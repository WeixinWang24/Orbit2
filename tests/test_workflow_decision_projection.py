from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from src.capability.models import CapabilityResult
from src.capability.mcp_servers.l3_workflow import SQLiteWorkflowRunStore
from src.capability.mcp_servers.l3_workflow import (
    DecisionOption,
    DecisionRequestPackage,
    WorkflowRun,
)
from src.capability.mcp_servers.l3_workflow.schemas import now_iso
from src.capability.mcp_servers.l3_workflow.store import default_workflow_db_path
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import ExecutionPlan, MessageRole, ToolRequest, TurnRequest
from src.core.runtime.workflow_decision import (
    WORKFLOW_DECISION_TOOL_NAME,
    WORKFLOW_DECISION_REQUEST_TAG,
    WorkflowResumeDispatchError,
    WorkflowDecisionRoundtripError,
    build_workflow_decision_tool_definition,
    build_workflow_resume_request,
    build_workflow_resume_request_from_tool_request,
    dispatch_workflow_resume_branch,
    parse_decision_response_text,
    project_decision_request_to_message,
    project_decision_request_to_turn_request,
    render_workflow_branch_result,
    run_workflow_decision_roundtrip,
    workflow_decision_package_from_payload,
    workflow_decision_package_from_result,
)


def _decision_request_package() -> dict:
    return DecisionRequestPackage(
        workflow_run_id="wfr_test",
        decision_id="dec_test",
        workflow_name="inspect_change_set_workflow",
        message_type="decision_request",
        factual_context={
            "decision_posture": "non_decisional",
            "changed_files": [{"path": "src/app.py"}],
        },
        artifact_refs=[
            {
                "artifact_type": "toolchain_result",
                "tool_name": "repo_scout_diff_digest",
                "ref_id": "digest_test",
            }
        ],
        admissible_options=[
            DecisionOption(
                option_id="read_exact_l1_evidence",
                label="Read exact evidence",
                description="Read exact source regions for already reported facts.",
                branch_type="resume_with_l1_reads",
            ).to_dict(),
            DecisionOption(
                option_id="assemble_review_summary",
                label="Assemble review summary",
                description="Use the existing factual package to assemble a compact review summary.",
                branch_type="resume_with_summary",
            ).to_dict(),
        ],
        constraints=["admissible_options_are_not_recommendations"],
        response_schema={
            "type": "object",
            "required": ["workflow_run_id", "decision_id", "selected_option_id"],
        },
        created_at=now_iso(),
    ).to_dict()


def _branch_package(branch_type: str, payload: dict) -> dict:
    return DecisionRequestPackage(
        workflow_run_id=f"wfr_{branch_type}",
        decision_id=f"dec_{branch_type}",
        workflow_name="inspect_change_set_workflow",
        message_type="decision_request",
        factual_context={
            "message_type": "fact_package",
            "fact_domain": "change_set_inspection",
            "decision_posture": "non_decisional",
            "diff_digest": {
                "run_id": "run_diff",
                "summary": {
                    "file_count": 1,
                    "hunk_count": 1,
                    "additions": 2,
                    "deletions": 1,
                },
                "audit": {"decision_posture": "non_decisional"},
            },
            "impact_scope": {
                "run_id": "run_impact",
                "summary": {
                    "changed_symbol_count": 1,
                    "test_adjacency_count": 1,
                    "impact_file_truncated": False,
                },
                "audit": {"decision_posture": "non_decisional"},
            },
        },
        artifact_refs=[
            {"source": "repo_scout_diff_digest", "run_id": "run_diff"},
            {"source": "repo_scout_impact_scope", "run_id": "run_impact"},
        ],
        admissible_options=[
            DecisionOption(
                option_id=f"option_{branch_type}",
                label=f"Option {branch_type}",
                description=f"Execute {branch_type}",
                branch_type=branch_type,
                payload=payload,
            ).to_dict(),
        ],
        constraints=["admissible_options_are_not_recommendations"],
        response_schema={
            "type": "object",
            "required": ["workflow_run_id", "decision_id", "selected_option_id"],
        },
        created_at=now_iso(),
    ).to_dict()


def _resume_request(package: dict) -> dict:
    return build_workflow_resume_request(
        decision_request_package=package,
        provider_response_text=json.dumps({
            "workflow_run_id": package["workflow_run_id"],
            "decision_id": package["decision_id"],
            "selected_option_id": package["admissible_options"][0]["option_id"],
            "rationale_summary": "Provider selected this admissible branch.",
        }),
    )


def _decision_tool_request(package: dict, selected_option_id: str) -> ToolRequest:
    return ToolRequest(
        tool_call_id="call_workflow_decision",
        tool_name=WORKFLOW_DECISION_TOOL_NAME,
        arguments={
            "workflow_run_id": package["workflow_run_id"],
            "decision_id": package["decision_id"],
            "selected_option_id": selected_option_id,
            "rationale_summary": "Provider selected this admissible branch.",
        },
    )


def _persist_waiting_run(tmp_path: Path, package: dict) -> None:
    store = SQLiteWorkflowRunStore(default_workflow_db_path(tmp_path))
    try:
        store.save_run(
            WorkflowRun(
                workflow_run_id=package["workflow_run_id"],
                workflow_name=package["workflow_name"],
                cwd=str(tmp_path),
                request={},
                status="waiting_for_decision",
                started_at=package["created_at"],
                current_decision_id=package["decision_id"],
            )
        )
    finally:
        store.close()


class _DecisionBackend(ExecutionBackend):
    def __init__(self, final_text: str, *, tool_requests: list[ToolRequest] | None = None) -> None:
        self.final_text = final_text
        self.tool_requests = tool_requests or []
        self.requests: list[TurnRequest] = []

    @property
    def backend_name(self) -> str:
        return "decision-test-backend"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        self.requests.append(request)
        if on_partial_text is not None:
            on_partial_text(self.final_text)
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="decision",
            final_text=self.final_text,
            model="test-model",
            tool_requests=self.tool_requests,
        )


def test_project_decision_request_to_message_uses_typed_user_block() -> None:
    package = _decision_request_package()

    message = project_decision_request_to_message(package)

    assert message.role == MessageRole.USER.value
    assert f"<{WORKFLOW_DECISION_REQUEST_TAG}" in (message.content or "")
    assert package["workflow_run_id"] in (message.content or "")
    assert "read_exact_l1_evidence" in (message.content or "")
    assert "assemble_review_summary" in (message.content or "")
    assert "admissible_options_are_not_recommendations" in (message.content or "")
    assert "Use the workflow decision tool" in (message.content or "")


def test_workflow_decision_tool_definition_scopes_options() -> None:
    package = _decision_request_package()

    definition = build_workflow_decision_tool_definition(package)

    assert definition["name"] == WORKFLOW_DECISION_TOOL_NAME
    assert definition["parameters"]["properties"]["workflow_run_id"]["const"] == "wfr_test"
    assert definition["parameters"]["properties"]["decision_id"]["const"] == "dec_test"
    assert definition["parameters"]["properties"]["selected_option_id"]["enum"] == [
        "read_exact_l1_evidence",
        "assemble_review_summary",
    ]


def test_workflow_decision_package_helpers_accept_waiting_payloads_only() -> None:
    package = _decision_request_package()
    payload = {
        "status": "waiting_for_decision",
        "decision_request_package": package,
    }

    assert workflow_decision_package_from_payload(payload) == package
    assert workflow_decision_package_from_payload({
        **payload,
        "status": "completed",
    }) is None


def test_workflow_decision_package_from_result_reads_data_raw_and_content() -> None:
    package = _decision_request_package()
    payload = {
        "status": "waiting_for_decision",
        "decision_request_package": package,
    }

    assert workflow_decision_package_from_result(CapabilityResult(
        tool_call_id="call_data",
        tool_name="workflow",
        ok=True,
        content="",
        data={"raw_result": payload},
    )) == package
    assert workflow_decision_package_from_result(CapabilityResult(
        tool_call_id="call_content",
        tool_name="workflow",
        ok=True,
        content=json.dumps(payload),
        data=None,
    )) == package


def test_render_workflow_branch_result_uses_typed_block() -> None:
    rendered = render_workflow_branch_result({"ok": True, "status": "done"})

    assert '<workflow-resume-branch-result schema_version="orbit2.workflow_resume.v0">' in rendered
    assert '"ok": true' in rendered
    assert rendered.endswith("</workflow-resume-branch-result>")


def test_project_decision_request_to_turn_request_preserves_system() -> None:
    package = _decision_request_package()

    request = project_decision_request_to_turn_request(
        package,
        system="Orbit2 runtime decision projection test.",
    )

    assert request.system == "Orbit2 runtime decision projection test."
    assert len(request.messages) == 1
    assert request.messages[0].role == MessageRole.USER.value
    assert request.tool_definitions is not None
    assert request.tool_definitions[0]["name"] == WORKFLOW_DECISION_TOOL_NAME


def test_parse_decision_response_accepts_json_fence_and_typed_block() -> None:
    response = {
        "workflow_run_id": "wfr_test",
        "decision_id": "dec_test",
        "selected_option_id": "assemble_review_summary",
    }

    assert parse_decision_response_text(json.dumps(response)) == response
    assert parse_decision_response_text(f"```json\n{json.dumps(response)}\n```") == response
    assert parse_decision_response_text(
        f"<workflow-decision-response>\n{json.dumps(response)}\n</workflow-decision-response>"
    ) == response


def test_build_workflow_resume_request_validates_provider_response() -> None:
    package = _decision_request_package()
    provider_response = {
        "workflow_run_id": package["workflow_run_id"],
        "decision_id": package["decision_id"],
        "selected_option_id": "assemble_review_summary",
        "rationale_summary": "The current factual package is enough for a compact review.",
    }

    resume = build_workflow_resume_request(
        decision_request_package=package,
        provider_response_text=json.dumps(provider_response),
    )

    assert resume["workflow_run_id"] == package["workflow_run_id"]
    assert resume["decision_id"] == package["decision_id"]
    assert resume["selected_option"]["option_id"] == "assemble_review_summary"
    assert resume["decision_response"]["rationale_summary"].startswith("The current")


def test_build_workflow_resume_request_from_decision_tool_call() -> None:
    package = _decision_request_package()

    resume = build_workflow_resume_request_from_tool_request(
        decision_request_package=package,
        tool_request=_decision_tool_request(package, "assemble_review_summary"),
    )

    assert resume["workflow_run_id"] == package["workflow_run_id"]
    assert resume["selected_option"]["option_id"] == "assemble_review_summary"


def test_build_workflow_resume_request_rejects_non_admissible_option() -> None:
    package = _decision_request_package()
    provider_response = {
        "workflow_run_id": package["workflow_run_id"],
        "decision_id": package["decision_id"],
        "selected_option_id": "hidden_planner_branch",
    }

    with pytest.raises(ValueError, match="not admissible"):
        build_workflow_resume_request(
            decision_request_package=package,
            provider_response_text=json.dumps(provider_response),
        )


def test_run_workflow_decision_roundtrip_persists_provider_response(
    tmp_path: Path,
) -> None:
    package = _decision_request_package()
    provider_response = {
        "workflow_run_id": package["workflow_run_id"],
        "decision_id": package["decision_id"],
        "selected_option_id": "read_exact_l1_evidence",
        "rationale_summary": "Read the exact evidence before summarizing.",
    }
    store = SQLiteWorkflowRunStore(default_workflow_db_path(tmp_path))
    try:
        store.save_run(
            WorkflowRun(
                workflow_run_id=package["workflow_run_id"],
                workflow_name=package["workflow_name"],
                cwd=str(tmp_path),
                request={},
                status="waiting_for_decision",
                started_at=package["created_at"],
                current_decision_id=package["decision_id"],
            )
        )
        store.save_decision_request(
            DecisionRequestPackage(
                workflow_run_id=package["workflow_run_id"],
                decision_id=package["decision_id"],
                workflow_name=package["workflow_name"],
                message_type=package["message_type"],
                factual_context=package["factual_context"],
                artifact_refs=package["artifact_refs"],
                admissible_options=package["admissible_options"],
                constraints=package["constraints"],
                response_schema=package["response_schema"],
                created_at=package["created_at"],
                status=package["status"],
            )
        )
    finally:
        store.close()

    backend = _DecisionBackend(
        "",
        tool_requests=[_decision_tool_request(package, "read_exact_l1_evidence")],
    )
    streamed: list[str] = []

    result = run_workflow_decision_roundtrip(
        backend=backend,
        workspace_root=tmp_path,
        decision_request_package=package,
        system="Orbit2 workflow decision test.",
        on_partial_text=streamed.append,
    )

    assert result["ok"] is True
    assert result["status"] == "decision_response_persisted"
    assert result["resume_request"]["selected_option"]["option_id"] == "read_exact_l1_evidence"
    assert result["provider_plan"]["source_backend"] == "decision-test-backend"
    assert streamed == [""]
    assert len(backend.requests) == 1
    assert backend.requests[0].system == "Orbit2 workflow decision test."
    assert backend.requests[0].tool_definitions is not None
    assert backend.requests[0].tool_definitions[0]["name"] == WORKFLOW_DECISION_TOOL_NAME
    assert f"<{WORKFLOW_DECISION_REQUEST_TAG}" in (backend.requests[0].messages[0].content or "")

    store = SQLiteWorkflowRunStore(default_workflow_db_path(tmp_path))
    try:
        persisted = store.list_decision_responses(package["workflow_run_id"])
    finally:
        store.close()
    assert persisted[0]["selected_option_id"] == "read_exact_l1_evidence"


def test_run_workflow_decision_roundtrip_rejects_provider_tool_requests(
    tmp_path: Path,
) -> None:
    package = _decision_request_package()
    backend = _DecisionBackend(
        "{}",
        tool_requests=[],
    )

    with pytest.raises(WorkflowDecisionRoundtripError, match="exactly one"):
        run_workflow_decision_roundtrip(
            backend=backend,
            workspace_root=tmp_path,
            decision_request_package=package,
        )


def test_dispatch_workflow_resume_branch_reads_selected_l1_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ORBIT_WORKSPACE_ROOT", str(tmp_path))
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text("line 1\nline 2\nline 3\nline 4\n", encoding="utf-8")
    package = _branch_package(
        "evidence_acquisition",
        {
            "candidate": {
                "tool": "structured_filesystem.read_file_region",
                "target": {"path": "src/app.py", "start_line": 2, "end_line": 3},
                "basis": "test selected exact file region",
            }
        },
    )
    _persist_waiting_run(tmp_path, package)

    result = dispatch_workflow_resume_branch(
        workspace_root=tmp_path,
        decision_request_package=package,
        resume_request=_resume_request(package),
    )

    assert result["ok"] is True
    assert result["status"] == "resume_branch_completed"
    assert result["branch_type"] == "evidence_acquisition"
    evidence = result["branch_result"]["evidence"]
    assert evidence["evidence_type"] == "file_region"
    assert evidence["content"] == "line 2\nline 3"

    store = SQLiteWorkflowRunStore(default_workflow_db_path(tmp_path))
    try:
        run = store.get_run(package["workflow_run_id"])
    finally:
        store.close()
    assert run is not None
    assert run["status"] == "resumed_completed"
    assert run["report"]["branch_type"] == "evidence_acquisition"


def test_dispatch_workflow_resume_branch_projects_summary(tmp_path: Path) -> None:
    package = _branch_package(
        "summary_projection",
        {"diff_run_id": "run_diff", "impact_run_id": "run_impact"},
    )
    _persist_waiting_run(tmp_path, package)

    result = dispatch_workflow_resume_branch(
        workspace_root=tmp_path,
        decision_request_package=package,
        resume_request=_resume_request(package),
    )

    assert result["ok"] is True
    assert result["branch_type"] == "summary_projection"
    summary = result["branch_result"]["summary_projection"]
    assert summary["message_type"] == "workflow_branch_fact_projection"
    assert summary["diff_summary"]["file_count"] == 1
    assert summary["impact_summary"]["test_adjacency_count"] == 1
    assert summary["decision_posture"] == "non_decisional_fact_projection"


def test_dispatch_workflow_resume_branch_stops_workflow(tmp_path: Path) -> None:
    package = _branch_package("stop", {})
    _persist_waiting_run(tmp_path, package)

    result = dispatch_workflow_resume_branch(
        workspace_root=tmp_path,
        decision_request_package=package,
        resume_request=_resume_request(package),
    )

    assert result["ok"] is True
    assert result["status"] == "resume_branch_stopped"
    assert result["branch_result"]["stopped"] is True

    store = SQLiteWorkflowRunStore(default_workflow_db_path(tmp_path))
    try:
        run = store.get_run(package["workflow_run_id"])
    finally:
        store.close()
    assert run is not None
    assert run["status"] == "stopped"


def test_dispatch_workflow_resume_branch_rejects_unknown_branch_type(
    tmp_path: Path,
) -> None:
    package = _branch_package("hidden_planner_branch", {})

    with pytest.raises(WorkflowResumeDispatchError, match="unsupported workflow branch_type"):
        dispatch_workflow_resume_branch(
            workspace_root=tmp_path,
            decision_request_package=package,
            resume_request=_resume_request(package),
        )
