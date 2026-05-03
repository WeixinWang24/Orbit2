from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from src.capability.models import CapabilityResult
from src.capability.mcp_servers.l1_structured.filesystem import (
    stdio_server as structured_filesystem,
)
from src.capability.mcp_servers.l1_structured.git import (
    stdio_server as structured_git,
)
from src.capability.mcp_servers.l2_toolchain.repo_scout import (
    stdio_server as repo_scout,
)
from src.capability.mcp_servers.l3_workflow import (
    inspect_change_set_workflow_result,
    persist_decision_response,
    validate_decision_response,
)
from src.capability.mcp_servers.l3_workflow.schemas import WorkflowRun, now_iso
from src.capability.mcp_servers.l3_workflow.store import (
    SQLiteWorkflowRunStore,
    default_workflow_db_path,
)
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import Message, MessageRole, TurnRequest


WORKFLOW_DECISION_SCHEMA_VERSION = "orbit2.workflow_decision.v0"
WORKFLOW_RESUME_SCHEMA_VERSION = "orbit2.workflow_resume.v0"
WORKFLOW_DECISION_REQUEST_TAG = "workflow-decision-request"
WORKFLOW_DECISION_RESPONSE_TAG = "workflow-decision-response"
WORKFLOW_RESUME_BRANCH_RESULT_TAG = "workflow-resume-branch-result"
WORKFLOW_DECISION_TOOL_NAME = "orbit2_workflow_decision"
WORKFLOW_DEFAULT_EVIDENCE_LINE_SPAN_ENV = "ORBIT2_WORKFLOW_DEFAULT_EVIDENCE_LINE_SPAN"
FALLBACK_WORKFLOW_DEFAULT_EVIDENCE_LINE_SPAN = 40


class WorkflowDecisionRoundtripError(RuntimeError):
    pass


class WorkflowResumeDispatchError(RuntimeError):
    pass


def project_decision_request_to_message(
    decision_request_package: dict[str, Any],
) -> Message:
    package = dict(decision_request_package)
    payload = {
        "schema_version": WORKFLOW_DECISION_SCHEMA_VERSION,
        "decision_request_package": package,
    }
    content = "\n".join(
        [
            "A workflow is waiting for a provider decision.",
            "Use the workflow decision tool exposed in this turn to select exactly one admissible branch.",
            "Do not call repository or filesystem tools from this decision turn unless Runtime exposes them separately.",
            "",
            f'<{WORKFLOW_DECISION_REQUEST_TAG} schema_version="{WORKFLOW_DECISION_SCHEMA_VERSION}">',
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            f"</{WORKFLOW_DECISION_REQUEST_TAG}>",
            "",
            "The admissible options are not recommendations.",
        ]
    )
    return Message(role=MessageRole.USER.value, content=content)


def project_decision_request_to_turn_request(
    decision_request_package: dict[str, Any],
    *,
    system: str | None = None,
) -> TurnRequest:
    return TurnRequest(
        messages=[project_decision_request_to_message(decision_request_package)],
        system=system,
        tool_definitions=[build_workflow_decision_tool_definition(decision_request_package)],
    )


def workflow_decision_package_from_result(
    result: CapabilityResult,
) -> dict[str, Any] | None:
    candidates: list[Any] = []
    if result.data is not None:
        candidates.append(result.data)
        if isinstance(result.data, dict):
            candidates.append(result.data.get("raw_result"))
    if result.content:
        try:
            candidates.append(json.loads(result.content))
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        package = workflow_decision_package_from_payload(candidate)
        if package is not None:
            return package
    return None


def workflow_decision_package_from_payload(
    payload: Any,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    package = payload.get("decision_request_package")
    if not isinstance(package, dict):
        return None
    if payload.get("status") != "waiting_for_decision":
        return None
    required = ("workflow_run_id", "decision_id", "workflow_name")
    if not all(isinstance(package.get(key), str) and package[key] for key in required):
        return None
    return package


def render_workflow_branch_result(dispatch_result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f'<{WORKFLOW_RESUME_BRANCH_RESULT_TAG} schema_version="{WORKFLOW_RESUME_SCHEMA_VERSION}">',
            json.dumps(dispatch_result, ensure_ascii=False, indent=2, sort_keys=True),
            f"</{WORKFLOW_RESUME_BRANCH_RESULT_TAG}>",
        ]
    )


def build_workflow_decision_tool_definition(
    decision_request_package: dict[str, Any],
) -> dict[str, Any]:
    option_ids = [
        option["option_id"]
        for option in decision_request_package.get("admissible_options", [])
        if isinstance(option, dict) and isinstance(option.get("option_id"), str)
    ]
    return {
        "name": WORKFLOW_DECISION_TOOL_NAME,
        "description": (
            "Select one admissible branch for the workflow decision currently "
            "presented in the transcript. This tool records the provider decision; "
            "it does not perform arbitrary repository actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_run_id": {
                    "type": "string",
                    "description": "Workflow run id from the decision request.",
                    "const": decision_request_package["workflow_run_id"],
                },
                "decision_id": {
                    "type": "string",
                    "description": "Decision id from the decision request.",
                    "const": decision_request_package["decision_id"],
                },
                "selected_option_id": {
                    "type": "string",
                    "description": "One admissible option id from the decision request.",
                    "enum": option_ids,
                },
                "rationale_summary": {
                    "type": "string",
                    "description": "Brief reason for selecting this branch.",
                },
                "requested_evidence": {
                    "type": "array",
                    "description": "Optional requested evidence descriptors.",
                    "items": {"type": "object"},
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional provider-side decision metadata.",
                },
            },
            "required": ["workflow_run_id", "decision_id", "selected_option_id"],
            "additionalProperties": False,
        },
    }


def parse_decision_response_text(text: str) -> dict[str, Any]:
    candidates = [
        text.strip(),
        *_extract_tagged_blocks(text, WORKFLOW_DECISION_RESPONSE_TAG),
        *_extract_json_fenced_blocks(text),
    ]
    brace_candidate = _extract_brace_object(text)
    if brace_candidate is not None:
        candidates.append(brace_candidate)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("provider decision response does not contain a JSON object")


def build_workflow_resume_request(
    *,
    decision_request_package: dict[str, Any],
    provider_response_text: str,
) -> dict[str, Any]:
    response = parse_decision_response_text(provider_response_text)
    return validate_decision_response(
        decision_request_package=decision_request_package,
        response=response,
    ).to_dict()


def build_workflow_resume_request_from_tool_request(
    *,
    decision_request_package: dict[str, Any],
    tool_request: Any,
) -> dict[str, Any]:
    tool_name = getattr(tool_request, "tool_name", None)
    arguments = getattr(tool_request, "arguments", None)
    if tool_name != WORKFLOW_DECISION_TOOL_NAME:
        raise WorkflowDecisionRoundtripError(
            f"workflow decision expected {WORKFLOW_DECISION_TOOL_NAME}, got {tool_name!r}"
        )
    if not isinstance(arguments, dict):
        raise WorkflowDecisionRoundtripError(
            "workflow decision tool arguments must be an object"
        )
    return validate_decision_response(
        decision_request_package=decision_request_package,
        response=arguments,
    ).to_dict()


def run_workflow_decision_roundtrip(
    *,
    backend: ExecutionBackend,
    workspace_root: str | Path,
    decision_request_package: dict[str, Any],
    system: str | None = None,
    on_partial_text: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = project_decision_request_to_turn_request(
        decision_request_package,
        system=system,
    )
    plan = backend.plan_from_messages(request, on_partial_text=on_partial_text)
    if len(plan.tool_requests) != 1:
        raise WorkflowDecisionRoundtripError(
            "workflow decision provider response must contain exactly one decision tool request"
        )

    resume_request = build_workflow_resume_request_from_tool_request(
        decision_request_package=decision_request_package,
        tool_request=plan.tool_requests[0],
    )
    persisted_resume_request = persist_decision_response(
        workspace_root=Path(workspace_root),
        decision_request_package=decision_request_package,
        response=resume_request["decision_response"],
    )
    return {
        "ok": True,
        "status": "decision_response_persisted",
        "workflow_run_id": persisted_resume_request["workflow_run_id"],
        "decision_id": persisted_resume_request["decision_id"],
        "resume_request": persisted_resume_request,
        "provider_plan": {
            "source_backend": plan.source_backend,
            "plan_label": plan.plan_label,
            "model": plan.model,
            "metadata": dict(plan.metadata),
        },
    }


def dispatch_workflow_resume_branch(
    *,
    workspace_root: str | Path,
    decision_request_package: dict[str, Any],
    resume_request: dict[str, Any],
) -> dict[str, Any]:
    workspace = Path(workspace_root)
    validated = validate_decision_response(
        decision_request_package=decision_request_package,
        response=resume_request.get("decision_response", {}),
    ).to_dict()
    selected_option = validated["selected_option"]
    branch_type = selected_option.get("branch_type")

    previous_workspace = os.environ.get(repo_scout.WORKSPACE_ROOT_ENV)
    os.environ[repo_scout.WORKSPACE_ROOT_ENV] = str(workspace)
    try:
        if branch_type == "evidence_acquisition":
            branch_result = _dispatch_evidence_acquisition(
                selected_option=selected_option,
                decision_response=validated["decision_response"],
            )
            status = "resume_branch_completed"
            workflow_status = "resumed_completed"
        elif branch_type == "summary_projection":
            branch_result = _dispatch_summary_projection(
                decision_request_package=decision_request_package,
                selected_option=selected_option,
            )
            status = "resume_branch_completed"
            workflow_status = "resumed_completed"
        elif branch_type == "stop":
            branch_result = _dispatch_stop(
                decision_request_package=decision_request_package,
                selected_option=selected_option,
            )
            status = "resume_branch_stopped"
            workflow_status = "stopped"
        elif branch_type == "continue_fact_collection":
            branch_result = _dispatch_continue_fact_collection(
                decision_request_package=decision_request_package,
                selected_option=selected_option,
            )
            status = "resume_branch_completed"
            workflow_status = "resumed_completed"
        elif branch_type == "workflow_transition":
            branch_result = _dispatch_workflow_transition(selected_option=selected_option)
            status = "resume_branch_completed"
            workflow_status = "resumed_completed"
        else:
            raise WorkflowResumeDispatchError(
                f"unsupported workflow branch_type: {branch_type!r}"
            )
    finally:
        if previous_workspace is None:
            os.environ.pop(repo_scout.WORKSPACE_ROOT_ENV, None)
        else:
            os.environ[repo_scout.WORKSPACE_ROOT_ENV] = previous_workspace

    result = {
        "ok": True,
        "status": status,
        "workflow_run_id": validated["workflow_run_id"],
        "decision_id": validated["decision_id"],
        "selected_option_id": selected_option["option_id"],
        "branch_type": branch_type,
        "branch_result": branch_result,
        "resume_request": validated,
        "audit": {
            "message_type": "workflow_resume_branch_result",
            "runtime_owner": "runtime_core",
            "capability_layer": "workflow",
            "branch_dispatch": "provider_selected_branch",
            "decision_posture": "provider_mediated",
        },
    }
    _persist_resume_dispatch_result(
        workspace_root=workspace,
        decision_request_package=decision_request_package,
        dispatch_result=result,
        workflow_status=workflow_status,
    )
    return result


def _dispatch_evidence_acquisition(
    *,
    selected_option: dict[str, Any],
    decision_response: dict[str, Any],
) -> dict[str, Any]:
    payload = selected_option.get("payload") or {}
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise WorkflowResumeDispatchError("evidence_acquisition branch has no candidate")
    tool = candidate.get("tool")
    target = candidate.get("target") or {}
    if not isinstance(target, dict):
        raise WorkflowResumeDispatchError("evidence_acquisition candidate target is invalid")

    evidence_gap = _candidate_evidence_gap(
        selected_option=selected_option,
        candidate=candidate,
    )
    reason = (
        decision_response.get("rationale_summary")
        or "provider selected exact structured evidence branch"
    )
    if tool == "structured_filesystem.read_file_region":
        evidence = structured_filesystem._read_file_region_result(
            path=str(target["path"]),
            start_line=int(target["start_line"]),
            end_line=_target_end_line(target),
            evidence_gap=evidence_gap,
            reason_context_pack_insufficient=str(reason),
        )
    elif tool == "structured_git.read_diff_hunk":
        evidence = structured_git._read_diff_hunk_result(
            path=str(target["path"]),
            hunk_index=int(target["hunk_index"]),
            evidence_gap=evidence_gap,
            reason_context_pack_insufficient=str(reason),
            staged=bool(candidate.get("state") == "staged" or target.get("staged")),
        )
    elif tool == "structured_git.read_git_show_region":
        evidence = structured_git._read_git_show_region_result(
            rev=str(target["rev"]),
            path=str(target["path"]),
            start_line=int(target["start_line"]),
            end_line=_target_end_line(target),
            evidence_gap_description=evidence_gap["description"],
            needed_evidence=evidence_gap["needed_evidence"],
            reason_context_pack_insufficient=str(reason),
            linked_context_pack_item=evidence_gap.get("linked_context_pack_item"),
        )
    else:
        raise WorkflowResumeDispatchError(f"unsupported evidence candidate tool: {tool!r}")

    return {
        "message_type": "workflow_branch_evidence_result",
        "selected_candidate": candidate,
        "evidence": evidence,
    }


def _dispatch_summary_projection(
    *,
    decision_request_package: dict[str, Any],
    selected_option: dict[str, Any],
) -> dict[str, Any]:
    factual_context = decision_request_package.get("factual_context", {})
    diff_digest = factual_context.get("diff_digest", {})
    impact_scope = factual_context.get("impact_scope", {})
    overview = factual_context.get("overview", {})
    changed_context = factual_context.get("changed_context", {})
    payload = selected_option.get("payload") or {}
    return {
        "summary_projection": {
            "message_type": "workflow_branch_fact_projection",
            "projection_type": payload.get("projection_type", "change_set_review_summary"),
            "decision_posture": "non_decisional_fact_projection",
            "workflow_run_id": decision_request_package["workflow_run_id"],
            "decision_id": decision_request_package["decision_id"],
            "selected_option_id": selected_option["option_id"],
            "overview_run_id": overview.get("run_id"),
            "changed_context_run_id": changed_context.get("run_id"),
            "diff_run_id": diff_digest.get("run_id") if isinstance(diff_digest, dict) else None,
            "impact_run_id": impact_scope.get("run_id") if isinstance(impact_scope, dict) else None,
            "overview_summary": overview.get("summary", {}) if isinstance(overview, dict) else {},
            "changed_context_summary": changed_context.get("summary", {}) if isinstance(changed_context, dict) else {},
            "diff_summary": diff_digest.get("summary", {}) if isinstance(diff_digest, dict) else {},
            "impact_summary": impact_scope.get("summary", {}) if isinstance(impact_scope, dict) else {},
            "artifact_refs": decision_request_package.get("artifact_refs", []),
        }
    }


def _dispatch_stop(
    *,
    decision_request_package: dict[str, Any],
    selected_option: dict[str, Any],
) -> dict[str, Any]:
    return {
        "message_type": "workflow_branch_stop_result",
        "stopped": True,
        "selected_option_id": selected_option["option_id"],
        "factual_context": decision_request_package.get("factual_context", {}),
        "artifact_refs": decision_request_package.get("artifact_refs", []),
    }


def _dispatch_continue_fact_collection(
    *,
    decision_request_package: dict[str, Any],
    selected_option: dict[str, Any],
) -> dict[str, Any]:
    payload = selected_option.get("payload") or {}
    fact_report = repo_scout._repo_scout_impact_scope_result(
        repo_id=f"{decision_request_package['workflow_run_id']}_resume_impact",
        label=decision_request_package.get("workflow_name"),
        include_untracked=True,
        max_impact_files=payload.get("max_impact_files"),
        max_impact_symbols=payload.get("max_impact_symbols"),
        max_impact_edges=payload.get("max_impact_edges"),
    )
    return {
        "message_type": "workflow_branch_fact_collection_result",
        "continuation_mode": "bounded_impact_scope_rerun_v0",
        "fact_report": fact_report,
    }


def _dispatch_workflow_transition(
    *,
    selected_option: dict[str, Any],
) -> dict[str, Any]:
    payload = selected_option.get("payload") or {}
    target_workflow = payload.get("target_workflow")
    if target_workflow != "inspect_change_set_workflow":
        raise WorkflowResumeDispatchError(
            f"unsupported workflow transition target: {target_workflow!r}"
        )
    result = inspect_change_set_workflow_result(
        repo_id=str(payload.get("repo_id") or "workspace"),
        label=payload.get("label") if isinstance(payload.get("label"), str) else None,
        include_untracked=bool(payload.get("include_untracked", True)),
    )
    return {
        "message_type": "workflow_branch_transition_result",
        "target_workflow": target_workflow,
        "workflow_result": result,
    }


def _persist_resume_dispatch_result(
    *,
    workspace_root: Path,
    decision_request_package: dict[str, Any],
    dispatch_result: dict[str, Any],
    workflow_status: str,
) -> None:
    store = SQLiteWorkflowRunStore(default_workflow_db_path(workspace_root))
    try:
        existing = store.get_run(decision_request_package["workflow_run_id"])
        store.save_run(
            WorkflowRun(
                workflow_run_id=decision_request_package["workflow_run_id"],
                workflow_name=decision_request_package["workflow_name"],
                cwd=existing["cwd"] if existing is not None else str(workspace_root),
                request=existing["request"] if existing is not None else {},
                status=workflow_status,
                started_at=(
                    existing["started_at"]
                    if existing is not None
                    else decision_request_package["created_at"]
                ),
                current_decision_id=decision_request_package["decision_id"],
                finished_at=now_iso(),
                report=dispatch_result,
                metadata={
                    **(existing["metadata"] if existing is not None else {}),
                    "capability_layer": "workflow",
                    "resume_branch_type": dispatch_result["branch_type"],
                },
            )
        )
    finally:
        store.close()


def _candidate_evidence_gap(
    *,
    selected_option: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    basis = candidate.get("basis")
    description = selected_option.get("description") or "Provider selected evidence branch"
    needed = basis if isinstance(basis, str) and basis.strip() else "exact structured evidence"
    return {
        "description": str(description),
        "needed_evidence": needed,
        "linked_context_pack_item": selected_option["option_id"],
    }


def _target_end_line(target: dict[str, Any]) -> int:
    if isinstance(target.get("end_line"), int):
        return int(target["end_line"])
    return int(target["start_line"]) + _default_evidence_line_span() - 1


def _default_evidence_line_span() -> int:
    raw = os.environ.get(WORKFLOW_DEFAULT_EVIDENCE_LINE_SPAN_ENV, "").strip()
    if not raw:
        return FALLBACK_WORKFLOW_DEFAULT_EVIDENCE_LINE_SPAN
    value = int(raw)
    if value <= 0:
        raise ValueError("workflow default evidence line span must be > 0")
    return value


def _extract_tagged_blocks(text: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}\b[^>]*>\s*(.*?)\s*</{tag}>", re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text)]


def _extract_json_fenced_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text)]


def _extract_brace_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1].strip()
