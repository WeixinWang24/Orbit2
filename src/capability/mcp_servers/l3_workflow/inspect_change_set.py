from __future__ import annotations

from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l2_toolchain.repo_scout import (
    stdio_server as repo_scout,
)
from src.capability.mcp_servers.l3_workflow.base import WorkflowRunRecorder
from src.capability.mcp_servers.l3_workflow.schemas import now_iso


WORKFLOW_NAME = "inspect_change_set_workflow"


def _workspace_root() -> Path:
    return repo_scout._workspace_root()


def _decision_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["workflow_run_id", "decision_id", "selected_option_id"],
        "properties": {
            "workflow_run_id": {"type": "string"},
            "decision_id": {"type": "string"},
            "selected_option_id": {"type": "string"},
            "rationale_summary": {"type": "string"},
            "requested_evidence": {
                "type": "array",
                "items": {"type": "object"},
            },
            "metadata": {"type": "object"},
        },
    }


def _admissible_options(
    *,
    diff_digest: dict[str, Any],
    impact_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    first_candidate = None
    for item in diff_digest["diff_digest"]["files"]:
        candidates = item.get("evidence_read_candidates", [])
        if candidates:
            first_candidate = {
                **candidates[0],
                "path": item["path"],
                "state": item.get("state"),
            }
            break
    options: list[dict[str, Any]] = []
    if first_candidate is not None:
        options.append({
            "option_id": "read_exact_l1_evidence",
            "label": "Read exact L1 evidence",
            "description": "Ask Runtime to acquire one exact structured evidence item named by the fact package.",
            "branch_type": "evidence_acquisition",
            "payload": {"candidate": first_candidate},
        })
    options.extend([
        {
            "option_id": "assemble_review_summary",
            "label": "Assemble review summary",
            "description": "Ask Runtime to project the existing fact package into a review-oriented summary.",
            "branch_type": "summary_projection",
            "payload": {
                "diff_run_id": diff_digest["run_id"],
                "impact_run_id": impact_scope["run_id"],
            },
        },
        {
            "option_id": "stop_and_report",
            "label": "Stop and report",
            "description": "Stop workflow progression and report the current bounded facts.",
            "branch_type": "stop",
            "payload": {},
        },
    ])
    if impact_scope["summary"].get("impact_file_truncated"):
        options.append({
            "option_id": "continue_impact_scope",
            "label": "Continue impact scope",
            "description": "Ask Runtime to run another bounded impact-scope slice.",
            "branch_type": "continue_fact_collection",
            "payload": {
                "observed_truncation": "impact_file_truncated",
            },
        })
    return options


def inspect_change_set_workflow_result(
    *,
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_diff_files: int | None = None,
    max_impact_files: int | None = None,
    max_impact_symbols: int | None = None,
    max_impact_edges: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    request = {
        "cwd": cwd,
        "repo_id": repo_id,
        "label": label,
        "include_untracked": include_untracked,
        "max_diff_files": max_diff_files,
        "max_impact_files": max_impact_files,
        "max_impact_symbols": max_impact_symbols,
        "max_impact_edges": max_impact_edges,
    }
    recorder = WorkflowRunRecorder(
        workflow_name=WORKFLOW_NAME,
        workspace_root=workspace,
        request=request,
    )
    try:
        diff_started = now_iso()
        diff_digest = repo_scout._repo_scout_diff_digest_result(
            cwd=cwd,
            repo_id=f"{repo_id}_diff",
            label=label,
            include_untracked=include_untracked,
            max_diff_files=max_diff_files,
        )
        recorder.record_step(
            step_id="step_001_repo_scout_diff_digest",
            name="collect diff fact digest",
            kind="toolchain",
            status="completed" if diff_digest["ok"] else "error",
            started_at=diff_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"files={diff_digest['summary']['file_count']} "
                f"hunks={diff_digest['summary']['hunk_count']}"
            ) if diff_digest["ok"] else str(diff_digest.get("summary")),
            refs=[{"kind": "toolchain_run", "run_id": diff_digest["run_id"]}],
        )

        impact_started = now_iso()
        impact_scope = repo_scout._repo_scout_impact_scope_result(
            cwd=cwd,
            repo_id=f"{repo_id}_impact",
            label=label,
            include_untracked=include_untracked,
            max_impact_files=max_impact_files,
            max_impact_symbols=max_impact_symbols,
            max_impact_edges=max_impact_edges,
        )
        recorder.record_step(
            step_id="step_002_repo_scout_impact_scope",
            name="collect impact-scope fact report",
            kind="toolchain",
            status="completed" if impact_scope["ok"] else "error",
            started_at=impact_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"changed_symbols={impact_scope['summary']['changed_symbol_count']} "
                f"test_adjacency={impact_scope['summary']['test_adjacency_count']}"
            ) if impact_scope["ok"] else str(impact_scope.get("summary")),
            refs=[{"kind": "toolchain_run", "run_id": impact_scope["run_id"]}],
        )

        factual_context = {
            "message_type": "fact_package",
            "fact_domain": "change_set_inspection",
            "decision_posture": "non_decisional",
            "diff_digest": {
                "run_id": diff_digest["run_id"],
                "summary": diff_digest["summary"],
                "audit": diff_digest["audit"],
            },
            "impact_scope": {
                "run_id": impact_scope["run_id"],
                "summary": impact_scope["summary"],
                "audit": impact_scope["audit"],
            },
        }
        artifact_refs = [
            {"source": "repo_scout_diff_digest", **ref}
            for ref in diff_digest.get("artifact_refs", [])
        ] + [
            {"source": "repo_scout_impact_scope", **ref}
            for ref in impact_scope.get("artifact_refs", [])
        ]
        decision_package = recorder.create_decision_request(
            factual_context=factual_context,
            artifact_refs=artifact_refs,
            admissible_options=_admissible_options(
                diff_digest=diff_digest,
                impact_scope=impact_scope,
            ),
            constraints=[
                "admissible_options_are_not_recommendations",
                "decision_response_must_select_one_option_id",
                "workflow_must_not_block_inside_mcp_waiting_for_provider",
            ],
            response_schema=_decision_response_schema(),
        )
        recorder.record_step(
            step_id="step_003_emit_decision_request",
            name="emit provider decision request",
            kind="decision_request",
            status="waiting_for_decision",
            started_at=now_iso(),
            input_summary="assembled diff and impact fact reports",
            output_summary=f"decision_id={decision_package['decision_id']}",
            refs=[{"kind": "decision_request", "decision_id": decision_package["decision_id"]}],
        )
        report: dict[str, Any] = {
            "ok": True,
            "workflow_run_id": recorder.workflow_run_id,
            "status": "waiting_for_decision",
            "workflow_name": WORKFLOW_NAME,
            "cwd": str(workspace),
            "request": request,
            "summary": {
                "diff_run_id": diff_digest["run_id"],
                "impact_run_id": impact_scope["run_id"],
                "decision_id": decision_package["decision_id"],
                "admissible_option_count": len(decision_package["admissible_options"]),
            },
            "decision_request_package": decision_package,
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                "capability_layer": "workflow",
                "message_type": "decision_request",
                "provider_roundtrip_owner": "runtime_core",
                "provider_wait": "not_performed_inside_tool_call",
                "decision_posture": "provider_mediated",
                "lower_level_reuse": [
                    "repo_scout_diff_digest",
                    "repo_scout_impact_scope",
                ],
            },
        }
        recorder.finalize_waiting(
            decision_request_package=decision_package,
            report=report,
        )
        return report
    finally:
        recorder.close()
