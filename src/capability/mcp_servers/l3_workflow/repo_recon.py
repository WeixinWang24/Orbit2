from __future__ import annotations

from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l2_toolchain.repo_scout import (
    stdio_server as repo_scout,
)
from src.capability.mcp_servers.l3_workflow.base import WorkflowRunRecorder
from src.capability.mcp_servers.l3_workflow.schemas import now_iso


WORKFLOW_NAME = "repo_recon_workflow"


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
    overview: dict[str, Any],
    changed_context: dict[str, Any],
    diff_digest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    options = [
        {
            "option_id": "summarize_workspace_orientation",
            "label": "Summarize workspace orientation",
            "description": "Ask Runtime to project the existing repo reconnaissance facts into a bounded workspace orientation summary.",
            "branch_type": "summary_projection",
            "payload": {
                "projection_type": "repo_recon_workspace_orientation",
                "overview_run_id": overview["run_id"],
                "changed_context_run_id": changed_context["run_id"],
                "diff_digest_run_id": diff_digest["run_id"] if diff_digest else None,
            },
        },
        {
            "option_id": "stop_and_report",
            "label": "Stop and report",
            "description": "Stop workflow progression and report the current bounded repo reconnaissance facts.",
            "branch_type": "stop",
            "payload": {},
        },
    ]
    first_candidate = _first_orientation_candidate(overview)
    if first_candidate is not None:
        options.insert(1, {
            "option_id": "read_repo_structure_evidence",
            "label": "Read repo structure evidence",
            "description": "Ask Runtime to acquire one exact structured evidence item named by the repo reconnaissance fact package.",
            "branch_type": "evidence_acquisition",
            "payload": {"candidate": first_candidate},
        })
    if changed_context["summary"].get("clean") is False:
        options.insert(1, {
            "option_id": "inspect_change_set",
            "label": "Inspect change set",
            "description": "Ask Runtime to start the dedicated change-set inspection workflow using the current bounded repo facts as the handoff point.",
            "branch_type": "workflow_transition",
            "payload": {
                "target_workflow": "inspect_change_set_workflow",
                "repo_id": changed_context["request"]["repo_id"],
                "label": changed_context["request"].get("label"),
                "include_untracked": changed_context["request"]["include_untracked"],
            },
        })
    return options


def _first_orientation_candidate(overview: dict[str, Any]) -> dict[str, Any] | None:
    candidates = overview["overview"].get("orientation_candidates", [])
    if not candidates:
        return None
    candidate = dict(candidates[0])
    return {
        "tool": candidate.get("candidate_tool", "structured_filesystem.read_file_region"),
        "target": {"path": candidate["path"], "start_line": 1},
        "basis": candidate.get("basis", "repo orientation candidate"),
        "path": candidate["path"],
    }


def repo_recon_workflow_result(
    *,
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    max_symbols: int | None = None,
    max_symbols_per_file: int | None = None,
    max_edges_per_file: int | None = None,
    max_diff_files: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    request = {
        "cwd": cwd,
        "repo_id": repo_id,
        "label": label,
        "include_untracked": include_untracked,
        "max_tree_depth": max_tree_depth,
        "max_tree_entries": max_tree_entries,
        "max_symbols": max_symbols,
        "max_symbols_per_file": max_symbols_per_file,
        "max_edges_per_file": max_edges_per_file,
        "max_diff_files": max_diff_files,
    }
    recorder = WorkflowRunRecorder(
        workflow_name=WORKFLOW_NAME,
        workspace_root=workspace,
        request=request,
    )
    try:
        overview_started = now_iso()
        overview = repo_scout._repo_scout_repository_overview_result(
            path=cwd or ".",
            repo_id=f"{repo_id}_overview",
            label=label,
            max_tree_depth=max_tree_depth,
            max_tree_entries=max_tree_entries,
            max_symbols=max_symbols,
        )
        recorder.record_step(
            step_id="step_001_repo_scout_repository_overview",
            name="collect repository overview facts",
            kind="toolchain",
            status="completed" if overview["ok"] else "error",
            started_at=overview_started,
            input_summary=f"cwd={cwd!r} repo_id={repo_id!r}",
            output_summary=(
                f"clean={overview['summary']['clean']} "
                f"tree_entries={overview['summary']['tree_entry_count']} "
                f"orientation_candidates={overview['summary']['orientation_candidate_count']}"
            ) if overview["ok"] else str(overview.get("summary")),
            refs=[{"kind": "toolchain_run", "run_id": overview["run_id"]}],
        )

        changed_started = now_iso()
        changed_context = repo_scout._repo_scout_changed_context_result(
            cwd=cwd,
            repo_id=f"{repo_id}_changed",
            label=label,
            include_untracked=include_untracked,
            max_symbols_per_file=max_symbols_per_file,
            max_edges_per_file=max_edges_per_file,
        )
        recorder.record_step(
            step_id="step_002_repo_scout_changed_context",
            name="collect changed workspace facts",
            kind="toolchain",
            status="completed" if changed_context["ok"] else "error",
            started_at=changed_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"clean={changed_context['summary']['clean']} "
                f"changed_file_states={changed_context['summary']['changed_file_state_count']}"
            ) if changed_context["ok"] else str(changed_context.get("summary")),
            refs=[{"kind": "toolchain_run", "run_id": changed_context["run_id"]}],
        )

        diff_digest: dict[str, Any] | None = None
        if changed_context["summary"].get("clean") is False:
            diff_started = now_iso()
            diff_digest = repo_scout._repo_scout_diff_digest_result(
                cwd=cwd,
                repo_id=f"{repo_id}_diff",
                label=label,
                include_untracked=include_untracked,
                max_diff_files=max_diff_files,
            )
            recorder.record_step(
                step_id="step_003_repo_scout_diff_digest",
                name="collect diff digest facts",
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

        factual_context = {
            "message_type": "fact_package",
            "fact_domain": "repo_recon",
            "decision_posture": "non_decisional",
            "overview": {
                "run_id": overview["run_id"],
                "summary": overview["summary"],
                "audit": overview["audit"],
            },
            "changed_context": {
                "run_id": changed_context["run_id"],
                "summary": changed_context["summary"],
                "audit": changed_context["audit"],
            },
            "diff_digest": (
                {
                    "run_id": diff_digest["run_id"],
                    "summary": diff_digest["summary"],
                    "audit": diff_digest["audit"],
                }
                if diff_digest is not None
                else None
            ),
        }
        artifact_refs = _artifact_refs("repo_scout_repository_overview", overview)
        artifact_refs.extend(_artifact_refs("repo_scout_changed_context", changed_context))
        if diff_digest is not None:
            artifact_refs.extend(_artifact_refs("repo_scout_diff_digest", diff_digest))

        decision_package = recorder.create_decision_request(
            factual_context=factual_context,
            artifact_refs=artifact_refs,
            admissible_options=_admissible_options(
                overview=overview,
                changed_context=changed_context,
                diff_digest=diff_digest,
            ),
            constraints=[
                "admissible_options_are_not_recommendations",
                "decision_response_must_select_one_option_id",
                "workflow_must_not_block_inside_mcp_waiting_for_provider",
            ],
            response_schema=_decision_response_schema(),
        )
        recorder.record_step(
            step_id="step_004_emit_decision_request",
            name="emit provider decision request",
            kind="decision_request",
            status="waiting_for_decision",
            started_at=now_iso(),
            input_summary="assembled repo overview, changed context, and optional diff digest facts",
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
                "overview_run_id": overview["run_id"],
                "changed_context_run_id": changed_context["run_id"],
                "diff_digest_run_id": diff_digest["run_id"] if diff_digest else None,
                "repo_clean": changed_context["summary"]["clean"],
                "changed_file_state_count": changed_context["summary"]["changed_file_state_count"],
                "orientation_candidate_count": overview["summary"]["orientation_candidate_count"],
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
                    "repo_scout_repository_overview",
                    "repo_scout_changed_context",
                    "repo_scout_diff_digest",
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


def _artifact_refs(source: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"source": source, **ref}
        for ref in report.get("artifact_refs", [])
    ]
