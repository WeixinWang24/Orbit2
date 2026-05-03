from __future__ import annotations

from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l3_workflow.schemas import (
    DecisionRequestPackage,
    DecisionResponse,
    WorkflowResumeRequest,
    WorkflowRun,
    WorkflowStep,
    make_decision_id,
    make_workflow_run_id,
    now_iso,
)
from src.capability.mcp_servers.l3_workflow.store import (
    SQLiteWorkflowRunStore,
    default_workflow_db_path,
)


class WorkflowRunRecorder:
    def __init__(
        self,
        *,
        workflow_name: str,
        workspace_root: Path,
        request: dict[str, Any],
    ) -> None:
        self.workflow_run_id = make_workflow_run_id()
        self.workflow_name = workflow_name
        self.workspace_root = workspace_root
        self.started_at = now_iso()
        self._step_index = 0
        self._steps: list[dict[str, Any]] = []
        self._store = SQLiteWorkflowRunStore(default_workflow_db_path(workspace_root))
        self._store.save_run(
            WorkflowRun(
                workflow_run_id=self.workflow_run_id,
                workflow_name=workflow_name,
                cwd=str(workspace_root),
                request=request,
                status="running",
                started_at=self.started_at,
                metadata={"capability_layer": "workflow"},
            )
        )

    @property
    def store(self) -> SQLiteWorkflowRunStore:
        return self._store

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._steps)

    def record_step(
        self,
        *,
        step_id: str,
        name: str,
        kind: str,
        status: str,
        started_at: str,
        input_summary: str,
        output_summary: str,
        refs: list[dict[str, Any]] | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        self._step_index += 1
        step = WorkflowStep(
            step_id=step_id,
            step_index=self._step_index,
            name=name,
            kind=kind,
            status=status,
            started_at=started_at,
            finished_at=now_iso(),
            input_summary=input_summary,
            output_summary=output_summary,
            refs=refs or [],
            error_summary=error_summary,
        )
        step_dict = step.to_dict()
        self._store.save_step(self.workflow_run_id, step)
        self._steps.append(step_dict)
        return step_dict

    def create_decision_request(
        self,
        *,
        factual_context: dict[str, Any],
        artifact_refs: list[dict[str, Any]],
        admissible_options: list[dict[str, Any]],
        constraints: list[str],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        package = DecisionRequestPackage(
            workflow_run_id=self.workflow_run_id,
            decision_id=make_decision_id(),
            workflow_name=self.workflow_name,
            message_type="decision_request",
            factual_context=factual_context,
            artifact_refs=artifact_refs,
            admissible_options=admissible_options,
            constraints=constraints,
            response_schema=response_schema,
            created_at=now_iso(),
        )
        self._store.save_decision_request(package)
        return package.to_dict()

    def finalize_waiting(
        self,
        *,
        decision_request_package: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        self._store.save_run(
            WorkflowRun(
                workflow_run_id=self.workflow_run_id,
                workflow_name=self.workflow_name,
                cwd=str(self.workspace_root),
                request=report.get("request", {}),
                status="waiting_for_decision",
                started_at=self.started_at,
                current_decision_id=decision_request_package["decision_id"],
                report=report,
                metadata={"capability_layer": "workflow"},
            )
        )

    def close(self) -> None:
        self._store.close()


def validate_decision_response(
    *,
    decision_request_package: dict[str, Any],
    response: DecisionResponse | dict[str, Any],
) -> WorkflowResumeRequest:
    response_dict = response.to_dict() if isinstance(response, DecisionResponse) else dict(response)
    workflow_run_id = decision_request_package["workflow_run_id"]
    decision_id = decision_request_package["decision_id"]
    if response_dict.get("workflow_run_id") != workflow_run_id:
        raise ValueError("decision response workflow_run_id does not match request")
    if response_dict.get("decision_id") != decision_id:
        raise ValueError("decision response decision_id does not match request")
    options = {
        option["option_id"]: option
        for option in decision_request_package.get("admissible_options", [])
    }
    selected_option_id = response_dict.get("selected_option_id")
    if selected_option_id not in options:
        raise ValueError("decision response selected_option_id is not admissible")
    return WorkflowResumeRequest(
        workflow_run_id=workflow_run_id,
        decision_id=decision_id,
        selected_option=options[selected_option_id],
        decision_response=response_dict,
    )


def persist_decision_response(
    *,
    workspace_root: Path,
    decision_request_package: dict[str, Any],
    response: DecisionResponse | dict[str, Any],
) -> dict[str, Any]:
    resume = validate_decision_response(
        decision_request_package=decision_request_package,
        response=response,
    )
    response_dict = resume.decision_response
    decision_response = DecisionResponse(
        workflow_run_id=response_dict["workflow_run_id"],
        decision_id=response_dict["decision_id"],
        selected_option_id=response_dict["selected_option_id"],
        rationale_summary=response_dict.get("rationale_summary"),
        requested_evidence=response_dict.get("requested_evidence", []),
        metadata=response_dict.get("metadata", {}),
        created_at=response_dict.get("created_at", now_iso()),
    )
    store = SQLiteWorkflowRunStore(default_workflow_db_path(workspace_root))
    try:
        store.save_decision_response(decision_response)
    finally:
        store.close()
    return resume.to_dict()
