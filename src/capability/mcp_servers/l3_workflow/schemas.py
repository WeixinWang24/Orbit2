from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_workflow_run_id(prefix: str = "wfr") -> str:
    return f"{prefix}_{uuid4().hex}"


def make_decision_id(prefix: str = "dec") -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    step_index: int
    name: str
    kind: str
    status: str
    started_at: str
    finished_at: str
    input_summary: str
    output_summary: str
    refs: list[dict[str, Any]] = field(default_factory=list)
    error_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_index": self.step_index,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "refs": list(self.refs),
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True)
class WorkflowRun:
    workflow_run_id: str
    workflow_name: str
    cwd: str
    request: dict[str, Any]
    status: str
    started_at: str
    current_decision_id: str | None = None
    finished_at: str | None = None
    report: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "workflow_name": self.workflow_name,
            "cwd": self.cwd,
            "request": self.request,
            "status": self.status,
            "started_at": self.started_at,
            "current_decision_id": self.current_decision_id,
            "finished_at": self.finished_at,
            "report": self.report,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DecisionOption:
    option_id: str
    label: str
    description: str
    branch_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "description": self.description,
            "branch_type": self.branch_type,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class DecisionRequestPackage:
    workflow_run_id: str
    decision_id: str
    workflow_name: str
    message_type: str
    factual_context: dict[str, Any]
    artifact_refs: list[dict[str, Any]]
    admissible_options: list[dict[str, Any]]
    constraints: list[str]
    response_schema: dict[str, Any]
    created_at: str
    status: str = "waiting_for_decision"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "decision_id": self.decision_id,
            "workflow_name": self.workflow_name,
            "message_type": self.message_type,
            "status": self.status,
            "factual_context": self.factual_context,
            "artifact_refs": list(self.artifact_refs),
            "admissible_options": list(self.admissible_options),
            "constraints": list(self.constraints),
            "response_schema": self.response_schema,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DecisionResponse:
    workflow_run_id: str
    decision_id: str
    selected_option_id: str
    rationale_summary: str | None = None
    requested_evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "decision_id": self.decision_id,
            "selected_option_id": self.selected_option_id,
            "rationale_summary": self.rationale_summary,
            "requested_evidence": list(self.requested_evidence),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WorkflowResumeRequest:
    workflow_run_id: str
    decision_id: str
    selected_option: dict[str, Any]
    decision_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "decision_id": self.decision_id,
            "selected_option": self.selected_option,
            "decision_response": self.decision_response,
        }
