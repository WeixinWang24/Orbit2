from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "tcr") -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True)
class ToolchainArtifactRef:
    artifact_id: str
    step_id: str
    name: str
    media_type: str = "text/plain"
    truncated: bool = False
    original_chars: int = 0
    stored_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "step_id": self.step_id,
            "name": self.name,
            "media_type": self.media_type,
            "truncated": self.truncated,
            "original_chars": self.original_chars,
            "stored_chars": self.stored_chars,
        }


@dataclass(frozen=True)
class ToolchainEvidenceRef:
    evidence_id: str
    evidence_type: str
    target: dict[str, Any]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "target": self.target,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ToolchainFinding:
    severity: str
    message: str
    code: str | None = None
    file_path: str | None = None
    line: int | None = None
    column: int | None = None
    test_id: str | None = None
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "code": self.code,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "test_id": self.test_id,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ToolchainStep:
    step_id: str
    step_index: int
    name: str
    kind: str
    status: str
    started_at: str
    finished_at: str
    input_summary: str
    output_summary: str
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
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
            "evidence_refs": list(self.evidence_refs),
            "artifact_refs": list(self.artifact_refs),
            "error_summary": self.error_summary,
        }


@dataclass(frozen=True)
class ToolchainRun:
    run_id: str
    toolchain_name: str
    cwd: str
    request: dict[str, Any]
    status: str
    started_at: str
    finished_at: str | None = None
    report: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "toolchain_name": self.toolchain_name,
            "cwd": self.cwd,
            "request": self.request,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "report": self.report,
            "metadata": self.metadata,
        }
