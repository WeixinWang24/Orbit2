from __future__ import annotations

from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l2_toolchain.schemas import (
    ToolchainRun,
    ToolchainStep,
    make_run_id,
    now_iso,
)
from src.capability.mcp_servers.l2_toolchain.store import (
    SQLiteToolchainRunStore,
    default_toolchain_db_path,
)


class ToolchainRunRecorder:
    def __init__(
        self,
        *,
        toolchain_name: str,
        workspace_root: Path,
        request: dict[str, Any],
    ) -> None:
        self.run_id = make_run_id()
        self.toolchain_name = toolchain_name
        self.workspace_root = workspace_root
        self.started_at = now_iso()
        self._step_index = 0
        self._steps: list[dict[str, Any]] = []
        self._store = SQLiteToolchainRunStore(default_toolchain_db_path(workspace_root))
        self._store.save_run(
            ToolchainRun(
                run_id=self.run_id,
                toolchain_name=toolchain_name,
                cwd=str(workspace_root),
                request=request,
                status="running",
                started_at=self.started_at,
                metadata={"capability_layer": "toolchain"},
            )
        )

    @property
    def store(self) -> SQLiteToolchainRunStore:
        return self._store

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._steps)

    def save_artifact(
        self,
        *,
        step_id: str,
        artifact_id: str,
        name: str,
        content: str,
        media_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._store.save_artifact(
            run_id=self.run_id,
            step_id=step_id,
            artifact_id=artifact_id,
            name=name,
            content=content,
            media_type=media_type,
            metadata=metadata,
        ).to_dict()

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
        evidence_refs: list[dict[str, Any]] | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
        error_summary: str | None = None,
    ) -> dict[str, Any]:
        self._step_index += 1
        step = ToolchainStep(
            step_id=step_id,
            step_index=self._step_index,
            name=name,
            kind=kind,
            status=status,
            started_at=started_at,
            finished_at=now_iso(),
            input_summary=input_summary,
            output_summary=output_summary,
            evidence_refs=evidence_refs or [],
            artifact_refs=artifact_refs or [],
            error_summary=error_summary,
        )
        step_dict = step.to_dict()
        self._store.save_step(self.run_id, step)
        self._steps.append(step_dict)
        return step_dict

    def finalize(self, *, status: str, report: dict[str, Any]) -> None:
        self._store.save_run(
            ToolchainRun(
                run_id=self.run_id,
                toolchain_name=self.toolchain_name,
                cwd=str(self.workspace_root),
                request=report.get("request", {}),
                status=status,
                started_at=self.started_at,
                finished_at=now_iso(),
                report=report,
                metadata={"capability_layer": "toolchain"},
            )
        )

    def close(self) -> None:
        self._store.close()
