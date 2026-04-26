from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l2_toolchain.schemas import (
    ToolchainArtifactRef,
    ToolchainRun,
    ToolchainStep,
    now_iso,
)


TOOLCHAIN_DB_PATH_ENV = "ORBIT2_MCP_TOOLCHAIN_DB_PATH"
ARTIFACT_MAX_CHARS_ENV = "ORBIT2_MCP_TOOLCHAIN_ARTIFACT_MAX_CHARS"
ARTIFACT_REGION_MAX_CHARS_ENV = "ORBIT2_MCP_TOOLCHAIN_ARTIFACT_REGION_MAX_CHARS"
FALLBACK_ARTIFACT_MAX_CHARS = 200_000
FALLBACK_ARTIFACT_REGION_MAX_CHARS = 12_000


def positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def artifact_max_chars() -> int:
    return positive_int_env(
        ARTIFACT_MAX_CHARS_ENV,
        FALLBACK_ARTIFACT_MAX_CHARS,
        label="toolchain artifact max chars",
    )


def artifact_region_max_chars() -> int:
    return positive_int_env(
        ARTIFACT_REGION_MAX_CHARS_ENV,
        FALLBACK_ARTIFACT_REGION_MAX_CHARS,
        label="toolchain artifact region max chars",
    )


def default_toolchain_db_path(workspace_root: Path) -> Path:
    raw = os.environ.get(TOOLCHAIN_DB_PATH_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return workspace_root / ".runtime" / "toolchain_runs.sqlite3"


class SQLiteToolchainRunStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_tables()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS toolchain_runs (
                run_id TEXT PRIMARY KEY,
                toolchain_name TEXT NOT NULL,
                cwd TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                report_json TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS toolchain_steps (
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                input_summary TEXT NOT NULL DEFAULT '',
                output_summary TEXT NOT NULL DEFAULT '',
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                artifact_refs_json TEXT NOT NULL DEFAULT '[]',
                error_summary TEXT,
                PRIMARY KEY (run_id, step_id),
                FOREIGN KEY (run_id) REFERENCES toolchain_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_toolchain_steps_run
                ON toolchain_steps(run_id, step_index);
            CREATE TABLE IF NOT EXISTS toolchain_artifacts (
                run_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                content TEXT NOT NULL,
                truncated INTEGER NOT NULL DEFAULT 0,
                original_chars INTEGER NOT NULL DEFAULT 0,
                stored_chars INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (run_id, artifact_id),
                FOREIGN KEY (run_id) REFERENCES toolchain_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_toolchain_artifacts_run
                ON toolchain_artifacts(run_id, step_id);
        """)

    def save_run(self, run: ToolchainRun) -> None:
        self._conn.execute(
            """
            INSERT INTO toolchain_runs
                (run_id, toolchain_name, cwd, request_json, status,
                 started_at, finished_at, report_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status,
                finished_at = excluded.finished_at,
                report_json = excluded.report_json,
                metadata_json = excluded.metadata_json
            """,
            (
                run.run_id,
                run.toolchain_name,
                run.cwd,
                json.dumps(run.request),
                run.status,
                run.started_at,
                run.finished_at,
                json.dumps(run.report) if run.report is not None else None,
                json.dumps(run.metadata),
            ),
        )
        self._conn.commit()

    def save_step(self, run_id: str, step: ToolchainStep) -> None:
        self._conn.execute(
            """
            INSERT INTO toolchain_steps
                (run_id, step_id, step_index, name, kind, status,
                 started_at, finished_at, input_summary, output_summary,
                 evidence_refs_json, artifact_refs_json, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step.step_id,
                step.step_index,
                step.name,
                step.kind,
                step.status,
                step.started_at,
                step.finished_at,
                step.input_summary,
                step.output_summary,
                json.dumps(step.evidence_refs),
                json.dumps(step.artifact_refs),
                step.error_summary,
            ),
        )
        self._conn.commit()

    def save_artifact(
        self,
        *,
        run_id: str,
        step_id: str,
        artifact_id: str,
        name: str,
        content: str,
        media_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        max_chars: int | None = None,
    ) -> ToolchainArtifactRef:
        cap = max_chars if isinstance(max_chars, int) and max_chars > 0 else artifact_max_chars()
        original_chars = len(content)
        stored = content[:cap]
        truncated = original_chars > len(stored)
        self._conn.execute(
            """
            INSERT INTO toolchain_artifacts
                (run_id, artifact_id, step_id, name, media_type, content,
                 truncated, original_chars, stored_chars, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                artifact_id,
                step_id,
                name,
                media_type,
                stored,
                1 if truncated else 0,
                original_chars,
                len(stored),
                now_iso(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return ToolchainArtifactRef(
            artifact_id=artifact_id,
            step_id=step_id,
            name=name,
            media_type=media_type,
            truncated=truncated,
            original_chars=original_chars,
            stored_chars=len(stored),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM toolchain_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        result = self._row_to_run(row)
        result["steps"] = self.list_steps(run_id)
        return result

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM toolchain_steps
            WHERE run_id = ?
            ORDER BY step_index ASC
            """,
            (run_id,),
        ).fetchall()
        return [self._row_to_step(row) for row in rows]

    def get_step(self, run_id: str, step_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT * FROM toolchain_steps
            WHERE run_id = ? AND step_id = ?
            """,
            (run_id, step_id),
        ).fetchone()
        return None if row is None else self._row_to_step(row)

    def read_artifact_region(
        self,
        *,
        run_id: str,
        artifact_id: str,
        start_char: int = 0,
        max_chars: int | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(start_char, int) or start_char < 0:
            raise ValueError("start_char must be a non-negative integer")
        cap = artifact_region_max_chars()
        if isinstance(max_chars, int) and max_chars > 0:
            cap = min(max_chars, cap)
        row = self._conn.execute(
            """
            SELECT * FROM toolchain_artifacts
            WHERE run_id = ? AND artifact_id = ?
            """,
            (run_id, artifact_id),
        ).fetchone()
        if row is None:
            return None
        content = row["content"]
        region = content[start_char : start_char + cap]
        return {
            "run_id": run_id,
            "artifact_id": artifact_id,
            "step_id": row["step_id"],
            "name": row["name"],
            "media_type": row["media_type"],
            "content": region,
            "range": {
                "start_char": start_char,
                "end_char": start_char + len(region),
                "stored_chars": row["stored_chars"],
                "original_chars": row["original_chars"],
                "source_truncated": bool(row["truncated"]),
                "region_truncated": start_char + len(region) < row["stored_chars"],
            },
            "limits": {
                "max_chars": cap,
            },
            "metadata": json.loads(row["metadata_json"]),
        }

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "toolchain_name": row["toolchain_name"],
            "cwd": row["cwd"],
            "request": json.loads(row["request_json"]),
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "metadata": json.loads(row["metadata_json"]),
        }

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "step_id": row["step_id"],
            "step_index": row["step_index"],
            "name": row["name"],
            "kind": row["kind"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "input_summary": row["input_summary"],
            "output_summary": row["output_summary"],
            "evidence_refs": json.loads(row["evidence_refs_json"]),
            "artifact_refs": json.loads(row["artifact_refs_json"]),
            "error_summary": row["error_summary"],
        }
