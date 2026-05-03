from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from src.capability.mcp_servers.l3_workflow.schemas import (
    DecisionRequestPackage,
    DecisionResponse,
    WorkflowRun,
    WorkflowStep,
)


WORKFLOW_DB_PATH_ENV = "ORBIT2_MCP_WORKFLOW_DB_PATH"


def default_workflow_db_path(workspace_root: Path) -> Path:
    raw = os.environ.get(WORKFLOW_DB_PATH_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return workspace_root / ".runtime" / "workflow_runs.sqlite3"


class SQLiteWorkflowRunStore:
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
            CREATE TABLE IF NOT EXISTS workflow_runs (
                workflow_run_id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                cwd TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                current_decision_id TEXT,
                finished_at TEXT,
                report_json TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS workflow_steps (
                workflow_run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                input_summary TEXT NOT NULL DEFAULT '',
                output_summary TEXT NOT NULL DEFAULT '',
                refs_json TEXT NOT NULL DEFAULT '[]',
                error_summary TEXT,
                PRIMARY KEY (workflow_run_id, step_id),
                FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(workflow_run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_workflow_steps_run
                ON workflow_steps(workflow_run_id, step_index);
            CREATE TABLE IF NOT EXISTS workflow_decision_requests (
                workflow_run_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                package_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (workflow_run_id, decision_id),
                FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(workflow_run_id)
            );
            CREATE TABLE IF NOT EXISTS workflow_decision_responses (
                workflow_run_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (workflow_run_id, decision_id),
                FOREIGN KEY (workflow_run_id, decision_id)
                    REFERENCES workflow_decision_requests(workflow_run_id, decision_id)
            );
        """)

    def save_run(self, run: WorkflowRun) -> None:
        self._conn.execute(
            """
            INSERT INTO workflow_runs
                (workflow_run_id, workflow_name, cwd, request_json, status,
                 started_at, current_decision_id, finished_at, report_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_run_id) DO UPDATE SET
                status = excluded.status,
                current_decision_id = excluded.current_decision_id,
                finished_at = excluded.finished_at,
                report_json = excluded.report_json,
                metadata_json = excluded.metadata_json
            """,
            (
                run.workflow_run_id,
                run.workflow_name,
                run.cwd,
                json.dumps(run.request),
                run.status,
                run.started_at,
                run.current_decision_id,
                run.finished_at,
                json.dumps(run.report) if run.report is not None else None,
                json.dumps(run.metadata),
            ),
        )
        self._conn.commit()

    def save_step(self, workflow_run_id: str, step: WorkflowStep) -> None:
        self._conn.execute(
            """
            INSERT INTO workflow_steps
                (workflow_run_id, step_id, step_index, name, kind, status,
                 started_at, finished_at, input_summary, output_summary,
                 refs_json, error_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_run_id,
                step.step_id,
                step.step_index,
                step.name,
                step.kind,
                step.status,
                step.started_at,
                step.finished_at,
                step.input_summary,
                step.output_summary,
                json.dumps(step.refs),
                step.error_summary,
            ),
        )
        self._conn.commit()

    def save_decision_request(self, package: DecisionRequestPackage) -> None:
        self._conn.execute(
            """
            INSERT INTO workflow_decision_requests
                (workflow_run_id, decision_id, package_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                package.workflow_run_id,
                package.decision_id,
                json.dumps(package.to_dict()),
                package.created_at,
            ),
        )
        self._conn.commit()

    def save_decision_response(self, response: DecisionResponse) -> None:
        self._conn.execute(
            """
            INSERT INTO workflow_decision_responses
                (workflow_run_id, decision_id, response_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(workflow_run_id, decision_id) DO UPDATE SET
                response_json = excluded.response_json,
                created_at = excluded.created_at
            """,
            (
                response.workflow_run_id,
                response.decision_id,
                json.dumps(response.to_dict()),
                response.created_at,
            ),
        )
        self._conn.commit()

    def get_run(self, workflow_run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM workflow_runs WHERE workflow_run_id = ?",
            (workflow_run_id,),
        ).fetchone()
        if row is None:
            return None
        result = self._row_to_run(row)
        result["steps"] = self.list_steps(workflow_run_id)
        result["decision_requests"] = self.list_decision_requests(workflow_run_id)
        result["decision_responses"] = self.list_decision_responses(workflow_run_id)
        return result

    def list_steps(self, workflow_run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM workflow_steps
            WHERE workflow_run_id = ?
            ORDER BY step_index ASC
            """,
            (workflow_run_id,),
        ).fetchall()
        return [self._row_to_step(row) for row in rows]

    def get_decision_request(
        self, workflow_run_id: str, decision_id: str
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT * FROM workflow_decision_requests
            WHERE workflow_run_id = ? AND decision_id = ?
            """,
            (workflow_run_id, decision_id),
        ).fetchone()
        return None if row is None else json.loads(row["package_json"])

    def list_decision_requests(self, workflow_run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM workflow_decision_requests
            WHERE workflow_run_id = ?
            ORDER BY created_at ASC
            """,
            (workflow_run_id,),
        ).fetchall()
        return [json.loads(row["package_json"]) for row in rows]

    def list_decision_responses(self, workflow_run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM workflow_decision_responses
            WHERE workflow_run_id = ?
            ORDER BY created_at ASC
            """,
            (workflow_run_id,),
        ).fetchall()
        return [json.loads(row["response_json"]) for row in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "workflow_run_id": row["workflow_run_id"],
            "workflow_name": row["workflow_name"],
            "cwd": row["cwd"],
            "request": json.loads(row["request_json"]),
            "status": row["status"],
            "started_at": row["started_at"],
            "current_decision_id": row["current_decision_id"],
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
            "refs": json.loads(row["refs_json"]),
            "error_summary": row["error_summary"],
        }
