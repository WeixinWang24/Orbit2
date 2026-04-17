from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.core.runtime.models import ConversationMessage, MessageRole, Session, SessionStatus
from src.core.store.base import SessionStore


class SQLiteSessionStore(SessionStore):
    def __init__(self, db_path: str | Path) -> None:
        # db_path is required so the store never silently lands in a cwd-relative
        # default; src.config.runtime.default_db_path is the canonical source.
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                backend_name TEXT NOT NULL,
                system_prompt TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, turn_index);
        """)

    def save_session(self, session: Session) -> None:
        self._conn.execute(
            """
            INSERT INTO sessions
                (session_id, backend_name, system_prompt, status,
                 created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json
            """,
            (
                session.session_id,
                session.backend_name,
                session.system_prompt,
                session.status.value,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                json.dumps(session.metadata),
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_session(row)

    def list_sessions(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def save_message(self, message: ConversationMessage) -> None:
        self._conn.execute(
            """
            INSERT INTO messages
                (message_id, session_id, role, content,
                 turn_index, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.message_id,
                message.session_id,
                message.role.value,
                message.content,
                message.turn_index,
                message.created_at.isoformat(),
                json.dumps(message.metadata),
            ),
        )
        self._conn.commit()

    def list_messages(self, session_id: str) -> list[ConversationMessage]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def delete_all_sessions(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        try:
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM sessions")
            deleted = cursor.rowcount
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        return deleted

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            session_id=row["session_id"],
            backend_name=row["backend_name"],
            system_prompt=row["system_prompt"],
            status=SessionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            turn_index=row["turn_index"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"]),
        )
