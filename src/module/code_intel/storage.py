from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.module.code_intel.models import (
    CodeEdge,
    CodeFile,
    Diagnostic,
    IndexSummary,
    RepositoryRecord,
    Symbol,
    SymbolKind,
)

SCHEMA_VERSION = 1


class SQLiteCodeIntelStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()

    def close(self) -> None:
        self._conn.close()

    def replace_repo_index(
        self,
        *,
        repository: RepositoryRecord,
        files: list[CodeFile],
        symbols: list[Symbol],
        edges: list[CodeEdge],
        diagnostics: list[Diagnostic],
    ) -> None:
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        try:
            cursor.execute(
                """
                INSERT INTO repositories
                    (repo_id, root_path, label, git_head, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    root_path = excluded.root_path,
                    label = excluded.label,
                    git_head = excluded.git_head,
                    indexed_at = excluded.indexed_at
                """,
                (
                    repository.repo_id,
                    repository.root_path,
                    repository.label,
                    repository.git_head,
                    repository.indexed_at.isoformat(),
                ),
            )
            for table in ("diagnostics", "edges", "symbols", "files"):
                cursor.execute(f"DELETE FROM {table} WHERE repo_id = ?", (repository.repo_id,))
            cursor.executemany(
                """
                INSERT INTO files
                    (file_id, repo_id, path, language, size_bytes, sha256)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (f.file_id, f.repo_id, f.path, f.language, f.size_bytes, f.sha256)
                    for f in files
                ],
            )
            cursor.executemany(
                """
                INSERT INTO symbols
                    (symbol_id, repo_id, file_path, kind, name, qualified_name,
                     start_line, end_line, parent_symbol_id, is_async, decorators_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.symbol_id,
                        s.repo_id,
                        s.file_path,
                        s.kind.value,
                        s.name,
                        s.qualified_name,
                        s.start_line,
                        s.end_line,
                        s.parent_symbol_id,
                        int(s.is_async),
                        json.dumps(s.decorators),
                    )
                    for s in symbols
                ],
            )
            cursor.executemany(
                """
                INSERT INTO edges
                    (edge_id, repo_id, file_path, kind, source_symbol_id, target_name, line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.edge_id,
                        e.repo_id,
                        e.file_path,
                        e.kind.value,
                        e.source_symbol_id,
                        e.target_name,
                        e.line,
                    )
                    for e in edges
                ],
            )
            cursor.executemany(
                """
                INSERT INTO diagnostics
                    (repo_id, file_path, severity, message, line)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (d.repo_id, d.file_path, d.severity, d.message, d.line)
                    for d in diagnostics
                ],
            )
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise

    def get_repository(self, repo_id: str) -> RepositoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM repositories WHERE repo_id = ?",
            (repo_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_repository(row)

    def get_index_summary(self, repo_id: str) -> IndexSummary | None:
        repo = self.get_repository(repo_id)
        if repo is None:
            return None
        counts = {
            name: self._conn.execute(
                f"SELECT COUNT(*) AS n FROM {name} WHERE repo_id = ?",
                (repo_id,),
            ).fetchone()["n"]
            for name in ("files", "symbols", "edges", "diagnostics")
        }
        rows = self._conn.execute(
            "SELECT DISTINCT language FROM files WHERE repo_id = ? ORDER BY language",
            (repo_id,),
        ).fetchall()
        return IndexSummary(
            repo_id=repo.repo_id,
            root_path=repo.root_path,
            label=repo.label,
            git_head=repo.git_head,
            indexed_at=repo.indexed_at,
            file_count=counts["files"],
            symbol_count=counts["symbols"],
            edge_count=counts["edges"],
            diagnostic_count=counts["diagnostics"],
            languages=[r["language"] for r in rows],
        )

    def find_symbols(
        self,
        *,
        repo_id: str,
        name: str | None = None,
        kind: SymbolKind | str | None = None,
        path_prefix: str | None = None,
        limit: int = 50,
    ) -> list[Symbol]:
        clauses = ["repo_id = ?"]
        args: list[object] = [repo_id]
        if name:
            clauses.append("(name = ? OR qualified_name = ?)")
            args.extend([name, name])
        if kind:
            kind_value = kind.value if isinstance(kind, SymbolKind) else str(kind)
            clauses.append("kind = ?")
            args.append(kind_value)
        if path_prefix:
            clauses.append("file_path LIKE ?")
            args.append(f"{path_prefix.rstrip('/')}%")
        args.append(max(1, min(limit, 500)))
        rows = self._conn.execute(
            f"""
            SELECT * FROM symbols
            WHERE {' AND '.join(clauses)}
            ORDER BY file_path, start_line, qualified_name
            LIMIT ?
            """,
            tuple(args),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def list_diagnostics(self, repo_id: str) -> list[Diagnostic]:
        rows = self._conn.execute(
            "SELECT * FROM diagnostics WHERE repo_id = ? ORDER BY file_path, line",
            (repo_id,),
        ).fetchall()
        return [self._row_to_diagnostic(r) for r in rows]

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS code_intel_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO code_intel_meta (key, value)
            VALUES ('schema_version', '1')
            ON CONFLICT(key) DO NOTHING;
            CREATE TABLE IF NOT EXISTS repositories (
                repo_id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                label TEXT NOT NULL,
                git_head TEXT,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                FOREIGN KEY (repo_id) REFERENCES repositories(repo_id)
            );
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent_symbol_id TEXT,
                is_async INTEGER NOT NULL DEFAULT 0,
                decorators_json TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY (repo_id) REFERENCES repositories(repo_id)
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_lookup
                ON symbols(repo_id, name, kind, file_path);
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_symbol_id TEXT,
                target_name TEXT NOT NULL,
                line INTEGER,
                FOREIGN KEY (repo_id) REFERENCES repositories(repo_id)
            );
            CREATE TABLE IF NOT EXISTS diagnostics (
                diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                line INTEGER,
                FOREIGN KEY (repo_id) REFERENCES repositories(repo_id)
            );
        """)

    @staticmethod
    def _row_to_repository(row: sqlite3.Row) -> RepositoryRecord:
        return RepositoryRecord(
            repo_id=row["repo_id"],
            root_path=row["root_path"],
            label=row["label"],
            git_head=row["git_head"],
            indexed_at=datetime.fromisoformat(row["indexed_at"]),
        )

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
        return Symbol(
            symbol_id=row["symbol_id"],
            repo_id=row["repo_id"],
            file_path=row["file_path"],
            kind=SymbolKind(row["kind"]),
            name=row["name"],
            qualified_name=row["qualified_name"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            parent_symbol_id=row["parent_symbol_id"],
            is_async=bool(row["is_async"]),
            decorators=json.loads(row["decorators_json"]),
        )

    @staticmethod
    def _row_to_diagnostic(row: sqlite3.Row) -> Diagnostic:
        return Diagnostic(
            repo_id=row["repo_id"],
            file_path=row["file_path"],
            severity=row["severity"],
            message=row["message"],
            line=row["line"],
        )
