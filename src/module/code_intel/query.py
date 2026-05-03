from __future__ import annotations

from src.module.code_intel.fragment import build_code_fragment
from src.module.code_intel.models import (
    CodeEdge,
    CodeFragment,
    EdgeKind,
    IndexSummary,
    Symbol,
    SymbolKind,
)
from src.module.code_intel.storage import SQLiteCodeIntelStore


class CodeIntelQuery:
    def __init__(self, store: SQLiteCodeIntelStore) -> None:
        self._store = store

    def get_index_summary(self, repo_id: str) -> IndexSummary | None:
        return self._store.get_index_summary(repo_id)

    def export_code_fragment(self, repo_id: str) -> CodeFragment | None:
        repository = self._store.get_repository(repo_id)
        if repository is None:
            return None
        return build_code_fragment(
            repository=repository,
            files=self._store.list_files(repo_id),
            symbols=self._store.list_symbols(repo_id),
            edges=self._store.list_all_edges(repo_id),
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
        return self._store.find_symbols(
            repo_id=repo_id,
            name=name,
            kind=kind,
            path_prefix=path_prefix,
            limit=limit,
        )

    def list_edges(
        self,
        *,
        repo_id: str,
        file_paths: list[str] | None = None,
        kind: EdgeKind | str | None = None,
        limit: int = 200,
    ) -> list[CodeEdge]:
        return self._store.list_edges(
            repo_id=repo_id,
            file_paths=file_paths,
            kind=kind,
            limit=limit,
        )

    def find_edges_by_target_names(
        self,
        *,
        repo_id: str,
        target_names: list[str],
        kind: EdgeKind | str | None = None,
        limit: int = 200,
    ) -> list[CodeEdge]:
        return self._store.find_edges_by_target_names(
            repo_id=repo_id,
            target_names=target_names,
            kind=kind,
            limit=limit,
        )
