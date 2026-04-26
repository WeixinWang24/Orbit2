from __future__ import annotations

from src.module.code_intel.models import IndexSummary, Symbol, SymbolKind
from src.module.code_intel.storage import SQLiteCodeIntelStore


class CodeIntelQuery:
    def __init__(self, store: SQLiteCodeIntelStore) -> None:
        self._store = store

    def get_index_summary(self, repo_id: str) -> IndexSummary | None:
        return self._store.get_index_summary(repo_id)

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
