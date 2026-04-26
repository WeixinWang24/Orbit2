from __future__ import annotations

from pathlib import Path

from src.config.runtime import code_intel_db_path
from src.module.code_intel import (
    CodeIntelIndexer,
    CodeIntelQuery,
    EdgeKind,
    PythonAstAnalyzer,
    SQLiteCodeIntelStore,
    SymbolKind,
    scan_python_files,
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_python_ast_analyzer_extracts_symbols_imports_and_calls(tmp_path: Path) -> None:
    write(
        tmp_path / "pkg" / "service.py",
        """
import os
from pkg.models import User

def deco(fn):
    return fn

class Service:
    @deco
    async def run(self):
        helper()
        self.save()

def helper():
    User()
""".lstrip(),
    )

    analysis = PythonAstAnalyzer().analyze_file(
        repo_id="sample",
        root=tmp_path,
        path=tmp_path / "pkg" / "service.py",
    )

    symbols = {s.qualified_name: s for s in analysis.symbols}
    assert "pkg.service.Service" in symbols
    assert symbols["pkg.service.Service.run"].kind == SymbolKind.METHOD
    assert symbols["pkg.service.Service.run"].is_async is True
    assert symbols["pkg.service.Service.run"].decorators == ["deco"]
    assert symbols["pkg.service.helper"].kind == SymbolKind.FUNCTION

    imports = [e.target_name for e in analysis.edges if e.kind == EdgeKind.IMPORTS]
    calls = [e.target_name for e in analysis.edges if e.kind == EdgeKind.CALLS]
    assert imports == ["os", "pkg.models.User"]
    assert "helper" in calls
    assert "self.save" in calls
    assert "User" in calls


def test_indexer_persists_summary_and_symbol_queries(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write(repo / "app.py", "def main():\n    return 1\n")
    store = SQLiteCodeIntelStore(code_intel_db_path(tmp_path))
    try:
        CodeIntelIndexer(store).index_repo(repo_id="repo", root=repo, label="Fixture")
        query = CodeIntelQuery(store)

        summary = query.get_index_summary("repo")
        assert summary is not None
        assert summary.label == "Fixture"
        assert summary.file_count == 1
        assert summary.symbol_count == 2
        assert summary.edge_count == 0
        assert summary.diagnostic_count == 0
        assert summary.languages == ["python"]

        matches = query.find_symbols(repo_id="repo", name="main")
        assert len(matches) == 1
        assert matches[0].qualified_name == "app.main"
        assert query.find_symbols(repo_id="repo", kind=SymbolKind.FUNCTION)[0].name == "main"
        assert query.find_symbols(repo_id="repo", path_prefix="app")[0].file_path == "app.py"
    finally:
        store.close()


def test_reindex_replaces_prior_symbols(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = repo / "app.py"
    write(target, "def old():\n    return 1\n")
    store = SQLiteCodeIntelStore(tmp_path / "code_intel.db")
    try:
        indexer = CodeIntelIndexer(store)
        indexer.index_repo(repo_id="repo", root=repo)
        write(target, "def new():\n    return 2\n")
        indexer.index_repo(repo_id="repo", root=repo)

        query = CodeIntelQuery(store)
        assert query.find_symbols(repo_id="repo", name="old") == []
        assert len(query.find_symbols(repo_id="repo", name="new")) == 1
    finally:
        store.close()


def test_invalid_python_records_diagnostic_without_aborting(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write(repo / "good.py", "def ok():\n    return True\n")
    write(repo / "bad.py", "def nope(:\n")
    store = SQLiteCodeIntelStore(tmp_path / "code_intel.db")
    try:
        CodeIntelIndexer(store).index_repo(repo_id="repo", root=repo)
        query = CodeIntelQuery(store)
        summary = query.get_index_summary("repo")
        assert summary is not None
        assert summary.file_count == 2
        assert summary.diagnostic_count == 1
        assert len(query.find_symbols(repo_id="repo", name="ok")) == 1
        diagnostics = store.list_diagnostics("repo")
        assert diagnostics[0].file_path == "bad.py"
        assert diagnostics[0].severity == "error"
    finally:
        store.close()


def test_scan_python_files_ignores_generated_and_cache_dirs(tmp_path: Path) -> None:
    write(tmp_path / "src" / "keep.py", "")
    write(tmp_path / ".runtime" / "skip.py", "")
    write(tmp_path / "node_modules" / "skip.py", "")
    write(tmp_path / "pkg" / "__pycache__" / "skip.py", "")

    assert [p.relative_to(tmp_path.resolve()).as_posix() for p in scan_python_files(tmp_path)] == [
        "src/keep.py"
    ]
