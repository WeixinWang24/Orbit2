from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.module.code_intel.base import LanguageAnalyzer
from src.module.code_intel.models import Diagnostic, RepositoryRecord
from src.module.code_intel.python_analyzer import PythonAstAnalyzer
from src.module.code_intel.storage import SQLiteCodeIntelStore

IGNORED_DIR_PARTS: frozenset[str] = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".runtime",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
})


def scan_python_files(root: Path) -> list[Path]:
    resolved = root.expanduser().resolve()
    files: list[Path] = []
    for path in resolved.rglob("*.py"):
        relative_parts = path.relative_to(resolved).parts
        if any(part in IGNORED_DIR_PARTS for part in relative_parts[:-1]):
            continue
        files.append(path)
    return sorted(files)


class CodeIntelIndexer:
    def __init__(
        self,
        store: SQLiteCodeIntelStore,
        *,
        python_analyzer: LanguageAnalyzer | None = None,
    ) -> None:
        self._store = store
        self._python = python_analyzer or PythonAstAnalyzer()

    def index_repo(
        self,
        *,
        repo_id: str,
        root: Path | str,
        label: str | None = None,
    ) -> RepositoryRecord:
        resolved = Path(root).expanduser().resolve()
        if not resolved.exists():
            raise ValueError("repo root does not exist")
        if not resolved.is_dir():
            raise ValueError("repo root is not a directory")

        files = []
        symbols = []
        edges = []
        diagnostics = []
        for path in scan_python_files(resolved):
            try:
                analysis = self._python.analyze_file(
                    repo_id=repo_id,
                    root=resolved,
                    path=path,
                )
            except UnicodeDecodeError as exc:
                relative = path.relative_to(resolved).as_posix()
                diagnostics.append(Diagnostic(
                    repo_id=repo_id,
                    file_path=relative,
                    severity="error",
                    message=str(exc),
                ))
                continue
            files.append(analysis.file)
            symbols.extend(analysis.symbols)
            edges.extend(analysis.edges)
            diagnostics.extend(analysis.diagnostics)

        repository = RepositoryRecord(
            repo_id=repo_id,
            root_path=str(resolved),
            label=label or repo_id,
            git_head=_git_head(resolved),
            indexed_at=datetime.now(timezone.utc),
        )
        self._store.replace_repo_index(
            repository=repository,
            files=files,
            symbols=symbols,
            edges=edges,
            diagnostics=diagnostics,
        )
        return repository


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None
