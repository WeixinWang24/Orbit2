from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.module.code_intel.models import FileAnalysis


class LanguageAnalyzer(Protocol):
    language: str

    def analyze_file(self, *, repo_id: str, root: Path, path: Path) -> FileAnalysis: ...
