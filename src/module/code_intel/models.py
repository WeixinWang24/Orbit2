from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"


class RepositoryRecord(BaseModel):
    repo_id: str
    root_path: str
    label: str
    git_head: str | None = None
    indexed_at: datetime


class CodeFile(BaseModel):
    file_id: str
    repo_id: str
    path: str
    language: str
    size_bytes: int
    sha256: str


class Symbol(BaseModel):
    symbol_id: str
    repo_id: str
    file_path: str
    kind: SymbolKind
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    parent_symbol_id: str | None = None
    is_async: bool = False
    decorators: list[str] = Field(default_factory=list)


class CodeEdge(BaseModel):
    edge_id: str
    repo_id: str
    file_path: str
    kind: EdgeKind
    source_symbol_id: str | None = None
    target_name: str
    line: int | None = None


class Diagnostic(BaseModel):
    repo_id: str
    file_path: str
    severity: str
    message: str
    line: int | None = None


class EvidenceRef(BaseModel):
    evidence_type: str
    repo_id: str
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    symbol_id: str | None = None


class ContextPack(BaseModel):
    pack_id: str
    repo_id: str
    summary: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class IndexSummary(BaseModel):
    repo_id: str
    root_path: str
    label: str
    git_head: str | None = None
    indexed_at: datetime
    file_count: int
    symbol_count: int
    edge_count: int
    diagnostic_count: int
    languages: list[str] = Field(default_factory=list)


class FileAnalysis(BaseModel):
    file: CodeFile
    symbols: list[Symbol] = Field(default_factory=list)
    edges: list[CodeEdge] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
