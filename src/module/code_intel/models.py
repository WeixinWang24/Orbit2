from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"


class CodeFragmentNodeKind(str, Enum):
    REPOSITORY = "repository"
    FILE = "file"
    SYMBOL = "symbol"


class CodeFragmentEdgeKind(str, Enum):
    CONTAINS = "contains"
    DEFINES = "defines"
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


class CodeFragmentSource(BaseModel):
    kind: str = "code_intel"
    repo_id: str
    root_path: str
    git_head: str | None = None
    indexed_at: datetime


class CodeFragmentNode(BaseModel):
    node_id: str
    kind: CodeFragmentNodeKind
    label: str
    source_path: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class CodeFragmentEdge(BaseModel):
    edge_id: str
    kind: CodeFragmentEdgeKind
    source_node_id: str
    target_node_id: str | None = None
    target_name: str | None = None
    confidence: float = 1.0
    reason: str = "code_intel"
    properties: dict[str, Any] = Field(default_factory=dict)


class CodeFragment(BaseModel):
    fragment_id: str
    repo_id: str
    source: CodeFragmentSource
    nodes: list[CodeFragmentNode] = Field(default_factory=list)
    edges: list[CodeFragmentEdge] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
