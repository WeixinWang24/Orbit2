"""L2 Code Intelligence MCP fact tools."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.timing import timed_mcp_tool

from src.capability.models import is_protected_relative_path
from src.config.runtime import code_intel_db_path
from src.module.code_intel import (
    CodeIntelIndexer,
    CodeIntelQuery,
    SQLiteCodeIntelStore,
)


SERVER_NAME = "code_intel"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DB_PATH_ENV = "ORBIT2_MCP_CODE_INTEL_DB_PATH"
MAX_SYMBOLS_ENV = "ORBIT2_MCP_CODE_INTEL_MAX_SYMBOLS"
MAX_EDGES_ENV = "ORBIT2_MCP_CODE_INTEL_MAX_EDGES"
MAX_FRAGMENT_NODES_ENV = "ORBIT2_MCP_CODE_INTEL_MAX_FRAGMENT_NODES"
MAX_FRAGMENT_EDGES_ENV = "ORBIT2_MCP_CODE_INTEL_MAX_FRAGMENT_EDGES"
FALLBACK_MAX_SYMBOLS = 80
FALLBACK_MAX_EDGES = 160
FALLBACK_MAX_FRAGMENT_NODES = 250
FALLBACK_MAX_FRAGMENT_EDGES = 400


mcp = FastMCP(SERVER_NAME)


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"code_intel MCP server requires workspace root via {WORKSPACE_ROOT_ENV} "
            "or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _db_path(root: Path) -> Path:
    override = os.environ.get(DB_PATH_ENV, "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return code_intel_db_path(root)


def _positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _max_symbols() -> int:
    return _positive_int_env(
        MAX_SYMBOLS_ENV,
        FALLBACK_MAX_SYMBOLS,
        label="code intel max symbols",
    )


def _max_edges() -> int:
    return _positive_int_env(
        MAX_EDGES_ENV,
        FALLBACK_MAX_EDGES,
        label="code intel max edges",
    )


def _max_fragment_nodes() -> int:
    return _positive_int_env(
        MAX_FRAGMENT_NODES_ENV,
        FALLBACK_MAX_FRAGMENT_NODES,
        label="code intel max fragment nodes",
    )


def _max_fragment_edges() -> int:
    return _positive_int_env(
        MAX_FRAGMENT_EDGES_ENV,
        FALLBACK_MAX_FRAGMENT_EDGES,
        label="code intel max fragment edges",
    )


def _resolve_workspace_path(path: str | None) -> Path:
    root = _workspace_root()
    raw = "." if path is None or not str(path).strip() else str(path).strip()
    candidate = Path(raw)
    if candidate.is_absolute():
        target = candidate.expanduser().resolve()
    else:
        target = (root / candidate).resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if relative not in ("", "."):
        matched = is_protected_relative_path(relative)
        if matched is not None:
            raise ValueError(f"path targets protected location: {matched}")
    return target


def _relative_path(path: str) -> str:
    root = _workspace_root()
    target = _resolve_workspace_path(path)
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if relative in ("", "."):
        return ""
    return relative


def _store() -> SQLiteCodeIntelStore:
    root = _workspace_root()
    return SQLiteCodeIntelStore(_db_path(root))


def _index_repo(
    *,
    repo_id: str,
    path: str = ".",
    label: str | None = None,
) -> dict[str, Any]:
    root = _resolve_workspace_path(path)
    if not root.exists():
        raise ValueError("index path does not exist")
    if not root.is_dir():
        raise ValueError("index path must be a directory")
    store = _store()
    try:
        repository = CodeIntelIndexer(store).index_repo(
            repo_id=repo_id,
            root=root,
            label=label,
        )
        query = CodeIntelQuery(store)
        summary = query.get_index_summary(repo_id)
        if summary is None:
            raise RuntimeError("index summary missing after indexing")
        return {
            "repository": repository.model_dump(mode="json"),
            "summary": summary.model_dump(mode="json"),
        }
    finally:
        store.close()


def _ensure_index(
    *,
    repo_id: str,
    path: str = ".",
    label: str | None = None,
    refresh_index: bool,
) -> None:
    store = _store()
    try:
        exists = CodeIntelQuery(store).get_index_summary(repo_id) is not None
    finally:
        store.close()
    if refresh_index or not exists:
        _index_repo(repo_id=repo_id, path=path, label=label)


def _audit() -> dict[str, str | list[str]]:
    return {
        "message_type": "fact_report",
        "fact_domain": "code_intel",
        "decision_posture": "non_decisional",
        "epistemic_posture": "observed_or_mechanically_derived",
        "capability_layer": "toolchain",
        "persistence": "sqlite_code_intel_cache",
    }


def _repository_summary_result(
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
) -> dict[str, Any]:
    indexed = _index_repo(repo_id=repo_id, path=path, label=label)
    return {
        "ok": True,
        "toolchain_name": "code_intel_repository_summary",
        "repo_id": repo_id,
        "summary": indexed["summary"],
        "repository": indexed["repository"],
        "audit": {
            **_audit(),
            "lower_level_reuse": ["code_intel.index"],
        },
    }


def _find_symbols_result(
    repo_id: str = "workspace",
    name: str | None = None,
    kind: str | None = None,
    path_prefix: str | None = None,
    limit: int | None = None,
    refresh_index: bool = True,
    path: str = ".",
    label: str | None = None,
) -> dict[str, Any]:
    if path_prefix:
        normalized_prefix = _relative_path(path_prefix)
    else:
        normalized_prefix = None
    cap = max(1, min(limit if limit is not None else _max_symbols(), _max_symbols()))
    _ensure_index(
        repo_id=repo_id,
        path=path,
        label=label,
        refresh_index=refresh_index,
    )
    store = _store()
    try:
        symbols = CodeIntelQuery(store).find_symbols(
            repo_id=repo_id,
            name=name,
            kind=kind,
            path_prefix=normalized_prefix,
            limit=cap,
        )
        return {
            "ok": True,
            "toolchain_name": "code_intel_find_symbols",
            "repo_id": repo_id,
            "query": {
                "name": name,
                "kind": kind,
                "path_prefix": normalized_prefix,
                "limit": cap,
            },
            "symbols": [symbol.model_dump(mode="json") for symbol in symbols],
            "summary": {
                "symbol_count": len(symbols),
                "truncated": len(symbols) >= cap,
            },
            "audit": {
                **_audit(),
                "lower_level_reuse": ["code_intel.index", "code_intel.query"],
            },
        }
    finally:
        store.close()


def _file_context_result(
    path: str,
    repo_id: str = "workspace",
    refresh_index: bool = True,
    label: str | None = None,
    max_symbols: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    relative = _relative_path(path)
    if not relative:
        raise ValueError("file context path must target a file")
    symbol_cap = max(
        1,
        min(max_symbols if max_symbols is not None else _max_symbols(), _max_symbols()),
    )
    edge_cap = max(1, min(max_edges if max_edges is not None else _max_edges(), _max_edges()))
    _ensure_index(
        repo_id=repo_id,
        path=".",
        label=label,
        refresh_index=refresh_index,
    )
    store = _store()
    try:
        query = CodeIntelQuery(store)
        symbols = [
            symbol
            for symbol in query.find_symbols(
                repo_id=repo_id,
                path_prefix=relative,
                limit=symbol_cap,
            )
            if symbol.file_path == relative
        ][:symbol_cap]
        edges = query.list_edges(
            repo_id=repo_id,
            file_paths=[relative],
            limit=edge_cap,
        )
        return {
            "ok": True,
            "toolchain_name": "code_intel_file_context",
            "repo_id": repo_id,
            "target": {"path": relative},
            "symbols": [symbol.model_dump(mode="json") for symbol in symbols],
            "imports": [
                edge.model_dump(mode="json")
                for edge in edges
                if edge.kind.value == "imports"
            ],
            "calls": [
                edge.model_dump(mode="json")
                for edge in edges
                if edge.kind.value == "calls"
            ],
            "summary": {
                "symbol_count": len(symbols),
                "edge_count": len(edges),
                "symbol_truncated": len(symbols) >= symbol_cap,
                "edge_truncated": len(edges) >= edge_cap,
            },
            "audit": {
                **_audit(),
                "lower_level_reuse": ["code_intel.index", "code_intel.query"],
            },
        }
    finally:
        store.close()


def _export_fragment_summary_result(
    repo_id: str = "workspace",
    path: str = ".",
    label: str | None = None,
    refresh_index: bool = True,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    node_cap = max(
        1,
        min(max_nodes if max_nodes is not None else _max_fragment_nodes(), _max_fragment_nodes()),
    )
    edge_cap = max(
        1,
        min(max_edges if max_edges is not None else _max_fragment_edges(), _max_fragment_edges()),
    )
    _ensure_index(
        repo_id=repo_id,
        path=path,
        label=label,
        refresh_index=refresh_index,
    )
    store = _store()
    try:
        fragment = CodeIntelQuery(store).export_code_fragment(repo_id)
        if fragment is None:
            return {
                "ok": False,
                "error": "code intel index not found",
                "repo_id": repo_id,
            }
        nodes = fragment.nodes[:node_cap]
        edges = fragment.edges[:edge_cap]
        return {
            "ok": True,
            "toolchain_name": "code_intel_export_fragment_summary",
            "repo_id": repo_id,
            "fragment_id": fragment.fragment_id,
            "source": fragment.source.model_dump(mode="json"),
            "summary": fragment.summary,
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "limits": {
                "max_nodes": node_cap,
                "max_edges": edge_cap,
                "nodes_returned": len(nodes),
                "edges_returned": len(edges),
                "nodes_truncated": len(fragment.nodes) > node_cap,
                "edges_truncated": len(fragment.edges) > edge_cap,
            },
            "audit": {
                **_audit(),
                "lower_level_reuse": ["code_intel.index", "code_intel.fragment"],
            },
        }
    finally:
        store.close()


@timed_mcp_tool(mcp, SERVER_NAME)
def code_intel_repository_summary(
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
) -> dict[str, Any]:
    """Index the workspace and return a compact Code Intelligence summary."""
    return _repository_summary_result(path=path, repo_id=repo_id, label=label)


@timed_mcp_tool(mcp, SERVER_NAME)
def code_intel_find_symbols(
    repo_id: str = "workspace",
    name: str | None = None,
    kind: str | None = None,
    path_prefix: str | None = None,
    limit: int | None = None,
    refresh_index: bool = True,
    path: str = ".",
    label: str | None = None,
) -> dict[str, Any]:
    """Find indexed symbols by name, kind, and path prefix."""
    return _find_symbols_result(
        repo_id=repo_id,
        name=name,
        kind=kind,
        path_prefix=path_prefix,
        limit=limit,
        refresh_index=refresh_index,
        path=path,
        label=label,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def code_intel_file_context(
    path: str,
    repo_id: str = "workspace",
    refresh_index: bool = True,
    label: str | None = None,
    max_symbols: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """Return symbols/imports/calls for one indexed workspace file."""
    return _file_context_result(
        path=path,
        repo_id=repo_id,
        refresh_index=refresh_index,
        label=label,
        max_symbols=max_symbols,
        max_edges=max_edges,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def code_intel_export_fragment_summary(
    repo_id: str = "workspace",
    path: str = ".",
    label: str | None = None,
    refresh_index: bool = True,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """Return a bounded detached Code Intelligence fragment summary."""
    return _export_fragment_summary_result(
        repo_id=repo_id,
        path=path,
        label=label,
        refresh_index=refresh_index,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


if __name__ == "__main__":
    mcp.run()
