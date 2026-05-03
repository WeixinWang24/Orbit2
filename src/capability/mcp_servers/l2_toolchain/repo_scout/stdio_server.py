"""L2 Repo Scout toolchain for summary-first git/code reconnaissance."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.mcp_servers.timing import timed_mcp_tool

from src.capability.models import is_protected_relative_path
from src.capability.mcp_servers.filesystem import stdio_server as raw_filesystem
from src.capability.mcp_servers.git import stdio_server as raw_git
from src.capability.mcp_servers.l1_structured.git import stdio_server as structured_git
from src.capability.mcp_servers.l2_toolchain.base import ToolchainRunRecorder
from src.capability.mcp_servers.l2_toolchain.schemas import now_iso
from src.capability.mcp_servers.l2_toolchain.store import (
    SQLiteToolchainRunStore,
    default_toolchain_db_path,
)
from src.config.runtime import code_intel_db_path
from src.module.code_intel import (
    CodeIntelIndexer,
    CodeIntelQuery,
    EdgeKind,
    SQLiteCodeIntelStore,
)


SERVER_NAME = "repo_scout"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
MAX_CHANGED_FILES_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_CHANGED_FILES"
MAX_SYMBOLS_PER_FILE_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_SYMBOLS_PER_FILE"
MAX_EDGES_PER_FILE_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_EDGES_PER_FILE"
MAX_OVERVIEW_TREE_DEPTH_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_OVERVIEW_TREE_DEPTH"
MAX_OVERVIEW_TREE_ENTRIES_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_OVERVIEW_TREE_ENTRIES"
MAX_OVERVIEW_SYMBOLS_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_OVERVIEW_SYMBOLS"
OVERVIEW_IGNORED_PARTS_ENV = "ORBIT2_MCP_REPO_SCOUT_OVERVIEW_IGNORED_PARTS"
OVERVIEW_IGNORED_SUFFIXES_ENV = "ORBIT2_MCP_REPO_SCOUT_OVERVIEW_IGNORED_SUFFIXES"
CHANGED_IGNORED_PARTS_ENV = "ORBIT2_MCP_REPO_SCOUT_CHANGED_IGNORED_PARTS"
CHANGED_IGNORED_SUFFIXES_ENV = "ORBIT2_MCP_REPO_SCOUT_CHANGED_IGNORED_SUFFIXES"
CHANGED_IGNORED_PATHS_ENV = "ORBIT2_MCP_REPO_SCOUT_CHANGED_IGNORED_PATHS"
MAX_DIFF_FILES_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_DIFF_FILES"
DIFF_SOURCE_MAX_CHARS_ENV = "ORBIT2_MCP_REPO_SCOUT_DIFF_SOURCE_MAX_CHARS"
MAX_IMPACT_FILES_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_IMPACT_FILES"
MAX_IMPACT_SYMBOLS_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_IMPACT_SYMBOLS"
MAX_IMPACT_EDGES_ENV = "ORBIT2_MCP_REPO_SCOUT_MAX_IMPACT_EDGES"
FALLBACK_MAX_CHANGED_FILES = 80
FALLBACK_MAX_SYMBOLS_PER_FILE = 20
FALLBACK_MAX_EDGES_PER_FILE = 30
FALLBACK_MAX_OVERVIEW_TREE_DEPTH = 3
FALLBACK_MAX_OVERVIEW_TREE_ENTRIES = 400
FALLBACK_MAX_OVERVIEW_SYMBOLS = 80
FALLBACK_MAX_DIFF_FILES = 80
FALLBACK_DIFF_SOURCE_MAX_CHARS = 200_000
FALLBACK_MAX_IMPACT_FILES = 40
FALLBACK_MAX_IMPACT_SYMBOLS = 80
FALLBACK_MAX_IMPACT_EDGES = 200
FALLBACK_OVERVIEW_IGNORED_PARTS = frozenset({
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
FALLBACK_OVERVIEW_IGNORED_SUFFIXES = frozenset({
    ".pyc",
    ".pyo",
})
FALLBACK_CHANGED_IGNORED_PARTS = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".runtime",
    "__pycache__",
})
FALLBACK_CHANGED_IGNORED_SUFFIXES = frozenset({
    ".pyc",
    ".pyo",
})
FALLBACK_CHANGED_IGNORED_PATHS = frozenset({
    ".DS_Store",
})


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"repo_scout MCP server requires workspace root via {WORKSPACE_ROOT_ENV} "
            "or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _positive_int_env(env_name: str, fallback: int, *, label: str) -> int:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _max_changed_files() -> int:
    return _positive_int_env(
        MAX_CHANGED_FILES_ENV,
        FALLBACK_MAX_CHANGED_FILES,
        label="repo scout max changed files",
    )


def _max_symbols_per_file() -> int:
    return _positive_int_env(
        MAX_SYMBOLS_PER_FILE_ENV,
        FALLBACK_MAX_SYMBOLS_PER_FILE,
        label="repo scout max symbols per file",
    )


def _max_edges_per_file() -> int:
    return _positive_int_env(
        MAX_EDGES_PER_FILE_ENV,
        FALLBACK_MAX_EDGES_PER_FILE,
        label="repo scout max edges per file",
    )


def _max_overview_tree_depth() -> int:
    return _positive_int_env(
        MAX_OVERVIEW_TREE_DEPTH_ENV,
        FALLBACK_MAX_OVERVIEW_TREE_DEPTH,
        label="repo scout max overview tree depth",
    )


def _max_overview_tree_entries() -> int:
    return _positive_int_env(
        MAX_OVERVIEW_TREE_ENTRIES_ENV,
        FALLBACK_MAX_OVERVIEW_TREE_ENTRIES,
        label="repo scout max overview tree entries",
    )


def _max_overview_symbols() -> int:
    return _positive_int_env(
        MAX_OVERVIEW_SYMBOLS_ENV,
        FALLBACK_MAX_OVERVIEW_SYMBOLS,
        label="repo scout max overview symbols",
    )


def _max_diff_files() -> int:
    return _positive_int_env(
        MAX_DIFF_FILES_ENV,
        FALLBACK_MAX_DIFF_FILES,
        label="repo scout max diff files",
    )


def _diff_source_max_chars() -> int:
    return _positive_int_env(
        DIFF_SOURCE_MAX_CHARS_ENV,
        FALLBACK_DIFF_SOURCE_MAX_CHARS,
        label="repo scout diff source max chars",
    )


def _max_impact_files() -> int:
    return _positive_int_env(
        MAX_IMPACT_FILES_ENV,
        FALLBACK_MAX_IMPACT_FILES,
        label="repo scout max impact files",
    )


def _max_impact_symbols() -> int:
    return _positive_int_env(
        MAX_IMPACT_SYMBOLS_ENV,
        FALLBACK_MAX_IMPACT_SYMBOLS,
        label="repo scout max impact symbols",
    )


def _max_impact_edges() -> int:
    return _positive_int_env(
        MAX_IMPACT_EDGES_ENV,
        FALLBACK_MAX_IMPACT_EDGES,
        label="repo scout max impact edges",
    )


def _string_set_env(env_name: str, fallback: frozenset[str]) -> frozenset[str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _overview_ignored_parts() -> frozenset[str]:
    return _string_set_env(OVERVIEW_IGNORED_PARTS_ENV, FALLBACK_OVERVIEW_IGNORED_PARTS)


def _overview_ignored_suffixes() -> frozenset[str]:
    return _string_set_env(
        OVERVIEW_IGNORED_SUFFIXES_ENV,
        FALLBACK_OVERVIEW_IGNORED_SUFFIXES,
    )


def _changed_ignored_parts() -> frozenset[str]:
    return _string_set_env(CHANGED_IGNORED_PARTS_ENV, FALLBACK_CHANGED_IGNORED_PARTS)


def _changed_ignored_suffixes() -> frozenset[str]:
    return _string_set_env(
        CHANGED_IGNORED_SUFFIXES_ENV,
        FALLBACK_CHANGED_IGNORED_SUFFIXES,
    )


def _changed_ignored_paths() -> frozenset[str]:
    return _string_set_env(CHANGED_IGNORED_PATHS_ENV, FALLBACK_CHANGED_IGNORED_PATHS)


def _is_overview_ignored_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        any(part in _overview_ignored_parts() for part in parts)
        or Path(path).suffix in _overview_ignored_suffixes()
    )


def _changed_path_kind(path: str) -> str:
    if path.endswith("/"):
        return "directory"
    target = (_workspace_root() / path).resolve()
    if target.exists() and target.is_dir():
        return "directory"
    return "file"


def _changed_ignore_reason(path: str) -> str | None:
    normalized = path.rstrip("/")
    parts = Path(normalized).parts
    if normalized in _changed_ignored_paths() or Path(path).name in _changed_ignored_paths():
        return "ignored_path"
    if is_protected_relative_path(normalized) is not None:
        return "protected_path"
    if any(part in _changed_ignored_parts() for part in parts):
        return "ignored_part"
    if Path(normalized).suffix in _changed_ignored_suffixes():
        return "ignored_suffix"
    return None


def _resolve_repo_path(path: str) -> tuple[Path, str]:
    raw = path.strip() if isinstance(path, str) else "."
    target = raw_filesystem._resolve_safe_path(raw or ".")
    if not target.is_dir():
        raise ValueError("repo scout path must resolve to a directory")
    workspace = _workspace_root()
    relative = target.relative_to(workspace).as_posix()
    normalized = "." if relative == "." else relative
    if normalized != "." and is_protected_relative_path(normalized) is not None:
        raise ValueError("repo scout path targets a protected location")
    return target, normalized


def _changed_manifest(*, cwd: str | None, include_untracked: bool) -> dict[str, Any]:
    status = raw_git._git_status_result(cwd=cwd)
    changed = raw_git._git_changed_files_result(cwd=cwd)
    raw_entries: list[dict[str, Any]] = []
    for entry in status["staged"]:
        raw_entries.append({"path": entry["path"], "state": "staged", "code": entry["code"]})
    for entry in status["unstaged"]:
        raw_entries.append({"path": entry["path"], "state": "unstaged", "code": entry["code"]})
    if include_untracked:
        for path in changed["untracked_files"]:
            raw_entries.append({"path": path, "state": "untracked", "code": "?"})

    normalized: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for item in sorted(
        {(entry["path"], entry["state"]): entry for entry in raw_entries}.values(),
        key=lambda entry: (entry["path"], entry["state"]),
    ):
        kind = _changed_path_kind(item["path"])
        reason = _changed_ignore_reason(item["path"])
        enriched = {**item, "kind": kind}
        if reason is not None:
            ignored.append({**enriched, "ignore_reason": reason})
        elif kind == "directory":
            directories.append(enriched)
        else:
            normalized.append(enriched)
    cap = _max_changed_files()
    capped_files = normalized[:cap]
    return {
        "status": status,
        "changed": changed,
        "files": capped_files,
        "entries": capped_files,
        "directories": directories,
        "ignored": ignored,
        "truncated": len(normalized) > cap,
        "total_file_state_count": len(normalized),
        "total_entry_state_count": len(raw_entries),
        "directory_state_count": len(directories),
        "ignored_state_count": len(ignored),
        "limits": {
            "max_changed_files": cap,
            "ignored_parts": sorted(_changed_ignored_parts()),
            "ignored_suffixes": sorted(_changed_ignored_suffixes()),
            "ignored_paths": sorted(_changed_ignored_paths()),
        },
    }


def _classify_path(path: str) -> dict[str, Any]:
    suffix = Path(path).suffix.lower()
    return {
        "path": path,
        "language": "python" if suffix == ".py" else "unknown",
        "is_python": suffix == ".py",
        "is_test": path.startswith("tests/") or Path(path).name.startswith("test_"),
    }


def _symbol_dict(symbol: Any) -> dict[str, Any]:
    return {
        "symbol_id": symbol.symbol_id,
        "kind": symbol.kind.value,
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "file_path": symbol.file_path,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "is_async": symbol.is_async,
        "decorators": list(symbol.decorators),
    }


def _edge_dict(edge: Any) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value,
        "file_path": edge.file_path,
        "target_name": edge.target_name,
        "line": edge.line,
        "source_symbol_id": edge.source_symbol_id,
    }


def _build_file_context(
    *,
    query: CodeIntelQuery,
    repo_id: str,
    file_path: str,
    max_symbols: int,
    max_edges: int,
) -> dict[str, Any]:
    classification = _classify_path(file_path)
    symbols: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if classification["is_python"]:
        symbols = [
            _symbol_dict(symbol)
            for symbol in query.find_symbols(
                repo_id=repo_id,
                path_prefix=file_path,
                limit=max_symbols,
            )
            if symbol.file_path == file_path
        ][:max_symbols]
        edges = [
            _edge_dict(edge)
            for edge in query.list_edges(
                repo_id=repo_id,
                file_paths=[file_path],
                limit=max_edges,
            )
        ]
    return {
        **classification,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "imports": [edge for edge in edges if edge["kind"] == EdgeKind.IMPORTS.value],
        "calls": [edge for edge in edges if edge["kind"] == EdgeKind.CALLS.value],
        "edge_count": len(edges),
    }


def _evidence_gaps(file_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for ctx in file_contexts:
        path = ctx["path"]
        if ctx["is_python"] and not ctx["symbols"]:
            gaps.append({
                "description": f"Changed Python file has no indexed symbols: {path}",
                "needed_evidence": "bounded file region or syntax diagnostic",
                "candidate_tool": "structured_filesystem.read_file_region",
                "target": {"path": path},
            })
        if not ctx["is_python"]:
            gaps.append({
                "description": f"Changed non-Python file needs direct evidence: {path}",
                "needed_evidence": "bounded file or diff region",
                "candidate_tool": "structured_filesystem.read_file_region",
                "target": {"path": path},
            })
    return gaps[:20]


def _hunk_line_stats(hunk: dict[str, Any]) -> dict[str, int]:
    additions = 0
    deletions = 0
    for line in hunk.get("content", "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return {"additions": additions, "deletions": deletions}


def _hunk_new_range(hunk: dict[str, Any]) -> dict[str, int]:
    start = int(hunk["new_start"])
    count = int(hunk["new_count"])
    end = start if count <= 0 else start + count - 1
    return {"start_line": start, "end_line": end}


def _symbols_touched_by_hunks(
    *,
    symbols: list[Any],
    hunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    touched: dict[str, dict[str, Any]] = {}
    for hunk in hunks:
        line_range = _hunk_new_range(hunk)
        for symbol in symbols:
            if symbol.kind.value == "module":
                continue
            if symbol.start_line <= line_range["end_line"] and symbol.end_line >= line_range["start_line"]:
                touched[symbol.symbol_id] = _symbol_dict(symbol)
    return list(touched.values())


def _classification_tags_for_file(
    *,
    path: str,
    state: str,
    touched_symbols: list[dict[str, Any]],
    additions: int,
    deletions: int,
) -> list[str]:
    flags: list[str] = []
    if state == "untracked":
        flags.append("untracked_file")
    if path.startswith("src/capability/mcp/") or "/mcp_servers/" in path:
        flags.append("capability_surface_or_mcp")
    if "governance" in path:
        flags.append("governance_surface")
    if "config" in path or path.endswith(".toml"):
        flags.append("configuration")
    if path.startswith("tests/") or Path(path).name.startswith("test_"):
        flags.append("test_surface")
    if additions + deletions >= 80:
        flags.append("large_diff")
    if state != "untracked" and not touched_symbols and path.endswith(".py"):
        flags.append("no_symbol_overlap")
    return flags


def _evidence_read_candidates(
    *,
    path: str,
    state: str,
    hunk_count: int,
    touched_symbols: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if state == "untracked":
        return [{
            "tool": "structured_filesystem.read_file_region",
            "target": {"path": path, "start_line": 1},
            "basis": "untracked file has no git diff hunk",
        }]
    if hunk_count > 0:
        return [{
            "tool": "structured_git.read_diff_hunk",
            "target": {"path": path, "hunk_index": 1},
            "basis": "first changed hunk exists in collected diff",
        }]
    if touched_symbols:
        first = touched_symbols[0]
        return [{
            "tool": "structured_filesystem.read_file_region",
            "target": {
                "path": path,
                "start_line": first["start_line"],
                "end_line": first["end_line"],
            },
            "basis": "touched symbol range is available from Code Intel",
        }]
    return []


def _fact_report_audit_metadata() -> dict[str, str]:
    return {
        "message_type": "fact_report",
        "fact_domain": "repo_recon",
        "decision_posture": "non_decisional",
        "epistemic_posture": "observed_or_mechanically_derived",
    }


def _collect_file_diff(entry: dict[str, Any], *, cwd: str | None) -> dict[str, Any]:
    if entry["state"] == "untracked":
        return {
            "path": entry["path"],
            "state": entry["state"],
            "diff": "",
            "hunks": [],
            "truncated": False,
            "diff_available": False,
            "unavailable_reason": "untracked_file",
        }
    diff = raw_git._git_diff_result(
        cwd=cwd,
        path=entry["path"],
        staged=entry["state"] == "staged",
        max_chars=_diff_source_max_chars(),
    )
    hunks = structured_git._parse_diff_hunks(diff["diff"])
    return {
        "path": entry["path"],
        "state": entry["state"],
        "diff": diff["diff"],
        "hunks": hunks,
        "truncated": diff["truncated"],
        "diff_available": bool(diff["diff"]),
    }


def _build_diff_digest(
    *,
    query: CodeIntelQuery,
    repo_id: str,
    manifest: dict[str, Any],
    file_diffs: list[dict[str, Any]],
) -> dict[str, Any]:
    file_entries = {entry["path"]: entry for entry in manifest["files"]}
    files: list[dict[str, Any]] = []
    summary_counts = Counter()
    classification_counts = Counter()
    for file_diff in file_diffs:
        path = file_diff["path"]
        entry = file_entries[path]
        hunks = file_diff["hunks"]
        stats = Counter()
        for hunk in hunks:
            stats.update(_hunk_line_stats(hunk))
        symbols = query.find_symbols(repo_id=repo_id, path_prefix=path, limit=200)
        file_symbols = [symbol for symbol in symbols if symbol.file_path == path]
        touched_symbols = _symbols_touched_by_hunks(symbols=file_symbols, hunks=hunks)
        classification_tags = _classification_tags_for_file(
            path=path,
            state=entry["state"],
            touched_symbols=touched_symbols,
            additions=stats["additions"],
            deletions=stats["deletions"],
        )
        classification_counts.update(classification_tags)
        summary_counts["files"] += 1
        summary_counts["hunks"] += len(hunks)
        summary_counts["additions"] += stats["additions"]
        summary_counts["deletions"] += stats["deletions"]
        files.append({
            "path": path,
            "state": entry["state"],
            "code": entry["code"],
            "language": _classify_path(path)["language"],
            "diff_available": file_diff["diff_available"],
            "diff_truncated": file_diff["truncated"],
            "hunk_count": len(hunks),
            "additions": stats["additions"],
            "deletions": stats["deletions"],
            "hunks": [
                {
                    "hunk_index": index,
                    "old_start": hunk["old_start"],
                    "old_count": hunk["old_count"],
                    "new_start": hunk["new_start"],
                    "new_count": hunk["new_count"],
                    "section": hunk["section"],
                    "line_count": hunk["line_count"],
                    **_hunk_line_stats(hunk),
                }
                for index, hunk in enumerate(hunks, start=1)
            ],
            "touched_symbols": touched_symbols,
            "classification_tags": classification_tags,
            "evidence_read_candidates": _evidence_read_candidates(
                path=path,
                state=entry["state"],
                hunk_count=len(hunks),
                touched_symbols=touched_symbols,
            ),
        })
    return {
        "summary": {
            "file_count": summary_counts["files"],
            "hunk_count": summary_counts["hunks"],
            "additions": summary_counts["additions"],
            "deletions": summary_counts["deletions"],
            "classification_tag_counts": dict(sorted(classification_counts.items())),
        },
        "files": files,
        "directories": manifest["directories"],
        "ignored": manifest["ignored"],
    }


def _symbol_target_names(symbols: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for symbol in symbols:
        name = symbol.get("name")
        qualified = symbol.get("qualified_name")
        if isinstance(name, str) and name:
            names.add(name)
        if isinstance(qualified, str) and qualified:
            names.add(qualified)
            names.add(qualified.rsplit(".", 1)[-1])
    return sorted(names)


def _edge_fact(edge: Any, *, relation: str) -> dict[str, Any]:
    payload = _edge_dict(edge)
    payload["relation"] = relation
    payload["source_is_test"] = _classify_path(edge.file_path)["is_test"]
    return payload


def _build_impact_scope(
    *,
    query: CodeIntelQuery,
    repo_id: str,
    manifest: dict[str, Any],
    file_diffs: list[dict[str, Any]],
    max_symbols: int,
    max_edges: int,
) -> dict[str, Any]:
    file_entries = {entry["path"]: entry for entry in manifest["files"]}
    changed_symbol_items: list[dict[str, Any]] = []
    symbol_facts_by_id: dict[str, dict[str, Any]] = {}
    files_without_symbol_overlap: list[dict[str, Any]] = []
    outgoing_edges: list[dict[str, Any]] = []
    changed_file_paths: set[str] = set()

    for file_diff in file_diffs:
        path = file_diff["path"]
        changed_file_paths.add(path)
        entry = file_entries[path]
        symbols = query.find_symbols(repo_id=repo_id, path_prefix=path, limit=max_symbols)
        file_symbols = [symbol for symbol in symbols if symbol.file_path == path]
        touched_symbols = _symbols_touched_by_hunks(
            symbols=file_symbols,
            hunks=file_diff["hunks"],
        )
        if entry["state"] == "untracked" and not touched_symbols:
            touched_symbols = [
                _symbol_dict(symbol)
                for symbol in file_symbols
                if symbol.kind.value != "module"
            ][:max_symbols]
        remaining_symbols = max(0, max_symbols - len(symbol_facts_by_id))
        touched_symbols = [
            symbol
            for symbol in touched_symbols
            if symbol["symbol_id"] not in symbol_facts_by_id
        ][:remaining_symbols]

        if file_diff["hunks"] and not touched_symbols and path.endswith(".py"):
            files_without_symbol_overlap.append({
                "path": path,
                "state": entry["state"],
                "hunk_count": len(file_diff["hunks"]),
                "basis": "changed Python hunks did not overlap indexed non-module symbols",
            })

        hunk_ranges = [
            {
                "hunk_index": index,
                **_hunk_new_range(hunk),
            }
            for index, hunk in enumerate(file_diff["hunks"], start=1)
        ]
        changed_symbol_items.append({
            "path": path,
            "state": entry["state"],
            "hunk_count": len(file_diff["hunks"]),
            "hunk_ranges": hunk_ranges,
            "symbols": touched_symbols,
            "basis": (
                "untracked file indexed symbols"
                if entry["state"] == "untracked"
                else "hunk range overlap with indexed symbols"
            ),
        })
        for symbol in touched_symbols:
            symbol_facts_by_id[symbol["symbol_id"]] = symbol

        remaining = max(0, max_edges - len(outgoing_edges))
        if remaining:
            outgoing_edges.extend([
                _edge_fact(edge, relation="edge_from_changed_file")
                for edge in query.list_edges(
                    repo_id=repo_id,
                    file_paths=[path],
                    limit=remaining,
                )
            ])

    target_names = _symbol_target_names(list(symbol_facts_by_id.values()))
    incoming_edges = [
        _edge_fact(edge, relation="edge_target_name_matches_changed_symbol")
        for edge in query.find_edges_by_target_names(
            repo_id=repo_id,
            target_names=target_names,
            limit=max_edges,
        )
    ]
    test_adjacency = [
        edge
        for edge in incoming_edges + outgoing_edges
        if edge["source_is_test"] or edge["file_path"].startswith("tests/")
    ]
    changed_test_files = [
        entry
        for entry in manifest["files"]
        if _classify_path(entry["path"])["is_test"]
    ]
    return {
        "summary": {
            "changed_file_count": len(file_diffs),
            "changed_symbol_count": len(symbol_facts_by_id),
            "outgoing_edge_count": len(outgoing_edges),
            "incoming_edge_count": len(incoming_edges),
            "test_adjacency_count": len(test_adjacency),
            "files_without_symbol_overlap_count": len(files_without_symbol_overlap),
        },
        "changed_symbols": changed_symbol_items,
        "adjacency": {
            "outgoing_edges": outgoing_edges,
            "incoming_references": incoming_edges,
            "target_names": target_names,
        },
        "test_adjacency": {
            "changed_test_files": changed_test_files,
            "edge_facts": test_adjacency,
        },
        "files_without_symbol_overlap": files_without_symbol_overlap,
        "directories": manifest["directories"],
        "ignored": manifest["ignored"],
        "boundary": {
            "scope_depth": "direct_edge_name_match_only",
            "decision_posture": "non_decisional",
        },
    }


def _repository_tree_manifest(
    *,
    path: str,
    max_depth: int,
    max_entries: int,
) -> dict[str, Any]:
    raw_tree = raw_filesystem._directory_tree_result(
        path,
        max_depth=max_depth,
        max_entries=max_entries,
    )
    entries = [
        entry
        for entry in raw_tree["entries"]
        if not _is_overview_ignored_path(entry["path"])
    ]
    top_level_dirs = sorted({
        Path(entry["path"]).parts[0]
        for entry in entries
        if entry["kind"] == "directory" and Path(entry["path"]).parts
    })
    file_entries = [entry for entry in entries if entry["kind"] == "file"]
    suffix_counts = Counter(Path(entry["path"]).suffix or "(none)" for entry in file_entries)
    return {
        "ok": True,
        "path": raw_tree["path"],
        "entries": entries,
        "entry_count": len(entries),
        "raw_entry_count": raw_tree["entry_count"],
        "truncated": raw_tree["truncated"],
        "top_level_dirs": top_level_dirs,
        "file_extension_counts": dict(sorted(suffix_counts.items())),
        "limits": {
            "max_depth": max_depth,
            "max_entries": max_entries,
            "ignored_parts": sorted(_overview_ignored_parts()),
            "ignored_suffixes": sorted(_overview_ignored_suffixes()),
        },
    }


def _orientation_candidates(
    *,
    tree_manifest: dict[str, Any],
    symbol_file_counts: Counter[str],
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    entries = tree_manifest["entries"]
    file_paths = {entry["path"] for entry in entries if entry["kind"] == "file"}
    candidates: list[dict[str, Any]] = []
    priority_names = (
        "README.md",
        "pyproject.toml",
        "setup.py",
        "main.py",
        "__main__.py",
        "src/operation/cli/harness.py",
    )
    for name in priority_names:
        if name in file_paths:
            candidates.append({
                "path": name,
                "basis": "standard orientation file",
                "candidate_tool": "structured_filesystem.read_file_region",
            })
    for path, count in symbol_file_counts.most_common(max_candidates):
        if path in file_paths and all(candidate["path"] != path for candidate in candidates):
            candidates.append({
                "path": path,
                "basis": f"high indexed symbol density ({count} symbols)",
                "candidate_tool": "structured_filesystem.read_file_region",
            })
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates]


def _build_repository_overview(
    *,
    tree_manifest: dict[str, Any],
    status: dict[str, Any],
    log: dict[str, Any],
    repository: Any,
    summary: Any,
    symbols: list[Any],
) -> dict[str, Any]:
    symbol_kind_counts = Counter(symbol.kind.value for symbol in symbols)
    symbol_file_counts = Counter(
        symbol.file_path
        for symbol in symbols
        if symbol.kind.value != "module"
    )
    top_symbol_files = [
        {"path": path, "symbol_count": count}
        for path, count in symbol_file_counts.most_common(20)
    ]
    overview_symbols = [
        _symbol_dict(symbol)
        for symbol in symbols
        if symbol.kind.value != "module"
    ][:40]
    return {
        "repository": repository.model_dump(mode="json"),
        "index_summary": summary.model_dump(mode="json") if summary is not None else None,
        "git": {
            "branch": status["branch"],
            "clean": status["clean"],
            "staged_count": status["staged_count"],
            "unstaged_count": status["unstaged_count"],
            "untracked_count": status["untracked_count"],
            "recent_commits": log.get("commits", []),
        },
        "tree": {
            "entry_count": tree_manifest["entry_count"],
            "raw_entry_count": tree_manifest["raw_entry_count"],
            "truncated": tree_manifest["truncated"],
            "top_level_dirs": tree_manifest["top_level_dirs"],
            "file_extension_counts": tree_manifest["file_extension_counts"],
        },
        "code_intel": {
            "symbol_kind_counts": dict(sorted(symbol_kind_counts.items())),
            "top_symbol_files": top_symbol_files,
            "sample_symbols": overview_symbols,
        },
        "orientation_candidates": _orientation_candidates(
            tree_manifest=tree_manifest,
            symbol_file_counts=symbol_file_counts,
        ),
    }


def _repo_scout_repository_overview_result(
    *,
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    repo_root, relative_path = _resolve_repo_path(path)
    depth = (
        max_tree_depth
        if isinstance(max_tree_depth, int) and max_tree_depth > 0
        else _max_overview_tree_depth()
    )
    entries_cap = (
        max_tree_entries
        if isinstance(max_tree_entries, int) and max_tree_entries > 0
        else _max_overview_tree_entries()
    )
    symbol_cap = (
        max_symbols
        if isinstance(max_symbols, int) and max_symbols > 0
        else _max_overview_symbols()
    )
    request = {
        "path": path,
        "repo_id": repo_id,
        "label": label,
        "max_tree_depth": max_tree_depth,
        "max_tree_entries": max_tree_entries,
        "max_symbols": max_symbols,
    }
    recorder = ToolchainRunRecorder(
        toolchain_name="repo_scout_repository_overview",
        workspace_root=workspace,
        request=request,
    )
    try:
        tree_started = now_iso()
        tree_manifest = _repository_tree_manifest(
            path=relative_path,
            max_depth=depth,
            max_entries=entries_cap,
        )
        tree_ref = recorder.save_artifact(
            step_id="step_001_collect_repository_tree",
            artifact_id="artifact_001_repository_tree",
            name="repository_tree",
            content=json.dumps(tree_manifest, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_001_collect_repository_tree",
            name="collect repository tree manifest",
            kind="raw_primitive",
            status="completed",
            started_at=tree_started,
            input_summary=f"path={relative_path!r} max_depth={depth} max_entries={entries_cap}",
            output_summary=(
                f"entries={tree_manifest['entry_count']} "
                f"truncated={tree_manifest['truncated']}"
            ),
            artifact_refs=[tree_ref],
        )

        git_started = now_iso()
        git_cwd = None if relative_path == "." else relative_path
        status = raw_git._git_status_result(cwd=git_cwd)
        log = raw_git._git_log_result(cwd=git_cwd, limit=5)
        recorder.record_step(
            step_id="step_002_collect_git_orientation",
            name="collect git orientation",
            kind="raw_primitive",
            status="completed",
            started_at=git_started,
            input_summary=f"cwd={git_cwd!r}",
            output_summary=(
                f"branch={status['branch']!r} clean={status['clean']} "
                f"commits={log['commit_count']}"
            ),
        )

        index_started = now_iso()
        store = SQLiteCodeIntelStore(code_intel_db_path(workspace))
        try:
            repository = CodeIntelIndexer(store).index_repo(
                repo_id=repo_id,
                root=repo_root,
                label=label or repo_id,
            )
            query = CodeIntelQuery(store)
            summary = query.get_index_summary(repo_id)
            symbols = query.find_symbols(repo_id=repo_id, limit=symbol_cap)
        finally:
            store.close()
        recorder.record_step(
            step_id="step_003_index_code_intel",
            name="index code intelligence",
            kind="code_intelligence",
            status="completed",
            started_at=index_started,
            input_summary=f"repo_id={repo_id!r} root={repo_root}",
            output_summary=(
                f"files={summary.file_count if summary else 0} "
                f"symbols={summary.symbol_count if summary else 0}"
            ),
        )

        overview_started = now_iso()
        overview = _build_repository_overview(
            tree_manifest=tree_manifest,
            status=status,
            log=log,
            repository=repository,
            summary=summary,
            symbols=symbols,
        )
        overview_ref = recorder.save_artifact(
            step_id="step_004_build_repository_overview",
            artifact_id="artifact_002_repository_overview",
            name="repository_overview",
            content=json.dumps(overview, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_004_build_repository_overview",
            name="build repository overview",
            kind="toolchain",
            status="completed",
            started_at=overview_started,
            input_summary=f"symbols={len(symbols)} entries={tree_manifest['entry_count']}",
            output_summary=(
                f"candidates={len(overview['orientation_candidates'])} "
                f"top_dirs={len(overview['tree']['top_level_dirs'])}"
            ),
            artifact_refs=[overview_ref],
        )

        report: dict[str, Any] = {
            "ok": True,
            "run_id": recorder.run_id,
            "status": "completed",
            "toolchain_name": "repo_scout_repository_overview",
            "cwd": str(workspace),
            "request": request,
            "summary": {
                "path": relative_path,
                "branch": status["branch"],
                "clean": status["clean"],
                "tree_entry_count": overview["tree"]["entry_count"],
                "tree_truncated": overview["tree"]["truncated"],
                "indexed_file_count": summary.file_count if summary else 0,
                "indexed_symbol_count": summary.symbol_count if summary else 0,
                "orientation_candidate_count": len(overview["orientation_candidates"]),
            },
            "overview": overview,
            "trace": recorder.steps,
            "trace_available": True,
            "artifact_refs": [tree_ref, overview_ref],
            "limits": {
                "max_tree_depth": depth,
                "max_tree_entries": entries_cap,
                "max_symbols": symbol_cap,
                "ignored_parts": sorted(_overview_ignored_parts()),
                "ignored_suffixes": sorted(_overview_ignored_suffixes()),
            },
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
                "lower_level_reuse": [
                    "filesystem.directory_tree",
                    "git.status",
                    "git.log",
                    "code_intel.index",
                ],
            },
        }
        recorder.finalize(status="completed", report=report)
        return report
    except Exception as exc:
        report = {
            "ok": False,
            "run_id": recorder.run_id,
            "status": "error",
            "toolchain_name": "repo_scout_repository_overview",
            "cwd": str(workspace),
            "request": request,
            "summary": f"repo overview errored: {exc}",
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status="error", report=report)
        return report
    finally:
        recorder.close()


def _repo_scout_diff_digest_result(
    *,
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_diff_files: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    file_cap = (
        max_diff_files
        if isinstance(max_diff_files, int) and max_diff_files > 0
        else _max_diff_files()
    )
    request = {
        "cwd": cwd,
        "repo_id": repo_id,
        "label": label,
        "include_untracked": include_untracked,
        "max_diff_files": max_diff_files,
    }
    recorder = ToolchainRunRecorder(
        toolchain_name="repo_scout_diff_digest",
        workspace_root=workspace,
        request=request,
    )
    try:
        manifest_started = now_iso()
        manifest = _changed_manifest(cwd=cwd, include_untracked=include_untracked)
        manifest_ref = recorder.save_artifact(
            step_id="step_001_collect_changed_manifest",
            artifact_id="artifact_001_changed_manifest",
            name="changed_manifest",
            content=json.dumps(manifest, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_001_collect_changed_manifest",
            name="collect normalized changed manifest",
            kind="raw_primitive",
            status="completed",
            started_at=manifest_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"files={len(manifest['files'])} directories={len(manifest['directories'])} "
                f"ignored={len(manifest['ignored'])}"
            ),
            artifact_refs=[manifest_ref],
        )

        diff_started = now_iso()
        selected_entries = manifest["files"][:file_cap]
        file_diffs = [
            _collect_file_diff(entry, cwd=cwd)
            for entry in selected_entries
        ]
        diffs_ref = recorder.save_artifact(
            step_id="step_002_collect_file_diffs",
            artifact_id="artifact_002_file_diffs",
            name="file_diffs",
            content=json.dumps(file_diffs, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_002_collect_file_diffs",
            name="collect per-file diffs",
            kind="raw_primitive",
            status="completed",
            started_at=diff_started,
            input_summary=f"files={len(selected_entries)} max_diff_files={file_cap}",
            output_summary=(
                f"diffs={sum(1 for item in file_diffs if item['diff_available'])} "
                f"untracked={sum(1 for item in file_diffs if item['state'] == 'untracked')}"
            ),
            artifact_refs=[diffs_ref],
        )

        index_started = now_iso()
        store = SQLiteCodeIntelStore(code_intel_db_path(workspace))
        try:
            repository = CodeIntelIndexer(store).index_repo(
                repo_id=repo_id,
                root=workspace,
                label=label or repo_id,
            )
            query = CodeIntelQuery(store)
            summary = query.get_index_summary(repo_id)
            digest = _build_diff_digest(
                query=query,
                repo_id=repo_id,
                manifest=manifest,
                file_diffs=file_diffs,
            )
        finally:
            store.close()
        recorder.record_step(
            step_id="step_003_index_and_digest",
            name="index code intelligence and build diff digest",
            kind="code_intelligence",
            status="completed",
            started_at=index_started,
            input_summary=f"repo_id={repo_id!r} root={workspace}",
            output_summary=(
                f"files={digest['summary']['file_count']} "
                f"hunks={digest['summary']['hunk_count']} "
                f"symbols={summary.symbol_count if summary else 0}"
            ),
        )

        digest_started = now_iso()
        digest_ref = recorder.save_artifact(
            step_id="step_004_persist_diff_digest",
            artifact_id="artifact_003_diff_digest",
            name="diff_digest",
            content=json.dumps(digest, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_004_persist_diff_digest",
            name="persist diff digest",
            kind="toolchain",
            status="completed",
            started_at=digest_started,
            input_summary=f"files={digest['summary']['file_count']}",
            output_summary=(
                f"additions={digest['summary']['additions']} "
                f"deletions={digest['summary']['deletions']}"
            ),
            artifact_refs=[digest_ref],
        )

        report: dict[str, Any] = {
            "ok": True,
            "run_id": recorder.run_id,
            "status": "completed",
            "toolchain_name": "repo_scout_diff_digest",
            "cwd": str(workspace),
            "request": request,
            "summary": {
                "branch": manifest["status"]["branch"],
                "clean": manifest["status"]["clean"],
                "file_count": digest["summary"]["file_count"],
                "hunk_count": digest["summary"]["hunk_count"],
                "additions": digest["summary"]["additions"],
                "deletions": digest["summary"]["deletions"],
                "directory_state_count": manifest["directory_state_count"],
                "ignored_state_count": manifest["ignored_state_count"],
                "manifest_truncated": manifest["truncated"],
                "diff_file_truncated": len(manifest["files"]) > file_cap,
                "indexed_file_count": summary.file_count if summary else 0,
                "indexed_symbol_count": summary.symbol_count if summary else 0,
            },
            "git": {
                "status": manifest["status"],
                "changed": manifest["changed"],
                "changed_files": manifest["files"],
                "changed_directories": manifest["directories"],
                "ignored": manifest["ignored"],
            },
            "code_intel": {
                "repository": repository.model_dump(mode="json"),
                "index_summary": summary.model_dump(mode="json") if summary else None,
            },
            "diff_digest": digest,
            "trace": recorder.steps,
            "trace_available": True,
            "artifact_refs": [manifest_ref, diffs_ref, digest_ref],
            "limits": {
                "max_changed_files": _max_changed_files(),
                "max_diff_files": file_cap,
                "diff_source_max_chars": _diff_source_max_chars(),
                "ignored_parts": sorted(_changed_ignored_parts()),
                "ignored_suffixes": sorted(_changed_ignored_suffixes()),
                "ignored_paths": sorted(_changed_ignored_paths()),
            },
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
                "lower_level_reuse": [
                    "git.status",
                    "git.changed_files",
                    "git.diff",
                    "code_intel.index",
                ],
                "call_chain_depth": "symbol_overlap_only",
            },
        }
        recorder.finalize(status="completed", report=report)
        return report
    except Exception as exc:
        report = {
            "ok": False,
            "run_id": recorder.run_id,
            "status": "error",
            "toolchain_name": "repo_scout_diff_digest",
            "cwd": str(workspace),
            "request": request,
            "summary": f"repo diff digest errored: {exc}",
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status="error", report=report)
        return report
    finally:
        recorder.close()


def _repo_scout_impact_scope_result(
    *,
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_impact_files: int | None = None,
    max_impact_symbols: int | None = None,
    max_impact_edges: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    file_cap = (
        max_impact_files
        if isinstance(max_impact_files, int) and max_impact_files > 0
        else _max_impact_files()
    )
    symbol_cap = (
        max_impact_symbols
        if isinstance(max_impact_symbols, int) and max_impact_symbols > 0
        else _max_impact_symbols()
    )
    edge_cap = (
        max_impact_edges
        if isinstance(max_impact_edges, int) and max_impact_edges > 0
        else _max_impact_edges()
    )
    request = {
        "cwd": cwd,
        "repo_id": repo_id,
        "label": label,
        "include_untracked": include_untracked,
        "max_impact_files": max_impact_files,
        "max_impact_symbols": max_impact_symbols,
        "max_impact_edges": max_impact_edges,
    }
    recorder = ToolchainRunRecorder(
        toolchain_name="repo_scout_impact_scope",
        workspace_root=workspace,
        request=request,
    )
    try:
        manifest_started = now_iso()
        manifest = _changed_manifest(cwd=cwd, include_untracked=include_untracked)
        manifest_ref = recorder.save_artifact(
            step_id="step_001_collect_changed_manifest",
            artifact_id="artifact_001_changed_manifest",
            name="changed_manifest",
            content=json.dumps(manifest, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_001_collect_changed_manifest",
            name="collect normalized changed manifest",
            kind="raw_primitive",
            status="completed",
            started_at=manifest_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"files={len(manifest['files'])} directories={len(manifest['directories'])} "
                f"ignored={len(manifest['ignored'])}"
            ),
            artifact_refs=[manifest_ref],
        )

        diff_started = now_iso()
        selected_entries = manifest["files"][:file_cap]
        file_diffs = [
            _collect_file_diff(entry, cwd=cwd)
            for entry in selected_entries
        ]
        diffs_ref = recorder.save_artifact(
            step_id="step_002_collect_file_diffs",
            artifact_id="artifact_002_file_diffs",
            name="file_diffs",
            content=json.dumps(file_diffs, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_002_collect_file_diffs",
            name="collect per-file diffs",
            kind="raw_primitive",
            status="completed",
            started_at=diff_started,
            input_summary=f"files={len(selected_entries)} max_impact_files={file_cap}",
            output_summary=(
                f"diffs={sum(1 for item in file_diffs if item['diff_available'])} "
                f"untracked={sum(1 for item in file_diffs if item['state'] == 'untracked')}"
            ),
            artifact_refs=[diffs_ref],
        )

        impact_started = now_iso()
        store = SQLiteCodeIntelStore(code_intel_db_path(workspace))
        try:
            repository = CodeIntelIndexer(store).index_repo(
                repo_id=repo_id,
                root=workspace,
                label=label or repo_id,
            )
            query = CodeIntelQuery(store)
            summary = query.get_index_summary(repo_id)
            impact_scope = _build_impact_scope(
                query=query,
                repo_id=repo_id,
                manifest=manifest,
                file_diffs=file_diffs,
                max_symbols=symbol_cap,
                max_edges=edge_cap,
            )
        finally:
            store.close()
        recorder.record_step(
            step_id="step_003_index_and_scope",
            name="index code intelligence and build impact scope",
            kind="code_intelligence",
            status="completed",
            started_at=impact_started,
            input_summary=f"repo_id={repo_id!r} root={workspace}",
            output_summary=(
                f"changed_symbols={impact_scope['summary']['changed_symbol_count']} "
                f"incoming_edges={impact_scope['summary']['incoming_edge_count']} "
                f"outgoing_edges={impact_scope['summary']['outgoing_edge_count']}"
            ),
        )

        persist_started = now_iso()
        impact_ref = recorder.save_artifact(
            step_id="step_004_persist_impact_scope",
            artifact_id="artifact_003_impact_scope",
            name="impact_scope",
            content=json.dumps(impact_scope, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_004_persist_impact_scope",
            name="persist impact scope",
            kind="toolchain",
            status="completed",
            started_at=persist_started,
            input_summary=f"files={impact_scope['summary']['changed_file_count']}",
            output_summary=(
                f"changed_symbols={impact_scope['summary']['changed_symbol_count']} "
                f"test_adjacency={impact_scope['summary']['test_adjacency_count']}"
            ),
            artifact_refs=[impact_ref],
        )

        report: dict[str, Any] = {
            "ok": True,
            "run_id": recorder.run_id,
            "status": "completed",
            "toolchain_name": "repo_scout_impact_scope",
            "cwd": str(workspace),
            "request": request,
            "summary": {
                "branch": manifest["status"]["branch"],
                "clean": manifest["status"]["clean"],
                "changed_file_count": impact_scope["summary"]["changed_file_count"],
                "changed_symbol_count": impact_scope["summary"]["changed_symbol_count"],
                "incoming_edge_count": impact_scope["summary"]["incoming_edge_count"],
                "outgoing_edge_count": impact_scope["summary"]["outgoing_edge_count"],
                "test_adjacency_count": impact_scope["summary"]["test_adjacency_count"],
                "manifest_truncated": manifest["truncated"],
                "impact_file_truncated": len(manifest["files"]) > file_cap,
                "indexed_file_count": summary.file_count if summary else 0,
                "indexed_symbol_count": summary.symbol_count if summary else 0,
            },
            "git": {
                "status": manifest["status"],
                "changed": manifest["changed"],
                "changed_files": manifest["files"],
                "changed_directories": manifest["directories"],
                "ignored": manifest["ignored"],
            },
            "code_intel": {
                "repository": repository.model_dump(mode="json"),
                "index_summary": summary.model_dump(mode="json") if summary else None,
            },
            "impact_scope": impact_scope,
            "trace": recorder.steps,
            "trace_available": True,
            "artifact_refs": [manifest_ref, diffs_ref, impact_ref],
            "limits": {
                "max_changed_files": _max_changed_files(),
                "max_impact_files": file_cap,
                "max_impact_symbols": symbol_cap,
                "max_impact_edges": edge_cap,
                "diff_source_max_chars": _diff_source_max_chars(),
                "ignored_parts": sorted(_changed_ignored_parts()),
                "ignored_suffixes": sorted(_changed_ignored_suffixes()),
                "ignored_paths": sorted(_changed_ignored_paths()),
            },
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
                "lower_level_reuse": [
                    "git.status",
                    "git.changed_files",
                    "git.diff",
                    "code_intel.index",
                ],
                "scope_depth": "direct_edge_name_match_only",
            },
        }
        recorder.finalize(status="completed", report=report)
        return report
    except Exception as exc:
        report = {
            "ok": False,
            "run_id": recorder.run_id,
            "status": "error",
            "toolchain_name": "repo_scout_impact_scope",
            "cwd": str(workspace),
            "request": request,
            "summary": f"repo impact scope errored: {exc}",
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status="error", report=report)
        return report
    finally:
        recorder.close()


def _repo_scout_changed_context_result(
    *,
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_symbols_per_file: int | None = None,
    max_edges_per_file: int | None = None,
) -> dict[str, Any]:
    workspace = _workspace_root()
    request = {
        "cwd": cwd,
        "repo_id": repo_id,
        "label": label,
        "include_untracked": include_untracked,
        "max_symbols_per_file": max_symbols_per_file,
        "max_edges_per_file": max_edges_per_file,
    }
    recorder = ToolchainRunRecorder(
        toolchain_name="repo_scout_changed_context",
        workspace_root=workspace,
        request=request,
    )
    try:
        manifest_started = now_iso()
        manifest = _changed_manifest(cwd=cwd, include_untracked=include_untracked)
        manifest_ref = recorder.save_artifact(
            step_id="step_001_collect_git_manifest",
            artifact_id="artifact_001_changed_manifest",
            name="changed_manifest",
            content=json.dumps(manifest, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        recorder.record_step(
            step_id="step_001_collect_git_manifest",
            name="collect git changed manifest",
            kind="raw_primitive",
            status="completed",
            started_at=manifest_started,
            input_summary=f"cwd={cwd!r} include_untracked={include_untracked}",
            output_summary=(
                f"changed_file_states={len(manifest['files'])} "
                f"clean={manifest['status']['clean']}"
            ),
            artifact_refs=[manifest_ref],
        )

        index_started = now_iso()
        store = SQLiteCodeIntelStore(code_intel_db_path(workspace))
        try:
            repository = CodeIntelIndexer(store).index_repo(
                repo_id=repo_id,
                root=workspace,
                label=label or repo_id,
            )
            query = CodeIntelQuery(store)
            summary = query.get_index_summary(repo_id)
            max_symbols = (
                max_symbols_per_file
                if isinstance(max_symbols_per_file, int) and max_symbols_per_file > 0
                else _max_symbols_per_file()
            )
            max_edges = (
                max_edges_per_file
                if isinstance(max_edges_per_file, int) and max_edges_per_file > 0
                else _max_edges_per_file()
            )
            file_contexts = [
                _build_file_context(
                    query=query,
                    repo_id=repo_id,
                    file_path=item["path"],
                    max_symbols=max_symbols,
                    max_edges=max_edges,
                )
                for item in manifest["files"]
            ]
        finally:
            store.close()
        summary_dict = summary.model_dump(mode="json") if summary is not None else None
        recorder.record_step(
            step_id="step_002_index_code_intel",
            name="index code intelligence",
            kind="code_intelligence",
            status="completed",
            started_at=index_started,
            input_summary=f"repo_id={repo_id!r} root={workspace}",
            output_summary=(
                f"files={summary.file_count if summary else 0} "
                f"symbols={summary.symbol_count if summary else 0}"
            ),
        )

        context_started = now_iso()
        context_ref = recorder.save_artifact(
            step_id="step_003_build_scout_context",
            artifact_id="artifact_002_file_contexts",
            name="file_contexts",
            content=json.dumps(file_contexts, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
        gaps = _evidence_gaps(file_contexts)
        recorder.record_step(
            step_id="step_003_build_scout_context",
            name="build changed-file scout context",
            kind="toolchain",
            status="completed",
            started_at=context_started,
            input_summary=f"changed_file_states={len(manifest['files'])}",
            output_summary=(
                f"context_files={len(file_contexts)} evidence_gaps={len(gaps)}"
            ),
            artifact_refs=[context_ref],
        )

        report: dict[str, Any] = {
            "ok": True,
            "run_id": recorder.run_id,
            "status": "completed",
            "toolchain_name": "repo_scout_changed_context",
            "cwd": str(workspace),
            "request": request,
            "summary": {
                "branch": manifest["status"]["branch"],
                "clean": manifest["status"]["clean"],
                "changed_file_state_count": len(manifest["files"]),
                "changed_file_state_count_total": manifest["total_file_state_count"],
                "manifest_truncated": manifest["truncated"],
                "python_file_count": sum(1 for ctx in file_contexts if ctx["is_python"]),
                "test_file_count": sum(1 for ctx in file_contexts if ctx["is_test"]),
                "indexed_file_count": summary.file_count if summary else 0,
                "indexed_symbol_count": summary.symbol_count if summary else 0,
            },
            "git": {
                "status": manifest["status"],
                "changed": manifest["changed"],
                "changed_files": manifest["files"],
            },
            "code_intel": {
                "repository": repository.model_dump(mode="json"),
                "index_summary": summary_dict,
                "file_contexts": file_contexts,
            },
            "evidence_gaps": gaps,
            "trace": recorder.steps,
            "trace_available": True,
            "artifact_refs": [manifest_ref, context_ref],
            "limits": {
                "max_changed_files": _max_changed_files(),
                "max_symbols_per_file": max_symbols,
                "max_edges_per_file": max_edges,
            },
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
                "lower_level_reuse": ["git.status", "git.changed_files", "code_intel.index"],
            },
        }
        recorder.finalize(status="completed", report=report)
        return report
    except Exception as exc:
        report = {
            "ok": False,
            "run_id": recorder.run_id,
            "status": "error",
            "toolchain_name": "repo_scout_changed_context",
            "cwd": str(workspace),
            "request": request,
            "summary": f"repo scout errored: {exc}",
            "trace": recorder.steps,
            "trace_available": True,
            "audit": {
                **_fact_report_audit_metadata(),
                "capability_layer": "toolchain",
                "provider_control": "provider_opaque_during_execution",
                "provider_audit": "provider_auditable_after_execution",
            },
        }
        recorder.finalize(status="error", report=report)
        return report
    finally:
        recorder.close()


def _toolchain_store() -> SQLiteToolchainRunStore:
    workspace = _workspace_root()
    return SQLiteToolchainRunStore(default_toolchain_db_path(workspace))


mcp = FastMCP(SERVER_NAME)


@timed_mcp_tool(mcp, SERVER_NAME)
def repo_scout_repository_overview(
    path: str = ".",
    repo_id: str = "workspace",
    label: str | None = None,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Build a summary-first repository overview for initial repo reconnaissance.

    Includes git branch, clean/dirty state, staged/unstaged/untracked counts,
    and recent commits; do not pair with git_status or git_log unless exact
    raw git output is required.
    """
    return _repo_scout_repository_overview_result(
        path=path,
        repo_id=repo_id,
        label=label,
        max_tree_depth=max_tree_depth,
        max_tree_entries=max_tree_entries,
        max_symbols=max_symbols,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def repo_scout_diff_digest(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_diff_files: int | None = None,
) -> dict[str, Any]:
    """Build a summary-first fact digest of changed diffs and touched symbols."""
    return _repo_scout_diff_digest_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_diff_files=max_diff_files,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def repo_scout_impact_scope(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_impact_files: int | None = None,
    max_impact_symbols: int | None = None,
    max_impact_edges: int | None = None,
) -> dict[str, Any]:
    """Build a non-decisional fact report of direct changed-symbol impact scope."""
    return _repo_scout_impact_scope_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_impact_files=max_impact_files,
        max_impact_symbols=max_impact_symbols,
        max_impact_edges=max_impact_edges,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def repo_scout_changed_context(
    cwd: str | None = None,
    repo_id: str = "workspace",
    label: str | None = None,
    include_untracked: bool = True,
    max_symbols_per_file: int | None = None,
    max_edges_per_file: int | None = None,
) -> dict[str, Any]:
    """Build summary-first changed-file scout context with Code Intel symbols.

    Includes normalized git status and changed-file manifest facts; do not pair
    with git_status or git_changed_files unless exact raw git output is required.
    """
    return _repo_scout_changed_context_result(
        cwd=cwd,
        repo_id=repo_id,
        label=label,
        include_untracked=include_untracked,
        max_symbols_per_file=max_symbols_per_file,
        max_edges_per_file=max_edges_per_file,
    )


@timed_mcp_tool(mcp, SERVER_NAME)
def toolchain_get_run(run_id: str) -> dict[str, Any]:
    """Read a persisted Repo Scout L2 toolchain run by run_id."""
    store = _toolchain_store()
    try:
        run = store.get_run(run_id)
        if run is None:
            return {"ok": False, "error": f"toolchain run not found: {run_id}"}
        return {"ok": True, "run": run}
    finally:
        store.close()


@timed_mcp_tool(mcp, SERVER_NAME)
def toolchain_get_step(run_id: str, step_id: str) -> dict[str, Any]:
    """Read one persisted Repo Scout L2 toolchain step."""
    store = _toolchain_store()
    try:
        step = store.get_step(run_id, step_id)
        if step is None:
            return {"ok": False, "error": f"toolchain step not found: {run_id}/{step_id}"}
        return {"ok": True, "step": step}
    finally:
        store.close()


@timed_mcp_tool(mcp, SERVER_NAME)
def toolchain_read_artifact_region(
    run_id: str,
    artifact_id: str,
    start_char: int = 0,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read a bounded region from a persisted Repo Scout L2 artifact."""
    store = _toolchain_store()
    try:
        region = store.read_artifact_region(
            run_id=run_id,
            artifact_id=artifact_id,
            start_char=start_char,
            max_chars=max_chars,
        )
        if region is None:
            return {
                "ok": False,
                "error": f"toolchain artifact not found: {run_id}/{artifact_id}",
            }
        return {"ok": True, "artifact_region": region}
    finally:
        store.close()


if __name__ == "__main__":
    mcp.run()
