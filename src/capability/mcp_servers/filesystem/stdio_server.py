"""Bounded filesystem MCP server for Orbit2 (Handoff 16, first slice).

Ships with five tools only: `read_file`, `list_directory`, `get_file_info`,
`write_file`, `replace_in_file`. Each is workspace-scoped: operations must
target paths inside the allowed root (set via `ORBIT_WORKSPACE_ROOT` env var
or the trailing positional argument at server launch).

Not migrated from Orbit1's 1151-line filesystem server: symbol search, glob,
grep, directory tree, media reads, todo_read/write, web_fetch, unified-patch
application. Deferred to future slices consistent with Handoff 16's bounded
intent.

Orbit2-side family-aware governance metadata for these tools lives at
`src/capability/mcp/governance.py`.
"""
from __future__ import annotations

import fnmatch
import glob as _py_glob
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.capability.models import is_protected_relative_path

SERVER_NAME = "filesystem"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_READ_BYTES = 64 * 1024
DEFAULT_MAX_DIRECTORY_ENTRIES = 200
DEFAULT_MAX_GLOB_RESULTS = 200
DEFAULT_MAX_GREP_MATCHES = 200
DEFAULT_GREP_MAX_FILE_BYTES = 256 * 1024
DEFAULT_GREP_MAX_TOTAL_BYTES = 4 * 1024 * 1024
DEFAULT_GREP_MAX_LINE_CHARS = 2000
DEFAULT_MAX_TREE_DEPTH = 4
DEFAULT_MAX_TREE_ENTRIES = 500
DEFAULT_MAX_MULTI_READ_FILES = 20
GREP_IGNORED_PARTS_ENV = "ORBIT2_MCP_FS_GREP_IGNORED_PARTS"
GREP_IGNORED_SUFFIXES_ENV = "ORBIT2_MCP_FS_GREP_IGNORED_SUFFIXES"
DEFAULT_GREP_IGNORED_PARTS = "__pycache__"
DEFAULT_GREP_IGNORED_SUFFIXES = ".pyc,.pyo"


def _workspace_root() -> Path:
    """Resolve the allowed filesystem root at tool-call time."""
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"filesystem MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _resolve_safe_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")
    workspace_root = _workspace_root()
    target = (workspace_root / candidate).resolve()
    try:
        target.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    return target


def _resolve_safe_existing_file(path: str) -> Path:
    target = _resolve_safe_path(path)
    if not target.exists():
        raise ValueError("path not found")
    if not target.is_file():
        raise ValueError("path is not a file")
    return target


def _csv_env_values(env_name: str, fallback_csv: str) -> frozenset[str]:
    raw = os.environ.get(env_name)
    source = fallback_csv if raw is None else raw
    return frozenset(part.strip() for part in source.split(",") if part.strip())


def _grep_ignored_parts() -> frozenset[str]:
    return _csv_env_values(GREP_IGNORED_PARTS_ENV, DEFAULT_GREP_IGNORED_PARTS)


def _grep_ignored_suffixes() -> frozenset[str]:
    return _csv_env_values(GREP_IGNORED_SUFFIXES_ENV, DEFAULT_GREP_IGNORED_SUFFIXES)


# ---------------------------------------------------------------------------
# Pure result helpers (importable by tests without running the MCP server)
# ---------------------------------------------------------------------------


def _read_file_result(path: str, *, max_bytes: int | None = None) -> dict[str, Any]:
    target = _resolve_safe_existing_file(path)
    cap = max_bytes if isinstance(max_bytes, int) and max_bytes > 0 else DEFAULT_MAX_READ_BYTES
    data = target.read_bytes()
    truncated = len(data) > cap
    payload = data[:cap].decode("utf-8", errors="replace")
    return {
        "ok": True,
        "path": target.as_posix(),
        "content": payload,
        "truncated": truncated,
        "byte_size": len(data),
    }


def _list_directory_result(path: str, *, max_entries: int | None = None) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    if not target.exists() or not target.is_dir():
        raise ValueError("path is not a directory")
    cap = max_entries if isinstance(max_entries, int) and max_entries > 0 else DEFAULT_MAX_DIRECTORY_ENTRIES
    entries: list[dict[str, Any]] = []
    for child in sorted(target.iterdir(), key=lambda p: p.name):
        entries.append({
            "name": child.name,
            "kind": "directory" if child.is_dir() else ("file" if child.is_file() else "other"),
        })
        if len(entries) >= cap:
            break
    return {
        "ok": True,
        "path": target.as_posix(),
        "entries": entries,
        "truncated": len(entries) >= cap,
    }


def _get_file_info_result(path: str) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    if not target.exists():
        raise ValueError("path not found")
    stat = target.stat()
    return {
        "ok": True,
        "path": target.as_posix(),
        "kind": "directory" if target.is_dir() else ("file" if target.is_file() else "other"),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _write_file_result(path: str, content: str) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "write_file",
    }


def _replace_in_file_result(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    target = _resolve_safe_existing_file(path)
    current = target.read_text(encoding="utf-8")
    if old_text not in current:
        return {
            "ok": False,
            "path": target.as_posix(),
            "mutation_kind": "replace_in_file",
            "failure_kind": "old_text_not_found",
            "replacement_count": 0,
        }
    updated = current.replace(old_text, new_text, 1)
    target.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "replace_in_file",
        "replacement_count": 1,
    }


# ---------------------------------------------------------------------------
# Handoff 24: filesystem mutation widening helpers
# ---------------------------------------------------------------------------


def _replace_all_in_file_result(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    target = _resolve_safe_existing_file(path)
    current = target.read_text(encoding="utf-8")
    count = current.count(old_text)
    if count == 0:
        return {
            "ok": False,
            "path": target.as_posix(),
            "mutation_kind": "replace_all_in_file",
            "failure_kind": "old_text_not_found",
            "replacement_count": 0,
        }
    updated = current.replace(old_text, new_text)
    target.write_text(updated, encoding="utf-8")
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "replace_all_in_file",
        "replacement_count": count,
    }


def _create_directory_result(path: str) -> dict[str, Any]:
    target = _resolve_safe_path(path)
    already_existed = target.exists() and target.is_dir()
    target.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "path": target.as_posix(),
        "mutation_kind": "create_directory",
        "already_existed": already_existed,
    }


def _move_file_result(source: str, destination: str) -> dict[str, Any]:
    src = _resolve_safe_path(source)
    dst = _resolve_safe_path(destination)
    if not src.exists():
        raise ValueError("source path not found")
    if not src.is_file():
        raise ValueError("source path is not a file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return {
        "ok": True,
        "source": src.as_posix(),
        "destination": dst.as_posix(),
        "mutation_kind": "move_file",
    }


# ---------------------------------------------------------------------------
# Handoff 23: mcp_fs_read widening helpers
# ---------------------------------------------------------------------------


def _glob_result(
    pattern: str,
    *,
    path: str = ".",
    max_results: int | None = None,
) -> dict[str, Any]:
    """Recursively match `pattern` against files under `path` inside the
    workspace. Uses Python's `glob.glob(..., recursive=True)` so shell-style
    `**` works as expected (e.g. `**/*.py` matches Python files at any
    depth). Returned paths are workspace-relative POSIX strings. Bounded to
    `max_results` entries."""
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    # Resolve the workspace root once, then use it both for containment
    # checks and for the root-dir anchor passed to `glob.glob`. Audit
    # HIGH-3: the earlier shape called `_workspace_root()` twice, which
    # could diverge if env/argv mutated between calls.
    workspace_root = _workspace_root()
    root = _resolve_safe_path(path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    cap = max_results if isinstance(max_results, int) and max_results > 0 else DEFAULT_MAX_GLOB_RESULTS
    raw_matches = _py_glob.glob(pattern, root_dir=str(root), recursive=True)
    matches: list[str] = []
    for rel_from_root in sorted(raw_matches):
        if len(matches) >= cap:
            break
        child = (root / rel_from_root).resolve()
        try:
            rel_from_workspace = child.relative_to(workspace_root).as_posix()
        except ValueError:
            continue
        if not child.is_file():
            continue
        matches.append(rel_from_workspace)
    return {
        "ok": True,
        "path": root.as_posix(),
        "pattern": pattern,
        "matches": matches,
        "truncated": len(matches) >= cap,
        "match_count": len(matches),
    }


def _search_files_result(
    name_pattern: str,
    *,
    path: str = ".",
    max_results: int | None = None,
) -> dict[str, Any]:
    """Find files under `path` whose basename matches `name_pattern`
    (fnmatch glob). Complements `glob` which matches the full relative
    path."""
    if not isinstance(name_pattern, str) or not name_pattern.strip():
        raise ValueError("name_pattern must be a non-empty string")
    root = _resolve_safe_path(path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    cap = max_results if isinstance(max_results, int) and max_results > 0 else DEFAULT_MAX_GLOB_RESULTS
    workspace_root = _workspace_root()
    matches: list[str] = []
    for child in root.rglob("*"):
        if len(matches) >= cap:
            break
        if not child.is_file():
            continue
        if not fnmatch.fnmatch(child.name, name_pattern):
            continue
        try:
            rel = child.relative_to(workspace_root).as_posix()
        except ValueError:
            continue
        matches.append(rel)
    return {
        "ok": True,
        "path": root.as_posix(),
        "name_pattern": name_pattern,
        "matches": matches,
        "truncated": len(matches) >= cap,
        "match_count": len(matches),
    }


def _grep_result(
    pattern: str,
    *,
    path: str = ".",
    file_pattern: str = "*",
    max_matches: int | None = None,
    case_insensitive: bool = False,
    max_file_bytes: int | None = None,
    max_total_bytes: int | None = None,
) -> dict[str, Any]:
    """Search file contents for `pattern` (regex). Returns matches with
    their path, line number, and line text.

    Bounds (audit HIGH-2 mitigation for regex DoS):
    - `max_matches` caps the number of matches returned.
    - `max_file_bytes` caps bytes read from any single file.
    - `max_total_bytes` caps total bytes read across the call.
    - Individual lines are truncated to `DEFAULT_GREP_MAX_LINE_CHARS`
      BEFORE the regex is applied, so a pathological pattern still can't
      backtrack over a multi-megabyte line. Python's `re` module has no
      timeout; bounding input is the only portable defense.
    """
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    try:
        regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc
    workspace_root = _workspace_root()
    root = _resolve_safe_path(path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    cap = max_matches if isinstance(max_matches, int) and max_matches > 0 else DEFAULT_MAX_GREP_MATCHES
    file_byte_cap = (
        max_file_bytes if isinstance(max_file_bytes, int) and max_file_bytes > 0
        else DEFAULT_GREP_MAX_FILE_BYTES
    )
    total_byte_cap = (
        max_total_bytes if isinstance(max_total_bytes, int) and max_total_bytes > 0
        else DEFAULT_GREP_MAX_TOTAL_BYTES
    )
    matches: list[dict[str, Any]] = []
    scanned_files = 0
    total_bytes = 0
    total_budget_exhausted = False
    for child in root.rglob("*"):
        if len(matches) >= cap:
            break
        if total_bytes >= total_byte_cap:
            total_budget_exhausted = True
            break
        if not child.is_file():
            continue
        if _is_grep_ignored_path(child):
            continue
        if not fnmatch.fnmatch(child.name, file_pattern):
            continue
        try:
            data = child.read_bytes()
        except OSError:
            continue
        # Per-file byte cap — bounds any single file's contribution.
        data = data[:file_byte_cap]
        total_bytes += len(data)
        try:
            text = data.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            continue
        scanned_files += 1
        try:
            rel = child.relative_to(workspace_root).as_posix()
        except ValueError:
            continue
        if is_protected_relative_path(rel) is not None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Truncate the line BEFORE regex so a greedy pattern can't
            # backtrack over a multi-megabyte line.
            bounded = line[:DEFAULT_GREP_MAX_LINE_CHARS]
            if regex.search(bounded):
                matches.append({"path": rel, "line": lineno, "text": bounded[:400]})
                if len(matches) >= cap:
                    break
    return {
        "ok": True,
        "path": root.as_posix(),
        "pattern": pattern,
        "file_pattern": file_pattern,
        "case_insensitive": case_insensitive,
        "matches": matches,
        "match_count": len(matches),
        "scanned_files": scanned_files,
        "scanned_bytes": total_bytes,
        "truncated": len(matches) >= cap,
        "total_byte_budget_exhausted": total_budget_exhausted,
    }


def _is_grep_ignored_path(path: Path) -> bool:
    ignored_parts = _grep_ignored_parts()
    ignored_suffixes = _grep_ignored_suffixes()
    return any(part in ignored_parts for part in path.parts) or path.suffix in ignored_suffixes


def _directory_tree_result(
    path: str = ".",
    *,
    max_depth: int | None = None,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Return a bounded recursive listing of `path` as a flat entry list
    with each entry's relative path and depth. Bounded by max_depth and
    max_entries so large trees don't explode the response."""
    root = _resolve_safe_path(path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    depth_cap = max_depth if isinstance(max_depth, int) and max_depth > 0 else DEFAULT_MAX_TREE_DEPTH
    entries_cap = max_entries if isinstance(max_entries, int) and max_entries > 0 else DEFAULT_MAX_TREE_ENTRIES
    workspace_root = _workspace_root()
    entries: list[dict[str, Any]] = []

    def _walk(cur: Path, depth: int) -> None:
        if depth > depth_cap:
            return
        if len(entries) >= entries_cap:
            return
        try:
            children = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if len(entries) >= entries_cap:
                return
            try:
                rel = child.relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            entry_kind = "directory" if child.is_dir() else ("file" if child.is_file() else "other")
            entries.append({"path": rel, "depth": depth, "kind": entry_kind})
            if child.is_dir():
                _walk(child, depth + 1)

    _walk(root, 1)
    return {
        "ok": True,
        "path": root.as_posix(),
        "max_depth": depth_cap,
        "entries": entries,
        "entry_count": len(entries),
        "truncated": len(entries) >= entries_cap,
    }


def _read_multiple_files_result(
    paths: list[str],
    *,
    max_bytes_per_file: int | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Batch read multiple files. Each file is resolved independently; one
    bad path doesn't abort the whole call. Returns per-file
    ok/content/byte_size/error — mirrors the single-file reader's shape."""
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list of strings")
    files_cap = max_files if isinstance(max_files, int) and max_files > 0 else DEFAULT_MAX_MULTI_READ_FILES
    byte_cap = max_bytes_per_file if isinstance(max_bytes_per_file, int) and max_bytes_per_file > 0 else DEFAULT_MAX_READ_BYTES
    results: list[dict[str, Any]] = []
    truncated_batch = len(paths) > files_cap
    for entry in paths[:files_cap]:
        if not isinstance(entry, str) or not entry.strip():
            results.append({"ok": False, "path": entry, "error": "path must be a non-empty string"})
            continue
        try:
            target = _resolve_safe_existing_file(entry)
        except ValueError as exc:
            results.append({"ok": False, "path": entry, "error": str(exc)})
            continue
        data = target.read_bytes()
        content = data[:byte_cap].decode("utf-8", errors="replace")
        results.append({
            "ok": True,
            "path": target.as_posix(),
            "content": content,
            "truncated": len(data) > byte_cap,
            "byte_size": len(data),
        })
    return {
        "ok": True,
        "results": results,
        "file_count": len(results),
        "batch_truncated": truncated_batch,
    }


def _list_directory_with_sizes_result(
    path: str,
    *,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Like `_list_directory_result` but includes per-entry size (for files)
    and entry count (for directories, when available). Does not recurse."""
    target = _resolve_safe_path(path)
    if not target.exists() or not target.is_dir():
        raise ValueError("path is not a directory")
    cap = max_entries if isinstance(max_entries, int) and max_entries > 0 else DEFAULT_MAX_DIRECTORY_ENTRIES
    entries: list[dict[str, Any]] = []
    for child in sorted(target.iterdir(), key=lambda p: p.name):
        if len(entries) >= cap:
            break
        if child.is_dir():
            try:
                child_count = sum(1 for _ in child.iterdir())
            except OSError:
                child_count = None
            entries.append({
                "name": child.name,
                "kind": "directory",
                "child_count": child_count,
            })
        elif child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = None
            entries.append({
                "name": child.name,
                "kind": "file",
                "size": size,
            })
        else:
            entries.append({"name": child.name, "kind": "other"})
    return {
        "ok": True,
        "path": target.as_posix(),
        "entries": entries,
        "truncated": len(entries) >= cap,
    }


# ---------------------------------------------------------------------------
# FastMCP server wiring
# ---------------------------------------------------------------------------


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def read_file(path: str) -> dict[str, Any]:
    """Read a workspace-relative text file. Returns up to 64 KiB."""
    return _read_file_result(path)


@mcp.tool()
def list_directory(path: str) -> dict[str, Any]:
    """List entries under a workspace-relative directory (first 200)."""
    return _list_directory_result(path)


@mcp.tool()
def get_file_info(path: str) -> dict[str, Any]:
    """Return size / mtime / kind for a workspace-relative path."""
    return _get_file_info_result(path)


@mcp.tool()
def write_file(path: str, content: str) -> dict[str, Any]:
    """Write UTF-8 text to a workspace-relative path. Creates parent dirs."""
    return _write_file_result(path, content)


@mcp.tool()
def replace_in_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace the first occurrence of `old_text` with `new_text`."""
    return _replace_in_file_result(path, old_text, new_text)


@mcp.tool()
def replace_all_in_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    """Replace ALL occurrences of `old_text` with `new_text`. Returns count."""
    return _replace_all_in_file_result(path, old_text, new_text)


@mcp.tool()
def create_directory(path: str) -> dict[str, Any]:
    """Create a workspace-relative directory (with parents). Idempotent."""
    return _create_directory_result(path)


@mcp.tool()
def move_file(source: str, destination: str) -> dict[str, Any]:
    """Move/rename a file inside the workspace. Creates destination parent dirs."""
    return _move_file_result(source, destination)


@mcp.tool()
def glob(
    pattern: str,
    path: str = ".",
    max_results: int = DEFAULT_MAX_GLOB_RESULTS,
) -> dict[str, Any]:
    """Recursively match a shell-style glob (`**/*.py`) against workspace file paths."""
    return _glob_result(pattern, path=path, max_results=max_results)


@mcp.tool()
def search_files(
    name_pattern: str,
    path: str = ".",
    max_results: int = DEFAULT_MAX_GLOB_RESULTS,
) -> dict[str, Any]:
    """Find files by basename glob under a workspace directory."""
    return _search_files_result(name_pattern, path=path, max_results=max_results)


@mcp.tool()
def grep(
    pattern: str,
    path: str = ".",
    file_pattern: str = "*",
    max_matches: int = DEFAULT_MAX_GREP_MATCHES,
    case_insensitive: bool = False,
) -> dict[str, Any]:
    """Regex search across workspace files. Returns bounded match list with line numbers."""
    return _grep_result(
        pattern, path=path, file_pattern=file_pattern,
        max_matches=max_matches, case_insensitive=case_insensitive,
    )


@mcp.tool()
def directory_tree(
    path: str = ".",
    max_depth: int = DEFAULT_MAX_TREE_DEPTH,
    max_entries: int = DEFAULT_MAX_TREE_ENTRIES,
) -> dict[str, Any]:
    """Bounded recursive directory listing with depth + entry caps."""
    return _directory_tree_result(path, max_depth=max_depth, max_entries=max_entries)


@mcp.tool()
def read_multiple_files(
    paths: list[str],
    max_bytes_per_file: int = DEFAULT_MAX_READ_BYTES,
    max_files: int = DEFAULT_MAX_MULTI_READ_FILES,
) -> dict[str, Any]:
    """Batch read; per-file ok/content/error. One bad path doesn't abort the batch."""
    return _read_multiple_files_result(
        paths, max_bytes_per_file=max_bytes_per_file, max_files=max_files,
    )


@mcp.tool()
def list_directory_with_sizes(
    path: str,
    max_entries: int = DEFAULT_MAX_DIRECTORY_ENTRIES,
) -> dict[str, Any]:
    """Non-recursive listing with file sizes and directory child counts."""
    return _list_directory_with_sizes_result(path, max_entries=max_entries)


if __name__ == "__main__":
    mcp.run()
