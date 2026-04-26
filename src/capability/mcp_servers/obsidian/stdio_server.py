"""Reusable read-only Obsidian MCP server for local Markdown vaults."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "obsidian"
VAULT_ROOT_ENVS = ("ORBIT_OBSIDIAN_VAULT_ROOT", "OBSIDIAN_VAULT_PATH")
MAX_READ_CHARS_ENV = "ORBIT_OBSIDIAN_MAX_READ_CHARS"
MAX_RESULTS_ENV = "ORBIT_OBSIDIAN_MAX_RESULTS"
DEFAULT_MAX_READ_CHARS = 12_000
DEFAULT_MAX_RESULTS = 50

WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]\n]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)")
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z0-9_/-]+)")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _configure_from_argv() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vault")
    parser.add_argument("vault_positional", nargs="?")
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    vault = (args.vault or args.vault_positional or "").strip()
    if vault:
        os.environ.setdefault("ORBIT_OBSIDIAN_VAULT_ROOT", vault)


def _vault_root() -> Path:
    raw = ""
    for env_name in VAULT_ROOT_ENVS:
        raw = os.environ.get(env_name, "").strip()
        if raw:
            break
    if not raw:
        raise ValueError(
            "missing vault root; set ORBIT_OBSIDIAN_VAULT_ROOT, "
            "OBSIDIAN_VAULT_PATH, or pass --vault"
        )
    root = Path(raw).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"vault root is invalid: {root}")
    return root


def _positive_int(value: Any, fallback: int, *, name: str) -> int:
    if value is None or value == "":
        return fallback
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0")
    return result


def _max_read_chars(value: int | None = None) -> int:
    return _positive_int(
        value if value is not None else os.environ.get(MAX_READ_CHARS_ENV, ""),
        DEFAULT_MAX_READ_CHARS,
        name="max read chars",
    )


def _max_results(value: int | None = None) -> int:
    return _positive_int(
        value if value is not None else os.environ.get(MAX_RESULTS_ENV, ""),
        DEFAULT_MAX_RESULTS,
        name="max results",
    )


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _resolve_safe_path(path: str | None = None, *, must_exist: bool = True) -> Path:
    root = _vault_root()
    if path is None or not str(path).strip():
        return root
    candidate = Path(str(path))
    if candidate.is_absolute():
        raise ValueError("absolute paths are not allowed")
    target = (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes vault root") from exc
    if must_exist and not target.exists():
        raise ValueError("path not found")
    return target


def _visible_note_paths(scope: str | None = None, *, recursive: bool = True) -> list[Path]:
    target = _resolve_safe_path(scope)
    if target.is_file():
        if target.suffix.lower() != ".md":
            return []
        rel = target.relative_to(_vault_root())
        return [] if _is_hidden(rel) else [target]
    if not target.is_dir():
        raise ValueError("path is not a directory")
    pattern = "**/*.md" if recursive else "*.md"
    root = _vault_root()
    notes: list[Path] = []
    for path in target.glob(pattern):
        rel = path.relative_to(root)
        if not _is_hidden(rel):
            notes.append(path)
    return sorted(notes, key=lambda item: item.relative_to(root).as_posix().lower())


def _strip_link_suffix(target: str) -> str:
    return target.split("|", 1)[0].split("#", 1)[0].strip()


def _resolve_note_path(path_or_title: str) -> Path:
    raw = _strip_link_suffix(str(path_or_title or ""))
    if not raw:
        raise ValueError("path is required")
    direct = _resolve_safe_path(raw, must_exist=False)
    candidates = [direct]
    if direct.suffix.lower() != ".md":
        candidates.append(direct.with_suffix(".md"))
    root = _vault_root()
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".md":
            rel = candidate.relative_to(root)
            if _is_hidden(rel):
                raise ValueError("hidden notes are not exposed")
            return candidate
    normalized = raw.removesuffix(".md").lower()
    matches: list[Path] = []
    for note in _visible_note_paths(recursive=True):
        rel_no_suffix = note.relative_to(root).with_suffix("").as_posix().lower()
        if note.stem.lower() == normalized or rel_no_suffix == normalized:
            matches.append(note)
    if not matches:
        raise ValueError(f"note not found: {path_or_title}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous note title: {path_or_title}")
    return matches[0]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_frontmatter_and_body(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw_frontmatter = text[4:end]
    body = text[end + 5:]
    frontmatter: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in raw_frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- ") and current_key:
            frontmatter.setdefault(current_key, [])
            if isinstance(frontmatter[current_key], list):
                frontmatter[current_key].append(line.lstrip()[2:].strip().strip("'\""))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if not value:
            frontmatter[current_key] = []
        elif value.startswith("[") and value.endswith("]"):
            frontmatter[current_key] = [
                item.strip().strip("'\"")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        elif value.lower() in {"true", "false"}:
            frontmatter[current_key] = value.lower() == "true"
        else:
            frontmatter[current_key] = value.strip("'\"")
    return frontmatter, body


def _extract_tags(text: str, frontmatter: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    fm_tags = frontmatter.get("tags")
    if isinstance(fm_tags, str) and fm_tags.strip():
        tags.append(fm_tags.strip().lstrip("#"))
    elif isinstance(fm_tags, list):
        tags.extend(str(item).strip().lstrip("#") for item in fm_tags if str(item).strip())
    tags.extend(TAG_RE.findall(text))
    return _dedupe(tags)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_headings(body: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        text = stripped[level:].strip()
        if text:
            headings.append({"level": level, "text": text})
    return headings


def _extract_links(body: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for raw in WIKILINK_RE.findall(body):
        target = _strip_link_suffix(raw)
        if target:
            links.append({"target": target, "kind": "wikilink"})
    for raw in MARKDOWN_LINK_RE.findall(body):
        if raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = _strip_link_suffix(raw)
        if target:
            links.append({"target": target, "kind": "markdown"})
    return links


def _build_summary(body: str, *, max_chars: int = 600) -> str:
    cleaned = CODE_FENCE_RE.sub(" ", body)
    cleaned = re.sub(r"\[\[([^\]]+)\]\]", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars].rstrip()


def _note_type_hint(path: Path, title: str, frontmatter: dict[str, Any]) -> str:
    explicit = frontmatter.get("type") or frontmatter.get("note_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower().replace(" ", "_").replace("-", "_")
    joined = f"{path.as_posix()} {title}".lower()
    for keyword in ("project", "decision", "procedure", "episode", "entity", "principle"):
        if keyword in joined:
            return keyword
    if "index" in joined or "moc" in joined:
        return "index"
    return "generic"


def _note_index() -> dict[str, str]:
    root = _vault_root()
    index: dict[str, str] = {}
    for note in _visible_note_paths(recursive=True):
        rel = note.relative_to(root).as_posix()
        rel_no_suffix = note.relative_to(root).with_suffix("").as_posix()
        for key in (note.stem, rel, rel_no_suffix):
            index.setdefault(key.lower(), rel)
    return index


def _resolved_link_path(target: str, index: dict[str, str]) -> str | None:
    normalized = _strip_link_suffix(target).removesuffix(".md").lower()
    return index.get(normalized) or index.get(f"{normalized}.md")


def _note_record(path: Path, *, include_raw_content: bool, max_chars: int | None = None) -> dict[str, Any]:
    root = _vault_root()
    text = _read_text(path)
    frontmatter, body = _extract_frontmatter_and_body(text)
    rel = path.relative_to(root)
    limit = _max_read_chars(max_chars)
    raw_content = text[:limit] if include_raw_content else None
    title = path.stem
    links = _extract_links(body)
    index = _note_index()
    enriched_links = [
        {
            **link,
            "resolved_path": _resolved_link_path(str(link.get("target") or ""), index),
        }
        for link in links
    ]
    return {
        "path": rel.as_posix(),
        "title": title,
        "note_type_hint": _note_type_hint(rel, title, frontmatter),
        "frontmatter": frontmatter,
        "tags": _extract_tags(text, frontmatter),
        "headings": _extract_headings(body),
        "links": enriched_links,
        "summary": _build_summary(body),
        "raw_content": raw_content,
        "truncated": bool(include_raw_content and len(text) > limit),
        "content_chars": min(len(text), limit) if include_raw_content else len(text),
        "modified_at_epoch": path.stat().st_mtime,
    }


def _list_notes_result(
    path: str | None = None,
    recursive: bool = False,
    max_results: int | None = None,
) -> dict[str, Any]:
    root = _vault_root()
    target = _resolve_safe_path(path)
    if not target.is_dir():
        raise ValueError("path is not a directory")
    limit = _max_results(max_results)
    note_paths = _visible_note_paths(path, recursive=recursive)
    visible = note_paths[:limit]
    notes = []
    for note_path in visible:
        rel = note_path.relative_to(root)
        notes.append(
            {
                "path": rel.as_posix(),
                "title": note_path.stem,
                "note_type_hint": _note_type_hint(rel, note_path.stem, {}),
                "modified_at_epoch": note_path.stat().st_mtime,
            }
        )
    return {
        "path": target.relative_to(root).as_posix() if target != root else "",
        "recursive": bool(recursive),
        "note_count": len(notes),
        "total_note_count": len(note_paths),
        "truncated": len(note_paths) > limit,
        "notes": notes,
    }


def _read_note_result(
    path: str,
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    return _note_record(
        _resolve_note_path(path),
        include_raw_content=bool(include_raw_content),
        max_chars=max_chars,
    )


def _read_notes_result(
    paths: list[str],
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list")
    notes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw_path in paths:
        try:
            notes.append(_read_note_result(str(raw_path), include_raw_content, max_chars))
        except Exception as exc:
            errors.append({"path": str(raw_path), "error": str(exc)})
    return {
        "requested_count": len(paths),
        "note_count": len(notes),
        "error_count": len(errors),
        "notes": notes,
        "errors": errors,
    }


def _search_notes_result(
    query: str,
    path: str | None = None,
    max_results: int | None = None,
    search_in: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    target = _resolve_safe_path(path)
    if not target.is_dir():
        raise ValueError("path is not a directory")
    root = _vault_root()
    limit = _max_results(max_results)
    surfaces = set(search_in or ["title", "frontmatter", "tags", "summary", "content", "path"])
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9_\-]+", query.lower())
        if len(token) >= 2
    ] or [query.lower().strip()]
    stopwords = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with"}
    tokens = [token for token in tokens if token not in stopwords] or [query.lower().strip()]
    matches: list[dict[str, Any]] = []
    for note_path in _visible_note_paths(path, recursive=True):
        text = _read_text(note_path)
        frontmatter, body = _extract_frontmatter_and_body(text)
        rel = note_path.relative_to(root).as_posix()
        title = note_path.stem
        summary = _build_summary(body)
        tags = _extract_tags(text, frontmatter)
        haystacks = {
            "title": title.lower(),
            "frontmatter": json.dumps(frontmatter, ensure_ascii=False).lower(),
            "tags": " ".join(tags).lower(),
            "summary": summary.lower(),
            "content": body.lower(),
            "path": rel.lower(),
        }
        matched_tokens: list[str] = []
        hit_surfaces: list[str] = []
        score = 0.0
        weights = {"title": 1.2, "path": 0.6, "summary": 0.8, "content": 0.3, "tags": 0.5, "frontmatter": 0.4}
        for token in tokens:
            for surface in surfaces:
                if token in haystacks.get(surface, ""):
                    score += weights.get(surface, 0.2)
                    if token not in matched_tokens:
                        matched_tokens.append(token)
                    if surface not in hit_surfaces:
                        hit_surfaces.append(surface)
        if not matched_tokens:
            continue
        note_type_hint = _note_type_hint(note_path.relative_to(root), title, frontmatter)
        if note_type_hint in tokens:
            score += 1.0
        matches.append(
            {
                "path": rel,
                "title": title,
                "note_type_hint": note_type_hint,
                "match_surfaces": hit_surfaces,
                "matched_tokens": matched_tokens,
                "preview": summary[:240],
                "score_hint": round(score, 4),
            }
        )
    matches.sort(key=lambda item: float(item.get("score_hint") or 0.0), reverse=True)
    return {
        "query": query,
        "query_tokens": tokens,
        "path_scope": target.relative_to(root).as_posix() if target != root else "",
        "match_count": min(len(matches), limit),
        "total_match_count": len(matches),
        "truncated": len(matches) > limit,
        "matches": matches[:limit],
    }


def _get_note_links_result(path: str) -> dict[str, Any]:
    note = _read_note_result(path, include_raw_content=False)
    return {
        "path": note["path"],
        "link_count": len(note["links"]),
        "links": note["links"],
    }


def _get_backlinks_result(path: str) -> dict[str, Any]:
    target_note = _read_note_result(path, include_raw_content=False)
    target_path = str(target_note["path"])
    target_stem = Path(target_path).stem
    target_no_suffix = str(Path(target_path).with_suffix(""))
    aliases = {target_path.lower(), target_no_suffix.lower(), target_stem.lower()}
    backlinks: list[dict[str, Any]] = []
    for note_path in _visible_note_paths(recursive=True):
        source = _note_record(note_path, include_raw_content=False)
        if source["path"] == target_path:
            continue
        for link in source["links"]:
            target = str(link.get("target") or "").removesuffix(".md").lower()
            resolved = str(link.get("resolved_path") or "").lower()
            if target in aliases or resolved == target_path.lower():
                backlinks.append(
                    {
                        "source_path": source["path"],
                        "source_title": source["title"],
                        "target": link.get("target"),
                        "kind": link.get("kind"),
                    }
                )
                break
    return {
        "path": target_path,
        "backlink_count": len(backlinks),
        "backlinks": backlinks,
    }


def _get_unresolved_links_result(path: str | None = None, max_results: int | None = None) -> dict[str, Any]:
    limit = _max_results(max_results)
    unresolved: list[dict[str, Any]] = []
    for note_path in _visible_note_paths(path, recursive=True):
        note = _note_record(note_path, include_raw_content=False)
        for link in note["links"]:
            if link.get("resolved_path") is None:
                unresolved.append(
                    {
                        "source_path": note["path"],
                        "target": link.get("target"),
                        "kind": link.get("kind"),
                    }
                )
    return {
        "path_scope": path or "",
        "unresolved_count": min(len(unresolved), limit),
        "total_unresolved_count": len(unresolved),
        "truncated": len(unresolved) > limit,
        "unresolved_links": unresolved[:limit],
    }


def _get_tag_summary_result(path: str | None = None, max_results: int | None = None) -> dict[str, Any]:
    tag_map: dict[str, list[str]] = {}
    for note_path in _visible_note_paths(path, recursive=True):
        note = _note_record(note_path, include_raw_content=False)
        for tag in note["tags"]:
            tag_map.setdefault(tag, []).append(note["path"])
    limit = _max_results(max_results)
    items = [
        {"tag": tag, "note_count": len(paths), "notes": paths[:limit]}
        for tag, paths in sorted(tag_map.items(), key=lambda item: (-len(item[1]), item[0].lower()))
    ]
    return {
        "path_scope": path or "",
        "tag_count": len(items),
        "tags": items[:limit],
        "truncated": len(items) > limit,
    }


def _get_vault_metadata_result(
    path: str | None = None,
    include_top_level_entries: bool | None = True,
    max_entries: int | None = None,
) -> dict[str, Any]:
    root = _vault_root()
    target = _resolve_safe_path(path)
    if not target.is_dir():
        raise ValueError("path is not a directory")
    visible: list[Path] = []
    for candidate in target.rglob("*"):
        rel = candidate.relative_to(root)
        if not _is_hidden(rel):
            visible.append(candidate)
    note_count = sum(1 for item in visible if item.is_file() and item.suffix.lower() == ".md")
    directory_count = sum(1 for item in visible if item.is_dir())
    latest_modified = target.stat().st_mtime
    for item in visible:
        try:
            latest_modified = max(latest_modified, item.stat().st_mtime)
        except OSError:
            pass
    entries: list[dict[str, Any]] = []
    if bool(include_top_level_entries):
        limit = _max_results(max_entries)
        children = [
            child
            for child in sorted(target.iterdir(), key=lambda item: item.name.lower())
            if not _is_hidden(child.relative_to(root))
        ]
        for child in children[:limit]:
            entries.append(
                {
                    "name": child.name,
                    "path": child.relative_to(root).as_posix(),
                    "kind": "directory" if child.is_dir() else "file",
                    "modified_at_epoch": child.stat().st_mtime,
                }
            )
    else:
        children = []
    return {
        "vault_root": str(root),
        "vault_name": root.name,
        "path_scope": target.relative_to(root).as_posix() if target != root else "",
        "resolved_path": str(target),
        "generated_at_epoch": int(time.time()),
        "note_count": note_count,
        "directory_count": directory_count,
        "latest_modified_at_epoch": latest_modified,
        "excluded_hidden_entries": True,
        "top_level_entries": entries,
        "truncated_top_level_entries": bool(include_top_level_entries and len(children) > len(entries)),
    }


def _check_availability_result() -> dict[str, Any]:
    raw_root = ""
    source_env = None
    for env_name in VAULT_ROOT_ENVS:
        value = os.environ.get(env_name, "").strip()
        if value:
            raw_root = value
            source_env = env_name
            break
    warnings: list[str] = []
    vault_root_configured = bool(raw_root)
    vault_root: str | None = None
    vault_root_exists = False
    vault_root_readable = False
    if raw_root:
        candidate = Path(raw_root).expanduser().resolve()
        vault_root = str(candidate)
        vault_root_exists = candidate.exists() and candidate.is_dir()
        vault_root_readable = os.access(candidate, os.R_OK) if vault_root_exists else False
        if not vault_root_exists:
            warnings.append("configured vault root does not exist or is not a directory")
        elif not vault_root_readable:
            warnings.append("configured vault root is not readable")
    else:
        warnings.append("vault root is not configured")
    cli_path = shutil.which("obsidian")
    if not cli_path:
        warnings.append("obsidian CLI not found on PATH")
    available = vault_root_exists and vault_root_readable
    return {
        "available": available,
        "availability_level": "vault_only" if available and not cli_path else "full" if available else "unavailable",
        "vault_root_configured": vault_root_configured,
        "vault_root_env": source_env,
        "vault_root_exists": vault_root_exists,
        "vault_root_readable": vault_root_readable,
        "vault_root": vault_root,
        "obsidian_cli_found": bool(cli_path),
        "obsidian_cli_path": cli_path,
        "recommended_mode": "mcp_vault_direct" if available else "disabled",
        "warnings": warnings,
    }


_configure_from_argv()
mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def obsidian_list_notes(
    path: str | None = None,
    recursive: bool = False,
    max_results: int | None = None,
) -> dict[str, Any]:
    """List Markdown notes in the configured vault or a scoped directory."""
    return _list_notes_result(path, recursive, max_results)


@mcp.tool()
def obsidian_read_note(
    path: str,
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read one note by vault-relative path, stem, or wikilink target."""
    return _read_note_result(path, include_raw_content, max_chars)


@mcp.tool()
def obsidian_read_notes(
    paths: list[str],
    include_raw_content: bool | None = False,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Read multiple notes without failing the whole call on one missing note."""
    return _read_notes_result(paths, include_raw_content, max_chars)


@mcp.tool()
def obsidian_search_notes(
    query: str,
    path: str | None = None,
    max_results: int | None = None,
    search_in: list[str] | None = None,
) -> dict[str, Any]:
    """Search notes across title, frontmatter, tags, summary, content, and path."""
    return _search_notes_result(query, path, max_results, search_in)


@mcp.tool()
def obsidian_get_note_links(path: str) -> dict[str, Any]:
    """Return outgoing local links for a note with best-effort resolution."""
    return _get_note_links_result(path)


@mcp.tool()
def obsidian_get_backlinks(path: str) -> dict[str, Any]:
    """Return notes that link to the target note."""
    return _get_backlinks_result(path)


@mcp.tool()
def obsidian_get_unresolved_links(
    path: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Return local note links that do not resolve to Markdown notes."""
    return _get_unresolved_links_result(path, max_results)


@mcp.tool()
def obsidian_get_tag_summary(
    path: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """Return tags and the notes where they appear."""
    return _get_tag_summary_result(path, max_results)


@mcp.tool()
def obsidian_get_vault_metadata(
    path: str | None = None,
    include_top_level_entries: bool | None = True,
    max_entries: int | None = None,
) -> dict[str, Any]:
    """Return bounded vault metadata for the whole vault or a scoped directory."""
    return _get_vault_metadata_result(path, include_top_level_entries, max_entries)


@mcp.tool()
def obsidian_check_availability() -> dict[str, Any]:
    """Check vault accessibility and optional Obsidian CLI presence."""
    return _check_availability_result()


if __name__ == "__main__":
    mcp.run()
