"""Bounded git MCP server for Orbit2 (Handoff 16 first slice; Handoff 24 expansion).

Read tools: `git_status`, `git_diff`, `git_log`, `git_show`,
`git_changed_files`. Classified safe/no-approval.

Mutation tools: `git_add`, `git_commit`, `git_restore`, `git_unstage`,
`git_checkout_branch`. Approval-required per the family-aware governance
overlay at `src/capability/mcp/governance.py`.

Runs `git` as a subprocess. The cwd is either the workspace root
(`ORBIT_WORKSPACE_ROOT` env var, or trailing positional arg at launch) or a
caller-supplied `cwd` argument that MUST resolve inside that root.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "git"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_DIFF_CHARS = 12_000
GIT_TIMEOUT_SECONDS = 20.0


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"git MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _resolve_cwd(cwd: str | None) -> Path:
    workspace = _workspace_root()
    if cwd is None or not str(cwd).strip() or cwd == ".":
        return workspace
    candidate = Path(cwd)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("cwd escapes workspace") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("cwd is not an existing directory")
    return resolved


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ---------------------------------------------------------------------------
# Pure result helpers
# ---------------------------------------------------------------------------


def _git_status_result(cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    branch_cp = _run_git(["branch", "--show-current"], cwd=resolved)
    branch = branch_cp.stdout.strip() if branch_cp.returncode == 0 else ""

    porcelain = _run_git(["status", "--short", "--branch"], cwd=resolved)
    if porcelain.returncode != 0:
        raise ValueError((porcelain.stderr or "git status failed").strip())

    lines = [line for line in (porcelain.stdout or "").splitlines() if line.strip()]
    file_lines = lines[1:] if lines and lines[0].startswith("## ") else lines

    staged: list[dict[str, str]] = []
    unstaged: list[dict[str, str]] = []
    untracked: list[str] = []
    for line in file_lines:
        if line.startswith("?? "):
            untracked.append(line[3:])
            continue
        if len(line) < 3:
            continue
        x, y, path_text = line[0], line[1], line[3:]
        if x != " ":
            staged.append({"path": path_text, "code": x})
        if y != " ":
            unstaged.append({"path": path_text, "code": y})

    return {
        "ok": True,
        "cwd": str(resolved),
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "staged_count": len(staged),
        "unstaged_count": len(unstaged),
        "untracked_count": len(untracked),
        "clean": not staged and not unstaged and not untracked,
    }


def _git_diff_result(
    *,
    cwd: str | None = None,
    path: str | None = None,
    staged: bool = False,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    args = ["diff"]
    if staged:
        args.append("--cached")
    if isinstance(path, str) and path.strip():
        args.extend(["--", path])
    completed = _run_git(args, cwd=resolved)
    if completed.returncode != 0:
        raise ValueError((completed.stderr or "git diff failed").strip())
    text, truncated = _truncate(completed.stdout or "", max_chars)
    return {
        "ok": True,
        "cwd": str(resolved),
        "staged": staged,
        "diff": text,
        "has_diff": bool(text),
        "truncated": truncated,
    }


def _git_log_result(*, cwd: str | None = None, limit: int = 10) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    format_str = "%H%x1f%h%x1f%an%x1f%ae%x1f%ad%x1f%s"
    completed = _run_git(
        ["log", f"-n{limit}", f"--pretty=format:{format_str}", "--date=iso-strict"],
        cwd=resolved,
    )
    if completed.returncode != 0:
        raise ValueError((completed.stderr or "git log failed").strip())
    commits: list[dict[str, str]] = []
    for line in (completed.stdout or "").splitlines():
        parts = line.split("\x1f")
        if len(parts) != 6:
            continue
        sha, short, author_name, author_email, date, subject = parts
        commits.append({
            "sha": sha,
            "short_sha": short,
            "author_name": author_name,
            "author_email": author_email,
            "date": date,
            "subject": subject,
        })
    return {
        "ok": True,
        "cwd": str(resolved),
        "limit": limit,
        "commits": commits,
        "commit_count": len(commits),
    }


def _git_add_result(*, paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    workspace = _workspace_root()
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list of strings")
    for p in paths:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("paths entries must be non-empty strings")
        # Defense-in-depth: individual path entries must remain inside the
        # workspace even though git itself only tracks paths under its repo
        # root. Prevents `git add ../../outside` from reaching git at all.
        candidate = Path(p)
        probe = candidate.resolve() if candidate.is_absolute() else (resolved / candidate).resolve()
        try:
            probe.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"paths entry {p!r} escapes workspace") from exc
    completed = _run_git(["add", "--", *paths], cwd=resolved)
    if completed.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_add",
            "failure_kind": "git_add_failed",
            "stderr": (completed.stderr or "").strip(),
        }
    return {
        "ok": True,
        "cwd": str(resolved),
        "mutation_kind": "git_add",
        "paths": paths,
    }


def _git_show_result(
    rev: str,
    path: str | None = None,
    *,
    cwd: str | None = None,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    if not isinstance(rev, str) or not rev.strip():
        raise ValueError("rev must be a non-empty string")
    args = ["show", rev]
    if isinstance(path, str) and path.strip():
        args.extend(["--", path])
    completed = _run_git(args, cwd=resolved)
    if completed.returncode != 0:
        raise ValueError((completed.stderr or "git show failed").strip())
    text, truncated = _truncate(completed.stdout or "", max_chars)
    return {
        "ok": True,
        "cwd": str(resolved),
        "rev": rev,
        "path": path,
        "output": text,
        "truncated": truncated,
    }


def _git_changed_files_result(cwd: str | None = None) -> dict[str, Any]:
    status = _git_status_result(cwd=cwd)
    staged_files = [e["path"] for e in status["staged"]]
    unstaged_files = [e["path"] for e in status["unstaged"]]
    untracked_files = status["untracked"]
    return {
        "ok": True,
        "cwd": status["cwd"],
        "branch": status["branch"],
        "staged_files": staged_files,
        "unstaged_files": unstaged_files,
        "untracked_files": untracked_files,
        "staged_count": len(staged_files),
        "unstaged_count": len(unstaged_files),
        "untracked_count": len(untracked_files),
        "total_changed_count": len(staged_files) + len(unstaged_files) + len(untracked_files),
    }


def _git_restore_result(paths: list[str], *, cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    workspace = _workspace_root()
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list of strings")
    for p in paths:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("paths entries must be non-empty strings")
        candidate = Path(p)
        probe = candidate.resolve() if candidate.is_absolute() else (resolved / candidate).resolve()
        try:
            probe.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"paths entry {p!r} escapes workspace") from exc
    completed = _run_git(["restore", "--worktree", "--source=HEAD", "--", *paths], cwd=resolved)
    if completed.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_restore",
            "failure_kind": "git_restore_failed",
            "stderr": (completed.stderr or "").strip(),
        }
    return {
        "ok": True,
        "cwd": str(resolved),
        "mutation_kind": "git_restore",
        "paths": paths,
    }


def _git_unstage_result(paths: list[str], *, cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    workspace = _workspace_root()
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list of strings")
    for p in paths:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("paths entries must be non-empty strings")
        candidate = Path(p)
        probe = candidate.resolve() if candidate.is_absolute() else (resolved / candidate).resolve()
        try:
            probe.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"paths entry {p!r} escapes workspace") from exc
    completed = _run_git(["restore", "--staged", "--", *paths], cwd=resolved)
    if completed.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_unstage",
            "failure_kind": "git_unstage_failed",
            "stderr": (completed.stderr or "").strip(),
        }
    return {
        "ok": True,
        "cwd": str(resolved),
        "mutation_kind": "git_unstage",
        "paths": paths,
    }


def _git_checkout_branch_result(branch: str, *, cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("branch must be a non-empty string")
    verify = _run_git(["rev-parse", "--verify", branch], cwd=resolved)
    if verify.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_checkout_branch",
            "failure_kind": "branch_not_found",
            "branch": branch,
            "stderr": (verify.stderr or "").strip(),
        }
    completed = _run_git(["checkout", branch], cwd=resolved)
    if completed.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_checkout_branch",
            "failure_kind": "git_checkout_branch_failed",
            "branch": branch,
            "stderr": (completed.stderr or "").strip(),
        }
    current = _run_git(["branch", "--show-current"], cwd=resolved)
    current_branch = current.stdout.strip() if current.returncode == 0 else branch
    return {
        "ok": True,
        "cwd": str(resolved),
        "mutation_kind": "git_checkout_branch",
        "branch": current_branch,
    }


def _git_commit_result(message: str, *, cwd: str | None = None) -> dict[str, Any]:
    resolved = _resolve_cwd(cwd)
    if not isinstance(message, str) or not message.strip():
        raise ValueError("commit message must be a non-empty string")
    completed = _run_git(["commit", "-m", message], cwd=resolved)
    if completed.returncode != 0:
        return {
            "ok": False,
            "cwd": str(resolved),
            "mutation_kind": "git_commit",
            "failure_kind": "git_commit_failed",
            "stderr": (completed.stderr or completed.stdout or "").strip(),
        }
    # Extract commit hash from the commit output; fall back to rev-parse HEAD.
    rev = _run_git(["rev-parse", "HEAD"], cwd=resolved)
    commit_sha = rev.stdout.strip() if rev.returncode == 0 else ""
    subject = message.splitlines()[0] if message else ""
    return {
        "ok": True,
        "cwd": str(resolved),
        "mutation_kind": "git_commit",
        "commit": commit_sha,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
# FastMCP server wiring
# ---------------------------------------------------------------------------


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def git_status(cwd: str | None = None) -> dict[str, Any]:
    """Return branch + staged/unstaged/untracked counts and file lists."""
    return _git_status_result(cwd=cwd)


@mcp.tool()
def git_diff(
    cwd: str | None = None,
    path: str | None = None,
    staged: bool = False,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    """Return bounded diff text for the working tree or index."""
    return _git_diff_result(cwd=cwd, path=path, staged=staged, max_chars=max_chars)


@mcp.tool()
def git_log(cwd: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Return recent commit summaries (sha / subject / author / date)."""
    return _git_log_result(cwd=cwd, limit=limit)


@mcp.tool()
def git_show(
    rev: str,
    path: str | None = None,
    cwd: str | None = None,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> dict[str, Any]:
    """Show the contents of a commit or file at a given revision."""
    return _git_show_result(rev, path, cwd=cwd, max_chars=max_chars)


@mcp.tool()
def git_changed_files(cwd: str | None = None) -> dict[str, Any]:
    """Return staged, unstaged, and untracked file lists (flat names only)."""
    return _git_changed_files_result(cwd=cwd)


@mcp.tool()
def git_restore(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Restore workspace files to HEAD state (discards working-tree changes)."""
    return _git_restore_result(paths, cwd=cwd)


@mcp.tool()
def git_unstage(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Unstage the given paths (move from index back to working tree)."""
    return _git_unstage_result(paths, cwd=cwd)


@mcp.tool()
def git_checkout_branch(branch: str, cwd: str | None = None) -> dict[str, Any]:
    """Switch to an existing branch. Refuses if the branch does not exist."""
    return _git_checkout_branch_result(branch, cwd=cwd)


@mcp.tool()
def git_add(paths: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Stage the given workspace-relative paths."""
    return _git_add_result(paths=paths, cwd=cwd)


@mcp.tool()
def git_commit(message: str, cwd: str | None = None) -> dict[str, Any]:
    """Create a commit from currently-staged changes."""
    return _git_commit_result(message, cwd=cwd)


if __name__ == "__main__":
    mcp.run()
