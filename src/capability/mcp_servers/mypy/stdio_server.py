"""Bounded mypy MCP server for Orbit2 (Handoff 17, first slice).

Ships one tool only: `run_mypy_structured(args, max_chars)`. Invokes mypy
(expected on PATH in the Orbit conda env) inside the workspace, captures
stdout/stderr and exit code, truncates per caller cap.

Not migrated from Orbit1's 272-line mypy server: strict-mode presets,
per-file severity rollup, incremental cache orchestration. Future per-family
slice.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

SERVER_NAME = "mypy"
WORKSPACE_ROOT_ENV = "ORBIT_WORKSPACE_ROOT"
DEFAULT_MAX_CHARS = 12_000
MYPY_TIMEOUT_SECONDS = 120.0


def _workspace_root() -> Path:
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif len(sys.argv) > 1 and sys.argv[-1].strip():
        root = Path(sys.argv[-1]).expanduser().resolve()
    else:
        raise ValueError(
            f"mypy MCP server requires allowed root via {WORKSPACE_ROOT_ENV} "
            "env var or trailing positional arg"
        )
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace root is invalid: {root}")
    return root


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _mypy_binary() -> str:
    binary = shutil.which("mypy")
    if binary is None:
        raise ValueError("mypy binary not found on PATH")
    return binary


def _validate_args(args: list[str] | None, default: list[str]) -> list[str]:
    """Reject absolute-path args in the forwarded CLI args (defense-in-depth).
    See the equivalent helper in the pytest server for rationale.
    """
    cli_args = list(args) if args else list(default)
    for a in cli_args:
        if not isinstance(a, str):
            raise ValueError(f"args entries must be strings; got {type(a).__name__}")
        if a.startswith("/") or a.startswith("\\\\"):
            raise ValueError(f"absolute paths are not allowed in args: {a!r}")
    return cli_args


def _run_mypy_structured_result(
    *,
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout_seconds: float = MYPY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    workspace = _workspace_root()
    cli_args = _validate_args(args, default=["--show-error-codes", "."])
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    try:
        completed = subprocess.run(
            [_mypy_binary(), *cli_args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "cwd": str(workspace),
            "args": cli_args,
            "returncode": None,
            "stdout": "",
            "stderr": f"mypy timed out after {timeout_seconds}s",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "failure_kind": "mypy_timeout",
            "timeout_seconds": timeout_seconds,
            "_exc": repr(exc),
        }
    stdout, stdout_truncated = _truncate(completed.stdout or "", max_chars)
    stderr, stderr_truncated = _truncate(completed.stderr or "", max_chars)
    return {
        # mypy returns 1 on type errors — that's a legitimate, non-fatal outcome.
        # We treat "ran" (exit 0 or 1) as ok=True; anything else is a tool failure.
        "ok": completed.returncode in (0, 1),
        "cwd": str(workspace),
        "args": cli_args,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "failure_kind": None if completed.returncode in (0, 1) else "mypy_nonzero_exit",
    }


mcp = FastMCP(SERVER_NAME)


@mcp.tool()
def run_mypy_structured(
    args: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Run mypy inside the workspace and return bounded output.

    `args` forwards directly to mypy (default: `["--show-error-codes", "."]`).
    Exit code 1 (type errors found) is treated as `ok=True` — it means mypy
    ran and produced diagnostics, not that the tool failed.
    """
    return _run_mypy_structured_result(args=args, max_chars=max_chars)


if __name__ == "__main__":
    mcp.run()
