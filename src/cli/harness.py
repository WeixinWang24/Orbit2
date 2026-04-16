from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.capability.boundary import CapabilityBoundary
from src.capability.registry import CapabilityRegistry
from src.capability.tools import (
    ApplyExactHunkTool,
    ReadFileTool,
    ReplaceAllInFileTool,
    ReplaceBlockInFileTool,
    ReplaceInFileTool,
    WriteFileTool,
)
from src.cli.style import (
    ACCENT_ASSISTANT,
    ACCENT_ERROR,
    ACCENT_MUTED,
    ACCENT_SUCCESS,
    ACCENT_SYSTEM,
    ACCENT_TOOL,
    ACCENT_USER,
    BOLD,
    CONTENT_ASSISTANT,
    CONTENT_USER,
    DIM,
    RESET,
    divider,
)
from src.providers.base import ExecutionBackend
from src.providers.codex import CodexBackend, CodexConfig
from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from src.runtime.session import SessionManager
from src.store.sqlite import SQLiteSessionStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_capability_boundary(workspace_root: Path) -> CapabilityBoundary:
    registry = CapabilityRegistry()
    registry.register(ReadFileTool(workspace_root))
    registry.register(WriteFileTool(workspace_root))
    registry.register(ReplaceInFileTool(workspace_root))
    registry.register(ReplaceAllInFileTool(workspace_root))
    registry.register(ReplaceBlockInFileTool(workspace_root))
    registry.register(ApplyExactHunkTool(workspace_root))
    return CapabilityBoundary(registry, workspace_root)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _build_backend(name: str) -> ExecutionBackend:
    if name == "codex":
        return CodexBackend(CodexConfig(), repo_root=REPO_ROOT)
    if name == "vllm":
        return OpenAICompatibleBackend(
            OpenAICompatibleConfig(base_url="http://localhost:8000/v1"),
        )
    raise ValueError(f"Unknown backend: {name}")


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _read_input(prompt: str) -> str:
    """Read user input with CJK-aware composer when on a real TTY."""
    if sys.stdin.isatty():
        try:
            from src.cli.composer import read_line
            return read_line(prompt)
        except (ImportError, OSError):
            pass
    return input(prompt)


def _run_interactive(manager: SessionManager, session_id: str) -> None:
    session = manager.get_session(session_id)
    width = _term_width()

    sys.stdout.write(
        f"{ACCENT_SUCCESS}{BOLD}Orbit2{RESET}"
        f" {ACCENT_MUTED}\u2502{RESET}"
        f" {ACCENT_SYSTEM}{session_id}{RESET}"
        f" {ACCENT_MUTED}\u2502{RESET}"
        f" {DIM}{session.backend_name}{RESET}\n"
    )
    sys.stdout.write(divider(width) + "\n")
    sys.stdout.write(
        f"{DIM}Commands: /quit  /history  /clear{RESET}\n\n"
    )
    sys.stdout.flush()

    prompt = f"{ACCENT_USER}{BOLD}you \u276f{RESET} "

    while True:
        try:
            user_input = _read_input(prompt)
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write(f"\n{DIM}Bye.{RESET}\n")
            sys.stdout.flush()
            return

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped == "/quit":
            sys.stdout.write(f"{DIM}Bye.{RESET}\n")
            sys.stdout.flush()
            return
        if stripped == "/history":
            _show_history(manager, session_id)
            continue
        if stripped == "/clear":
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()
            continue

        sys.stdout.write(f"\n{ACCENT_ASSISTANT}{BOLD}assistant \u276f{RESET} ")
        sys.stdout.flush()

        last_len = 0

        def on_partial(accumulated: str) -> None:
            nonlocal last_len
            delta = accumulated[last_len:]
            last_len = len(accumulated)
            sys.stdout.write(f"{CONTENT_ASSISTANT}{delta}{RESET}")
            sys.stdout.flush()

        plan = manager.run_turn(session_id, stripped, on_partial_text=on_partial)

        if last_len == 0 and plan.final_text:
            sys.stdout.write(f"{CONTENT_ASSISTANT}{plan.final_text}{RESET}")
        if plan.final_text is None and last_len == 0:
            sys.stdout.write(f"{ACCENT_ERROR}[no response]{RESET}")

        if plan.tool_requests:
            sys.stdout.write("\n")
            for tr in plan.tool_requests:
                status = "\u2713" if tr.tool_name else "\u2717"
                sys.stdout.write(
                    f"  {ACCENT_TOOL}{status} {tr.tool_name or 'unknown'}{RESET}"
                    f" {DIM}{tr.tool_call_id or ''}{RESET}\n"
                )

        sys.stdout.write("\n\n")
        sys.stdout.flush()


def _show_history(manager: SessionManager, session_id: str) -> None:
    messages = manager.list_messages(session_id)
    width = _term_width()
    if not messages:
        sys.stdout.write(f"  {DIM}(empty transcript){RESET}\n")
        sys.stdout.flush()
        return

    sys.stdout.write(f"\n{divider(width)}\n")
    sys.stdout.write(
        f"  {ACCENT_SYSTEM}Transcript{RESET}"
        f" {DIM}({len(messages)} messages){RESET}\n"
    )
    sys.stdout.write(f"{divider(width)}\n")

    for msg in messages:
        role = msg.role.value
        content = msg.content
        if len(content) > 120:
            content = content[:117] + "..."

        if role == "user":
            label_color = ACCENT_USER
            body_color = CONTENT_USER
        elif role == "assistant":
            label_color = ACCENT_ASSISTANT
            body_color = CONTENT_ASSISTANT
        elif role == "tool":
            label_color = ACCENT_TOOL
            body_color = DIM
        else:
            label_color = ACCENT_MUTED
            body_color = DIM

        sys.stdout.write(
            f"  {label_color}{BOLD}{role}{RESET}"
            f" {body_color}{content}{RESET}\n"
        )

    sys.stdout.write(f"{divider(width)}\n\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit2 CLI")
    parser.add_argument(
        "--backend", "-b",
        default="codex",
        choices=["codex", "vllm"],
        help="Execution backend (default: codex)",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Resume an existing session by ID",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for new sessions",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List existing sessions and exit",
    )
    args = parser.parse_args()

    db_path = REPO_ROOT / ".runtime" / "sessions.db"
    store = SQLiteSessionStore(db_path)
    try:
        backend = _build_backend(args.backend)
        boundary = _build_capability_boundary(REPO_ROOT)
        manager = SessionManager(backend=backend, store=store, capability_boundary=boundary)

        if args.list_sessions:
            sessions = manager.list_sessions()
            if not sessions:
                sys.stdout.write(f"{DIM}No sessions.{RESET}\n")
            else:
                for s in sessions:
                    msg_count = len(manager.list_messages(s.session_id))
                    sys.stdout.write(
                        f"  {ACCENT_SYSTEM}{s.session_id}{RESET}"
                        f"  {DIM}[{s.backend_name}]{RESET}"
                        f"  {DIM}{msg_count} msgs{RESET}"
                        f"  {DIM}{s.status.value}{RESET}\n"
                    )
            sys.stdout.flush()
            return

        if args.session:
            session = manager.get_session(args.session)
        else:
            session = manager.create_session(system_prompt=args.system)

        _run_interactive(manager, session.session_id)
    finally:
        store.close()


if __name__ == "__main__":
    main()
