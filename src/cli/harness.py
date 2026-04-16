from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.capability.boundary import CapabilityBoundary
from src.capability.registry import CapabilityRegistry
from src.capability.tools import ReadFileTool
from src.providers.base import ExecutionBackend
from src.providers.codex import CodexBackend, CodexConfig
from src.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from src.runtime.session import SessionManager
from src.store.sqlite import SQLiteSessionStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_capability_boundary(workspace_root: Path) -> CapabilityBoundary:
    registry = CapabilityRegistry()
    registry.register(ReadFileTool(workspace_root))
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


def _run_interactive(manager: SessionManager, session_id: str) -> None:
    session = manager.get_session(session_id)
    print(f"Session: {session_id}")
    print(f"Backend: {session.backend_name}")
    print("Type your message. /quit to exit, /history to show transcript.\n")

    while True:
        try:
            user_input = input("\033[1myou>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped == "/quit":
            print("Bye.")
            return
        if stripped == "/history":
            _show_history(manager, session_id)
            continue

        sys.stdout.write("\033[2massistant>\033[0m ")
        sys.stdout.flush()

        last_len = 0

        def on_partial(accumulated: str) -> None:
            nonlocal last_len
            delta = accumulated[last_len:]
            last_len = len(accumulated)
            sys.stdout.write(delta)
            sys.stdout.flush()

        plan = manager.run_turn(session_id, stripped, on_partial_text=on_partial)

        if last_len == 0 and plan.final_text:
            sys.stdout.write(plan.final_text)
        if plan.final_text is None and last_len == 0:
            sys.stdout.write("[no response]")

        sys.stdout.write("\n\n")
        sys.stdout.flush()


def _show_history(manager: SessionManager, session_id: str) -> None:
    messages = manager.list_messages(session_id)
    if not messages:
        print("  (empty transcript)")
        return
    print(f"\n  Transcript ({len(messages)} messages):")
    for msg in messages:
        role = msg.role.value
        content = msg.content
        if len(content) > 120:
            content = content[:117] + "..."
        print(f"  [{role}] {content}")
    print()


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
                print("No sessions.")
            else:
                for s in sessions:
                    msg_count = len(manager.list_messages(s.session_id))
                    print(f"  {s.session_id}  [{s.backend_name}]  {msg_count} msgs  {s.status.value}")
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
