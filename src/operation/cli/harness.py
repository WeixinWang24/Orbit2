from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from src.capability.boundary import CapabilityBoundary
from src.capability.discovery import ListAvailableToolsTool
from src.capability.mcp.attach import attach_mcp_server
from src.capability.mcp.models import McpClientBootstrap
from src.capability.mcp_servers import (
    DEFAULT_WORKSPACE_MCP_SERVER_MODULES,
    OBSIDIAN_MCP_SERVER_MODULE,
)
from src.capability.registry import CapabilityRegistry
from src.capability.tools import (
    ApplyExactHunkTool,
    ReadFileTool,
    ReplaceAllInFileTool,
    ReplaceBlockInFileTool,
    ReplaceInFileTool,
    WriteFileTool,
)
from src.governance.approval import ApprovalGate, ApprovalMemory
from src.governance.capability_awareness_disclosure import (
    BasicCapabilityAwarenessDisclosurePolicy,
)
from src.governance.disclosure import LayerAwareDisclosureStrategy
from src.governance.policies import RevealGroupSessionApprovalPolicy
from src.governance.runtime_context_disclosure import (
    BasicSelfLocationDisclosurePolicy,
)
from src.governance.workspace_instructions_disclosure import (
    BasicWorkspaceInstructionsDisclosurePolicy,
)
from src.knowledge.assembly import StructuredContextAssembler
from src.knowledge.capability_awareness import CapabilityAwarenessCollector
from src.knowledge.runtime_context import RuntimeContextCollector
from src.knowledge.workspace_instructions import WorkspaceInstructionsCollector
from src.operation.cli.approval import CLIApprovalInteractor
from src.operation.cli.composer import (
    ComposerAction,
    PageDownAction,
    PageUpAction,
)
from src.operation.cli.markdown import (
    render_markdown_for_terminal,
    wrap_ansi_text_for_terminal,
)
from src.operation.cli.style import (
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
from src.core.providers.base import ExecutionBackend
from src.core.providers.codex import CodexBackend, CodexConfig
from src.core.providers.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from src.config.runtime import (
    REPO_ROOT,
    default_db_path,
    resolve_obsidian_vault_root,
    resolve_provider_model,
    resolve_runtime_root,
    resolve_vllm_provider_settings,
)
from src.core.runtime.session import SessionManager
from src.core.store.sqlite import SQLiteSessionStore


def _build_capability_boundary(
    workspace_root: Path,
    *,
    obsidian_vault_root: Path | None = None,
) -> CapabilityBoundary:
    registry = CapabilityRegistry()
    registry.register(ReadFileTool(workspace_root))
    registry.register(WriteFileTool(workspace_root))
    registry.register(ReplaceInFileTool(workspace_root))
    registry.register(ReplaceAllInFileTool(workspace_root))
    registry.register(ReplaceBlockInFileTool(workspace_root))
    registry.register(ApplyExactHunkTool(workspace_root))

    shared_env = _workspace_mcp_env(workspace_root)
    # MCP families that are workspace-root scoped are wired here so the
    # default operator capability boundary reaches them without per-session
    # configuration. Obsidian is attached below only when the runtime provides
    # a vault path, because a vault is not necessarily the Orbit2 workspace.
    for server_module in DEFAULT_WORKSPACE_MCP_SERVER_MODULES:
        attach_mcp_server(
            McpClientBootstrap(
                server_name=server_module.server_name,
                command=sys.executable,
                args=("-m", server_module.module_path, str(workspace_root)),
                env=shared_env,
            ),
            registry,
        )

    if obsidian_vault_root is not None:
        vault_root = obsidian_vault_root.expanduser().resolve()
        if not vault_root.exists() or not vault_root.is_dir():
            raise ValueError(f"obsidian vault root is invalid: {vault_root}")
        attach_mcp_server(
            McpClientBootstrap(
                server_name="obsidian",
                command=sys.executable,
                args=(
                    "-m",
                    OBSIDIAN_MCP_SERVER_MODULE.module_path,
                    "--vault",
                    str(vault_root),
                ),
                env=shared_env,
            ),
            registry,
        )

    # Progressive-exposure discovery tool (Handoff 19). Registered AFTER all
    # native + MCP attachments so its view of the registry matches what the
    # discovery summary will report. Default-exposed + self-lives in the
    # `discovery` reveal group via the tool class itself.
    registry.register(ListAvailableToolsTool(registry))

    # Governance Surface approval gate (Handoff 20). Default policy is
    # session-scope reveal-group reuse. Interactor prompts via stdin on the
    # CLI operator surface; approval truth stays with the gate, not CLI.
    approval_memory = ApprovalMemory()
    approval_policy = RevealGroupSessionApprovalPolicy(memory=approval_memory)
    approval_interactor = CLIApprovalInteractor()
    approval_gate = ApprovalGate(
        policy=approval_policy,
        interactor=approval_interactor,
    )
    return CapabilityBoundary(
        registry, workspace_root, approval_gate=approval_gate,
    )


def _workspace_mcp_env(workspace_root: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("ORBIT2_MCP_")
    }
    env["ORBIT_WORKSPACE_ROOT"] = str(workspace_root)
    return env


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _build_backend(name: str, model: str, runtime_root: Path | None = None) -> ExecutionBackend:
    if name == "codex":
        return CodexBackend(CodexConfig(model=model), repo_root=REPO_ROOT)
    if name == "vllm":
        vllm_settings = resolve_vllm_provider_settings(runtime_root or REPO_ROOT)
        return OpenAICompatibleBackend(
            OpenAICompatibleConfig(
                model=model,
                base_url=vllm_settings.base_url,
                api_key=vllm_settings.api_key,
                basic_auth_username=vllm_settings.basic_auth_username,
                basic_auth_password=vllm_settings.basic_auth_password,
            ),
        )
    raise ValueError(f"Unknown backend: {name}")


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def _read_input(prompt: str) -> str:
    """Read user input with CJK-aware composer when on a real TTY.

    May raise PageUpAction or PageDownAction from the composer.
    """
    if sys.stdin.isatty():
        try:
            from src.operation.cli.composer import read_line
            return read_line(prompt)
        except (ImportError, OSError):
            pass
    return input(prompt)


def _write(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _wrap_for_tty(text: str, width: int) -> str:
    return wrap_ansi_text_for_terminal(text, width) if width > 0 else text


def _switch_session_relative(
    manager: SessionManager, current_id: str, *, direction: ComposerAction,
) -> str | None:
    """Move to previous (PageUp) or next (PageDown) session. Returns new session_id or None."""
    sessions = manager.list_sessions()
    if len(sessions) <= 1:
        return None
    current_idx = next(
        (i for i, s in enumerate(sessions) if s.session_id == current_id), -1,
    )
    if current_idx < 0:
        return None
    if isinstance(direction, PageUpAction):
        new_idx = current_idx - 1
    elif isinstance(direction, PageDownAction):
        new_idx = current_idx + 1
    else:
        return None
    if 0 <= new_idx < len(sessions):
        return sessions[new_idx].session_id
    return None


def _run_interactive(
    manager: SessionManager,
    session_id: str,
    *,
    boundary: CapabilityBoundary | None = None,
) -> None:
    active_session_id = session_id
    session = manager.get_session(active_session_id)
    width = _term_width()

    _write(
        f"{ACCENT_SUCCESS}{BOLD}Orbit2{RESET}"
        f" {ACCENT_MUTED}\u2502{RESET}"
        f" {ACCENT_SYSTEM}{active_session_id}{RESET}"
        f" {ACCENT_MUTED}\u2502{RESET}"
        f" {DIM}{session.backend_name}{RESET}\n"
    )
    _write(divider(width) + "\n")
    _write(
        f"{DIM}Commands: /quit  /history  /clear  /sessions  /switch  /new"
        f"  /delete-all  /reset-permission{RESET}\n\n"
    )

    prompt = f"{ACCENT_USER}{BOLD}you \u276f{RESET} "

    while True:
        try:
            user_input = _read_input(prompt)
        except (EOFError, KeyboardInterrupt):
            _write(f"\n{DIM}Bye.{RESET}\n")
            return
        except ComposerAction as action:
            new_id = _switch_session_relative(
                manager, active_session_id, direction=action,
            )
            if new_id is not None and new_id != active_session_id:
                active_session_id = new_id
                session = manager.get_session(active_session_id)
                _write(
                    f"\r{ACCENT_SUCCESS}Switched to{RESET}"
                    f" {ACCENT_SYSTEM}{active_session_id}{RESET}"
                    f" {DIM}({len(manager.list_messages(active_session_id))} msgs){RESET}\n"
                )
            continue

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped == "/quit":
            _write(f"{DIM}Bye.{RESET}\n")
            return
        if stripped == "/history":
            _show_history(manager, active_session_id)
            continue
        if stripped == "/clear":
            _write("\x1b[2J\x1b[H")
            continue
        if stripped == "/sessions":
            _show_sessions(manager, active_session_id)
            continue
        if stripped.startswith("/switch"):
            result = _handle_switch(manager, active_session_id, stripped)
            if result is not None and result != active_session_id:
                active_session_id = result
                session = manager.get_session(active_session_id)
                _write(
                    f"{ACCENT_SUCCESS}Switched to{RESET}"
                    f" {ACCENT_SYSTEM}{active_session_id}{RESET}"
                    f" {DIM}({len(manager.list_messages(active_session_id))} msgs){RESET}\n\n"
                )
            elif result == active_session_id:
                _write(f"  {DIM}Already on this session.{RESET}\n")
            continue
        if stripped == "/new":
            # Reload current session from store to get latest system_prompt
            current = manager.get_session(active_session_id)
            session = manager.create_session(system_prompt=current.system_prompt)
            active_session_id = session.session_id
            _write(
                f"{ACCENT_SUCCESS}New session{RESET}"
                f" {ACCENT_SYSTEM}{active_session_id}{RESET}\n\n"
            )
            continue
        if stripped == "/delete-all":
            deleted = _handle_delete_all(manager)
            if deleted:
                # Deletion confirmed — create a fresh session
                session = manager.create_session(system_prompt=DEFAULT_SYSTEM_PROMPT)
                active_session_id = session.session_id
                _write(
                    f"{ACCENT_SUCCESS}New session{RESET}"
                    f" {ACCENT_SYSTEM}{active_session_id}{RESET}\n\n"
                )
            continue
        if stripped == "/reset-permission":
            _handle_reset_permission(boundary, active_session_id)
            continue

        _write(f"\n{ACCENT_ASSISTANT}{BOLD}assistant \u276f{RESET}\n")

        is_tty = sys.stdout.isatty()
        term_width = _term_width() if is_tty else 0
        last_len = 0
        status_frames = ("-", "\\", "|", "/")
        status_index = 0
        status_shown = False

        def on_partial(accumulated: str) -> None:
            nonlocal last_len, status_index, status_shown
            delta = accumulated[last_len:]
            last_len = len(accumulated)
            if is_tty:
                frame = status_frames[status_index % len(status_frames)]
                status_index += 1
                status_shown = True
                _write(f"\r\x1b[K{DIM}thinking {frame} {last_len} chars{RESET}")
            else:
                _write(f"{CONTENT_ASSISTANT}{delta}{RESET}")

        plan = manager.run_turn(active_session_id, stripped, on_partial_text=on_partial)

        if plan.final_text:
            rendered = render_markdown_for_terminal(
                plan.final_text, base_color=CONTENT_ASSISTANT,
            )
            if is_tty and last_len > 0:
                if status_shown:
                    _write("\r\x1b[K")
                _write(f"{CONTENT_ASSISTANT}{_wrap_for_tty(rendered, term_width)}{RESET}")
            elif last_len == 0:
                rendered = _wrap_for_tty(rendered, term_width if is_tty else 0)
                _write(f"{CONTENT_ASSISTANT}{rendered}{RESET}")
            # Non-TTY with streaming: raw deltas already emitted; leave
            # them as the canonical output for piped consumers.
        elif last_len == 0:
            _write(f"{ACCENT_ERROR}[no response]{RESET}")
        elif is_tty and status_shown:
            _write("\r\x1b[K")

        if plan.tool_requests:
            _write("\n")
            for tr in plan.tool_requests:
                status = "\u2713" if tr.tool_name else "\u2717"
                _write(
                    f"  {ACCENT_TOOL}{status} {tr.tool_name or 'unknown'}{RESET}"
                    f" {DIM}{tr.tool_call_id or ''}{RESET}\n"
                )

        _write("\n\n")


def _show_sessions(manager: SessionManager, active_session_id: str) -> None:
    sessions = manager.list_sessions()
    width = _term_width()
    if not sessions:
        _write(f"  {DIM}(no sessions){RESET}\n")
        return

    _write(f"\n{divider(width)}\n")
    _write(f"  {ACCENT_SYSTEM}Sessions{RESET} {DIM}({len(sessions)}){RESET}\n")
    _write(f"{divider(width)}\n")

    for i, s in enumerate(sessions):
        msg_count = len(manager.list_messages(s.session_id))
        marker = f"{ACCENT_SUCCESS}\u25b6{RESET} " if s.session_id == active_session_id else "  "
        _write(
            f"  {marker}"
            f"{DIM}{i + 1}.{RESET} "
            f"{ACCENT_SYSTEM}{s.session_id}{RESET}"
            f"  {DIM}[{s.backend_name}]{RESET}"
            f"  {DIM}{msg_count} msgs{RESET}"
            f"  {DIM}{s.status.value}{RESET}\n"
        )

    _write(f"{divider(width)}\n")
    _write(f"  {DIM}/switch <N> or /switch <session_id>{RESET}\n\n")


def _handle_switch(
    manager: SessionManager, current_id: str, command: str,
) -> str | None:
    """Parse /switch command and return new session_id, or None on failure."""
    parts = command.split(None, 1)
    if len(parts) < 2:
        _write(f"  {ACCENT_ERROR}Usage: /switch <N> or /switch <session_id>{RESET}\n")
        return None

    target = parts[1].strip()
    sessions = manager.list_sessions()

    # Try numeric index first
    try:
        idx = int(target)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1].session_id
        _write(f"  {ACCENT_ERROR}Index out of range (1-{len(sessions)}){RESET}\n")
        return None
    except ValueError:
        pass

    # Try session_id match (prefix or full)
    matches = [s for s in sessions if s.session_id == target]
    if not matches:
        matches = [s for s in sessions if s.session_id.startswith(target)]
    if len(matches) == 1:
        return matches[0].session_id
    if len(matches) > 1:
        _write(f"  {ACCENT_ERROR}Ambiguous prefix, {len(matches)} matches{RESET}\n")
        return None

    _write(f"  {ACCENT_ERROR}Session not found: {target}{RESET}\n")
    return None


def _handle_reset_permission(
    boundary: CapabilityBoundary | None, session_id: str,
) -> None:
    """Reset only the current session's approval reuse state.

    Scope: approval reuse memory only. Transcript, session records, and
    progressive-exposure reveal state are all left untouched so operators can
    retract a blanket 'approve similar' without losing unrelated session
    history or exposure decisions.
    """
    if boundary is None or boundary.approval_gate is None:
        _write(
            f"  {DIM}(no approval gate configured — nothing to reset){RESET}\n\n"
        )
        return
    boundary.approval_gate.reset_session(session_id)
    _write(
        f"  {ACCENT_SUCCESS}Reset approval memory for{RESET}"
        f" {ACCENT_SYSTEM}{session_id}{RESET}"
        f" {DIM}— similar write actions will prompt again.{RESET}\n\n"
    )


def _handle_delete_all(manager: SessionManager) -> bool:
    """Delete all sessions with explicit confirmation. Returns True if deletion was confirmed."""
    sessions = manager.list_sessions()
    if not sessions:
        _write(f"  {DIM}No sessions to delete.{RESET}\n")
        return False

    count = len(sessions)
    _write(
        f"\n  {ACCENT_ERROR}{BOLD}WARNING:{RESET}"
        f" This will permanently delete {ACCENT_ERROR}{count}{RESET}"
        f" session(s) and all their messages.\n"
    )
    _write(f"  {DIM}This action cannot be undone.{RESET}\n\n")

    try:
        confirmation = _read_input(
            f"  {ACCENT_ERROR}Type 'yes' to confirm:{RESET} "
        )
    except (EOFError, KeyboardInterrupt, ComposerAction):
        _write(f"\n  {DIM}Cancelled.{RESET}\n\n")
        return False

    if confirmation.strip() != "yes":
        _write(f"  {DIM}Cancelled.{RESET}\n\n")
        return False

    deleted = manager.delete_all_sessions()
    _write(f"  {ACCENT_SUCCESS}Deleted {deleted} session(s).{RESET}\n\n")
    return True


def _show_history(manager: SessionManager, session_id: str) -> None:
    messages = manager.list_messages(session_id)
    width = _term_width()
    if not messages:
        _write(f"  {DIM}(empty transcript){RESET}\n")
        return

    _write(f"\n{divider(width)}\n")
    _write(
        f"  {ACCENT_SYSTEM}Transcript{RESET}"
        f" {DIM}({len(messages)} messages){RESET}\n"
    )
    _write(f"{divider(width)}\n")

    for msg in messages:
        role = msg.role.value
        content = msg.content

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

        if role == "assistant":
            # Markdown-aware rendering for assistant messages only. Fenced
            # code blocks + inline code + bold are recognised; other roles
            # stay verbatim so user input and tool output aren't reformatted.
            rendered = render_markdown_for_terminal(
                content, base_color=body_color,
            )
            rendered = _wrap_for_tty(rendered, max(0, width - len(role) - 3))
            _write(
                f"  {label_color}{BOLD}{role}{RESET}"
                f" {body_color}{rendered}{RESET}\n"
            )
        else:
            if len(content) > 120:
                content = content[:117] + "..."
            _write(
                f"  {label_color}{BOLD}{role}{RESET}"
                f" {body_color}{content}{RESET}\n"
            )

    _write(f"{divider(width)}\n\n")


def main(argv: Sequence[str] | None = None) -> None:
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
    parser.add_argument(
        "--runtime-root",
        default=None,
        help=(
            "Effective runtime/workspace/store root. Overrides ORBIT2_RUNTIME_ROOT. "
            "Defaults to the Orbit2 repo checkout. Process cwd is never consulted."
        ),
    )
    parser.add_argument(
        "--obsidian-vault-root",
        default=None,
        help=(
            "Optional Obsidian vault root to attach as the obsidian MCP server. "
            "Overrides .runtime/agent_runtime.toml [obsidian].vault_root."
        ),
    )
    args = parser.parse_args(argv)

    runtime_root = resolve_runtime_root(args.runtime_root)
    db_path = default_db_path(runtime_root)
    _write(
        f"{DIM}runtime_root={runtime_root.path} "
        f"(source={runtime_root.source})  store={db_path}{RESET}\n"
    )
    store = SQLiteSessionStore(db_path)
    try:
        provider_model = resolve_provider_model(runtime_root)
        backend = _build_backend(args.backend, provider_model.value, runtime_root.path)
        obsidian_vault_root = resolve_obsidian_vault_root(
            runtime_root,
            args.obsidian_vault_root,
        )
        boundary = _build_capability_boundary(
            runtime_root.path,
            obsidian_vault_root=(
                obsidian_vault_root.path if obsidian_vault_root else None
            ),
        )
        assembler = StructuredContextAssembler(
            runtime_context_collector=RuntimeContextCollector(runtime_root, db_path),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
            capability_awareness_collector=CapabilityAwarenessCollector(
                boundary.registry
            ),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
            workspace_instructions_collector=WorkspaceInstructionsCollector(
                runtime_root.path
            ),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        manager = SessionManager(
            backend=backend,
            store=store,
            assembler=assembler,
            capability_boundary=boundary,
            disclosure_strategy=LayerAwareDisclosureStrategy(),
        )

        if args.list_sessions:
            sessions = manager.list_sessions()
            if not sessions:
                _write(f"{DIM}No sessions.{RESET}\n")
            else:
                for s in sessions:
                    msg_count = len(manager.list_messages(s.session_id))
                    _write(
                        f"  {ACCENT_SYSTEM}{s.session_id}{RESET}"
                        f"  {DIM}[{s.backend_name}]{RESET}"
                        f"  {DIM}{msg_count} msgs{RESET}"
                        f"  {DIM}{s.status.value}{RESET}\n"
                    )
            return

        if args.session:
            session = manager.get_session(args.session)
        else:
            session = manager.create_session(system_prompt=args.system)

        _run_interactive(manager, session.session_id, boundary=boundary)
    finally:
        store.close()


if __name__ == "__main__":
    main()
