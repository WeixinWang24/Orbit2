from __future__ import annotations

import json
import sys
from typing import Callable, TextIO

from src.governance.approval import (
    ApprovalDecision,
    ApprovalInteractor,
    ApprovalRequest,
)
from src.operation.cli.style import (
    ACCENT_ERROR,
    ACCENT_MUTED,
    ACCENT_SUCCESS,
    ACCENT_SYSTEM,
    ACCENT_TOOL,
    BOLD,
    DIM,
    RESET,
    divider,
)

PromptReader = Callable[[str], str]


def _default_reader(prompt: str) -> str:
    return input(prompt)


class CLIApprovalInteractor(ApprovalInteractor):
    """Operator-surface approval interactor for the CLI.

    Presents an approval request with tool name, reveal group, side-effect
    class, and a bounded argument preview, then reads a single operator
    choice:
      - `a` / `allow`   → ALLOW_ONCE
      - `s` / `similar` → ALLOW_SIMILAR_IN_SESSION
      - `d` / `deny`    → DENY

    Default on EOF / KeyboardInterrupt / unrecognised input is DENY. This
    interactor is a view over the gate's decision; it does not own approval
    truth or store any state.
    """

    _ARG_PREVIEW_LIMIT = 240

    def __init__(
        self,
        *,
        reader: PromptReader | None = None,
        out: TextIO | None = None,
    ) -> None:
        self._reader = reader or _default_reader
        self._out = out or sys.stdout

    def prompt(self, request: ApprovalRequest) -> ApprovalDecision:
        self._render_header(request)
        try:
            choice = self._reader(
                f"  {ACCENT_SYSTEM}[a]llow once / [s]imilar in session / [d]eny"
                f" {DIM}(default: deny){RESET}: "
            )
        except (EOFError, KeyboardInterrupt):
            self._write(f"\n  {DIM}No response; denying.{RESET}\n\n")
            return ApprovalDecision.DENY

        normalized = choice.strip().lower()
        if normalized in ("a", "allow", "allow-once", "allow_once", "o", "once"):
            self._write(f"  {ACCENT_SUCCESS}Allowed once.{RESET}\n\n")
            return ApprovalDecision.ALLOW_ONCE
        if normalized in ("s", "similar", "allow-similar", "allow_similar"):
            self._write(
                f"  {ACCENT_SUCCESS}Allowed — will reuse for"
                f" {ACCENT_TOOL}{request.reveal_group}{RESET}"
                f" {ACCENT_SUCCESS}this session.{RESET}\n\n"
            )
            return ApprovalDecision.ALLOW_SIMILAR_IN_SESSION
        # Anything else (including "d", "deny", empty, unrecognised input) denies.
        self._write(f"  {ACCENT_ERROR}Denied.{RESET}\n\n")
        return ApprovalDecision.DENY

    def _render_header(self, request: ApprovalRequest) -> None:
        self._write("\n")
        self._write(divider(60) + "\n")
        self._write(
            f"  {ACCENT_TOOL}{BOLD}Approval required{RESET}"
            f" {ACCENT_MUTED}\u2502{RESET}"
            f" {ACCENT_SYSTEM}{request.tool_name}{RESET}\n"
        )
        self._write(
            f"    {DIM}reveal group:{RESET} {ACCENT_TOOL}{request.reveal_group}{RESET}"
            f"  {DIM}side effect:{RESET} {ACCENT_TOOL}{request.side_effect_class}{RESET}\n"
        )
        if request.summary:
            summary = request.summary.strip().splitlines()[0]
            if len(summary) > 160:
                summary = summary[:157] + "..."
            self._write(f"    {DIM}summary:{RESET} {summary}\n")
        preview = self._format_arguments(request.arguments)
        if preview:
            self._write(f"    {DIM}arguments:{RESET} {preview}\n")
        self._write(divider(60) + "\n")

    def _format_arguments(self, arguments: dict) -> str:
        if not arguments:
            return ""
        try:
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            rendered = repr(arguments)
        if len(rendered) > self._ARG_PREVIEW_LIMIT:
            return rendered[: self._ARG_PREVIEW_LIMIT - 3] + "..."
        return rendered

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()
