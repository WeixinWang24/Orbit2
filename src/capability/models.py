from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# Path prefixes that capability tools must never touch, even within workspace.
# Any path whose first segment equals one of these, or begins with one of
# these followed by "/" or ".", is treated as a protected location.
PROTECTED_PATH_PREFIXES: tuple[str, ...] = (".runtime", ".env", ".envrc", ".git")


def is_protected_relative_path(relative: str) -> str | None:
    """Return the matched protected prefix if `relative` targets one, else None.

    `relative` must already be a workspace-relative POSIX-style path string.
    Matches `.git`, `.git/hooks/...`, `.env.local`, `.env/foo`, etc.
    """
    normalized = relative.replace("\\", "/")
    for prefix in PROTECTED_PATH_PREFIXES:
        if normalized == prefix:
            return prefix
        if normalized.startswith(prefix + "/"):
            return prefix
        if normalized.startswith(prefix + "."):
            return prefix
    return None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    content: str
    data: dict[str, Any] | None = None


class GovernanceOutcome(BaseModel):
    allowed: bool
    reason: str = ""


class CapabilityResult(BaseModel):
    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    governance_outcome: str = "allowed"
