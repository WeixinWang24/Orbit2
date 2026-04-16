from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


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
