from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None


class ToolRequest(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    provider_item_id: Optional[str] = None


class TurnRequest(BaseModel):
    messages: list[Message]
    system: Optional[str] = None
    tool_definitions: Optional[list[dict]] = None


class ProviderNormalizedResult(BaseModel):
    source_backend: str
    plan_label: str
    final_text: Optional[str] = None
    model: str
    metadata: dict = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    source_backend: str
    plan_label: str
    final_text: Optional[str] = None
    model: str
    metadata: dict = Field(default_factory=dict)
    tool_requests: list[ToolRequest] = Field(default_factory=list)


class Session(BaseModel):
    session_id: str
    backend_name: str
    system_prompt: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    metadata: dict = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    message_id: str
    session_id: str
    role: MessageRole
    content: str
    turn_index: int
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


def make_session_id() -> str:
    return f"session_{uuid4().hex[:12]}"


def make_message_id() -> str:
    return f"msg_{uuid4().hex[:12]}"
