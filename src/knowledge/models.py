from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.core.runtime.models import ConversationMessage, Message, MessageRole, TurnRequest


@dataclass
class ContextFragment:
    fragment_name: str
    visibility_scope: str
    content: str
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssembledContext:
    instruction_fragments: list[ContextFragment] = field(default_factory=list)
    transcript_messages: list[ConversationMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_turn_request(self) -> TurnRequest:
        system_parts = [
            fragment.content.strip()
            for fragment in sorted(
                self.instruction_fragments, key=lambda f: -f.priority
            )
            if fragment.content.strip()
        ]
        system = "\n\n".join(system_parts) if system_parts else None

        assembled: list[Message] = []
        for m in self.transcript_messages:
            if m.role == MessageRole.ASSISTANT and "tool_calls" in m.metadata:
                # Deep-copy the tool_calls blob so downstream provider adapters
                # cannot mutate canonical transcript metadata in place.
                assembled.append(Message(
                    role=m.role.value,
                    content=m.content or None,
                    tool_calls=copy.deepcopy(m.metadata["tool_calls"]),
                ))
            elif m.role == MessageRole.TOOL:
                assembled.append(Message(
                    role=m.role.value,
                    content=m.content,
                    tool_call_id=m.metadata.get("tool_call_id"),
                ))
            else:
                assembled.append(Message(role=m.role.value, content=m.content))
        return TurnRequest(system=system, messages=assembled)
