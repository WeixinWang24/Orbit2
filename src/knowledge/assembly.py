from __future__ import annotations

from abc import ABC, abstractmethod

from src.runtime.models import ConversationMessage, Message, MessageRole, TurnRequest


class ContextAssembler(ABC):
    @abstractmethod
    def assemble(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> TurnRequest: ...


class TranscriptContextAssembler(ContextAssembler):
    def assemble(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> TurnRequest:
        assembled: list[Message] = []
        for m in messages:
            if m.role == MessageRole.ASSISTANT and "tool_calls" in m.metadata:
                assembled.append(Message(
                    role=m.role.value,
                    content=m.content or None,
                    tool_calls=m.metadata["tool_calls"],
                ))
            elif m.role == MessageRole.TOOL:
                assembled.append(Message(
                    role=m.role.value,
                    content=m.content,
                    tool_call_id=m.metadata.get("tool_call_id"),
                ))
            else:
                assembled.append(Message(role=m.role.value, content=m.content))
        return TurnRequest(system=system_prompt, messages=assembled)
