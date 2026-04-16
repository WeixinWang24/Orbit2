from __future__ import annotations

from abc import ABC, abstractmethod

from src.runtime.models import ConversationMessage, TurnRequest


class ContextAssembler(ABC):
    @abstractmethod
    def assemble(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> TurnRequest: ...
