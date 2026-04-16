from __future__ import annotations

import copy

from src.knowledge.assembly.base import ContextAssembler
from src.knowledge.models import AssembledContext, ContextFragment
from src.runtime.models import ConversationMessage, TurnRequest

SESSION_SYSTEM_PROMPT_FRAGMENT_NAME = "session_system_prompt"
SESSION_SYSTEM_PROMPT_PRIORITY = 100


class StructuredContextAssembler(ContextAssembler):
    def __init__(
        self,
        *,
        extra_instruction_fragments: list[ContextFragment] | None = None,
    ) -> None:
        self._extra_instruction_fragments = [
            copy.deepcopy(f) for f in (extra_instruction_fragments or [])
        ]

    def assemble(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> TurnRequest:
        return self.assemble_structured(
            messages, system_prompt=system_prompt
        ).to_turn_request()

    def assemble_structured(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> AssembledContext:
        instruction_fragments: list[ContextFragment] = []
        if system_prompt is not None:
            instruction_fragments.append(
                ContextFragment(
                    fragment_name=SESSION_SYSTEM_PROMPT_FRAGMENT_NAME,
                    visibility_scope="instruction",
                    content=system_prompt,
                    priority=SESSION_SYSTEM_PROMPT_PRIORITY,
                    metadata={"origin": "session"},
                )
            )
        instruction_fragments.extend(
            copy.deepcopy(f) for f in self._extra_instruction_fragments
        )

        return AssembledContext(
            instruction_fragments=instruction_fragments,
            transcript_messages=list(messages),
            metadata={
                "assembler": "structured",
                "transcript_message_count": len(messages),
                "instruction_fragment_count": len(instruction_fragments),
            },
        )
