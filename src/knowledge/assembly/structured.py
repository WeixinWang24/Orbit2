from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from src.knowledge.assembly.base import ContextAssembler
from src.knowledge.models import AssembledContext, ContextFragment
from src.knowledge.runtime_context import build_runtime_context_fragment
from src.core.runtime.models import ConversationMessage, TurnRequest

if TYPE_CHECKING:
    from src.governance.runtime_context_disclosure import (
        RuntimeContextDisclosurePolicy,
    )
    from src.knowledge.runtime_context import RuntimeContextCollector

SESSION_SYSTEM_PROMPT_FRAGMENT_NAME = "session_system_prompt"
SESSION_SYSTEM_PROMPT_PRIORITY = 100


class StructuredContextAssembler(ContextAssembler):
    def __init__(
        self,
        *,
        extra_instruction_fragments: list[ContextFragment] | None = None,
        runtime_context_collector: "RuntimeContextCollector | None" = None,
        runtime_context_disclosure_policy: "RuntimeContextDisclosurePolicy | None" = None,
    ) -> None:
        self._extra_instruction_fragments = [
            copy.deepcopy(f) for f in (extra_instruction_fragments or [])
        ]
        self._runtime_context_collector = runtime_context_collector
        self._runtime_context_disclosure_policy = runtime_context_disclosure_policy

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
        runtime_fragment = self._runtime_context_fragment_or_none()
        if runtime_fragment is not None:
            instruction_fragments.append(runtime_fragment)
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

    def _runtime_context_fragment_or_none(self) -> ContextFragment | None:
        if (
            self._runtime_context_collector is None
            or self._runtime_context_disclosure_policy is None
        ):
            return None
        snapshot = self._runtime_context_collector.collect()
        decision = self._runtime_context_disclosure_policy.decide(snapshot)
        if not decision.disclose or not decision.visible_fields:
            return None
        return build_runtime_context_fragment(
            snapshot,
            visible_fields=decision.visible_fields,
            policy_name=decision.policy_name,
        )
