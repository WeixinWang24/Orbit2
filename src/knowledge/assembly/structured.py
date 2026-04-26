from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from src.knowledge.assembly.base import ContextAssembler
from src.knowledge.capability_awareness import build_capability_awareness_fragment
from src.knowledge.models import AssembledContext, ContextFragment
from src.knowledge.runtime_context import build_runtime_context_fragment
from src.knowledge.workspace_instructions import build_workspace_instructions_fragment
from src.core.runtime.models import ConversationMessage, TurnRequest

if TYPE_CHECKING:
    from src.governance.capability_awareness_disclosure import (
        CapabilityAwarenessDisclosurePolicy,
    )
    from src.governance.disclosure import ExposureDecision
    from src.governance.runtime_context_disclosure import (
        RuntimeContextDisclosurePolicy,
    )
    from src.governance.workspace_instructions_disclosure import (
        WorkspaceInstructionsDisclosurePolicy,
    )
    from src.knowledge.capability_awareness import CapabilityAwarenessCollector
    from src.knowledge.runtime_context import RuntimeContextCollector
    from src.knowledge.workspace_instructions import (
        WorkspaceInstructionsCollector,
    )

SESSION_SYSTEM_PROMPT_FRAGMENT_NAME = "session_system_prompt"
SESSION_SYSTEM_PROMPT_PRIORITY = 100


class StructuredContextAssembler(ContextAssembler):
    def __init__(
        self,
        *,
        extra_instruction_fragments: list[ContextFragment] | None = None,
        runtime_context_collector: "RuntimeContextCollector | None" = None,
        runtime_context_disclosure_policy: "RuntimeContextDisclosurePolicy | None" = None,
        capability_awareness_collector: "CapabilityAwarenessCollector | None" = None,
        capability_awareness_disclosure_policy: "CapabilityAwarenessDisclosurePolicy | None" = None,
        workspace_instructions_collector: "WorkspaceInstructionsCollector | None" = None,
        workspace_instructions_disclosure_policy: "WorkspaceInstructionsDisclosurePolicy | None" = None,
    ) -> None:
        self._extra_instruction_fragments = [
            copy.deepcopy(f) for f in (extra_instruction_fragments or [])
        ]
        self._runtime_context_collector = runtime_context_collector
        self._runtime_context_disclosure_policy = runtime_context_disclosure_policy
        self._capability_awareness_collector = capability_awareness_collector
        self._capability_awareness_disclosure_policy = capability_awareness_disclosure_policy
        self._workspace_instructions_collector = workspace_instructions_collector
        self._workspace_instructions_disclosure_policy = (
            workspace_instructions_disclosure_policy
        )

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
        exposure_decision: "ExposureDecision | None" = None,
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
        workspace_fragment = self._workspace_instructions_fragment_or_none()
        if workspace_fragment is not None:
            instruction_fragments.append(workspace_fragment)
        runtime_fragment = self._runtime_context_fragment_or_none()
        if runtime_fragment is not None:
            instruction_fragments.append(runtime_fragment)
        awareness_fragment = self._capability_awareness_fragment_or_none(exposure_decision)
        if awareness_fragment is not None:
            instruction_fragments.append(awareness_fragment)
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

    def _workspace_instructions_fragment_or_none(self) -> ContextFragment | None:
        if (
            self._workspace_instructions_collector is None
            or self._workspace_instructions_disclosure_policy is None
        ):
            return None
        snapshot = self._workspace_instructions_collector.collect()
        decision = self._workspace_instructions_disclosure_policy.decide(snapshot)
        if not decision.disclose:
            return None
        return build_workspace_instructions_fragment(
            snapshot, policy_name=decision.policy_name
        )

    def _capability_awareness_fragment_or_none(
        self, exposure_decision: "ExposureDecision | None"
    ) -> ContextFragment | None:
        if (
            self._capability_awareness_collector is None
            or self._capability_awareness_disclosure_policy is None
        ):
            return None
        snapshot = self._capability_awareness_collector.collect(
            exposure_decision=exposure_decision
        )
        decision = self._capability_awareness_disclosure_policy.decide(snapshot)
        if not decision.disclose:
            return None
        return build_capability_awareness_fragment(
            snapshot, policy_name=decision.policy_name
        )
