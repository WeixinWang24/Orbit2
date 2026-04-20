"""Capability-awareness disclosure policy — Governance Surface owner of
whether the capability-awareness fragment is emitted on a given turn.

Mirrors the shape of `runtime_context_disclosure.py`: an abstract policy
with a `decide(snapshot) -> decision` contract and a basic default. The
basic default discloses only when progressive disclosure has actually
hidden something — if the session's exposure strategy has widened to cover
every group, the awareness block adds no value and is suppressed to keep
the instruction fragment set honest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.knowledge.capability_awareness import CapabilityAwarenessSnapshot


@dataclass(frozen=True)
class CapabilityAwarenessDisclosureDecision:
    disclose: bool
    policy_name: str


class CapabilityAwarenessDisclosurePolicy(ABC):
    @property
    @abstractmethod
    def policy_name(self) -> str: ...

    @abstractmethod
    def decide(
        self, snapshot: CapabilityAwarenessSnapshot
    ) -> CapabilityAwarenessDisclosureDecision: ...


class BasicCapabilityAwarenessDisclosurePolicy(CapabilityAwarenessDisclosurePolicy):
    _POLICY_NAME = "basic_capability_awareness"

    @property
    def policy_name(self) -> str:
        return self._POLICY_NAME

    def decide(
        self, snapshot: CapabilityAwarenessSnapshot
    ) -> CapabilityAwarenessDisclosureDecision:
        return CapabilityAwarenessDisclosureDecision(
            disclose=bool(snapshot.hidden_reveal_groups),
            policy_name=self._POLICY_NAME,
        )


DEFAULT_CAPABILITY_AWARENESS_DISCLOSURE_POLICY: CapabilityAwarenessDisclosurePolicy = (
    BasicCapabilityAwarenessDisclosurePolicy()
)
