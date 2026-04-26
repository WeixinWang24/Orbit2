"""Workspace-instructions disclosure policy — Governance Surface owner of
whether the `orbit.md` fragment is emitted on a given turn.

Mirrors the shape of `runtime_context_disclosure.py` and
`capability_awareness_disclosure.py`: an abstract policy with a
`decide(snapshot) -> decision` contract and a basic default. The basic
default discloses only when the `orbit.md` file exists and has
non-whitespace content; absence or empty content suppresses the
fragment so the instruction set does not gain a dangling empty block.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.knowledge.workspace_instructions import WorkspaceInstructionsSnapshot


@dataclass(frozen=True)
class WorkspaceInstructionsDisclosureDecision:
    disclose: bool
    policy_name: str


class WorkspaceInstructionsDisclosurePolicy(ABC):
    @property
    @abstractmethod
    def policy_name(self) -> str: ...

    @abstractmethod
    def decide(
        self, snapshot: WorkspaceInstructionsSnapshot
    ) -> WorkspaceInstructionsDisclosureDecision: ...


class BasicWorkspaceInstructionsDisclosurePolicy(WorkspaceInstructionsDisclosurePolicy):
    _POLICY_NAME = "basic_workspace_instructions"

    @property
    def policy_name(self) -> str:
        return self._POLICY_NAME

    def decide(
        self, snapshot: WorkspaceInstructionsSnapshot
    ) -> WorkspaceInstructionsDisclosureDecision:
        return WorkspaceInstructionsDisclosureDecision(
            disclose=snapshot.exists and bool(snapshot.content.strip()),
            policy_name=self._POLICY_NAME,
        )


DEFAULT_WORKSPACE_INSTRUCTIONS_DISCLOSURE_POLICY: WorkspaceInstructionsDisclosurePolicy = (
    BasicWorkspaceInstructionsDisclosurePolicy()
)
