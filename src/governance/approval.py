from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SIMILAR_IN_SESSION = "allow_similar_in_session"
    DENY = "deny"


# Reason codes for ApprovalOutcome.reason. Audit-observable strings — they
# end up inside `CapabilityResult.governance_outcome` so post-hoc transcript
# review can distinguish (a) tools that didn't need approval, (b) approvals
# reused from session memory, (c) fresh operator approvals, and (d) operator
# denials. Keep this list closed: each code maps to exactly one code path
# through the gate, and extending the set is a governance-visible change.
APPROVAL_REASON_NO_APPROVAL_REQUIRED = "no_approval_required"
APPROVAL_REASON_REUSED_FROM_SESSION = "reused_from_session"
APPROVAL_REASON_OPERATOR_ALLOW_ONCE = "operator_allow_once"
APPROVAL_REASON_OPERATOR_ALLOW_SIMILAR = "operator_allow_similar_in_session"
APPROVAL_REASON_OPERATOR_DENY = "operator_deny"


class ApprovalRequest(BaseModel):
    session_id: str
    tool_name: str
    reveal_group: str
    side_effect_class: str
    requires_approval: bool
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class ApprovalOutcome(BaseModel):
    decision: ApprovalDecision
    reason: str


class ApprovalInteractor(ABC):
    @abstractmethod
    def prompt(self, request: ApprovalRequest) -> ApprovalDecision: ...


class ApprovalPolicy(ABC):
    @abstractmethod
    def check_reuse(self, request: ApprovalRequest) -> bool: ...

    @abstractmethod
    def record(self, request: ApprovalRequest, decision: ApprovalDecision) -> None: ...

    @abstractmethod
    def reset_session(self, session_id: str) -> None: ...


class ApprovalMemory:
    """Session-scoped approval reuse state.

    Holds a per-session set of reveal groups that the operator has allowed to
    reuse within that session. Lives entirely in process memory — no
    persistence — so restarting the CLI forgets all approvals.
    """

    def __init__(self) -> None:
        self._approved: dict[str, set[str]] = {}

    def is_approved(self, session_id: str, reveal_group: str) -> bool:
        return reveal_group in self._approved.get(session_id, set())

    def remember(self, session_id: str, reveal_group: str) -> None:
        self._approved.setdefault(session_id, set()).add(reveal_group)

    def reset(self, session_id: str) -> int:
        removed = self._approved.pop(session_id, set())
        return len(removed)

    def snapshot(self, session_id: str) -> list[str]:
        return sorted(self._approved.get(session_id, set()))


class ApprovalGate:
    """Runtime approval decision seam.

    The gate is the governance-surface owner of approval truth. CapabilityBoundary
    calls `resolve()` for any `requires_approval=True` tool before executing it;
    the gate consults its policy (reuse eligibility) and, if no reuse is granted,
    its interactor (operator prompt). Any decision that grants session-scope
    similar-approval reuse is recorded back through the policy so later calls in
    the same reveal group skip the prompt.
    """

    def __init__(
        self,
        *,
        policy: ApprovalPolicy,
        interactor: ApprovalInteractor,
    ) -> None:
        self._policy = policy
        self._interactor = interactor

    def resolve(self, request: ApprovalRequest) -> ApprovalOutcome:
        if not request.requires_approval:
            return ApprovalOutcome(
                decision=ApprovalDecision.ALLOW_ONCE,
                reason=APPROVAL_REASON_NO_APPROVAL_REQUIRED,
            )
        if self._policy.check_reuse(request):
            return ApprovalOutcome(
                decision=ApprovalDecision.ALLOW_ONCE,
                reason=APPROVAL_REASON_REUSED_FROM_SESSION,
            )
        decision = self._interactor.prompt(request)
        if decision == ApprovalDecision.ALLOW_SIMILAR_IN_SESSION:
            self._policy.record(request, decision)
            return ApprovalOutcome(
                decision=decision,
                reason=APPROVAL_REASON_OPERATOR_ALLOW_SIMILAR,
            )
        if decision == ApprovalDecision.ALLOW_ONCE:
            return ApprovalOutcome(
                decision=decision,
                reason=APPROVAL_REASON_OPERATOR_ALLOW_ONCE,
            )
        return ApprovalOutcome(
            decision=ApprovalDecision.DENY,
            reason=APPROVAL_REASON_OPERATOR_DENY,
        )

    def reset_session(self, session_id: str) -> None:
        self._policy.reset_session(session_id)
