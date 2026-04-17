"""Governance Surface approval-gate tests (Handoff 20)."""

from __future__ import annotations

from src.governance.approval import (
    APPROVAL_REASON_NO_APPROVAL_REQUIRED,
    APPROVAL_REASON_OPERATOR_ALLOW_ONCE,
    APPROVAL_REASON_OPERATOR_ALLOW_SIMILAR,
    APPROVAL_REASON_OPERATOR_DENY,
    APPROVAL_REASON_REUSED_FROM_SESSION,
    ApprovalDecision,
    ApprovalGate,
    ApprovalInteractor,
    ApprovalMemory,
    ApprovalOutcome,
    ApprovalPolicy,
    ApprovalRequest,
)
from src.governance.policies import RevealGroupSessionApprovalPolicy


def _request(
    *,
    session_id: str = "s1",
    tool_name: str = "write_file",
    reveal_group: str = "native_fs_mutate",
    side_effect_class: str = "write",
    requires_approval: bool = True,
    arguments: dict | None = None,
    summary: str = "write to disk",
) -> ApprovalRequest:
    return ApprovalRequest(
        session_id=session_id,
        tool_name=tool_name,
        reveal_group=reveal_group,
        side_effect_class=side_effect_class,
        requires_approval=requires_approval,
        arguments=arguments or {"path": "notes.md", "content": "hi"},
        summary=summary,
    )


class _ScriptedInteractor(ApprovalInteractor):
    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[ApprovalRequest] = []

    def prompt(self, request: ApprovalRequest) -> ApprovalDecision:
        self.calls.append(request)
        if not self._decisions:
            raise AssertionError("interactor prompted more than scripted")
        return self._decisions.pop(0)


# ---------------------------------------------------------------------------
# ApprovalMemory — session-scoped state
# ---------------------------------------------------------------------------


class TestApprovalMemory:
    def test_empty_session_has_no_approvals(self) -> None:
        mem = ApprovalMemory()
        assert mem.is_approved("s1", "native_fs_mutate") is False
        assert mem.snapshot("s1") == []

    def test_remember_scopes_to_session(self) -> None:
        mem = ApprovalMemory()
        mem.remember("s1", "native_fs_mutate")
        assert mem.is_approved("s1", "native_fs_mutate") is True
        assert mem.is_approved("s2", "native_fs_mutate") is False

    def test_remember_is_idempotent(self) -> None:
        mem = ApprovalMemory()
        mem.remember("s1", "native_fs_mutate")
        mem.remember("s1", "native_fs_mutate")
        assert mem.snapshot("s1") == ["native_fs_mutate"]

    def test_reset_clears_only_target_session(self) -> None:
        mem = ApprovalMemory()
        mem.remember("s1", "native_fs_mutate")
        mem.remember("s2", "mcp_git_mutate")
        removed = mem.reset("s1")
        assert removed == 1
        assert mem.is_approved("s1", "native_fs_mutate") is False
        assert mem.is_approved("s2", "mcp_git_mutate") is True

    def test_reset_on_unknown_session_is_noop(self) -> None:
        mem = ApprovalMemory()
        removed = mem.reset("nonexistent")
        assert removed == 0


# ---------------------------------------------------------------------------
# RevealGroupSessionApprovalPolicy — reveal-group scoped reuse
# ---------------------------------------------------------------------------


class TestRevealGroupSessionApprovalPolicy:
    def test_check_reuse_false_without_prior_approval(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        assert policy.check_reuse(_request()) is False

    def test_record_allow_similar_enables_reuse(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req = _request()
        policy.record(req, ApprovalDecision.ALLOW_SIMILAR_IN_SESSION)
        assert policy.check_reuse(req) is True

    def test_record_allow_once_does_not_persist(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req = _request()
        policy.record(req, ApprovalDecision.ALLOW_ONCE)
        assert policy.check_reuse(req) is False

    def test_record_deny_does_not_persist(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req = _request()
        policy.record(req, ApprovalDecision.DENY)
        assert policy.check_reuse(req) is False

    def test_reuse_is_scoped_to_reveal_group(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req_fs = _request(reveal_group="native_fs_mutate")
        req_git = _request(reveal_group="mcp_git_mutate")
        policy.record(req_fs, ApprovalDecision.ALLOW_SIMILAR_IN_SESSION)
        assert policy.check_reuse(req_fs) is True
        assert policy.check_reuse(req_git) is False

    def test_reuse_is_scoped_to_session(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req_s1 = _request(session_id="s1")
        req_s2 = _request(session_id="s2")
        policy.record(req_s1, ApprovalDecision.ALLOW_SIMILAR_IN_SESSION)
        assert policy.check_reuse(req_s1) is True
        assert policy.check_reuse(req_s2) is False

    def test_reset_clears_session_reuse(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req = _request()
        policy.record(req, ApprovalDecision.ALLOW_SIMILAR_IN_SESSION)
        policy.reset_session("s1")
        assert policy.check_reuse(req) is False

    def test_empty_reveal_group_never_reuses(self) -> None:
        policy = RevealGroupSessionApprovalPolicy()
        req = _request(reveal_group="")
        policy.record(req, ApprovalDecision.ALLOW_SIMILAR_IN_SESSION)
        assert policy.check_reuse(req) is False


# ---------------------------------------------------------------------------
# ApprovalGate — the decision seam
# ---------------------------------------------------------------------------


class TestApprovalGate:
    def test_requires_approval_false_bypasses_interactor(self) -> None:
        interactor = _ScriptedInteractor([])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        req = _request(requires_approval=False)
        outcome = gate.resolve(req)
        assert outcome.decision == ApprovalDecision.ALLOW_ONCE
        assert outcome.reason == APPROVAL_REASON_NO_APPROVAL_REQUIRED
        assert interactor.calls == []

    def test_first_call_prompts_interactor(self) -> None:
        interactor = _ScriptedInteractor([ApprovalDecision.ALLOW_ONCE])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        outcome = gate.resolve(_request())
        assert outcome.decision == ApprovalDecision.ALLOW_ONCE
        assert outcome.reason == APPROVAL_REASON_OPERATOR_ALLOW_ONCE
        assert len(interactor.calls) == 1

    def test_allow_once_does_not_persist(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_ONCE, ApprovalDecision.ALLOW_ONCE,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        gate.resolve(_request())
        gate.resolve(_request())
        assert len(interactor.calls) == 2

    def test_allow_similar_enables_reuse_in_session(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        first = gate.resolve(_request())
        second = gate.resolve(_request(tool_name="apply_exact_hunk"))
        assert first.decision == ApprovalDecision.ALLOW_SIMILAR_IN_SESSION
        assert first.reason == APPROVAL_REASON_OPERATOR_ALLOW_SIMILAR
        assert second.decision == ApprovalDecision.ALLOW_ONCE
        assert second.reason == APPROVAL_REASON_REUSED_FROM_SESSION
        assert len(interactor.calls) == 1

    def test_reuse_does_not_cross_reveal_groups(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
            ApprovalDecision.DENY,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        gate.resolve(_request(reveal_group="native_fs_mutate"))
        second = gate.resolve(_request(reveal_group="mcp_git_mutate"))
        assert second.decision == ApprovalDecision.DENY
        assert second.reason == APPROVAL_REASON_OPERATOR_DENY
        assert len(interactor.calls) == 2

    def test_reuse_does_not_cross_sessions(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
            ApprovalDecision.DENY,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        gate.resolve(_request(session_id="s1"))
        second = gate.resolve(_request(session_id="s2"))
        assert second.decision == ApprovalDecision.DENY
        assert len(interactor.calls) == 2

    def test_deny_does_not_persist(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.DENY, ApprovalDecision.ALLOW_ONCE,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        first = gate.resolve(_request())
        second = gate.resolve(_request())
        assert first.decision == ApprovalDecision.DENY
        assert second.decision == ApprovalDecision.ALLOW_ONCE
        assert len(interactor.calls) == 2

    def test_reset_session_clears_reuse(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
            ApprovalDecision.DENY,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        gate.resolve(_request())
        gate.reset_session("s1")
        second = gate.resolve(_request())
        assert second.decision == ApprovalDecision.DENY
        assert len(interactor.calls) == 2  # reset forced a re-prompt

    def test_reset_session_does_not_affect_other_sessions(self) -> None:
        interactor = _ScriptedInteractor([
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
            ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
        ])
        gate = ApprovalGate(
            policy=RevealGroupSessionApprovalPolicy(),
            interactor=interactor,
        )
        gate.resolve(_request(session_id="s1"))
        gate.resolve(_request(session_id="s2"))
        gate.reset_session("s1")
        # s2 reuse still works
        reused = gate.resolve(_request(session_id="s2"))
        assert reused.decision == ApprovalDecision.ALLOW_ONCE
        assert reused.reason == APPROVAL_REASON_REUSED_FROM_SESSION
        assert len(interactor.calls) == 2  # no third prompt


# ---------------------------------------------------------------------------
# Policy plug-and-play seam — confirm gate works with a foreign ApprovalPolicy
# ---------------------------------------------------------------------------


class _AlwaysReusePolicy(ApprovalPolicy):
    def __init__(self) -> None:
        self.recorded: list[tuple[ApprovalRequest, ApprovalDecision]] = []
        self.reset_calls: list[str] = []

    def check_reuse(self, request: ApprovalRequest) -> bool:
        return True

    def record(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        self.recorded.append((request, decision))

    def reset_session(self, session_id: str) -> None:
        self.reset_calls.append(session_id)


class TestPolicyPluggability:
    def test_gate_uses_foreign_policy_for_reuse(self) -> None:
        interactor = _ScriptedInteractor([])  # should never be called
        policy = _AlwaysReusePolicy()
        gate = ApprovalGate(policy=policy, interactor=interactor)
        outcome = gate.resolve(_request())
        assert outcome.decision == ApprovalDecision.ALLOW_ONCE
        assert outcome.reason == APPROVAL_REASON_REUSED_FROM_SESSION
        assert interactor.calls == []

    def test_gate_reset_forwards_to_policy(self) -> None:
        policy = _AlwaysReusePolicy()
        gate = ApprovalGate(
            policy=policy, interactor=_ScriptedInteractor([]),
        )
        gate.reset_session("s-abc")
        assert policy.reset_calls == ["s-abc"]
