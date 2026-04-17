"""Boundary integration tests for the approval gate (Handoff 20)."""

from __future__ import annotations

from pathlib import Path

from src.capability.boundary import CapabilityBoundary
from src.capability.models import ToolDefinition, ToolResult
from src.capability.registry import CapabilityRegistry
from src.capability.tools.base import Tool
from src.core.runtime.models import ToolRequest
from src.governance.approval import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalInteractor,
    ApprovalRequest,
)
from src.governance.policies import RevealGroupSessionApprovalPolicy


class _RecordingWriteTool(Tool):
    def __init__(self) -> None:
        self.execute_calls: int = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="test_write",
            description="A test write tool",
            parameters={"type": "object", "properties": {}},
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def reveal_group(self) -> str:
        return "native_fs_mutate"

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        return ()

    def execute(self, **kwargs) -> ToolResult:
        self.execute_calls += 1
        return ToolResult(ok=True, content="wrote")


class _SafeTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="test_safe", description="safe", parameters={"type": "object", "properties": {}},
        )

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        return ()

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(ok=True, content="safe")


class _ScriptedInteractor(ApprovalInteractor):
    def __init__(self, decisions: list[ApprovalDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[ApprovalRequest] = []

    def prompt(self, request: ApprovalRequest) -> ApprovalDecision:
        self.calls.append(request)
        return self._decisions.pop(0)


def _build(
    decisions: list[ApprovalDecision], tmp_path: Path,
) -> tuple[CapabilityBoundary, _RecordingWriteTool, _ScriptedInteractor, ApprovalGate]:
    registry = CapabilityRegistry()
    tool = _RecordingWriteTool()
    registry.register(tool)
    registry.register(_SafeTool())
    interactor = _ScriptedInteractor(decisions)
    gate = ApprovalGate(
        policy=RevealGroupSessionApprovalPolicy(), interactor=interactor,
    )
    boundary = CapabilityBoundary(registry, tmp_path, approval_gate=gate)
    return boundary, tool, interactor, gate


class TestBoundaryApprovalGateFiring:
    def test_approval_required_tool_without_session_id_is_denied(
        self, tmp_path: Path,
    ) -> None:
        boundary, tool, interactor, _ = _build(
            [ApprovalDecision.ALLOW_ONCE], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        result = boundary.execute(req)  # no session_id
        assert result.ok is False
        assert "no session" in result.content
        assert "denied" in result.governance_outcome
        assert tool.execute_calls == 0
        assert interactor.calls == []

    def test_allow_once_runs_execute(self, tmp_path: Path) -> None:
        boundary, tool, interactor, _ = _build(
            [ApprovalDecision.ALLOW_ONCE], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        result = boundary.execute(req, session_id="s1")
        assert result.ok is True
        assert result.governance_outcome == "allowed: operator_allow_once"
        assert tool.execute_calls == 1
        assert len(interactor.calls) == 1

    def test_empty_session_id_is_denied(self, tmp_path: Path) -> None:
        boundary, tool, interactor, _ = _build(
            [ApprovalDecision.ALLOW_ONCE], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        result = boundary.execute(req, session_id="")
        assert result.ok is False
        assert "no_session" in result.governance_outcome
        assert tool.execute_calls == 0
        assert interactor.calls == []

    def test_deny_blocks_execute_and_reports_denial(self, tmp_path: Path) -> None:
        boundary, tool, interactor, _ = _build(
            [ApprovalDecision.DENY], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        result = boundary.execute(req, session_id="s1")
        assert result.ok is False
        assert tool.execute_calls == 0
        assert "operator_deny" in result.governance_outcome

    def test_allow_similar_reuses_in_session(self, tmp_path: Path) -> None:
        boundary, tool, interactor, _ = _build(
            [ApprovalDecision.ALLOW_SIMILAR_IN_SESSION], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        r1 = boundary.execute(req, session_id="s1")
        r2 = boundary.execute(
            ToolRequest(tool_call_id="c2", tool_name="test_write", arguments={}),
            session_id="s1",
        )
        assert r1.ok is True
        assert r1.governance_outcome == "allowed: operator_allow_similar_in_session"
        assert r2.ok is True
        assert r2.governance_outcome == "allowed: reused_from_session"
        assert tool.execute_calls == 2
        assert len(interactor.calls) == 1  # only the first call prompted

    def test_reset_session_re_triggers_prompt(self, tmp_path: Path) -> None:
        boundary, tool, interactor, gate = _build(
            [
                ApprovalDecision.ALLOW_SIMILAR_IN_SESSION,
                ApprovalDecision.ALLOW_ONCE,
            ],
            tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        boundary.execute(req, session_id="s1")
        gate.reset_session("s1")
        boundary.execute(
            ToolRequest(tool_call_id="c2", tool_name="test_write", arguments={}),
            session_id="s1",
        )
        assert len(interactor.calls) == 2
        assert tool.execute_calls == 2

    def test_safe_tool_does_not_consult_gate(self, tmp_path: Path) -> None:
        boundary, _, interactor, _ = _build(
            [], tmp_path,
        )
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_safe", arguments={},
        )
        result = boundary.execute(req, session_id="s1")
        assert result.ok is True
        assert interactor.calls == []


class TestBoundaryBackwardCompat:
    def test_no_gate_attached_preserves_legacy_behavior(
        self, tmp_path: Path,
    ) -> None:
        registry = CapabilityRegistry()
        tool = _RecordingWriteTool()
        registry.register(tool)
        # No approval_gate
        boundary = CapabilityBoundary(registry, tmp_path)
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        # Even without a session_id and with requires_approval=True, the
        # boundary proceeds because legacy callers (tests, non-CLI use) have
        # no gate wired.
        result = boundary.execute(req)
        assert result.ok is True
        assert tool.execute_calls == 1

    def test_no_gate_marks_outcome_as_allowed_no_gate(
        self, tmp_path: Path,
    ) -> None:
        """Audit observability: ungated requires_approval calls are
        distinguishable from gate-approved calls in the transcript."""
        registry = CapabilityRegistry()
        tool = _RecordingWriteTool()
        registry.register(tool)
        boundary = CapabilityBoundary(registry, tmp_path)
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_write", arguments={},
        )
        result = boundary.execute(req)
        assert result.governance_outcome == "allowed: no_gate"

    def test_no_gate_safe_tool_keeps_plain_allowed(
        self, tmp_path: Path,
    ) -> None:
        """Safe tools (no requires_approval) keep the plain `allowed`
        outcome regardless of whether a gate is attached."""
        registry = CapabilityRegistry()
        registry.register(_SafeTool())
        boundary = CapabilityBoundary(registry, tmp_path)
        req = ToolRequest(
            tool_call_id="c1", tool_name="test_safe", arguments={},
        )
        result = boundary.execute(req)
        assert result.governance_outcome == "allowed"
