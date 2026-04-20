"""Capability-awareness fragment tests — Handoff 27.

Covers the collector, disclosure policy, fragment builder, StructuredContext
Assembler integration, and a regression shape replicating session
7b475ca42e77's task-triggered capability-incompleteness failure. ADR-0011
frames this block as awareness-shaping context: typed, governance-conditioned,
bounded, and inspectable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.capability.boundary import CapabilityBoundary
from src.capability.discovery import ListAvailableToolsTool
from src.capability.registry import CapabilityRegistry
from src.capability.tools import ReadFileTool, WriteFileTool
from src.governance.capability_awareness_disclosure import (
    DEFAULT_CAPABILITY_AWARENESS_DISCLOSURE_POLICY,
    BasicCapabilityAwarenessDisclosurePolicy,
    CapabilityAwarenessDisclosureDecision,
    CapabilityAwarenessDisclosurePolicy,
)
from src.governance.disclosure import (
    BatchRevealDisclosureStrategy,
    ExposureDecision,
    SingleRevealDisclosureStrategy,
)
from src.knowledge import (
    CAPABILITY_AWARENESS_FRAGMENT_NAME,
    CAPABILITY_AWARENESS_POSTURE_TEXT,
    CAPABILITY_AWARENESS_PRIORITY,
    CAPABILITY_AWARENESS_VISIBILITY_SCOPE,
    CapabilityAwarenessCollector,
    CapabilityAwarenessSnapshot,
    StructuredContextAssembler,
    build_capability_awareness_fragment,
)
from src.knowledge.assembly.structured import SESSION_SYSTEM_PROMPT_PRIORITY


# ---------------------------------------------------------------------------
# Fixtures: a small registry that emulates the production shape
# ---------------------------------------------------------------------------


def _registry_with_discovery_and_groups(tmp_path: Path) -> CapabilityRegistry:
    """Registry with native_fs_read (default-exposed), native_fs_mutate (hidden),
    and the discovery tool (always-on). Mirrors the `session_7b475ca42e77`
    minimal shape without needing to spin up MCP servers."""
    registry = CapabilityRegistry()
    registry.register(ReadFileTool(tmp_path))       # reveal_group: native_fs_read, default_exposed
    registry.register(WriteFileTool(tmp_path))      # reveal_group: native_fs_mutate, hidden
    registry.register(ListAvailableToolsTool(registry))  # reveal_group: discovery, default_exposed
    return registry


# ---------------------------------------------------------------------------
# CapabilityAwarenessCollector
# ---------------------------------------------------------------------------


class TestCapabilityAwarenessCollector:
    def test_collect_without_decision_falls_back_to_default_exposed(
        self, tmp_path: Path
    ) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        snapshot = CapabilityAwarenessCollector(registry).collect()
        # discovery + native_fs_read are visible by default
        assert "discovery" in snapshot.visible_reveal_groups
        assert "native_fs_read" in snapshot.visible_reveal_groups
        # native_fs_mutate is hidden
        assert "native_fs_mutate" in snapshot.hidden_reveal_groups
        assert snapshot.visible_tool_count >= 2

    def test_collect_with_decision_uses_active_groups(
        self, tmp_path: Path
    ) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        decision = ExposureDecision(
            exposed_tool_names={"native__read_file", "list_available_tools"},
            active_reveal_groups=["discovery", "native_fs_read"],
            strategy_name="single_reveal",
        )
        snapshot = CapabilityAwarenessCollector(registry).collect(
            exposure_decision=decision
        )
        assert snapshot.visible_tool_count == 2
        assert snapshot.visible_reveal_groups == ("discovery", "native_fs_read")
        assert "native_fs_mutate" in snapshot.hidden_reveal_groups

    def test_revealed_group_becomes_visible(self, tmp_path: Path) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        decision = ExposureDecision(
            exposed_tool_names={
                "native__read_file", "native__write_file", "list_available_tools"
            },
            active_reveal_groups=[
                "discovery", "native_fs_read", "native_fs_mutate"
            ],
            strategy_name="single_reveal",
        )
        snapshot = CapabilityAwarenessCollector(registry).collect(
            exposure_decision=decision
        )
        assert "native_fs_mutate" not in snapshot.hidden_reveal_groups
        assert "native_fs_mutate" in snapshot.visible_reveal_groups

    def test_hidden_group_descriptions_present(self, tmp_path: Path) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        snapshot = CapabilityAwarenessCollector(registry).collect()
        for g in snapshot.hidden_reveal_groups:
            assert g in snapshot.hidden_group_descriptions
            assert snapshot.hidden_group_descriptions[g]  # non-empty

    def test_empty_registry_produces_empty_snapshot(self) -> None:
        registry = CapabilityRegistry()
        snapshot = CapabilityAwarenessCollector(registry).collect()
        assert snapshot.visible_reveal_groups == ()
        assert snapshot.hidden_reveal_groups == ()
        assert snapshot.visible_tool_count == 0


# ---------------------------------------------------------------------------
# BasicCapabilityAwarenessDisclosurePolicy
# ---------------------------------------------------------------------------


class TestBasicDisclosurePolicy:
    def test_policy_name_is_stable(self) -> None:
        assert (
            BasicCapabilityAwarenessDisclosurePolicy().policy_name
            == "basic_capability_awareness"
        )

    def test_discloses_when_hidden_groups_exist(self) -> None:
        snapshot = CapabilityAwarenessSnapshot(
            visible_tool_count=2,
            visible_reveal_groups=("discovery",),
            hidden_reveal_groups=("mcp_fs_mutate",),
        )
        decision = BasicCapabilityAwarenessDisclosurePolicy().decide(snapshot)
        assert decision.disclose is True

    def test_suppresses_when_no_hidden_groups(self) -> None:
        snapshot = CapabilityAwarenessSnapshot(
            visible_tool_count=5,
            visible_reveal_groups=("discovery", "native_fs_read"),
            hidden_reveal_groups=(),
        )
        decision = BasicCapabilityAwarenessDisclosurePolicy().decide(snapshot)
        assert decision.disclose is False

    def test_module_default_is_basic_policy(self) -> None:
        assert isinstance(
            DEFAULT_CAPABILITY_AWARENESS_DISCLOSURE_POLICY,
            BasicCapabilityAwarenessDisclosurePolicy,
        )


# ---------------------------------------------------------------------------
# build_capability_awareness_fragment
# ---------------------------------------------------------------------------


class TestBuildFragment:
    def _snap(self) -> CapabilityAwarenessSnapshot:
        return CapabilityAwarenessSnapshot(
            visible_tool_count=2,
            visible_reveal_groups=("discovery", "native_fs_read"),
            hidden_reveal_groups=("mcp_fs_mutate", "mcp_git_read"),
            hidden_group_descriptions={
                "mcp_fs_mutate": "MCP filesystem mutations (write / replace).",
                "mcp_git_read": "MCP git read tools (status / diff / log).",
            },
        )

    def test_fragment_identity_matches_constants(self) -> None:
        frag = build_capability_awareness_fragment(
            self._snap(), policy_name="test"
        )
        assert frag.fragment_name == CAPABILITY_AWARENESS_FRAGMENT_NAME
        assert frag.visibility_scope == CAPABILITY_AWARENESS_VISIBILITY_SCOPE
        assert frag.priority == CAPABILITY_AWARENESS_PRIORITY

    def test_priority_below_system_prompt(self) -> None:
        assert CAPABILITY_AWARENESS_PRIORITY < SESSION_SYSTEM_PROMPT_PRIORITY

    def test_content_wraps_in_typed_tags(self) -> None:
        frag = build_capability_awareness_fragment(
            self._snap(), policy_name="test"
        )
        assert frag.content.startswith("<capability-awareness>\n")
        assert frag.content.endswith("\n</capability-awareness>")

    def test_content_lists_hidden_groups_with_descriptions(self) -> None:
        frag = build_capability_awareness_fragment(
            self._snap(), policy_name="test"
        )
        assert "mcp_fs_mutate" in frag.content
        assert "MCP filesystem mutations" in frag.content
        assert "mcp_git_read" in frag.content

    def test_content_includes_posture_text(self) -> None:
        frag = build_capability_awareness_fragment(
            self._snap(), policy_name="test"
        )
        assert CAPABILITY_AWARENESS_POSTURE_TEXT in frag.content

    def test_metadata_carries_visible_and_hidden_lists(self) -> None:
        frag = build_capability_awareness_fragment(
            self._snap(), policy_name="basic_capability_awareness"
        )
        assert frag.metadata["origin"] == "capability_awareness"
        assert frag.metadata["policy_name"] == "basic_capability_awareness"
        assert frag.metadata["visible_tool_count"] == 2
        assert frag.metadata["visible_reveal_groups"] == [
            "discovery", "native_fs_read"
        ]
        assert frag.metadata["hidden_reveal_groups"] == [
            "mcp_fs_mutate", "mcp_git_read"
        ]


# ---------------------------------------------------------------------------
# StructuredContextAssembler integration
# ---------------------------------------------------------------------------


class _StubAwarenessPolicy(CapabilityAwarenessDisclosurePolicy):
    def __init__(self, *, disclose: bool) -> None:
        self._disclose = disclose

    @property
    def policy_name(self) -> str:
        return "stub_awareness"

    def decide(
        self, snapshot: CapabilityAwarenessSnapshot
    ) -> CapabilityAwarenessDisclosureDecision:
        return CapabilityAwarenessDisclosureDecision(
            disclose=self._disclose, policy_name=self.policy_name
        )


class TestAssemblerIntegration:
    def test_no_collector_emits_no_awareness_fragment(self, tmp_path: Path) -> None:
        assembler = StructuredContextAssembler()
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert CAPABILITY_AWARENESS_FRAGMENT_NAME not in names

    def test_collector_without_policy_emits_no_fragment(self, tmp_path: Path) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert CAPABILITY_AWARENESS_FRAGMENT_NAME not in names

    def test_basic_policy_emits_fragment_when_hidden_groups_exist(
        self, tmp_path: Path
    ) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        fragments = {f.fragment_name: f for f in ac.instruction_fragments}
        assert CAPABILITY_AWARENESS_FRAGMENT_NAME in fragments
        assert (
            fragments[CAPABILITY_AWARENESS_FRAGMENT_NAME].priority
            == CAPABILITY_AWARENESS_PRIORITY
        )

    def test_disclose_false_suppresses_fragment(self, tmp_path: Path) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=_StubAwarenessPolicy(disclose=False),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert CAPABILITY_AWARENESS_FRAGMENT_NAME not in names

    def test_system_prompt_renders_before_awareness_in_turn_request(
        self, tmp_path: Path
    ) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt="You are helpful.")
        assert req.system is not None
        sys_idx = req.system.find("You are helpful.")
        cap_idx = req.system.find("<capability-awareness>")
        assert sys_idx >= 0 and cap_idx >= 0
        assert sys_idx < cap_idx

    def test_exposure_decision_changes_hidden_set(self, tmp_path: Path) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        # First turn: native_fs_mutate is hidden → fragment lists it
        ac1 = assembler.assemble_structured([], system_prompt=None)
        first_frag = next(
            f for f in ac1.instruction_fragments
            if f.fragment_name == CAPABILITY_AWARENESS_FRAGMENT_NAME
        )
        assert "native_fs_mutate" in first_frag.content

        # After reveal: native_fs_mutate moves to visible → fragment should
        # no longer list it as hidden (and may be suppressed if nothing is hidden)
        revealed_decision = ExposureDecision(
            exposed_tool_names={
                "native__read_file", "native__write_file", "list_available_tools"
            },
            active_reveal_groups=[
                "discovery", "native_fs_read", "native_fs_mutate"
            ],
            strategy_name="single_reveal",
        )
        ac2 = assembler.assemble_structured(
            [], system_prompt=None, exposure_decision=revealed_decision
        )
        frags_after = [
            f for f in ac2.instruction_fragments
            if f.fragment_name == CAPABILITY_AWARENESS_FRAGMENT_NAME
        ]
        # All groups visible → BasicPolicy suppresses; list should be empty
        assert frags_after == []


# ---------------------------------------------------------------------------
# Regression replica: session_7b475ca42e77 shape
# ---------------------------------------------------------------------------


class TestSession7b475ca42e77RegressionShape:
    """The original failure: minimal visible set {discovery, native_fs_read}
    with filesystem mutation hidden. After Handoff 27, the assembled context
    must include a capability-awareness fragment telling the provider to
    prefer discovery over shell-command fallback.
    """

    def test_assembled_system_carries_awareness_and_posture(
        self, tmp_path: Path
    ) -> None:
        registry = _registry_with_discovery_and_groups(tmp_path)
        # First-turn exposure decision under single-reveal: only defaults are active
        decision = SingleRevealDisclosureStrategy().compute(registry, [])
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt=None)
        assert req.system is not None
        # Structural signals
        assert "<capability-awareness>" in req.system
        assert "hidden_reveal_groups" in req.system
        # The hidden mutate group must be surfaced
        assert "native_fs_mutate" in req.system
        # The posture targets the specific failure (shell-command fallback)
        assert "shell-command" in req.system
        assert "list_available_tools" in req.system
        # Negative check: no tool schemas dumped (bounded)
        assert "native__write_file" not in req.system

    def test_awareness_suppressed_when_all_safe_revealed(
        self, tmp_path: Path
    ) -> None:
        """Under batch-reveal + reveal_all_safe, every safe group becomes
        visible. When nothing mutation-bearing remains hidden for a simple
        registry like this, the fragment should naturally suppress rather
        than emit redundant empty-state text."""
        registry = _registry_with_discovery_and_groups(tmp_path)
        # Simulate an exposure decision where every group is active
        decision = ExposureDecision(
            exposed_tool_names={
                "native__read_file", "native__write_file", "list_available_tools"
            },
            active_reveal_groups=[
                "discovery", "native_fs_read", "native_fs_mutate"
            ],
            strategy_name="batch_reveal",
        )
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        req = assembler.assemble_structured(
            [], system_prompt=None, exposure_decision=decision
        ).to_turn_request()
        assert req.system is None or "<capability-awareness>" not in (req.system or "")


# ---------------------------------------------------------------------------
# Debug envelope projection: awareness fragment is inspectable
# ---------------------------------------------------------------------------


class TestDebugEnvelopeProjection:
    def test_awareness_fragment_appears_in_envelope_preview(
        self, tmp_path: Path
    ) -> None:
        from src.knowledge.assembly import build_envelope

        registry = _registry_with_discovery_and_groups(tmp_path)
        assembler = StructuredContextAssembler(
            capability_awareness_collector=CapabilityAwarenessCollector(registry),
            capability_awareness_disclosure_policy=(
                BasicCapabilityAwarenessDisclosurePolicy()
            ),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        req = ac.to_turn_request()
        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=0,
            request=req,
            assembled_context=ac,
            exposure_decision=None,
        )
        names = [fp.fragment_name for fp in env.instruction_fragments]
        assert CAPABILITY_AWARENESS_FRAGMENT_NAME in names
