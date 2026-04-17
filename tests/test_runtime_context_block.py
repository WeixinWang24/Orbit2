"""Runtime-context block tests — Handoff 26.

Covers the collector, the disclosure policy, and their integration into
StructuredContextAssembler. The block is awareness-shaping context under
ADR-0011: typed, governance-conditioned, and inspectable.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.runtime import REPO_ROOT, RuntimeRoot
from src.governance.runtime_context_disclosure import (
    DEFAULT_RUNTIME_CONTEXT_DISCLOSURE_POLICY,
    BasicSelfLocationDisclosurePolicy,
    RuntimeContextDisclosureDecision,
    RuntimeContextDisclosurePolicy,
)
from src.knowledge import (
    RUNTIME_CONTEXT_FRAGMENT_NAME,
    RUNTIME_CONTEXT_PRIORITY,
    RUNTIME_CONTEXT_VISIBILITY_SCOPE,
    RuntimeContextCollector,
    RuntimeContextSnapshot,
    StructuredContextAssembler,
    build_runtime_context_fragment,
)
from src.knowledge.assembly.structured import SESSION_SYSTEM_PROMPT_PRIORITY


# ---------------------------------------------------------------------------
# RuntimeContextCollector
# ---------------------------------------------------------------------------


class TestRuntimeContextCollector:
    def test_collects_runtime_root_path_and_source(self, tmp_path: Path) -> None:
        rr = RuntimeRoot(tmp_path, "cli-flag")
        store = tmp_path / ".runtime" / "sessions.db"
        snap = RuntimeContextCollector(rr, store).collect()
        assert snap.runtime_root_path == tmp_path
        assert snap.runtime_root_source == "cli-flag"
        assert snap.store_path == store

    def test_cwd_is_resolved_from_live_process(self, tmp_path: Path) -> None:
        rr = RuntimeRoot(tmp_path, "repo-root")
        collector = RuntimeContextCollector(rr, tmp_path / "db")
        with patch("os.getcwd", return_value=str(tmp_path)):
            snap = collector.collect()
        assert snap.cwd == tmp_path.resolve()

    def test_repo_root_sourced_from_config(self, tmp_path: Path) -> None:
        rr = RuntimeRoot(tmp_path, "cli-flag")
        snap = RuntimeContextCollector(rr, tmp_path / "x").collect()
        assert snap.repo_root == REPO_ROOT

    def test_store_path_defaults_to_runtime_root_layout(self, tmp_path: Path) -> None:
        rr = RuntimeRoot(tmp_path, "cli-flag")
        snap = RuntimeContextCollector(rr).collect()
        assert snap.store_path == tmp_path / ".runtime" / "sessions.db"

    def test_collect_method_does_not_create_directories(self, tmp_path: Path) -> None:
        """The collector's `collect()` call is the part that must be pure.

        NOTE: `default_db_path(runtime_root)` (called in CLI/inspector
        startup before constructing the collector) does mkdir the
        `.runtime/` directory. That side-effect lives in `default_db_path`,
        not in the collector. This test narrowly confirms the collector
        itself never mkdirs — which is the property the assembler relies on
        every turn.
        """
        rr = RuntimeRoot(tmp_path, "cli-flag")
        RuntimeContextCollector(rr).collect()
        assert not (tmp_path / ".runtime").exists()

    def test_cwd_and_runtime_root_can_diverge(self, tmp_path: Path) -> None:
        root = tmp_path / "runtime_root"
        root.mkdir()
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        rr = RuntimeRoot(root, "cli-flag")
        collector = RuntimeContextCollector(rr)
        with patch("os.getcwd", return_value=str(other_cwd)):
            snap = collector.collect()
        assert snap.cwd == other_cwd.resolve()
        assert snap.runtime_root_path == root
        assert snap.cwd != snap.runtime_root_path

    def test_env_runtime_root_source_is_preserved_in_snapshot(
        self, tmp_path: Path
    ) -> None:
        rr = RuntimeRoot(tmp_path, "env:ORBIT2_RUNTIME_ROOT")
        snap = RuntimeContextCollector(rr).collect()
        assert snap.runtime_root_source == "env:ORBIT2_RUNTIME_ROOT"


# ---------------------------------------------------------------------------
# RuntimeContextSnapshot.to_visible_dict
# ---------------------------------------------------------------------------


class TestSnapshotToVisibleDict:
    def _snap(self, tmp_path: Path) -> RuntimeContextSnapshot:
        return RuntimeContextSnapshot(
            cwd=tmp_path,
            runtime_root_path=tmp_path,
            runtime_root_source="cli-flag",
            repo_root=tmp_path,
            store_path=tmp_path / "db",
        )

    def test_subsets_to_visible_fields_only(self, tmp_path: Path) -> None:
        snap = self._snap(tmp_path)
        visible = snap.to_visible_dict(frozenset({"cwd", "runtime_root_source"}))
        assert set(visible) == {"cwd", "runtime_root_source"}

    def test_unknown_field_names_are_ignored(self, tmp_path: Path) -> None:
        snap = self._snap(tmp_path)
        visible = snap.to_visible_dict(frozenset({"cwd", "bogus_field"}))
        assert set(visible) == {"cwd"}

    def test_empty_visible_fields_returns_empty_dict(self, tmp_path: Path) -> None:
        snap = self._snap(tmp_path)
        assert snap.to_visible_dict(frozenset()) == {}


# ---------------------------------------------------------------------------
# BasicSelfLocationDisclosurePolicy
# ---------------------------------------------------------------------------


class TestBasicSelfLocationPolicy:
    def _snap(self) -> RuntimeContextSnapshot:
        return RuntimeContextSnapshot(
            cwd=Path("/x"),
            runtime_root_path=Path("/x"),
            runtime_root_source="repo-root",
            repo_root=Path("/x"),
            store_path=Path("/x/db"),
        )

    def test_policy_name_is_stable(self) -> None:
        assert BasicSelfLocationDisclosurePolicy().policy_name == "basic_self_location"

    def test_policy_discloses_self_location_fields(self) -> None:
        decision = BasicSelfLocationDisclosurePolicy().decide(self._snap())
        assert decision.disclose is True
        assert decision.visible_fields == frozenset({
            "cwd",
            "runtime_root_path",
            "runtime_root_source",
            "repo_root",
            "store_path",
        })
        assert decision.policy_name == "basic_self_location"

    def test_module_default_policy_is_basic_self_location(self) -> None:
        assert isinstance(
            DEFAULT_RUNTIME_CONTEXT_DISCLOSURE_POLICY,
            BasicSelfLocationDisclosurePolicy,
        )


# ---------------------------------------------------------------------------
# build_runtime_context_fragment
# ---------------------------------------------------------------------------


class TestBuildFragment:
    def _snap(self, tmp_path: Path) -> RuntimeContextSnapshot:
        return RuntimeContextSnapshot(
            cwd=tmp_path,
            runtime_root_path=tmp_path,
            runtime_root_source="cli-flag",
            repo_root=tmp_path,
            store_path=tmp_path / "db",
        )

    def test_fragment_identity_matches_constants(self, tmp_path: Path) -> None:
        fragment = build_runtime_context_fragment(
            self._snap(tmp_path),
            visible_fields=frozenset({"cwd"}),
            policy_name="test",
        )
        assert fragment.fragment_name == RUNTIME_CONTEXT_FRAGMENT_NAME
        assert fragment.visibility_scope == RUNTIME_CONTEXT_VISIBILITY_SCOPE
        assert fragment.priority == RUNTIME_CONTEXT_PRIORITY

    def test_fragment_priority_is_below_system_prompt(self) -> None:
        assert RUNTIME_CONTEXT_PRIORITY < SESSION_SYSTEM_PROMPT_PRIORITY

    def test_fragment_content_wraps_in_typed_tags(self, tmp_path: Path) -> None:
        fragment = build_runtime_context_fragment(
            self._snap(tmp_path),
            visible_fields=frozenset({"cwd"}),
            policy_name="test",
        )
        assert fragment.content.startswith("<runtime-context>\n")
        assert fragment.content.endswith("\n</runtime-context>")

    def test_fragment_content_includes_requested_fields_only(
        self, tmp_path: Path
    ) -> None:
        fragment = build_runtime_context_fragment(
            self._snap(tmp_path),
            visible_fields=frozenset({"cwd", "runtime_root_source"}),
            policy_name="test",
        )
        assert "cwd:" in fragment.content
        assert "runtime_root_source:" in fragment.content
        assert "store_path:" not in fragment.content
        assert "repo_root:" not in fragment.content

    def test_fragment_metadata_carries_policy_and_visible_fields(
        self, tmp_path: Path
    ) -> None:
        fragment = build_runtime_context_fragment(
            self._snap(tmp_path),
            visible_fields=frozenset({"cwd", "runtime_root_source"}),
            policy_name="basic_self_location",
        )
        assert fragment.metadata["origin"] == "runtime_context"
        assert fragment.metadata["policy_name"] == "basic_self_location"
        assert fragment.metadata["visible_fields"] == ["cwd", "runtime_root_source"]


# ---------------------------------------------------------------------------
# StructuredContextAssembler integration
# ---------------------------------------------------------------------------


class _StubPolicy(RuntimeContextDisclosurePolicy):
    def __init__(self, *, disclose: bool, fields: frozenset[str] | None = None) -> None:
        self._disclose = disclose
        self._fields = fields if fields is not None else frozenset({"cwd"})

    @property
    def policy_name(self) -> str:
        return "stub"

    def decide(self, snapshot: RuntimeContextSnapshot) -> RuntimeContextDisclosureDecision:
        return RuntimeContextDisclosureDecision(
            disclose=self._disclose,
            visible_fields=self._fields,
            policy_name=self.policy_name,
        )


def _fixture_collector(tmp_path: Path) -> RuntimeContextCollector:
    return RuntimeContextCollector(
        RuntimeRoot(tmp_path, "cli-flag"), tmp_path / "db"
    )


class TestAssemblerIntegration:
    def test_assembler_without_collector_emits_no_runtime_context(self) -> None:
        assembler = StructuredContextAssembler()
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert RUNTIME_CONTEXT_FRAGMENT_NAME not in names

    def test_assembler_with_collector_but_no_policy_emits_no_runtime_context(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert RUNTIME_CONTEXT_FRAGMENT_NAME not in names

    def test_assembler_with_collector_and_policy_emits_fragment(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        fragments = {f.fragment_name: f for f in ac.instruction_fragments}
        assert RUNTIME_CONTEXT_FRAGMENT_NAME in fragments
        assert fragments[RUNTIME_CONTEXT_FRAGMENT_NAME].priority == RUNTIME_CONTEXT_PRIORITY

    def test_policy_returning_false_suppresses_fragment(self, tmp_path: Path) -> None:
        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
            runtime_context_disclosure_policy=_StubPolicy(disclose=False),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert RUNTIME_CONTEXT_FRAGMENT_NAME not in names

    def test_empty_visible_fields_suppresses_fragment(self, tmp_path: Path) -> None:
        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
            runtime_context_disclosure_policy=_StubPolicy(
                disclose=True, fields=frozenset()
            ),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert RUNTIME_CONTEXT_FRAGMENT_NAME not in names

    def test_system_prompt_renders_before_runtime_context_in_turn_request(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
        )
        req = assembler.assemble([], system_prompt="You are helpful.")
        assert req.system is not None
        sys_index = req.system.find("You are helpful.")
        rc_index = req.system.find("<runtime-context>")
        assert sys_index >= 0
        assert rc_index >= 0
        assert sys_index < rc_index

    def test_runtime_context_block_exposes_cwd_and_root(self, tmp_path: Path) -> None:
        with patch("os.getcwd", return_value=str(tmp_path)):
            assembler = StructuredContextAssembler(
                runtime_context_collector=_fixture_collector(tmp_path),
                runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
            )
            req = assembler.assemble([], system_prompt=None)
        assert req.system is not None
        assert f"cwd: {tmp_path.resolve()}" in req.system
        assert "runtime_root_path:" in req.system
        assert "runtime_root_source: cli-flag" in req.system

    def test_fragment_reports_cwd_independently_of_runtime_root(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "runtime_root"
        root.mkdir()
        other_cwd = tmp_path / "live_cwd"
        other_cwd.mkdir()
        assembler = StructuredContextAssembler(
            runtime_context_collector=RuntimeContextCollector(
                RuntimeRoot(root, "env:ORBIT2_RUNTIME_ROOT"), root / "db"
            ),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
        )
        with patch("os.getcwd", return_value=str(other_cwd)):
            req = assembler.assemble([], system_prompt=None)
        assert req.system is not None
        assert f"cwd: {other_cwd.resolve()}" in req.system
        assert f"runtime_root_path: {root}" in req.system
        assert "runtime_root_source: env:ORBIT2_RUNTIME_ROOT" in req.system

    def test_runtime_context_block_does_not_leak_env_or_secrets(
        self, tmp_path: Path
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-should-not-appear",
                "ORBIT2_SECRET_TEST_VALUE": "secret-should-not-appear",
            },
            clear=False,
        ):
            assembler = StructuredContextAssembler(
                runtime_context_collector=_fixture_collector(tmp_path),
                runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
            )
            req = assembler.assemble([], system_prompt=None)
        assert req.system is not None
        assert "sk-should-not-appear" not in req.system
        assert "secret-should-not-appear" not in req.system
        assert "OPENAI_API_KEY" not in req.system


# ---------------------------------------------------------------------------
# Debug envelope integration: runtime-context fragment is inspectable
# ---------------------------------------------------------------------------


class TestDebugEnvelopeProjection:
    def test_runtime_context_fragment_appears_in_envelope_preview(
        self, tmp_path: Path
    ) -> None:
        from src.knowledge.assembly import build_envelope

        assembler = StructuredContextAssembler(
            runtime_context_collector=_fixture_collector(tmp_path),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
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
        assert RUNTIME_CONTEXT_FRAGMENT_NAME in names
