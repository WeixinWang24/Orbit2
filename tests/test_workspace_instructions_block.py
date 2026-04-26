"""Workspace-instructions block tests — Handoff 29.

Covers the collector, the disclosure policy, assembler integration, the
debug envelope projection, and the transcript-truth separation invariant.

The block loads a repo-local `orbit.md` through the converged runtime-root
seam without consulting process cwd. Absence of the file is a first-class
state; canonical transcript persistence is unaffected by the fragment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.governance.workspace_instructions_disclosure import (
    DEFAULT_WORKSPACE_INSTRUCTIONS_DISCLOSURE_POLICY,
    BasicWorkspaceInstructionsDisclosurePolicy,
    WorkspaceInstructionsDisclosureDecision,
    WorkspaceInstructionsDisclosurePolicy,
)
from src.knowledge import (
    MAX_WORKSPACE_INSTRUCTIONS_BYTES,
    WORKSPACE_INSTRUCTIONS_FILENAME,
    WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME,
    WORKSPACE_INSTRUCTIONS_PRIORITY,
    WORKSPACE_INSTRUCTIONS_VISIBILITY_SCOPE,
    StructuredContextAssembler,
    WorkspaceInstructionsCollector,
    WorkspaceInstructionsSnapshot,
    build_workspace_instructions_fragment,
)
from src.knowledge.assembly.structured import SESSION_SYSTEM_PROMPT_PRIORITY


# ---------------------------------------------------------------------------
# WorkspaceInstructionsCollector
# ---------------------------------------------------------------------------


class TestWorkspaceInstructionsCollector:
    def test_reads_orbit_md_content_when_present(self, tmp_path: Path) -> None:
        (tmp_path / "orbit.md").write_text("# Workspace rules\nBe careful.\n")
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.exists is True
        assert snap.source_path == tmp_path / "orbit.md"
        assert "Workspace rules" in snap.content
        assert "Be careful." in snap.content
        assert snap.truncated is False

    def test_reports_absence_when_file_missing(self, tmp_path: Path) -> None:
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.exists is False
        assert snap.content == ""
        assert snap.source_path == tmp_path / "orbit.md"
        assert snap.truncated is False

    def test_directory_at_orbit_md_path_is_treated_as_absent(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "orbit.md").mkdir()
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.exists is False
        assert snap.content == ""

    def test_source_path_uses_workspace_root(self, tmp_path: Path) -> None:
        collector = WorkspaceInstructionsCollector(tmp_path)
        assert collector.workspace_root == tmp_path
        assert collector.source_path == tmp_path / WORKSPACE_INSTRUCTIONS_FILENAME

    def test_default_filename_is_orbit_md(self) -> None:
        assert WORKSPACE_INSTRUCTIONS_FILENAME == "orbit.md"

    def test_truncates_oversize_file(self, tmp_path: Path) -> None:
        (tmp_path / "orbit.md").write_bytes(b"A" * (MAX_WORKSPACE_INSTRUCTIONS_BYTES + 512))
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.truncated is True
        assert len(snap.content.encode("utf-8")) == MAX_WORKSPACE_INSTRUCTIONS_BYTES

    def test_truncation_limit_is_configurable(self, tmp_path: Path) -> None:
        (tmp_path / "orbit.md").write_text("abcdefgh")
        snap = WorkspaceInstructionsCollector(tmp_path, max_bytes=4).collect()
        assert snap.truncated is True
        assert snap.content == "abcd"

    def test_non_utf8_bytes_are_replaced_not_raised(self, tmp_path: Path) -> None:
        (tmp_path / "orbit.md").write_bytes(b"hello \xff world")
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.exists is True
        assert "hello" in snap.content
        assert "world" in snap.content

    def test_alternative_filename_override(self, tmp_path: Path) -> None:
        (tmp_path / "rules.md").write_text("custom filename content")
        snap = WorkspaceInstructionsCollector(tmp_path, filename="rules.md").collect()
        assert snap.exists is True
        assert "custom filename content" in snap.content

    def test_truncation_at_multibyte_boundary_does_not_inject_replacement_char(
        self, tmp_path: Path
    ) -> None:
        # Three-byte UTF-8 CJK character at the boundary: if we cut bytes
        # mid-sequence, a naive decode would splatter U+FFFD. The
        # incremental decoder should buffer the partial sequence instead.
        prefix_bytes = 4  # "abcd"
        cjk = "字"  # 0xE5 0xAD 0x97 in UTF-8
        body = ("abcd" + cjk * 100).encode("utf-8")
        (tmp_path / "orbit.md").write_bytes(body)
        # max_bytes falls mid-CJK: prefix (4) + first CJK start (1) + mid (1)
        snap = WorkspaceInstructionsCollector(
            tmp_path, max_bytes=prefix_bytes + 2
        ).collect()
        assert snap.truncated is True
        # The clean boundary is "abcd" — the partial CJK must not surface
        # as a replacement char; it must be buffered away.
        assert snap.content == "abcd"
        assert "\ufffd" not in snap.content

    def test_is_file_oserror_returns_absent_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate an OSError from Path.is_file() — e.g. a permission
        # problem on the containing directory. The collector must not
        # let the exception escape into the assembler.
        def _raise(self: Path) -> bool:
            raise OSError("simulated permission error")

        monkeypatch.setattr(Path, "is_file", _raise)
        snap = WorkspaceInstructionsCollector(tmp_path).collect()
        assert snap.exists is False
        assert snap.content == ""
        assert snap.truncated is False

    def test_collector_ignores_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collector must never consult process cwd — workspace_root is
        the only authoritative lookup anchor."""
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        (other_cwd / "orbit.md").write_text("from elsewhere (should NOT appear)")

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "orbit.md").write_text("from workspace root (authoritative)")

        monkeypatch.chdir(other_cwd)
        snap = WorkspaceInstructionsCollector(workspace).collect()
        assert snap.exists is True
        assert "authoritative" in snap.content
        assert "should NOT appear" not in snap.content


# ---------------------------------------------------------------------------
# BasicWorkspaceInstructionsDisclosurePolicy
# ---------------------------------------------------------------------------


class TestBasicDisclosurePolicy:
    def _snap(
        self, *, exists: bool, content: str, truncated: bool = False
    ) -> WorkspaceInstructionsSnapshot:
        return WorkspaceInstructionsSnapshot(
            source_path=Path("/x/orbit.md"),
            exists=exists,
            content=content,
            truncated=truncated,
        )

    def test_policy_name_is_stable(self) -> None:
        assert (
            BasicWorkspaceInstructionsDisclosurePolicy().policy_name
            == "basic_workspace_instructions"
        )

    def test_discloses_when_file_exists_with_content(self) -> None:
        decision = BasicWorkspaceInstructionsDisclosurePolicy().decide(
            self._snap(exists=True, content="hello")
        )
        assert decision.disclose is True
        assert decision.policy_name == "basic_workspace_instructions"

    def test_suppresses_when_file_absent(self) -> None:
        decision = BasicWorkspaceInstructionsDisclosurePolicy().decide(
            self._snap(exists=False, content="")
        )
        assert decision.disclose is False

    def test_suppresses_when_content_whitespace_only(self) -> None:
        decision = BasicWorkspaceInstructionsDisclosurePolicy().decide(
            self._snap(exists=True, content="   \n   \t\n")
        )
        assert decision.disclose is False

    def test_suppresses_when_content_empty(self) -> None:
        decision = BasicWorkspaceInstructionsDisclosurePolicy().decide(
            self._snap(exists=True, content="")
        )
        assert decision.disclose is False

    def test_module_default_policy_is_basic(self) -> None:
        assert isinstance(
            DEFAULT_WORKSPACE_INSTRUCTIONS_DISCLOSURE_POLICY,
            BasicWorkspaceInstructionsDisclosurePolicy,
        )


# ---------------------------------------------------------------------------
# build_workspace_instructions_fragment
# ---------------------------------------------------------------------------


class TestBuildFragment:
    def _snap(self, content: str) -> WorkspaceInstructionsSnapshot:
        return WorkspaceInstructionsSnapshot(
            source_path=Path("/w/orbit.md"),
            exists=True,
            content=content,
            truncated=False,
        )

    def test_fragment_identity_matches_constants(self) -> None:
        fragment = build_workspace_instructions_fragment(
            self._snap("body"), policy_name="test"
        )
        assert fragment.fragment_name == WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME
        assert fragment.visibility_scope == WORKSPACE_INSTRUCTIONS_VISIBILITY_SCOPE
        assert fragment.priority == WORKSPACE_INSTRUCTIONS_PRIORITY

    def test_fragment_priority_is_below_session_system_prompt(self) -> None:
        assert WORKSPACE_INSTRUCTIONS_PRIORITY < SESSION_SYSTEM_PROMPT_PRIORITY

    def test_fragment_content_wraps_in_typed_tags(self) -> None:
        fragment = build_workspace_instructions_fragment(
            self._snap("# rules\nbe careful"), policy_name="test"
        )
        assert fragment.content.startswith("<workspace-instructions>\n")
        assert fragment.content.endswith("\n</workspace-instructions>")
        assert "# rules" in fragment.content
        assert "be careful" in fragment.content

    def test_fragment_metadata_carries_policy_and_source(self) -> None:
        fragment = build_workspace_instructions_fragment(
            self._snap("abc"), policy_name="basic_workspace_instructions"
        )
        assert fragment.metadata["origin"] == "workspace_instructions"
        assert fragment.metadata["policy_name"] == "basic_workspace_instructions"
        assert fragment.metadata["source_path"] == "/w/orbit.md"
        assert fragment.metadata["content_length"] == 3
        assert fragment.metadata["truncated"] is False

    def test_fragment_metadata_reports_truncation(self) -> None:
        truncated_snap = WorkspaceInstructionsSnapshot(
            source_path=Path("/w/orbit.md"),
            exists=True,
            content="abc",
            truncated=True,
        )
        fragment = build_workspace_instructions_fragment(
            truncated_snap, policy_name="test"
        )
        assert fragment.metadata["truncated"] is True


# ---------------------------------------------------------------------------
# StructuredContextAssembler integration
# ---------------------------------------------------------------------------


class _StubPolicy(WorkspaceInstructionsDisclosurePolicy):
    def __init__(self, *, disclose: bool) -> None:
        self._disclose = disclose

    @property
    def policy_name(self) -> str:
        return "stub"

    def decide(
        self, snapshot: WorkspaceInstructionsSnapshot
    ) -> WorkspaceInstructionsDisclosureDecision:
        return WorkspaceInstructionsDisclosureDecision(
            disclose=self._disclose, policy_name=self.policy_name
        )


def _fixture_collector(
    tmp_path: Path, *, content: str | None = "workspace rule one"
) -> WorkspaceInstructionsCollector:
    if content is not None:
        (tmp_path / "orbit.md").write_text(content)
    return WorkspaceInstructionsCollector(tmp_path)


class TestAssemblerIntegration:
    def test_assembler_without_collector_emits_no_workspace_block(self) -> None:
        assembler = StructuredContextAssembler()
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME not in names

    def test_assembler_with_collector_but_no_policy_emits_no_workspace_block(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=_fixture_collector(tmp_path),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME not in names

    def test_assembler_with_collector_and_policy_emits_fragment(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=_fixture_collector(tmp_path),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        fragments = {f.fragment_name: f for f in ac.instruction_fragments}
        assert WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME in fragments
        fragment = fragments[WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME]
        assert fragment.priority == WORKSPACE_INSTRUCTIONS_PRIORITY
        assert "workspace rule one" in fragment.content

    def test_policy_returning_false_suppresses_fragment(self, tmp_path: Path) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=_fixture_collector(tmp_path),
            workspace_instructions_disclosure_policy=_StubPolicy(disclose=False),
        )
        ac = assembler.assemble_structured([], system_prompt="sys")
        names = [f.fragment_name for f in ac.instruction_fragments]
        assert WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME not in names

    def test_absent_orbit_md_is_handled_cleanly(self, tmp_path: Path) -> None:
        # No orbit.md at workspace_root; runtime path still works.
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=WorkspaceInstructionsCollector(tmp_path),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt="Default prompt.")
        assert req.system == "Default prompt."
        assert "<workspace-instructions>" not in (req.system or "")

    def test_absent_orbit_md_with_no_system_prompt_yields_none_system(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=WorkspaceInstructionsCollector(tmp_path),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt=None)
        assert req.system is None

    def test_system_prompt_renders_before_workspace_instructions(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=_fixture_collector(
                tmp_path, content="Tailored workspace rule."
            ),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt="You are helpful.")
        assert req.system is not None
        sys_idx = req.system.find("You are helpful.")
        wi_idx = req.system.find("<workspace-instructions>")
        assert sys_idx >= 0
        assert wi_idx >= 0
        assert sys_idx < wi_idx

    def test_workspace_instructions_render_before_runtime_context(
        self, tmp_path: Path
    ) -> None:
        from src.config.runtime import RuntimeRoot
        from src.governance.runtime_context_disclosure import (
            BasicSelfLocationDisclosurePolicy,
        )
        from src.knowledge import RuntimeContextCollector

        assembler = StructuredContextAssembler(
            runtime_context_collector=RuntimeContextCollector(
                RuntimeRoot(tmp_path, "cli-flag"), tmp_path / "db"
            ),
            runtime_context_disclosure_policy=BasicSelfLocationDisclosurePolicy(),
            workspace_instructions_collector=_fixture_collector(
                tmp_path, content="workspace authoritative rule"
            ),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt=None)
        assert req.system is not None
        wi_idx = req.system.find("<workspace-instructions>")
        rc_idx = req.system.find("<runtime-context>")
        assert wi_idx >= 0
        assert rc_idx >= 0
        assert wi_idx < rc_idx

    def test_workspace_instructions_content_appears_in_system(
        self, tmp_path: Path
    ) -> None:
        assembler = StructuredContextAssembler(
            workspace_instructions_collector=_fixture_collector(
                tmp_path,
                content="# Project rules\n- test rule 1\n- test rule 2\n",
            ),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        req = assembler.assemble([], system_prompt="You are helpful.")
        assert req.system is not None
        assert "# Project rules" in req.system
        assert "test rule 1" in req.system
        assert "test rule 2" in req.system


# ---------------------------------------------------------------------------
# Transcript-truth separation: canonical transcript is untouched
# ---------------------------------------------------------------------------


class TestTranscriptSeparation:
    def test_fragment_does_not_appear_in_transcript_messages(
        self, tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone

        from src.core.runtime.models import (
            ConversationMessage,
            MessageRole,
            make_message_id,
        )

        (tmp_path / "orbit.md").write_text("workspace secret rule")

        user_msg = ConversationMessage(
            message_id=make_message_id(),
            session_id="s1",
            role=MessageRole.USER,
            content="Hello.",
            turn_index=1,
            created_at=datetime.now(timezone.utc),
        )
        original_content = user_msg.content

        assembler = StructuredContextAssembler(
            workspace_instructions_collector=WorkspaceInstructionsCollector(tmp_path),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
            ),
        )
        ac = assembler.assemble_structured([user_msg], system_prompt="sys")

        # User message reference is preserved and its content is untouched.
        assert ac.transcript_messages[0] is user_msg
        assert user_msg.content == original_content
        # The workspace rule ends up in instruction fragments, never in the
        # user message contents.
        assert "workspace secret rule" not in user_msg.content
        instruction_text = "\n".join(
            f.content for f in ac.instruction_fragments
        )
        assert "workspace secret rule" in instruction_text


# ---------------------------------------------------------------------------
# Debug envelope projection
# ---------------------------------------------------------------------------


class TestDebugEnvelopeProjection:
    def test_workspace_instructions_fragment_appears_in_envelope_preview(
        self, tmp_path: Path
    ) -> None:
        from src.knowledge.assembly import build_envelope

        (tmp_path / "orbit.md").write_text("envelope visibility check")

        assembler = StructuredContextAssembler(
            workspace_instructions_collector=WorkspaceInstructionsCollector(tmp_path),
            workspace_instructions_disclosure_policy=(
                BasicWorkspaceInstructionsDisclosurePolicy()
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
        assert WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME in names
