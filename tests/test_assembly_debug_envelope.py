"""Tests for the Knowledge Surface assembly debug envelope + inspector projection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    MessageRole,
    TurnRequest,
    make_message_id,
    make_session_id,
)
from src.core.runtime.session import SessionManager
from src.core.store.sqlite import SQLiteSessionStore
from src.knowledge.assembly import (
    AssemblyDebugEnvelope,
    StructuredContextAssembler,
    TranscriptContextAssembler,
    build_envelope,
)
from src.knowledge.assembly.debug import (
    _FRAGMENT_PREVIEW_CHARS,
    _MAX_MESSAGE_PREVIEW_ROWS,
    _MESSAGE_PREVIEW_CHARS,
    _SYSTEM_PREVIEW_CHARS,
)
from src.knowledge.exposure import ExposureDecision
from src.operation.inspector.web_inspector import (
    _extract_assembly_envelopes,
    _html_page,
    _render_assembly_panel,
    _render_transcript_panel,
)


NOW = datetime(2026, 4, 17, 13, 0, 0, tzinfo=timezone.utc)


def _msg(role: MessageRole, content: str, turn_index: int, **meta) -> ConversationMessage:
    return ConversationMessage(
        message_id=make_message_id(),
        session_id="s1",
        role=role,
        content=content,
        turn_index=turn_index,
        created_at=NOW,
        metadata=meta,
    )


class DummyBackend(ExecutionBackend):
    def __init__(self, *, final_text: str = "ok") -> None:
        self._final_text = final_text
        self.last_request: TurnRequest | None = None

    @property
    def backend_name(self) -> str:
        return "dummy"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        self.last_request = request
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="dummy-final",
            final_text=self._final_text,
            model="dummy-model",
        )


# ---------------------------------------------------------------------------
# Envelope shape
# ---------------------------------------------------------------------------


class TestEnvelopeShape:
    def test_structured_assembler_surfaces_instruction_fragments(self) -> None:
        assembler = StructuredContextAssembler()
        msgs = [
            _msg(MessageRole.USER, "hello", 1),
            _msg(MessageRole.ASSISTANT, "world", 2),
        ]
        ctx = assembler.assemble_structured(msgs, system_prompt="be helpful")
        req = ctx.to_turn_request()

        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=len(msgs),
            request=req,
            assembled_context=ctx,
            exposure_decision=None,
        )

        assert env.assembler_name == "StructuredContextAssembler"
        assert env.transcript_message_count == 2
        assert len(env.instruction_fragments) == 1
        assert env.instruction_fragments[0].fragment_name == "session_system_prompt"
        assert env.instruction_fragments[0].visibility_scope == "instruction"
        assert env.assembled_system_preview == "be helpful"
        assert env.assembled_message_count == 2
        assert len(env.assembled_messages_preview) == 2
        assert env.assembled_messages_preview[0].role == "user"
        assert env.assembler_metadata["assembler"] == "structured"

    def test_no_structured_context_still_builds_envelope(self) -> None:
        req = TurnRequest(
            system="sys",
            messages=[],
            tool_definitions=None,
        )
        env = build_envelope(
            assembler_name="TranscriptContextAssembler",
            transcript_message_count=0,
            request=req,
            assembled_context=None,
            exposure_decision=None,
        )
        assert env.instruction_fragments == []
        assert env.assembled_system_preview == "sys"
        assert env.assembler_metadata == {}

    def test_exposure_decision_is_projected(self) -> None:
        req = TurnRequest(system=None, messages=[], tool_definitions=[])
        decision = ExposureDecision(
            exposed_tool_names={"tool_a", "tool_b"},
            active_reveal_groups=["default", "filesystem"],
            exposure_reason={"tool_a": "default_exposed"},
            rejected_reveal_requests=["bogus_group"],
        )
        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=0,
            request=req,
            assembled_context=None,
            exposure_decision=decision,
        )
        assert env.exposed_tool_names == ["tool_a", "tool_b"]  # sorted
        assert env.exposed_tool_groups == ["default", "filesystem"]
        assert env.rejected_reveal_requests == ["bogus_group"]


class TestPreviewBounds:
    def test_fragment_preview_truncates_long_content(self) -> None:
        long = "x" * (_FRAGMENT_PREVIEW_CHARS * 2)
        assembler = StructuredContextAssembler()
        ctx = assembler.assemble_structured([], system_prompt=long)
        req = ctx.to_turn_request()
        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=0,
            request=req,
            assembled_context=ctx,
            exposure_decision=None,
        )
        preview = env.instruction_fragments[0].content_preview
        assert len(preview) <= _FRAGMENT_PREVIEW_CHARS
        assert env.instruction_fragments[0].content_length == len(long)

    def test_system_preview_truncates(self) -> None:
        long = "s" * (_SYSTEM_PREVIEW_CHARS * 2)
        req = TurnRequest(system=long, messages=[])
        env = build_envelope(
            assembler_name="TranscriptContextAssembler",
            transcript_message_count=0,
            request=req,
            assembled_context=None,
            exposure_decision=None,
        )
        assert len(env.assembled_system_preview) <= _SYSTEM_PREVIEW_CHARS

    def test_message_rows_are_capped(self) -> None:
        from src.core.runtime.models import Message
        msgs = [Message(role="user", content=f"m{i}") for i in range(_MAX_MESSAGE_PREVIEW_ROWS + 5)]
        req = TurnRequest(system=None, messages=msgs)
        env = build_envelope(
            assembler_name="TranscriptContextAssembler",
            transcript_message_count=len(msgs),
            request=req,
            assembled_context=None,
            exposure_decision=None,
        )
        assert len(env.assembled_messages_preview) == _MAX_MESSAGE_PREVIEW_ROWS
        assert env.assembled_messages_truncated is True
        assert env.assembled_message_count == len(msgs)


class TestSerialization:
    def test_envelope_round_trips_through_json(self) -> None:
        assembler = StructuredContextAssembler()
        ctx = assembler.assemble_structured(
            [_msg(MessageRole.USER, "q", 1)], system_prompt="s"
        )
        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=1,
            request=ctx.to_turn_request(),
            assembled_context=ctx,
            exposure_decision=None,
        )
        blob = env.to_metadata_dict()
        # JSON round-trip must succeed (store persists as JSON)
        text = json.dumps(blob)
        restored = json.loads(text)
        assert restored["assembler_name"] == "StructuredContextAssembler"
        assert restored["instruction_fragments"][0]["fragment_name"] == "session_system_prompt"


# ---------------------------------------------------------------------------
# Session manager integration — envelope persists on assistant metadata
# ---------------------------------------------------------------------------


class TestSessionManagerPersists:
    def test_final_assistant_message_carries_envelope(self, tmp_path: Path) -> None:
        backend = DummyBackend(final_text="reply")
        store = SQLiteSessionStore(tmp_path / "t.db")
        manager = SessionManager(backend=backend, store=store)
        session = manager.create_session(system_prompt="be terse")
        manager.run_turn(session.session_id, "hello")
        msgs = manager.list_messages(session.session_id)
        assistants = [m for m in msgs if m.role == MessageRole.ASSISTANT]
        assert len(assistants) == 1
        env = assistants[0].metadata.get("assembly_envelope")
        assert isinstance(env, dict)
        assert env["assembler_name"] == "StructuredContextAssembler"
        # User turn is already written before planning, so the envelope
        # reflects that — transcript count at planning time = 1.
        assert env["transcript_message_count"] == 1
        fragments = env["instruction_fragments"]
        assert any(f["fragment_name"] == "session_system_prompt" for f in fragments)

    def test_envelope_last_update_accessor_matches_persisted(self, tmp_path: Path) -> None:
        backend = DummyBackend()
        store = SQLiteSessionStore(tmp_path / "t.db")
        manager = SessionManager(backend=backend, store=store)
        session = manager.create_session(system_prompt="sys")
        manager.run_turn(session.session_id, "hi")
        assert manager._last_assembly_envelope is not None
        assert isinstance(manager._last_assembly_envelope, AssemblyDebugEnvelope)
        persisted = next(
            m.metadata["assembly_envelope"]
            for m in manager.list_messages(session.session_id)
            if m.role == MessageRole.ASSISTANT
        )
        assert (
            persisted["assembler_name"]
            == manager._last_assembly_envelope.assembler_name
        )

    def test_multi_tool_loop_each_assistant_gets_its_own_envelope(
        self, tmp_path: Path,
    ) -> None:
        """Pairing invariant: every assistant message written by run_turn
        carries the envelope from the planning call that produced its plan,
        not the envelope from a later planning call. Exercised by chaining
        two planning calls — first returns a tool_request, then final text.
        """
        from src.capability.boundary import CapabilityBoundary
        from src.capability.registry import CapabilityRegistry
        from src.capability.tools.base import Tool
        from src.capability.models import ToolDefinition, ToolResult
        from src.core.runtime.models import ToolRequest

        class EchoTool(Tool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="echo",
                    description="Echo a message back.",
                    parameters={
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                )

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(ok=True, content=f"echoed: {kwargs.get('msg','')}")

        class TwoPhaseBackend(ExecutionBackend):
            def __init__(self) -> None:
                self.phase = 0

            @property
            def backend_name(self) -> str:
                return "two-phase"

            def plan_from_messages(self, request, *, on_partial_text=None):
                self.phase += 1
                if self.phase == 1:
                    return ExecutionPlan(
                        source_backend=self.backend_name,
                        plan_label="call-tool",
                        final_text=None,
                        model="m",
                        tool_requests=[
                            ToolRequest(
                                tool_call_id="c1",
                                tool_name="echo",
                                arguments={"msg": "hi"},
                            )
                        ],
                    )
                return ExecutionPlan(
                    source_backend=self.backend_name,
                    plan_label="final",
                    final_text="done",
                    model="m",
                )

        registry = CapabilityRegistry()
        registry.register(EchoTool())
        boundary = CapabilityBoundary(registry, tmp_path)
        store = SQLiteSessionStore(tmp_path / "t.db")
        mgr = SessionManager(
            backend=TwoPhaseBackend(),
            store=store,
            capability_boundary=boundary,
        )
        session = mgr.create_session(system_prompt="sys")
        mgr.run_turn(session.session_id, "please call echo")

        msgs = mgr.list_messages(session.session_id)
        assistants = [m for m in msgs if m.role == MessageRole.ASSISTANT]
        # Two assistant messages: one with tool_calls, one with final text.
        assert len(assistants) == 2
        env_first = assistants[0].metadata.get("assembly_envelope")
        env_final = assistants[1].metadata.get("assembly_envelope")
        assert isinstance(env_first, dict) and isinstance(env_final, dict)
        # Pairing invariant: the first assistant's envelope saw only the
        # user message (transcript count 1); the final assistant's envelope
        # saw user + first-assistant + tool-result (transcript count 3).
        assert env_first["transcript_message_count"] == 1
        assert env_final["transcript_message_count"] == 3
        # Both envelopes came from the same assembler.
        assert env_first["assembler_name"] == env_final["assembler_name"]


# ---------------------------------------------------------------------------
# Inspector projection — transcript vs assembly separation
# ---------------------------------------------------------------------------


class TestInspectorProjection:
    def _envelope_meta(self) -> dict:
        req = TurnRequest(system="system txt", messages=[])
        env = build_envelope(
            assembler_name="StructuredContextAssembler",
            transcript_message_count=1,
            request=req,
            assembled_context=None,
            exposure_decision=None,
        )
        return env.to_metadata_dict()

    def test_extract_picks_up_assembly_envelopes_in_order(self) -> None:
        msgs = [
            _msg(MessageRole.USER, "u1", 1),
            _msg(MessageRole.ASSISTANT, "a1", 2, assembly_envelope=self._envelope_meta()),
            _msg(MessageRole.USER, "u2", 3),
            _msg(MessageRole.ASSISTANT, "a2", 4, assembly_envelope=self._envelope_meta()),
        ]
        rows = _extract_assembly_envelopes(msgs)
        assert [turn for turn, _ in rows] == [2, 4]

    def test_extract_skips_assistants_without_envelope(self) -> None:
        msgs = [
            _msg(MessageRole.USER, "u", 1),
            _msg(MessageRole.ASSISTANT, "a", 2),  # no envelope
        ]
        assert _extract_assembly_envelopes(msgs) == []

    def test_panel_shows_projection_banner(self) -> None:
        msgs = [
            _msg(MessageRole.ASSISTANT, "a", 1, assembly_envelope=self._envelope_meta())
        ]
        html = _render_assembly_panel(messages=msgs)
        assert "projection only" in html.lower()
        assert "transcript truth" in html.lower()

    def test_panel_empty_state_when_no_envelopes(self) -> None:
        msgs = [_msg(MessageRole.USER, "u", 1)]
        html = _render_assembly_panel(messages=msgs)
        assert "no assembly envelopes" in html.lower()

    def test_transcript_and_assembly_tabs_render_distinct_html(self) -> None:
        msgs = [
            _msg(MessageRole.USER, "u1", 1),
            _msg(MessageRole.ASSISTANT, "a1", 2, assembly_envelope=self._envelope_meta()),
        ]
        transcript_html = _render_transcript_panel(msgs)
        assembly_html = _render_assembly_panel(messages=msgs)
        # Assembly-specific content must not appear in the transcript panel.
        assert "projection only" not in transcript_html.lower()
        assert "assembled provider" not in transcript_html.lower()
        # Transcript-only structural element should not leak into assembly.
        # (message-card class is transcript rendering's wrapper; assembly
        # uses section-card + fragment-card instead.)
        assert 'class="message-card' not in assembly_html

    def test_full_page_exposes_assembly_tab(self) -> None:
        from src.core.runtime.models import Session as SessionModel
        session = SessionModel(
            session_id="s1",
            backend_name="dummy",
            system_prompt=None,
            status=__import__(
                "src.core.runtime.models", fromlist=["SessionStatus"],
            ).SessionStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )
        msgs = [
            _msg(MessageRole.ASSISTANT, "a", 1, assembly_envelope=self._envelope_meta()),
        ]
        html = _html_page(
            sessions=[session],
            current_session=session,
            messages=msgs,
            active_tab="assembly",
            right_tab="metadata",
            db_path="/tmp/x",
        )
        assert ">Assembly<" in html
        assert ">Transcript<" in html
        assert ">Debug<" in html
        assert "projection only" in html.lower()
