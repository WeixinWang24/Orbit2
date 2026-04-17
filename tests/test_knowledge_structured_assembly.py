"""Knowledge Surface structured context-assembly tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.knowledge import (
    AssembledContext,
    ContextFragment,
    StructuredContextAssembler,
)
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    MessageRole,
    TurnRequest,
)
from src.core.runtime.session import SessionManager
from src.core.store.sqlite import SQLiteSessionStore


def _msg(role: MessageRole, content: str, turn_index: int, session_id: str = "s1") -> ConversationMessage:
    return ConversationMessage(
        message_id=f"msg_{turn_index}",
        session_id=session_id,
        role=role,
        content=content,
        turn_index=turn_index,
        created_at=datetime.now(timezone.utc),
    )


class TestAssembledContextProjection:
    def test_empty_context_projects_to_none_system_and_empty_messages(self) -> None:
        ctx = AssembledContext()
        req = ctx.to_turn_request()
        assert req.system is None
        assert req.messages == []

    def test_single_fragment_becomes_system(self) -> None:
        ctx = AssembledContext(
            instruction_fragments=[
                ContextFragment(
                    fragment_name="a",
                    visibility_scope="instruction",
                    content="be concise",
                    priority=50,
                )
            ]
        )
        req = ctx.to_turn_request()
        assert req.system == "be concise"

    def test_multiple_fragments_join_in_priority_order(self) -> None:
        ctx = AssembledContext(
            instruction_fragments=[
                ContextFragment(
                    fragment_name="low",
                    visibility_scope="instruction",
                    content="low priority line",
                    priority=10,
                ),
                ContextFragment(
                    fragment_name="high",
                    visibility_scope="instruction",
                    content="high priority line",
                    priority=100,
                ),
            ]
        )
        req = ctx.to_turn_request()
        assert req.system == "high priority line\n\nlow priority line"

    def test_blank_fragments_drop_out(self) -> None:
        ctx = AssembledContext(
            instruction_fragments=[
                ContextFragment(
                    fragment_name="blank",
                    visibility_scope="instruction",
                    content="   ",
                    priority=50,
                ),
                ContextFragment(
                    fragment_name="real",
                    visibility_scope="instruction",
                    content="real content",
                    priority=10,
                ),
            ]
        )
        req = ctx.to_turn_request()
        assert req.system == "real content"

    def test_transcript_messages_are_projected_to_turn_request(self) -> None:
        messages = [
            _msg(MessageRole.USER, "hi", 1),
            _msg(MessageRole.ASSISTANT, "hello", 2),
        ]
        ctx = AssembledContext(transcript_messages=messages)
        req = ctx.to_turn_request()
        assert [m.role for m in req.messages] == ["user", "assistant"]
        assert [m.content for m in req.messages] == ["hi", "hello"]

    def test_assistant_tool_call_metadata_round_trips(self) -> None:
        tool_call_blob = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
        assistant = _msg(MessageRole.ASSISTANT, "", 1)
        assistant.metadata = {"tool_calls": tool_call_blob}
        ctx = AssembledContext(transcript_messages=[assistant])
        req = ctx.to_turn_request()
        assert req.messages[0].tool_calls == tool_call_blob
        assert req.messages[0].content is None

    def test_tool_role_preserves_tool_call_id(self) -> None:
        tool_msg = _msg(MessageRole.TOOL, "result", 1)
        tool_msg.metadata = {"tool_call_id": "c1"}
        ctx = AssembledContext(transcript_messages=[tool_msg])
        req = ctx.to_turn_request()
        assert req.messages[0].tool_call_id == "c1"
        assert req.messages[0].content == "result"


class TestStructuredContextAssembler:
    def test_system_prompt_becomes_instruction_fragment(self) -> None:
        assembler = StructuredContextAssembler()
        plan = assembler.assemble_structured([], system_prompt="be concise")
        assert len(plan.instruction_fragments) == 1
        frag = plan.instruction_fragments[0]
        assert frag.fragment_name == "session_system_prompt"
        assert frag.content == "be concise"
        assert frag.visibility_scope == "instruction"
        assert frag.metadata["origin"] == "session"

    def test_no_system_prompt_means_no_session_fragment(self) -> None:
        assembler = StructuredContextAssembler()
        plan = assembler.assemble_structured([])
        assert plan.instruction_fragments == []

    def test_extra_instruction_fragments_are_preserved(self) -> None:
        extra = [
            ContextFragment(
                fragment_name="runtime_mode",
                visibility_scope="instruction",
                content="runtime mode: dev",
                priority=80,
            ),
        ]
        assembler = StructuredContextAssembler(extra_instruction_fragments=extra)
        plan = assembler.assemble_structured([], system_prompt="be concise")
        names = [f.fragment_name for f in plan.instruction_fragments]
        assert "session_system_prompt" in names
        assert "runtime_mode" in names

    def test_transcript_messages_are_held_canonically(self) -> None:
        messages = [_msg(MessageRole.USER, "hi", 1)]
        assembler = StructuredContextAssembler()
        plan = assembler.assemble_structured(messages)
        assert len(plan.transcript_messages) == 1
        assert plan.transcript_messages[0].content == "hi"
        assert plan.transcript_messages[0].role == MessageRole.USER

    def test_mutating_plan_transcript_list_does_not_affect_caller(self) -> None:
        messages = [_msg(MessageRole.USER, "hi", 1)]
        assembler = StructuredContextAssembler()
        plan = assembler.assemble_structured(messages)
        plan.transcript_messages.clear()
        assert len(messages) == 1
        assert messages[0].content == "hi"

    def test_projected_tool_calls_are_not_aliased_to_transcript_metadata(self) -> None:
        tool_call_blob = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
        assistant = _msg(MessageRole.ASSISTANT, "", 1)
        assistant.metadata = {"tool_calls": tool_call_blob}
        assembler = StructuredContextAssembler()
        req = assembler.assemble([assistant])
        # Mutating the projected Message tool_calls must not bleed into the canonical transcript
        req.messages[0].tool_calls.append({"id": "intruder"})
        assert assistant.metadata["tool_calls"] == tool_call_blob
        assert len(assistant.metadata["tool_calls"]) == 1

    def test_extra_fragment_mutation_does_not_affect_assembler(self) -> None:
        seed = [ContextFragment(fragment_name="n", visibility_scope="instruction", content="x", priority=50, metadata={"k": "v"})]
        assembler = StructuredContextAssembler(extra_instruction_fragments=seed)
        # Mutate the seed list and the contained fragment after construction
        seed.append(ContextFragment(fragment_name="intruder", visibility_scope="instruction", content="y", priority=1))
        seed[0].metadata["k"] = "mutated"
        plan = assembler.assemble_structured([])
        names = [f.fragment_name for f in plan.instruction_fragments]
        assert "intruder" not in names
        assert plan.instruction_fragments[0].metadata["k"] == "v"

    def test_extra_fragment_priority_wins_in_projected_system(self) -> None:
        extra = [ContextFragment(
            fragment_name="override",
            visibility_scope="instruction",
            content="OVERRIDE_LINE",
            priority=150,  # higher than SESSION_SYSTEM_PROMPT_PRIORITY (100)
        )]
        assembler = StructuredContextAssembler(extra_instruction_fragments=extra)
        req = assembler.assemble([], system_prompt="session line")
        assert req.system is not None
        override_idx = req.system.find("OVERRIDE_LINE")
        session_idx = req.system.find("session line")
        assert override_idx >= 0
        assert session_idx >= 0
        assert override_idx < session_idx

    def test_metadata_reports_counts(self) -> None:
        messages = [_msg(MessageRole.USER, "a", 1), _msg(MessageRole.ASSISTANT, "b", 2)]
        assembler = StructuredContextAssembler()
        plan = assembler.assemble_structured(messages, system_prompt="sys")
        assert plan.metadata["assembler"] == "structured"
        assert plan.metadata["transcript_message_count"] == 2
        assert plan.metadata["instruction_fragment_count"] == 1

    def test_assemble_returns_turn_request(self) -> None:
        assembler = StructuredContextAssembler()
        result = assembler.assemble([], system_prompt="be concise")
        assert isinstance(result, TurnRequest)
        assert result.system == "be concise"

    def test_assemble_matches_transcript_baseline_for_simple_case(self) -> None:
        messages = [
            _msg(MessageRole.USER, "first", 1),
            _msg(MessageRole.ASSISTANT, "reply", 2),
        ]
        structured = StructuredContextAssembler().assemble(messages, system_prompt="sys")
        assert structured.system == "sys"
        assert [m.role for m in structured.messages] == ["user", "assistant"]
        assert [m.content for m in structured.messages] == ["first", "reply"]

    def test_transcript_messages_not_mutated(self) -> None:
        original = [_msg(MessageRole.USER, "hello", 1)]
        assembler = StructuredContextAssembler()
        assembler.assemble(original, system_prompt="sys")
        assert original[0].content == "hello"
        assert original[0].role == MessageRole.USER


class _RecordingBackend(ExecutionBackend):
    def __init__(self) -> None:
        self.last_request: TurnRequest | None = None

    @property
    def backend_name(self) -> str:
        return "recording"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        self.last_request = request
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="recording",
            final_text="ok",
            model="recording-model",
        )


class TestSessionManagerStructuredDefault:
    def test_default_assembler_is_structured(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        backend = _RecordingBackend()
        manager = SessionManager(backend=backend, store=store)

        session = manager.create_session(system_prompt="sys")
        manager.run_turn(session.session_id, "hello")

        assert backend.last_request is not None
        assert backend.last_request.system == "sys"
        assert backend.last_request.messages[0].content == "hello"

    def test_structured_path_does_not_mutate_canonical_transcript(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        backend = _RecordingBackend()
        assembler = StructuredContextAssembler(
            extra_instruction_fragments=[
                ContextFragment(
                    fragment_name="runtime_mode",
                    visibility_scope="instruction",
                    content="runtime mode: dev",
                    priority=80,
                ),
            ]
        )
        manager = SessionManager(backend=backend, store=store, assembler=assembler)

        session = manager.create_session(system_prompt="be concise")
        manager.run_turn(session.session_id, "hello")

        # Provider saw a structurally composed system prompt (two fragments joined)
        assert "be concise" in (backend.last_request.system or "")
        assert "runtime mode: dev" in (backend.last_request.system or "")
        assert "\n\n" in (backend.last_request.system or "")
        # Canonical transcript is unchanged — user input only, no fragment content leaked in
        transcript = manager.list_messages(session.session_id)
        assert transcript[0].content == "hello"
        assert "runtime mode" not in transcript[0].content
