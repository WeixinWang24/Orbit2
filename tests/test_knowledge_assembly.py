"""Knowledge Surface context assembly tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from src.knowledge.assembly import ContextAssembler, TranscriptContextAssembler
from src.providers.base import ExecutionBackend
from src.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    Message,
    MessageRole,
    TurnRequest,
)
from src.runtime.session import SessionManager
from src.store.sqlite import SQLiteSessionStore


def _msg(role: MessageRole, content: str, turn_index: int, session_id: str = "s1") -> ConversationMessage:
    return ConversationMessage(
        message_id=f"msg_{turn_index}",
        session_id=session_id,
        role=role,
        content=content,
        turn_index=turn_index,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# TranscriptContextAssembler unit tests
# ---------------------------------------------------------------------------

class TestTranscriptContextAssembler:
    def test_empty_transcript(self) -> None:
        assembler = TranscriptContextAssembler()
        result = assembler.assemble([], system_prompt="be concise")
        assert result.system == "be concise"
        assert result.messages == []

    def test_single_user_message(self) -> None:
        assembler = TranscriptContextAssembler()
        messages = [_msg(MessageRole.USER, "hello", 1)]
        result = assembler.assemble(messages)
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "hello"

    def test_multi_turn_preserves_order(self) -> None:
        assembler = TranscriptContextAssembler()
        messages = [
            _msg(MessageRole.USER, "first", 1),
            _msg(MessageRole.ASSISTANT, "reply one", 2),
            _msg(MessageRole.USER, "second", 3),
            _msg(MessageRole.ASSISTANT, "reply two", 4),
        ]
        result = assembler.assemble(messages, system_prompt="test")
        assert result.system == "test"
        assert len(result.messages) == 4
        assert [m.role for m in result.messages] == ["user", "assistant", "user", "assistant"]
        assert [m.content for m in result.messages] == ["first", "reply one", "second", "reply two"]

    def test_no_system_prompt(self) -> None:
        assembler = TranscriptContextAssembler()
        messages = [_msg(MessageRole.USER, "hi", 1)]
        result = assembler.assemble(messages)
        assert result.system is None

    def test_returns_turn_request(self) -> None:
        assembler = TranscriptContextAssembler()
        result = assembler.assemble([])
        assert isinstance(result, TurnRequest)

    def test_transcript_messages_not_mutated(self) -> None:
        assembler = TranscriptContextAssembler()
        original = [_msg(MessageRole.USER, "hello", 1)]
        assembler.assemble(original, system_prompt="test")
        assert original[0].content == "hello"
        assert original[0].role == MessageRole.USER


# ---------------------------------------------------------------------------
# Custom assembler (demonstrates plug-and-play seam)
# ---------------------------------------------------------------------------

class PrefixAssembler(ContextAssembler):
    """Test assembler that prepends a context note to each message."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def assemble(
        self,
        messages: list[ConversationMessage],
        *,
        system_prompt: str | None = None,
    ) -> TurnRequest:
        return TurnRequest(
            system=system_prompt,
            messages=[
                Message(role=m.role.value, content=f"{self._prefix}: {m.content}")
                for m in messages
            ],
        )


class RecordingBackend(ExecutionBackend):
    """Backend that records the TurnRequest it receives."""

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
            plan_label="recording-final-text",
            final_text="ok",
            model="recording-model",
        )


class TestCustomAssemblerIntegration:
    def test_session_manager_uses_custom_assembler(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        backend = RecordingBackend()
        assembler = PrefixAssembler("ctx")
        manager = SessionManager(backend=backend, store=store, assembler=assembler)

        session = manager.create_session(system_prompt="test system")
        manager.run_turn(session.session_id, "hello")

        assert backend.last_request is not None
        assert backend.last_request.system == "test system"
        assert len(backend.last_request.messages) == 1
        assert backend.last_request.messages[0].content == "ctx: hello"

    def test_default_assembler_is_transcript(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        backend = RecordingBackend()
        manager = SessionManager(backend=backend, store=store)

        session = manager.create_session()
        manager.run_turn(session.session_id, "hello")

        assert backend.last_request is not None
        assert backend.last_request.messages[0].content == "hello"


# ---------------------------------------------------------------------------
# Transcript/context separation verification
# ---------------------------------------------------------------------------

class TestTranscriptContextSeparation:
    def test_transcript_unchanged_after_assembly(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        backend = RecordingBackend()
        assembler = PrefixAssembler("modified")
        manager = SessionManager(backend=backend, store=store, assembler=assembler)

        session = manager.create_session()
        manager.run_turn(session.session_id, "original")

        # The provider saw modified content
        assert backend.last_request.messages[0].content == "modified: original"
        # But the canonical transcript is unchanged
        transcript = manager.list_messages(session.session_id)
        assert transcript[0].content == "original"
        assert transcript[1].content == "ok"  # assistant reply stored verbatim
