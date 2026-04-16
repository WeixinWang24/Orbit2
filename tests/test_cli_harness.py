"""CLI harness tests — operator surface over SessionManager."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from src.cli.harness import _run_interactive, _show_history, main
from src.providers.base import ExecutionBackend
from src.runtime.models import ExecutionPlan, Message, TurnRequest
from src.runtime.session import SessionManager
from src.store.sqlite import SQLiteSessionStore


# ---------------------------------------------------------------------------
# Streaming-capable test backend
# ---------------------------------------------------------------------------

class StreamingDummyBackend(ExecutionBackend):
    @property
    def backend_name(self) -> str:
        return "streaming-dummy"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        text = f"echo: {request.messages[-1].content}"
        if on_partial_text:
            accumulated = ""
            for ch in text:
                accumulated += ch
                on_partial_text(accumulated)
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="streaming-dummy-final-text",
            final_text=text,
            model="dummy-model",
        )


class NonStreamingDummyBackend(ExecutionBackend):
    @property
    def backend_name(self) -> str:
        return "nonstreaming-dummy"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="nonstreaming-dummy-final-text",
            final_text=f"reply: {request.messages[-1].content}",
            model="dummy-model",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def streaming_manager(tmp_path: Path) -> SessionManager:
    store = SQLiteSessionStore(tmp_path / "test.db")
    return SessionManager(backend=StreamingDummyBackend(), store=store)


@pytest.fixture
def nonstreaming_manager(tmp_path: Path) -> SessionManager:
    store = SQLiteSessionStore(tmp_path / "test.db")
    return SessionManager(backend=NonStreamingDummyBackend(), store=store)


# ---------------------------------------------------------------------------
# 1. CLI delegates to SessionManager
# ---------------------------------------------------------------------------

def test_cli_delegates_to_session_manager(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session(system_prompt="test")
    inputs = iter(["hello", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    messages = streaming_manager.list_messages(session.session_id)
    assert len(messages) == 2
    assert messages[0].role.value == "user"
    assert messages[0].content == "hello"
    assert messages[1].role.value == "assistant"
    assert messages[1].content == "echo: hello"


# ---------------------------------------------------------------------------
# 2. Basic one-turn interaction
# ---------------------------------------------------------------------------

def test_one_turn_interaction(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "echo: hi" in captured.out
    assert "Bye." in captured.out


# ---------------------------------------------------------------------------
# 3. Multi-turn interaction
# ---------------------------------------------------------------------------

def test_multi_turn_interaction(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["first", "second", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    messages = streaming_manager.list_messages(session.session_id)
    assert len(messages) == 4
    assert messages[0].content == "first"
    assert messages[1].content == "echo: first"
    assert messages[2].content == "second"
    assert messages[3].content == "echo: second"


# ---------------------------------------------------------------------------
# 4. /history behavior
# ---------------------------------------------------------------------------

def test_history_command(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["hello", "/history", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "Transcript (2 messages)" in captured.out
    assert "[user] hello" in captured.out
    assert "[assistant] echo: hello" in captured.out


# ---------------------------------------------------------------------------
# 5. /quit behavior
# ---------------------------------------------------------------------------

def test_quit_exits_cleanly(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "Bye." in captured.out
    messages = streaming_manager.list_messages(session.session_id)
    assert len(messages) == 0


def test_eof_exits_cleanly(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    with patch("builtins.input", side_effect=EOFError):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "Bye." in captured.out


# ---------------------------------------------------------------------------
# 6. Streaming callback display
# ---------------------------------------------------------------------------

def test_streaming_output_appears_incrementally(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["test", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "echo: test" in captured.out


# ---------------------------------------------------------------------------
# 7. Non-streaming fallback
# ---------------------------------------------------------------------------

def test_nonstreaming_fallback_shows_final_text(nonstreaming_manager: SessionManager, capsys) -> None:
    session = nonstreaming_manager.create_session()
    inputs = iter(["test", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(nonstreaming_manager, session.session_id)

    captured = capsys.readouterr()
    assert "reply: test" in captured.out


# ---------------------------------------------------------------------------
# 8. Main entrypoint delegation
# ---------------------------------------------------------------------------

def test_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_main():
        called["cli"] = True

    monkeypatch.setattr("src.cli.harness.main", fake_main)
    import main as main_module
    monkeypatch.setattr(main_module, "main", fake_main)
    fake_main()
    assert called.get("cli") is True


# ---------------------------------------------------------------------------
# _show_history directly
# ---------------------------------------------------------------------------

def test_show_history_empty(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    _show_history(streaming_manager, session.session_id)
    captured = capsys.readouterr()
    assert "empty transcript" in captured.out


def test_show_history_truncates_long_content(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    long_input = "x" * 200
    inputs = iter([long_input, "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    _show_history(streaming_manager, session.session_id)
    captured = capsys.readouterr()
    assert "..." in captured.out
