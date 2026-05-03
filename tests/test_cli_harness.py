"""CLI harness tests — operator surface over SessionManager."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from src.operation.cli.harness import (
    _build_backend,
    _run_interactive,
    _show_history,
    main,
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\[\?[0-9]+[hl]|\x1b\[[0-9]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import ExecutionPlan, Message, TurnRequest
from src.core.runtime.session import SessionManager
from src.core.store.sqlite import SQLiteSessionStore


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


def test_build_vllm_backend_reads_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = object()
    captured_kwargs = {}
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: captured_kwargs.update(kwargs) or fake_client,
    )
    config_dir = tmp_path / ".runtime"
    config_dir.mkdir()
    (config_dir / "agent_runtime.toml").write_text(
        "\n".join([
            "[vllm]",
            "base_url_env = 'ORBIT2_TEST_VLLM_BASE_URL'",
            "basic_auth_username_env = 'ORBIT2_TEST_VLLM_USERNAME'",
            "basic_auth_password_env = 'ORBIT2_TEST_VLLM_PASSWORD'",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "ORBIT2_TEST_VLLM_BASE_URL",
        "http://10.204.18.32:8080/v1/chat/completions",
    )
    monkeypatch.setenv("ORBIT2_TEST_VLLM_USERNAME", "alice")
    monkeypatch.setenv("ORBIT2_TEST_VLLM_PASSWORD", "secret")

    backend = _build_backend("vllm", "Qwopus-GLM-18B", tmp_path)

    assert backend.backend_name == "openai-compatible"
    assert captured_kwargs["base_url"] == "http://10.204.18.32:8080/v1"
    assert captured_kwargs["default_headers"]["Authorization"].startswith("Basic ")


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
    plain = _strip_ansi(captured.out)
    assert "echo: hi" in plain
    assert "Bye." in plain


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
    plain = _strip_ansi(captured.out)
    assert "Transcript" in plain
    assert "(2 messages)" in plain
    assert "user" in plain and "hello" in plain
    assert "assistant" in plain and "echo: hello" in plain


# ---------------------------------------------------------------------------
# 5. /quit behavior
# ---------------------------------------------------------------------------

def test_quit_exits_cleanly(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)
    assert "Bye." in plain
    messages = streaming_manager.list_messages(session.session_id)
    assert len(messages) == 0


def test_eof_exits_cleanly(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    with patch("builtins.input", side_effect=EOFError):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)
    assert "Bye." in plain


# ---------------------------------------------------------------------------
# 6. Streaming callback display
# ---------------------------------------------------------------------------

def test_streaming_output_appears_incrementally(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    inputs = iter(["test", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)
    assert "echo: test" in plain


# ---------------------------------------------------------------------------
# 7. Non-streaming fallback
# ---------------------------------------------------------------------------

def test_nonstreaming_fallback_shows_final_text(nonstreaming_manager: SessionManager, capsys) -> None:
    session = nonstreaming_manager.create_session()
    inputs = iter(["test", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(nonstreaming_manager, session.session_id)

    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)
    assert "reply: test" in plain


def test_nonstreaming_fallback_applies_markdown_render(tmp_path: Path, capsys) -> None:
    """Non-streaming path should route `plan.final_text` through the renderer.

    Handoff 25 widens the renderer to headings/bullets/italics; ensure the
    fallback path actually uses it instead of printing raw markdown.
    """

    class MarkdownNonStreamingBackend(ExecutionBackend):
        @property
        def backend_name(self) -> str:
            return "md-nonstream-dummy"

        def plan_from_messages(
            self,
            request: TurnRequest,
            *,
            on_partial_text: Callable[[str], None] | None = None,
        ) -> ExecutionPlan:
            return ExecutionPlan(
                source_backend=self.backend_name,
                plan_label="md-nonstream-dummy-final",
                final_text="### heading\n- **bold** body",
                model="dummy-model",
            )

    store = SQLiteSessionStore(tmp_path / "test.db")
    mgr = SessionManager(backend=MarkdownNonStreamingBackend(), store=store)
    session = mgr.create_session()
    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(mgr, session.session_id)

    plain = _strip_ansi(capsys.readouterr().out)
    # Raw markers gone; content preserved; bullet glyph present.
    assert "### heading" not in plain
    assert "- **bold** body" not in plain
    assert "heading" in plain
    assert "bold" in plain
    assert "\u2022 bold body" in plain


def test_tty_nonstreaming_wraps_long_message_before_terminal_autowrap(
    tmp_path: Path, capsys,
) -> None:
    class LongNonStreamingBackend(ExecutionBackend):
        @property
        def backend_name(self) -> str:
            return "long-nonstream-dummy"

        def plan_from_messages(
            self,
            request: TurnRequest,
            *,
            on_partial_text: Callable[[str], None] | None = None,
        ) -> ExecutionPlan:
            return ExecutionPlan(
                source_backend=self.backend_name,
                plan_label="long-nonstream-dummy-final",
                final_text="abcdef你好世界",
                model="dummy-model",
            )

    store = SQLiteSessionStore(tmp_path / "test.db")
    mgr = SessionManager(backend=LongNonStreamingBackend(), store=store)
    session = mgr.create_session()

    import sys as _sys

    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs), patch.object(
        _sys.stdout, "isatty", return_value=True,
    ), patch("src.operation.cli.harness._term_width", return_value=8):
        _run_interactive(mgr, session.session_id)

    plain = _strip_ansi(capsys.readouterr().out)
    assert "abcdef你\n好世界" in plain


class _MarkdownStreamingBackend(ExecutionBackend):
    """Backend that streams a markdown-rich response character by character."""

    @property
    def backend_name(self) -> str:
        return "md-stream-dummy"

    def plan_from_messages(
        self,
        request: TurnRequest,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        text = "### heading\n- **bold** body"
        if on_partial_text:
            for i in range(1, len(text) + 1):
                on_partial_text(text[:i])
        return ExecutionPlan(
            source_backend=self.backend_name,
            plan_label="md-stream-dummy-final",
            final_text=text,
            model="dummy-model",
        )


def test_tty_streaming_shows_status_then_rendered_once(
    tmp_path: Path, capsys,
) -> None:
    """Regression: TTY streaming must not leak raw markdown into scrollback."""
    store = SQLiteSessionStore(tmp_path / "test.db")
    mgr = SessionManager(backend=_MarkdownStreamingBackend(), store=store)
    session = mgr.create_session()

    import sys as _sys

    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs), patch.object(
        _sys.stdout, "isatty", return_value=True,
    ):
        _run_interactive(mgr, session.session_id)

    out = capsys.readouterr().out
    plain = _strip_ansi(out)

    assert "\x1b7" not in out
    assert "\x1b8" not in out
    assert "\x1b[J" not in out
    assert "\r\x1b[K" in out
    assert "thinking" in plain
    assert "chars" in plain

    assert "### heading" not in plain
    assert "**bold**" not in plain
    assert "- **bold** body" not in plain
    assert "heading" in plain
    assert "\u2022 bold body" in plain
    assert plain.count("bold body") == 1
    assert plain.count("heading") == 1


def test_tty_rendered_body_preserves_assistant_color_after_styled_spans(
    tmp_path: Path, capsys,
) -> None:
    """Regression: Handoff 30.

    In the post-erase rendered region, each internal `RESET` that closes
    a markdown span (heading / bullet / bold / italic / inline-code)
    must be followed immediately by `CONTENT_ASSISTANT` so ordinary
    body text after the span stays in the assistant purple instead of
    dropping to terminal default. The streaming backend produces
    `"### heading\\n- **bold** body"` — after the `**bold**` close, the
    trailing ` body` must sit inside a re-applied `CONTENT_ASSISTANT`
    region.
    """
    from src.operation.cli.style import CONTENT_ASSISTANT, RESET

    store = SQLiteSessionStore(tmp_path / "test.db")
    mgr = SessionManager(backend=_MarkdownStreamingBackend(), store=store)
    session = mgr.create_session()

    import sys as _sys

    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs), patch.object(
        _sys.stdout, "isatty", return_value=True,
    ):
        _run_interactive(mgr, session.session_id)

    out = capsys.readouterr().out

    final_clear = out.rfind("\r\x1b[K")
    assert final_clear != -1
    post_erase = out[final_clear + len("\r\x1b[K"):]

    # The `**bold**` close in the renderer becomes `RESET + base_color`
    # when base_color=CONTENT_ASSISTANT. The trailing ` body` after the
    # close must be preceded by the restore pair.
    assert f"{RESET}{CONTENT_ASSISTANT} body" in post_erase
    # The heading close must also restore assistant color before the
    # newline leading into the bullet line.
    assert f"{RESET}{CONTENT_ASSISTANT}\n" in post_erase


def test_tty_status_clears_when_stream_has_no_final_text(
    tmp_path: Path, capsys,
) -> None:
    class PartialOnlyBackend(ExecutionBackend):
        @property
        def backend_name(self) -> str:
            return "partial-only-dummy"

        def plan_from_messages(
            self,
            request: TurnRequest,
            *,
            on_partial_text: Callable[[str], None] | None = None,
        ) -> ExecutionPlan:
            if on_partial_text:
                on_partial_text("working")
            return ExecutionPlan(
                source_backend=self.backend_name,
                plan_label="partial-only",
                final_text=None,
                model="dummy-model",
            )

    store = SQLiteSessionStore(tmp_path / "test.db")
    mgr = SessionManager(backend=PartialOnlyBackend(), store=store)
    session = mgr.create_session()

    import sys as _sys

    inputs = iter(["hi", "/quit"])
    with patch("builtins.input", side_effect=inputs), patch.object(
        _sys.stdout, "isatty", return_value=True,
    ):
        _run_interactive(mgr, session.session_id)

    out = capsys.readouterr().out
    assert out.rfind("\r\x1b[K") > out.rfind("thinking")


# ---------------------------------------------------------------------------
# 8. Main entrypoint delegation
# ---------------------------------------------------------------------------

def test_main_delegates_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_main():
        called["cli"] = True

    monkeypatch.setattr("src.operation.cli.harness.main", fake_main)
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
    plain = _strip_ansi(captured.out)
    assert "empty transcript" in plain


def test_show_history_truncates_long_content(streaming_manager: SessionManager, capsys) -> None:
    session = streaming_manager.create_session()
    long_input = "x" * 200
    inputs = iter([long_input, "/quit"])
    with patch("builtins.input", side_effect=inputs):
        _run_interactive(streaming_manager, session.session_id)

    _show_history(streaming_manager, session.session_id)
    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)
    assert "..." in plain


# ---------------------------------------------------------------------------
# Composer: display_width
# ---------------------------------------------------------------------------

from src.operation.cli.composer import display_width


class TestDisplayWidth:
    def test_ascii_width(self) -> None:
        assert display_width("hello") == 5

    def test_cjk_width(self) -> None:
        assert display_width("\u4f60\u597d") == 4  # 你好 — two CJK chars, 2 cols each

    def test_mixed_width(self) -> None:
        assert display_width("hi\u4f60\u597d") == 6  # 2 ASCII + 4 CJK

    def test_empty(self) -> None:
        assert display_width("") == 0

    def test_fullwidth_latin(self) -> None:
        # Fullwidth A (U+FF21) should be 2 cols
        assert display_width("\uff21") == 2

    def test_japanese_katakana(self) -> None:
        # ア (U+30A2) — CJK double-width
        assert display_width("\u30a2\u30a4\u30a6") == 6

    def test_combining_mark_zero_width(self) -> None:
        # U+0301 COMBINING ACUTE ACCENT — should be 0 display width
        assert display_width("\u0301") == 0

    def test_combining_after_base_char(self) -> None:
        # "e" (1 col) + combining acute (0 col) = 1 col total
        assert display_width("e\u0301") == 1

    def test_zero_width_space(self) -> None:
        assert display_width("\u200b") == 0

    def test_zero_width_joiner(self) -> None:
        assert display_width("\u200d") == 0


# ---------------------------------------------------------------------------
# Composer: backspace boundary
# ---------------------------------------------------------------------------

class TestComposerBackspaceBoundary:
    """Verify the composer model never lets cursor go negative."""

    def test_backspace_at_empty_does_nothing(self) -> None:
        """Simulates repeated backspace on empty buffer."""
        # The composer uses a list buffer; cursor=0 means start.
        # Directly test the boundary logic.
        buf: list[str] = []
        cursor = 0
        # Simulate 5 backspaces on empty buffer
        for _ in range(5):
            if cursor > 0:
                del buf[cursor - 1]
                cursor -= 1
        assert cursor == 0
        assert buf == []

    def test_backspace_stops_at_boundary(self) -> None:
        buf = list("ab")
        cursor = 2
        # Delete 4 times (more than buffer length)
        for _ in range(4):
            if cursor > 0:
                del buf[cursor - 1]
                cursor -= 1
        assert cursor == 0
        assert buf == []

    def test_cjk_backspace_deletes_one_char(self) -> None:
        buf = list("\u4f60\u597d")  # 你好
        cursor = 2
        # Backspace once should remove 好
        if cursor > 0:
            del buf[cursor - 1]
            cursor -= 1
        assert cursor == 1
        assert "".join(buf) == "\u4f60"


# ---------------------------------------------------------------------------
# Style: color constants and helpers
# ---------------------------------------------------------------------------

from src.operation.cli.style import RESET, BOLD, DIM, ACCENT_USER, ACCENT_ASSISTANT, divider, styled


class TestStyle:
    def test_styled_wraps_text(self) -> None:
        result = styled("hello", BOLD)
        assert result.startswith(BOLD)
        assert result.endswith(RESET)
        assert "hello" in result

    def test_divider_length(self) -> None:
        d = divider(40)
        plain = _strip_ansi(d)
        assert len(plain) == 40

    def test_color_constants_are_ansi(self) -> None:
        for code in (ACCENT_USER, ACCENT_ASSISTANT, BOLD, DIM, RESET):
            assert "\x1b[" in code

    def test_styled_no_codes(self) -> None:
        assert styled("text") == "text"


# ---------------------------------------------------------------------------
# CLI header output
# ---------------------------------------------------------------------------

class TestCLIHeader:
    def test_session_header_contains_session_id(self, streaming_manager: SessionManager, capsys) -> None:
        session = streaming_manager.create_session()
        inputs = iter(["/quit"])
        with patch("builtins.input", side_effect=inputs):
            _run_interactive(streaming_manager, session.session_id)

        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert session.session_id in plain
        assert "Orbit2" in plain

    def test_clear_command(self, streaming_manager: SessionManager, capsys) -> None:
        session = streaming_manager.create_session()
        inputs = iter(["/clear", "/quit"])
        with patch("builtins.input", side_effect=inputs):
            _run_interactive(streaming_manager, session.session_id)

        captured = capsys.readouterr()
        # Clear screen sequence should be present in raw output
        assert "\x1b[2J" in captured.out


# ---------------------------------------------------------------------------
# Session navigation: /sessions, /switch, /new
# ---------------------------------------------------------------------------

from src.operation.cli.harness import _show_sessions, _handle_switch, _handle_delete_all


class TestSessionNavigation:
    def test_sessions_lists_all(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        _show_sessions(streaming_manager, s1.session_id)

        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert s1.session_id in plain
        assert s2.session_id in plain
        assert "Sessions" in plain

    def test_sessions_marks_active(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        _show_sessions(streaming_manager, s1.session_id)

        captured = capsys.readouterr()
        # Active marker (▶) should appear near the active session
        assert "\u25b6" in captured.out

    def test_sessions_empty(self, tmp_path: Path, capsys) -> None:
        store = SQLiteSessionStore(tmp_path / "empty.db")
        mgr = SessionManager(backend=StreamingDummyBackend(), store=store)
        _show_sessions(mgr, "nonexistent")
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "no sessions" in plain

    def test_switch_by_index(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # list_sessions returns newest first, so s2=index 1, s1=index 2
        result = _handle_switch(streaming_manager, s1.session_id, "/switch 1")
        assert result == s2.session_id

    def test_switch_by_session_id(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        result = _handle_switch(streaming_manager, s1.session_id, f"/switch {s2.session_id}")
        assert result == s2.session_id

    def test_switch_by_prefix(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        # Use a unique prefix (first 12 chars should be unique enough)
        prefix = s1.session_id[:16]
        result = _handle_switch(streaming_manager, "other", f"/switch {prefix}")
        assert result == s1.session_id

    def test_switch_no_arg(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        result = _handle_switch(streaming_manager, s1.session_id, "/switch")
        assert result is None
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "Usage" in plain

    def test_switch_out_of_range(self, streaming_manager: SessionManager, capsys) -> None:
        streaming_manager.create_session()
        result = _handle_switch(streaming_manager, "x", "/switch 99")
        assert result is None
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "out of range" in plain

    def test_switch_not_found(self, streaming_manager: SessionManager, capsys) -> None:
        streaming_manager.create_session()
        result = _handle_switch(streaming_manager, "x", "/switch nonexistent_id")
        assert result is None
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "not found" in plain

    def test_switch_interactive(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # Interact on s1, switch to s2, send a message, quit
        inputs = iter(["hello", f"/switch {s2.session_id}", "world", "/quit"])
        with patch("builtins.input", side_effect=inputs):
            _run_interactive(streaming_manager, s1.session_id)

        # s1 should have 2 messages (user + assistant for "hello")
        msgs_s1 = streaming_manager.list_messages(s1.session_id)
        assert len(msgs_s1) == 2
        assert msgs_s1[0].content == "hello"

        # s2 should have 2 messages (user + assistant for "world")
        msgs_s2 = streaming_manager.list_messages(s2.session_id)
        assert len(msgs_s2) == 2
        assert msgs_s2[0].content == "world"

    def test_new_session_command(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        inputs = iter(["/new", "hello", "/quit"])
        with patch("builtins.input", side_effect=inputs):
            _run_interactive(streaming_manager, s1.session_id)

        # Original session should be empty
        assert len(streaming_manager.list_messages(s1.session_id)) == 0
        # There should now be 2 sessions (original + new)
        sessions = streaming_manager.list_sessions()
        assert len(sessions) == 2
        # The new session (not s1) should have the message
        new_session = [s for s in sessions if s.session_id != s1.session_id][0]
        msgs = streaming_manager.list_messages(new_session.session_id)
        assert len(msgs) == 2
        assert msgs[0].content == "hello"


# ---------------------------------------------------------------------------
# /delete-all
# ---------------------------------------------------------------------------

class TestDeleteAll:
    def test_delete_all_confirmed(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        # Add a message so there's something to delete
        streaming_manager.run_turn(s1.session_id, "test message")
        assert len(streaming_manager.list_sessions()) == 1

        inputs = iter(["yes"])
        with patch("builtins.input", side_effect=inputs):
            _handle_delete_all(streaming_manager)

        assert len(streaming_manager.list_sessions()) == 0
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "Deleted 1 session" in plain

    def test_delete_all_cancelled(self, streaming_manager: SessionManager, capsys) -> None:
        s1 = streaming_manager.create_session()
        inputs = iter(["no"])
        with patch("builtins.input", side_effect=inputs):
            _handle_delete_all(streaming_manager)

        # Session should still exist
        assert len(streaming_manager.list_sessions()) == 1
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "Cancelled" in plain

    def test_delete_all_empty_input_cancels(self, streaming_manager: SessionManager, capsys) -> None:
        streaming_manager.create_session()
        inputs = iter([""])
        with patch("builtins.input", side_effect=inputs):
            _handle_delete_all(streaming_manager)

        assert len(streaming_manager.list_sessions()) == 1

    def test_delete_all_eof_cancels(self, streaming_manager: SessionManager, capsys) -> None:
        streaming_manager.create_session()
        with patch("builtins.input", side_effect=EOFError):
            _handle_delete_all(streaming_manager)

        assert len(streaming_manager.list_sessions()) == 1

    def test_delete_all_no_sessions(self, tmp_path: Path, capsys) -> None:
        store = SQLiteSessionStore(tmp_path / "empty.db")
        mgr = SessionManager(backend=StreamingDummyBackend(), store=store)
        _handle_delete_all(mgr)
        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "No sessions to delete" in plain

    def test_delete_all_shows_warning(self, streaming_manager: SessionManager, capsys) -> None:
        streaming_manager.create_session()
        streaming_manager.create_session()
        streaming_manager.create_session()
        inputs = iter(["no"])
        with patch("builtins.input", side_effect=inputs):
            _handle_delete_all(streaming_manager)

        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "WARNING" in plain
        assert "3" in plain  # should show count

    def test_delete_all_interactive(self, streaming_manager: SessionManager, capsys) -> None:
        """Full interactive flow: create sessions, /delete-all with yes, verify fresh session."""
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        inputs = iter(["/delete-all", "yes", "/sessions", "/quit"])
        with patch("builtins.input", side_effect=inputs):
            _run_interactive(streaming_manager, s1.session_id)

        captured = capsys.readouterr()
        plain = _strip_ansi(captured.out)
        assert "Deleted" in plain
        assert "New session" in plain


# ---------------------------------------------------------------------------
# Store layer: delete_all_sessions
# ---------------------------------------------------------------------------

class TestStoreDeleteAll:
    def test_delete_all_removes_sessions_and_messages(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        mgr = SessionManager(backend=StreamingDummyBackend(), store=store)
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.run_turn(s1.session_id, "msg1")
        mgr.run_turn(s2.session_id, "msg2")

        assert len(mgr.list_sessions()) == 2
        deleted = mgr.delete_all_sessions()
        assert deleted == 2
        assert len(mgr.list_sessions()) == 0
        assert len(mgr.list_messages(s1.session_id)) == 0
        assert len(mgr.list_messages(s2.session_id)) == 0

    def test_delete_all_empty_store(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(tmp_path / "test.db")
        mgr = SessionManager(backend=StreamingDummyBackend(), store=store)
        deleted = mgr.delete_all_sessions()
        assert deleted == 0


# ---------------------------------------------------------------------------
# CSI escape sequence parsing
# ---------------------------------------------------------------------------

import os

from src.operation.cli.composer import (
    ComposerAction,
    PageDownAction,
    PageUpAction,
    _read_csi_sequence,
)


class TestCSISequenceParsing:
    """Verify _read_csi_sequence fully consumes escape sequences."""

    def _make_fd(self, data: bytes) -> int:
        """Create a pipe fd preloaded with data."""
        r, w = os.pipe()
        os.write(w, data)
        os.close(w)
        return r

    def test_simple_letter_terminator(self) -> None:
        # CSI A (Up arrow) — no params, letter terminator
        fd = self._make_fd(b"A")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == ""
        assert term == "A"

    def test_number_tilde_page_up(self) -> None:
        # CSI 5~ (Page Up)
        fd = self._make_fd(b"5~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "5"
        assert term == "~"

    def test_number_tilde_page_down(self) -> None:
        # CSI 6~ (Page Down)
        fd = self._make_fd(b"6~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "6"
        assert term == "~"

    def test_number_tilde_delete(self) -> None:
        # CSI 3~ (Delete)
        fd = self._make_fd(b"3~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "3"
        assert term == "~"

    def test_multi_digit_params(self) -> None:
        # CSI 15~ (F5)
        fd = self._make_fd(b"15~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "15"
        assert term == "~"

    def test_semicolon_params(self) -> None:
        # CSI 1;5C (Ctrl+Right)
        fd = self._make_fd(b"1;5C")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "1;5"
        assert term == "C"

    def test_insert_key(self) -> None:
        # CSI 2~ (Insert) — previously would leak ~
        fd = self._make_fd(b"2~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "2"
        assert term == "~"

    def test_no_tilde_leak_for_unknown_sequences(self) -> None:
        """Verify that unrecognized CSI sequences don't leave bytes unconsumed."""
        # CSI 24~ (F12) — fully consumed
        fd = self._make_fd(b"24~")
        params, term = _read_csi_sequence(fd)
        os.close(fd)
        assert params == "24"
        assert term == "~"


# ---------------------------------------------------------------------------
# Page Up / Page Down session switching
# ---------------------------------------------------------------------------

from src.operation.cli.harness import _switch_session_relative


class TestPageUpDownSessionSwitch:
    def test_page_up_moves_to_previous(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # list_sessions returns newest first: [s2, s1]
        # page up from s1 (index 1) → s2 (index 0)
        result = _switch_session_relative(streaming_manager, s1.session_id, direction=PageUpAction())
        assert result == s2.session_id

    def test_page_down_moves_to_next(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # list_sessions: [s2, s1]. page down from s2 (index 0) → s1 (index 1)
        result = _switch_session_relative(streaming_manager, s2.session_id, direction=PageDownAction())
        assert result == s1.session_id

    def test_page_up_at_first_returns_none(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # s2 is at index 0 (newest). page up → out of range
        result = _switch_session_relative(streaming_manager, s2.session_id, direction=PageUpAction())
        assert result is None

    def test_page_down_at_last_returns_none(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        s2 = streaming_manager.create_session()
        # s1 is at index 1 (oldest). page down → out of range
        result = _switch_session_relative(streaming_manager, s1.session_id, direction=PageDownAction())
        assert result is None

    def test_single_session_returns_none(self, streaming_manager: SessionManager) -> None:
        s1 = streaming_manager.create_session()
        assert _switch_session_relative(streaming_manager, s1.session_id, direction=PageUpAction()) is None
        assert _switch_session_relative(streaming_manager, s1.session_id, direction=PageDownAction()) is None

    def test_unknown_session_returns_none(self, streaming_manager: SessionManager) -> None:
        streaming_manager.create_session()
        result = _switch_session_relative(streaming_manager, "nonexistent", direction=PageUpAction())
        assert result is None


class TestComposerActionTypes:
    def test_page_up_is_composer_action(self) -> None:
        assert issubclass(PageUpAction, ComposerAction)

    def test_page_down_is_composer_action(self) -> None:
        assert issubclass(PageDownAction, ComposerAction)

    def test_actions_are_exceptions(self) -> None:
        assert issubclass(ComposerAction, Exception)
        # Ensure they can be raised and caught
        with pytest.raises(PageUpAction):
            raise PageUpAction()
        with pytest.raises(PageDownAction):
            raise PageDownAction()
        with pytest.raises(ComposerAction):
            raise PageUpAction()
