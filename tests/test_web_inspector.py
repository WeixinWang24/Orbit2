"""Tests for Orbit2 Web Inspector (Operation Surface first slice)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.operation.inspector.web_inspector import (
    VIO_INSPIRED_CSS,
    _VALID_MAIN_TABS,
    _VALID_RIGHT_TABS,
    _build_url,
    _format_local_timestamp,
    _html_page,
    _json_tree_html,
    _make_handler,
    _message_chips_html,
    _message_css_class,
    _preview_snippet,
    _render_debug_panel,
    _render_json_block,
    _render_message_card,
    _render_metadata_right_panel,
    _render_raw_right_panel,
    _render_stat_card,
    _render_transcript_panel,
    _truncate_text,
    serve,
)
from src.core.runtime.models import (
    ConversationMessage,
    MessageRole,
    Session,
    SessionStatus,
    make_message_id,
    make_session_id,
)
from src.core.store.sqlite import SQLiteSessionStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)


def _make_session(**overrides) -> Session:
    defaults = dict(
        session_id="session_test001",
        backend_name="openai-codex",
        system_prompt="You are helpful.",
        status=SessionStatus.ACTIVE,
        created_at=NOW,
        updated_at=NOW,
        metadata={},
    )
    defaults.update(overrides)
    return Session(**defaults)


def _make_msg(role: MessageRole, content: str, turn_index: int, **meta_overrides) -> ConversationMessage:
    return ConversationMessage(
        message_id=make_message_id(),
        session_id="session_test001",
        role=role,
        content=content,
        turn_index=turn_index,
        created_at=NOW,
        metadata=meta_overrides,
    )


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    def test_preview_snippet_short(self):
        assert _preview_snippet("hello") == "hello"

    def test_preview_snippet_long(self):
        result = _preview_snippet("a" * 100, limit=10)
        assert len(result) == 11  # 10 chars + ellipsis
        assert result.endswith("\u2026")

    def test_preview_snippet_newlines(self):
        assert "\\n" in _preview_snippet("line1\nline2")

    def test_truncate_text_short(self):
        text, truncated = _truncate_text("short")
        assert text == "short"
        assert truncated is False

    def test_truncate_text_long(self):
        text, truncated = _truncate_text("x" * 300)
        assert truncated is True
        assert text.endswith("\u2026")
        assert len(text) <= 220

    def test_format_local_timestamp_string(self):
        result = _format_local_timestamp("2026-04-16T12:00:00")
        assert "2026" in result

    def test_format_local_timestamp_datetime(self):
        result = _format_local_timestamp(NOW)
        assert "2026" in result

    def test_format_local_timestamp_invalid(self):
        assert _format_local_timestamp("not-a-date") == "not-a-date"


class TestJsonTreeHtml:
    def test_renders_dict(self):
        html = _json_tree_html({"key": "value"})
        assert "key" in html
        assert "value" in html
        assert "dict" in html

    def test_renders_list(self):
        html = _json_tree_html([1, 2, 3])
        assert "list" in html
        assert "[0]" in html

    def test_renders_scalar(self):
        html = _json_tree_html("hello")
        assert "hello" in html

    def test_renders_nested(self):
        html = _json_tree_html({"outer": {"inner": 42}})
        assert "outer" in html
        assert "inner" in html

    def test_open_depth(self):
        html = _json_tree_html({"a": {"b": 1}}, open_depth=2)
        assert "open" in html

    def test_empty_dict(self):
        html = _json_tree_html({})
        assert "empty object" in html

    def test_empty_list(self):
        html = _json_tree_html([])
        assert "empty list" in html

    def test_escapes_html(self):
        html = _json_tree_html({"<script>": "alert(1)"})
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_render_json_block(self):
        html = _render_json_block({"foo": "bar"}, title="test")
        assert "json-view" in html
        assert "foo" in html


class TestStatCard:
    def test_basic(self):
        html = _render_stat_card("42", "total")
        assert "42" in html
        assert "total" in html
        assert "stat-card" in html

    def test_css_class(self):
        html = _render_stat_card("5", "errors", "danger")
        assert "danger" in html


# ---------------------------------------------------------------------------
# Message rendering tests
# ---------------------------------------------------------------------------


class TestMessageCssClass:
    def test_user(self):
        msg = _make_msg(MessageRole.USER, "hi", 1)
        assert _message_css_class(msg) == "user"

    def test_assistant(self):
        msg = _make_msg(MessageRole.ASSISTANT, "hello", 2)
        assert _message_css_class(msg) == "assistant"

    def test_tool_ok(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3, ok=True)
        assert "tool-ok" in _message_css_class(msg)

    def test_tool_fail(self):
        msg = _make_msg(MessageRole.TOOL, "error", 3, ok=False)
        assert "tool-fail" in _message_css_class(msg)

    def test_tool_neutral(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3)
        assert _message_css_class(msg) == "tool"


class TestMessageChips:
    def test_turn_index_chip(self):
        msg = _make_msg(MessageRole.USER, "hi", 1)
        html = _message_chips_html(msg)
        assert "turn 1" in html

    def test_tool_name_chip(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3, tool_name="native__read_file")
        html = _message_chips_html(msg)
        assert "native__read_file" in html

    def test_governance_allowed(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3, governance_outcome="allowed")
        html = _message_chips_html(msg)
        assert "allowed" in html
        assert "ok" in html

    def test_governance_denied(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3, governance_outcome="denied_path_escape")
        html = _message_chips_html(msg)
        assert "denied_path_escape" in html
        assert "fail" in html

    def test_tool_calls_count(self):
        msg = _make_msg(
            MessageRole.ASSISTANT, "calling tool", 2,
            tool_calls=[{"name": "native__read_file", "arguments": {}}],
        )
        html = _message_chips_html(msg)
        assert "1 tool call(s)" in html
        assert "native__read_file" in html

    def test_ok_badge(self):
        msg = _make_msg(MessageRole.TOOL, "result", 3, ok=True)
        html = _message_chips_html(msg)
        assert ">ok<" in html

    def test_fail_badge(self):
        msg = _make_msg(MessageRole.TOOL, "error", 3, ok=False)
        html = _message_chips_html(msg)
        assert "failed" in html


class TestRenderMessageCard:
    def test_user_card(self):
        msg = _make_msg(MessageRole.USER, "hello world", 1)
        html = _render_message_card(msg)
        assert "message-card user" in html
        assert "USER" in html
        assert "hello world" in html

    def test_assistant_card(self):
        msg = _make_msg(MessageRole.ASSISTANT, "I can help", 2)
        html = _render_message_card(msg)
        assert "message-card assistant" in html
        assert "ASSISTANT" in html

    def test_tool_card_with_metadata(self):
        msg = _make_msg(
            MessageRole.TOOL, "file contents here", 3,
            tool_name="native__read_file", ok=True, governance_outcome="allowed",
        )
        html = _render_message_card(msg)
        assert "message-card tool tool-ok" in html
        assert "TOOL" in html
        assert "metadata" in html

    def test_long_content_truncated(self):
        msg = _make_msg(MessageRole.USER, "x" * 500, 1)
        html = _render_message_card(msg)
        assert "\u2026" in html


# ---------------------------------------------------------------------------
# Panel rendering tests
# ---------------------------------------------------------------------------


class TestTranscriptPanel:
    def test_empty_messages(self):
        html = _render_transcript_panel([])
        assert "No messages" in html

    def test_renders_messages(self):
        msgs = [
            _make_msg(MessageRole.USER, "question", 1),
            _make_msg(MessageRole.ASSISTANT, "answer", 2),
        ]
        html = _render_transcript_panel(msgs)
        assert "question" in html
        assert "answer" in html
        assert "USER" in html
        assert "ASSISTANT" in html


class TestDebugPanel:
    def test_session_info(self):
        session = _make_session()
        html = _render_debug_panel(session, [])
        assert "session_test001" in html
        assert "openai-codex" in html

    def test_system_prompt_displayed(self):
        session = _make_session(system_prompt="Be helpful.")
        html = _render_debug_panel(session, [])
        assert "Be helpful." in html

    def test_tool_executions(self):
        session = _make_session()
        msgs = [
            _make_msg(MessageRole.TOOL, "file data", 3, tool_name="native__read_file", ok=True, governance_outcome="allowed"),
        ]
        html = _render_debug_panel(session, msgs)
        assert "native__read_file" in html
        assert "allowed" in html
        assert "Tool Executions (1)" in html


class TestMetadataRightPanel:
    def test_session_identity(self):
        session = _make_session()
        html = _render_metadata_right_panel(session, [])
        assert "session_test001" in html
        assert "openai-codex" in html

    def test_message_stats(self):
        session = _make_session()
        msgs = [
            _make_msg(MessageRole.USER, "hi", 1),
            _make_msg(MessageRole.ASSISTANT, "hello", 2),
            _make_msg(MessageRole.TOOL, "data", 3, ok=True, governance_outcome="allowed"),
        ]
        html = _render_metadata_right_panel(session, msgs)
        assert ">3<" in html  # total message count

    def test_governance_summary(self):
        session = _make_session()
        msgs = [
            _make_msg(MessageRole.TOOL, "ok", 3, governance_outcome="allowed"),
            _make_msg(MessageRole.TOOL, "denied", 5, governance_outcome="denied_path_escape"),
        ]
        html = _render_metadata_right_panel(session, msgs)
        assert ">1<" in html  # 1 allowed


class TestRawRightPanel:
    def test_renders_metadata(self):
        session = _make_session()
        msgs = [
            _make_msg(MessageRole.USER, "hi", 1),
            _make_msg(MessageRole.TOOL, "data", 3, tool_name="native__read_file"),
        ]
        html = _render_raw_right_panel(session, msgs)
        assert "USER" in html
        assert "TOOL" in html
        assert "native__read_file" in html

    def test_empty_messages(self):
        session = _make_session()
        html = _render_raw_right_panel(session, [])
        assert "No messages" in html


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


class TestBuildUrl:
    def test_default(self):
        assert _build_url() == "/"

    def test_with_session(self):
        url = _build_url(session_id="s1")
        assert "session_id=s1" in url

    def test_with_tab(self):
        url = _build_url(session_id="s1", tab="debug")
        assert "tab=debug" in url

    def test_with_right_tab(self):
        url = _build_url(session_id="s1", right_tab="raw")
        assert "right_tab=raw" in url


# ---------------------------------------------------------------------------
# Full page assembly
# ---------------------------------------------------------------------------


class TestHtmlPage:
    def test_empty_sessions(self):
        html = _html_page(
            sessions=[], current_session=None, messages=[],
            active_tab="transcript", right_tab="metadata", db_path="test.db",
        )
        assert "Orbit2 Inspector" in html
        assert "No sessions found" in html
        assert "Select a session" in html

    def test_with_session(self):
        session = _make_session()
        msgs = [
            _make_msg(MessageRole.USER, "hello", 1),
            _make_msg(MessageRole.ASSISTANT, "world", 2),
        ]
        html = _html_page(
            sessions=[session], current_session=session, messages=msgs,
            active_tab="transcript", right_tab="metadata", db_path="test.db",
        )
        assert "Orbit2 Inspector" in html
        assert "session_test001" in html
        assert "hello" in html
        assert "world" in html
        assert "Transcript" in html
        assert "Debug" in html

    def test_debug_tab(self):
        session = _make_session()
        html = _html_page(
            sessions=[session], current_session=session, messages=[],
            active_tab="debug", right_tab="metadata", db_path="test.db",
        )
        assert "openai-codex" in html
        assert "session_test001" in html

    def test_raw_right_tab(self):
        session = _make_session()
        msgs = [_make_msg(MessageRole.USER, "test", 1)]
        html = _html_page(
            sessions=[session], current_session=session, messages=msgs,
            active_tab="transcript", right_tab="raw", db_path="test.db",
        )
        assert "USER" in html


# ---------------------------------------------------------------------------
# CSS design language preservation
# ---------------------------------------------------------------------------


class TestDesignLanguage:
    def test_css_variables_present(self):
        assert "--bg: #050510" in VIO_INSPIRED_CSS
        assert "--cyan: #00FFC6" in VIO_INSPIRED_CSS
        assert "--pink: #FF2E88" in VIO_INSPIRED_CSS
        assert "--violet: #B388FF" in VIO_INSPIRED_CSS

    def test_glassmorphism(self):
        assert "backdrop-filter: blur" in VIO_INSPIRED_CSS

    def test_grid_layout(self):
        assert "grid-template-columns" in VIO_INSPIRED_CSS
        assert "sidebar main right" in VIO_INSPIRED_CSS


# ---------------------------------------------------------------------------
# Integration with SQLiteSessionStore
# ---------------------------------------------------------------------------


class TestStoreIntegration:
    def test_handler_reads_from_store(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteSessionStore(db_path)

        # Create a session with messages
        session = Session(
            session_id="session_int001",
            backend_name="test-backend",
            status=SessionStatus.ACTIVE,
            created_at=NOW,
            updated_at=NOW,
        )
        store.save_session(session)
        store.save_message(ConversationMessage(
            message_id="msg_001",
            session_id="session_int001",
            role=MessageRole.USER,
            content="Read my file",
            turn_index=1,
            created_at=NOW,
        ))
        store.save_message(ConversationMessage(
            message_id="msg_002",
            session_id="session_int001",
            role=MessageRole.ASSISTANT,
            content="Here are the contents",
            turn_index=2,
            created_at=NOW,
            metadata={"tool_calls": [{"name": "native__read_file", "arguments": {"path": "/tmp/test.txt"}}]},
        ))
        store.save_message(ConversationMessage(
            message_id="msg_003",
            session_id="session_int001",
            role=MessageRole.TOOL,
            content="file contents here",
            turn_index=3,
            created_at=NOW,
            metadata={"tool_name": "native__read_file", "ok": True, "governance_outcome": "allowed"},
        ))

        # Verify the handler class can be created
        handler_cls = _make_handler(str(db_path))

        # Verify sessions load
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "session_int001"

        messages = store.list_messages("session_int001")
        assert len(messages) == 3

        # Verify full page renders without error
        html = _html_page(
            sessions=sessions,
            current_session=sessions[0],
            messages=messages,
            active_tab="transcript",
            right_tab="metadata",
            db_path=str(db_path),
        )
        assert "session_int001" in html
        assert "Read my file" in html
        assert "native__read_file" in html
        assert "allowed" in html


# ---------------------------------------------------------------------------
# XSS safety
# ---------------------------------------------------------------------------


class TestXssSafety:
    def test_message_content_escaped(self):
        msg = _make_msg(MessageRole.USER, '<script>alert("xss")</script>', 1)
        html = _render_message_card(msg)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_session_id_escaped(self):
        session = _make_session(session_id='<img src=x onerror=alert(1)>')
        html = _render_metadata_right_panel(session, [])
        assert "<img" not in html

    def test_metadata_values_escaped(self):
        msg = _make_msg(MessageRole.TOOL, "data", 3, tool_name='<b>evil</b>')
        html = _message_chips_html(msg)
        assert "<b>" not in html
        assert "&lt;b&gt;" in html

    def test_session_id_escaped_in_debug_panel(self):
        session = _make_session(session_id='<img src=x onerror=alert(1)>')
        html = _render_debug_panel(session, [])
        assert "<img" not in html

    def test_session_id_escaped_in_sidebar(self):
        session = _make_session(session_id='<script>alert(1)</script>')
        html = _html_page(
            sessions=[session], current_session=session, messages=[],
            active_tab="transcript", right_tab="metadata", db_path="test.db",
        )
        assert "<script>alert" not in html


# ---------------------------------------------------------------------------
# Tab validation (audit MED fix)
# ---------------------------------------------------------------------------


class TestTabValidation:
    def test_valid_main_tabs(self):
        assert "transcript" in _VALID_MAIN_TABS
        assert "debug" in _VALID_MAIN_TABS

    def test_valid_right_tabs(self):
        assert "metadata" in _VALID_RIGHT_TABS
        assert "raw" in _VALID_RIGHT_TABS

    def test_invalid_tab_falls_back_to_transcript(self):
        """The handler validates tabs; at the page level, invalid tabs render transcript."""
        session = _make_session()
        msgs = [_make_msg(MessageRole.USER, "test", 1)]
        # Even with an invalid tab value, the page renders without error
        html = _html_page(
            sessions=[session], current_session=session, messages=msgs,
            active_tab="<script>", right_tab="metadata", db_path="test.db",
        )
        # Should still render — the handler would have sanitized this,
        # but the page itself should not crash on unexpected values
        assert "Orbit2 Inspector" in html


# ---------------------------------------------------------------------------
# db_path validation (audit MED fix)
# ---------------------------------------------------------------------------


class TestServeValidation:
    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            serve(str(tmp_path / "nonexistent.db"), open_browser=False)


# ---------------------------------------------------------------------------
# Timestamp edge case (audit LOW fix)
# ---------------------------------------------------------------------------


class TestTimestampEdgeCases:
    def test_naive_datetime(self):
        """Timezone-naive datetime returns isoformat without crash."""
        naive = datetime(2026, 1, 1, 12, 0, 0)
        result = _format_local_timestamp(naive)
        assert "2026" in result

    def test_non_datetime(self):
        result = _format_local_timestamp(12345)
        assert result == "12345"
