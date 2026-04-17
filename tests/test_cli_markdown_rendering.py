"""Tests for `src/operation/cli/markdown.py::render_markdown_for_terminal`."""

from __future__ import annotations

import re

from src.operation.cli.markdown import render_markdown_for_terminal


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


class TestFencedCodeBlocks:
    def test_fenced_block_is_rendered_and_content_preserved(self) -> None:
        src = "before\n```python\ndef f():\n    return 1\n```\nafter"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "def f():" in stripped
        assert "return 1" in stripped
        assert "before" in stripped
        assert "after" in stripped

    def test_fenced_block_gets_an_ansi_frame_header(self) -> None:
        src = "```\nhello\n```"
        out = render_markdown_for_terminal(src)
        assert "\x1b[" in out  # ANSI codes present
        stripped = _strip_ansi(out)
        assert "hello" in stripped
        assert "code" in stripped  # header label

    def test_fenced_block_language_label_shown(self) -> None:
        src = "```python\nx = 1\n```"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "(python)" in stripped

    def test_fenced_block_without_language_has_no_lang_label(self) -> None:
        src = "```\nhello\n```"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        # No "()" label when no language given
        assert "()" not in stripped

    def test_multiple_fenced_blocks(self) -> None:
        src = "```\nfirst\n```\nmiddle\n```\nsecond\n```"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "first" in stripped
        assert "second" in stripped
        assert "middle" in stripped


class TestInlineFormatting:
    def test_inline_code_is_styled(self) -> None:
        src = "call `run()` first"
        out = render_markdown_for_terminal(src)
        assert "\x1b[" in out
        stripped = _strip_ansi(out)
        assert "run()" in stripped
        # backticks dropped after rendering
        assert "`run()`" not in stripped

    def test_bold_is_styled(self) -> None:
        src = "this is **important** text"
        out = render_markdown_for_terminal(src)
        assert "\x1b[" in out
        stripped = _strip_ansi(out)
        assert "important" in stripped
        assert "**important**" not in stripped

    def test_plain_text_unchanged(self) -> None:
        src = "no markdown here at all."
        out = render_markdown_for_terminal(src)
        assert out == "no markdown here at all."

    def test_empty_string_passes_through(self) -> None:
        assert render_markdown_for_terminal("") == ""


class TestBoundedness:
    def test_does_not_touch_italics(self) -> None:
        # First-slice scope excludes italics; asterisks around single words
        # without pairs should pass through unmodified.
        src = "one *word* here"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "*word*" in stripped

    def test_does_not_touch_headings(self) -> None:
        src = "# Big heading\nbody"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "# Big heading" in stripped

    def test_does_not_touch_links(self) -> None:
        src = "[name](https://example.com)"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "[name](https://example.com)" in stripped
