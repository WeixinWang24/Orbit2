"""Tests for `src/operation/cli/markdown.py::render_markdown_for_terminal`."""

from __future__ import annotations

import re

from src.operation.cli.markdown import (
    render_markdown_for_terminal,
    wrap_ansi_text_for_terminal,
)
from src.operation.cli.style import RESET


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


class TestAnsiAwareWrapping:
    def test_wraps_ascii_by_visible_width(self) -> None:
        assert wrap_ansi_text_for_terminal("abcdef", 4) == "abcd\nef"

    def test_wraps_cjk_by_visible_width(self) -> None:
        assert wrap_ansi_text_for_terminal("你好世界", 4) == "你好\n世界"

    def test_ansi_sequences_do_not_count_toward_width(self) -> None:
        src = "\x1b[95mabcd\x1b[0mef"
        out = wrap_ansi_text_for_terminal(src, 4)
        assert _strip_ansi(out) == "abcd\nef"
        assert "\x1b[95m" in out
        assert "\x1b[0m" in out

    def test_existing_newlines_reset_width(self) -> None:
        assert wrap_ansi_text_for_terminal("abc\ndefgh", 4) == "abc\ndefg\nh"


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

    def test_italic_is_styled(self) -> None:
        src = "this is *subtle* text"
        out = render_markdown_for_terminal(src)
        assert "\x1b[" in out
        stripped = _strip_ansi(out)
        assert "subtle" in stripped
        assert "*subtle*" not in stripped

    def test_bold_and_italic_coexist_without_double_rendering(self) -> None:
        # Bold runs first; its content gets ANSI-wrapped and no `*` survives
        # inside the bold span, so italic cannot double-render over it.
        src = "leading **bold phrase** then *italic phrase* tail"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "bold phrase" in stripped
        assert "italic phrase" in stripped
        assert "**" not in stripped
        # Italic markers consumed.
        assert "*italic phrase*" not in stripped

    def test_plain_text_unchanged(self) -> None:
        src = "no markdown here at all."
        out = render_markdown_for_terminal(src)
        assert out == "no markdown here at all."

    def test_empty_string_passes_through(self) -> None:
        assert render_markdown_for_terminal("") == ""


class TestBlockFormatting:
    def test_heading_marker_removed_and_body_styled(self) -> None:
        src = "### 当前已可用\nbody text"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "### 当前已可用" not in stripped
        assert "当前已可用" in stripped
        assert "body text" in stripped

    def test_heading_levels_all_render(self) -> None:
        for level in range(1, 7):
            prefix = "#" * level
            src = f"{prefix} heading{level}"
            out = render_markdown_for_terminal(src)
            stripped = _strip_ansi(out)
            assert f"heading{level}" in stripped
            assert prefix not in stripped

    def test_heading_marker_inside_prose_is_kept(self) -> None:
        # Mid-line `#` should not be treated as a heading.
        src = "see tag #123 for context"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "#123" in stripped

    def test_bullet_marker_is_styled_and_body_preserved(self) -> None:
        src = "- first item\n- second item"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "first item" in stripped
        assert "second item" in stripped
        assert "\u2022 first item" in stripped
        # Raw dash markers should no longer lead the items.
        assert "- first item" not in stripped

    def test_star_bullet_also_renders(self) -> None:
        src = "* alpha\n* beta"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "\u2022 alpha" in stripped
        assert "\u2022 beta" in stripped

    def test_bullet_preserves_leading_indent(self) -> None:
        src = "  - nested"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "  \u2022 nested" in stripped

    def test_mid_line_dash_is_not_a_bullet(self) -> None:
        src = "plain - dash in the middle"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "plain - dash in the middle" in stripped

    def test_excess_blank_lines_collapse(self) -> None:
        src = "first\n\n\n\n\nsecond"
        out = render_markdown_for_terminal(src)
        # Runs of 3+ newlines collapse down to exactly 2 (one blank line).
        assert out == "first\n\nsecond"

    def test_links_pass_through(self) -> None:
        # Links remain out of scope; the literal syntax should still show.
        src = "[name](https://example.com)"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "[name](https://example.com)" in stripped


class TestRealDebugSample:
    """Regression coverage anchored on the handoff-supplied CLI sample.

    The sample combines headings, bold bullet bodies, inline code, mixed
    CJK/English text, and wrapped continuation lines — i.e. the exact mix
    that broke the first-slice renderer. Raw `###` / `**` / `` ` `` tokens
    are expected to be gone after rendering.
    """

    SAMPLE = (
        "### 当前已可用\n"
        "- **list_available_tools**：查看/申请解锁工具组\n"
        "- **native__read_file**：读取工作区文本文件\n"
        "\n"
        "### 可解锁的 MCP / 扩展工具组\n"
        "- **mcp_diagnostics** \n"
        " 诊断工具：`pytest` / `ruff` / `mypy`\n"
        "- **mcp_fs_read** \n"
        " 文件读取：目录树、文件信息、glob、grep 等\n"
        "\n"
        "### 数量概览\n"
        "- **当前暴露工具数**：2\n"
        "- **隐藏但可解锁工具数**：33\n"
    )

    def test_raw_heading_markers_are_gone(self) -> None:
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        assert "### " not in stripped
        assert "#### " not in stripped

    def test_raw_bullet_markers_are_gone(self) -> None:
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        for line in stripped.splitlines():
            assert not line.startswith("- ")
            assert not line.startswith("* ")

    def test_raw_bold_markers_are_gone(self) -> None:
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        assert "**" not in stripped

    def test_raw_inline_code_backticks_are_gone(self) -> None:
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        assert "`pytest`" not in stripped
        assert "`ruff`" not in stripped

    def test_content_words_still_present(self) -> None:
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        for needle in (
            "当前已可用",
            "list_available_tools",
            "native__read_file",
            "mcp_diagnostics",
            "mcp_fs_read",
            "pytest",
            "ruff",
            "mypy",
            "当前暴露工具数",
            "隐藏但可解锁工具数",
        ):
            assert needle in stripped

    def test_output_contains_ansi_styling(self) -> None:
        out = render_markdown_for_terminal(self.SAMPLE)
        assert "\x1b[" in out
        assert "\u2022" in out  # bullet char rendered

    def test_cjk_continuation_line_is_preserved(self) -> None:
        # The description line under each bullet starts with a single space
        # in the model output; we keep it verbatim so it stays visually
        # attached to its bullet.
        stripped = _strip_ansi(render_markdown_for_terminal(self.SAMPLE))
        assert " 诊断工具：pytest / ruff / mypy" in stripped
        assert " 文件读取：目录树、文件信息、glob、grep 等" in stripped


class TestFencedCoexistence:
    def test_heading_outside_fenced_block_still_renders(self) -> None:
        src = "# outer\n```\n# not a heading\n```\n## tail"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        # Outer heading markers are stripped.
        assert "outer" in stripped
        assert stripped.splitlines()[0].lstrip().startswith("outer")
        # Fenced content keeps its literal `#` payload.
        assert "# not a heading" in stripped

    def test_bullet_inside_fenced_block_is_preserved_literally(self) -> None:
        src = "```\n- keep literal\n```"
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "- keep literal" in stripped

    def test_blockquote_triple_backtick_does_not_open_fence(self) -> None:
        # Regression: a model test response had `> ```python` on a line
        # (triple-backtick inside a blockquote). The old regex treated it
        # as a fence opening and swallowed every following heading/bullet
        # until the next bare triple-backtick anywhere in the text.
        src = (
            "> ```python\n"
            "> print(\"quote code\")\n"
            "> ```\n"
            "\n"
            "## real heading\n"
            "- real bullet\n"
            "\n"
            "```mermaid\n"
            "graph TD\n"
            "```\n"
        )
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        # The real heading and bullet must still render (not be absorbed
        # into a phantom fence).
        assert "real heading" in stripped
        assert "## real heading" not in stripped
        assert "\u2022 real bullet" in stripped
        # The true mermaid fence must still render with its lang tag.
        assert "(mermaid)" in stripped
        assert "graph TD" in stripped

    def test_fence_with_trailing_lang_on_closing_is_not_closing(self) -> None:
        # `\n```mermaid` mid-text is a NEW opening, not a closing of the
        # previous fence. The tightened closing pattern rejects it.
        src = (
            "```python\n"
            "x = 1\n"
            "```\n"
            "\n"
            "```mermaid\n"
            "graph\n"
            "```\n"
        )
        out = render_markdown_for_terminal(src)
        stripped = _strip_ansi(out)
        assert "(python)" in stripped
        assert "(mermaid)" in stripped
        assert "x = 1" in stripped
        assert "graph" in stripped


class TestBaseColorPreservation:
    """Regression: Handoff 30.

    When the rendered output is embedded inside an outer colored region
    (the CLI harness wraps the assistant body in `CONTENT_ASSISTANT …
    RESET`), every internal `RESET` that closes an inline span must
    also re-apply the outer color, otherwise ordinary body text
    following the first styled span drops to terminal default (the
    observed "assistant text reverts to white" bug).
    """

    BASE = "\x1b[95m"  # sentinel base color, stand-in for CONTENT_ASSISTANT

    def test_default_base_color_preserves_old_behavior(self) -> None:
        # Without an explicit base_color the renderer must keep emitting
        # a bare RESET — no stray sentinel bytes.
        src = "this is **important** text"
        out = render_markdown_for_terminal(src)
        assert self.BASE not in out

    def test_plain_text_never_injects_base_color(self) -> None:
        # No styled spans -> no internal RESET -> nothing to restore.
        src = "no markdown here at all."
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert out == src
        assert self.BASE not in out

    def test_bold_close_restores_base_color(self) -> None:
        src = "prefix **bold body** suffix"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"{RESET}{self.BASE} suffix" in out

    def test_italic_close_restores_base_color(self) -> None:
        src = "prefix *subtle* suffix"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"{RESET}{self.BASE} suffix" in out

    def test_inline_code_close_restores_base_color(self) -> None:
        src = "call `run()` first"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"{RESET}{self.BASE} first" in out

    def test_heading_close_restores_base_color_before_next_line(self) -> None:
        src = "### heading\nbody text"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"{RESET}{self.BASE}\nbody text" in out

    def test_bullet_close_restores_base_color_before_body(self) -> None:
        src = "- item body"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"\u2022{RESET}{self.BASE} item body" in out

    def test_mixed_markdown_trailing_plain_text_stays_in_base_color(self) -> None:
        # Canonical Handoff-30 failure shape: bullet + bold + trailing
        # Chinese prose. The trailing prose after the bold close must
        # be preceded by the base-color restore.
        src = "- **list_available_tools**：查看/申请解锁工具组"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        assert f"{RESET}{self.BASE}：查看/申请解锁工具组" in out

    def test_fenced_block_close_restores_base_color(self) -> None:
        # Between fenced blocks the outer color must be active too, so
        # surrounding prose is not painted terminal-default.
        src = "before\n```\nhello\n```\nafter"
        out = render_markdown_for_terminal(src, base_color=self.BASE)
        # Fenced body lines close with RESET+base; between the block
        # border and surrounding prose the base color is always restored
        # by every internal close.
        assert f"{RESET}{self.BASE}" in out
        # Stripping ANSI must still give the expected plain text.
        stripped = _strip_ansi(out)
        assert "before" in stripped
        assert "hello" in stripped
        assert "after" in stripped
