from __future__ import annotations

import re

from src.operation.cli.composer import display_width
from src.operation.cli.style import (
    ACCENT_MUTED,
    ACCENT_SUCCESS,
    ACCENT_SYSTEM,
    BOLD,
    CONTENT_CODE,
    CONTENT_INLINE_CODE,
    DIM,
    ITALIC,
    RESET,
)

# Fenced code block: triple backticks at a line start, optional language
# tag, body, and a closing triple-backtick line. Requiring both the
# opening and closing fences to sit alone at a line start stops
# blockquote prefixes (`> ```lang`) from being mistaken for openings and
# stops inline-with-lang fences (` ```mermaid`) from being mistaken for
# closings — the pathology that would otherwise let a single bad nested
# fence swallow surrounding headings and bullets.
_FENCED_RE = re.compile(
    r"^```([a-zA-Z0-9_+-]*)\n(.*?)\n```[ \t]*(?=\n|\Z)",
    re.DOTALL | re.MULTILINE,
)
# ATX-style heading: 1-6 leading `#`, at least one space, then the heading
# body. Matches only at line starts so `# tag` inside prose stays verbatim.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# Bullet item: optional leading indent, `-` or `*`, one or more spaces,
# then the body. Anchored to line starts so mid-line `- ` sequences and
# `*emphasis*` don't become accidental bullets.
_BULLET_RE = re.compile(r"^([ \t]*)[-*][ \t]+(.*)$", re.MULTILINE)
# Inline `code` — non-greedy, no newlines inside
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
# **bold** — non-greedy
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
# *italic* — single-asterisk pair around non-whitespace content, with
# negative lookbehind/ahead to avoid matching inside `**bold**`. Bold is
# always rendered first so live `**`-pairs are gone before italic sees
# the text; the lookarounds protect against unmatched stray asterisks.
_ITALIC_RE = re.compile(
    r"(?<![*\w])\*(\S(?:[^*\n]*?\S)?|\S)\*(?![*\w])",
)
# Collapse runs of 3+ newlines down to a single blank line so model
# output with doubled blank lines stops looking fragmented on the
# terminal. Two newlines (single blank line between paragraphs) are left
# untouched on purpose.
_BLANK_SQUASH_RE = re.compile(r"\n{3,}")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def render_markdown_for_terminal(
    text: str, *, indent: str = "    ", base_color: str = "",
) -> str:
    """Render a bounded subset of markdown for terminal display.

    Scope:
      - fenced code blocks (``` ... ```)
      - inline code (`foo`)
      - bold (**foo**)
      - italic (*foo*)
      - ATX headings (# through ######)
      - bullet lists (`-` / `*` at line start)
      - paragraph-spacing normalization (collapse 3+ blank lines)

    Out of scope: ordered lists, tables, links, blockquotes, setext
    headings, syntax highlighting.

    `indent` is applied to every fenced-code-block line so blocks stand
    out from surrounding prose in the transcript view.

    `base_color`: when the rendered output is embedded inside an outer
    colored region (e.g. the CLI harness wraps the rendered assistant
    body in `CONTENT_ASSISTANT … RESET`), a bare `RESET` at the close of
    each inline span would drop the outer color for the remainder of the
    line. Setting `base_color` makes every internal span close with
    `RESET + base_color` so the outer color is re-applied immediately
    and ordinary text following styled spans keeps the intended color.
    The default empty string preserves pre-existing behavior.
    """
    if not text:
        return text

    close = _close(base_color)
    rendered_parts: list[str] = []
    last_end = 0
    for match in _FENCED_RE.finditer(text):
        prose = text[last_end : match.start()]
        if prose:
            rendered_parts.append(_render_prose(prose, close=close))
        lang = match.group(1).strip()
        code_body = match.group(2)
        rendered_parts.append(
            _render_fenced(lang, code_body, indent=indent, close=close),
        )
        last_end = match.end()
    tail = text[last_end:]
    if tail:
        rendered_parts.append(_render_prose(tail, close=close))

    return "".join(rendered_parts)


def wrap_ansi_text_for_terminal(text: str, width: int) -> str:
    """Soft-wrap rendered terminal text by visible display width.

    ANSI SGR sequences are preserved and treated as zero-width. This avoids
    delegating long message wrapping to terminal auto-wrap, which is less
    predictable once colored spans and CJK double-width characters mix.
    """
    if width <= 0 or not text:
        return text

    parts: list[str] = []
    col = 0
    i = 0
    while i < len(text):
        ansi = _ANSI_RE.match(text, i)
        if ansi:
            parts.append(ansi.group(0))
            i = ansi.end()
            continue

        ch = text[i]
        if ch == "\n":
            parts.append(ch)
            col = 0
            i += 1
            continue

        w = display_width(ch)
        if w > 0 and col > 0 and col + w > width:
            parts.append("\n")
            col = 0
        parts.append(ch)
        col += w
        i += 1

    return "".join(parts)


def _close(base_color: str) -> str:
    return f"{RESET}{base_color}" if base_color else RESET


def _render_prose(text: str, *, close: str) -> str:
    text = _HEADING_RE.sub(lambda m: _render_heading_match(m, close), text)
    text = _BULLET_RE.sub(lambda m: _render_bullet_match(m, close), text)
    text = _BOLD_RE.sub(lambda m: f"{BOLD}{m.group(1)}{close}", text)
    text = _ITALIC_RE.sub(lambda m: f"{ITALIC}{m.group(1)}{close}", text)
    text = _INLINE_CODE_RE.sub(
        lambda m: f"{CONTENT_INLINE_CODE}{m.group(1)}{close}", text,
    )
    text = _BLANK_SQUASH_RE.sub("\n\n", text)
    return text


def _render_heading_match(match: re.Match[str], close: str) -> str:
    level = len(match.group(1))
    body = match.group(2).strip()
    if level <= 2:
        accent = ACCENT_SYSTEM
    elif level == 3:
        accent = ACCENT_SUCCESS
    else:
        accent = ACCENT_MUTED
    return f"{BOLD}{accent}{body}{close}"


def _render_bullet_match(match: re.Match[str], close: str) -> str:
    lead = match.group(1)
    body = match.group(2)
    return f"{lead}{ACCENT_MUTED}\u2022{close} {body}"


def _render_fenced(lang: str, body: str, *, indent: str, close: str) -> str:
    body = body.rstrip("\n")
    indented_lines = [
        f"{indent}{CONTENT_CODE}{line}{close}" for line in body.split("\n")
    ]
    indented_block = "\n".join(indented_lines)
    header = (
        f"{indent}{ACCENT_SYSTEM}\u250c{close}"
        f" {DIM}code{close}"
        + (f" {ACCENT_MUTED}({lang}){close}" if lang else "")
        + "\n"
    )
    footer = f"\n{indent}{ACCENT_SYSTEM}\u2514{close}"
    return header + indented_block + footer
