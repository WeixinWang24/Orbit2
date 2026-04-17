from __future__ import annotations

import re

from src.operation.cli.style import (
    ACCENT_MUTED,
    ACCENT_SUCCESS,
    ACCENT_SYSTEM,
    BOLD,
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


def render_markdown_for_terminal(text: str, *, indent: str = "    ") -> str:
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
    """
    if not text:
        return text

    rendered_parts: list[str] = []
    last_end = 0
    for match in _FENCED_RE.finditer(text):
        prose = text[last_end : match.start()]
        if prose:
            rendered_parts.append(_render_prose(prose))
        lang = match.group(1).strip()
        code_body = match.group(2)
        rendered_parts.append(_render_fenced(lang, code_body, indent=indent))
        last_end = match.end()
    tail = text[last_end:]
    if tail:
        rendered_parts.append(_render_prose(tail))

    return "".join(rendered_parts)


def _render_prose(text: str) -> str:
    text = _HEADING_RE.sub(_render_heading_match, text)
    text = _BULLET_RE.sub(_render_bullet_match, text)
    text = _BOLD_RE.sub(lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    text = _ITALIC_RE.sub(lambda m: f"{ITALIC}{m.group(1)}{RESET}", text)
    text = _INLINE_CODE_RE.sub(
        lambda m: f"{ACCENT_MUTED}{m.group(1)}{RESET}", text,
    )
    text = _BLANK_SQUASH_RE.sub("\n\n", text)
    return text


def _render_heading_match(match: re.Match[str]) -> str:
    level = len(match.group(1))
    body = match.group(2).strip()
    if level <= 2:
        accent = ACCENT_SYSTEM
    elif level == 3:
        accent = ACCENT_SUCCESS
    else:
        accent = ACCENT_MUTED
    return f"{BOLD}{accent}{body}{RESET}"


def _render_bullet_match(match: re.Match[str]) -> str:
    lead = match.group(1)
    body = match.group(2)
    return f"{lead}{ACCENT_MUTED}\u2022{RESET} {body}"


def _render_fenced(lang: str, body: str, *, indent: str) -> str:
    body = body.rstrip("\n")
    indented_lines = [f"{indent}{DIM}{line}{RESET}" for line in body.split("\n")]
    indented_block = "\n".join(indented_lines)
    header = (
        f"{indent}{ACCENT_SYSTEM}\u250c{RESET}"
        f" {DIM}code{RESET}"
        + (f" {ACCENT_MUTED}({lang}){RESET}" if lang else "")
        + "\n"
    )
    footer = f"\n{indent}{ACCENT_SYSTEM}\u2514{RESET}"
    return header + indented_block + footer
