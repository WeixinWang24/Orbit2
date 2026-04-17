from __future__ import annotations

import re

from src.operation.cli.style import (
    ACCENT_MUTED,
    ACCENT_SYSTEM,
    BOLD,
    DIM,
    RESET,
)

# Fenced code block: exactly three backticks optionally followed by a
# language tag, then content, then a closing `\n````. The negative lookbehind
# and lookahead stop 4+-backtick CommonMark escape fences from matching the
# wrong boundary and leaving an orphan backtick in the surrounding prose.
# Requiring `\n` before the closing fence means the footer always lands on
# its own line — stray prose after the closing fence can never be attached
# to the block footer when rendered.
_FENCED_RE = re.compile(
    r"(?<!`)```([a-zA-Z0-9_+-]*)\n(.*?)\n```(?!`)",
    re.DOTALL,
)
# Inline `code` — non-greedy, no newlines inside
_INLINE_CODE_RE = re.compile(r"`([^`\n]+?)`")
# **bold** — non-greedy
_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")


def render_markdown_for_terminal(text: str, *, indent: str = "    ") -> str:
    """Render a bounded subset of markdown for terminal display.

    Scope — first-slice only:
      - fenced code blocks (``` ... ```)
      - inline code (`foo`)
      - bold (**foo**)
    Out of scope: italics, headings, lists, links, tables, syntax highlighting.

    `indent` is applied to every fenced-code-block line so blocks stand out from
    surrounding prose in the transcript view.
    """
    if not text:
        return text

    rendered_parts: list[str] = []
    last_end = 0
    for match in _FENCED_RE.finditer(text):
        prose = text[last_end : match.start()]
        if prose:
            rendered_parts.append(_render_inline(prose))
        lang = match.group(1).strip()
        code_body = match.group(2)
        rendered_parts.append(_render_fenced(lang, code_body, indent=indent))
        last_end = match.end()
    tail = text[last_end:]
    if tail:
        rendered_parts.append(_render_inline(tail))

    return "".join(rendered_parts)


def _render_inline(text: str) -> str:
    text = _BOLD_RE.sub(lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    text = _INLINE_CODE_RE.sub(
        lambda m: f"{ACCENT_MUTED}{m.group(1)}{RESET}", text,
    )
    return text


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
