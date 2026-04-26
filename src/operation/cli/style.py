"""Terminal style constants for Orbit2 CLI.

Salvaged from Orbit1 termio.py and runtime_cli_render.py.
Provides role-specific colors and ANSI escape helpers for the CLI operator surface.
"""

from __future__ import annotations

import os

# ── ANSI escape primitives ───────────────────────────────────────────────────

ESC = "\x1b"
CSI = ESC + "["

RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
DIM = f"{CSI}2m"
ITALIC = f"{CSI}3m"
UNDERLINE = f"{CSI}4m"

FG_BRIGHT_BLACK = f"{CSI}90m"
FG_BRIGHT_GREEN = f"{CSI}92m"
FG_BRIGHT_YELLOW = f"{CSI}93m"
FG_BRIGHT_MAGENTA = f"{CSI}95m"
FG_BRIGHT_CYAN = f"{CSI}96m"
FG_BRIGHT_RED = f"{CSI}91m"


def fg_rgb(r: int, g: int, b: int) -> str:
    return f"{CSI}38;2;{r};{g};{b}m"


# ── Role colors (from Orbit1 runtime_cli_render.py) ─────────────────────────

# Label accents
ACCENT_USER = fg_rgb(120, 185, 255)       # bright blue — user label
ACCENT_ASSISTANT = FG_BRIGHT_MAGENTA      # magenta — assistant label
ACCENT_TOOL = FG_BRIGHT_YELLOW            # yellow — tool messages
ACCENT_SYSTEM = FG_BRIGHT_CYAN            # cyan — system/session info
ACCENT_SUCCESS = FG_BRIGHT_GREEN          # green — success signals
ACCENT_ERROR = FG_BRIGHT_RED              # red — errors
ACCENT_MUTED = FG_BRIGHT_BLACK            # grey — dim metadata

# Content body colors
# Keep message text readable on black terminal backgrounds.
CONTENT_USER = fg_rgb(135, 195, 255)       # bright steel-blue
CONTENT_ASSISTANT = fg_rgb(215, 190, 255)  # bright violet/lavender
CONTENT_CODE = fg_rgb(225, 225, 235)       # bright neutral for code blocks
CONTENT_INLINE_CODE = fg_rgb(245, 220, 140) # warm bright inline code

# Divider
DIVIDER_COLOR = fg_rgb(60, 60, 80)        # very dim purple-grey


def supports_truecolor() -> bool:
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return True
    term_program = os.environ.get("TERM_PROGRAM", "")
    return term_program in ("iTerm.app", "WezTerm", "ghostty", "vscode")


def _detect_truecolor() -> bool:
    """Detect truecolor once; called at import time."""
    return supports_truecolor()


TRUECOLOR = _detect_truecolor()

# Fallback to standard colors if truecolor not available
if not TRUECOLOR:
    ACCENT_USER = f"{CSI}94m"             # bright blue
    CONTENT_USER = f"{CSI}94m"            # bright blue
    CONTENT_ASSISTANT = f"{CSI}95m"       # bright magenta
    CONTENT_CODE = f"{CSI}97m"            # bright white
    CONTENT_INLINE_CODE = f"{CSI}93m"     # bright yellow
    DIVIDER_COLOR = DIM


def styled(text: str, *codes: str) -> str:
    if not codes:
        return text
    return "".join(codes) + text + RESET


def divider(width: int = 60) -> str:
    return DIVIDER_COLOR + ("\u2500" * width) + RESET
