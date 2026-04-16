"""Display-width-aware input composer for Orbit2 CLI.

Provides CJK-safe input handling with correct cursor positioning and
backspace boundaries. Salvaged from Orbit1 composer_state.py, simplified
for the Orbit2 CLI operator surface.

Design rules:
- Cursor position tracks display width, not byte offset.
- Backspace deletes one character (even if 2-column wide) and never
  escapes past the prompt boundary.
- Arrow keys move by character, accounting for display width.
- All CSI escape sequences are fully consumed so no trailing bytes
  leak into text input.
"""

from __future__ import annotations

import os
import sys
import tty
import termios
from unicodedata import category as unicode_category, east_asian_width


# ---------------------------------------------------------------------------
# Composer actions — raised to signal non-text events to the harness
# ---------------------------------------------------------------------------

class ComposerAction(Exception):
    """Base class for non-text actions from the composer."""
    pass


class PageUpAction(ComposerAction):
    """Page Up pressed — signals previous session switch."""
    pass


class PageDownAction(ComposerAction):
    """Page Down pressed — signals next session switch."""
    pass


# ---------------------------------------------------------------------------
# Display width
# ---------------------------------------------------------------------------

def display_width(text: str) -> int:
    """Calculate terminal display width, CJK double-width aware."""
    w = 0
    for ch in text:
        w += _char_width(ch)
    return w


def _char_width(ch: str) -> int:
    """Display width of a single character.

    Returns 0 for combining marks and zero-width characters,
    2 for CJK/fullwidth, 1 for everything else.
    """
    cat = unicode_category(ch)
    # Combining marks (Mn, Mc, Me) and zero-width chars
    if cat.startswith("M") or ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        return 0
    # Control characters
    if cat == "Cc":
        return 0
    eaw = east_asian_width(ch)
    return 2 if eaw in ("W", "F") else 1


# ---------------------------------------------------------------------------
# Raw terminal I/O
# ---------------------------------------------------------------------------

def _read_utf8_char(fd: int) -> str:
    """Read one complete UTF-8 character from fd in raw mode."""
    first = _read_byte(fd)
    if first is None:
        raise EOFError
    b = first[0]
    if b < 0x80:
        return chr(b)
    elif b < 0xE0:
        n = 2
    elif b < 0xF0:
        n = 3
    else:
        n = 4
    buf = first
    for _ in range(n - 1):
        cont = _read_byte(fd)
        if cont is None:
            return chr(0xFFFD)
        buf += cont
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return chr(0xFFFD)


def _read_byte(fd: int) -> bytes | None:
    try:
        b = os.read(fd, 1)
        return b if b else None
    except OSError:
        return None


def _write(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _redraw_from_cursor(buf: list[str], cursor: int, old_tail_width: int) -> None:
    """Redraw everything from cursor position to end, then restore cursor."""
    tail = "".join(buf[cursor:])
    tail_width = display_width(tail)
    _write(tail)
    extra = old_tail_width - tail_width
    if extra > 0:
        _write(" " * extra)
        _write(f"\x1b[{extra}D")
    if tail_width > 0:
        _write(f"\x1b[{tail_width}D")


# ---------------------------------------------------------------------------
# CSI sequence parser
# ---------------------------------------------------------------------------

def _read_csi_sequence(fd: int) -> tuple[str, str]:
    """Read a complete CSI sequence after ESC[ has been consumed.

    Returns (params, terminator) where:
    - params: the parameter bytes (digits, semicolons) as a string
    - terminator: the final byte that ends the sequence (letter or ~)

    Standard CSI format: ESC [ <params> <terminator>
    Examples:
        ESC [ A       → ("", "A")     — Up arrow
        ESC [ 5 ~     → ("5", "~")    — Page Up
        ESC [ 1 5 ~   → ("15", "~")   — F5
        ESC [ 1 ; 5 C → ("1;5", "C")  — Ctrl+Right
    """
    params = ""
    while True:
        ch = _read_utf8_char(fd)
        code = ord(ch)
        # Parameter bytes: digits 0-9 and semicolons
        if ch.isdigit() or ch == ";":
            params += ch
            continue
        # Intermediate bytes (space through /) — consume but don't store
        if 0x20 <= code <= 0x2F:
            continue
        # Valid CSI final byte (0x40-0x7E) — terminates the sequence
        if 0x40 <= code <= 0x7E:
            return (params, ch)
        # Control character or other unexpected byte mid-sequence — abort
        # Return empty params with a sentinel so the caller discards cleanly
        return ("", "")


# ---------------------------------------------------------------------------
# Main input function
# ---------------------------------------------------------------------------

def read_line(prompt: str = "") -> str:
    """Read one line of input with CJK-aware cursor positioning.

    The prompt is written first; backspace cannot delete past it.
    Returns the entered text (without trailing newline), or raises
    EOFError, KeyboardInterrupt, PageUpAction, or PageDownAction.
    """
    _write(prompt)

    buf: list[str] = []
    cursor: int = 0  # index into buf (character position)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = _read_utf8_char(fd)

            # Enter
            if ch in ("\r", "\n"):
                _write("\r\n")
                return "".join(buf)

            # Ctrl+C
            if ch == "\x03":
                _write("\r\n")
                raise KeyboardInterrupt

            # Ctrl+D
            if ch == "\x04":
                if not buf:
                    raise EOFError
                continue

            # Backspace (DEL or BS)
            if ch in ("\x7f", "\x08"):
                if cursor > 0:
                    deleted = buf[cursor - 1]
                    del_width = _char_width(deleted)
                    old_tail_width = display_width("".join(buf[cursor:]))
                    del buf[cursor - 1]
                    cursor -= 1
                    if del_width > 0:
                        _write(f"\x1b[{del_width}D")
                    _redraw_from_cursor(buf, cursor, old_tail_width + del_width)
                continue

            # Escape sequences
            if ch == "\x1b":
                seq1 = _read_utf8_char(fd)
                if seq1 == "[":
                    params, term = _read_csi_sequence(fd)

                    # Arrow keys (no params)
                    if term == "D" and not params:  # Left
                        if cursor > 0:
                            cursor -= 1
                            w = _char_width(buf[cursor])
                            if w > 0:
                                _write(f"\x1b[{w}D")
                        continue

                    if term == "C" and not params:  # Right
                        if cursor < len(buf):
                            w = _char_width(buf[cursor])
                            if w > 0:
                                _write(f"\x1b[{w}C")
                            cursor += 1
                        continue

                    # Home: CSI H or CSI 1~
                    if (term == "H" and not params) or (term == "~" and params == "1"):
                        if cursor > 0:
                            w = display_width("".join(buf[:cursor]))
                            if w > 0:
                                _write(f"\x1b[{w}D")
                            cursor = 0
                        continue

                    # End: CSI F or CSI 4~
                    if (term == "F" and not params) or (term == "~" and params == "4"):
                        if cursor < len(buf):
                            w = display_width("".join(buf[cursor:]))
                            if w > 0:
                                _write(f"\x1b[{w}C")
                            cursor = len(buf)
                        continue

                    # Delete: CSI 3~
                    if term == "~" and params == "3":
                        if cursor < len(buf):
                            old_tail_width = display_width("".join(buf[cursor:]))
                            del buf[cursor]
                            _redraw_from_cursor(buf, cursor, old_tail_width)
                        continue

                    # Page Up: CSI 5~
                    if term == "~" and params == "5":
                        raise PageUpAction()

                    # Page Down: CSI 6~
                    if term == "~" and params == "6":
                        raise PageDownAction()

                    # All other CSI sequences are fully consumed and discarded
                    continue

                # SS3 sequences (ESC O x) — consume the key code byte
                if seq1 == "O":
                    _read_utf8_char(fd)  # consume the key code (e.g. A/B/C/D for arrows)
                # All other non-CSI escape sequences: seq1 already consumed
                continue

            # Ctrl+A — home
            if ch == "\x01":
                if cursor > 0:
                    w = display_width("".join(buf[:cursor]))
                    if w > 0:
                        _write(f"\x1b[{w}D")
                    cursor = 0
                continue

            # Ctrl+E — end
            if ch == "\x05":
                if cursor < len(buf):
                    w = display_width("".join(buf[cursor:]))
                    if w > 0:
                        _write(f"\x1b[{w}C")
                    cursor = len(buf)
                continue

            # Ctrl+U — kill line (clear input)
            if ch == "\x15":
                if buf:
                    if cursor > 0:
                        w = display_width("".join(buf[:cursor]))
                        if w > 0:
                            _write(f"\x1b[{w}D")
                    _write("\x1b[K")
                    buf.clear()
                    cursor = 0
                continue

            # Ctrl+K — kill to end of line
            if ch == "\x0b":
                if cursor < len(buf):
                    _write("\x1b[K")
                    del buf[cursor:]
                continue

            # Ctrl+W — delete word backward
            if ch == "\x17":
                if cursor > 0:
                    orig_cursor = cursor
                    while cursor > 0 and buf[cursor - 1] in (" ", "\t"):
                        cursor -= 1
                    while cursor > 0 and buf[cursor - 1] not in (" ", "\t"):
                        cursor -= 1
                    deleted = buf[cursor:orig_cursor]
                    del_width = display_width("".join(deleted))
                    old_tail_width = display_width("".join(buf[orig_cursor:]))
                    del buf[cursor:orig_cursor]
                    if del_width > 0:
                        _write(f"\x1b[{del_width}D")
                    _redraw_from_cursor(buf, cursor, old_tail_width + del_width)
                continue

            # Regular printable character (skip zero-width for display safety)
            if ord(ch) >= 32:
                w = _char_width(ch)
                if w == 0:
                    # Zero-width / combining character — skip to avoid cursor desync
                    continue
                old_tail_width = display_width("".join(buf[cursor:]))
                buf.insert(cursor, ch)
                cursor += 1
                _write(ch)
                if cursor < len(buf):
                    _redraw_from_cursor(buf, cursor, old_tail_width)

    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except OSError:
            pass
