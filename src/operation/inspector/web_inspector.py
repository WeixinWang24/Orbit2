from __future__ import annotations

import json
import sqlite3
import sys
import traceback
import webbrowser
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from src.core.store.sqlite import SQLiteSessionStore

_VALID_MAIN_TABS = {"transcript", "debug", "assembly"}
_VALID_RIGHT_TABS = {"metadata", "raw"}

# ---------------------------------------------------------------------------
# CSS — ported verbatim from Orbit1 VIO_INSPIRED_CSS
# ---------------------------------------------------------------------------

VIO_INSPIRED_CSS = """
:root {
  --bg: #050510;
  --panel: #121222;
  --text: #BFE8FF;
  --text-dim: #7FC8E8;
  --cyan: #00FFC6;
  --pink: #FF2E88;
  --green: #3CFF9F;
  --amber: #FFCC66;
  --red: #FF6B8A;
  --violet: #B388FF;
  --blue: #61B8FF;
  --radius: 14px;
  --shadow-cyan: 0 0 0 1px rgba(0,255,198,.18), 0 0 18px rgba(0,255,198,.08);
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; width: 100%; min-height: 100%; color-scheme: dark;
  background:
    linear-gradient(180deg, rgba(122,0,255,.06), transparent 30%),
    radial-gradient(circle at top right, rgba(0,255,198,.06), transparent 24%),
    radial-gradient(circle at bottom left, rgba(255,46,136,.05), transparent 24%),
    var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 20px;
}
body { overflow: hidden; }
.dashboard {
  width: calc(100vw - 24px);
  height: calc(100vh - 24px);
  margin: 12px;
  display: grid;
  gap: 14px;
  grid-template-columns: 300px minmax(0, 1.2fr) minmax(0, 1fr);
  grid-template-rows: 56px minmax(0, 1fr);
  grid-template-areas:
    'top top top'
    'sidebar main right';
}
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
  border: 1px solid rgba(0,255,198,.18);
  border-radius: var(--radius);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.03), 0 0 0 1px rgba(0,255,198,.08);
  backdrop-filter: blur(10px);
}
.topbar { grid-area: top; padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.brand h1 { margin: 0; font-size: 24px; text-transform: uppercase; letter-spacing: .12em; color: #D59BFF; }
.brand p { margin: 4px 0 0; color: var(--text-dim); font-size: 18px; letter-spacing: .06em; text-transform: uppercase; }
.sidebar, .main, .right { min-height: 0; overflow: auto; padding: 14px; }
.sidebar { grid-area: sidebar; }
.main { grid-area: main; display: flex; flex-direction: column; }
.right { grid-area: right; display: flex; flex-direction: column; min-width: 0; }
.section-title { margin: 0 0 12px; font-size: 18px; letter-spacing: .12em; text-transform: uppercase; color: var(--cyan); }
.session-link {
  display: block; padding: 10px 12px; margin-bottom: 8px; text-decoration: none; color: var(--text);
  border: 1px solid rgba(0,255,198,.14); border-radius: 12px; background: rgba(18,18,34,.68);
}
.session-link:hover { border-color: rgba(0,255,198,.24); }
.session-link.active { box-shadow: var(--shadow-cyan); border-color: rgba(0,255,198,.32); }
.session-meta { color: var(--text-dim); font-size: 14px; margin-top: 4px; }
.tabbar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.tab {
  display: inline-flex; align-items: center; padding: 10px 14px; border-radius: 999px; text-decoration: none;
  border: 1px solid rgba(0,255,198,.18); color: var(--text-dim); background: rgba(18,18,34,.72); font-size: 18px;
}
.tab:hover { border-color: rgba(0,255,198,.28); color: var(--text); }
.tab.active { color: var(--text); box-shadow: var(--shadow-cyan); border-color: rgba(0,255,198,.30); }
.main-panel, .right-panel-body { min-height: 0; overflow: auto; flex: 1; }
.meta { color: var(--text-dim); font-size: 18px; }
.message-card { margin-bottom: 12px; padding: 16px; border-radius: 12px; border: 1px solid rgba(0,255,198,.12); background: rgba(18,18,34,.75); }
.message-card.user { border-color: rgba(97,184,255,.28); background: linear-gradient(180deg, rgba(97,184,255,.08), rgba(18,18,34,.78)); }
.message-card.assistant { border-color: rgba(255,46,136,.22); background: linear-gradient(180deg, rgba(255,46,136,.05), rgba(18,18,34,.78)); }
.message-card.tool { border-color: rgba(179,136,255,.38); background: linear-gradient(180deg, rgba(179,136,255,.10), rgba(18,18,34,.78)); }
.message-card.tool.tool-ok { border-color: rgba(60,255,159,.30); background: linear-gradient(180deg, rgba(60,255,159,.08), rgba(18,18,34,.78)); }
.message-card.tool.tool-fail { border-color: rgba(255,107,138,.30); background: linear-gradient(180deg, rgba(255,107,138,.08), rgba(18,18,34,.78)); }
.message-role { font-size: 18px; letter-spacing: .10em; text-transform: uppercase; color: var(--pink); margin-bottom: 8px; }
.message-card.user .message-role { color: var(--blue); }
.message-card.assistant .message-role { color: var(--pink); }
.message-card.tool .message-role { color: var(--violet); }
.message-summary { margin-bottom: 10px; color: var(--text); font-size: 17px; line-height: 1.6; }
.message-meta-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.message-chip { padding: 6px 10px; border-radius: 999px; font-size: 15px; border: 1px solid rgba(0,255,198,.16); color: var(--text-dim); }
.message-chip.ok { border-color: rgba(60,255,159,.34); color: var(--green); }
.message-chip.fail { border-color: rgba(255,107,138,.34); color: var(--red); }
.message-chip.governance { border-color: rgba(255,204,102,.34); color: var(--amber); }
pre {
  margin: 0; white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 17px; line-height: 1.8;
}
.panel-block { padding: 16px; border-radius: 12px; border: 1px solid rgba(0,255,198,.12); background: rgba(18,18,34,.75); overflow: auto; margin-bottom: 12px; }
.fragment-card { margin-bottom: 12px; padding: 14px; border-radius: 12px; border: 1px solid rgba(0,255,198,.12); background: rgba(18,18,34,.75); }
.fragment-title { font-size: 18px; color: var(--cyan); margin: 0 0 8px; letter-spacing: .08em; text-transform: uppercase; }
.empty { color: var(--text-dim); font-style: italic; }
.chip { padding: 8px 12px; border-radius: 999px; border: 1px solid rgba(255,46,136,.22); color: #ffd1e6; font-size: 16px; }
.json-view details { margin: 6px 0; padding-left: 10px; border-left: 1px solid rgba(0,255,198,.14); }
.json-view summary { cursor: pointer; color: var(--cyan); }
.json-kv { margin: 4px 0; }
.json-key { color: var(--violet); }
.json-type { color: var(--amber); font-size: 12px; margin-left: 6px; }
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.stat-card {
  flex: 1; min-width: 100px; padding: 10px 12px; border-radius: 10px;
  border: 1px solid rgba(0,255,198,.14); background: rgba(18,18,34,.7); text-align: center;
}
.stat-value { font-size: 22px; font-weight: 700; color: var(--cyan); line-height: 1.2; }
.stat-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }
.stat-card.ok .stat-value { color: var(--green); }
.stat-card.warn .stat-value { color: var(--amber); }
.stat-card.danger .stat-value { color: var(--red); }
.section-card {
  padding: 14px; border-radius: 10px; border: 1px solid rgba(0,255,198,.12);
  background: rgba(18,18,34,.65); margin-bottom: 10px;
}
.section-card h3 {
  margin: 0 0 8px; font-size: 14px; color: var(--cyan);
  letter-spacing: .08em; text-transform: uppercase;
}
.section-card.cyan { border-color: rgba(0,255,198,.25); }
.section-card.violet { border-color: rgba(179,136,255,.25); }
.section-card.blue { border-color: rgba(97,184,255,.25); }
.section-card.green { border-color: rgba(60,255,159,.25); }
.kv-row {
  display: flex; justify-content: space-between; padding: 4px 0;
  border-bottom: 1px solid rgba(0,255,198,.06); font-size: 14px;
}
.kv-row:last-child { border-bottom: none; }
.kv-key { color: var(--text-dim); }
.kv-val { color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (max-width: 1400px) {
  .dashboard { grid-template-columns: 260px minmax(0, 1fr) minmax(0, 1fr); }
}
"""

# ---------------------------------------------------------------------------
# Utility functions — ported from Orbit1
# ---------------------------------------------------------------------------


def _preview_snippet(value: str, *, limit: int = 80) -> str:
    preview = value.replace("\n", "\\n")
    return preview[:limit] + ("\u2026" if len(preview) > limit else "")


def _truncate_text(value: str, *, limit: int = 220) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[: limit - 1].rstrip() + "\u2026", True


def _format_local_timestamp(value) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value
    else:
        return str(value)
    if dt.tzinfo is None:
        return dt.isoformat()
    return dt.astimezone().isoformat()


def _json_tree_html(
    value, *, label: str | None = None, open_depth: int = 0, _depth: int = 0,
) -> str:
    if isinstance(value, dict):
        title = label or f"object ({len(value)})"
        open_attr = " open" if _depth < open_depth else ""
        rows = "".join(
            _json_tree_html(item, label=str(key), open_depth=open_depth, _depth=_depth + 1)
            for key, item in value.items()
        )
        rows = rows or '<div class="empty">empty object</div>'
        return (
            f'<details{open_attr}><summary>{escape(title)}'
            f'<span class="json-type">dict</span></summary>'
            f'<div class="json-view">{rows}</div></details>'
        )
    if isinstance(value, list):
        title = label or f"array ({len(value)})"
        open_attr = " open" if _depth < open_depth else ""
        rows = "".join(
            _json_tree_html(item, label=f"[{idx}]", open_depth=open_depth, _depth=_depth + 1)
            for idx, item in enumerate(value)
        )
        rows = rows or '<div class="empty">empty list</div>'
        return (
            f'<details{open_attr}><summary>{escape(title)}'
            f'<span class="json-type">list</span></summary>'
            f'<div class="json-view">{rows}</div></details>'
        )
    rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    prefix = f'<span class="json-key">{escape(label)}</span>: ' if label is not None else ""
    return f'<div class="json-kv">{prefix}<code>{escape(rendered)}</code></div>'


def _render_json_block(value, *, title: str | None = None, open_depth: int = 0) -> str:
    return '<div class="json-view">' + _json_tree_html(value, label=title, open_depth=open_depth) + "</div>"


def _render_stat_card(value: str, label: str, css_class: str = "") -> str:
    cls = f' {css_class}' if css_class else ""
    return (
        f'<div class="stat-card{cls}">'
        f'<div class="stat-value">{escape(str(value))}</div>'
        f'<div class="stat-label">{escape(label)}</div>'
        f"</div>"
    )


def _render_kv_row(key: str, val: str) -> str:
    return (
        f'<div class="kv-row">'
        f'<span class="kv-key">{escape(key)}</span>'
        f'<span class="kv-val">{escape(str(val))}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------


def _message_css_class(msg) -> str:
    """Return CSS class string for a ConversationMessage."""
    role = msg.role.value.lower() if hasattr(msg.role, "value") else str(msg.role).lower()
    if role == "tool":
        meta = msg.metadata or {}
        ok = meta.get("ok")
        if ok is True:
            return "tool tool-ok"
        if ok is False:
            return "tool tool-fail"
        return "tool"
    return role


def _message_chips_html(msg) -> str:
    """Render metadata chips for a message (tool name, governance, etc.)."""
    meta = msg.metadata or {}
    chips = []

    # Turn index
    chips.append(f'<span class="message-chip">turn {msg.turn_index}</span>')

    # Tool-specific chips
    tool_name = meta.get("tool_name")
    if tool_name:
        chips.append(f'<span class="message-chip">{escape(str(tool_name))}</span>')

    governance = meta.get("governance_outcome")
    if governance:
        css = "ok" if governance == "allowed" else "fail" if "denied" in str(governance) else "governance"
        chips.append(f'<span class="message-chip {css}">{escape(str(governance))}</span>')

    ok = meta.get("ok")
    if ok is True:
        chips.append('<span class="message-chip ok">ok</span>')
    elif ok is False:
        chips.append('<span class="message-chip fail">failed</span>')

    # Tool calls count (for assistant messages with tool_calls)
    tool_calls = meta.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        chips.append(f'<span class="message-chip">{len(tool_calls)} tool call(s)</span>')
        for tc in tool_calls:
            tc_name = tc.get("name") or tc.get("tool_name") or ""
            if tc_name:
                chips.append(f'<span class="message-chip">{escape(tc_name)}</span>')

    return "".join(chips)


def _render_message_card(msg) -> str:
    """Render a single message card."""
    css_cls = _message_css_class(msg)
    role_label = msg.role.value.upper() if hasattr(msg.role, "value") else str(msg.role).upper()
    content = msg.content or ""
    preview, truncated = _truncate_text(content)
    chips = _message_chips_html(msg)
    meta = msg.metadata or {}

    parts = [f'<div class="message-card {css_cls}">']
    parts.append(f'<div class="message-role">{escape(role_label)}</div>')

    if chips:
        parts.append(f'<div class="message-meta-row">{chips}</div>')

    parts.append(f'<div class="message-summary"><pre>{escape(preview)}</pre></div>')

    if meta:
        parts.append(
            '<details><summary style="cursor:pointer;color:var(--text-dim);font-size:14px;">'
            "metadata</summary>"
        )
        parts.append(_render_json_block(meta, title="metadata", open_depth=1))
        parts.append("</details>")

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _render_transcript_panel(messages: list) -> str:
    if not messages:
        return '<div class="empty">No messages in this session.</div>'
    return "".join(_render_message_card(msg) for msg in messages)


def _render_debug_panel(session, messages: list) -> str:
    """Debug panel: session-level metadata + tool call details."""
    parts = []

    # Session info card
    parts.append('<div class="section-card cyan">')
    parts.append("<h3>Session</h3>")
    parts.append(_render_kv_row("session_id", session.session_id))
    parts.append(_render_kv_row("backend", session.backend_name))
    parts.append(_render_kv_row("status", session.status.value if hasattr(session.status, "value") else str(session.status)))
    parts.append(_render_kv_row("created_at", _format_local_timestamp(session.created_at)))
    parts.append(_render_kv_row("updated_at", _format_local_timestamp(session.updated_at)))
    parts.append("</div>")

    if session.system_prompt:
        parts.append('<div class="section-card violet">')
        parts.append("<h3>System Prompt</h3>")
        parts.append(f"<pre>{escape(session.system_prompt)}</pre>")
        parts.append("</div>")

    if session.metadata:
        parts.append('<div class="section-card blue">')
        parts.append("<h3>Session Metadata</h3>")
        parts.append(_render_json_block(session.metadata, title="session metadata", open_depth=1))
        parts.append("</div>")

    # Tool call details extracted from messages
    tool_msgs = [m for m in messages if (m.role.value if hasattr(m.role, "value") else str(m.role)).lower() == "tool"]
    if tool_msgs:
        parts.append('<div class="section-card green">')
        parts.append(f"<h3>Tool Executions ({len(tool_msgs)})</h3>")
        for tm in tool_msgs:
            meta = tm.metadata or {}
            tool_name = meta.get("tool_name", "unknown")
            ok = meta.get("ok")
            governance = meta.get("governance_outcome", "")
            status_cls = "ok" if ok else "fail" if ok is False else ""
            parts.append(f'<div class="fragment-card">')
            parts.append(
                f'<div class="fragment-title">{escape(str(tool_name))}'
                f' <span class="message-chip {status_cls}">{escape(str(governance))}</span>'
                f"</div>"
            )
            tool_call_id = meta.get("tool_call_id", "")
            if tool_call_id:
                parts.append(f'<div style="font-size:13px;color:var(--text-dim);">call_id: {escape(str(tool_call_id))}</div>')
            content_preview = _preview_snippet(tm.content or "", limit=120)
            parts.append(f"<pre>{escape(content_preview)}</pre>")
            parts.append("</div>")
        parts.append("</div>")

    return "".join(parts)


def _extract_assembly_envelopes(messages: list) -> list[tuple]:
    """Walk the transcript and pull (turn_index, envelope_dict) pairs from
    every assistant message that has an `assembly_envelope` attached.

    The inspector is a projection — it never reconstructs envelopes from
    transcript state. If the runtime did not persist one on a given turn,
    the inspector simply does not render one for that turn. Order is
    transcript order (ascending turn_index), so operators see how the
    assembled payload evolved across a multi-tool-loop turn.
    """
    rows: list[tuple] = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role.lower() != "assistant":
            continue
        meta = msg.metadata or {}
        envelope = meta.get("assembly_envelope")
        if not isinstance(envelope, dict):
            continue
        rows.append((msg.turn_index, envelope))
    return rows


def _render_envelope_card(turn_index: int, envelope: dict) -> str:
    parts = ['<div class="fragment-card">']
    parts.append(
        f'<div class="fragment-title">Turn {escape(str(turn_index))}'
        f' \u00b7 {escape(str(envelope.get("assembler_name", "unknown")))}'
        "</div>"
    )

    # Top-line stats
    parts.append('<div class="stat-row">')
    parts.append(
        _render_stat_card(
            str(envelope.get("transcript_message_count", 0)),
            "transcript msgs",
        )
    )
    parts.append(
        _render_stat_card(
            str(envelope.get("assembled_message_count", 0)),
            "assembled msgs",
        )
    )
    parts.append(
        _render_stat_card(
            str(len(envelope.get("instruction_fragments") or [])),
            "fragments",
        )
    )
    exposed_names = envelope.get("exposed_tool_names")
    if exposed_names is not None:
        parts.append(
            _render_stat_card(str(len(exposed_names)), "exposed tools")
        )
    parts.append("</div>")

    # Assembled system preview
    system_preview = envelope.get("assembled_system_preview")
    if system_preview:
        parts.append('<div class="section-card violet">')
        parts.append("<h3>Assembled System</h3>")
        parts.append(f"<pre>{escape(str(system_preview))}</pre>")
        parts.append("</div>")

    # Instruction fragments
    fragments = envelope.get("instruction_fragments") or []
    if fragments:
        parts.append('<div class="section-card cyan">')
        parts.append(f"<h3>Instruction Fragments ({len(fragments)})</h3>")
        for frag in fragments:
            parts.append('<div class="fragment-card">')
            parts.append(
                f'<div class="fragment-title">'
                f'{escape(str(frag.get("fragment_name", "")))}'
                f' <span class="message-chip">'
                f'{escape(str(frag.get("visibility_scope", "")))}</span>'
                f' <span class="message-chip">priority '
                f'{escape(str(frag.get("priority", 0)))}</span>'
                f' <span class="message-chip">len '
                f'{escape(str(frag.get("content_length", 0)))}</span>'
                "</div>"
            )
            preview = frag.get("content_preview") or ""
            parts.append(f"<pre>{escape(str(preview))}</pre>")
            parts.append("</div>")
        parts.append("</div>")

    # Assembled messages preview (bounded)
    preview_rows = envelope.get("assembled_messages_preview") or []
    truncated = envelope.get("assembled_messages_truncated", False)
    if preview_rows:
        parts.append('<div class="section-card blue">')
        suffix = " (truncated)" if truncated else ""
        parts.append(
            f"<h3>Assembled Provider-Facing Messages "
            f"({len(preview_rows)}{suffix})</h3>"
        )
        for row in preview_rows:
            role = row.get("role", "")
            content_preview = row.get("content_preview")
            chips: list[str] = [
                f'<span class="message-chip">{escape(role)}</span>',
                f'<span class="message-chip">len '
                f'{escape(str(row.get("content_length", 0)))}</span>',
            ]
            tcc = row.get("tool_calls_count", 0)
            if tcc:
                chips.append(
                    f'<span class="message-chip">{escape(str(tcc))} tool call(s)</span>'
                )
            if row.get("has_tool_call_id"):
                chips.append('<span class="message-chip">tool_call_id</span>')
            parts.append('<div class="fragment-card">')
            parts.append(f'<div class="message-meta-row">{"".join(chips)}</div>')
            if content_preview is not None:
                parts.append(f"<pre>{escape(str(content_preview))}</pre>")
            parts.append("</div>")
        parts.append("</div>")

    # Exposure details
    if exposed_names is not None:
        parts.append('<div class="section-card green">')
        parts.append("<h3>Exposed Capability</h3>")
        groups = envelope.get("exposed_tool_groups") or []
        rejected = envelope.get("rejected_reveal_requests") or []
        if groups:
            parts.append(
                f'<div class="message-meta-row">'
                + "".join(
                    f'<span class="message-chip">{escape(str(g))}</span>'
                    for g in groups
                )
                + "</div>"
            )
        else:
            parts.append('<div class="empty">no active reveal groups</div>')
        parts.append(
            '<div class="message-meta-row">'
            + "".join(
                f'<span class="message-chip">{escape(str(n))}</span>'
                for n in exposed_names
            )
            + "</div>"
        )
        if rejected:
            parts.append(
                "<div style='margin-top:8px;'><strong>Rejected reveals:</strong></div>"
                '<div class="message-meta-row">'
                + "".join(
                    f'<span class="message-chip fail">{escape(str(r))}</span>'
                    for r in rejected
                )
                + "</div>"
            )
        parts.append("</div>")

    # Assembler metadata (raw view)
    asm_meta = envelope.get("assembler_metadata") or {}
    if asm_meta:
        parts.append('<div class="section-card">')
        parts.append("<h3>Assembler Metadata</h3>")
        parts.append(_render_json_block(asm_meta, title="metadata", open_depth=1))
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def _render_assembly_panel(messages: list) -> str:
    """Knowledge Surface payload/context assembly debug projection.

    This panel is intentionally separate from the transcript view: transcript
    is canonical turn-by-turn truth, this panel is the assembled provider-
    facing payload that the backend actually received. The framing banner is
    load-bearing — without it, operators can confuse the assembled payload
    with the transcript itself.
    """
    parts = []
    parts.append('<div class="section-card violet">')
    parts.append("<h3>Assembly / Payload Debug \u00b7 projection only</h3>")
    parts.append(
        '<div class="meta">Transcript truth lives in the Transcript tab. '
        "This tab projects the assembled provider-facing payload the "
        "Knowledge Surface produced for each planning call, as it was "
        "persisted by the runtime. The inspector never recomputes "
        "assembly; if a turn has no envelope, the runtime did not "
        "produce one.</div>"
    )
    parts.append("</div>")

    envelopes = _extract_assembly_envelopes(messages)
    if not envelopes:
        parts.append(
            '<div class="empty">No assembly envelopes persisted for this '
            "session yet. Run a turn to populate one.</div>"
        )
        return "".join(parts)

    for turn_index, env in envelopes:
        parts.append(_render_envelope_card(turn_index, env))
    return "".join(parts)


def _render_metadata_right_panel(session, messages: list) -> str:
    """Right panel: session identity + message stats."""
    parts = []

    # Session identity
    parts.append('<div class="section-card cyan">')
    parts.append("<h3>Session Identity</h3>")
    parts.append(_render_kv_row("session_id", session.session_id))
    parts.append(_render_kv_row("backend", session.backend_name))
    parts.append(_render_kv_row("status", session.status.value if hasattr(session.status, "value") else str(session.status)))
    parts.append(_render_kv_row("created_at", _format_local_timestamp(session.created_at)))
    parts.append(_render_kv_row("updated_at", _format_local_timestamp(session.updated_at)))
    parts.append("</div>")

    # Message stats
    role_counts: dict[str, int] = {}
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        role_counts[role] = role_counts.get(role, 0) + 1

    parts.append('<div class="stat-row">')
    parts.append(_render_stat_card(str(len(messages)), "messages"))
    for role, count in sorted(role_counts.items()):
        parts.append(_render_stat_card(str(count), role.lower()))
    parts.append("</div>")

    # Governance summary
    governed_msgs = [m for m in messages if (m.metadata or {}).get("governance_outcome")]
    if governed_msgs:
        allowed = sum(1 for m in governed_msgs if m.metadata.get("governance_outcome") == "allowed")
        denied = len(governed_msgs) - allowed
        parts.append('<div class="stat-row">')
        parts.append(_render_stat_card(str(allowed), "allowed", "ok"))
        parts.append(_render_stat_card(str(denied), "denied", "danger" if denied else "ok"))
        parts.append("</div>")

    return "".join(parts)


def _render_raw_right_panel(session, messages: list) -> str:
    """Right panel: raw metadata for all messages."""
    parts = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        parts.append(f'<div class="fragment-card">')
        parts.append(f'<div class="fragment-title">{escape(role.upper())} (turn {msg.turn_index})</div>')
        meta = msg.metadata or {}
        if meta:
            parts.append(_render_json_block(meta, title=f"turn {msg.turn_index} metadata", open_depth=1))
        else:
            parts.append('<div class="empty">no metadata</div>')
        parts.append("</div>")
    return "".join(parts) or '<div class="empty">No messages.</div>'


# ---------------------------------------------------------------------------
# Full page assembly
# ---------------------------------------------------------------------------


def _build_url(session_id: str | None = None, tab: str = "transcript", right_tab: str = "metadata") -> str:
    params: dict[str, str] = {}
    if session_id:
        params["session_id"] = session_id
    if tab != "transcript":
        params["tab"] = tab
    if right_tab != "metadata":
        params["right_tab"] = right_tab
    return "/?" + urlencode(params) if params else "/"


def _html_page(
    sessions: list,
    current_session,
    messages: list,
    active_tab: str,
    right_tab: str,
    db_path: str,
) -> str:
    """Generate the full inspector HTML page."""
    parts = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Orbit2 Inspector</title>",
        f"<style>{VIO_INSPIRED_CSS}</style>",
        "</head><body>",
        '<div class="dashboard">',
    ]

    # ---- Topbar ----
    sid = current_session.session_id if current_session else ""
    parts.append('<div class="topbar card">')
    parts.append('<div class="brand">')
    parts.append("<h1>Orbit2 Inspector</h1>")
    parts.append("<p>Operation Surface \u00b7 Transcript &amp; Debug</p>")
    parts.append("</div>")
    parts.append(f'<span class="chip">{escape(db_path)}</span>')
    parts.append("</div>")

    # ---- Sidebar ----
    parts.append('<div class="sidebar card">')
    parts.append('<h2 class="section-title">Sessions</h2>')
    if not sessions:
        parts.append('<div class="empty">No sessions found.</div>')
    else:
        for s in sessions:
            active_cls = " active" if current_session and s.session_id == current_session.session_id else ""
            url = _build_url(session_id=s.session_id, tab=active_tab, right_tab=right_tab)
            status = s.status.value if hasattr(s.status, "value") else str(s.status)
            parts.append(f'<a href="{escape(url)}" class="session-link{active_cls}">')
            parts.append(f'<div style="font-size:14px;font-family:monospace;">{escape(s.session_id)}</div>')
            parts.append(f'<div class="session-meta">{escape(status)} \u00b7 {escape(_format_local_timestamp(s.updated_at))}</div>')
            parts.append("</a>")
    parts.append("</div>")

    # ---- Main panel ----
    parts.append('<div class="main card">')

    if current_session:
        # Tab bar
        main_tabs = [
            ("transcript", "Transcript"),
            ("assembly", "Assembly"),
            ("debug", "Debug"),
        ]
        parts.append('<div class="tabbar">')
        for tab_id, tab_label in main_tabs:
            active_cls = " active" if active_tab == tab_id else ""
            url = _build_url(session_id=sid, tab=tab_id, right_tab=right_tab)
            parts.append(f'<a href="{escape(url)}" class="tab{active_cls}">{escape(tab_label)}</a>')
        parts.append("</div>")

        # Main content
        parts.append('<div class="main-panel">')
        if active_tab == "debug":
            parts.append(_render_debug_panel(current_session, messages))
        elif active_tab == "assembly":
            parts.append(_render_assembly_panel(messages))
        else:
            parts.append(_render_transcript_panel(messages))
        parts.append("</div>")
    else:
        parts.append('<div class="main-panel"><div class="empty">Select a session from the sidebar.</div></div>')

    parts.append("</div>")

    # ---- Right panel ----
    parts.append('<div class="right card">')

    if current_session:
        right_tabs = [("metadata", "Metadata"), ("raw", "Raw")]
        parts.append('<div class="tabbar">')
        for tab_id, tab_label in right_tabs:
            active_cls = " active" if right_tab == tab_id else ""
            url = _build_url(session_id=sid, tab=active_tab, right_tab=tab_id)
            parts.append(f'<a href="{escape(url)}" class="tab{active_cls}">{escape(tab_label)}</a>')
        parts.append("</div>")

        parts.append('<div class="right-panel-body">')
        if right_tab == "raw":
            parts.append(_render_raw_right_panel(current_session, messages))
        else:
            parts.append(_render_metadata_right_panel(current_session, messages))
        parts.append("</div>")
    else:
        parts.append('<div class="right-panel-body"><div class="empty">No session selected.</div></div>')

    parts.append("</div>")

    # ---- Close ----
    parts.append("</div></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTTP handler + server
# ---------------------------------------------------------------------------


def _make_handler(db_path_str: str):
    """Create a request handler class that opens a per-request store connection.

    Each request creates its own SQLiteSessionStore to avoid cross-thread
    sqlite3 sharing (ThreadingHTTPServer uses one thread per request).
    """
    _db_path = Path(db_path_str)

    class InspectorHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                self._handle_get()
            except Exception:
                tb = traceback.format_exc()
                print(f"[inspector] ERROR: {tb}", file=sys.stderr)
                body = b"Internal server error"
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def _handle_get(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            session_id = params.get("session_id", [None])[0]
            active_tab = params.get("tab", ["transcript"])[0]
            right_tab = params.get("right_tab", ["metadata"])[0]

            # Validate tab parameters against allowlist
            if active_tab not in _VALID_MAIN_TABS:
                active_tab = "transcript"
            if right_tab not in _VALID_RIGHT_TABS:
                right_tab = "metadata"

            # Per-request store connection (thread-safe)
            store = SQLiteSessionStore(_db_path)

            sessions = sorted(
                store.list_sessions(),
                key=lambda s: s.updated_at,
                reverse=True,
            )

            current_session = None
            messages: list = []

            if session_id:
                current_session = store.get_session(session_id)
            if not current_session and sessions:
                current_session = sessions[0]

            if current_session:
                messages = store.list_messages(current_session.session_id)

            html = _html_page(
                sessions=sessions,
                current_session=current_session,
                messages=messages,
                active_tab=active_tab,
                right_tab=right_tab,
                db_path=db_path_str,
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            # Route to stderr with prefix instead of suppressing
            print(f"[inspector] {fmt % args}", file=sys.stderr)

    return InspectorHandler


def serve(
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 8789,
    open_browser: bool = True,
) -> None:
    """Start the Orbit2 web inspector server.

    Args:
        db_path: Path to the SQLite sessions database.
        host: Bind address.
        port: Bind port.
        open_browser: Whether to open the browser automatically.
    """
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Database not found: {resolved}")
    handler = _make_handler(str(resolved))
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Orbit2 Inspector running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInspector stopped.")
    finally:
        server.server_close()
