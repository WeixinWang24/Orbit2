"""Assembly debug envelope — Knowledge Surface projection artifact.

The envelope exists so the Operation Surface inspector can visibly surface
*assembled* provider-facing context and the source state it came from without
collapsing three distinct things into one:
  1. canonical transcript truth (ConversationMessage rows in the store)
  2. assembled provider-facing payload (TurnRequest actually sent)
  3. the raw provider transport wire payload (not in scope for this slice)

The envelope is a read-only projection of (1) and (2). It is produced at
planning time by the SessionManager, serialized into assistant-message
metadata, and then rendered by the inspector. The inspector must NEVER
recompute an envelope from transcript state — it projects whatever the
runtime persisted.

Previews are bounded. Full assembled payload is intentionally not
serialized; this is a debug aid, not a full provenance browser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.runtime.models import TurnRequest
from src.knowledge.exposure import ExposureDecision
from src.knowledge.models import AssembledContext

_FRAGMENT_PREVIEW_CHARS = 160
_MESSAGE_PREVIEW_CHARS = 160
_SYSTEM_PREVIEW_CHARS = 400
_MAX_MESSAGE_PREVIEW_ROWS = 40


def _preview(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    compact = text.replace("\r\n", "\n")
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "\u2026"


@dataclass
class FragmentPreview:
    fragment_name: str
    visibility_scope: str
    priority: int
    content_preview: str
    content_length: int


@dataclass
class AssembledMessagePreview:
    role: str
    content_preview: str | None
    content_length: int
    tool_calls_count: int
    has_tool_call_id: bool


@dataclass
class AssemblyDebugEnvelope:
    """Bounded per-turn projection of what the assembler produced.

    Content is deliberately lossy: previews are truncated, the full transcript
    is represented only by its length, and the full provider-facing messages
    list is represented by at most `_MAX_MESSAGE_PREVIEW_ROWS` rows. The
    envelope carries enough signal to debug "what did the provider see this
    turn and why does it look like that", not to reconstruct inputs.
    """

    assembler_name: str
    transcript_message_count: int
    instruction_fragments: list[FragmentPreview] = field(default_factory=list)
    assembled_system_preview: str | None = None
    assembled_message_count: int = 0
    assembled_messages_preview: list[AssembledMessagePreview] = field(
        default_factory=list,
    )
    assembled_messages_truncated: bool = False
    exposed_tool_groups: list[str] | None = None
    exposed_tool_names: list[str] | None = None
    rejected_reveal_requests: list[str] | None = None
    assembler_metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_envelope(
    *,
    assembler_name: str,
    transcript_message_count: int,
    request: TurnRequest,
    assembled_context: AssembledContext | None,
    exposure_decision: ExposureDecision | None,
) -> AssemblyDebugEnvelope:
    """Build the envelope from the same artifacts the runtime already has.

    `assembled_context` is optional because not every assembler exposes a
    structured intermediate. When it is None, fragment/metadata fields
    fall back to what is recoverable from `request` alone.
    """
    fragments: list[FragmentPreview] = []
    assembler_metadata: dict[str, Any] = {}
    if assembled_context is not None:
        for fragment in assembled_context.instruction_fragments:
            content = fragment.content or ""
            fragments.append(
                FragmentPreview(
                    fragment_name=fragment.fragment_name,
                    visibility_scope=fragment.visibility_scope,
                    priority=fragment.priority,
                    content_preview=_preview(content, _FRAGMENT_PREVIEW_CHARS) or "",
                    content_length=len(content),
                )
            )
        assembler_metadata = dict(assembled_context.metadata)

    preview_rows: list[AssembledMessagePreview] = []
    for msg in request.messages[:_MAX_MESSAGE_PREVIEW_ROWS]:
        content = msg.content or ""
        preview_rows.append(
            AssembledMessagePreview(
                role=msg.role,
                content_preview=_preview(content, _MESSAGE_PREVIEW_CHARS),
                content_length=len(content),
                tool_calls_count=len(msg.tool_calls) if msg.tool_calls else 0,
                has_tool_call_id=bool(msg.tool_call_id),
            )
        )
    truncated = len(request.messages) > _MAX_MESSAGE_PREVIEW_ROWS

    exposed_groups: list[str] | None = None
    exposed_names: list[str] | None = None
    rejected: list[str] | None = None
    if exposure_decision is not None:
        exposed_groups = list(exposure_decision.active_reveal_groups)
        exposed_names = sorted(exposure_decision.exposed_tool_names)
        rejected = list(exposure_decision.rejected_reveal_requests)

    return AssemblyDebugEnvelope(
        assembler_name=assembler_name,
        transcript_message_count=transcript_message_count,
        instruction_fragments=fragments,
        assembled_system_preview=_preview(request.system, _SYSTEM_PREVIEW_CHARS),
        assembled_message_count=len(request.messages),
        assembled_messages_preview=preview_rows,
        assembled_messages_truncated=truncated,
        exposed_tool_groups=exposed_groups,
        exposed_tool_names=exposed_names,
        rejected_reveal_requests=rejected,
        assembler_metadata=assembler_metadata,
    )
