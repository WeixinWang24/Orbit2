"""Knowledge-side thin adapter over the governance disclosure strategy.

Handoff 19 created this module with a single hard-coded compute function
embedded here. Handoff 23 moved the policy decision to
`src.governance.disclosure` (where task-shape-driven disclosure rules
naturally live) and left this module as a thin adapter so existing
callers (`SessionManager._plan_with_tools`, tests, inspector) don't need
to track the move.

`ExposureDecision` is re-exported from its new home. Legacy call sites
that pass only `(registry, messages)` keep working — they get the
single-reveal strategy (the Handoff-19 default).
"""
from __future__ import annotations

from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ConversationMessage, MessageRole
from src.governance.disclosure import (
    DEFAULT_DISCLOSURE_STRATEGY,
    REVEAL_BATCH_REQUEST_MARKER,
    REVEAL_REQUEST_MARKER,
    DisclosureStrategy,
    ExposureDecision,
)


def compute_exposed_tools(
    registry: CapabilityRegistry,
    messages: list[ConversationMessage],
    *,
    strategy: DisclosureStrategy | None = None,
) -> ExposureDecision:
    """Back-compat entry point. Defaults to the single-reveal strategy
    unless the caller passes an explicit disclosure strategy."""
    active_strategy = strategy or DEFAULT_DISCLOSURE_STRATEGY
    return active_strategy.compute(registry, messages)


def collect_reveal_requests(messages: list[ConversationMessage]) -> list[str]:
    """First-seen-order list of literal `reveal_request` markers on TOOL
    metadata. Still used directly by some tests and by the inspector's
    exposure view — preserved here for stability. Does NOT surface
    batch/all-safe markers; callers that need those should read them
    explicitly via `src.governance.disclosure` marker keys.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for message in messages:
        if message.role != MessageRole.TOOL:
            continue
        if not isinstance(message.metadata, dict):
            continue
        requested = message.metadata.get(REVEAL_REQUEST_MARKER)
        if isinstance(requested, str) and requested and requested not in seen_set:
            seen.append(requested)
            seen_set.add(requested)
    return seen


def filter_definitions_by_exposure(
    definitions: list[dict],
    exposed_tool_names: set[str],
) -> list[dict]:
    """Order-preserving filter helper. Unchanged from Handoff 19."""
    return [d for d in definitions if d.get("name") in exposed_tool_names]


__all__ = [
    "ExposureDecision",
    "compute_exposed_tools",
    "collect_reveal_requests",
    "filter_definitions_by_exposure",
    "REVEAL_REQUEST_MARKER",
    "REVEAL_BATCH_REQUEST_MARKER",
]
