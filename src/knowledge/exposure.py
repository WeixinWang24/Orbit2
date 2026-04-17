"""Turn-scoped progressive tool exposure (Handoff 19, 19B).

Computes the subset of the full capability inventory that should be visible
to the provider on the current turn. The subset starts from the
`default_exposed` tools and widens to include any reveal groups the
provider has explicitly requested via recent TOOL-role transcript
messages (the discovery tool writes a `reveal_request` marker into TOOL
metadata).

This is the Knowledge Surface's assembly seam for staged exposure: given
the registry inventory + the transcript, decide what the backend sees on
this turn. Execution continues to route through the full `CapabilityBoundary`
— exposure only constrains what the model is offered to call, not what the
boundary accepts if asked. That asymmetry is intentional: a misbehaving
provider that fabricates a hidden tool name still goes through the
registry's `denied_unknown_tool` path (the tool ISN'T hidden from the
registry, only from the provider's list).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.capability.registry import CapabilityRegistry
from src.core.runtime.models import ConversationMessage, MessageRole


@dataclass
class ExposureDecision:
    """Result of one round of staged-exposure computation.

    Attributes:
    - `exposed_tool_names`: alphabetical set of tool names the provider
      should see this turn.
    - `active_reveal_groups`: reveal groups included in the exposed set
      (default groups + revealed groups).
    - `exposure_reason`: per-tool string explaining why it's exposed
      (`"default_exposed"`, `"revealed_by:<group>"`). Debug/inspection only;
      not machine-consumed.
    - `rejected_reveal_requests`: reveal requests found in the transcript
      that didn't match any known group (unrecognized group names). Kept
      so callers can surface them in inspection traces.
    """

    exposed_tool_names: set[str] = field(default_factory=set)
    active_reveal_groups: list[str] = field(default_factory=list)
    exposure_reason: dict[str, str] = field(default_factory=dict)
    rejected_reveal_requests: list[str] = field(default_factory=list)

    def to_inspection_metadata(self) -> dict:
        return {
            "exposed_tool_names": sorted(self.exposed_tool_names),
            "active_reveal_groups": sorted(self.active_reveal_groups),
            "exposure_reason": dict(self.exposure_reason),
            "rejected_reveal_requests": list(self.rejected_reveal_requests),
        }


def collect_reveal_requests(messages: list[ConversationMessage]) -> list[str]:
    """Walk the transcript and return every reveal group requested via a
    TOOL-role `reveal_request` metadata marker, in first-seen order. A group
    requested on turn N stays exposed for every subsequent turn in the
    session — Handoff 19's minimum contract is monotonic widening, not
    per-turn decay."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for message in messages:
        if message.role != MessageRole.TOOL:
            continue
        if not isinstance(message.metadata, dict):
            continue
        requested = message.metadata.get("reveal_request")
        if isinstance(requested, str) and requested and requested not in seen_set:
            seen.append(requested)
            seen_set.add(requested)
    return seen


def compute_exposed_tools(
    registry: CapabilityRegistry,
    messages: list[ConversationMessage],
) -> ExposureDecision:
    """Compute the ExposureDecision for the next provider turn.

    Discovery of available reveal groups and the default-exposed subset
    both come from the live registry — this keeps the decision consistent
    with whatever has been attached since the last call.
    """
    decision = ExposureDecision()

    # Map every registered tool to its reveal group and default-exposed
    # flag. Built once so a long request list doesn't repeatedly re-inspect
    # the registry.
    tool_group: dict[str, str] = {}
    tool_default: dict[str, bool] = {}
    for name in registry.list_names():
        tool = registry.get(name)
        if tool is None:
            continue
        tool_group[name] = tool.reveal_group
        tool_default[name] = tool.default_exposed

    # Step 1: include the always-exposed default set.
    for name, is_default in tool_default.items():
        if is_default:
            decision.exposed_tool_names.add(name)
            decision.exposure_reason[name] = "default_exposed"
    default_groups = {
        tool_group[name] for name in decision.exposed_tool_names
    }
    for group in sorted(default_groups):
        if group not in decision.active_reveal_groups:
            decision.active_reveal_groups.append(group)

    # Step 2: widen by reveal requests seen in the transcript.
    available_groups = set(tool_group.values())
    for requested in collect_reveal_requests(messages):
        if requested not in available_groups:
            decision.rejected_reveal_requests.append(requested)
            continue
        if requested not in decision.active_reveal_groups:
            decision.active_reveal_groups.append(requested)
        for name, group in tool_group.items():
            if group == requested and name not in decision.exposed_tool_names:
                decision.exposed_tool_names.add(name)
                decision.exposure_reason[name] = f"revealed_by:{requested}"

    return decision


def filter_definitions_by_exposure(
    definitions: list[dict],
    exposed_tool_names: set[str],
) -> list[dict]:
    """Given the full list of `ToolDefinition.model_dump()` dicts the
    provider could in principle see, return only those in
    `exposed_tool_names`. Preserves input order."""
    return [d for d in definitions if d.get("name") in exposed_tool_names]
