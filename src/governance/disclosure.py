"""Disclosure strategies — Governance Surface ownership of progressive
exposure policy (Handoff 23).

Handoff 19 established progressive exposure with a single implicit rule:
monotonic, one-group-per-call reveals widen the provider-visible tool
surface. That rule was fine for the first slice but it's a policy choice,
not an architectural invariant. Different task shapes want different
disclosure modes — e.g. a pure exploration/overview run benefits from
revealing every safe tool in one batch, while a mutation-heavy workflow
still wants the single-step reveal discipline.

This module factors the policy out into a plug-and-play seam:

- `DisclosureStrategy` — abstract base. `compute(registry, messages)` →
  `ExposureDecision`. Adding a new mode means adding a subclass, not
  editing `compute_exposed_tools`.
- `SingleRevealDisclosureStrategy` — the Handoff-19 default. Scans the
  transcript for `reveal_request` markers and widens one group at a time.
- `BatchRevealDisclosureStrategy` — new. Understands three markers:
  `reveal_request` (single group, back-compat), `reveal_batch_request`
  (explicit list of groups), `reveal_all_safe_request` (every group whose
  member tools are all `side_effect_class == "safe"`). Provides the
  "overview ergonomics" improvement Handoff 23 calls out without
  collapsing progressive exposure into an uncontrolled escape hatch.

The strategy is Governance Surface owned (it decides a policy question)
but the computation itself stays inside a single planning turn — no
cross-session state, no persisted policy decisions. Reset logic lives
with approvals, not with disclosure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from src.capability.models import CapabilityLayer
from src.core.runtime.models import ConversationMessage, MessageRole

if TYPE_CHECKING:
    from src.capability.registry import CapabilityRegistry


# Marker keys carried on TOOL-role message metadata. The session manager
# already restricts promotion of these markers to the discovery tool's
# result (Handoff 19 audit HIGH-1 fix), so adding new marker names here
# is governance-visible.
REVEAL_REQUEST_MARKER = "reveal_request"                    # single group
REVEAL_BATCH_REQUEST_MARKER = "reveal_batch_request"        # list[str]
REVEAL_ALL_SAFE_REQUEST_MARKER = "reveal_all_safe_request"  # bool


# Kept here so both strategies and the session-manager persistence layer
# share one source of truth about which markers are legal disclosure signals.
DISCLOSURE_MARKER_KEYS: frozenset[str] = frozenset({
    REVEAL_REQUEST_MARKER,
    REVEAL_BATCH_REQUEST_MARKER,
    REVEAL_ALL_SAFE_REQUEST_MARKER,
})


@dataclass
class ExposureDecision:
    """Result of one disclosure computation. Re-homed from
    `src.knowledge.exposure` into the governance surface where policy
    naturally lives. Kept shape-compatible with the original dataclass
    so existing callers (inspector, session manager) don't churn.
    """

    exposed_tool_names: set[str] = field(default_factory=set)
    active_reveal_groups: list[str] = field(default_factory=list)
    exposure_reason: dict[str, str] = field(default_factory=dict)
    rejected_reveal_requests: list[str] = field(default_factory=list)
    # Name of the strategy that produced this decision, for inspection.
    strategy_name: str = ""

    def to_inspection_metadata(self) -> dict:
        return {
            "exposed_tool_names": sorted(self.exposed_tool_names),
            "active_reveal_groups": sorted(self.active_reveal_groups),
            "exposure_reason": dict(self.exposure_reason),
            "rejected_reveal_requests": list(self.rejected_reveal_requests),
            "strategy_name": self.strategy_name,
        }


# ---------------------------------------------------------------------------
# Base strategy + shared helpers
# ---------------------------------------------------------------------------


class DisclosureStrategy(ABC):
    """Policy hook: given a registry inventory and the transcript so far,
    return the ExposureDecision that should govern the next provider turn.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str: ...

    @abstractmethod
    def compute(
        self, registry: CapabilityRegistry, messages: list[ConversationMessage]
    ) -> ExposureDecision: ...


def _scan_tool_messages(
    messages: list[ConversationMessage],
) -> list[tuple[ConversationMessage, dict]]:
    """Yield (message, metadata) pairs for every TOOL-role message whose
    metadata is a dict. Centralized so strategies iterate consistently."""
    return [
        (m, m.metadata)
        for m in messages
        if m.role == MessageRole.TOOL and isinstance(m.metadata, dict)
    ]


def _collect_registry_snapshot(
    registry: CapabilityRegistry,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    """Return `(tool_group, tool_default_exposed, tool_side_effect)`. Built
    once per compute so long reveal chains don't repeatedly re-read the
    registry."""
    tool_group: dict[str, str] = {}
    tool_default: dict[str, bool] = {}
    tool_side_effect: dict[str, str] = {}
    for name in registry.list_names():
        tool = registry.get(name)
        if tool is None:
            continue
        tool_group[name] = tool.reveal_group
        tool_default[name] = tool.default_exposed
        tool_side_effect[name] = tool.side_effect_class
    return tool_group, tool_default, tool_side_effect


def _collect_layer_registry_snapshot(
    registry: CapabilityRegistry,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str], dict[str, CapabilityLayer], dict[str, bool]]:
    tool_group: dict[str, str] = {}
    tool_default: dict[str, bool] = {}
    tool_side_effect: dict[str, str] = {}
    tool_layer: dict[str, CapabilityLayer] = {}
    tool_requires_approval: dict[str, bool] = {}
    for name in registry.list_names():
        tool = registry.get(name)
        if tool is None:
            continue
        tool_group[name] = tool.reveal_group
        tool_default[name] = tool.default_exposed
        tool_side_effect[name] = tool.side_effect_class
        tool_layer[name] = tool.capability_layer
        tool_requires_approval[name] = tool.requires_approval
    return tool_group, tool_default, tool_side_effect, tool_layer, tool_requires_approval


def _add_defaults(
    decision: ExposureDecision,
    tool_group: dict[str, str],
    tool_default: dict[str, bool],
) -> None:
    for name, is_default in tool_default.items():
        if is_default:
            decision.exposed_tool_names.add(name)
            decision.exposure_reason[name] = "default_exposed"
    default_groups = {tool_group[name] for name in decision.exposed_tool_names}
    for group in sorted(default_groups):
        if group not in decision.active_reveal_groups:
            decision.active_reveal_groups.append(group)


def _widen_with_group(
    decision: ExposureDecision,
    tool_group: dict[str, str],
    available_groups: set[str],
    requested: str,
    reason_verb: str,
) -> None:
    """Shared widening: given a requested group, add every tool in that
    group to the decision with an explicit reason string. No-op if the
    group is unknown (caller adds to rejected_reveal_requests)."""
    if requested not in available_groups:
        decision.rejected_reveal_requests.append(requested)
        return
    if requested not in decision.active_reveal_groups:
        decision.active_reveal_groups.append(requested)
    for name, group in tool_group.items():
        if group == requested and name not in decision.exposed_tool_names:
            decision.exposed_tool_names.add(name)
            decision.exposure_reason[name] = f"{reason_verb}:{requested}"


# ---------------------------------------------------------------------------
# Single-reveal strategy (Handoff 19 behavior)
# ---------------------------------------------------------------------------


class SingleRevealDisclosureStrategy(DisclosureStrategy):
    """One-group-per-request monotonic exposure. Matches the Handoff-19
    default. Reads only the literal `reveal_request` marker; ignores any
    batch or all-safe markers (which are off-policy for this strategy —
    they simply don't widen exposure under single-reveal mode).
    """

    @property
    def strategy_name(self) -> str:
        return "single_reveal"

    def compute(
        self, registry: CapabilityRegistry, messages: list[ConversationMessage]
    ) -> ExposureDecision:
        decision = ExposureDecision(strategy_name=self.strategy_name)
        tool_group, tool_default, _ = _collect_registry_snapshot(registry)
        _add_defaults(decision, tool_group, tool_default)
        available_groups = set(tool_group.values())
        for _msg, meta in _scan_tool_messages(messages):
            requested = meta.get(REVEAL_REQUEST_MARKER)
            if isinstance(requested, str) and requested:
                _widen_with_group(
                    decision, tool_group, available_groups,
                    requested, reason_verb="revealed_by",
                )
            # Surface strategy mismatches: a session running SingleReveal
            # that sees batch / all-safe markers records them as rejected
            # rather than silently ignoring them. Makes the mismatch
            # visible to operators via `rejected_reveal_requests` and the
            # inspector's exposure panel. Audit MED-3 fix.
            batch = meta.get(REVEAL_BATCH_REQUEST_MARKER)
            if isinstance(batch, list):
                for b in batch:
                    if isinstance(b, str) and b:
                        decision.rejected_reveal_requests.append(
                            f"ignored_under_single_reveal:batch:{b}"
                        )
            all_safe = meta.get(REVEAL_ALL_SAFE_REQUEST_MARKER)
            if all_safe is True:
                decision.rejected_reveal_requests.append(
                    "ignored_under_single_reveal:all_safe"
                )
        return decision


# ---------------------------------------------------------------------------
# Batch / overview strategy (Handoff 23)
# ---------------------------------------------------------------------------


class BatchRevealDisclosureStrategy(DisclosureStrategy):
    """Overview-oriented disclosure. Understands three markers:

    - `reveal_request: str` — single group (same as single-reveal).
    - `reveal_batch_request: list[str]` — explicit multi-group unlock.
    - `reveal_all_safe_request: bool` — unlock every group whose tools are
      all `side_effect_class == "safe"`. Mutation-bearing groups are
      explicitly NOT unlocked by this marker; operators who want them must
      go through single-reveal or batch-reveal with explicit names.

    The strategy still ignores non-discovery markers (the session manager
    already filters those at persistence time). Under batch mode, exposure
    still widens monotonically — a session that opens with overview +
    later switches to single-reveal keeps the batch-revealed groups.
    """

    @property
    def strategy_name(self) -> str:
        return "batch_reveal"

    def compute(
        self, registry: CapabilityRegistry, messages: list[ConversationMessage]
    ) -> ExposureDecision:
        decision = ExposureDecision(strategy_name=self.strategy_name)
        tool_group, tool_default, tool_side_effect = _collect_registry_snapshot(registry)
        _add_defaults(decision, tool_group, tool_default)
        available_groups = set(tool_group.values())

        # Compute the "all-safe" group set: every group where ALL its
        # member tools are safe. A mixed group (any write member) does NOT
        # qualify for reveal_all_safe — that's the escape-hatch guard.
        safe_groups = _compute_safe_groups(tool_group, tool_side_effect)

        for _msg, meta in _scan_tool_messages(messages):
            # Single-group marker — back-compat with SingleReveal.
            single = meta.get(REVEAL_REQUEST_MARKER)
            if isinstance(single, str) and single:
                _widen_with_group(
                    decision, tool_group, available_groups,
                    single, reason_verb="revealed_by",
                )
            # Batch marker — list of groups.
            batch = meta.get(REVEAL_BATCH_REQUEST_MARKER)
            if isinstance(batch, list):
                for requested in batch:
                    if isinstance(requested, str) and requested:
                        _widen_with_group(
                            decision, tool_group, available_groups,
                            requested, reason_verb="batch_revealed_by",
                        )
            # All-safe marker — flips on every safe group.
            all_safe = meta.get(REVEAL_ALL_SAFE_REQUEST_MARKER)
            if all_safe is True:
                for safe_group in sorted(safe_groups):
                    _widen_with_group(
                        decision, tool_group, available_groups,
                        safe_group, reason_verb="all_safe_revealed",
                    )

        return decision


class LayerAwareDisclosureStrategy(DisclosureStrategy):
    @property
    def strategy_name(self) -> str:
        return "layer_aware_v0"

    def compute(
        self, registry: CapabilityRegistry, messages: list[ConversationMessage]
    ) -> ExposureDecision:
        decision = ExposureDecision(strategy_name=self.strategy_name)
        (
            tool_group,
            _tool_default,
            tool_side_effect,
            tool_layer,
            tool_requires_approval,
        ) = _collect_layer_registry_snapshot(registry)
        available_groups = set(tool_group.values())
        self._add_layer_defaults(
            decision,
            tool_group=tool_group,
            tool_side_effect=tool_side_effect,
            tool_layer=tool_layer,
            tool_requires_approval=tool_requires_approval,
        )

        safe_context_groups = _compute_safe_context_groups(
            tool_group=tool_group,
            tool_side_effect=tool_side_effect,
            tool_layer=tool_layer,
            tool_requires_approval=tool_requires_approval,
        )
        for _msg, meta in _scan_tool_messages(messages):
            single = meta.get(REVEAL_REQUEST_MARKER)
            if isinstance(single, str) and single:
                _widen_with_group(
                    decision,
                    tool_group,
                    available_groups,
                    single,
                    reason_verb="layer_revealed_by",
                )

            batch = meta.get(REVEAL_BATCH_REQUEST_MARKER)
            if isinstance(batch, list):
                for requested in batch:
                    if isinstance(requested, str) and requested:
                        _widen_with_group(
                            decision,
                            tool_group,
                            available_groups,
                            requested,
                            reason_verb="layer_batch_revealed_by",
                        )

            all_safe = meta.get(REVEAL_ALL_SAFE_REQUEST_MARKER)
            if all_safe is True:
                for group in sorted(safe_context_groups):
                    _widen_with_group(
                        decision,
                        tool_group,
                        available_groups,
                        group,
                        reason_verb="layer_all_safe_revealed",
                    )
        return decision

    def _add_layer_defaults(
        self,
        decision: ExposureDecision,
        *,
        tool_group: dict[str, str],
        tool_side_effect: dict[str, str],
        tool_layer: dict[str, CapabilityLayer],
        tool_requires_approval: dict[str, bool],
    ) -> None:
        for name, layer in tool_layer.items():
            if layer not in {CapabilityLayer.TOOLCHAIN, CapabilityLayer.WORKFLOW}:
                continue
            if tool_side_effect.get(name) != "safe":
                continue
            if tool_requires_approval.get(name) is True:
                continue
            decision.exposed_tool_names.add(name)
            decision.exposure_reason[name] = f"layer_default:{layer.value}"
        for group in sorted({tool_group[name] for name in decision.exposed_tool_names}):
            if group not in decision.active_reveal_groups:
                decision.active_reveal_groups.append(group)


def _compute_safe_groups(
    tool_group: dict[str, str], tool_side_effect: dict[str, str]
) -> set[str]:
    """A group is safe iff it has at least one member AND every member has
    `side_effect_class == "safe"`. The non-empty requirement is a hardening
    guard (audit CRIT-1): `all()` over an empty iterable returns `True`, so
    a momentarily-empty group name would otherwise qualify; if a write tool
    later attached to that same group, the replayed `reveal_all_safe_request`
    marker would surface the mutation on a subsequent turn. Requiring at
    least one member makes the predicate robust under dynamic attachment.
    """
    group_members: dict[str, list[str]] = {}
    for name, group in tool_group.items():
        group_members.setdefault(group, []).append(name)
    safe: set[str] = set()
    for group, members in group_members.items():
        if members and all(tool_side_effect.get(m) == "safe" for m in members):
            safe.add(group)
    return safe


def _compute_safe_context_groups(
    *,
    tool_group: dict[str, str],
    tool_side_effect: dict[str, str],
    tool_layer: dict[str, CapabilityLayer],
    tool_requires_approval: dict[str, bool],
) -> set[str]:
    group_members: dict[str, list[str]] = {}
    for name, group in tool_group.items():
        group_members.setdefault(group, []).append(name)
    safe: set[str] = set()
    for group, members in group_members.items():
        if not members:
            continue
        if any(
            tool_layer.get(member) not in {
                CapabilityLayer.TOOLCHAIN,
                CapabilityLayer.WORKFLOW,
            }
            for member in members
        ):
            continue
        if all(
            tool_side_effect.get(member) == "safe"
            and tool_requires_approval.get(member) is False
            for member in members
        ):
            safe.add(group)
    return safe


# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------


DEFAULT_DISCLOSURE_STRATEGY: DisclosureStrategy = SingleRevealDisclosureStrategy()
