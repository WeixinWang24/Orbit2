from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from src.capability.discovery import DISCOVERY_TOOL_NAME
from src.governance.disclosure import (
    DEFAULT_DISCLOSURE_STRATEGY,
    DISCLOSURE_MARKER_KEYS,
    REVEAL_ALL_SAFE_REQUEST_MARKER,
    REVEAL_BATCH_REQUEST_MARKER,
    REVEAL_REQUEST_MARKER,
    DisclosureStrategy,
    ExposureDecision,
)
from src.knowledge.assembly import (
    AssemblyDebugEnvelope,
    ContextAssembler,
    StructuredContextAssembler,
    build_envelope,
)
from src.knowledge.exposure import (
    compute_exposed_tools,
    filter_definitions_by_exposure,
)
from src.core.providers.base import ExecutionBackend
from src.core.runtime.models import (
    ConversationMessage,
    ExecutionPlan,
    MessageRole,
    Session,
    SessionStatus,
    make_message_id,
    make_session_id,
)
from src.config.runtime import MAX_TOOL_TURNS
from src.core.store.base import SessionStore


class CapabilityBoundaryUnavailableError(RuntimeError):
    pass


class SessionManager:
    def __init__(
        self,
        backend: ExecutionBackend,
        store: SessionStore,
        assembler: ContextAssembler | None = None,
        capability_boundary: object | None = None,
        disclosure_strategy: DisclosureStrategy | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._assembler = assembler or StructuredContextAssembler()
        self._capability_boundary = capability_boundary
        # Progressive-disclosure policy. Default preserves Handoff-19
        # single-reveal behavior. Operators who want overview ergonomics
        # pass `BatchRevealDisclosureStrategy()`; future modes can be
        # added as new DisclosureStrategy subclasses without touching
        # this constructor.
        self._disclosure_strategy = disclosure_strategy or DEFAULT_DISCLOSURE_STRATEGY
        # Initialised once at construction. `_plan_with_tools` only overwrites
        # it on a successful exposure computation, so a later call that
        # raises before reaching the boundary branch leaves the previous
        # decision visible for post-mortem inspection.
        self._last_exposure_decision: ExposureDecision | None = None
        # Assembly debug envelope captured at the most recent planning call.
        # Consumed by run_turn when writing the subsequent assistant message
        # so the envelope is persisted alongside the transcript turn it
        # actually corresponds to. Purely a projection artifact — the
        # envelope never feeds back into assembly or routing.
        self._last_assembly_envelope: AssemblyDebugEnvelope | None = None
        # Wire the discovery tool's summary to the manager's live exposure
        # decision so the model sees a summary consistent with the tool
        # definitions it actually receives this turn. Lookup is best-effort:
        # if the discovery tool isn't attached, staged exposure still works,
        # it just falls back to the static `default_exposed` flag in the
        # discovery summary (acceptable — there's nothing to be inconsistent
        # with).
        self._wire_discovery_exposure_provider()

    def create_session(self, *, system_prompt: str | None = None) -> Session:
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=make_session_id(),
            backend_name=self._backend.backend_name,
            system_prompt=system_prompt,
            status=SessionStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        self._store.save_session(session)
        return session

    def get_session(self, session_id: str) -> Session:
        session = self._store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        return session

    def list_sessions(self) -> list[Session]:
        return self._store.list_sessions()

    def list_messages(self, session_id: str) -> list[ConversationMessage]:
        return self._store.list_messages(session_id)

    def delete_all_sessions(self) -> int:
        return self._store.delete_all_sessions()

    def run_turn(
        self,
        session_id: str,
        user_input: str,
        *,
        on_partial_text: Callable[[str], None] | None = None,
    ) -> ExecutionPlan:
        session = self.get_session(session_id)
        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session {session_id} is not active")

        existing = self._store.list_messages(session_id)
        turn_index = len(existing) + 1

        user_msg = ConversationMessage(
            message_id=make_message_id(),
            session_id=session_id,
            role=MessageRole.USER,
            content=user_input,
            turn_index=turn_index,
            created_at=datetime.now(timezone.utc),
        )
        self._store.save_message(user_msg)

        plan = self._plan_with_tools(session, session_id, on_partial_text)

        tool_loops = 0
        while plan.tool_requests and tool_loops < MAX_TOOL_TURNS:
            if self._capability_boundary is None:
                raise CapabilityBoundaryUnavailableError(
                    "Provider returned tool requests but no capability boundary is configured"
                )

            tool_loops += 1
            turn_index = self._next_turn_index(session_id)

            # Save assistant message with tool_calls metadata
            assistant_metadata: dict = {
                "source_backend": plan.source_backend,
                "model": plan.model,
                "tool_calls": [tr.model_dump() for tr in plan.tool_requests],
            }
            envelope = self._envelope_metadata_snapshot()
            if envelope is not None:
                assistant_metadata["assembly_envelope"] = envelope
            self._store.save_message(ConversationMessage(
                message_id=make_message_id(),
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=plan.final_text or "",
                turn_index=turn_index,
                created_at=datetime.now(timezone.utc),
                metadata=assistant_metadata,
            ))

            # Execute each tool through the governed capability boundary
            for tr in plan.tool_requests:
                turn_index = self._next_turn_index(session_id)
                result = self._capability_boundary.execute(tr, session_id=session_id)
                tool_metadata: dict = {
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "ok": result.ok,
                    "governance_outcome": result.governance_outcome,
                }
                # Preserve progressive-exposure reveal requests on the
                # TOOL-role transcript metadata so the next turn's assembler
                # can compute a wider exposed tool subset. Only surface the
                # marker itself; the rest of `result.data` stays with the
                # live result.
                #
                # HARDENING (Handoff 19 audit HIGH-1): only the discovery tool
                # is allowed to emit a reveal_request marker. Any other tool's
                # `data["reveal_request"]` is ignored here so a compromised
                # or misbehaving MCP server cannot forge cross-turn exposure
                # by echoing the marker in its own result payload.
                if (
                    isinstance(result.data, dict)
                    and result.tool_name == DISCOVERY_TOOL_NAME
                ):
                    # Preserve every known disclosure marker the discovery
                    # tool may have set: single-group reveal (Handoff 19),
                    # batch list (Handoff 23), or the one-shot all-safe
                    # flag. Promotion stays restricted to the discovery
                    # tool so a non-discovery result still cannot forge
                    # cross-turn exposure.
                    for marker_key in DISCLOSURE_MARKER_KEYS:
                        value = result.data.get(marker_key)
                        if marker_key == REVEAL_REQUEST_MARKER:
                            if isinstance(value, str) and value:
                                tool_metadata[marker_key] = value
                        elif marker_key == REVEAL_BATCH_REQUEST_MARKER:
                            if isinstance(value, list) and all(
                                isinstance(v, str) and v for v in value
                            ):
                                tool_metadata[marker_key] = list(value)
                        elif marker_key == REVEAL_ALL_SAFE_REQUEST_MARKER:
                            if value is True:
                                tool_metadata[marker_key] = True
                self._store.save_message(ConversationMessage(
                    message_id=make_message_id(),
                    session_id=session_id,
                    role=MessageRole.TOOL,
                    content=result.content,
                    turn_index=turn_index,
                    created_at=datetime.now(timezone.utc),
                    metadata=tool_metadata,
                ))

            # Continue with updated transcript
            plan = self._plan_with_tools(session, session_id, on_partial_text)

        # Handle MAX_TOOL_TURNS exhaustion: save the last assistant+tool_calls
        # and a sentinel tool result so the transcript is never silently truncated
        if plan.tool_requests and tool_loops >= MAX_TOOL_TURNS:
            turn_index = self._next_turn_index(session_id)
            exhausted_metadata: dict = {
                "source_backend": plan.source_backend,
                "model": plan.model,
                "tool_calls": [tr.model_dump() for tr in plan.tool_requests],
                "tool_loop_exhausted": True,
            }
            envelope = self._envelope_metadata_snapshot()
            if envelope is not None:
                exhausted_metadata["assembly_envelope"] = envelope
            self._store.save_message(ConversationMessage(
                message_id=make_message_id(),
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=plan.final_text or "",
                turn_index=turn_index,
                created_at=datetime.now(timezone.utc),
                metadata=exhausted_metadata,
            ))
            for tr in plan.tool_requests:
                turn_index = self._next_turn_index(session_id)
                self._store.save_message(ConversationMessage(
                    message_id=make_message_id(),
                    session_id=session_id,
                    role=MessageRole.TOOL,
                    content=f"tool loop exhausted after {MAX_TOOL_TURNS} iterations",
                    turn_index=turn_index,
                    created_at=datetime.now(timezone.utc),
                    metadata={
                        "tool_call_id": tr.tool_call_id,
                        "tool_name": tr.tool_name,
                        "ok": False,
                        "governance_outcome": "denied_loop_exhaustion",
                    },
                ))
        elif plan.final_text is not None:
            turn_index = self._next_turn_index(session_id)
            final_metadata: dict = {
                "source_backend": plan.source_backend,
                "model": plan.model,
            }
            envelope = self._envelope_metadata_snapshot()
            if envelope is not None:
                final_metadata["assembly_envelope"] = envelope
            self._store.save_message(ConversationMessage(
                message_id=make_message_id(),
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=plan.final_text,
                turn_index=turn_index,
                created_at=datetime.now(timezone.utc),
                metadata=final_metadata,
            ))

        updated = session.model_copy(update={"updated_at": datetime.now(timezone.utc)})
        self._store.save_session(updated)

        return plan

    def _plan_with_tools(
        self,
        session: Session,
        session_id: str,
        on_partial_text: Callable[[str], None] | None,
    ) -> ExecutionPlan:
        all_messages = self._store.list_messages(session_id)

        # Compute progressive-exposure subset BEFORE assembling context so the
        # assembler can emit awareness-shaping fragments (Handoff 27) keyed to
        # the current turn's visible/hidden reveal groups. Execution still
        # routes through the full boundary; exposure only constrains which
        # tools the provider sees this turn.
        decision: ExposureDecision | None = None
        if self._capability_boundary is not None:
            decision = compute_exposed_tools(
                self._capability_boundary.registry, all_messages,
                strategy=self._disclosure_strategy,
            )
            self._last_exposure_decision = decision

        # Prefer the structured intermediate when the assembler exposes it so
        # the debug envelope can project instruction fragments. Fall back to
        # the plain assemble() path for assemblers without that shape.
        assembled_context = None
        if hasattr(self._assembler, "assemble_structured"):
            assembled_context = self._assembler.assemble_structured(
                all_messages,
                system_prompt=session.system_prompt,
                exposure_decision=decision,
            )
            request = assembled_context.to_turn_request()
        else:
            request = self._assembler.assemble(
                all_messages, system_prompt=session.system_prompt,
            )

        if self._capability_boundary is not None and decision is not None:
            full_definitions = [
                d.model_dump() for d in self._capability_boundary.list_definitions()
            ]
            request.tool_definitions = filter_definitions_by_exposure(
                full_definitions, decision.exposed_tool_names
            )

        # Build the envelope AFTER exposure has been applied to `request`,
        # so snapshots match what the backend actually received. Each call
        # to `_plan_with_tools` overwrites `_last_assembly_envelope`; the
        # pairing invariant is that every assistant-message write site in
        # `run_turn` snapshots the envelope immediately for the plan that
        # was just produced, before the next `_plan_with_tools` call can
        # overwrite it. If a future `build_envelope` extension reads
        # `request.tool_definitions`, it will see the filtered list.
        self._last_assembly_envelope = build_envelope(
            assembler_name=type(self._assembler).__name__,
            transcript_message_count=len(all_messages),
            request=request,
            assembled_context=assembled_context,
            exposure_decision=decision,
        )

        return self._backend.plan_from_messages(request, on_partial_text=on_partial_text)

    def _envelope_metadata_snapshot(self) -> dict | None:
        envelope = self._last_assembly_envelope
        if envelope is None:
            return None
        return envelope.to_metadata_dict()

    def _wire_discovery_exposure_provider(self) -> None:
        if self._capability_boundary is None:
            return
        registry = getattr(self._capability_boundary, "registry", None)
        if registry is None:
            return
        discovery = registry.get(DISCOVERY_TOOL_NAME)
        if discovery is None:
            return
        # The tool was constructed without a provider — attach ours now.
        if not hasattr(discovery, "_active_reveal_groups_provider"):
            return
        discovery._active_reveal_groups_provider = self._active_reveal_groups_snapshot

    def _active_reveal_groups_snapshot(self) -> list[str]:
        decision = self._last_exposure_decision
        if decision is None:
            return []
        return list(decision.active_reveal_groups)

    @property
    def last_exposure_decision(self) -> ExposureDecision | None:
        """Debug/inspection accessor for the most recent successful
        staged-exposure computation. `None` means the manager has no
        capability boundary configured (exposure filtering is a no-op
        there). The attribute is not cleared between turns; operators can
        read it after an exception to inspect the last-known decision.
        """
        return self._last_exposure_decision

    def _next_turn_index(self, session_id: str) -> int:
        return len(self._store.list_messages(session_id)) + 1
