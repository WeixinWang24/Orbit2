from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.capability.boundary import workspace_root_from_boundary
from src.capability.discovery import DISCOVERY_TOOL_NAME
from src.capability.mcp_servers.l3_workflow import persist_decision_response
from src.capability.tool_relationships import overlap_notices_for_tool_names
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
from src.core.runtime.workflow_decision import (
    WORKFLOW_DECISION_TOOL_NAME,
    WorkflowDecisionRoundtripError,
    build_workflow_decision_tool_definition,
    build_workflow_resume_request_from_tool_request,
    dispatch_workflow_resume_branch,
    project_decision_request_to_message,
    render_workflow_branch_result,
    workflow_decision_package_from_result,
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
        self._disclosure_strategy = disclosure_strategy or DEFAULT_DISCLOSURE_STRATEGY
        self._last_exposure_decision: ExposureDecision | None = None
        self._last_assembly_envelope: AssemblyDebugEnvelope | None = None
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

        plan = self._plan_from_transcript(session, session_id, on_partial_text)

        tool_loops = 0
        while plan.tool_requests and tool_loops < MAX_TOOL_TURNS:
            if self._capability_boundary is None:
                raise CapabilityBoundaryUnavailableError(
                    "Provider returned tool requests but no capability boundary is configured"
                )

            tool_loops += 1
            turn_index = self._next_turn_index(session_id)

            assistant_metadata: dict = {
                "source_backend": plan.source_backend,
                "model": plan.model,
                "tool_calls": [tr.model_dump() for tr in plan.tool_requests],
            }
            overlap_notices = overlap_notices_for_tool_names(
                tr.tool_name for tr in plan.tool_requests
            )
            if overlap_notices:
                assistant_metadata["tool_overlap_notice"] = overlap_notices
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

            workflow_decision_packages: list[dict[str, Any]] = []
            for tr in plan.tool_requests:
                turn_index = self._next_turn_index(session_id)
                result = self._capability_boundary.execute(tr, session_id=session_id)
                workflow_decision_package = workflow_decision_package_from_result(result)
                tool_metadata: dict = {
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "ok": result.ok,
                    "governance_outcome": result.governance_outcome,
                }
                if workflow_decision_package is not None:
                    workflow_decision_packages.append(workflow_decision_package)
                    tool_metadata["workflow_decision"] = {
                        "status": "waiting_for_decision",
                        "workflow_run_id": workflow_decision_package["workflow_run_id"],
                        "decision_id": workflow_decision_package["decision_id"],
                        "workflow_name": workflow_decision_package["workflow_name"],
                    }
                if (
                    isinstance(result.data, dict)
                    and result.tool_name == DISCOVERY_TOOL_NAME
                ):
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

            for decision_package in workflow_decision_packages:
                self._run_workflow_decision_message_flow(
                    session=session,
                    session_id=session_id,
                    decision_request_package=decision_package,
                    on_partial_text=on_partial_text,
                )

            plan = self._plan_from_transcript(session, session_id, on_partial_text)

        if plan.tool_requests and tool_loops >= MAX_TOOL_TURNS:
            turn_index = self._next_turn_index(session_id)
            exhausted_metadata: dict = {
                "source_backend": plan.source_backend,
                "model": plan.model,
                "tool_calls": [tr.model_dump() for tr in plan.tool_requests],
                "tool_loop_exhausted": True,
            }
            overlap_notices = overlap_notices_for_tool_names(
                tr.tool_name for tr in plan.tool_requests
            )
            if overlap_notices:
                exhausted_metadata["tool_overlap_notice"] = overlap_notices
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

    def _plan_from_transcript(
        self,
        session: Session,
        session_id: str,
        on_partial_text: Callable[[str], None] | None,
        *,
        use_exposed_tools: bool = True,
        tool_definitions: list[dict] | None = None,
    ) -> ExecutionPlan:
        all_messages = self._store.list_messages(session_id)
        decision: ExposureDecision | None = None
        if use_exposed_tools and self._capability_boundary is not None:
            decision = compute_exposed_tools(
                self._capability_boundary.registry, all_messages,
                strategy=self._disclosure_strategy,
            )
            self._last_exposure_decision = decision

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

        if use_exposed_tools and self._capability_boundary is not None and decision is not None:
            full_definitions = [
                d.model_dump() for d in self._capability_boundary.list_definitions()
            ]
            request.tool_definitions = filter_definitions_by_exposure(
                full_definitions, decision.exposed_tool_names
            )
        elif not use_exposed_tools:
            request.tool_definitions = tool_definitions

        self._last_assembly_envelope = build_envelope(
            assembler_name=type(self._assembler).__name__,
            transcript_message_count=len(all_messages),
            request=request,
            assembled_context=assembled_context,
            exposure_decision=decision,
        )

        return self._backend.plan_from_messages(request, on_partial_text=on_partial_text)

    def _run_workflow_decision_message_flow(
        self,
        *,
        session: Session,
        session_id: str,
        decision_request_package: dict[str, Any],
        on_partial_text: Callable[[str], None] | None,
    ) -> None:
        decision_message = project_decision_request_to_message(decision_request_package)
        turn_index = self._next_turn_index(session_id)
        self._store.save_message(ConversationMessage(
            message_id=make_message_id(),
            session_id=session_id,
            role=MessageRole.USER,
            content=decision_message.content or "",
            turn_index=turn_index,
            created_at=datetime.now(timezone.utc),
            metadata={
                "origin": "runtime_core",
                "message_type": "workflow_decision_request",
                "workflow_run_id": decision_request_package["workflow_run_id"],
                "decision_id": decision_request_package["decision_id"],
                "workflow_name": decision_request_package["workflow_name"],
            },
        ))

        decision_plan = self._plan_from_transcript(
            session,
            session_id,
            on_partial_text,
            use_exposed_tools=False,
            tool_definitions=[
                build_workflow_decision_tool_definition(decision_request_package)
            ],
        )
        turn_index = self._next_turn_index(session_id)
        response_metadata: dict[str, Any] = {
            "source_backend": decision_plan.source_backend,
            "model": decision_plan.model,
            "message_type": "workflow_decision_response",
            "workflow_run_id": decision_request_package["workflow_run_id"],
            "decision_id": decision_request_package["decision_id"],
            "workflow_name": decision_request_package["workflow_name"],
        }
        envelope = self._envelope_metadata_snapshot()
        if envelope is not None:
            response_metadata["assembly_envelope"] = envelope
        if decision_plan.tool_requests:
            response_metadata["tool_calls"] = [
                tr.model_dump() for tr in decision_plan.tool_requests
            ]
        self._store.save_message(ConversationMessage(
            message_id=make_message_id(),
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=decision_plan.final_text or "",
            turn_index=turn_index,
            created_at=datetime.now(timezone.utc),
            metadata=response_metadata,
        ))
        if len(decision_plan.tool_requests) != 1:
            raise WorkflowDecisionRoundtripError(
                "workflow decision provider response must contain exactly one decision tool request"
            )

        decision_tool_request = decision_plan.tool_requests[0]
        resume_request = build_workflow_resume_request_from_tool_request(
            decision_request_package=decision_request_package,
            tool_request=decision_tool_request,
        )
        workspace_root = workspace_root_from_boundary(self._capability_boundary)
        if workspace_root is None:
            raise CapabilityBoundaryUnavailableError(
                "workflow decision resume requires a capability boundary workspace root"
            )
        persist_decision_response(
            workspace_root=workspace_root,
            decision_request_package=decision_request_package,
            response=resume_request["decision_response"],
        )
        dispatch_result = dispatch_workflow_resume_branch(
            workspace_root=workspace_root,
            decision_request_package=decision_request_package,
            resume_request=resume_request,
        )
        turn_index = self._next_turn_index(session_id)
        self._store.save_message(ConversationMessage(
            message_id=make_message_id(),
            session_id=session_id,
            role=MessageRole.TOOL,
            content=render_workflow_branch_result(dispatch_result),
            turn_index=turn_index,
            created_at=datetime.now(timezone.utc),
            metadata={
                "tool_call_id": decision_tool_request.tool_call_id,
                "tool_name": WORKFLOW_DECISION_TOOL_NAME,
                "ok": True,
                "governance_outcome": "runtime_workflow_decision",
                "origin": "runtime_core",
                "message_type": "workflow_resume_branch_result",
                "workflow_run_id": dispatch_result["workflow_run_id"],
                "decision_id": dispatch_result["decision_id"],
                "selected_option_id": dispatch_result["selected_option_id"],
                "branch_type": dispatch_result["branch_type"],
                "status": dispatch_result["status"],
            },
        ))

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
        return self._last_exposure_decision

    def _next_turn_index(self, session_id: str) -> int:
        return len(self._store.list_messages(session_id)) + 1
