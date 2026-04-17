from __future__ import annotations

from src.governance.approval import (
    ApprovalDecision,
    ApprovalMemory,
    ApprovalPolicy,
    ApprovalRequest,
)


class RevealGroupSessionApprovalPolicy(ApprovalPolicy):
    """First concrete policy: session-scoped reuse keyed on `reveal_group`.

    Conservative by design — similarity is literal reveal-group match, not
    fuzzy semantic matching. An operator granting `allow_similar_in_session`
    for one tool in `native_fs_mutate` allows later tools in the same group
    within the same session without further prompting; any other group still
    prompts. `/reset-permission` clears the per-session set so the next call
    in any group prompts again.
    """

    def __init__(self, memory: ApprovalMemory | None = None) -> None:
        self._memory = memory or ApprovalMemory()

    @property
    def memory(self) -> ApprovalMemory:
        return self._memory

    def check_reuse(self, request: ApprovalRequest) -> bool:
        if not request.reveal_group:
            return False
        return self._memory.is_approved(request.session_id, request.reveal_group)

    def record(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        if decision != ApprovalDecision.ALLOW_SIMILAR_IN_SESSION:
            return
        if not request.reveal_group:
            return
        self._memory.remember(request.session_id, request.reveal_group)

    def reset_session(self, session_id: str) -> None:
        self._memory.reset(session_id)
