from __future__ import annotations

from abc import ABC, abstractmethod

from src.runtime.models import ConversationMessage, Session


class SessionStore(ABC):
    @abstractmethod
    def save_session(self, session: Session) -> None: ...

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None: ...

    @abstractmethod
    def list_sessions(self) -> list[Session]: ...

    @abstractmethod
    def save_message(self, message: ConversationMessage) -> None: ...

    @abstractmethod
    def list_messages(self, session_id: str) -> list[ConversationMessage]: ...

    @abstractmethod
    def delete_all_sessions(self) -> int:
        """Delete all sessions and their messages. Returns count of deleted sessions."""
        ...
