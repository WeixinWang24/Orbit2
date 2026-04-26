"""Workspace-instructions context block (Handoff 29).

User-customizable system-prompt supplement sourced from a repo-local
`orbit.md` file at the converged runtime-root path. The file is read
lazily through the Knowledge Surface collector pattern so file-reading
responsibility does not smear into CLI or provider glue, and so
canonical transcript truth (ADR-0002) remains distinct from the
assembled provider-facing payload.

The collector accepts an explicit `workspace_root` — it never consults
process cwd. Absence of `orbit.md` is a first-class state (`exists=False`)
that the disclosure policy can suppress cleanly; no exception bubbles out
of the assembly path.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path

from src.knowledge.models import ContextFragment

WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME = "workspace_instructions"
WORKSPACE_INSTRUCTIONS_VISIBILITY_SCOPE = "workspace_instruction"
WORKSPACE_INSTRUCTIONS_PRIORITY = 95
WORKSPACE_INSTRUCTIONS_FILENAME = "orbit.md"
MAX_WORKSPACE_INSTRUCTIONS_BYTES = 65536

_OPEN_TAG = "<workspace-instructions>"
_CLOSE_TAG = "</workspace-instructions>"


@dataclass(frozen=True)
class WorkspaceInstructionsSnapshot:
    source_path: Path
    exists: bool
    content: str
    truncated: bool


class WorkspaceInstructionsCollector:
    def __init__(
        self,
        workspace_root: Path,
        *,
        filename: str = WORKSPACE_INSTRUCTIONS_FILENAME,
        max_bytes: int = MAX_WORKSPACE_INSTRUCTIONS_BYTES,
    ) -> None:
        self._workspace_root = workspace_root
        self._filename = filename
        self._max_bytes = max_bytes

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    @property
    def source_path(self) -> Path:
        return self._workspace_root / self._filename

    def collect(self) -> WorkspaceInstructionsSnapshot:
        path = self.source_path
        try:
            if not path.is_file():
                return WorkspaceInstructionsSnapshot(
                    source_path=path,
                    exists=False,
                    content="",
                    truncated=False,
                )
            raw = path.read_bytes()
        except OSError:
            return WorkspaceInstructionsSnapshot(
                source_path=path,
                exists=False,
                content="",
                truncated=False,
            )
        truncated = len(raw) > self._max_bytes
        if truncated:
            # Use the incremental decoder with final=False so an
            # incomplete UTF-8 sequence at our byte-budget boundary is
            # BUFFERED rather than replaced with U+FFFD. Malformed bytes
            # elsewhere in the file still surface as U+FFFD via
            # errors="replace" — that signal belongs to the file, not to
            # our truncation.
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            text = decoder.decode(raw[: self._max_bytes], final=False)
        else:
            text = raw.decode("utf-8", errors="replace")
        return WorkspaceInstructionsSnapshot(
            source_path=path,
            exists=True,
            content=text,
            truncated=truncated,
        )


def build_workspace_instructions_fragment(
    snapshot: WorkspaceInstructionsSnapshot,
    *,
    policy_name: str,
) -> ContextFragment:
    body = snapshot.content.strip("\n")
    if body:
        content = "\n".join([_OPEN_TAG, body, _CLOSE_TAG])
    else:
        content = "\n".join([_OPEN_TAG, _CLOSE_TAG])
    return ContextFragment(
        fragment_name=WORKSPACE_INSTRUCTIONS_FRAGMENT_NAME,
        visibility_scope=WORKSPACE_INSTRUCTIONS_VISIBILITY_SCOPE,
        content=content,
        priority=WORKSPACE_INSTRUCTIONS_PRIORITY,
        metadata={
            "origin": "workspace_instructions",
            "policy_name": policy_name,
            "source_path": str(snapshot.source_path),
            "content_length": len(snapshot.content),
            "truncated": snapshot.truncated,
        },
    )
