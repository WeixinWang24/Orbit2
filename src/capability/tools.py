from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.capability.models import ToolDefinition, ToolResult


class Tool(ABC):
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @property
    def side_effect_class(self) -> str:
        return "safe"

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def environment_check_kind(self) -> str:
        return "none"

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult: ...


class ReadFileTool(Tool):
    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__read_file",
            description="Read the contents of a text file within the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "safe"

    @property
    def environment_check_kind(self) -> str:
        return "path_exists"

    def execute(self, *, path: str) -> ToolResult:
        target = (self._workspace_root / path).resolve()
        try:
            target.relative_to(self._workspace_root)
        except ValueError:
            return ToolResult(ok=False, content="path escapes workspace")
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        return ToolResult(ok=True, content=target.read_text(), data={"path": str(target)})
