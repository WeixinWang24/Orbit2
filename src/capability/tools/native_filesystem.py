from __future__ import annotations

from pathlib import Path

from src.capability.models import ToolDefinition, ToolResult, is_protected_relative_path
from src.capability.tools.base import Tool


class _WorkspaceScopedTool(Tool):
    """Shared base for native filesystem tools bound to a workspace root.

    CapabilityBoundary normally governs the `path` argument before `execute`
    runs, but `_resolve` also enforces workspace containment and
    protected-prefix denial directly so the tool remains safe under direct
    invocation (tests, scripts, or any code path that hasn't routed through
    the boundary yet).
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    def _resolve(self, path: str) -> Path | ToolResult:
        target = (self._workspace_root / path).resolve()
        try:
            relative = target.relative_to(self._workspace_root).as_posix()
        except ValueError:
            return ToolResult(ok=False, content="path escapes workspace")
        matched = is_protected_relative_path(relative)
        if matched is not None:
            return ToolResult(
                ok=False,
                content=f"path targets protected location: {matched}",
            )
        return target


class ReadFileTool(_WorkspaceScopedTool):
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
                    },
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
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        return ToolResult(ok=True, content=target.read_text(encoding="utf-8"), data={"path": str(target)})


class WriteFileTool(_WorkspaceScopedTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__write_file",
            description="Write UTF-8 text to a file within the workspace. Creates parent directories if needed. Overwrites existing files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "UTF-8 text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    def execute(self, *, path: str, content: str) -> ToolResult:
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"wrote {target}",
            data={"mutation_kind": "write_file", "path": str(target)},
        )


class ReplaceInFileTool(_WorkspaceScopedTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__replace_in_file",
            description="Replace the first occurrence of old_text with new_text in a workspace file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_text": {"type": "string", "description": "Exact text to find."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def environment_check_kind(self) -> str:
        return "path_exists"

    def execute(self, *, path: str, old_text: str, new_text: str) -> ToolResult:
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        content = target.read_text(encoding="utf-8")
        if old_text not in content:
            return ToolResult(
                ok=False,
                content="old_text not found",
                data={
                    "mutation_kind": "replace_in_file",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                },
            )
        updated = content.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"replaced text in {target}",
            data={
                "mutation_kind": "replace_in_file",
                "path": str(target),
                "replacement_count": 1,
                "before_excerpt": old_text,
                "after_excerpt": new_text,
            },
        )


class ReplaceAllInFileTool(_WorkspaceScopedTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__replace_all_in_file",
            description="Replace every occurrence of old_text with new_text in a workspace file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_text": {"type": "string", "description": "Exact text to find."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def environment_check_kind(self) -> str:
        return "path_exists"

    def execute(self, *, path: str, old_text: str, new_text: str) -> ToolResult:
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        content = target.read_text(encoding="utf-8")
        replacement_count = content.count(old_text)
        if replacement_count == 0:
            return ToolResult(
                ok=False,
                content="old_text not found",
                data={
                    "mutation_kind": "replace_all_in_file",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                    "replacement_count": 0,
                },
            )
        updated = content.replace(old_text, new_text)
        target.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"replaced {replacement_count} occurrence(s) in {target}",
            data={
                "mutation_kind": "replace_all_in_file",
                "path": str(target),
                "replacement_count": replacement_count,
                "before_excerpt": old_text,
                "after_excerpt": new_text,
            },
        )


class ReplaceBlockInFileTool(_WorkspaceScopedTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__replace_block_in_file",
            description="Replace a uniquely matching block of text in a workspace file. Fails if the block is absent or matches more than one region.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_block": {"type": "string", "description": "Exact block to find (must match exactly once)."},
                    "new_block": {"type": "string", "description": "Replacement block."},
                },
                "required": ["path", "old_block", "new_block"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def environment_check_kind(self) -> str:
        return "path_exists"

    def execute(self, *, path: str, old_block: str, new_block: str) -> ToolResult:
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        content = target.read_text(encoding="utf-8")
        match_count = content.count(old_block)
        if match_count == 0:
            return ToolResult(
                ok=False,
                content="old_block not found",
                data={
                    "mutation_kind": "replace_block_in_file",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                    "match_count": 0,
                    "replacement_count": 0,
                },
            )
        if match_count > 1:
            return ToolResult(
                ok=False,
                content="old_block matched multiple regions",
                data={
                    "mutation_kind": "replace_block_in_file",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                    "match_count": match_count,
                    "replacement_count": 0,
                },
            )
        updated = content.replace(old_block, new_block, 1)
        target.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"replaced block in {target}",
            data={
                "mutation_kind": "replace_block_in_file",
                "path": str(target),
                "match_count": 1,
                "replacement_count": 1,
                "before_excerpt": old_block,
                "after_excerpt": new_block,
            },
        )


class ApplyExactHunkTool(_WorkspaceScopedTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="native__apply_exact_hunk",
            description="Apply an exact hunk replacement anchored by before_context and after_context. The full before_context+old_block+after_context must match exactly once in the file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "before_context": {"type": "string", "description": "Text immediately preceding old_block (stays unchanged)."},
                    "old_block": {"type": "string", "description": "Block to replace."},
                    "after_context": {"type": "string", "description": "Text immediately following old_block (stays unchanged)."},
                    "new_block": {"type": "string", "description": "Replacement for old_block."},
                },
                "required": ["path", "before_context", "old_block", "after_context", "new_block"],
            },
        )

    @property
    def side_effect_class(self) -> str:
        return "write"

    @property
    def requires_approval(self) -> bool:
        return True

    @property
    def environment_check_kind(self) -> str:
        return "path_exists"

    def execute(
        self,
        *,
        path: str,
        before_context: str,
        old_block: str,
        after_context: str,
        new_block: str,
    ) -> ToolResult:
        target = self._resolve(path)
        if isinstance(target, ToolResult):
            return target
        if not target.exists() or not target.is_file():
            return ToolResult(ok=False, content="file not found")
        content = target.read_text(encoding="utf-8")
        old_hunk = before_context + old_block + after_context
        new_hunk = before_context + new_block + after_context
        match_count = content.count(old_hunk)
        if match_count == 0:
            return ToolResult(
                ok=False,
                content="exact hunk not found",
                data={
                    "mutation_kind": "apply_exact_hunk",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                    "match_count": 0,
                    "replacement_count": 0,
                    "change_summary": "0 exact hunk matches",
                },
            )
        if match_count > 1:
            return ToolResult(
                ok=False,
                content="exact hunk matched multiple regions",
                data={
                    "mutation_kind": "apply_exact_hunk",
                    "failure_layer": "tool_semantic",
                    "path": str(target),
                    "match_count": match_count,
                    "replacement_count": 0,
                    "change_summary": f"{match_count} exact hunk matches",
                },
            )
        updated = content.replace(old_hunk, new_hunk, 1)
        target.write_text(updated, encoding="utf-8")
        return ToolResult(
            ok=True,
            content=f"applied exact hunk in {target}",
            data={
                "mutation_kind": "apply_exact_hunk",
                "path": str(target),
                "match_count": 1,
                "replacement_count": 1,
                "before_excerpt": old_block,
                "after_excerpt": new_block,
                "change_summary": "1 exact hunk applied",
            },
        )
