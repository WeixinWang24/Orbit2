from __future__ import annotations

from typing import Any

from src.capability.models import (
    CapabilityLayer,
    ToolDefinition,
    ToolResult,
    is_protected_relative_path,
)
from src.capability.mcp.client import McpClient
from src.capability.mcp.governance import (
    DEFAULT_MCP_GOVERNANCE,
    McpGovernanceMetadata,
    resolve_filesystem_mcp_target_path,
)
from src.capability.mcp.layers import classify_mcp_capability_layer
from src.capability.mcp.models import McpClientBootstrap, McpToolDescriptor
from src.capability.tools.base import Tool


# Marker key on `ToolResult.data` that signals a tool-layer governance denial.
# The CapabilityBoundary surfaces this in `CapabilityResult.governance_outcome`
# instead of the default "allowed" when present.
GOVERNANCE_DENIED_MARKER = "_governance_denied_reason"


class McpToolWrapper(Tool):
    """Wrap one MCP-discovered tool as an Orbit2 `Tool`.

    The wrapper is the Capability Surface seam for MCP-backed capabilities:
    the same `CapabilityRegistry` + `CapabilityBoundary` pair governs both
    native and MCP tools. MCP execution never bypasses Orbit2's attachment
    path — every call enters through `CapabilityBoundary.execute`, has its
    arguments validated against the schema, and returns a `CapabilityResult`.

    Family-aware governance: when the attachment layer resolves a
    `McpGovernanceMetadata` for the tool (e.g. via Handoff 16's overlay), the
    wrapper surfaces it through the Tool metadata attributes. When no
    metadata is supplied the wrapper falls back to the conservative default
    (`side_effect_class='unknown'`, `requires_approval=True`) — matching the
    Handoff 13 posture for opaque MCP calls.

    The metadata drives `side_effect_class` / `requires_approval` /
    `environment_check_kind`. These are metadata-only today: the Governance
    Surface approval-gate slice will wire a consumer.
    """

    def __init__(
        self,
        *,
        descriptor: McpToolDescriptor,
        client: McpClient,
        governance: McpGovernanceMetadata | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._client = client
        self._governance: McpGovernanceMetadata = governance or dict(DEFAULT_MCP_GOVERNANCE)
        # Family-aware override of the boundary's path-arg whitelist.
        # Git tools expose `path` as a git pathspec, not a filesystem path
        # relative to workspace — applying filesystem containment to it
        # would false-positive (e.g. pathspecs under `.git/` subtree during
        # object-database diffs). Git tools opt out of boundary filesystem
        # governance; future family wrappers can do the same.
        server = descriptor.server_name.strip().lower()
        if server == "git":
            self._governance_path_arg_keys: tuple[str, ...] | None = ()
        else:
            self._governance_path_arg_keys = None

        # All MCP wrappers start hidden; the harness explicitly flips
        # select families on when it decides the default-exposed minimum.
        self._default_exposed = False

    # Handoff 19 reveal-group family map. Single source of truth: used by
    # the `reveal_group` property below. Adding a new family server here
    # means both read-side and write-side paths update atomically — there
    # is no independent fallback in __init__ that could disagree.
    _MCP_FAMILY_READ_GROUPS: dict[str, str] = {
        "filesystem": "mcp_fs_read",
        "git": "mcp_git_read",
    }
    _MCP_FAMILY_WRITE_GROUPS: dict[str, str] = {
        "filesystem": "mcp_fs_mutate",
        "git": "mcp_git_mutate",
        "process": "mcp_process",
    }
    _MCP_DIAGNOSTICS_SERVERS: frozenset[str] = frozenset({"pytest", "ruff", "mypy"})

    @property
    def governance_path_arg_keys(self) -> tuple[str, ...] | None:
        return self._governance_path_arg_keys

    @property
    def reveal_group(self) -> str:
        server = self._descriptor.server_name.strip().lower()
        is_write = self._governance["side_effect_class"] == "write"
        if server in self._MCP_DIAGNOSTICS_SERVERS:
            return "mcp_diagnostics"
        if is_write:
            write_group = self._MCP_FAMILY_WRITE_GROUPS.get(server)
            if write_group is not None:
                return write_group
        read_group = self._MCP_FAMILY_READ_GROUPS.get(server)
        if read_group is not None:
            return read_group
        return f"mcp_{server}"

    @property
    def default_exposed(self) -> bool:
        return self._default_exposed

    def set_default_exposed(self, exposed: bool) -> None:
        """Operator-surface hook for declaring whether this wrapper is part
        of the default-exposed minimum. Called by the harness after attach
        so the decision lives with the CLI policy layer, not the wrapper."""
        self._default_exposed = bool(exposed)

    @property
    def capability_layer(self) -> CapabilityLayer:
        return classify_mcp_capability_layer(
            server_name=self._descriptor.server_name,
            original_tool_name=self._descriptor.original_name,
        )

    @property
    def descriptor(self) -> McpToolDescriptor:
        return self._descriptor

    @property
    def governance(self) -> McpGovernanceMetadata:
        return dict(self._governance)

    @property
    def definition(self) -> ToolDefinition:
        parameters = self._descriptor.input_schema or {
            "type": "object",
            "properties": {},
        }
        description = self._descriptor.description or (
            f"MCP tool {self._descriptor.original_name!r} "
            f"from server {self._descriptor.server_name!r}."
        )
        return ToolDefinition(
            name=self._descriptor.orbit_tool_name,
            description=description,
            parameters=parameters,
        )

    @property
    def side_effect_class(self) -> str:
        return self._governance["side_effect_class"]

    @property
    def requires_approval(self) -> bool:
        return self._governance["requires_approval"]

    @property
    def environment_check_kind(self) -> str:
        return self._governance["environment_check_kind"]

    def execute(self, **kwargs: Any) -> ToolResult:
        return self._client.call_tool(self._descriptor.original_name, kwargs)


class FilesystemMcpToolWrapper(McpToolWrapper):
    """MCP wrapper that knows the `filesystem` family's allowed-root/target-path
    contract (Handoff 16).

    Handoff 16 explicitly requires Orbit2-side target-path resolution for
    filesystem MCP tools: the server's `ORBIT_WORKSPACE_ROOT` env var or
    trailing positional arg defines the allowed root, and tool calls whose
    resolved path escapes that root (or targets a protected prefix like
    `.env`, `.runtime`, `.git`) must be refused on the Orbit2 side before
    the MCP server is invoked — preventing the server from becoming a
    filesystem escape hatch even if the native governance whitelist keys
    are absent or bypassed.

    Tool-layer governance refusal: this subclass returns a `ToolResult`
    carrying `GOVERNANCE_DENIED_MARKER` on its `data`; the CapabilityBoundary
    surfaces it as `governance_outcome='denied: ...'`.
    """

    def __init__(
        self,
        *,
        descriptor: McpToolDescriptor,
        client: McpClient,
        governance: McpGovernanceMetadata | None,
        bootstrap: McpClientBootstrap,
    ) -> None:
        super().__init__(descriptor=descriptor, client=client, governance=governance)
        self._bootstrap = bootstrap
        # FilesystemMcpToolWrapper is the sole enforcement point for
        # filesystem-family path governance: it knows the MCP server's
        # `allowed_root` (which may differ from the CapabilityBoundary's
        # `workspace_root`). Letting the boundary ALSO govern the `path`
        # arg produces a double-check where the two enforcement layers use
        # different reference roots. Opt out of the boundary's generic path
        # whitelist so the wrapper is the single source of truth.
        self._governance_path_arg_keys = ()

    def execute(self, **kwargs: Any) -> ToolResult:
        refusal = self._governance_check(kwargs)
        if refusal is not None:
            return refusal
        return super().execute(**kwargs)

    def _governance_check(self, arguments: dict[str, Any]) -> ToolResult | None:
        from src.capability.mcp.governance import filesystem_server_allowed_root

        path_arg = arguments.get("path")

        allowed_root = filesystem_server_allowed_root(
            list(self._bootstrap.args),
            dict(self._bootstrap.env) if self._bootstrap.env else None,
        )
        if allowed_root is None:
            # A filesystem MCP server with no resolvable allowed root cannot
            # be governed by Orbit2. Refuse unconditionally so misconfigured
            # servers are unusable rather than silently bypassed. Also
            # independent of whether the current call carries a path arg —
            # future tools may exist without path args yet still mutate
            # filesystem state, and we must not rely on per-call argument
            # shape to enforce governance.
            reason = (
                "filesystem MCP server has no resolvable allowed root "
                "(ORBIT_WORKSPACE_ROOT env or trailing positional arg required)"
            )
            return _governance_refusal(reason)

        target = resolve_filesystem_mcp_target_path(
            input_payload=arguments,
            server_args=list(self._bootstrap.args),
            server_env=dict(self._bootstrap.env) if self._bootstrap.env else None,
        )
        if target is None:
            return None

        try:
            relative = target.relative_to(allowed_root).as_posix()
        except ValueError:
            return _governance_refusal(
                f"path {path_arg!r} escapes filesystem MCP allowed root"
            )

        matched = is_protected_relative_path(relative)
        if matched is not None:
            return _governance_refusal(
                f"path targets protected location: {matched}"
            )
        return None


def _governance_refusal(reason: str) -> ToolResult:
    return ToolResult(
        ok=False,
        content=f"governance denied: {reason}",
        data={GOVERNANCE_DENIED_MARKER: reason},
    )
