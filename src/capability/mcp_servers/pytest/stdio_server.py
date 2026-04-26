from __future__ import annotations

from src.capability.mcp_servers.l2_toolchain.pytest import stdio_server as _impl
from src.capability.mcp_servers.l2_toolchain.pytest.stdio_server import (
    DEFAULT_MAX_CHARS,
    PYTEST_TIMEOUT_SECONDS,
    SERVER_NAME,
    WORKSPACE_ROOT_ENV,
    _pytest_diagnose_failures_result,
    _run_pytest_structured_result,
    _validate_args,
    _workspace_root,
    mcp,
    pytest_diagnose_failures,
    run_pytest_structured,
    toolchain_get_run,
    toolchain_get_step,
    toolchain_read_artifact_region,
)

subprocess = _impl.subprocess


if __name__ == "__main__":
    mcp.run()
