"""Launch the Orbit2 Web Inspector bound to the same runtime root as the CLI."""

from __future__ import annotations

import argparse

from src.operation.inspector.web_inspector import serve
from src.core.runtime.paths import default_db_path, resolve_runtime_root
from src.core.store.sqlite import SQLiteSessionStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit2 Web Inspector")
    parser.add_argument(
        "--runtime-root",
        default=None,
        help=(
            "Effective runtime/store root. Overrides ORBIT2_RUNTIME_ROOT. "
            "Defaults to the Orbit2 repo checkout. Must match the root the CLI writes to."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser")
    args = parser.parse_args()

    runtime_root = resolve_runtime_root(args.runtime_root)
    db_path = default_db_path(runtime_root)
    print(
        f"runtime_root={runtime_root.path} (source={runtime_root.source})  store={db_path}"
    )
    # Ensure the DB file and schema exist so an inspector launched before the
    # CLI (or against an empty isolated root) shows "no sessions" instead of
    # a missing-file error.
    SQLiteSessionStore(db_path).close()
    serve(str(db_path), host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
