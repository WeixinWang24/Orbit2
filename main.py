"""Orbit2 main entrypoint."""

from __future__ import annotations

import sys
from typing import Sequence

from src.operation.cli.codex_auth import main as codex_auth_main
from src.operation.cli.harness import main as harness_main


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"codex-auth", "auth"}:
        return codex_auth_main(args[1:])
    harness_main(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
