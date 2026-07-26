#!/usr/bin/env python3
"""Verify every Lambda source file parses as Python 3.

Docker COPY is byte-for-byte, so a source file Python cannot parse (e.g. text saved
as cp1252 instead of UTF-8) builds and pushes cleanly, then fails at Lambda import
time with Runtime.UserCodeSyntaxError. Catch it before the build instead.

Used by both scripts/lambda-build-push.sh and the CI test job.

Usage: check_lambda_sources.py [lambdas_dir]
"""

from __future__ import annotations

import pathlib
import sys

DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "lambdas"


def check(root: pathlib.Path) -> list[str]:
    """Return a list of "path: error" strings, empty when every file compiles."""
    failures = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            compile(path.read_bytes(), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"{path}: {exc}")
    return failures


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1]).resolve() if len(argv) > 1 else DEFAULT_DIR
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    failures = check(root)
    if failures:
        print("ERROR: Lambda sources failed to compile:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"OK: all Lambda sources under {root} compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
