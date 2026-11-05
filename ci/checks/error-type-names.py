#!/usr/bin/env python3
"""Name every `thiserror` type with an `Error` suffix, as std does (`ParseIntError`).

A type counts when its derive names `thiserror::Error`, or `Error` in a file that
imports `thiserror`. The derive's item is the next `struct`, `enum` or `union`.
"""

import re
import subprocess
import sys
from pathlib import Path

DERIVE = re.compile(r"^[ \t]*#\[derive\(([^)]*)\)\]", re.MULTILINE)
ITEM = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|union)\s+(\w+)", re.MULTILINE
)
THISERROR_PATH = re.compile(r"\bthiserror::Error\b")
BARE_ERROR = re.compile(r"(?<!::)\bError\b")


def tracked_rust() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.rs"], capture_output=True, text=True, check=True
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def violations(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    imports_thiserror = "thiserror" in src
    found = []
    for derive in DERIVE.finditer(src):
        names = derive.group(1)
        if not (
            THISERROR_PATH.search(names)
            or (imports_thiserror and BARE_ERROR.search(names))
        ):
            continue
        item = ITEM.search(src, derive.end())
        if item is None or item.group(1).endswith("Error"):
            continue
        line = src.count("\n", 0, item.start()) + 1
        found.append(f"{path}:{line}: {item.group(1)}")
    return found


def main() -> int:
    files = tracked_rust()
    found = [v for path in files for v in violations(path)]
    if found:
        print(
            "error: a type deriving thiserror::Error must end in `Error`.",
            file=sys.stderr,
        )
        print(
            "Rename `<Condition>` to `<Condition>Error`, "
            "and `Invalid<X>` to `<X>Error`.",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("\n".join(found), file=sys.stderr)
        return 1
    print(f"error type names OK ({len(files)} Rust files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
