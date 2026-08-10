#!/usr/bin/env python3
"""Enforce the import-grouping convention rustfmt can't express.

`use` imports — at file top level **and** inside nested modules — must form
blank-line-separated groups, in this order:

  1. std / core / alloc
  2. third-party crates (crates.io)
  3. workspace crates (pipette_*)
  4. crate-local, with first-segments ordered self → super → crate

rustfmt only knows std / external / local, so it can't split workspace crates
from third-party. Within a group, rustfmt owns lexicographic order; this check
only enforces the self/super/crate relative order inside the crate-local group
(so `use crate::…` before `use super::*` is rejected).

Scope: every same-indent run of plain `use` statements (joined across
multi-line `use a::{ ... };`). `pub use` re-exports are ignored. Comments and
attributes inside a block are skipped so they cannot hide order violations.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

STD = {"std", "core", "alloc"}
# rustfmt's local-import order (not pure alphabetical).
LOCAL_ORDER = {"self": 0, "super": 1, "crate": 2}
RANK_NAME = {0: "std", 1: "third-party", 2: "workspace", 3: "crate-local"}

USE = re.compile(r"^(\s*)use\s")
PUB_USE = re.compile(r"^(\s*)pub(?:\([^)]*\))?\s+use\s")  # incl. pub(crate)/pub(super)
# Comments + attributes stay inside the block (same as the old top-level check).
NEUTRAL = re.compile(r"^\s*(//|#!?\[)")
FIRST_SEGMENT = re.compile(r"\s*use\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)")
BLANK = re.compile(r"^\s*$")

HINT = (
    "Group `use` imports (file-level and nested), blank-line separated:\n"
    "  std -> third-party -> workspace (pipette_*) -> self/super/crate\n"
    "Within crate-local: self, then super, then crate (rustfmt order)."
)


def rank(seg: str) -> int:
    if seg in STD:
        return 0
    if seg in LOCAL_ORDER:
        return 3
    if seg.startswith("pipette_"):
        return 2
    return 1


def sort_key(seg: str) -> tuple:
    """Group rank, then self/super/crate order inside crate-local."""
    r = rank(seg)
    if r == 3:
        return (r, LOCAL_ORDER.get(seg, 99))
    return (r, 0)


def leading_ws(line: str) -> str:
    return re.match(r"^(\s*)", line).group(1)


def order_violation(prev_key, key, blank_before):
    """Message for a `use` with `key` following one with `prev_key`, or None."""
    if prev_key is None:
        return None
    prev_r, r = prev_key[0], key[0]
    if key < prev_key:
        if r < prev_r:
            return f"{RANK_NAME[r]} import after the {RANK_NAME[prev_r]} group"
        return "crate-local imports must be ordered self, super, crate"
    if r > prev_r and not blank_before:
        return f"{RANK_NAME[prev_r]}/{RANK_NAME[r]} groups need a blank line between"
    return None


def check(path):
    """Yield (line, message) for each grouping violation in any import block."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    n = len(lines)
    idx = 0
    while idx < n:
        line = lines[idx]
        if not USE.match(line) or PUB_USE.match(line):
            idx += 1
            continue

        indent = leading_ws(line)
        prev_key = None
        blank_before = False
        j = idx
        while j < n:
            L = lines[j]
            if BLANK.match(L):
                blank_before = blank_before or prev_key is not None
                j += 1
                continue
            if leading_ws(L) != indent:
                break
            if USE.match(L) and not PUB_USE.match(L):
                stmt, k = L, j
                while ";" not in stmt and k + 1 < n:
                    k += 1
                    stmt += " " + lines[k]
                m = FIRST_SEGMENT.match(stmt)
                seg = m.group(1) if m else ""
                key = sort_key(seg)
                msg = order_violation(prev_key, key, blank_before)
                if msg:
                    yield j + 1, msg
                prev_key, blank_before = key, False
                j = k + 1
                continue
            if NEUTRAL.match(L):
                j += 1
                continue
            break
        idx = j if j > idx else idx + 1


def check_source(src: str) -> list[tuple[int, str]]:
    """Run [`check`] on `src` via a temp `.rs` file; return violation list."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.rs"
        path.write_text(src, encoding="utf-8")
        return list(check(path))


def self_test() -> None:
    """Regression fixtures for the scanner. Raises AssertionError on failure."""
    # (name, source, expected message substring or None if clean)
    cases = [
        (
            "ok nested groups",
            "mod t {\n"
            "    use std::io;\n"
            "\n"
            "    use anyhow::Context;\n"
            "\n"
            "    use pipette_ops::X;\n"
            "\n"
            "    use super::*;\n"
            "}\n",
            None,
        ),
        (
            "std after crate-local",
            "mod t {\n    use super::*;\n    use std::io;\n}\n",
            "std import after the crate-local group",
        ),
        (
            "missing blank between groups",
            "mod t {\n    use std::io;\n    use super::*;\n}\n",
            "groups need a blank line between",
        ),
        (
            "comment does not hide missing blank",
            "use std::io;\n// note\nuse super::x;\n",
            "groups need a blank line between",
        ),
        (
            "comment does not hide order violation",
            "use super::*;\n// note\nuse std::io;\n",
            "std import after the crate-local group",
        ),
        (
            "attr does not end block",
            "use std::io;\n#[allow(unused_imports)]\nuse super::x;\n",
            "groups need a blank line between",
        ),
        (
            "crate before super",
            "mod t {\n    use crate::x;\n    use super::*;\n}\n",
            "crate-local imports must be ordered self, super, crate",
        ),
        (
            "super before crate ok",
            "mod t {\n    use super::*;\n    use crate::x;\n}\n",
            None,
        ),
        (
            "self super crate ok",
            "mod t {\n    use self::a;\n    use super::*;\n    use crate::x;\n}\n",
            None,
        ),
        (
            "nested module scanned",
            "mod outer {\n"
            "    use std::io;\n"
            "\n"
            "    use super::*;\n"
            "\n"
            "    mod inner {\n"
            "        use super::*;\n"
            "        use std::fs;\n"
            "    }\n"
            "}\n",
            "std import after the crate-local group",
        ),
        (
            "pub use ignored mid-block",
            "use std::io;\npub use crate::x;\nuse super::y;\n",
            None,
        ),
    ]
    failed = []
    for name, src, expect in cases:
        got = check_source(src)
        msgs = [m for _, m in got]
        if expect is None:
            if got:
                failed.append(f"{name}: expected clean, got {got}")
        elif not any(expect in m for m in msgs):
            failed.append(f"{name}: expected message containing {expect!r}, got {got}")
    if failed:
        detail = "\n  ".join(failed)
        raise AssertionError(f"import-groups self-test failed:\n  {detail}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only_self_test = argv == ["--self-test"]
    if argv and not only_self_test:
        print(f"usage: {sys.argv[0]} [--self-test]", file=sys.stderr)
        return 2

    # Fixtures always run first so CI catches scanner regressions.
    try:
        self_test()
    except AssertionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if only_self_test:
        print("import-groups self-test OK")
        return 0

    files = subprocess.run(
        ["git", "ls-files", "*.rs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    violations = [f"{f}:{line}: {msg}" for f in files for line, msg in check(f)]
    if violations:
        print("error: import grouping violates the convention.", file=sys.stderr)
        print(HINT, file=sys.stderr)
        print("", file=sys.stderr)
        print("\n".join(violations), file=sys.stderr)
        return 1
    print(f"import grouping OK ({len(files)} Rust files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
