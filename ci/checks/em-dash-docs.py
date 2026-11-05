#!/usr/bin/env python3
"""Reject em dashes in tracked Markdown prose.

An em dash is uncommon to type by hand, so it reads as machine-written.

Exempt, because the character is not punctuation there:

- Fenced code blocks and inline `` `code` `` spans: data, not prose.
- A table cell that is, or opens with, the glyph: it means "not applicable".

Every rendering is rejected, not just U+2014. The needles are built from the
code point so this file does not trip its own check.
"""

import re
import subprocess
import sys
from pathlib import Path

EM_DASH = chr(0x2014)

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`+[^`]*`+")
TABLE_ROW = re.compile(r"^\s*\|.*\|")


def tracked_markdown() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(out)


def renderings() -> list[str]:
    """Every spelling that renders as an em dash, lowercased.

    Split into a prefix and its digits so zero-padded forms (`&#08212;`) are caught.
    """
    return [
        ("&", f"{'mdash'};"),
        ("&#", f"{8212};"),
        ("&#x", f"{0x2014:x};"),
        ("\\u{", f"{0x2014:x}}}"),
    ]


def contains_padded(haystack: str, prefix: str, digits: str) -> bool:
    at = haystack.find(prefix)
    while at != -1:
        if haystack[at + len(prefix) :].lstrip("0").startswith(digits):
            return True
        at = haystack.find(prefix, at + 1)
    return False


def na_cell(line: str, col: int) -> bool:
    """True when this dash is a table cell that is, or opens with, the glyph."""
    if not TABLE_ROW.match(line):
        return False
    start = line.index("|") + 1
    for i, ch in enumerate(line[start:], start=start):
        if ch == "|":
            if start <= col < i:
                return _is_na(line[start:i], col - start)
            start = i + 1
    return _is_na(line[start:], col - start)


def _is_na(cell: str, at: int) -> bool:
    body = cell.strip()
    return body == EM_DASH or (body.startswith(EM_DASH) and at == cell.index(EM_DASH))


def offenders(path: str) -> list[str]:
    return scan_source(Path(path).read_text(errors="replace"), path)


def scan_source(text: str, path: str) -> list[str]:
    hits: list[str] = []
    in_fence = False
    lines = text.split("\n")
    for lineno, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        code = [(m.start(), m.end()) for m in INLINE_CODE.finditer(line)]
        lower = line.lower()
        flagged = False
        for m in re.finditer(re.escape(EM_DASH), line):
            if any(a <= m.start() < b for a, b in code):
                continue
            if na_cell(line, m.start()):
                continue
            flagged = True
            break
        if not flagged:
            for prefix, digits in renderings():
                if contains_padded(lower, prefix, digits):
                    flagged = True
                    break
        if flagged:
            hits.append(f"{path}:{lineno}: {line.strip()[:120]}")
    return hits


def selftest() -> list[str]:
    """Fixtures for the exemptions, run on every invocation.

    An over-eager exemption still passes, so nothing else reveals it. Spellings
    are composed so this file does not trip its own check.
    """
    d = EM_DASH
    ent = "&" + "mdash" + ";"
    num = "&#" + f"{8212};"
    num_padded = "&#" + f"0{8212};"
    hex_ent = "&#x" + f"{0x2014:x};"
    cases: list[tuple[str, int, str]] = [
        (f"a {d} b\n", 1, "plain prose"),
        (f"```\na {d} b\n```\n", 0, "fenced code is data, not prose"),
        (f"~~~\na {d} b\n~~~\n", 0, "tilde fence"),
        (f"```\nfenced\n```\na {d} b\n", 1, "prose AFTER a fence still counts"),
        (f"a `x {d} y` b\n", 0, "inline code span"),
        (f"| a | {d} |\n", 0, "a cell that IS the glyph reads as not-applicable"),
        (
            f"| a | {d} (a function, no shape) |\n",
            0,
            "a cell OPENING with the glyph is the same idiom",
        ),
        (f"| a | b {d} c |\n", 1, "a glyph mid-cell is punctuation, not the glyph"),
        (f"| {d} | b {d} c |\n", 1, "one exempt cell does not excuse the other"),
        (f"a {ent} b\n", 1, "html entity"),
        (f"a {num} b\n", 1, "numeric entity"),
        (f"a {num_padded} b\n", 1, "zero-padded numeric entity"),
        (f"a {hex_ent} b\n", 1, "hex entity"),
        ("a plain line\n", 0, "no glyph"),
    ]
    failures = []
    for src, want, why in cases:
        got = len(scan_source(src, "<selftest>"))
        if got != want:
            failures.append(f"selftest: expected {want} hit(s), got {got} for {why}")
    return failures


def main() -> int:
    failures = selftest()
    if failures:
        print("the guard's own fixtures failed, so its verdict cannot be trusted:")
        for f in failures:
            print(f"  {f}")
        return 1
    files = tracked_markdown()
    if len(files) < 20:
        # A broken walk would make the check vacuously green.
        print(f"expected to scan the docs tree, found {len(files)} markdown files")
        return 1
    hits = [h for f in files for h in offenders(f)]
    if not hits:
        return 0
    print(f"em dashes in Markdown prose ({len(hits)}):")
    for h in hits:
        print(f"  {h}")
    print()
    print("Rewrite the sentence instead of swapping in an en dash or '--'.")
    print("A period, comma, colon, or parentheses usually carries the same break;")
    print("in a list item or heading naming a term, a colon is usually what was meant.")
    print("Fenced code blocks, and a table cell holding the 'n/a' glyph, are exempt.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
