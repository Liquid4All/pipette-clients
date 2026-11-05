#!/usr/bin/env python3
"""Hold the iOS and Android preset model catalogs to a declared divergence set.

The two catalogs are hand-maintained in different languages and drift silently.
They may legitimately differ (a model ships on a platform only when that
platform's phone finished the ladder resident, without swapping), but every
difference must be declared here with the device result behind it.

Both directions fail:

- a row on one platform and not the other, not listed in `IOS_ONLY` /
  `ANDROID_ONLY`, is undeclared drift;
- a listed row that no longer diverges is a stale exemption, so the list
  cannot only grow.

Two formats are compared, each on its own identity:

- GGUF, as `(repo, quant)`: a quant offered on one platform only is a gap too;
- ONNX Runtime GenAI, as `(repo, normalized prefix, declared revision)`: one repo
  publishes several packages, and the same repo at another commit is other weights.

MLX is excluded: Apple-silicon bundles have no Android counterpart.

Do not resolve a failure by deleting the entry that fails. Either add it to the
other catalog, or declare it below with its reason.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

IOS_CATALOG = Path("ios/Pipette/Pipette/Models/CatalogEntry.swift")
ANDROID_CATALOG = Path(
    "android/Pipette/app/src/main/java/ai/liquid/pipette/ModelTemplates.kt"
)

# `.gguf("Display Name", "org/repo", "QUANT", bytes: 123)`
IOS_ENTRY = re.compile(r'\.gguf\(\s*"[^"]*",\s*"([^"]+)",\s*"([^"]+)"')

# A PresetModel's `"org/repo:QUANT"` identifier. Matched on its own, not by
# position, because Android wraps longer entries across lines.
ANDROID_ENTRY = re.compile(r'"([\w.\-]+/[\w.\-]+):([A-Za-z0-9_]+)"')

# Row counters, compared with the coordinate count by `extract`. The lookbehind skips
# the `data class PresetModel(` declaration; iOS's `func gguf(` has no leading dot.
IOS_ROW = re.compile(r"\.gguf\(")
ANDROID_ROW = re.compile(r"(?<!class )PresetModel\(")


# Rows a platform offers alone, on purpose (PIP-511). Each catalog is the set of
# configurations the results dashboard publishes for that handset's llama.cpp engine:
#
#   https://pipette.liquid.ai/results?hw=Galaxy+S26+Ultra+%2F+llama.cpp%3Aandroid-arm64-v8a
#   https://pipette.liquid.ai/results?hw=iPhone+17+Pro+%2F+llama.cpp%3Aios
#
# `unsloth/Olmo-3-7B-Think-GGUF` has published cells on the iPhone 17 Pro and none on
# the SM-S948U1 (Galaxy S26 Ultra).
IOS_ONLY: set[tuple[str, str]] = {
    ("unsloth/Olmo-3-7B-Think-GGUF", "Q4_K_M"),
}

ANDROID_ONLY: set[tuple[str, str]] = set()

# `.onnxGenai("Display Name", "org/repo", prefix: "a/b", revision: "sha", bytes: 123)`,
# wrapped across lines. Not a widened GGUF regex: an ONNX row could then satisfy the
# GGUF row count and vanish from both comparisons.
IOS_ONNX_ENTRY = re.compile(
    r'\.onnxGenai\(\s*"[^"]*",\s*"([^"]+)",'
    r'\s*prefix:\s*"([^"]*)",'
    r'\s*revision:\s*"([^"]+)"'
)
# Anchored on the display-name literal: a bare `.onnxGenai(` also matches the enum
# case the builder constructs in its own body.
IOS_ONNX_ROW = re.compile(r'\.onnxGenai\(\s*"')

# Android has no ONNX Runtime GenAI catalog yet (PIP-593). Any mention of ONNX there
# fails, so the Android coordinate grammar gets written with its first rows instead of
# comparing against an empty set forever.
ANDROID_ONNX_MARKER = re.compile(r"onnx", re.IGNORECASE)

# iOS-only because Android has no ONNX runtime yet; PIP-593 removes this set.
IOS_ONLY_ONNX: set[tuple[str, str, str]] = {
    (
        "microsoft/Phi-3-mini-4k-instruct-onnx",
        "cpu_and_mobile/cpu-int4-rtn-block-32",
        "5f5f794c1c23c9d5ee142af85df02a6cc52d6945",
    ),
}


def normalized_prefix(prefix: str) -> str:
    """A repo subpath in one spelling: `a/b`, `/a/b` and `a//b/` compare equal."""
    return "/".join(part for part in prefix.split("/") if part)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def extract(
    path: Path, pattern: re.Pattern[str], row: re.Pattern[str], name: str
) -> set[tuple[str, str]]:
    """Every `(repo, quant)` in one catalog, refusing to read it partially.

    `pattern` reads a row's coordinate; `row` counts the rows that are there. A
    mismatch fails, so a row the coordinate regex cannot read is not skipped in
    silence (Android's `DownloadCoordinator.parseDownloadInput` accepts more shapes
    than `ANDROID_ENTRY`). Prose containing a row marker fails too, which is the
    safe direction.
    """
    if not path.exists():
        print(f"catalog not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    text = path.read_text(encoding="utf-8")
    found = pattern.findall(text)
    authored = len(row.findall(text))
    if len(found) != authored:
        print(
            f"{name}: read {len(found)} coordinates from {authored} catalog rows in "
            f"{path}.\n  A row this check cannot parse is invisible to it, so an "
            "undeclared divergence there would pass. Widen the coordinate regex, or "
            "the row counter if the extra hit is prose.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return set(found)


def extract_onnx(path: Path) -> set[tuple[str, str, str]]:
    """Every `(repo, normalized prefix, revision)` in the iOS ONNX catalog.

    Same read-it-all-or-fail rule as `extract`.
    """
    text = path.read_text(encoding="utf-8")
    found = IOS_ONNX_ENTRY.findall(text)
    authored = len(IOS_ONNX_ROW.findall(text))
    if len(found) != authored:
        print(
            f"iOS ONNX: read {len(found)} coordinates from {authored} catalog rows in "
            f"{path}.\n  A row this check cannot parse is invisible to it. Widen "
            "IOS_ONNX_ENTRY, or IOS_ONNX_ROW if the extra hit is prose.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return {
        (repo, normalized_prefix(prefix), revision) for repo, prefix, revision in found
    }


def report(title: str, rows: set[tuple[str, ...]], fix: str) -> None:
    if not rows:
        return
    print(f"{title} ({fix}):", file=sys.stderr)
    for row in sorted(rows):
        print(f"  {':'.join(row)}", file=sys.stderr)


def main() -> int:
    root = repo_root()
    ios = extract(root / IOS_CATALOG, IOS_ENTRY, IOS_ROW, "iOS")
    android = extract(root / ANDROID_CATALOG, ANDROID_ENTRY, ANDROID_ROW, "Android")

    # An empty side means the regex stopped matching, not that a platform ships nothing.
    catalogs = (("iOS", ios, IOS_CATALOG), ("Android", android, ANDROID_CATALOG))
    for name, found, path in catalogs:
        if not found:
            print(
                f"parsed 0 GGUF entries from the {name} catalog ({path}); "
                "its shape probably changed and this check needs updating",
                file=sys.stderr,
            )
            return 1

    # `stale` covers a declared row now on both platforms and one dropped from both.
    sides = (
        ("iOS", "Android", ios - android, IOS_ONLY, ANDROID_CATALOG),
        ("Android", "iOS", android - ios, ANDROID_ONLY, IOS_CATALOG),
    )
    failed = False
    for side, other, diverging, declared, add_to in sides:
        undeclared = diverging - declared
        stale = declared - diverging
        report(
            f"present on {side} but missing on {other}",
            undeclared,
            f"add to {add_to}, or declare it in this check",
        )
        report(
            f"declared {side}-only but no longer diverging",
            stale,
            "drop from this check's exemption list",
        )
        failed = failed or bool(undeclared or stale)

    android_text = (root / ANDROID_CATALOG).read_text(encoding="utf-8")
    if ANDROID_ONNX_MARKER.search(android_text):
        print(
            f"the Android catalog ({ANDROID_CATALOG}) mentions ONNX, but this check "
            "has no coordinate grammar for its rows. Add one beside IOS_ONNX_ENTRY "
            "and compare both sides, then clear IOS_ONLY_ONNX of anything Android now "
            "offers (PIP-593).",
            file=sys.stderr,
        )
        return 1

    ios_onnx = extract_onnx(root / IOS_CATALOG)
    undeclared_onnx = ios_onnx - IOS_ONLY_ONNX
    stale_onnx = IOS_ONLY_ONNX - ios_onnx
    report(
        "present in the iOS ONNX catalog but not declared",
        undeclared_onnx,
        "declare it in IOS_ONLY_ONNX with its reason",
    )
    report(
        "declared iOS-only ONNX but no longer in the catalog",
        stale_onnx,
        "drop from IOS_ONLY_ONNX",
    )
    failed = failed or bool(undeclared_onnx or stale_onnx)

    if failed:
        return 1

    shared = len(ios & android)
    divergent = len(IOS_ONLY) + len(ANDROID_ONLY)
    print(f"GGUF catalogs agree on {shared} entries, {divergent} declared divergent")
    print(f"ONNX: {len(ios_onnx)} iOS rows, all declared iOS-only pending PIP-593")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
