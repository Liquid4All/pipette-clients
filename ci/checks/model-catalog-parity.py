#!/usr/bin/env python3
"""Pin the iOS and Android preset model catalogs to the same GGUF set.

The two clients hand-maintain separate catalogs in different languages, so they
drift silently: a model added to one is simply absent from the other, and
nothing fails. That is how `LiquidAI/LFM2.5-230M-GGUF` and
`LiquidAI/LFM2.5-8B-A1B-GGUF` sat on iOS only until a manual parity audit found
them (PIP-435). This check turns the next divergence into a failed build.

Compared as a set of `(repo, quant)` pairs rather than repos alone: offering a
quant on one platform and not the other is the same class of gap, and the
catalogs agree at that granularity today.

Scope is deliberately GGUF only. MLX entries are Apple-silicon bundles with no
Android counterpart, so they are not a divergence.

If a model genuinely belongs on one platform only, this check is the wrong place
to encode that: add it to both and gate it at the UI, or extend this script with
an explicit, commented exception list. Do not simply delete the entry that fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

IOS_CATALOG = Path("ios/Pipette/Pipette/Contracts/ModelSourceCatalog.swift")
ANDROID_CATALOG = Path(
    "android/Pipette/app/src/main/java/ai/liquid/pipette/ModelTemplates.kt"
)

# `.gguf("Display Name", "org/repo", "QUANT", bytes: 123)`
IOS_ENTRY = re.compile(r'\.gguf\(\s*"[^"]*",\s*"([^"]+)",\s*"([^"]+)"')

# The `identifier` argument of a PresetModel: `"org/repo:QUANT"`. Matched on its
# own rather than by position because Android wraps longer entries across lines,
# and this shape appears nowhere else in the file.
ANDROID_ENTRY = re.compile(r'"([\w.\-]+/[\w.\-]+):([A-Za-z0-9_]+)"')


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def extract(path: Path, pattern: re.Pattern[str]) -> set[tuple[str, str]]:
    if not path.exists():
        print(f"catalog not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def report(title: str, missing: set[tuple[str, str]], target: Path) -> None:
    if not missing:
        return
    print(f"{title} (add to {target}):", file=sys.stderr)
    for repo, quant in sorted(missing):
        print(f"  {repo}:{quant}", file=sys.stderr)


def main() -> int:
    root = repo_root()
    ios = extract(root / IOS_CATALOG, IOS_ENTRY)
    android = extract(root / ANDROID_CATALOG, ANDROID_ENTRY)

    # An empty side means the regex stopped matching (a refactor of either
    # catalog's shape), not that a platform genuinely ships no models. Fail
    # loudly rather than reporting a spurious set difference.
    catalogs = (("iOS", ios, IOS_CATALOG), ("Android", android, ANDROID_CATALOG))
    for name, found, path in catalogs:
        if not found:
            print(
                f"parsed 0 GGUF entries from the {name} catalog ({path}); "
                "its shape probably changed and this check needs updating",
                file=sys.stderr,
            )
            return 1

    report("present on iOS but missing on Android", ios - android, ANDROID_CATALOG)
    report("present on Android but missing on iOS", android - ios, IOS_CATALOG)
    if ios != android:
        return 1

    print(f"iOS and Android GGUF catalogs match ({len(ios)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
