#!/usr/bin/env python3
"""Generate (or verify) the Android app's bundled third-party attribution file.

    ./ci/android-attribution.py --write    # regenerate and commit the result
    ./ci/android-attribution.py --check    # fail if the committed file is stale

The output is `android/Pipette/app/src/main/assets/ThirdPartyLicenses.json`,
read at runtime by Acknowledgements.kt (schema: a list of
`{name, license, text}`, same shape the iOS client bundles).

It covers everything distributed inside the APK/AAB:

1. Vendored native sources compiled into the shipped .so files (`vendor/*`).
2. Rust crates statically linked into `libpipette_android.so`: the
   `pipette-android` graph for `aarch64-linux-android`, resolved per package,
   normal dependencies only (dev, build and proc-macro crates never ship).
3. Maven artifacts on the Gradle runtime classpath (`--configuration`).

License text is the dependency's own license file where one exists (vendored
sources, Rust crates in the local cargo registry), otherwise the canonical SPDX
text from a pinned SPDX release. `--check` re-renders, so it needs network.

Known limitation: Maven POMs name a license but carry no text, so Maven
artifacts share the canonical SPDX text and MIT copyright lines are not
reproduced per project.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

REPO = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
OUT = os.path.join(REPO, "android/Pipette/app/src/main/assets/ThirdPartyLicenses.json")

# Reuses ci/android-licenses.py for Maven resolution and license classification.
# Loaded by path because the hyphenated filename is not importable.
_spec = importlib.util.spec_from_file_location(
    "android_licenses", os.path.join(REPO, "ci", "android-licenses.py")
)
android_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(android_licenses)

# KleidiAI is listed even though `PIPETTE_ENABLE_KLEIDIAI` defaults to off
# (android/Pipette/build-rust-android.sh), so the output does not depend on the
# environment `--check` runs in. Over-crediting is the safe direction.
VENDORED = [
    ("llama.cpp", "MIT", "vendor/llama.cpp/LICENSE"),
    ("sentry-native", "MIT", "vendor/sentry-native/LICENSE"),
    ("KleidiAI", "Apache-2.0", "vendor/kleidiai/LICENSES/Apache-2.0.txt"),
]

# SPDX id -> SPDX full name, shown in the app's acknowledgements screen.
DISPLAY = {
    "Apache-2.0": "Apache License 2.0",
    "Apache-2.0 WITH LLVM-exception": "Apache License 2.0 with LLVM Exception",
    "MIT": "MIT License",
    "BSD-2-Clause": 'BSD 2-Clause "Simplified" License',
    "BSD-3-Clause": 'BSD 3-Clause "New" or "Revised" License',
    "ISC": "ISC License",
    "Zlib": "zlib License",
    "Unicode-3.0": "Unicode License v3",
    "Unlicense": "The Unlicense",
    "CC0-1.0": "Creative Commons Zero v1.0 Universal",
    "MPL-2.0": "Mozilla Public License 2.0",
    "0BSD": "BSD Zero Clause License",
    "CDLA-Permissive-2.0": "Community Data License Agreement Permissive 2.0",
}

# Under a choice ("MIT OR Apache-2.0"), attribute under the first of these
# present. MIT leads because it is the most common, which keeps the screen short.
ELECTION_ORDER = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "ISC",
    "Zlib",
    "0BSD",
    "Unlicense",
    "CC0-1.0",
    "MPL-2.0",
    "Unicode-3.0",
    "CDLA-Permissive-2.0",
]

# android-licenses.py canonical id -> SPDX id, for Maven canonical-text lookup.
MAVEN_SPDX = {
    "Apache-2.0": "Apache-2.0",
    "MIT": "MIT",
    "ISC": "ISC",
    "MPL-2.0": "MPL-2.0",
    "Unicode": "Unicode-3.0",
    "EDL-1.0 (BSD-style)": "BSD-3-Clause",
    "Bouncy Castle (MIT-style)": "MIT",
    "JDOM (Apache-style)": "Apache-2.0",
    "Public Domain / CC0": "CC0-1.0",
}


def maven_spdx(canonical: str, declared: str) -> str:
    """SPDX id for a Maven artifact's license.

    The classifier collapses BSD variants (equivalent for gating); attribution
    needs the exact text, so the variant is recovered from the declared string.
    """
    if canonical == "BSD":
        return "BSD-2-Clause" if "2" in declared else "BSD-3-Clause"
    spdx = MAVEN_SPDX.get(canonical)
    if not spdx:
        sys.exit(f"no SPDX mapping for {canonical!r}: add it to MAVEN_SPDX")
    return spdx


# Non-OSS vendor terms carry no redistributable license text, so the entry
# points at the terms instead of pretending to quote them.
VENDOR_TEXT = {
    "Google: Android SDK Terms": (
        "Distributed under the Android Software Development Kit License "
        "Agreement.\nSee https://developer.android.com/studio/terms"
    ),
    "Google: Play Core Terms": (
        "Distributed under the Play Core Software Development Kit Terms of "
        "Service.\nSee https://developer.android.com/guide/playcore/license"
    ),
    "Google: Play Integrity Terms": (
        "Distributed under the Play Integrity API Terms of Service.\n"
        "See https://developer.android.com/google/play/integrity/terms"
    ),
}

# A release tag, not `main`: an upstream reformat would otherwise fail --check
# with "stale" when no dependency changed. Bumping it means regenerating.
SPDX_REF = "v3.27.0"
SPDX_TEXT_URL = (
    f"https://raw.githubusercontent.com/spdx/license-list-data/{SPDX_REF}/text/{{}}.txt"
)
_spdx_cache: dict[str, str] = {}


def spdx_text(spdx_id: str) -> str:
    """Canonical license text for an SPDX id or `<id> WITH <exception>`.

    SPDX publishes exception texts as their own files (text/LLVM-exception.txt),
    so a WITH expression is fetched as two documents and concatenated.
    """
    if spdx_id not in _spdx_cache:
        if " WITH " in spdx_id:
            base, exception = (p.strip() for p in spdx_id.split(" WITH ", 1))
            _spdx_cache[spdx_id] = (
                f"{_fetch(base)}\n--- {exception} ---\n\n{_fetch(exception)}"
            )
        else:
            _spdx_cache[spdx_id] = _fetch(spdx_id)
    return _spdx_cache[spdx_id]


def _fetch(spdx_id: str) -> str:
    # Not just OSError: a malformed id raises http.client.InvalidURL, which is not one.
    try:
        with urllib.request.urlopen(SPDX_TEXT_URL.format(spdx_id), timeout=30) as resp:
            return resp.read().decode("utf-8").strip() + "\n"
    except Exception as exc:  # noqa: BLE001 - reported, then fatal
        sys.exit(
            f"could not fetch canonical text for {spdx_id!r} from SPDX {SPDX_REF}: "
            f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# 1. vendored native sources
# ---------------------------------------------------------------------------


def vendored_entries() -> list[tuple[str, str, str]]:
    out = []
    for name, spdx, rel in VENDORED:
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            sys.exit(f"missing license file: {rel} (is the submodule checked out?)")
        with open(path, encoding="utf-8", errors="replace") as fh:
            out.append((name, spdx, fh.read().strip() + "\n"))
    return out


# ---------------------------------------------------------------------------
# 2. Rust crates linked into libpipette_android.so
# ---------------------------------------------------------------------------

LICENSE_FILE_PREFERENCE = {
    "MIT": ("LICENSE-MIT", "LICENSE-MIT.md", "LICENSE-MIT.txt"),
    "Apache-2.0": ("LICENSE-APACHE", "LICENSE-APACHE-2.0", "LICENSE-APACHE.md"),
    "Apache-2.0 WITH LLVM-exception": ("LICENSE-APACHE",),
    "Unlicense": ("UNLICENSE", "LICENSE-UNLICENSE"),
}
GENERIC_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING")


def elect(expression: str) -> str:
    """Pick one SPDX id from an OR expression ("MIT/Apache-2.0", "MIT OR Apache-2.0").

    AND-ed parts are split off by the caller, since every one of them applies.
    """
    normalized = expression.replace("/", " OR ")
    options = [t.strip(" ()") for t in normalized.split(" OR ")]
    for preferred in ELECTION_ORDER:
        if preferred in options:
            return preferred
    sys.exit(f"no electable license in {expression!r}: add it to ELECTION_ORDER")


def split_expression(expression: str) -> list[str]:
    """Every license to attribute for one crate.

    "(MIT OR Apache-2.0) AND Unicode-3.0" needs the elected half of the OR plus
    Unicode-3.0.
    """
    parts = [p.strip() for p in expression.split(" AND ")]
    return [elect(p) for p in parts]


def crate_dir(name: str, version: str) -> str | None:
    for src in glob.glob(os.path.expanduser("~/.cargo/registry/src/*/")):
        candidate = os.path.join(src, f"{name}-{version}")
        if os.path.isdir(candidate):
            return candidate
    return None


def crate_license_text(name: str, version: str, spdx: str, expression: str) -> str:
    """The crate's own license file if it has a matching one, else SPDX text.

    `spdx` (the elected id) picks the named license file to look for;
    `expression` decides whether a generic `LICENSE` file can be trusted.
    """
    directory = crate_dir(name, version)
    if directory:
        for filename in LICENSE_FILE_PREFERENCE.get(spdx, ()):
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    return fh.read().strip() + "\n"
        # Under a choice, a generic LICENSE file may hold the option we did not
        # elect. Test the expression: `spdx` is always a single id.
        single = " OR " not in expression and "/" not in expression
        if single:
            for filename in GENERIC_LICENSE_FILES:
                path = os.path.join(directory, filename)
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        return fh.read().strip() + "\n"
    return spdx_text(spdx)


def rust_entries() -> list[tuple[str, str, str]]:
    # Membership comes from `cargo tree -p`, which resolves per package like the
    # real build (CMakeLists.txt). `cargo metadata`'s resolve is workspace-unified
    # and would include features other members enable (e.g. `pipette-torch-oai`
    # turns on `clap` in `pipette-plan-types`). Metadata supplies only licenses.
    tree = subprocess.run(
        [
            "cargo",
            "tree",
            "-p",
            "pipette-android",
            "--target",
            "aarch64-linux-android",
            "-e",
            "normal,no-proc-macro",
            "--prefix",
            "none",
            "-f",
            "{p}",
            # CI sets CARGO_TERM_COLOR=always, which would break the `(*)` strip below.
            "--color",
            "never",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    linked: set[tuple[str, str]] = set()
    for line in tree.splitlines():
        # `name vX.Y.Z`, optionally ` (/path)` for a path dependency and ` (*)`
        # where the tree has already shown that subtree.
        line = line.removesuffix(" (*)")
        if " (" in line:
            line = line[: line.index(" (")]
        name, _, version = line.partition(" v")
        if name and version:
            linked.add((name, version))
    if not any(name == "pipette-android" for name, _ in linked):
        sys.exit("pipette-android not found in cargo tree")

    meta = json.loads(
        subprocess.run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--filter-platform",
                "aarch64-linux-android",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    by_coord = {(p["name"], p["version"]): p for p in meta["packages"]}

    out = []
    # Sorted: set order varies per process (salted hashing) and would flake --check.
    for name, version in sorted(linked):
        if name.startswith("pipette-"):
            continue  # first-party
        pkg = by_coord.get((name, version))
        if pkg is None:
            sys.exit(f"{name} {version} is in the link graph but not in cargo metadata")
        expression = pkg.get("license")
        if not expression:
            sys.exit(f"{name} {version} declares no license")
        for spdx in split_expression(expression):
            out.append(
                (
                    name,
                    spdx,
                    crate_license_text(name, version, spdx, expression),
                )
            )
    return out


# ---------------------------------------------------------------------------
# 3. Maven artifacts on the release runtime classpath
# ---------------------------------------------------------------------------


def maven_entries(
    configuration: str = "kotlinEngineReleaseRuntimeClasspath",
) -> list[tuple[str, str, str]]:
    coords = android_licenses.resolve_configuration(configuration)
    out = []
    for group, artifact, version in coords:
        names, _ = android_licenses.licenses_of(group, artifact, version)
        if not names:
            sys.exit(f"{group}:{artifact}:{version} declares no license in its POM")
        for declared in names:
            verdict, canonical = android_licenses.classify(declared)
            if verdict == "copyleft":
                sys.exit(
                    f"{group}:{artifact}:{version} is copyleft ({canonical}); "
                    "ci/android-licenses.py should have caught this"
                )
            if verdict == "unknown":
                sys.exit(
                    f"{group}:{artifact}:{version} declares an unclassified "
                    f"license {declared!r}: classify it in ci/android-licenses.py"
                )
            if verdict == "vendor":
                text = VENDOR_TEXT.get(canonical)
                if not text:
                    sys.exit(
                        f"no terms text for vendor license {canonical!r}: "
                        "add it to VENDOR_TEXT"
                    )
                out.append((f"{group}:{artifact}", canonical, text))
                continue
            spdx = maven_spdx(canonical, declared)
            out.append((f"{group}:{artifact}", spdx, spdx_text(spdx)))
    return out


# ---------------------------------------------------------------------------
# grouping + output
# ---------------------------------------------------------------------------


def group(entries: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Collapse entries sharing a license id and text into one row.

    The label is short ("a, b + N more") so the list layout survives a group of
    hundreds; the full roster of names, which is the actual obligation, is
    prefixed to the detail text instead.
    """
    buckets: dict[tuple[str, str], list[str]] = {}
    for name, spdx, text in entries:
        buckets.setdefault((spdx, text), [])
        if name not in buckets[(spdx, text)]:
            buckets[(spdx, text)].append(name)

    rows = []
    for (spdx, text), names in buckets.items():
        names = sorted(names)
        if len(names) <= 3:
            label = ", ".join(names)
        else:
            label = f"{names[0]}, {names[1]} + {len(names) - 2} more"
        body = text
        if len(names) > 1:
            roster = "\n".join(f"  - {n}" for n in names)
            body = (
                f"The following {len(names)} components are distributed under "
                f"the {DISPLAY.get(spdx, spdx)}:\n\n{roster}\n\n"
                f"{'-' * 72}\n\n{text}"
            )
        rows.append(
            {
                "name": label,
                "license": DISPLAY.get(spdx, spdx),
                "text": body,
            }
        )
    # Text is in the key because two versions of one crate with different license
    # files tie on (license, name), and a tie could flip order between runs.
    return sorted(rows, key=lambda r: (r["license"], r["name"], r["text"]))


def render(configuration: str = "kotlinEngineReleaseRuntimeClasspath") -> str:
    entries = vendored_entries() + rust_entries() + maven_entries(configuration)
    rows = group(entries)
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the file")
    mode.add_argument(
        "--check", action="store_true", help="fail if the committed file is stale"
    )
    ap.add_argument(
        "--configuration",
        default="kotlinEngineReleaseRuntimeClasspath",
        help="Gradle configuration to query for Maven artifacts (default: %(default)s)",
    )
    args = ap.parse_args()

    generated = render(args.configuration)
    count = len(json.loads(generated))

    if args.write:
        with open(OUT, "w", encoding="utf-8") as fh:
            fh.write(generated)
        print(f"wrote {os.path.relpath(OUT, REPO)} ({count} entries)")
        return 0

    if not os.path.exists(OUT):
        print(f"FAIL: {os.path.relpath(OUT, REPO)} does not exist", file=sys.stderr)
        return 1
    with open(OUT, encoding="utf-8") as fh:
        committed = fh.read()
    if committed == generated:
        print(
            f"ok: {os.path.relpath(OUT, REPO)} matches the resolved graph "
            f"({count} entries)"
        )
        return 0

    print(
        f"FAIL: {os.path.relpath(OUT, REPO)} does not match the dependency graph\n"
        "  the app ships. As committed, the app would display attribution for\n"
        "  dependencies it does not have, and omit ones it does.\n\n"
        "  Regenerate and commit:\n"
        "    ./ci/android-attribution.py --write",
        file=sys.stderr,
    )
    old = json.loads(committed) if committed.strip() else []
    old_names = {r.get("name") for r in old}
    new_names = {r["name"] for r in json.loads(generated)}
    for label, diff in (
        ("only in committed file", old_names - new_names),
        ("only in regenerated file", new_names - old_names),
    ):
        if diff:
            print(f"\n  {label} ({len(diff)}):", file=sys.stderr)
            for name in sorted(diff)[:15]:
                print(f"    {name}", file=sys.stderr)
            if len(diff) > 15:
                print(f"    … and {len(diff) - 15} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
