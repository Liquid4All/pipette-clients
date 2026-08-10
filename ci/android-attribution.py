#!/usr/bin/env python3
"""Generate (or verify) the Android app's bundled third-party attribution file.

    ./ci/android-attribution.py --write    # regenerate and commit the result
    ./ci/android-attribution.py --check    # fail if the committed file is stale

The output is `android/Pipette/app/src/main/assets/ThirdPartyLicenses.json`,
read at runtime by Acknowledgements.kt (schema: a list of
`{name, license, text}`, same shape the iOS client bundles).

Attribution is the one obligation Apache-2.0 and MIT impose, and it applies to
everything distributed inside the APK/AAB. That is three separate graphs, all
covered here:

1. **Vendored native sources** compiled into the shipped .so files
   (`vendor/*` submodules).
2. **Rust crates** statically linked into `libpipette_android.so` — the
   `pipette-android` dependency graph for `aarch64-linux-android`, normal
   (linking) dependencies only. Dev- and build-only dependencies are excluded,
   and so are proc macros and their subtrees: all of them run at compile time
   and none reaches the shipped binary, so they carry no distribution
   obligation. Note that a proc macro is a *normal* dependency as far as
   `dep_kinds` is concerned — it is identified by its own target kind.
3. **Maven artifacts** on `releaseRuntimeClasspath` — the ~212 Java/Kotlin
   dependencies that become dex in the APK.

License text sources, in order of preference:

- the dependency's own license file (vendored submodules, and Rust crates
  extracted in the local cargo registry) — preferred because it carries that
  project's own copyright line;
- otherwise the canonical SPDX text, fetched from a pinned SPDX release tag and
  embedded in the committed JSON, so the *app build* reads it straight from
  assets. Note that `--check` re-renders in order to compare, so the check
  itself does need network — as does resolving Gradle in the same job.

Known limitation: Maven POMs declare a license *name*, never the text, and
`.aar`/`.jar` files only sometimes embed one. So Maven artifacts are grouped
under the canonical SPDX text for their license. For MIT that means the
per-project copyright lines are not reproduced verbatim. This is what mainstream
Android attribution tooling does, and it is a deliberate, documented tradeoff —
not an oversight. Rust crates and vendored sources do get their own texts.
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

# ci/android-licenses.py owns Maven resolution and license classification; reuse
# it rather than keeping a second copy of either. Loaded by path because the
# filename is hyphenated (the ci/ naming convention) and so not importable.
_spec = importlib.util.spec_from_file_location(
    "android_licenses", os.path.join(REPO, "ci", "android-licenses.py")
)
android_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(android_licenses)

# Vendored native source trees compiled into the shipped .so files.
#
# KleidiAI is listed unconditionally even though `PIPETTE_ENABLE_KLEIDIAI`
# defaults to off (see build-rust-android.sh), so this file stays byte-identical
# regardless of the environment `--check` runs in. Crediting a library a build
# flag can pull in is the safe direction; omitting one it did would not be.
VENDORED = [
    ("llama.cpp", "MIT", "vendor/llama.cpp/LICENSE"),
    ("sentry-native", "MIT", "vendor/sentry-native/LICENSE"),
    ("KleidiAI", "Apache-2.0", "vendor/kleidiai/LICENSES/Apache-2.0.txt"),
]

# SPDX id -> the display name shown in the app's acknowledgements screen. These
# are the SPDX full names, which is what the screen is designed to render.
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

# When a crate offers a choice ("MIT OR Apache-2.0"), attribute under the first
# of these present. MIT leads because it is the most common option across this
# graph, which keeps the acknowledgements screen short: every crate electing MIT
# shares one entry rather than splitting across two licenses.
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

# Classifier canonical id (from android-licenses.py) -> SPDX id, for Maven
# artifacts whose POM gives a name but no text. Anything unmapped is a hard
# error rather than a silent omission.
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
    """SPDX id for a Maven artifact's license, for canonical-text lookup.

    The classifier deliberately collapses BSD variants into one verdict — for
    *gating*, 2-clause and 3-clause are equivalent. Attribution needs the exact
    text, so recover the variant from the declared string here.
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

# Pinned to an immutable SPDX release tag, NOT to `main`. Both --write and
# --check derive the embedded text from here, so tracking `main` would mean an
# upstream reformat of any license text turns every CI and release run red with
# "stale — regenerate and commit", which is exactly the wrong diagnosis when no
# dependency changed. Bumping this tag is a deliberate act that regenerates the
# file. (--check does still need network: it re-renders to compare. That is no
# new constraint on the jobs that run it — they already resolve Gradle and cargo
# metadata over the network in the same step.)
SPDX_REF = "v3.27.0"
SPDX_TEXT_URL = (
    f"https://raw.githubusercontent.com/spdx/license-list-data/{SPDX_REF}/text/{{}}.txt"
)
_spdx_cache: dict[str, str] = {}


def spdx_text(spdx_id: str) -> str:
    """Canonical license text for an SPDX id or `<id> WITH <exception>`.

    SPDX publishes exception texts as their own flat files (text/LLVM-exception.txt),
    so a WITH expression is fetched as two documents and concatenated. Passing the
    whole expression to one URL would embed a space and raise before any request.
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
    # Catch Exception, not just OSError: a malformed id yields an unquotable URL
    # and http.client.InvalidURL, which derives from HTTPException — not from
    # OSError — and would otherwise escape as a bare traceback.
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
    """Pick one SPDX id from a crate's license expression.

    Cargo manifests use several spellings ("MIT/Apache-2.0", "MIT OR
    Apache-2.0", "(MIT OR Apache-2.0) AND Unicode-3.0"). An AND means every
    listed license applies, so electing one would under-attribute; those are
    handled by the caller, which keeps all AND-ed ids.
    """
    normalized = expression.replace("/", " OR ")
    options = [t.strip(" ()") for t in normalized.split(" OR ")]
    for preferred in ELECTION_ORDER:
        if preferred in options:
            return preferred
    sys.exit(f"no electable license in {expression!r}: add it to ELECTION_ORDER")


def split_expression(expression: str) -> list[str]:
    """Every license that must be attributed for one crate.

    `AND` is conjunctive: "(MIT OR Apache-2.0) AND Unicode-3.0" needs the
    elected half of the OR *plus* Unicode-3.0.
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

    `spdx` is the single elected id; `expression` is the crate's full manifest
    license. Both are needed: the elected id picks which named license file to
    look for, while the expression decides whether an unnamed `LICENSE` file can
    be trusted at all.
    """
    directory = crate_dir(name, version)
    if directory:
        for filename in LICENSE_FILE_PREFERENCE.get(spdx, ()):
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    return fh.read().strip() + "\n"
        # A generic LICENSE file is only safe when the crate offers exactly one
        # license. Under a choice, the file may hold the option we did not elect
        # — which would pair, say, Apache-2.0 text with an "MIT License" label.
        # This must test the *expression*: `spdx` is always a single elected id,
        # so checking it here would make the guard unconditionally true.
        single = " OR " not in expression and "/" not in expression
        if single:
            for filename in GENERIC_LICENSE_FILES:
                path = os.path.join(directory, filename)
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        return fh.read().strip() + "\n"
    return spdx_text(spdx)


def rust_entries() -> list[tuple[str, str, str]]:
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
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    roots = [i for i, p in packages.items() if p["name"] == "pipette-android"]
    if not roots:
        sys.exit("pipette-android not found in cargo metadata")

    def is_proc_macro(pkg_id: str) -> bool:
        # A proc macro is a *normal* dependency — `dep_kinds` says nothing about
        # it — so the only way to tell is the package's own target kind. Its code
        # runs in the compiler and never reaches the .so, and neither does its
        # subtree (syn, quote, proc-macro2), so the whole branch is pruned.
        return any(
            "proc-macro" in target.get("kind", [])
            for target in packages[pkg_id].get("targets", [])
        )

    seen: set[str] = set()

    def walk(pkg_id: str) -> None:
        if pkg_id in seen:
            return
        seen.add(pkg_id)
        for dep in nodes.get(pkg_id, {}).get("deps", []):
            kinds = {(k.get("kind") or "normal") for k in dep.get("dep_kinds", [])}
            if "normal" not in kinds:  # dev-/build-only: not in the shipped .so
                continue
            if is_proc_macro(dep["pkg"]):
                continue
            walk(dep["pkg"])

    for root in roots:
        walk(root)

    out = []
    # Sorted, not raw set order: string hashing is salted per process, so
    # iterating `seen` directly would vary the output between runs and make
    # --check flake on an unchanged tree.
    for pkg_id in sorted(seen):
        pkg = packages[pkg_id]
        if pkg["name"].startswith("pipette-"):
            continue  # first-party
        expression = pkg.get("license")
        if not expression:
            sys.exit(f"{pkg['name']} {pkg['version']} declares no license")
        for spdx in split_expression(expression):
            out.append(
                (
                    pkg["name"],
                    spdx,
                    crate_license_text(pkg["name"], pkg["version"], spdx, expression),
                )
            )
    return out


# ---------------------------------------------------------------------------
# 3. Maven artifacts on releaseRuntimeClasspath
# ---------------------------------------------------------------------------


def maven_entries() -> list[tuple[str, str, str]]:
    coords = android_licenses.resolve_configuration("releaseRuntimeClasspath")
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

    The row label keeps the format the app already displayed — up to three names
    in full, otherwise the first two plus "+ N more" — because a row that listed
    181 Apache-2.0 artifacts would wreck the list layout and truncate the detail
    view's title bar.

    But a label alone would leave most components unnamed, and naming them is
    the whole obligation. So when a group covers more than one component, the
    detail text is prefixed with the full roster. Every shipped dependency is
    then named in the asset and readable in the app, while the list stays short.
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
    # Stable order so --check is deterministic and diffs stay readable. The text
    # is part of the key because (license, name) alone can tie: two versions of
    # one crate with differing license files produce two rows under the same
    # label, and an unbroken tie would let their order flip between runs.
    return sorted(rows, key=lambda r: (r["license"], r["name"], r["text"]))


def render() -> str:
    entries = vendored_entries() + rust_entries() + maven_entries()
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
    args = ap.parse_args()

    generated = render()
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
