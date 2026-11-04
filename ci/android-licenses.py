#!/usr/bin/env python3
"""License gate for the Android app's *shipping* dependencies.

Resolves a Gradle configuration, reads each resolved artifact's declared
license from its POM, and fails on anything outside the allow-list below.

    ./ci/android-licenses.py                     # gate the release runtime classpath
    ./ci/android-licenses.py --list              # print the graph, gate nothing
    ./ci/android-licenses.py --configuration kotlinEngineDebugRuntimeClasspath

Runs in the `android-app` job, which has Gradle and the Android SDK (the
`conventions` job that runs ci/checks/ has no toolchain). It parses
`gradle dependencies` output because resolving POMs inside a Gradle task needs
`ArtifactResolutionQuery`, which is incompatible with the configuration cache.

Only the release runtime classpath is gated: build-time and test configurations
carry copyleft artifacts (trove4j, JNA, JUnit) that are never distributed.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
GRADLE_PROJECT = os.path.join(REPO_ROOT, "android", "Pipette")
GRADLE_CACHE = os.path.expanduser("~/.gradle/caches/modules-2/files-2.1")

# Fallbacks when a POM is not in the local Gradle cache; they mirror
# android/Pipette/settings.gradle.kts.
REPOS = (
    "https://repo1.maven.org/maven2",
    "https://dl.google.com/dl/android/maven2",
    "https://plugins.gradle.org/m2",
)

POM_NS = {"m": "http://maven.apache.org/POM/4.0.0"}

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
#
# POM license strings are free text, so an unrecognized one FAILS: a new
# copyleft dependency must not slide through under an unpredicted spelling.
#
# Patterns match the normalized name (see normalize()): "Apache-2.0" becomes
# "apache 2 0", "The MIT License (MIT)" becomes "the mit license mit". Run
# `--selftest` after editing them.

# Checked first, so "LGPL, version 2.1 / Apache License v2.0" cannot match on
# its Apache half below.
COPYLEFT = (
    # `\bl?gpl\b` alone misses "GPLv2" and "LGPL2.1": no word boundary before the digit.
    (
        r"\bl?gpl\b|\bl?gpl ?v? ?\d|general public license|gnu (lesser|library)",
        "GPL/LGPL",
    ),
    (r"\bagpl\b|affero", "AGPL"),
    (r"\bsspl\b|server side public license", "SSPL"),
    (r"mozilla public license 1|\bmpl 1", "MPL-1.x"),
    (r"common development and distribution|\bcddl\b", "CDDL"),
    (r"eclipse public license|\bepl\b", "EPL"),
    (r"common public license|\bcpl\b", "CPL"),
)

# Keep in sync with the Rust allow-list in deny.toml.
ALLOWED = (
    (
        r"^(the )?apache( software)?( licen[cs]e)?,? ?(version |v ?)?2( 0)?$",
        "Apache-2.0",
    ),
    (r"^(the )?mit licen[cs]e( mit)?$|^mit$", "MIT"),
    (
        r"^(the )?(new |simplified |revised )?bsd"
        r"( style| 2 clause| 3 clause)?( licen[cs]e)?$",
        "BSD",
    ),
    (r"^isc( licen[cs]e)?$", "ISC"),
    # Eclipse Distribution License is BSD-style, unlike the Eclipse Public License.
    (r"^eclipse distribution license( v ?1( 0)?)?$|^edl 1 0$", "EDL-1.0 (BSD-style)"),
    (r"^bouncy castle licen[cs]e$", "Bouncy Castle (MIT-style)"),
    (
        r"^mozilla public license( version)? 2( 0)?( mpl 2( 0)?)?$|^mpl 2( 0)?$",
        "MPL-2.0",
    ),
    (r"^public domain$|^cc0", "Public Domain / CC0"),
    (r"^unicode", "Unicode"),
    (
        r"^similar to apache license but with the acknowledgment clause removed$",
        "JDOM (Apache-style)",
    ),
)

# Non-OSS vendor terms accepted as a product decision; they impose no
# source-disclosure obligation.
VENDOR_TERMS = (
    # All arrive transitively through com.clerk:clerk-android-api (Play Services
    # via credentials-play-services-auth, Play Integrity, googleid).
    (
        r"^android software development kit licen[cs]e( agreement)?$",
        "Google: Android SDK Terms",
    ),
    (
        r"^play core software development kit terms of service$",
        "Google: Play Core Terms",
    ),
    (r"^play integrity api terms of service$", "Google: Play Integrity Terms"),
)


# Strings this graph declares, plus copyleft strings that must be rejected, so
# a loosened pattern cannot quietly allow copyleft. CI runs --selftest first.
SELFTEST = (
    ("The Apache Software License, Version 2.0", "allow", "Apache-2.0"),
    ("The Apache License, Version 2.0", "allow", "Apache-2.0"),
    ("Apache License, Version 2.0", "allow", "Apache-2.0"),
    ("Apache License V2.0", "allow", "Apache-2.0"),
    ("Apache 2.0", "allow", "Apache-2.0"),
    ("Apache-2.0", "allow", "Apache-2.0"),
    ("Apache 2", "allow", "Apache-2.0"),
    ("The MIT License", "allow", "MIT"),
    ("The MIT License (MIT)", "allow", "MIT"),
    ("MIT License", "allow", "MIT"),
    ("MIT license", "allow", "MIT"),
    ("MIT", "allow", "MIT"),
    ("BSD-3-Clause", "allow", "BSD"),
    ("New BSD License", "allow", "BSD"),
    ("BSD style", "allow", "BSD"),
    ("ISC", "allow", "ISC"),
    ("Eclipse Distribution License - v 1.0", "allow", "EDL-1.0 (BSD-style)"),
    ("EDL 1.0", "allow", "EDL-1.0 (BSD-style)"),
    ("Bouncy Castle Licence", "allow", "Bouncy Castle (MIT-style)"),
    ("Public Domain", "allow", "Public Domain / CC0"),
    ("Unicode-3.0", "allow", "Unicode"),
    (
        "Similar to Apache License but with the acknowledgment clause removed",
        "allow",
        "JDOM (Apache-style)",
    ),
    ("Android Software Development Kit License", "vendor", "Google: Android SDK Terms"),
    (
        "Android Software Development Kit License Agreement",
        "vendor",
        "Google: Android SDK Terms",
    ),
    (
        "Play Core Software Development Kit Terms of Service",
        "vendor",
        "Google: Play Core Terms",
    ),
    ("Play Integrity API Terms of Service", "vendor", "Google: Play Integrity Terms"),
    # The JNA strings contain "Apache License v2.0", and trove4j never writes "LGPL".
    ("GNU LESSER GENERAL PUBLIC LICENSE 2.1", "copyleft", "GPL/LGPL"),
    ("LGPL, version 2.1", "copyleft", "GPL/LGPL"),
    ("LGPL, version 2.1 / Apache License v2.0", "copyleft", "GPL/LGPL"),
    ("Apache License v2.0 / LGPL, version 2.1", "copyleft", "GPL/LGPL"),
    ("ASL, version 2 / LGPL, version 2.1", "copyleft", "GPL/LGPL"),
    ("GPL-3.0", "copyleft", "GPL/LGPL"),
    ("GNU General Public License, version 2", "copyleft", "GPL/LGPL"),
    ("CDDL + GPLv2 with classpath exception", "copyleft", "GPL/LGPL"),
    ("GNU Affero General Public License v3.0", "copyleft", "GPL/LGPL"),
    ("Server Side Public License", "copyleft", "SSPL"),
    ("Mozilla Public License 1.1 (MPL 1.1)", "copyleft", "MPL-1.x"),
    ("Eclipse Public License 1.0", "copyleft", "EPL"),
    ("Eclipse Public License - v 2.0", "copyleft", "EPL"),
    ("Common Development and Distribution License", "copyleft", "CDDL"),
    ("Some Proprietary Vendor Agreement", "unknown", None),
)


def selftest() -> int:
    failures = []
    for declared, want_verdict, want_canonical in SELFTEST:
        verdict, canonical = classify(declared)
        if verdict != want_verdict or (want_canonical and canonical != want_canonical):
            failures.append(
                (declared, want_verdict, want_canonical, verdict, canonical)
            )
    for declared, wv, wc, gv, gc in failures:
        print(f"FAIL {declared!r}\n     want {wv}/{wc}  got {gv}/{gc}", file=sys.stderr)
    if failures:
        print(
            f"\n{len(failures)}/{len(SELFTEST)} classification self-tests failed",
            file=sys.stderr,
        )
        return 1
    print(f"ok: {len(SELFTEST)} classification self-tests passed")
    return 0


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def classify(name: str) -> tuple[str, str]:
    """Return (verdict, canonical); verdict is allow | vendor | copyleft | unknown."""
    n = normalize(name)
    for pattern, canonical in COPYLEFT:
        if re.search(pattern, n):
            return "copyleft", canonical
    for pattern, canonical in ALLOWED:
        if re.search(pattern, n):
            return "allow", canonical
    for pattern, canonical in VENDOR_TERMS:
        if re.search(pattern, n):
            return "vendor", canonical
    return "unknown", name


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

# Matches `gradle dependencies` tree lines such as:
#   +--- androidx.core:core-ktx:1.17.0
#   |    \--- androidx.core:core:1.5.0 -> 1.18.0 (*)
#   +--- com.squareup.okhttp3:okhttp:{strictly 5.3.2} -> 5.3.2 (c)
COORD_RE = re.compile(
    r"[\\+|`\-\s]*---\s+"
    r"(?P<group>[A-Za-z0-9_.\-]+):(?P<artifact>[A-Za-z0-9_.\-]+)"
    r"(?::(?P<version>[^\s(]+))?"
    r"(?:\s+->\s+(?P<resolved>[^\s(]+))?"
)


def resolve_configuration(configuration: str) -> list[tuple[str, str, str]]:
    """Run `gradle dependencies` and return the resolved coordinates.

    Sentry's dependencies depend on whether a DSN is configured, so
    PIPETTE_ENABLE_CRASH_REPORTING=1 forces them in: the graph is then the one we
    distribute, whatever this machine has configured.
    """
    env = {**os.environ, "PIPETTE_ENABLE_CRASH_REPORTING": "1"}
    proc = subprocess.run(
        [
            "./gradlew",
            ":app:dependencies",
            "--configuration",
            configuration,
            "--console=plain",
            "-q",
        ],
        cwd=GRADLE_PROJECT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            f"gradle failed to resolve {configuration}:\n{proc.stdout}\n{proc.stderr}"
        )

    found = set()
    unparsed = set()
    for line in proc.stdout.splitlines():
        if "---" not in line or "project :" in line:
            continue
        m = COORD_RE.search(line)
        if not m:
            continue
        # "1.5.0 -> 1.18.0": the resolved version is what ships.
        version = m.group("resolved") or m.group("version")
        if not version:
            continue  # BOM-managed entry with no version on this line
        version = version.strip("{}").replace("strictly ", "").strip()
        if not re.match(r"^[0-9]", version):
            # Fail closed: a skipped artifact would ship ungated.
            unparsed.add(f"{m.group('group')}:{m.group('artifact')} -> {version!r}")
            continue
        found.add((m.group("group"), m.group("artifact"), version))
    if unparsed:
        sys.exit(
            f"could not parse a resolved version for {len(unparsed)} coordinate(s) "
            f"in {configuration}; they would ship ungated:\n  "
            + "\n  ".join(sorted(unparsed))
            + "\n\nFix COORD_RE / resolve_configuration in ci/android-licenses.py "
            "to understand this version form."
        )
    if not found:
        sys.exit(f"no coordinates parsed from {configuration}: output format changed?")
    return sorted(found)


def pom_from_cache(group: str, artifact: str, version: str) -> bytes | None:
    d = os.path.join(GRADLE_CACHE, group, artifact, version)
    if not os.path.isdir(d):
        return None
    for sha in os.listdir(d):
        p = os.path.join(d, sha, f"{artifact}-{version}.pom")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                return fh.read()
    return None


def pom_from_network(group: str, artifact: str, version: str) -> bytes | None:
    path = f"{group.replace('.', '/')}/{artifact}/{version}/{artifact}-{version}.pom"
    for repo in REPOS:
        try:
            with urllib.request.urlopen(f"{repo}/{path}", timeout=30) as resp:
                body = resp.read()
                if body.lstrip().startswith(b"<"):
                    return body
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return None


_pom_cache: dict[tuple[str, str, str], bytes | None] = {}


def pom(group: str, artifact: str, version: str) -> bytes | None:
    key = (group, artifact, version)
    if key not in _pom_cache:
        _pom_cache[key] = pom_from_cache(*key) or pom_from_network(*key)
    return _pom_cache[key]


def _text(el) -> str | None:
    return " ".join(el.text.split()) if el is not None and el.text else None


def _find(el, tag):
    """Look up a child tag with and without the Maven namespace."""
    return (
        el.find(f"m:{tag}", POM_NS)
        if el.find(f"m:{tag}", POM_NS) is not None
        else el.find(tag)
    )


def licenses_of(
    group: str,
    artifact: str,
    version: str,
    depth: int = 0,
    seen: frozenset = frozenset(),
) -> tuple[list[str], str | None]:
    """Declared license names, walking <parent> when a POM declares none itself.

    Many POMs (AndroidX, Kotlin, JAXB) inherit their license from an ancestor,
    so a self-only read would report them as undeclared.
    """
    if depth > 6:
        return [], None
    body = pom(group, artifact, version)
    if body is None:
        return [], None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return [], None

    names = []
    for lic in root.findall(".//m:licenses/m:license", POM_NS) or root.findall(
        ".//licenses/license"
    ):
        name = _text(_find(lic, "name"))
        if name:
            names.append(name)
    if names:
        via = None if depth == 0 else f"{group}:{artifact}:{version}"
        return names, via

    parent = _find(root, "parent")
    if parent is not None:
        pg, pa, pv = (
            _text(_find(parent, "groupId")),
            _text(_find(parent, "artifactId")),
            _text(_find(parent, "version")),
        )
        if pg and pa and pv and (pg, pa, pv) not in seen:
            return licenses_of(pg, pa, pv, depth + 1, seen | {(pg, pa, pv)})
    return [], None


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--configuration",
        default="kotlinEngineReleaseRuntimeClasspath",
        help="Gradle configuration to gate (default: %(default)s)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every artifact and its license, then exit 0",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="check the license classifier against known strings, then exit",
    )
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    coords = resolve_configuration(args.configuration)
    print(f"{args.configuration}: {len(coords)} resolved artifacts")

    undeclared, copyleft, unknown = [], [], []
    summary: collections.Counter[str] = collections.Counter()
    rows = []

    for group, artifact, version in coords:
        coord = f"{group}:{artifact}:{version}"
        names, via = licenses_of(group, artifact, version)
        if not names:
            undeclared.append(coord)
            rows.append((coord, "<undeclared>", "unknown"))
            continue
        for name in names:
            verdict, canonical = classify(name)
            summary[canonical] += 1
            rows.append(
                (coord, name if via is None else f"{name} (via {via})", verdict)
            )
            if verdict == "copyleft":
                copyleft.append((coord, name, canonical))
            elif verdict == "unknown":
                unknown.append((coord, name))

    if args.list:
        for coord, name, verdict in rows:
            print(f"  {verdict:8} {coord:70} {name}")

    print("\nlicense summary:")
    for canonical, count in summary.most_common():
        print(f"  {count:4}  {canonical}")

    if args.list:
        return 0

    failed = False

    if copyleft:
        failed = True
        print(
            f"\nFAIL: {len(copyleft)} copyleft-licensed artifact(s) would ship in "
            f"{args.configuration}:",
            file=sys.stderr,
        )
        for coord, name, canonical in copyleft:
            print(f"  {coord}\n      declares: {name}  [{canonical}]", file=sys.stderr)
        print(
            "\n  A copyleft dependency in a distributed configuration is a\n"
            "  licensing obligation, not a lint failure. Do not add it to the\n"
            "  allow-list to get green: either drop the dependency, replace it,\n"
            "  confirm it is dual-licensed and elect the permissive option, or\n"
            "  get a licensing decision on record.\n"
            "  Trace it with: cd android/Pipette && ./gradlew :app:dependencies "
            f"--configuration {args.configuration}",
            file=sys.stderr,
        )

    if unknown:
        failed = True
        print(
            f"\nFAIL: {len(unknown)} artifact(s) declare a license this gate does not "
            "recognize:",
            file=sys.stderr,
        )
        for coord, name in unknown:
            print(f"  {coord}\n      declares: {name!r}", file=sys.stderr)
        print(
            "\n  Unrecognized is a failure by design. It means a human has not\n"
            "  yet classified this license. Identify it, then add a pattern to\n"
            "  ALLOWED (if permissive) or COPYLEFT (if not) in this script.",
            file=sys.stderr,
        )

    if undeclared:
        failed = True
        print(
            f"\nFAIL: {len(undeclared)} artifact(s) declare no license in their POM:",
            file=sys.stderr,
        )
        for coord in undeclared:
            print(f"  {coord}", file=sys.stderr)
        print(
            "\n  Check the project's own LICENSE file and record the finding,\n"
            "  either by adding a pattern here or by removing the dependency.",
            file=sys.stderr,
        )

    if failed:
        return 1
    print(
        f"\nok: every artifact in {args.configuration} is permissively licensed "
        "or ships under accepted vendor terms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
