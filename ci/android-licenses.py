#!/usr/bin/env python3
"""License gate for the Android app's *shipping* dependencies.

Resolves a Gradle configuration, reads each resolved artifact's declared
license from its POM, and fails on anything outside the allow-list below.

    ./ci/android-licenses.py                     # gate releaseRuntimeClasspath
    ./ci/android-licenses.py --list              # print the graph, gate nothing
    ./ci/android-licenses.py --configuration debugRuntimeClasspath

It runs as a step in the `android-app` job, where Gradle, a JDK and the Android
SDK are on hand (ci/checks/* runs in `conventions`, which carries no toolchain —
see ci/README.md). It reads the graph by parsing `gradle dependencies`, which
leaves the build untouched: fetching POMs from inside a Gradle task needs
`ArtifactResolutionQuery`, and that is incompatible with the configuration cache
this project enables.

Scope: `releaseRuntimeClasspath`, which is what lands in the APK/AAB. Build-time
and test-only configurations are out of scope by design. They carry copyleft
artifacts — detekt pulls LGPL-2.1 trove4j, lint pulls dual-licensed JNA, JUnit is
EPL-1.0 — that are never distributed and so impose no obligation, and a gate that
fails on dependencies which do not matter is one people learn to bypass.
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

# Maven repositories to fall back to when a POM is not in the local Gradle
# cache. Order matters only for speed. These mirror the repositories declared in
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
# POM license strings are free text, not SPDX ids: this graph alone spells
# Apache-2.0 four different ways. So each declared string is normalized and
# matched against the patterns below to reach a canonical id, and only canonical
# ids are allowed. An unrecognized string is a FAILURE, not a pass — that is
# what makes a newly introduced copyleft dependency turn CI red instead of
# sliding through under a spelling nobody predicted.
#
# Patterns are matched against the license name lowercased with runs of
# non-alphanumerics collapsed to single spaces. Note what that means for version
# numbers and parentheses: "Apache-2.0", "Apache 2.0" and "The Apache Software
# License, Version 2.0" all normalize with "2 0" (the dot becomes a space), and
# "The MIT License (MIT)" normalizes to "the mit license mit". Write patterns
# against the normalized form, and run `--selftest` after editing them.

# Checked FIRST, and never allowed. Listed explicitly rather than relying on
# "unrecognized fails", so a copyleft license can never be mistaken for a
# permissive one by a loose pattern below (e.g. "LGPL, version 2.1 / Apache
# License v2.0" must not match on the Apache half).
COPYLEFT = (
    # `\bl?gpl\b` alone misses the version-suffixed spellings ("GPLv2",
    # "LGPL2.1") because the digit leaves no word boundary after "gpl".
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

# Permissive licenses, allowed to ship. Keep in sync with the Rust workspace's
# allow-list in //deny.toml — those two are the whole policy.
ALLOWED = (
    # "apache 2", "apache 2 0", "apache license 2 0", "apache license v2 0",
    # "the apache software license version 2 0", …
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
    # Eclipse *Distribution* License is BSD-style — unrelated to the Eclipse
    # Public License, which COPYLEFT catches above and which is checked first.
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

# Non-OSS vendor terms that ship knowingly. These are NOT open-source licenses;
# they are accepted because using the vendor's SDK is a deliberate product
# decision, and they impose no source-disclosure obligation. Every entry here is
# a choice someone made, so each one names what pulls it in.
VENDOR_TERMS = (
    # All 12 arrive transitively through com.clerk:clerk-android-api — the
    # play-services-* set via androidx.credentials:credentials-play-services-auth
    # (passkeys/FIDO), plus Play Integrity and googleid. Nothing in this repo
    # depends on Play Services directly.
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


# Every license string this graph actually declares, plus copyleft strings that
# MUST be rejected. Guards the patterns above against a subtle failure mode: an
# over-tight pattern silently reclassifies a permissive license as "unknown"
# (noisy but safe), while an over-loose one can classify a copyleft license as
# permissive (quietly wrong). Run with --selftest; CI runs it before the gate.
SELFTEST = (
    # (declared string, expected verdict, expected canonical)
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
    # Vendor terms — allowed to ship, but must NOT be mistaken for open source.
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
    # Copyleft — every one of these must be rejected. The dual-licensed and
    # spelled-out forms are the ones a naive pattern gets wrong: the JNA strings
    # contain "Apache License v2.0", and trove4j never writes "LGPL".
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
    # Genuinely unclassified must stay a failure, not fall through to allow.
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
    """Return (verdict, canonical).

    verdict is one of: allow | vendor | copyleft | unknown.
    """
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

# Matches a coordinate in `gradle dependencies` tree output, e.g.
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

    Resolves the CANONICAL shipping graph, not whatever this machine happens to
    be configured for. Sentry's dependencies are conditional on a DSN being
    configured (see app/build.gradle.kts), so without pinning this the resolved
    set would differ between a maintainer with a DSN in local.properties, CI with
    the secret, and a fork with neither. The committed attribution asset could
    then only ever match one of them, and the other two would fail the freshness
    check on a file they cannot legitimately change.

    PIPETTE_ENABLE_CRASH_REPORTING=1 forces Sentry to be wired in regardless of
    the DSN, so the graph is the one we distribute. It does not make anything
    report: the manifest DSN stays empty, so the SDK never initializes.
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
        # A line may state a requested version and the version conflict
        # resolution actually picked ("1.5.0 -> 1.18.0"); the latter is what
        # ships, so it wins.
        version = m.group("resolved") or m.group("version")
        if not version:
            continue  # BOM-managed entry with no version on this line
        version = version.strip("{}").replace("strictly ", "").strip()
        if not re.match(r"^[0-9]", version):
            # A version this parser cannot read must never be dropped silently:
            # everything else here fails closed, and an artifact skipped at this
            # point would ship completely ungated. Collect and fail instead.
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
        default="releaseRuntimeClasspath",
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

    # --list is an inventory mode, so it reports and stops. Falling through to
    # the gate would exit 1 on a graph someone asked only to look at.
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
