# Third-party licensing

What may enter each client's dependency graph, what enforces it, and what is
still unenforced.

## Policy

Permissive licenses only in anything distributed: MIT, Apache-2.0, BSD, ISC,
Zlib, Unicode, and the small permissive tail (0BSD, CC0, CDLA-Permissive-2.0,
EDL-1.0, Bouncy Castle). MPL-2.0 is accepted. Its copyleft is per-file and does
not reach our sources.

**No GPL, LGPL, AGPL, SSPL, CDDL, EPL, CPL or MPL-1.x in a distributed
artifact.** A dual-licensed dependency offering a permissive option (`MIT OR
Apache-2.0 OR LGPL-2.1-or-later`) is fine. The permissive option is elected.
Election is automatic only for Rust, where cargo-deny evaluates the SPDX `OR`.
A Maven POM instead lists separate `<license>` elements with no operator, so the
Android gate cannot tell a choice from a conjunction and fails on any copyleft
element; an artifact of that shape (JNA is the canonical example) needs a
per-coordinate decision recorded in `ci/android-licenses.py`, not an allow-list
entry.

Two deliberate carve-outs:

- **Android build-time and test-only dependencies are not gated.** They carry
  copyleft today and it imposes no obligation, because nothing is distributed:
  detekt pulls LGPL-2.1 `trove4j`, Android lint pulls dual-licensed `JNA`,
  JUnit 4 is EPL-1.0, `javax.annotation-api` is CDDL +
  GPLv2-with-classpath-exception. Gating these would fail on dependencies that
  do not matter, which is how a gate gets routed around.

  This carve-out is **Android only**. `cargo deny` leaves `exclude-dev` at its
  default of `false`, so the Rust gate covers dev- and build-dependencies too.
  a copyleft `[dev-dependencies]` entry turns `rust-check` red. That is
  deliberate (stricter costs nothing there, since the Rust test graph is ours),
  but it means the two gates have genuinely different scopes.
- **Non-OSS vendor terms ship knowingly.** The Android app distributes 12
  artifacts under Google's Android SDK / Play terms, all arriving transitively
  through `com.clerk:clerk-android-api` (the `play-services-*` set via
  `androidx.credentials:credentials-play-services-auth` for passkeys/FIDO, plus
  Play Integrity and `googleid`). Not copyleft, but not open source either, and
  they carry Play-ecosystem obligations the MIT-licensed Clerk SDK does not.

## What enforces it

| Surface | Gate | Runs in |
|---|---|---|
| Rust workspace | `cargo deny check licenses` ([`deny.toml`](../deny.toml)) | `rust-check` (linux leg) |
| Rust advisories | `cargo deny check advisories` ([`deny.toml`](../deny.toml)) | `rust-check` (linux leg) |
| Android shipping deps | [`ci/android-licenses.py`](../ci/android-licenses.py) | `android-app` |
| iOS SwiftPM graph | hand audit at review time | — |

Run either locally:

```sh
cargo deny check licenses                 # needs cargo-deny
./ci/android-licenses.py                  # needs Gradle + Android SDK
./ci/android-licenses.py --selftest       # just the license classifier
./ci/android-licenses.py --list           # every artifact and its license
```

### Rust

`deny.toml` is an allow-list, so an unrecognized license fails rather than
passes. It resolves the graph for all six shipped targets with `all-features`,
so a dependency that only appears under `cfg(target_os = "android")` or behind a
feature flag is still checked.

The same file also carries the RUSTSEC security policy in `[advisories]`. That
gate is not a licensing concern, so it is not described here; see the comments in
`deny.toml` for what it denies and which advisories are deliberately ignored.

The `pipette-*` workspace members declare no `license` field, since nothing here
is published. Each sets `publish = false`, which is what `[licenses.private]
ignore = true` keys off to skip first-party crates while still checking every
third-party one. **A new member crate must set `publish = false` or the gate
fails it as unlicensed.**

### Android

Gates `releaseRuntimeClasspath` (what actually lands in the APK/AAB) by
resolving it with Gradle and reading each artifact's declared license from its
POM, following `<parent>` chains (AndroidX, Kotlin and JAXB all inherit their
license from an ancestor POM).

POM license strings are free text, not SPDX ids: the current graph spells
Apache-2.0 four different ways. So each string is normalized and matched against
patterns to reach a canonical id, copyleft patterns are checked *first* (so
`"LGPL, version 2.1 / Apache License v2.0"` cannot match on its Apache half),
and anything unrecognized fails. `--selftest` covers every spelling the graph
declares plus copyleft strings that must be rejected; CI runs it before the gate.

### iOS: hand-audited

The 22 SwiftPM packages are audited by hand and are all MIT or Apache-2.0. This
is the one graph held to the policy by review rather than by a gate.

Automating it takes more work than the other two, because SwiftPM carries no
license metadata: `Package.swift` has no license field, and `Package.resolved`
records identity, repo URL and revision only. So a gate must derive the license
from the source: either by matching each checkout's `LICENSE` text, or by asking
the forge, e.g. GitHub's `/repos/{owner}/{repo}/license` endpoint for an SPDX id.

Attribution here is already covered, and is a separate concern: the LicenseList
SwiftPM plugin generates it at build time for every package in the graph, and
`ios/generate-third-party-licenses.py` covers the one thing the plugin cannot
see; the vendored llama.cpp.

## Attribution

Distinct from the gates above: the gates decide what may ship, attribution
discharges the notice requirement for what does. Apache-2.0 and MIT both require
it, and it is the only obligation they impose.

| Client | What produces it | Covers |
|---|---|---|
| Android | [`ci/android-attribution.py`](../ci/android-attribution.py) → `app/src/main/assets/ThirdPartyLicenses.json` | vendored native + linked Rust crates + all Maven artifacts |
| iOS | LicenseList SwiftPM plugin (build time) + `ios/generate-third-party-licenses.py` | SwiftPM graph + vendored llama.cpp |
| CLIs | nothing: not distributed as binaries | — |

```sh
./ci/android-attribution.py --write   # regenerate, then commit the result
./ci/android-attribution.py --check   # CI: fail if the committed file is stale
```

The Android file is generated, never edited by hand: the graphs it covers move
with every dependency change, in both directions; a component dropped from the
build must lose its entry, and one added must gain it. `--check` runs in both the
CI and release workflows to hold the committed file to the resolved graph.

Two things to know about the output:

- Rust crates and vendored sources get **their own** license text, so per-project
  copyright lines survive. Maven artifacts cannot: POMs declare a license name,
  never its text. They are grouped under the canonical SPDX text, which is what
  mainstream Android attribution tooling does.
- Groups are labelled compactly (`"a, b + N more"`) to keep the settings list
  readable, but every component in a group is named in that entry's license
  text. Nothing relies on the label to discharge attribution.

## Vendored native code

Statically linked into the mobile apps, licenses checked at submodule bump:

| Submodule | License |
|---|---|
| `vendor/llama.cpp` | MIT |
| `vendor/kleidiai` | Apache-2.0 / BSD-3-Clause |
| `vendor/sentry-native` | MIT |

Engines the `pipette` CLI drives as external processes (upstream `llama.cpp`
release binaries (MIT), `mlx-lm` in a managed `uv` venv (MIT), Docker-hosted
vLLM/SGLang (Apache-2.0)) are downloaded at runtime, not linked or
redistributed, so they are outside both gates.
