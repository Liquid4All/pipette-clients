# Contributing

Thanks for your interest in Pipette Clients. This file covers how to get a
checkout that builds, which checks to run before you open a pull request, and
the conventions the repository expects.

Start at the [top-level README](README.md) for what the repository contains,
and [docs/README.md](docs/README.md) for the full documentation index. The
local checks below mirror the relevant CI gates where noted; the authoritative
definition is [`.github/workflows/build.yml`](.github/workflows/build.yml).

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

Clone with submodules. There are three, all under `vendor/`: `llama.cpp`
(shared by the iOS and Android builds), `kleidiai`, and `sentry-native`.

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without them, initialize them in place; the same command
the [Android build guide](docs/pipette-android/build.md#vendored-submodules)
gives:

```bash
git submodule update --init --recursive
```

**Git LFS.** One tracked file is stored in LFS: the statically linked
`crates/pipette-llamacpp/vendor/toybox/toybox-aarch64` binary, which the Android
memory-measurement path embeds with `include_bytes!`. Have Git LFS installed
*before* you clone or the working tree gets a 131-byte pointer file instead of
the binary, the artifact bakes that pointer in, and the wrapped
`toybox time -v llama-bench` exits 127 at run time. This is why CI checks out
with `lfs: true` on the main-workflow jobs that build Android artifacts
(`android-app`, `build-artifact`) and not on the rest; the Android Firebase
Distribution workflow does the same for APK distribution builds.

A host build of the CLI needs neither the mobile toolchains nor LFS:

```bash
cargo build --release -p pipette-cli   # produces the `pipette` binary
```

Per-platform build guides: [iOS](docs/pipette-ios/build.md) ·
[Android](docs/pipette-android/build.md).

### Toolchain versions

These are pinned in `build.yml`. Matching them locally is the cheapest way to
reduce CI drift.

| Tool | Version | Where it is pinned |
|------|---------|--------------------|
| Rust | stable | `dtolnay/rust-toolchain@stable` (all Rust jobs) |
| JDK | 17 (temurin) | `android-app`, `android-lint` |
| Android NDK | 27.2.12479018 | `android-app` |
| Xcode | 26.4, via `DEVELOPER_DIR` | `ios-check` |
| Python | 3.12 | `python-check` (also `target-version` in `ruff.toml`) |
| SwiftLint | 0.65.0 | `ios-check` |

**Do not "helpfully" bump the Xcode pin.** `ios-check` sets
`DEVELOPER_DIR=/Applications/Xcode_26.4.app` instead of taking the runner image's
default. Keep the workflow comment and the pin in sync if this changes: the pin
exists to keep the iOS app build, simulator tests, and `xcrun` calls inside
`build-llama.sh` on the same validated SDK/toolchain. Moving it without
revalidating iOS CI can reintroduce hard-to-attribute compiler or runtime
failures.

## Local checks, by area

Run the checks for whatever you touched. They mirror the relevant CI gates, but
CI remains authoritative.

### Repository conventions: always

`./ci/lint.sh` is the one check that runs on **every** PR: the `conventions` job
has no path filter. Run it whatever you changed.

```bash
./ci/lint.sh
```

### Rust

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo nextest run --workspace
cargo test --doc --workspace
```

Nothing is excluded from the clippy and test passes. `nextest` runs the test
binaries in parallel but does not execute doctests, which is why the separate
`--doc` pass exists. Run both. `cargo-nextest` is a separate install (CI
installs it in the `rust-check` job). `pipette-mlx` is `cfg(target_os = "macos")`
and resolves to an empty crate elsewhere, so its lints and tests only mean
anything on a Mac.

### Python

```bash
ruff check .
ruff format --check .
./crates/pipette-mlx/test-python.sh
```

The current Python-backed crate, `pipette-mlx`, owns its Python tests and dev
dependencies; `test-python.sh` requires `uv`.

### iOS

```bash
./ios/build-llama.sh sim
./ios/gen-mlx-build-info.sh
```

SwiftLint runs in strict mode from the `ios/` directory, against the config
there:

```bash
cd ios && swiftlint lint --strict --config .swiftlint.yml
```

CI uses SwiftLint 0.65.0 exactly; a different version can disagree about what is
a violation. The lint is source-only and runs before the compile so it fails
fast. Compiler warnings are gated separately, via
`SWIFT_TREAT_WARNINGS_AS_ERRORS` on the app target. For building and testing the
app itself, see [docs/pipette-ios/build.md](docs/pipette-ios/build.md); CI's
exact `xcodebuild build-for-testing` / `test-without-building` invocations (and
the flags they must carry) live in the `ios-check` job.

### Android

From `android/Pipette/`:

```bash
./gradlew :app:assembleDebug :app:assembleRelease :app:testDebugUnitTest
./gradlew :app:ktfmtCheck :app:detekt
```

Formatting failures are fixed with `./gradlew :app:ktfmtFormat`. Detekt findings
must be fixed, or (rarely) baselined with `./gradlew :app:detektBaseline`.
`ktfmtCheck`/`detekt` need only the JDK and Android SDK, so they are quick; the
assemble tasks pull in the NDK, Rust and the vendored native build.

## Three things that will otherwise cost you an afternoon

**1. Do not add `-- -D warnings` to clippy.** Lint levels come from
`[workspace.lints]` in the root `Cargo.toml`, and each member opts in with
`[lints] workspace = true`. That is the single source of truth: `warnings` is
denied there, along with the ban on panicking constructs (`unwrap_used`,
`expect_used`, `panic`, `todo`, `unreachable`). The CI clippy steps deliberately
pass no `-D` flags. See the comment on the Clippy step in `rust-check`. Adding
them locally papers over a workspace lint policy that should be edited in one
place instead.

**2. A skipped job is not a failed job.** The `changes` job runs
`dorny/paths-filter`, and `rust-check`, `python-check`, `ios-check`,
`android-app`, `android-lint` and `build-artifact` are gated on it, so on a PR
that touches none of their paths they are skipped rather than run. (Pushes to
`main` always run everything.) `build-complete` is the single aggregate gate:
it `needs` all of them, always runs, and treats `skipped` as OK while failing on
any `failure` or `cancelled`. It is the check to require for branch protection
when you want one status that represents the whole path-filtered CI matrix.

**3. Regenerate and commit the build-info stamps.** `ios-check` has a freshness
gate, `Verify build-info stamps are committed and current`. Both generators run
in CI and the step then fails if either committed stamp differs from what was
just generated. If you bump the `vendor/llama.cpp` submodule or the SwiftPM
dependency graph (`Package.resolved`), re-run both generators and commit the
results:

```bash
./ios/build-llama.sh sim
./ios/gen-mlx-build-info.sh
```

then commit `ios/Pipette/Pipette/Generated/Llama/LlamaBuildInfo.swift` and
`ios/Pipette/Pipette/Generated/MLX/MLXBuildInfo.swift`. These stamps are what
the app's reported `runtime_version` is baked from, so a stale one can ship an
incorrect runtime version. Forget the regen and CI fails on a diff, which is not
an obvious symptom of "you bumped a submodule".

## Repository conventions

`ci/checks/` holds repo-convention checks that Rust's own tooling cannot
express. `ci/lint.sh` runs every executable file in that directory via its own
shebang, so a check can be written in any language, and reports a pass/fail
summary. CI calls the same script (one step in the `conventions` job), so a
green local run matches the PR check. Full detail in [ci/README.md](ci/README.md).

Three checks exist today:

| Check | Enforces |
|-------|----------|
| `anyhow-imports` | Only `use anyhow::Context;` may be imported; `Result` / `bail!` / `anyhow!` / `Error` stay `anyhow::`-qualified, so a bare `Result` is never ambiguous with a crate-local one. |
| `import-groups` | `use` statements form blank-line-separated groups (std, third-party, workspace (`pipette_*`), crate-local), which rustfmt cannot split. |
| `rust-path-refs` | Selected Rust module paths cited from Kotlin, Swift and Markdown still resolve, so a Rust-side rename cannot silently strand those references. |

To add a check, drop a self-contained executable `ci/checks/<name>.{sh,py,…}`
into the directory (with a shebang and `chmod +x`) that:

- inspects tracked files via `git ls-files`,
- exits non-zero on a violation, with a message stating both the rule and the
  fix,
- exits zero otherwise.

`lint.sh` picks it up with no workflow edit. Reserve this for conventions
rustfmt and clippy genuinely cannot express; if a lint can, prefer the lint.
Checks live here rather than in a per-language job precisely because
`conventions` has no path filter. That is what lets `rust-path-refs` catch a
Rust-only change breaking a reference from `android/`, `ios/` or `docs/`, which
those languages' own jobs never see.

## Commits and pull requests

Prefer commit subjects that follow the style already in `git log`:

```
type(scope): imperative summary (PIP-123) (#456)
```

- **type** is one of `feat`, `fix`, `refactor`, `chore`, `docs`, `ci`, `test`.
- **scope** is optional and usually names the crate, runtime, platform, or shared
  area the change lands in. Omit it for changes that span the repository.
- The summary is lower-case, imperative, and says what the change does rather
  than what it touches.
- The issue key (`PIP-123`) goes in parentheses when the change has one.
- The trailing `(#456)` is the PR number added by GitHub on squash merge: don't
  write it by hand.

Use a **Problem** / **Solution** / **Testing** structure in PR descriptions:
what was wrong, what the change does about it (a bulleted list when there is
more than one move), and what you actually ran to convince yourself. Close out
issue keys with `Closes PIP-123` / `Refs PIP-123` at the end.

Use the GitHub issue templates for bug reports and feature requests.

**Stacked PRs are supported.** The `pull_request` trigger carries no
base-branch filter, so a PR targeting an intermediate head (not just `main`)
triggers the CI workflow. Jobs still follow the path filters described above.
Stack freely rather than folding unrelated work into one branch.

## License

Copyright 2026 Liquid AI, Inc.

Licensed under the Apache License, Version 2.0 (the "License"). You may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this repository is licensed under Apache-2.0, without any
additional terms or conditions.
