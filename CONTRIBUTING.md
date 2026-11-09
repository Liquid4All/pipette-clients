# Contributing

Thanks for your interest in Pipette Clients. This file covers how to get a
checkout that builds, which checks to run before you open a pull request, and
the conventions the repository expects.

Start at the [top-level README](README.md) for what the repository contains,
and [docs/README.md](docs/README.md) for the full documentation index. The
local checks below mirror the relevant CI gates where noted; the authoritative
definition is [`.github/workflows/build.yml`](.github/workflows/build.yml), plus
[`security.yml`](.github/workflows/security.yml) for the supply-chain and
secret-scanning gates. [`codeql.yml`](.github/workflows/codeql.yml) runs GitHub's
CodeQL over Rust, Python and the workflow files. It keeps its findings as a SARIF
artifact on each run for 14 days and does not publish them to code scanning. Its
`codeql-complete` check fails only when CodeQL itself fails, not on a finding.
A push to a `release/<train>` branch publishes a release; see
[Releasing](#releasing).

`build.yml`, `security.yml` and `codeql.yml` also run on every push to `main`.
A failure there belongs to no pull request, so it appears on no one's checks
list. [`main-ci-watch.yml`](.github/workflows/main-ci-watch.yml) reports it: it
opens one issue per red workflow, labeled `main-red`, that names the commit and
the failed jobs. The next green run of that workflow closes the issue.

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

Clone with submodules. There are eight, all under `vendor/` and listed in
`.gitmodules`: `llama.cpp` (shared by the iOS and Android builds), `kleidiai`,
`sentry-native`, `onnxruntime`, `onnxruntime-genai`, `llamacpp-prismml`,
`llamacpp-prismml-b10828` and `llamacpp-lfm3` (upstream llama.cpp at a later
commit). The last three are iOS
[llama.cpp variants](docs/pipette-ios/build.md#llamacpp-variants).

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without them, initialize them in place:

```bash
git submodule update --init --recursive
```

**Git LFS.** Nothing in the tree is LFS-tracked. `.gitattributes` is the list
to check, and CI fetches LFS only on the Android legs.

A host build of the CLI needs neither the mobile toolchains nor LFS:

```bash
cargo build --release -p pipette-cli   # produces the `pipette` binary
```

**`pipette-plan-types` comes from
[pipette-mgmt](https://github.com/Liquid4All/pipette-mgmt).** The management
server owns the plan vocabulary, and the root `Cargo.toml` takes it by git over
HTTPS, pinned to one commit. The repository is public, so a build needs no
credentials. A change to the vocabulary lands in pipette-mgmt first; then bump
`rev` in the root `Cargo.toml` and run `cargo update -p pipette-plan-types`.

To build against a local pipette-mgmt checkout, override the git source for one
command. Use this for a change that spans both repositories. A relative `path`
starts at the current directory:

```bash
cargo --config 'patch."https://github.com/Liquid4All/pipette-mgmt.git".pipette-plan-types.path="../pipette-mgmt/crates/pipette-plan-types"' \
  check -p pipette-ctl
```

The override removes the crate's `source` line from `Cargo.lock`. Run
`git checkout Cargo.lock` before you commit.

Per-platform build guides: [iOS](docs/pipette-ios/build.md) ·
[Android](docs/pipette-android/build.md).

### Toolchain versions

Matching CI's toolchains locally is the cheapest way to reduce drift. Read the
versions where they are pinned rather than from a copy here:

- Rust tracks the current stable release. The JDK, the Android NDK, Xcode,
  Python and SwiftLint are pinned in each `build.yml` job that uses them (search
  `build.yml` for the tool's name before you bump one). Python's version is also
  `target-version` in `ruff.toml`.
- cargo-deny, cargo-machete and gitleaks are pinned in `ci/install-tool.sh`,
  each as a release tarball with a SHA-256. Dependabot cannot see these pins;
  [ci/README.md](ci/README.md#whats-here) says how to bump them.

**Do not change the Xcode pin without revalidating iOS CI.** `ios-check` sets
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

`./ci/lint.sh` runs on **every** PR whatever it touches (the `conventions` job
has no path filter), so run it whatever you changed. How to add a check is in
[ci/README.md](ci/README.md); each check's header comment says what it enforces
and why.

```bash
./ci/lint.sh
```

The other unfiltered gates are the `security` workflow (see
[Supply chain and secrets](#supply-chain-and-secrets)) and the `pr-body`
workflow, which reads the pull request description rather than the tree. See
[Commits and pull requests](#commits-and-pull-requests).

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

The `rust-coverage` job measures line coverage for every crate, posts a
per-file table to its run summary and uploads `lcov.info`. It gates one crate:
the job fails if `pipette-cli`'s line coverage is below 90 %. To read the same
numbers locally for one crate (`cargo-llvm-cov` is a separate install):

```bash
cargo llvm-cov nextest -p pipette-cli --summary-only
```

### Supply chain and secrets

The `security` workflow runs these on every PR whatever it touched, plus daily
against `main`, because an advisory is published without anyone changing the
tree.

```bash
cargo deny check -D advisory-not-detected -D unnecessary-skip -D unmatched-skip advisories bans sources
cargo-machete
gitleaks git --redact --config .gitleaks.toml --log-opts="origin/main..HEAD" .
```

Needs cargo-deny 0.19 or newer; an older one aborts on any advisory carrying a
CVSS 4.0 vector before it reads `deny.toml`, which looks like a policy failure
but is a stale tool. `cargo deny check licenses` is the same config file
and the same pinned tool, but a different job (`rust-check`).

`-D advisory-not-detected` turns a stale `[advisories].ignore` entry, one whose
crate has left the lockfile, into an error rather than a warning that still
exits 0, so the change that drops the last affected crate deletes the ignore
with it. It is a lint level, so it only works on the command line.

cargo-machete greps the sources under a crate's own directory, so three shapes
read as dead when they are not:

- a crate reached only through a macro expansion (`regex`, via nutype's
  `validate(regex = ...)`);
- a crate named only by a module `#[path]`-included from outside the crate
  directory (`android/native/benchmarks.rs`, in `pipette-android`);
- a dependency listed only to turn a feature on for a crate something else also
  pulls. Nothing names it, and deleting it compiles fine wherever another crate
  happens to activate the same feature, so the loss shows up as a missing
  feature on some other target rather than as an error here.

Suppress these in that crate's `[package.metadata.cargo-machete]` with the
reason written next to it. Deleting the dependency is the wrong fix for all
three.

A gitleaks finding is a secret until shown otherwise: rotate first, then decide
what to do about the commit. `.gitleaks.toml` allow-lists values that are public
by design (the Clerk publishable key, the PostHog project key, the Firebase
Android key) and test fixtures. Adding a real credential to it would hide it
from the `main` and cron scans as well, so it is not a way to make a red check
go away. PostHog's configuration is in each app's doc:
[iOS](docs/pipette-ios/architecture.md#analytics-posthog),
[Android](docs/pipette-android/implementation.md#analytics-posthog).

### Python

```bash
ruff check .
ruff format --check .
./crates/pipette-mlx/test-python.sh
./crates/pipette-onnx-genai/test-python.sh
./crates/pipette-openvino/test-python.sh
```

Each Python-backed crate owns its Python tests and dev dependencies, and
`python-check` runs all three scripts. They require `uv`.

### iOS

```bash
./ios/build-llama.sh sim
./ios/build-onnx-genai.sh sim
./ios/gen-mlx-build-info.sh
./ios/gen-onnx-genai-build-info.sh
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
./gradlew :app:assembleKotlinEngineDebug :app:assembleKotlinEngineRelease :app:testKotlinEngineDebugUnitTest
./gradlew :app:ktfmtCheck :app:detekt
```

Formatting failures are fixed with `./gradlew :app:ktfmtFormat`. Detekt findings
must be fixed, or (rarely) baselined with `./gradlew :app:detektBaseline`.
`ktfmtCheck`/`detekt` need only the JDK and Android SDK, so they are quick. The
assemble tasks also pull in the NDK and the vendored native build (CMake). The
`kotlinEngine` flavor needs no Rust.

## Three common causes of a red CI run

**1. Do not add `-- -D warnings` to clippy.** Lint levels come from
`[workspace.lints]` in the root `Cargo.toml`, and each member opts in with
`[lints] workspace = true`. That is the single source of truth: `warnings` is
denied there, along with the ban on panicking constructs (`unwrap_used`,
`expect_used`, `panic`, `todo`, `unreachable`). The CI clippy steps deliberately
pass no `-D` flags. See the comment on the Clippy step in `rust-check`. To
change a lint level, edit `[workspace.lints]`, not the command line.

**2. A skipped job is not a failed job.** The `changes` job runs
`dorny/paths-filter`, and every compile and test job in `build.yml` is gated on
it, so on a PR that touches none of a job's paths that job is skipped rather
than run. (Pushes to `main` and releases always run everything.)
`build-complete` is the single aggregate gate: it `needs` every job, always runs, and treats `skipped` as OK while failing on
any `failure` or `cancelled`. It is the check to require for branch protection
when you want one status that represents the whole path-filtered CI matrix.

**3. Regenerate and commit the build-info stamps.** `ios-check` has a freshness
gate, `Verify build-info stamps are committed and current`. The generators run
in CI, and the step fails if a committed stamp differs from what they generate.
If you bump the SwiftPM dependency graph (`Package.resolved`) or the vendored
ONNX Runtime GenAI, re-run both generators:

```bash
./ios/gen-mlx-build-info.sh
./ios/gen-onnx-genai-build-info.sh
```

Then commit the `.swift` and `.json` files they write under
`ios/Pipette/Pipette/Generated/MLX/` and `ios/Pipette/Pipette/Generated/OnnxGenai/`.
The app bakes its reported `runtime_version` from these stamps, so a stale stamp
can ship an incorrect runtime version. If you skip this step, CI fails on a diff
in the generated files.

## Releasing

Merging to `main` does not release. To publish, push a `release/<train>` branch,
for example `release/2026.11.1`. Every push to that branch runs the whole
`build.yml` pipeline with the release version baked into the binaries. The
`github-release` job then creates a GitHub release with the tarballs, APKs,
installers, the iOS app archive (`pipette-ios-<tag>-internal.xcarchive.zip`,
the [private-API build](docs/pipette-ios/private-thermal-release-build.md),
ad-hoc signed and not signed for any device),
its manifest (`pipette-ios-<tag>-internal.json`) and a `.sha256` for each asset.
The manifest lists the runtimes that the build carries, with every llama.cpp pin
among them. `pipette-ios-proxy` reads these archives to run jobs on a device
lab's phones ([On the farm](docs/pipette-ios/build.md#on-the-farm-pipette-ios-proxy)).

[`ci/version.sh`](ci/version.sh) computes the version. Run it locally to see
what a commit publishes as:

- A release branch gives `<train>-<n>-g<sha>`, where `<n>` is the distance from
  the branch point on `main`.
- Any other ref gives `<train>-dev-<n>-g<sha>` from the nearest release tag, or
  `dev-<n>-g<sha>` when no tag is reachable.

Push and pull request runs on other branches build the same jobs but skip
`github-release`, and their binaries report a `dev` version so the build caches
stay warm. Publishes against one release branch queue rather than race on the
tag, and a re-run of a published commit skips the publish.

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

Every PR description carries a **Problem** / **Solution** / **Testing**
structure: what was wrong, what the change does about it (a bulleted list when
there is more than one move), and what you actually ran to convince yourself.
Close out issue keys with `Closes PIP-123` / `Refs PIP-123` at the end.

```markdown
**Problem**

What problem does this solve?

**Solution**

What changed?

**Testing**

What did you run?
```

[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) holds the
same skeleton. Replace each prompt with your answer.

[`ci/pr-body.py`](ci/pr-body.py) enforces this in the `pr-body` workflow. Three rules about content: the sections are present, they are in that
order, and each says something (the template's prompts left in place count as
saying nothing; a fenced block of the commands you ran counts as an answer). Two
about form: exactly one heading per section, and bold. Extra sections are free,
in any form and any position, which is what `Provenance`, `Stack`, `Follow-up`
and `Notes` already are.

**Headings are bold, not `##`.** The check takes only bold, so a description
skimmed at speed always has its sections in the same shape. `## Problem` fails
with a message naming the fix.

The check runs on `opened`, on `edited`, and on every push, so a description
fixed after the fact clears it without an empty commit. Run it before you push:

```bash
./ci/pr-body.py --pr 1234   # or a file path, or - for stdin
```

It is its own workflow rather than a job in `build.yml` because that one
deliberately does not trigger on `edited`: re-running the full Rust, iOS and
Android matrix every time somebody fixes a typo in a description is not worth
it.

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
