# ci/

Repo-convention checks that run in CI and locally.

Run them all before pushing:

```sh
./ci/lint.sh
```

## What's here

- `lint.sh` + `checks/`: the convention checks, run by the `conventions` job.
  They need no Rust, Swift or Android toolchain. They need `python3` 3.10 or
  later, because most checks are Python, and `shellcheck` on PATH for
  `checks/shellcheck.sh` (the CI runner image ships both; locally,
  `brew install shellcheck`). See "Adding a check" below.
- `android-licenses.py`: the license gate for the Android app's shipping
  dependencies.
- `android-attribution.py`: generates (`--write`), and verifies (`--check`) the
  `ThirdPartyLicenses.json` the Android app bundles.
- `android-16kb-page-size.sh`: checks that the Android native libraries, given
  as a directory or a packaged APK/AAB, load on 16 KB page-size devices. Run by
  the `android-app` job and by `android-release-distribution.yml`.
- `pr-body.py`: the Problem / Solution / Testing structure required of a pull
  request description.
- `install-tool.sh`: installs a pinned release tarball after verifying its
  SHA-256, and holds the pins. The `security` workflow's three tools
  (cargo-deny, cargo-machete, gitleaks) go through it, as does the cargo-deny
  that `build.yml` checks licenses with, so the pins exist once. Dependabot
  cannot see these pins: bump one by hand, and change the URL and its SHA-256
  (`shasum -a 256` of the new asset) together. The other Pipette repositories
  carry the same pins in their copy of this file, so bump them together. It
  runs on Linux CI runners only: it installs into `/usr/local/bin` and needs
  `sha256sum`.

The `android-*` scripts need the Android toolchain (Gradle and a JDK for the two
license scripts; the NDK's `llvm-readelf`, the built libraries and, for an
APK/AAB, the SDK build-tools' `zipalign` for the page-size check), so they run
as steps in the Android jobs, and live directly in `ci/` rather than under
`checks/`. See [docs/licensing.md](../docs/licensing.md).

`pr-body.py` is in `ci/` for a different reason: `lint.sh` runs each check with
no arguments against `git ls-files`, and a PR description is not a tracked file.
There is no body to read in a clone, so it takes one (`--pr`, `--env`, a file
path, or stdin) and runs from its own workflow,
[pr-body.yml](../.github/workflows/pr-body.yml), which unlike `build.yml`
triggers on `edited` so a fixed description clears the check without a push.
The rules, and the reason headings are bold rather than `##`, are in
[CONTRIBUTING.md](../CONTRIBUTING.md#commits-and-pull-requests).

```sh
./ci/pr-body.py --pr 1234
```

`lint.sh` runs every executable file in `ci/checks/` via its own shebang, so a
check can be any language. CI calls the same script (one step
in the `conventions` job), so a green local run matches the PR check.

## Adding a check

Drop a self-contained, executable `ci/checks/<name>` (with a shebang,
`chmod +x`) that:

- inspects tracked files (`git ls-files`),
- exits non-zero on violation with a message stating the rule and how to fix it,
- exits zero otherwise.

`lint.sh` picks it up; no workflow edit is needed. Write a check here only for
a convention that rustfmt, clippy or rustdoc cannot express. The `conventions`
job has no Rust toolchain, and a check that runs cargo would download and
compile the whole dependency tree. Broken doc links are the worked example:
`rust-check` catches them with a `cargo doc` step, because that check also has
to run on each platform.

Checks live here rather than in a per-language job because `conventions` has no
path filter. It runs on every PR. That is what lets `rust-path-refs` catch a
Rust-only change breaking a reference from `android/`, `ios/` or `docs/`, which
those languages' own jobs never see.
