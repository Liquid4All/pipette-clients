# ci/

Repo-convention checks that run in CI and locally.

Run them all before pushing:

```sh
./ci/lint.sh
```

## What's here

- `lint.sh` + `checks/`: the convention checks, run by the `conventions` job.
  No toolchain required; see "Adding a check" below.
- `android-licenses.py`: the license gate for the Android app's shipping
  dependencies.
- `android-attribution.py`: generates (`--write`), and verifies (`--check`) the
  `ThirdPartyLicenses.json` the Android app bundles.

The two `android-*.py` scripts need Gradle, a JDK and the Android SDK, so they run
as steps in the `android-app` job, and live directly in `ci/` rather than under
`checks/`. See [docs/licensing.md](../docs/licensing.md).

`lint.sh` runs every executable file in `ci/checks/` via its own shebang, so a
check can be any language (bash, python, …). CI calls the same script (one step
in the `conventions` job), so a green local run matches the PR check.

## Adding a check

Drop a self-contained, executable `ci/checks/<name>.{sh,py,…}` (with a shebang,
`chmod +x`) that:

- inspects tracked files (`git ls-files`),
- exits non-zero on violation with a message stating the rule and how to fix it,
- exits zero otherwise.

`lint.sh` picks it up; no workflow edit. Reserve these for conventions that
Rust's own tooling (rustfmt, clippy, rustdoc) can't express; otherwise prefer
that; `conventions` needs no Rust toolchain, and a check that invokes cargo
here would download and compile the whole dependency tree. Doc-link rot is the
worked example: it is a `cargo doc` step in `rust-check`, not a check here,
because it also has to run per-platform.

Checks live here rather than in a per-language job because `conventions` has no
path filter. It runs on every PR. That is what lets `rust-path-refs` catch a
Rust-only change breaking a reference from `android/`, `ios/` or `docs/`, which
those languages' own jobs never see.
