#!/usr/bin/env bash
#
# Stamp this build's identity — the git commit, and on CI the release version —
# into the built Info.plist, so a submitted result can be traced to the source
# and the build that produced it.
#
# Runs as the last build phase — after "Process Info.plist", whose output this
# rewrites. Nothing under source control changes.
#
# Why not the Generated/ pattern that build-llama.sh and gen-mlx-build-info.sh
# use: those emit a *committed* .swift and rewrite it only when the contents
# change, which is cheap because a vendored llama.cpp or an SwiftPM pin moves
# rarely. This value moves with HEAD, so a committed constant would be stale the
# instant it was committed and the next build would dirty the tree again — every
# commit, forever. The built Info.plist is a build artifact, so stamping it costs
# no churn.
#
# Both keys are absent, not empty, when they cannot be resolved (no git, no repo,
# an export from a tarball; a local build with no CI version).
# `Bundle.normalizedInfoString` already treats an absent or placeholder value as
# "unknown", so the app degrades to reporting what it does know rather than
# reporting a lie.
set -euo pipefail

PLIST="${TARGET_BUILD_DIR:?must run as an Xcode build phase}/${INFOPLIST_PATH:?}"

# stamp <key> <value> — replace-or-add, since "Process Info.plist" may have
# already written the key from the source plist.
stamp() {
    /usr/libexec/PlistBuddy -c "Delete :$1" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :$1 string $2" "$PLIST"
    echo "note: $1 = $2"
}

# The version this build publishes as (ci/version.sh), injected by the
# ios-archive CI job. This is what reaches the management server as
# `client_version`, so an archive attached to a release names that release.
#
# A separate key rather than CFBundleVersion: that one is a build *number*, is
# validated by App Store Connect, and on the TestFlight path is already owned by
# ci_scripts/ci_pre_xcodebuild.sh (`agvtool new-version` with $CI_BUILD_NUMBER).
# Overloading it with "2026.08.1-3-ga1b2c3d4ab" would collide with that and put a
# non-conforming value in front of App Review. `PipetteBuildVersion` is ours, so
# it is free-form — as `client_version` upstream already is.
if [[ -n "${PIPETTE_BUILD_VERSION:-}" ]]; then
    stamp PipetteBuildVersion "$PIPETTE_BUILD_VERSION"
else
    echo "note: PIPETTE_BUILD_VERSION unset — PipetteBuildVersion left unset (local build)"
fi

if ! commit=$(git -C "$SRCROOT" rev-parse --short HEAD 2>/dev/null); then
    echo "note: no git metadata — PipetteGitCommit left unset"
    exit 0
fi

# A dirty tree means the binary does not correspond to any commit. Saying so is
# the point: an unmarked hash claims a provenance the build does not have.
if ! git -C "$SRCROOT" diff --quiet HEAD 2>/dev/null; then
    commit="$commit-dirty"
fi

stamp PipetteGitCommit "$commit"
