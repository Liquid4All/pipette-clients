#!/usr/bin/env bash
#
# Stamp this build's identity (the git commit, and on CI the release version)
# into the built Info.plist, so a submitted result can be traced to its source.
#
# Runs as the last build phase, after "Process Info.plist", whose output this
# rewrites. Not a committed Generated/ .swift like the other build-info scripts:
# this value moves with every commit, so a committed copy would be stale once
# committed and dirty the tree on the next build.
#
# A key that cannot be resolved (no git metadata, a local build with no CI
# version) is left absent, not empty: `Bundle.normalizedInfoString` reads an
# absent or placeholder value as unknown.
set -euo pipefail

PLIST="${TARGET_BUILD_DIR:?must run as an Xcode build phase}/${INFOPLIST_PATH:?}"

# Replace-or-add: "Process Info.plist" may have already written the key.
stamp() {
    /usr/libexec/PlistBuddy -c "Delete :$1" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :$1 string $2" "$PLIST"
    echo "note: $1 = $2"
}

# The release version from build.yml's `release-metadata` job, set by CI. The app
# reports it to the management server as `client_version`.
#
# Not CFBundleVersion: that is a build number App Store Connect validates, and on
# the TestFlight path ci_scripts/ci_pre_xcodebuild.sh already sets it
# (`agvtool new-version` with $CI_BUILD_NUMBER). A value like
# "2026.08.11-640d96e8" would not conform.
if [[ -n "${PIPETTE_BUILD_VERSION:-}" ]]; then
    stamp PipetteBuildVersion "$PIPETTE_BUILD_VERSION"
else
    echo "note: PIPETTE_BUILD_VERSION unset — PipetteBuildVersion left unset (local build)"
fi

if ! commit=$(git -C "$SRCROOT" rev-parse --short HEAD 2>/dev/null); then
    echo "note: no git metadata — PipetteGitCommit left unset"
    exit 0
fi

# A dirty tree means the binary does not correspond to any commit, and an
# unmarked hash would claim a provenance the build does not have.
if ! git -C "$SRCROOT" diff --quiet HEAD 2>/dev/null; then
    commit="$commit-dirty"
fi

stamp PipetteGitCommit "$commit"
