#!/usr/bin/env bash
#
# Stamp the app's git commit into the built Info.plist, so a submitted result can
# be traced to the source that produced it.
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
# The key is absent, not empty, when it cannot be resolved (no git, no repo, an
# export from a tarball). `Bundle.normalizedInfoString` already treats an absent
# or placeholder value as "unknown", so the app degrades to reporting the version
# alone rather than reporting a lie.
set -euo pipefail

KEY=PipetteGitCommit
PLIST="${TARGET_BUILD_DIR:?must run as an Xcode build phase}/${INFOPLIST_PATH:?}"

if ! commit=$(git -C "$SRCROOT" rev-parse --short HEAD 2>/dev/null); then
    echo "note: no git metadata — $KEY left unset"
    exit 0
fi

# A dirty tree means the binary does not correspond to any commit. Saying so is
# the point: an unmarked hash claims a provenance the build does not have.
if ! git -C "$SRCROOT" diff --quiet HEAD 2>/dev/null; then
    commit="$commit-dirty"
fi

/usr/libexec/PlistBuddy -c "Delete :$KEY" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :$KEY string $commit" "$PLIST"
echo "note: $KEY = $commit"
