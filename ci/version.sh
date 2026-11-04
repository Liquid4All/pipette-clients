#!/usr/bin/env bash
# The version a build publishes as. Also runnable locally: `ci/version.sh`
# answers "what would this commit publish as?" without pushing.
#
#   release/2026.08.1 three commits in -> 2026.08.1-3-ga1b2c3d4ab
#   three commits past that release    -> 2026.08.1-dev-3-g323eeda3ab
#   no tag in sight                    -> dev-1284-g323eeda3ab
#
# Every form is a train, a number, and the commit — the shape `git describe`
# emits.
#
# On a release branch the train is the branch name and the number is the
# distance from the branch point. Nothing reads the clock and nothing needs a
# pre-existing tag, so a released commit always names itself the same way. The
# one thing that can renumber it is rewriting main: the distance is measured
# from `git merge-base origin/main HEAD`, so a force-push that moves the branch
# point moves the count with it.
#
# Off a release branch the train is the nearest reachable tag with a `-dev`
# marker, so a branch or PR build says which release it descends from and how
# far past it the commit sits. Releases are tagged with their whole version
# string, so the tag's own `-<n>-g<sha>` tail is stripped back to the train
# first. The marker is what keeps this from being a lie: `client_version` maps
# a warehouse row to a downloadable release by string equality, and without it
# a PR build would be indistinguishable from a published release on that train.
#
# Unlike the release form, this one does move under a commit: it reads whatever
# tags the checkout can see, so the same commit built before and after a release
# is cut answers differently, and a checkout with no tags at all falls back to
# `dev` and the commit height. That is the trade for saying what a build
# descends from — nothing consumes these strings except a human reading
# `--version`, and a release still names itself from its branch alone.
set -euo pipefail

ref="${GITHUB_REF:-refs/heads/$(git rev-parse --abbrev-ref HEAD)}"
# The commit to name. On a `pull_request` the checkout is the ephemeral merge
# commit, whose sha appears nowhere in the PR, so CI passes the PR's head sha
# here; everywhere else HEAD is already the commit being built.
sha="${PIPETTE_VERSION_SHA:-HEAD}"

case "$ref" in
  refs/heads/release/*)
    train="${ref#refs/heads/release/}"
    # `release/**` matches nested names. A `/` would nest inside the release
    # tag here, and is outright illegal as an image tag in the sibling repos,
    # so keep trains flat everywhere and fail on the branch name.
    case "$train" in
      */*) echo "release branch must not nest: ${train}" >&2; exit 1 ;;
    esac
    # Distance from where the branch left main. Merging main back in adds only
    # the merge commit, so this never goes backwards.
    base="$(git merge-base origin/main HEAD)"
    echo "${train}-$(git rev-list --count "${base}..HEAD")-g$(git rev-parse --short=10 HEAD)"
    ;;
  *)
    # `--abbrev=0` asks only for the tag name; the distance and the commit are
    # counted below so both arms spell them the same way. It fails when no tag
    # is reachable, which is the untagged fallback and not an error.
    if tag="$(git describe --tags --abbrev=0 "$sha" 2>/dev/null)"; then
      # Shortest-suffix strip, so a train that itself carries hyphens survives:
      # `2026.08.1-hotfix-2-gabcdef1234` -> `2026.08.1-hotfix`.
      train="${tag%-*-g*}-dev"
      steps="$(git rev-list --count "${tag}..${sha}")"
    else
      train="dev"
      steps="$(git rev-list --count "$sha")"
    fi
    echo "${train}-${steps}-g$(git rev-parse --short=10 "$sha")"
    ;;
esac
