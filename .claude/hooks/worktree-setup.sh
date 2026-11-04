#!/usr/bin/env bash
# Claude Code worktree setup, run by the SessionStart hook.
# Provisions linked git worktrees only, once each.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

git_dir="$(cd "$(git rev-parse --git-dir)" && pwd)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
[[ "$git_dir" == "$common_dir" ]] && exit 0

# The marker lives in the per-worktree git dir, so it is never committed.
marker="$git_dir/.claude-worktree-setup-done"
[[ -f "$marker" ]] && exit 0

git submodule update --init --recursive vendor/llama.cpp
# Best-effort: the real build runs under xcodebuild; this only warms editor
# `import llama` resolution.
./ios/build-llama.sh sim || echo "worktree-setup: llama pre-build failed — editor import may lag"

touch "$marker"
