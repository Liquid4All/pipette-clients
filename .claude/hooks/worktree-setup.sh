#!/usr/bin/env bash
# Claude Code worktree setup.
# Mirrors the [setup] script in .codex/environments/environment.toml.
# Fires from the SessionStart hook; only runs inside a linked git worktree.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

# Skip in the primary checkout — only provision linked worktrees.
git_dir="$(cd "$(git rev-parse --git-dir)" && pwd)"
common_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
[[ "$git_dir" == "$common_dir" ]] && exit 0

# Run once per worktree (marker lives in the per-worktree git dir, never committed).
marker="$git_dir/.claude-worktree-setup-done"
[[ -f "$marker" ]] && exit 0

git submodule update --init --recursive vendor/llama.cpp
# Best-effort: a pre-build failure shouldn't fail worktree setup — the real build
# runs under xcodebuild; this only warms editor `import llama` resolution.
./ios/build-llama.sh sim || echo "worktree-setup: llama pre-build failed — editor import may lag"

touch "$marker"
