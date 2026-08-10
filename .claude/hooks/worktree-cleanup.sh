#!/usr/bin/env bash
# Claude Code worktree cleanup.
# Mirrors the [cleanup] script in .codex/environments/environment.toml.
# Fires from the WorktreeRemove hook, which passes the worktree directory being
# removed as JSON on stdin (the ".directory" field). Runs only on actual
# worktree removal — not on /clear or normal session exit.
set -euo pipefail

dir="$(python3 -c 'import sys, json; print(json.load(sys.stdin).get("directory", ""))' 2>/dev/null || true)"
[ -n "$dir" ] && [ -d "$dir" ] || exit 0

cd "$dir"
rm -rf ios/DerivedData ios/build ios/Pipette/build
find ios/Pipette/Pipette/Generated -mindepth 1 ! -name .gitkeep -delete 2>/dev/null || true
