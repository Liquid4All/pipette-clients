#!/usr/bin/env bash
# Claude Code worktree cleanup, run by the WorktreeRemove hook.
# Mirrors the [cleanup] script in .codex/environments/environment.toml.
# The hook passes the removed worktree's directory as JSON on stdin (".directory").
set -euo pipefail

dir="$(python3 -c 'import sys, json; print(json.load(sys.stdin).get("directory", ""))' 2>/dev/null || true)"
[ -n "$dir" ] && [ -d "$dir" ] || exit 0

cd "$dir"
rm -rf ios/DerivedData ios/build ios/Pipette/build
find ios/Pipette/Pipette/Generated -mindepth 1 ! -name .gitkeep -delete 2>/dev/null || true
