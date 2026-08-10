#!/usr/bin/env bash
#
# Runs every executable check in ci/checks/ and reports a summary. Each check is
# run via its own shebang, so a check can be any language (bash, python, …) — no
# per-language runner here. The same entry point runs locally (`./ci/lint.sh`)
# and in CI, so a green local run matches the PR check. See ci/README.md to add
# one.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

checks=()
for f in ci/checks/*; do
  [[ -f "$f" && -x "$f" ]] && checks+=("$f")
done
if [[ ${#checks[@]} -eq 0 ]]; then
  echo "no checks found in ci/checks/"
  exit 0
fi

failed=0
for check in "${checks[@]}"; do
  name=$(basename "$check")
  name=${name%.*}
  if out=$("$check" 2>&1); then
    printf '  ok   %s\n' "$name"
  else
    printf '  FAIL %s\n' "$name"
    printf '%s\n' "$out" | sed 's/^/       /'
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  echo
  echo "convention checks failed — see output above" >&2
  exit 1
fi
echo "all convention checks passed"
