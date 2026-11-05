#!/usr/bin/env bash
#
# Runs every executable check in ci/checks/ via its own shebang and reports a
# summary. CI runs this same script, so a green local run matches the PR check.
# See ci/README.md to add a check.
set -uo pipefail

# No `set -e` here, so without `|| exit` a failed cd would lint the wrong tree.
root=$(git rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1

checks=()
for f in ci/checks/*; do
  [[ -f "$f" && -x "$f" ]] && checks+=("$f")
done
if [[ ${#checks[@]} -eq 0 ]]; then
  echo "no checks found in ci/checks/"
  exit 0
fi

# The Python checks evaluate PEP 604 unions (`X | None`) at import time, so on
# python3 < 3.10 each one would print as `FAIL <name>`, indistinguishable from a
# real violation. 3.10 is the language floor; CI runs 3.12.
if printf '%s\n' "${checks[@]}" | grep -q '\.py$'; then
  if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    have=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null) || have="missing"
    echo "ci/lint.sh: python3 is ${have}, but the Python checks in ci/checks/ need 3.10+." >&2
    echo "  Running them on an older interpreter reports each as a failure with a" >&2
    echo "  'TypeError: unsupported operand type(s) for |', which looks like a" >&2
    echo "  convention violation rather than the environment problem it is." >&2
    echo "  Put a newer python3 first on PATH (pyenv, uv, or homebrew). CI runs 3.12." >&2
    exit 1
  fi
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
