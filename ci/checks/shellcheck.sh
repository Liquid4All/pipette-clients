#!/usr/bin/env bash
#
# Runs shellcheck over every tracked shell script. (Mind the wording of comments
# in this file: a line whose first word after `#` is "shellcheck" parses as a
# directive, and an unparseable one is an SC1073 error.)
#
# Scope is "first line is a shell shebang" rather than `*.sh`, because some
# scripts are not named *.sh (scripts/slurm/schedule, *.sbatch). Gradle's
# vendored `gradlew` wrappers are excluded: a Gradle upgrade would undo any fix.
#
# Severity floor is `warning`, so SC2086 (`info`, unquoted expansions) passes.
#
# Runs whatever shellcheck is on PATH, so the `conventions` job needs no install.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! command -v shellcheck >/dev/null 2>&1; then
  echo "error: shellcheck not found on PATH." >&2
  echo "  macOS:  brew install shellcheck" >&2
  echo "  Debian: apt-get install shellcheck" >&2
  echo "  nix:    nix profile install nixpkgs#shellcheck" >&2
  exit 1
fi

files=()
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] || continue
  case "$f" in */gradlew | gradlew) continue ;; esac
  shebang=
  IFS= read -r shebang 2>/dev/null <"$f" || true
  case "${shebang%$'\r'}" in
    '#!'*[\ /]bash* | '#!'*[\ /]sh | '#!'*[\ /]sh\ *) files+=("$f") ;;
  esac
done < <(git ls-files -z)

# Tripwire: a broken enumeration would otherwise report a clean run over zero files.
min_scripts=20
if [[ ${#files[@]} -lt $min_scripts ]]; then
  echo "error: found ${#files[@]} shell scripts, expected at least ${min_scripts}." >&2
  echo "Either the file enumeration in this check broke, or that many scripts" >&2
  echo "really did go away and the floor needs lowering." >&2
  exit 1
fi

status=0
out=$(shellcheck --severity=warning -- "${files[@]}" 2>&1) || status=$?

# Exit 1 is findings; anything higher means shellcheck itself failed to run.
if [[ $status -ne 0 ]]; then
  if [[ $status -eq 1 ]]; then
    echo "error: shellcheck found issues (severity warning and above)." >&2
    echo "Fix them, or annotate a justified exception at the offending line with" >&2
    echo "  # shellcheck disable=SCxxxx  # <why>" >&2
  else
    echo "error: shellcheck exited ${status}, meaning it failed to run rather" >&2
    echo "than finding a violation. Its output follows." >&2
  fi
  echo >&2
  echo "$out" >&2
  exit 1
fi

echo "shellcheck OK (${#files[@]} shell scripts, severity warning)"
