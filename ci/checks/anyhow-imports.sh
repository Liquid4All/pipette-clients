#!/usr/bin/env bash
#
# Enforce the anyhow import convention. This repo pairs anyhow with per-crate
# typed errors (e.g. `pipette_ops::error::Result`), so a bare `use anyhow::Result;`
# silently shadows the crate-local Result and hides which Result a signature
# returns. Rule: import only the `anyhow::Context` trait (needed in scope for
# `.context()`); write Result / bail! / anyhow! / Error `anyhow::`-qualified at
# the use site. The only allowed anyhow import is `use anyhow::Context;`.
#
# Statement-aware: joins multi-line `use anyhow::{ ... };` before judging.
# Limitation: an anyhow item in a combined `use { anyhow::Result, ... }` tree
# isn't parsed — rustfmt never emits that form.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

files=()
while IFS= read -r f; do files+=("$f"); done < <(git ls-files '*.rs')
if [[ ${#files[@]} -eq 0 ]]; then
  echo "anyhow import convention OK (0 Rust files)"
  exit 0
fi

violations=$(awk '
  FNR == 1 { inuse = 0; buf = ""; startln = 0 }
  {
    line = $0
    sub(/\/\/.*/, "", line)                       # drop // line comments
    if (inuse) {
      buf = buf " " line
    } else if (line ~ /^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]/) {
      buf = line; startln = FNR; inuse = 1
    }
    if (inuse && index(buf, ";") > 0) {           # full statement collected
      norm = buf; gsub(/[[:space:]]/, "", norm)
      if (norm ~ /^(pub)?useanyhow(::|;)/ &&
          norm != "useanyhow::Context;" &&
          norm != "pubuseanyhow::Context;") {
        r = buf; gsub(/[[:space:]]+/, " ", r); sub(/^ +/, "", r)
        printf "%s:%d: %s\n", FILENAME, startln, r
      }
      inuse = 0; buf = ""
    }
  }
' "${files[@]}")

if [[ -n "$violations" ]]; then
  echo "error: disallowed anyhow import(s) found." >&2
  echo "Only 'use anyhow::Context;' may be imported from anyhow." >&2
  echo "Write Result / bail! / anyhow! / Error as 'anyhow::'-qualified at the use site." >&2
  echo >&2
  echo "$violations" >&2
  exit 1
fi

echo "anyhow import convention OK (${#files[@]} Rust files checked)"
