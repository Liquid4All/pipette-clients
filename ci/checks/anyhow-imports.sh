#!/usr/bin/env bash
#
# Only `use anyhow::Context;` may import from anyhow. Crates here have their own
# typed `Result`, so a bare `use anyhow::Result;` would shadow it and hide which
# Result a signature returns; write anyhow items `anyhow::`-qualified instead.
#
# Joins multi-line `use` statements before judging. A combined
# `use { anyhow::Result, ... }` tree is not parsed; rustfmt never emits that form.
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
    sub(/\/\/.*/, "", line)
    if (inuse) {
      buf = buf " " line
    } else if (line ~ /^[[:space:]]*(pub[[:space:]]+)?use[[:space:]]/) {
      buf = line; startln = FNR; inuse = 1
    }
    if (inuse && index(buf, ";") > 0) {
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
