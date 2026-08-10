#!/usr/bin/env bash
#
# Verify Android native-library compatibility with 16 KB page-size devices.
#
# The check can run on a generated native-lib directory or on a packaged APK/AAB.
# CI uses both: the generated directory catches our computed native libs close to
# where they are produced, and the packaged artifact covers third-party .so files.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <native-lib-dir-or-apk-or-aab>" >&2
  exit 2
fi

input="$1"
if [ ! -e "$input" ]; then
  echo "error: input not found: $input" >&2
  exit 1
fi

sdk_candidates=()
if [ -n "${ANDROID_NDK:-}" ]; then sdk_candidates+=("$(dirname "$(dirname "$ANDROID_NDK")")"); fi
if [ -n "${ANDROID_NDK_HOME:-}" ]; then sdk_candidates+=("$(dirname "$(dirname "$ANDROID_NDK_HOME")")"); fi
if [ -n "${ANDROID_HOME:-}" ]; then sdk_candidates+=("$ANDROID_HOME"); fi
if [ -n "${ANDROID_SDK_ROOT:-}" ]; then sdk_candidates+=("$ANDROID_SDK_ROOT"); fi
sdk_candidates+=("$HOME/Library/Android/sdk" "$HOME/Android/Sdk")

find_latest_tool() {
  local rel_dir="$1"
  local name="$2"
  local tool=""
  for sdk in "${sdk_candidates[@]}"; do
    [ -d "$sdk/$rel_dir" ] || continue
    tool="$(find "$sdk/$rel_dir" \( -type f -o -type l \) -name "$name" | sort -V | tail -n 1)"
    if [ -n "$tool" ]; then
      printf '%s\n' "$tool"
      return 0
    fi
  done
  return 1
}

readelf="$(find_latest_tool ndk llvm-readelf || true)"
if [ -z "$readelf" ]; then
  echo "error: llvm-readelf not found under Android SDK NDK installations" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

case "$input" in
  *.apk | *.aab)
    archive="$input"
    so_entries="$(unzip -Z1 "$archive" | grep -E '(^|/)lib/[^/]+/[^/]+\.so$' || true)"
    if [ -z "$so_entries" ]; then
      echo "No native libraries found in $archive."
      exit 0
    fi

    printf '%s\n' "$so_entries" | while IFS= read -r entry; do
      unzip -q "$archive" "$entry" -d "$tmp"
    done
    check_root="$tmp"
    ;;
  *)
    if [ ! -d "$input" ]; then
      echo "error: expected a native-lib directory, APK, or AAB: $input" >&2
      exit 1
    fi
    check_root="$input"
    ;;
esac

failures=0
checked=0
while IFS= read -r -d '' so; do
  checked=$((checked + 1))
  rel="${so#"$check_root"/}"
  phdr="$("$readelf" -lW "$so")"

  bad_alignments=""
  while IFS= read -r align; do
    [ -n "$align" ] || continue
    case "$align" in
      0x* | 0X*)
        hex="${align#0x}"
        hex="${hex#0X}"
        value=$((16#$hex))
        ;;
      *) value="$align" ;;
    esac
    if [ "$value" -lt 16384 ]; then
      bad_alignments="${bad_alignments}${bad_alignments:+,}${align}"
    fi
  done < <(printf '%s\n' "$phdr" | awk '/LOAD/ { print $NF }')

  if [ -n "$bad_alignments" ]; then
    echo "error: $rel has LOAD segment alignment below 16 KB: $bad_alignments" >&2
    failures=$((failures + 1))
  fi

  if ! grep -q 'GNU_RELRO' <<< "$phdr"; then
    echo "error: $rel is missing GNU_RELRO" >&2
    failures=$((failures + 1))
  fi
done < <(find "$check_root" -type f -name '*.so' -print0)

if [ "$checked" -eq 0 ]; then
  echo "No native libraries found in $input."
  exit 0
fi

case "$input" in
  *.apk)
    zipalign="$(find_latest_tool build-tools zipalign || true)"
    if [ -z "$zipalign" ]; then
      echo "error: zipalign not found under Android SDK build-tools installations" >&2
      exit 1
    fi
    zipalign_log="$tmp/zipalign.log"
    if ! "$zipalign" -v -c -P 16 4 "$archive" > "$zipalign_log"; then
      cat "$zipalign_log" >&2
      exit 1
    fi
    tail -n 1 "$zipalign_log"
    ;;
  *.aab)
    bundletool_cmd=()
    if command -v bundletool >/dev/null 2>&1; then
      bundletool_cmd=("$(command -v bundletool)")
    else
      candidate="$(find "$HOME/.gradle/caches/modules-2/files-2.1/com.android.tools.build/bundletool" -type f -name '*.jar' 2>/dev/null | sort -V | tail -n 1 || true)"
      if [ -n "$candidate" ]; then
        manifest="$(unzip -p "$candidate" META-INF/MANIFEST.MF 2>/dev/null || true)"
        if grep -q '^Main-Class:' <<< "$manifest"; then
          bundletool_cmd=(java -jar "$candidate")
        fi
      fi
    fi
    if [ "${#bundletool_cmd[@]}" -ne 0 ]; then
      bundletool_log="$tmp/bundletool-config.txt"
      "${bundletool_cmd[@]}" dump config --bundle="$archive" > "$bundletool_log"
      if grep -q 'PAGE_ALIGNMENT_4K' "$bundletool_log"; then
        echo "error: bundletool reports PAGE_ALIGNMENT_4K for $archive" >&2
        cat "$bundletool_log" >&2
        exit 1
      fi
      if grep -q 'PAGE_ALIGNMENT_16K' "$bundletool_log"; then
        echo "bundletool reports PAGE_ALIGNMENT_16K."
      else
        echo "bundletool config has no explicit page-alignment entry; ELF alignment was still verified."
      fi
    else
      echo "runnable bundletool not found; ELF alignment was verified, but bundle page-alignment metadata was not checked."
    fi
    ;;
esac

if [ "$failures" -ne 0 ]; then
  echo "16 KB page-size verification failed for $failures native-library issue(s)." >&2
  exit 1
fi

echo "Verified $checked native libraries in $input for 16 KB LOAD alignment and RELRO."
