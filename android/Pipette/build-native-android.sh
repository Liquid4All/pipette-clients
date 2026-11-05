#!/usr/bin/env bash
#
# Thin entry point for the Android C++ native build (kotlinEngine flavor).
# CMake is the single source of truth. This script resolves the NDK,
# KleidiAI knob, and compiler launchers, then invokes CMake with the
# android-arm64-v8a-cpp preset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ABI="arm64-v8a"
ABI_DIR="$SCRIPT_DIR/app/build/generated/kotlinJniLibs/$ABI"
SYMBOLS_DIR="$SCRIPT_DIR/app/build/generated/kotlinNativeSymbols/$ABI"
API_LEVEL="${ANDROID_API_LEVEL:-31}"

# Resolve the NDK: explicit ANDROID_NDK_HOME / ANDROID_NDK wins, else the newest installed under SDK.
NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
if [[ -z "$NDK_ROOT" ]]; then
  ANDROID_SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
  if [[ ! -d "$ANDROID_SDK_ROOT/ndk" ]]; then
    echo "Android NDK not found. Set ANDROID_NDK_HOME or install the NDK under $ANDROID_SDK_ROOT/ndk." >&2
    exit 1
  fi
  NDK_ROOT="$(find "$ANDROID_SDK_ROOT/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi
export ANDROID_NDK="$NDK_ROOT"

if [[ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
  echo "NDK at '$ANDROID_NDK' is missing build/cmake/android.toolchain.cmake: incomplete or wrong NDK path?" >&2
  exit 1
fi

for tool in cmake ninja; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool not found on PATH (required for the CMake llama.cpp build)." >&2
    exit 1
  fi
done

case "${PIPETTE_ENABLE_KLEIDIAI:-0}" in
  1 | on | ON | true | TRUE) KLEIDIAI="ON" ;;
  0 | off | OFF | false | FALSE | "") KLEIDIAI="OFF" ;;
  *)
    echo "error: PIPETTE_ENABLE_KLEIDIAI must be 1/0 (on/off, true/false); got '${PIPETTE_ENABLE_KLEIDIAI}'" >&2
    exit 1
    ;;
esac

case "${PIPETTE_ENABLE_CRASH_REPORTING:-0}" in
  1 | on | ON | true | TRUE) CRASH_REPORTING="ON" ;;
  0 | off | OFF | false | FALSE) CRASH_REPORTING="OFF" ;;
  *)
    echo "error: PIPETTE_ENABLE_CRASH_REPORTING must be 1/0 (on/off, true/false); got '${PIPETTE_ENABLE_CRASH_REPORTING}'" >&2
    exit 1
    ;;
esac

SCCACHE_ARGS=()
if [[ "${PIPETTE_NO_SCCACHE:-}" != "1" ]] && command -v sccache >/dev/null 2>&1; then
  SCCACHE="$(command -v sccache)"
  SCCACHE_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE" -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE")
  echo "==> sccache enabled"
fi

cd "$REPO_ROOT"

# 1. Configure
cmake --preset android-arm64-v8a-cpp \
  -DANDROID_PLATFORM="android-$API_LEVEL" \
  -DGGML_CPU_KLEIDIAI="$KLEIDIAI" \
  -DPIPETTE_ENABLE_CRASH_REPORTING="$CRASH_REPORTING" \
  -DPIPETTE_JNILIBS_DIR="$ABI_DIR" \
  -DPIPETTE_SYMBOLS_DIR="$SYMBOLS_DIR" \
  ${SCCACHE_ARGS[@]+"${SCCACHE_ARGS[@]}"}

# 2. Build + stage
cmake --build --preset android-arm64-v8a-cpp --target pipette_android_stage

echo "Staged native libs in $ABI_DIR:"
ls -1 "$ABI_DIR"
