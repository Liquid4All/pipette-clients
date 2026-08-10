#!/usr/bin/env bash
#
# Thin entry point for the Android native build. CMake is the single source of
# truth (see repo-root CMakeLists.txt). This script only resolves the NDK +
# KleidiAI knob and invokes CMake.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ABI="arm64-v8a"
ABI_DIR="$SCRIPT_DIR/app/build/generated/rustJniLibs/$ABI"
TARGET="aarch64-linux-android"
# Min API level; ANDROID_API_LEVEL overrides (kept for parity with the prior
# script). Forwarded to the preset's ANDROID_PLATFORM below.
API_LEVEL="${ANDROID_API_LEVEL:-31}"

# Resolve the NDK: an explicit ANDROID_NDK_HOME / ANDROID_NDK wins (accept both,
# since the CMake toolchain consumes ANDROID_NDK), else the newest installed
# under the SDK. Exported as ANDROID_NDK for cmake/android.toolchain.cmake.
NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
if [[ -z "$NDK_ROOT" ]]; then
  ANDROID_SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
  if [[ ! -d "$ANDROID_SDK_ROOT/ndk" ]]; then
    echo "Android NDK not found. Set ANDROID_NDK_HOME or install the NDK under $ANDROID_SDK_ROOT/ndk." >&2
    exit 1
  fi
  # Version-sort (-V): lexicographic sort mis-orders once major-version digit
  # widths differ (e.g. "9.x" after "10.x").
  NDK_ROOT="$(find "$ANDROID_SDK_ROOT/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi
export ANDROID_NDK="$NDK_ROOT"

# Preflight: a missing/corrupt/partially-downloaded NDK (dir exists but the CMake
# toolchain file does not) should fail fast here with an actionable message
# rather than deep inside the CMake/cargo build.
if [[ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]]; then
  echo "NDK at '$ANDROID_NDK' is missing build/cmake/android.toolchain.cmake — incomplete or wrong NDK path?" >&2
  exit 1
fi

for tool in cmake ninja; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool not found on PATH (required for the CMake llama.cpp build)." >&2
    exit 1
  fi
done

if ! rustup target list --installed | grep -qx "$TARGET"; then
  rustup target add "$TARGET"
fi

# KleidiAI is opt-in (prefill-only win, decode-neutral). Parse explicitly so a
# typo can't quietly pick a side; pass the flag ON or OFF (never omit it) since
# the build dir is reused across invocations and a stale value would silently
# produce a mismatched A/B.
case "${PIPETTE_ENABLE_KLEIDIAI:-0}" in
  1 | on | ON | true | TRUE) KLEIDIAI="ON" ;;
  0 | off | OFF | false | FALSE | "") KLEIDIAI="OFF" ;;
  *)
    echo "error: PIPETTE_ENABLE_KLEIDIAI must be 1/0 (on/off, true/false); got '${PIPETTE_ENABLE_KLEIDIAI}'" >&2
    exit 1
    ;;
esac

# Native crash capture (sentry-native, linked into the :benchmark engine) defaults
# OFF: whether crash reporting is wanted is decided one level up by whether a Sentry
# DSN was supplied (see app/build.gradle.kts), and Gradle always passes an explicit
# value here. A bare run of this script is therefore a native-development build that
# does not need sentry-native checked out at all.
# Parsed and passed explicitly for the same reason as KleidiAI above: the CMake build
# dir is reused across invocations, so omitting the flag would leave a stale cached
# value from an earlier build.
case "${PIPETTE_ENABLE_CRASH_REPORTING:-0}" in
  1 | on | ON | true | TRUE) CRASH_REPORTING="ON" ;;
  0 | off | OFF | false | FALSE) CRASH_REPORTING="OFF" ;;
  *)
    echo "error: PIPETTE_ENABLE_CRASH_REPORTING must be 1/0 (on/off, true/false); got '${PIPETTE_ENABLE_CRASH_REPORTING}'" >&2
    exit 1
    ;;
esac

# Route the llama.cpp/ggml C/C++ compiles through sccache when it's on PATH (CI
# sets SCCACHE_GHA_ENABLED for the GHA backend; local builds opt out with
# PIPETTE_NO_SCCACHE=1). Mirrors ios/build-llama.sh. The Rust bridge keeps its
# own target-dir cache, so there's no RUSTC_WRAPPER here.
SCCACHE_ARGS=()
if [[ "${PIPETTE_NO_SCCACHE:-}" != "1" ]] && command -v sccache >/dev/null 2>&1; then
  SCCACHE="$(command -v sccache)"
  SCCACHE_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE" -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE")
  echo "==> sccache enabled"
fi

# `cmake --preset` reads CMakePresets.json from the CWD, and the preset's
# ${sourceDir} resolves relative to it — so run from the repo root (Gradle
# invokes this script with CWD = the app module dir).
cd "$REPO_ROOT"

# 1. Configure (ANDROID_PLATFORM + GGML_CPU_KLEIDIAI + jniLibs dest override the
#    preset defaults).
cmake --preset android-arm64-v8a \
  -DANDROID_PLATFORM="android-$API_LEVEL" \
  -DGGML_CPU_KLEIDIAI="$KLEIDIAI" \
  -DPIPETTE_ENABLE_CRASH_REPORTING="$CRASH_REPORTING" \
  -DPIPETTE_JNILIBS_DIR="$ABI_DIR" \
  ${SCCACHE_ARGS[@]+"${SCCACHE_ARGS[@]}"}  # +"${..}" guard: empty array under set -u errors on bash 3.2

# 2. Build + stage in one invocation. The stage target depends on the cdylib,
#    which pulls the bridge -> llama/mtmd/ggml/ggml-base and (via ggml's
#    add_dependencies) the runtime-loaded CPU variants — so everything needed is
#    built, and cargo runs exactly once.
cmake --build --preset android-arm64-v8a --target pipette_android_stage

echo "Staged native libs in $ABI_DIR:"
ls -1 "$ABI_DIR"
