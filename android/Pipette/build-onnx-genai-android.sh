#!/usr/bin/env bash
#
# Build the in-process ONNX Runtime GenAI engine for the Android app (kotlinEngine
# flavor) from the two vendored trees iOS builds from (ios/build-onnx-genai.sh):
# onnxruntime-genai (the generation loop and the C API android/native/onnx_genai_jni.cpp
# calls) and onnxruntime, which genai loads at run time but does not vendor.
#
#   1. Build onnxruntime as a shared library.
#   2. Assemble an `ort_home` from it.
#   3. Build onnxruntime-genai against that `ort_home`.
#   4. Publish both libraries and genai's C header, and check their page alignment.
#
# arm64-v8a with the CPU provider alone, as the iOS engine runs.
#
# Outputs into target/onnx-genai-android-arm64-v8a/out/:
#   lib/libonnxruntime.so, lib/libonnxruntime-genai.so, include/ort_genai_c.h
# which build-native-android.sh hands to CMake as PIPETTE_ONNX_GENAI_DIR.
#
# Usage:
#   ./android/Pipette/build-onnx-genai-android.sh
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,20s/^# \{0,1\}//p' "$0"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENDOR_ORT="$REPO_ROOT/vendor/onnxruntime"
VENDOR_GENAI="$REPO_ROOT/vendor/onnxruntime-genai"
ABI="arm64-v8a"
API_LEVEL="${ANDROID_API_LEVEL:-31}"
BUILD_ROOT="$REPO_ROOT/target/onnx-genai-android-$ABI"
OUT_DIR="$BUILD_ROOT/out"

# Resolve the SDK and NDK as build-native-android.sh does: explicit
# ANDROID_NDK_HOME / ANDROID_NDK wins, else the newest installed under the SDK.
ANDROID_SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
if [[ -z "$NDK_ROOT" ]]; then
  if [[ ! -d "$ANDROID_SDK_ROOT/ndk" ]]; then
    echo "Android NDK not found. Set ANDROID_NDK_HOME or install the NDK under $ANDROID_SDK_ROOT/ndk." >&2
    exit 1
  fi
  NDK_ROOT="$(find "$ANDROID_SDK_ROOT/ndk" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi
if [[ ! -f "$NDK_ROOT/build/cmake/android.toolchain.cmake" ]]; then
  echo "NDK at '$NDK_ROOT' is missing build/cmake/android.toolchain.cmake: incomplete or wrong NDK path?" >&2
  exit 1
fi
OBJDUMP="$(find "$NDK_ROOT/toolchains/llvm/prebuilt" -name llvm-objdump -type f | head -n 1)"

for tool in cmake ninja; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool not found on PATH (required for the ONNX Runtime GenAI build)." >&2
    exit 1
  fi
done

# A submodule's pinned commit: its checkout's when it has one, else the gitlink the
# index records. The second is what a CI runner has when the build cache hits, so the
# stamp below can be read without cloning either tree.
pinned_commit() {
  if [[ -e "$REPO_ROOT/$1/.git" ]]; then
    git -C "$REPO_ROOT/$1" rev-parse HEAD
  else
    git -C "$REPO_ROOT" ls-files -s "$1" | awk '{print $2}'
  fi
}
ORT_COMMIT="$(pinned_commit vendor/onnxruntime)"
GENAI_COMMIT="$(pinned_commit vendor/onnxruntime-genai)"

# Both pins reach a result through the committed stamp iOS generates from the same
# two trees (the app's BuildConfig and the engine both read it), so a pin bumped
# without it would ship a version that names other sources. Checked before anything
# is built, on every run.
BUILD_INFO="$REPO_ROOT/ios/Pipette/Pipette/Generated/OnnxGenai/OnnxGenaiBuildInfo.json"
for tree_commit in "onnxruntime\" $ORT_COMMIT" "onnxruntime_genai\" $GENAI_COMMIT"; do
  tree="${tree_commit%% *}"
  commit="${tree_commit##* }"
  recorded="$(grep -F "\"$tree: " "$BUILD_INFO" | sed -E 's/.*"commit": "([0-9a-f]+)".*/\1/')"
  if [[ -z "$recorded" || "$commit" != "$recorded"* ]]; then
    echo "error: $BUILD_INFO records ${tree%\"} at '${recorded:-nothing}', and it is pinned to $commit." >&2
    echo "       run ./ios/gen-onnx-genai-build-info.sh and commit the result." >&2
    exit 1
  fi
done

# Which inputs the published libraries are from, so an unchanged tree rebuilds
# nothing and a bumped pin rebuilds everything: the NDK, the API level, both
# submodule commits and this script.
STAMP="$OUT_DIR/.stamp"
STAMP_ID="$(basename "$NDK_ROOT") $API_LEVEL $ORT_COMMIT $GENAI_COMMIT $(shasum -a 256 "$0" | cut -d' ' -f1)"
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$STAMP_ID" \
  && -f "$OUT_DIR/lib/libonnxruntime-genai.so" && -f "$OUT_DIR/lib/libonnxruntime.so" ]]; then
  echo "==> ONNX Runtime GenAI already built for $ABI: nothing to do"
  exit 0
fi

# Recursive because onnxruntime has nested submodules (onnx, emsdk,
# libprotobuf-mutator). Scoped per path, as ios/build-onnx-genai.sh scopes it.
require_submodule() {
  local path="$1" marker="$2"
  if [[ ! -f "$REPO_ROOT/$path/$marker" ]] \
    || git -C "$REPO_ROOT" submodule status --recursive "$path" | grep -q '^-'; then
    echo "==> initializing the vendored $path submodule"
    git -C "$REPO_ROOT" submodule update --init --recursive "$path" || true
  fi
  if [[ ! -f "$REPO_ROOT/$path/$marker" ]]; then
    echo "error: vendored sources missing at $path" >&2
    echo "       run: git submodule update --init --recursive $path" >&2
    exit 1
  fi
}
require_submodule vendor/onnxruntime VERSION_NUMBER
require_submodule vendor/onnxruntime-genai VERSION_INFO

# ONNX Runtime's build scripts need a Python with a few packages, kept in a
# build-tree venv. After the stamp check, as the submodules above are, so an
# up-to-date tree clones and installs nothing.
VENV="$BUILD_ROOT/python"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> Creating the build venv"
  command -v uv >/dev/null 2>&1 \
    || { echo "error: uv not found (brew install uv)" >&2; exit 1; }
  uv venv --python 3.12 "$VENV" >/dev/null
fi
VIRTUAL_ENV="$VENV" uv pip install -q -r "$VENDOR_ORT/requirements.txt" requests

# ---------------------------------------------------------------------------
# 1. onnxruntime, as a shared library.
#
# CMAKE_POLICY_VERSION_MINIMUM: CMake 4 dropped compatibility with
# `cmake_minimum_required` below 3.5, which a dependency still declares
# (ios/build-onnx-genai.sh has the same escape hatch).
#
# No page-size flag here or below: both trees add -Wl,-z,max-page-size=16384 for
# Android themselves, and step 4 checks that they did.
# ---------------------------------------------------------------------------
echo "==> Building onnxruntime ($ABI, API $API_LEVEL)"
ORT_BUILD="$BUILD_ROOT/ort"
"$VENV/bin/python" "$VENDOR_ORT/tools/ci_build/build.py" \
  --build_dir "$ORT_BUILD" \
  --config Release \
  --parallel \
  --skip_tests \
  --skip_submodule_sync \
  --no_kleidiai \
  --build_shared_lib \
  --android \
  --android_sdk_path "$ANDROID_SDK_ROOT" \
  --android_ndk_path "$NDK_ROOT" \
  --android_abi "$ABI" \
  --android_api "$API_LEVEL" \
  --cmake_generator Ninja \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5
ORT_SO="$ORT_BUILD/Release/libonnxruntime.so"
[[ -f "$ORT_SO" ]] || { echo "error: onnxruntime library missing at $ORT_SO" >&2; exit 3; }

# ---------------------------------------------------------------------------
# 2. Assemble an ort_home for genai.
#
# For Android, genai reads the layout of the onnxruntime AAR: headers in
# `$ORT_HOME/headers` and the library in `$ORT_HOME/jni/<abi>` (cmake/ortlib.cmake).
# ---------------------------------------------------------------------------
echo "==> Assembling ort_home"
ORT_HOME="$BUILD_ROOT/ort-home"
rm -rf "$ORT_HOME"
mkdir -p "$ORT_HOME/headers" "$ORT_HOME/jni/$ABI"
cp "$VENDOR_ORT"/include/onnxruntime/core/session/*.h "$ORT_HOME/headers/"
cp "$VENDOR_ORT/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h" "$ORT_HOME/headers/"
cp "$ORT_SO" "$ORT_HOME/jni/$ABI/"

# ---------------------------------------------------------------------------
# 3. onnxruntime-genai against that ort_home.
#
# cmake directly rather than genai's build.py, as on iOS: the flags below are the
# ones it would pass, minus the Python wheel and the Java bindings. The app calls
# the C API through its own JNI file, so the timed loop makes no JNI call per token,
# which genai's Java bindings would.
#
# ENABLE_TELEMETRY=OFF is not a default. Left on, the engine links Microsoft's
# 1DS client, which reports to Microsoft; Pipette reports its own measurements.
#
# Unlike iOS, the submodule builds unpatched: its two carried patches are for the
# Objective-C surface and a static library, and Android has neither.
# ---------------------------------------------------------------------------
echo "==> Building onnxruntime-genai ($ABI, API $API_LEVEL)"
GENAI_BUILD="$BUILD_ROOT/genai"
cmake -G Ninja -S "$VENDOR_GENAI" -B "$GENAI_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="android-$API_LEVEL" \
  -DORT_HOME="$ORT_HOME" \
  -DENABLE_TESTS=OFF \
  -DENABLE_PYTHON=OFF \
  -DENABLE_JAVA=OFF \
  -DENABLE_MODEL_BENCHMARK=OFF \
  -DENABLE_TELEMETRY=OFF \
  -DBUILD_WHEEL=OFF \
  -DUSE_CUDA=OFF -DUSE_DML=OFF -DUSE_WINML=OFF -DUSE_GUIDANCE=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$GENAI_BUILD" --config Release --target onnxruntime-genai
GENAI_SO="$GENAI_BUILD/libonnxruntime-genai.so"
[[ -f "$GENAI_SO" ]] || { echo "error: onnxruntime-genai library missing at $GENAI_SO" >&2; exit 4; }

# ---------------------------------------------------------------------------
# 4. Publish, and check the alignment Android 15+ devices with 16 KB pages need.
# ---------------------------------------------------------------------------
echo "==> Publishing to ${OUT_DIR#"$REPO_ROOT/"}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/lib" "$OUT_DIR/include"
cp "$ORT_SO" "$GENAI_SO" "$OUT_DIR/lib/"
cp "$VENDOR_GENAI/src/ort_genai_c.h" "$OUT_DIR/include/"

for so in "$OUT_DIR"/lib/*.so; do
  if "$OBJDUMP" -p "$so" | grep LOAD | grep -v 'align 2\*\*14' | grep -q .; then
    echo "error: $(basename "$so") has a load segment that is not 16 KB aligned" >&2
    exit 5
  fi
done

printf '%s' "$STAMP_ID" > "$STAMP"
echo "==> ONNX Runtime GenAI ready ($ABI): $(du -h "$OUT_DIR/lib/libonnxruntime-genai.so" | cut -f1) engine, $(du -h "$OUT_DIR/lib/libonnxruntime.so" | cut -f1) runtime"
