#!/usr/bin/env bash
#
# Build the in-process ONNX Runtime GenAI engine for the iOS app from two
# vendored trees: onnxruntime-genai (the generation loop and the Objective-C
# surface the app calls) and onnxruntime, which genai links against but does
# not vendor.
#
#   1. Build onnxruntime.
#   2. Assemble an `ort_home` from it.
#   3. Copy the onnxruntime-genai tree and patch the copy.
#   4. Build onnxruntime-genai against that `ort_home`.
#   5. Compile its Objective-C surface.
#   6. Combine and publish the archives and headers.
#
# Device and simulator slices are both arm64 with CoreML + XNNPACK.
#
# Outputs into ios/Pipette/Pipette/Generated/OnnxGenai/:
#   libonnxruntime.a, libonnxruntime-genai.a (with the Objective-C surface
#   compiled in), the genai + ORT headers and ort_genai_objc.h, and the
#   OnnxGenaiBuildInfo files from gen-onnx-genai-build-info.sh.
#
# Usage:
#   ./ios/build-onnx-genai.sh           # auto-detect from $PLATFORM_NAME, else device
#   ./ios/build-onnx-genai.sh device
#   ./ios/build-onnx-genai.sh sim

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25s/^# \{0,1\}//p' "$0"
    exit 0
fi

# CI runs this script in its own step before xcodebuild, then sets this so the
# in-Xcode build phase is a no-op.
if [[ "${PIPETTE_SKIP_BUILD_ONNX_GENAI:-}" == "1" ]]; then
    echo "==> PIPETTE_SKIP_BUILD_ONNX_GENAI=1 — skipping (outputs assumed present)"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_ORT="$REPO_ROOT/vendor/onnxruntime"
VENDOR_GENAI="$REPO_ROOT/vendor/onnxruntime-genai"
OUT_DIR="$REPO_ROOT/ios/Pipette/Pipette/Generated/OnnxGenai"
PATCHES_DIR="$REPO_ROOT/ios/patches/onnx-genai"

# Recursive because onnxruntime has nested submodules (onnx, emsdk,
# libprotobuf-mutator). Scoped per path so an iOS build never pulls the
# Android-only kleidiai / sentry-native trees.
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

# ---------------------------------------------------------------------------
# Platform / SDK selection. Both slices build the same providers: unlike
# llama.cpp there is no Metal split.
# ---------------------------------------------------------------------------
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    case "${PLATFORM_NAME:-}" in
        iphoneos)        MODE="device" ;;
        iphonesimulator) MODE="sim" ;;
        *)               MODE="device" ;;
    esac
fi
case "$MODE" in
    device) SDK="iphoneos" ;;
    sim)    SDK="iphonesimulator" ;;
    *)      echo "error: unknown mode '$MODE' (expected: device or sim)" >&2; exit 2 ;;
esac

# As in build-llama.sh: Xcode Run Script phases get a bare system PATH. Append
# rather than prepend so a cmake chosen earlier in PATH still wins.
for brew_bin in /opt/homebrew/bin /usr/local/bin; do
    [[ -d "$brew_bin" && ":$PATH:" != *":$brew_bin:"* ]] && PATH="$PATH:$brew_bin"
done
export PATH
command -v cmake >/dev/null 2>&1 || { echo "error: cmake not found (brew install cmake)" >&2; exit 1; }

LIBTOOL="$(xcrun --sdk "$SDK" -f libtool)"
SDK_PATH="$(xcrun --sdk "$SDK" --show-sdk-path)"
# The app's deployment floor: Xcode supplies it to the build phase, and the
# default is the project's value for standalone runs.
DEPLOY_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-26.0}"

BUILD_ROOT="$REPO_ROOT/ios/.build-onnx-genai"
ORT_BUILD="$BUILD_ROOT/ort-$MODE"
ORT_HOME="$BUILD_ROOT/ort-home-$MODE"
GENAI_SRC="$BUILD_ROOT/src-genai-$MODE"
GENAI_BUILD="$BUILD_ROOT/genai-$MODE"

# Which slice and inputs the published archives are from. Both slices write the
# same filenames, so without it a device build followed by a simulator one links
# the wrong architecture, and rebuilding every time costs minutes. Keyed on the
# SDK, both submodule commits, the genai patches and this script: the same
# inputs as CI's cache key.
SLICE_STAMP="$OUT_DIR/.slice"
SLICE_INPUTS="$(cat "$PATCHES_DIR"/[0-9][0-9][0-9]-*.patch "$0" | shasum -a 256 | cut -d' ' -f1)"
SLICE_ID="$SDK $(git -C "$VENDOR_ORT" rev-parse HEAD 2>/dev/null) $(git -C "$VENDOR_GENAI" rev-parse HEAD 2>/dev/null) $SLICE_INPUTS"
if [[ -f "$SLICE_STAMP" && "$(cat "$SLICE_STAMP")" == "$SLICE_ID" \
      && -f "$OUT_DIR/libonnxruntime-genai.a" && -f "$OUT_DIR/libonnxruntime.a" ]]; then
    echo "==> ONNX Runtime GenAI already built for $MODE — nothing to do"
    exit 0
fi

# ONNX Runtime's build scripts need a Python with a few packages, kept in a
# build-tree venv. After the slice check, so an up-to-date slice (a CI cache
# hit) installs nothing.
VENV="$BUILD_ROOT/python"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "==> Creating the build venv"
    command -v uv >/dev/null 2>&1 \
        || { echo "error: uv not found (brew install uv)" >&2; exit 1; }
    uv venv --python 3.12 "$VENV" >/dev/null
fi
VIRTUAL_ENV="$VENV" uv pip install -q -r "$VENDOR_ORT/requirements.txt" requests
PYTHON="$VENV/bin/python"

# CMake 4 dropped compatibility with `cmake_minimum_required` below 3.5, which
# psimd (an XNNPACK dependency) still declares. Both trees need the escape hatch
# CMake itself names; without it configure fails before compiling anything.
CMAKE_POLICY_ARG=-DCMAKE_POLICY_VERSION_MINIMUM=3.5

# ---------------------------------------------------------------------------
# 1. onnxruntime, as a static framework for this slice.
# ---------------------------------------------------------------------------
echo "==> Building onnxruntime ($MODE / $SDK)"
"$PYTHON" "$VENDOR_ORT/tools/ci_build/build.py" \
    --build_dir "$ORT_BUILD" \
    --config Release \
    --parallel \
    --skip_tests \
    --no_kleidiai \
    --build_apple_framework \
    --use_coreml \
    --use_xnnpack \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    --ios \
    --use_xcode \
    --apple_sysroot "$SDK" \
    --osx_arch arm64 \
    --apple_deploy_target "$DEPLOY_TARGET"

ORT_FRAMEWORK="$ORT_BUILD/Release/Release-$SDK/static_framework/onnxruntime.framework"
[[ -d "$ORT_FRAMEWORK" ]] \
    || { echo "error: onnxruntime framework missing at $ORT_FRAMEWORK" >&2; exit 3; }

# ---------------------------------------------------------------------------
# 2. Assemble an ort_home for genai.
#
# For iOS, genai reads headers from `$ORT_HOME/Headers` and requires an
# xcframework at `$ORT_HOME/onnxruntime.xcframework` (cmake/ortlib.cmake). One
# slice per ort_home, so the device/simulator split runs all the way down.
# ---------------------------------------------------------------------------
echo "==> Assembling ort_home for $MODE"
rm -rf "$ORT_HOME"
mkdir -p "$ORT_HOME"
cp -R "$ORT_FRAMEWORK/Headers" "$ORT_HOME/Headers"
xcodebuild -create-xcframework -framework "$ORT_FRAMEWORK" \
    -output "$ORT_HOME/onnxruntime.xcframework" >/dev/null

# ---------------------------------------------------------------------------
# 3. Sync a build COPY of the genai submodule and apply ios/patches/onnx-genai/
# to it (each patch says why it exists), so the submodule is never modified.
# `rsync --delete` re-pristines the copy, so every patch applies fresh and one
# that no longer applies fails loudly.
# ---------------------------------------------------------------------------
echo "==> Syncing build copy: vendor/onnxruntime-genai -> ${GENAI_SRC#"$REPO_ROOT/"}"
mkdir -p "$GENAI_SRC"
rsync -a --delete --exclude='.git' "$VENDOR_GENAI/" "$GENAI_SRC/"

# `patch`, not `git apply`: the copy lives inside the repo's gitignored tree,
# where `git apply` resolves paths against the repo root and silently no-ops.
shopt -s nullglob
for p in "$PATCHES_DIR"/[0-9][0-9][0-9]-*.patch; do
    echo "==> Applying patch: $(basename "$p")"
    ( cd "$GENAI_SRC" && patch -p1 --forward < "$p" ) \
        || { echo "error: failed to apply $(basename "$p") — does it still match this onnxruntime-genai tag?" >&2; exit 3; }
done
shopt -u nullglob

# Defense in depth: confirm the two patches that decide what we ship actually
# landed. A patch that applied with fuzz to the wrong place would otherwise
# surface as a link error hundreds of lines later.
grep -q "add_library(onnxruntime-genai STATIC" "$GENAI_SRC/CMakeLists.txt" \
    || { echo "error: the static-library patch did not take" >&2; exit 3; }
grep -q "processImagesWithPrompts:" "$GENAI_SRC/src/objectivec/include/ort_genai_objc.h" \
    || { echo "error: the Objective-C compile-fix patch did not take" >&2; exit 3; }

# ---------------------------------------------------------------------------
# 4. onnxruntime-genai against that ort_home.
#
# cmake directly rather than genai's build.py: that wrapper defaults to a
# generator its own iOS toolchain contradicts, and drives a Python-wheel target
# that fails. The flags below are the ones it would pass, minus the wheel.
#
# ENABLE_TELEMETRY=OFF is not a default. Left on, the engine links Microsoft's
# 1DS client, which reports to Microsoft and drags in UIKit and Network.
#
# CMAKE_SKIP_INSTALL_RULES=ON: nothing installs from this tree, and a static
# target propagates a link interface that genai's export set does not cover.
# ---------------------------------------------------------------------------
echo "==> Configuring onnxruntime-genai ($MODE / $SDK)"
rm -rf "$GENAI_BUILD"
cmake -G Xcode -S "$GENAI_SRC" -B "$GENAI_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$SDK" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOY_TARGET" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DPLATFORM_NAME="$SDK" \
    -DORT_HOME="$ORT_HOME" \
    -DBUILD_APPLE_FRAMEWORK=ON \
    -DENABLE_TESTS=OFF \
    -DENABLE_PYTHON=OFF \
    -DENABLE_JAVA=OFF \
    -DENABLE_MODEL_BENCHMARK=OFF \
    -DENABLE_TELEMETRY=OFF \
    -DBUILD_WHEEL=OFF \
    -DUSE_CUDA=OFF -DUSE_DML=OFF -DUSE_WINML=OFF -DUSE_GUIDANCE=OFF \
    -DCMAKE_SKIP_INSTALL_RULES=ON \
    "$CMAKE_POLICY_ARG"

echo "==> Building onnxruntime-genai"
cmake --build "$GENAI_BUILD" --config Release --target onnxruntime-genai

# No `Release/` level: that comes from build.py, which the genai build bypasses.
GENAI_LIB="$GENAI_BUILD/Release-$SDK/static_framework/onnxruntime-genai.framework"
[[ -f "$GENAI_LIB/onnxruntime-genai" ]] \
    || { echo "error: onnxruntime-genai archive missing at $GENAI_LIB" >&2; exit 4; }

# ---------------------------------------------------------------------------
# 5. Compile the Objective-C surface.
#
# `src/objectivec` is not a CMake target: upstream ships it as a CocoaPods
# source pod, compiled by whoever consumes it. Compiled here rather than in the
# app target to keep third-party sources out of the app's synchronized group.
# ---------------------------------------------------------------------------
echo "==> Compiling the Objective-C surface"
OBJC_SRC="$GENAI_SRC/src/objectivec"
OBJC_OBJ="$BUILD_ROOT/objc-$MODE"
rm -rf "$OBJC_OBJ"
mkdir -p "$OBJC_OBJ"
for mm in "$OBJC_SRC"/*.mm; do
    xcrun clang++ -x objective-c++ -fobjc-arc -std=c++20 -O2 \
        -target "arm64-apple-ios$DEPLOY_TARGET$([[ "$MODE" == sim ]] && echo -simulator)" \
        -isysroot "$SDK_PATH" \
        -I"$OBJC_SRC" -I"$OBJC_SRC/include" \
        -I"$GENAI_LIB/Headers" -I"$ORT_HOME/Headers" \
        -c "$mm" -o "$OBJC_OBJ/$(basename "${mm%.mm}").o"
done

# ---------------------------------------------------------------------------
# 6. Combine into one archive per tree and publish.
#
# genai's own archive holds only its objects; its dependencies (the custom-op
# libraries) are separate archives. Collect every .a, as build-llama.sh does;
# libtool dedups object files.
# ---------------------------------------------------------------------------
echo "==> Combining static libs"
mkdir -p "$OUT_DIR"
ARCHIVES=("$GENAI_LIB/onnxruntime-genai")
while IFS= read -r -d '' a; do ARCHIVES+=("$a"); done \
    < <(find "$GENAI_BUILD" -name '*.a' -print0)
ARCHIVES+=("$OBJC_OBJ"/*.o)
printf '    + %s\n' "${ARCHIVES[@]##*/}"
rm -f "$OUT_DIR/libonnxruntime-genai.a" "$OUT_DIR/libonnxruntime.a"
"$LIBTOOL" -static -o "$OUT_DIR/libonnxruntime-genai.a" "${ARCHIVES[@]}" 2>/dev/null
cp "$ORT_FRAMEWORK/onnxruntime" "$OUT_DIR/libonnxruntime.a"

echo "==> Publishing headers to ${OUT_DIR#"$REPO_ROOT/"}"
cp "$OBJC_SRC/include/ort_genai_objc.h" "$OUT_DIR/"
cp "$GENAI_LIB/Headers"/*.h "$OUT_DIR/"
cp "$ORT_HOME/Headers"/*.h "$OUT_DIR/"

printf '%s' "$SLICE_ID" > "$SLICE_STAMP"

"$REPO_ROOT/ios/gen-onnx-genai-build-info.sh"

echo "==> ONNX Runtime GenAI ready ($MODE): $(du -h "$OUT_DIR/libonnxruntime-genai.a" | cut -f1) engine, $(du -h "$OUT_DIR/libonnxruntime.a" | cut -f1) runtime"
