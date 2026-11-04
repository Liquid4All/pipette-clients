#!/usr/bin/env bash
#
# Build one vendored llama.cpp into the framework the Pipette app loads it from:
# llama.cpp / ggml / mtmd and the ios/native veneers, linked into one dynamic
# library that exports only `pipette_llama_table` (ios/native/pipette_llama.h).
#
# Builds a COPY of the `vendor/llama.cpp` submodule (shared with Android) with
# the ios/patches/ series applied, using llama.cpp's own CMake as upstream
# build-xcframework.sh does, then links the per-target static libs.
#
# Device (iphoneos)           -> Metal ON  (GGML_METAL + GGML_METAL_EMBED_LIBRARY)
# Simulator (iphonesimulator) -> Metal OFF (CPU only)
#
# Output: ios/.build-llama/frameworks/<sdk>/<pin>.framework and its .dSYM.
# ios/embed-llama.sh copies them into the app. The build of vendor/llama.cpp also
# writes the ggml, llama, mtmd and ios/native headers and a module map to
# ios/Pipette/Pipette/Generated/Llama/. The Swift engine compiles against these
# files, so the app can `import llama`. The same build writes
# Generated/PromptSeed/prompt_seed.txt.
#
# Usage:
#   ./ios/build-llama.sh           # auto-detect from $PLATFORM_NAME, else device
#   ./ios/build-llama.sh device
#   ./ios/build-llama.sh sim
#
# Variants: set PIPETTE_LLAMA_VARIANT=<name> to build the vendor/llamacpp-<name>
# submodule into llamacpp-<name>.framework. That submodule holds an iOS-only
# llama.cpp, and vendor/llamacpp-<name>.tag holds its tag. Its patch series is
# ios/patches/llamacpp-<name>/ when that exists, else the main one. The internal
# release carries every variant beside main (build.yml, ios-archive).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,30s/^# \{0,1\}//p' "$0"
    exit 0
fi

# CI runs this script in its own step before xcodebuild, then sets this so the
# in-Xcode build phase is a no-op.
if [[ "${PIPETTE_SKIP_BUILD_LLAMA:-}" == "1" ]]; then
    echo "==> PIPETTE_SKIP_BUILD_LLAMA=1 — skipping (outputs assumed present)"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/ios/Pipette/Pipette/Generated/Llama"
PATCHES_DIR="$REPO_ROOT/ios/patches"   # numbered patch series (NNN-*.patch), applied in order
VARIANT="${PIPETTE_LLAMA_VARIANT:-}"
if [[ -z "$VARIANT" ]]; then
    SUBMODULE="vendor/llama.cpp"        # shared with Android
    PIN="llamacpp"                      # PIPETTE_LLAMA_MAIN_PIN in ios/native/pipette_llama.h
else
    # The name ends up in the framework's file and bundle id names.
    if [[ ! "$VARIANT" =~ ^[a-z0-9][a-z0-9.-]*$ ]]; then
        echo "error: PIPETTE_LLAMA_VARIANT '$VARIANT' must match [a-z0-9][a-z0-9.-]*" >&2
        exit 2
    fi
    SUBMODULE="vendor/llamacpp-$VARIANT"
    PIN="llamacpp-$VARIANT"
    # PIPETTE_LLAMA_PIN_MAX in ios/native/pipette_llama.h: the app refuses a longer pin.
    if (( ${#PIN} > 63 )); then
        echo "error: PIPETTE_LLAMA_VARIANT '$VARIANT' is too long: $PIN must fit in 63 bytes" >&2
        exit 2
    fi
    if [[ -d "$REPO_ROOT/ios/patches/llamacpp-$VARIANT" ]]; then
        PATCHES_DIR="$REPO_ROOT/ios/patches/llamacpp-$VARIANT"
    fi
fi
VENDOR="$REPO_ROOT/$SUBMODULE"
# Becomes the runtime descriptor's repository_url below, so only an https URL.
LLAMA_URL="$(git -C "$REPO_ROOT" config -f .gitmodules "submodule.$SUBMODULE.url" || true)"
if [[ ! "$LLAMA_URL" =~ ^https://[A-Za-z0-9./_-]+$ ]]; then
    echo "error: $SUBMODULE needs an https URL in .gitmodules (got '$LLAMA_URL')" >&2
    exit 2
fi

# Recursive in case llama.cpp grows nested submodules (it has none today), and
# scoped to this one path so an iOS build never pulls another llama.cpp or the
# Android-only kleidiai / sentry-native trees. `--checkout` is needed for a
# variant whose .gitmodules entry says `update = none` (a private repository):
# a plain update skips it.
if [[ ! -f "$VENDOR/ggml/src/ggml.c" ]] \
   || git -C "$REPO_ROOT" submodule status --recursive "$SUBMODULE" | grep -q '^-'; then
    echo "==> initializing the vendored llama.cpp submodule $SUBMODULE"
    git -C "$REPO_ROOT" submodule update --init --checkout --recursive "$SUBMODULE" || true
fi
if [[ ! -f "$VENDOR/ggml/src/ggml.c" ]]; then
    echo "error: vendored llama.cpp sources missing at $VENDOR" >&2
    echo "       run: git submodule update --init --checkout --recursive $SUBMODULE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Platform / SDK selection (device → Metal, simulator → CPU only).
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
    device)
        SDK="iphoneos"
        USE_METAL=1
        METAL_ARGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
        ;;
    sim)
        SDK="iphonesimulator"
        USE_METAL=0
        METAL_ARGS=(-DGGML_METAL=OFF)
        ;;
    *)
        echo "error: unknown mode '$MODE' (expected: device or sim)" >&2
        exit 2
        ;;
esac

# Xcode Run Script phases get a bare system PATH without Homebrew's bin dir, so a
# cmake/ninja that resolves in a terminal is "not found" here. Append (not
# prepend) the Homebrew prefixes so a cmake chosen earlier in PATH still wins.
for brew_bin in /opt/homebrew/bin /usr/local/bin; do
    [[ -d "$brew_bin" && ":$PATH:" != *":$brew_bin:"* ]] && PATH="$PATH:$brew_bin"
done
export PATH

# ninja is as required as cmake (the -G Ninja generator below); without this check
# it surfaces as a cryptic CMAKE_MAKE_PROGRAM configure failure instead.
for tool in cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 \
        || { echo "error: $tool not found (brew install $tool)" >&2; exit 1; }
done
LIBTOOL="$(xcrun --sdk "$SDK" -f libtool)"

BUILD_SRC="$REPO_ROOT/ios/.build-llama/src-$MODE${VARIANT:+-$VARIANT}"
BUILD_DIR="$REPO_ROOT/ios/.build-llama/cmake-$MODE${VARIANT:+-$VARIANT}"

# `--short=9`, not bare `--short`: the auto length grows with the object db (9
# chars on a dev machine, 7 on a fresh CI checkout), which would break CI's
# committed-stamp check and make the submitted runtime_version non-deterministic.
GGML_COMMIT="$(git -C "$VENDOR" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)"

# The upstream tag on this exact commit, when there is one. Not `git describe`,
# which reports the nearest tag and would label a commit past b10216 as b10216.
# A shallow submodule clone has no tag objects, hence the fallback below.
GGML_TAG="$(git -C "$VENDOR" tag --points-at HEAD 2>/dev/null | grep -E '^b[0-9]+$' | head -1 || true)"

# Fallback: the tag recorded beside the pin, as `<tag> <full-sha>`. Bump the
# submodule without updating the file and the sha mismatch rejects the tag
# rather than stamping it onto the wrong commit.
TAG_FILE="$VENDOR.tag"
if [[ -z "$GGML_TAG" && -f "$TAG_FILE" ]]; then
    read -r RECORDED_TAG RECORDED_SHA _ < "$TAG_FILE" || true
    HEAD_SHA="$(git -C "$VENDOR" rev-parse HEAD 2>/dev/null || echo unknown)"
    # Upstream tags are bNNNN; a variant may be a fork with its own scheme.
    TAG_PATTERN='^b[0-9]+$'
    [[ -n "$VARIANT" ]] && TAG_PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*$'
    if [[ ! "${RECORDED_TAG:-}" =~ $TAG_PATTERN ]]; then
        echo "warning: $TAG_FILE: '${RECORDED_TAG:-}' is not a tag ($TAG_PATTERN) — ignoring" >&2
    elif [[ "${RECORDED_SHA:-}" != "$HEAD_SHA" ]]; then
        echo "warning: $TAG_FILE records ${RECORDED_TAG} at ${RECORDED_SHA:-<none>}, but the" >&2
        echo "         submodule is at ${HEAD_SHA} — update it when bumping the pin. Reporting untagged." >&2
    else
        GGML_TAG="$RECORDED_TAG"
    fi
fi

# ---------------------------------------------------------------------------
# Sync a build COPY of the submodule and patch the copy, so the submodule is
# never modified. The series is `NNN-*.patch` in numeric order: 002 takes host
# buffers from vm_allocate so a freed model's pages leave the process; 003 stages
# weight loads through one bounded buffer instead of a heap block per tensor.
# `rsync --delete` re-pristines the copy each run, and `--fuzz=0` makes a patch
# whose context moved on a bump fail loudly instead of landing somewhere else.
# Upstream a patch to retire it from the series.
# ---------------------------------------------------------------------------
echo "==> Syncing build copy: ${VENDOR#"$REPO_ROOT"/} -> ${BUILD_SRC#"$REPO_ROOT"/}"
mkdir -p "$BUILD_SRC"
rsync -a --delete --exclude='.git' "$VENDOR/" "$BUILD_SRC/"

# `patch`, not `git apply`: the copy lives inside the repo's gitignored tree,
# where `git apply` resolves paths against the repo root and silently no-ops.
shopt -s nullglob
for p in "$PATCHES_DIR"/[0-9][0-9][0-9]-*.patch; do
    echo "==> Applying patch: $(basename "$p")"
    ( cd "$BUILD_SRC" && patch -N -p1 --fuzz=0 < "$p" ) \
        || { echo "error: failed to apply $(basename "$p") — does it still match this llama.cpp commit?" >&2; exit 3; }
done
shopt -u nullglob

# Defense-in-depth: without the OOM null-check a Metal OOM faults with
# EXC_BAD_ACCESS instead of returning NULL, so a bump that loses it fails here.
grep -qF 'if (res == NULL)' "$BUILD_SRC/ggml/src/ggml-metal/ggml-metal.cpp" \
    || { echo "error: OOM null-check missing from ggml-metal.cpp in the build copy" >&2; exit 3; }

GGML_INCLUDE="$BUILD_SRC/ggml/include"
LLAMA_INCLUDE="$BUILD_SRC/include"
MTMD_INCLUDE="$BUILD_SRC/tools/mtmd"

# ---------------------------------------------------------------------------
# llama.cpp's own CMake against the build copy (not the repo-root Android CMake).
# ---------------------------------------------------------------------------
SCCACHE_ARGS=()
if [[ "${PIPETTE_NO_SCCACHE:-}" != "1" ]] && command -v sccache >/dev/null 2>&1; then
    SCCACHE="$(command -v sccache)"
    SCCACHE_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER="$SCCACHE" -DCMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE")
    echo "==> sccache enabled"
fi

# Reconfigure cleanly if a previous run cached a different source dir.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$BUILD_SRC" "$BUILD_DIR/CMakeCache.txt"; then
    echo "==> build dir cached a different source — reconfiguring from scratch"
    rm -rf "$BUILD_DIR"
fi

echo "==> Configuring llama.cpp ($MODE / $SDK / metal=$USE_METAL, commit=$GGML_COMMIT)"
cmake -G Ninja -S "$BUILD_SRC" -B "$BUILD_DIR" \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="$SDK" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_ACCELERATE=ON \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_MTMD=ON \
    -DMTMD_VIDEO=OFF \
    -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_SUBPROCESS=OFF \
    "${METAL_ARGS[@]}" \
    ${SCCACHE_ARGS[@]+"${SCCACHE_ARGS[@]}"}  # +"${..}" guard: expanding an empty array under `set -u` errors on bash 3.2 (macOS/Xcode Cloud)

# Named explicitly, though llama + ggml would bring the rest, so a missing
# target fails loudly.
#
# `mtmd` is the multimodal library (the vision cells' encoder + projector).
# Upstream's LLAMA_BUILD_MTMD adds it without the tools tree, for library-only
# packaging, so it links only llama + ggml. Video input is off: it shells out to
# ffmpeg, which iOS does not allow.
#
# `llama-common` supplies two pieces, each behind a veneer below:
# `common/speculative.cpp` (draft-and-verify, run by `vl_spec_decode` with
# speculation on) and `common/chat.cpp` with its Jinja engine (chat templates the
# legacy C call does not know). It builds for a phone only with curl, OpenSSL
# and subprocess off (`subproc.cpp` wants fork/exec).
BUILD_TARGETS=(llama ggml ggml-base ggml-cpu mtmd llama-common)
[[ "$USE_METAL" == "1" ]] && BUILD_TARGETS+=(ggml-metal)
echo "==> Building: ${BUILD_TARGETS[*]}"
cmake --build "$BUILD_DIR" --target "${BUILD_TARGETS[@]}"


# ---------------------------------------------------------------------------
# Our C++ over llama.cpp, compiled against the build COPY so a stale header from
# an older pin cannot be picked up: veneers over the two pieces with no C API,
# common/speculative (ios/native/pipette_vlspec.h) and common/chat
# (ios/native/pipette_chat.h), and the table the app calls this build through
# (ios/native/pipette_llama.h).
# ---------------------------------------------------------------------------
NATIVE_DIR="$REPO_ROOT/ios/native"
NATIVE_SOURCES=(pipette_vlspec pipette_chat pipette_llama_table)
NATIVE_OBJS=()
SDK_PATH="$(xcrun --sdk "$SDK" --show-sdk-path)"
# The min-version flag names the platform, so it must match the SDK: an
# `-mios-version-min` object in a simulator link fails with "built for 'iOS'",
# which reads like a stale artifact. Same 16.4 as the CMake configure.
if [[ "$MODE" == "sim" ]]; then
    NATIVE_MIN_VERSION="-mios-simulator-version-min=16.4"
else
    NATIVE_MIN_VERSION="-mios-version-min=16.4"
fi
NATIVE_DEFS=()
if ! grep -q '^struct common_batch {' "$BUILD_SRC/common/common.h"; then
    NATIVE_DEFS=(-DPIPETTE_VLSPEC_LLAMA_BATCH=1)
    echo "==> No common_batch in this llama.cpp: the speculative veneer takes its llama_batch path"
fi
for source in "${NATIVE_SOURCES[@]}"; do
    echo "==> Compiling $source.cpp"
    "$(xcrun --sdk "$SDK" -f clang++)" \
        -isysroot "$SDK_PATH" \
        -arch arm64 \
        "$NATIVE_MIN_VERSION" \
        -std=gnu++17 -O2 -c \
        ${NATIVE_DEFS[@]+"${NATIVE_DEFS[@]}"} \
        -I"$NATIVE_DIR" \
        -I"$BUILD_SRC/common" \
        -I"$BUILD_SRC/include" \
        -I"$BUILD_SRC/ggml/include" \
        -I"$BUILD_SRC/tools/mtmd" \
        -I"$BUILD_SRC/vendor" \
        -o "$BUILD_DIR/$source.o" "$NATIVE_DIR/$source.cpp"
    NATIVE_OBJS+=("$BUILD_DIR/$source.o")
done

# ---------------------------------------------------------------------------
# One dynamic library from every .a CMake produced, as build-xcframework.sh's
# combine_static_libraries makes upstream's framework. It exports only the
# table: everything else stays private to this build, so a C++ definition it
# shares with another image (the app, another build) is never bound across them.
# ---------------------------------------------------------------------------
FRAMEWORK="$REPO_ROOT/ios/.build-llama/frameworks/$SDK/$PIN.framework"
ARCHIVES=()
while IFS= read -r -d '' a; do ARCHIVES+=("$a"); done \
    < <(find "$BUILD_DIR" -name '*.a' ! -name combined.a -print0)
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
    echo "error: no static libraries produced under $BUILD_DIR" >&2
    exit 4
fi
printf '    + %s\n' "${ARCHIVES[@]##*/}"
rm -f "$BUILD_DIR/combined.a"
"$LIBTOOL" -static -o "$BUILD_DIR/combined.a" "${ARCHIVES[@]}" 2>/dev/null

echo "==> Linking $PIN.framework"
rm -rf "$FRAMEWORK" "$FRAMEWORK.dSYM"
mkdir -p "$FRAMEWORK"
"$(xcrun --sdk "$SDK" -f clang++)" -dynamiclib \
    -isysroot "$SDK_PATH" \
    -arch arm64 \
    "$NATIVE_MIN_VERSION" \
    "${NATIVE_OBJS[@]}" "$BUILD_DIR/combined.a" \
    -framework Foundation -framework Metal -framework Accelerate \
    -install_name "@rpath/$PIN.framework/$PIN" \
    -Wl,-exported_symbol,_pipette_llama_table \
    -Wl,-dead_strip \
    -o "$FRAMEWORK/$PIN"
xcrun dsymutil "$FRAMEWORK/$PIN" -o "$FRAMEWORK.dSYM"
xcrun strip -x "$FRAMEWORK/$PIN"

# Upstream's framework Info.plist (build-xcframework.sh), plus the build's
# identity, which the app and release-manifest.py read without loading the build.
# The URL comes from .gitmodules, not `git remote get-url`: a submodule cloned
# through a mirror reports the mirror, and a variant's pin may be from a fork.
LLAMA_URL="${LLAMA_URL#https://}"
LLAMA_URL="${LLAMA_URL%.git}"
echo "==> Identity: $LLAMA_URL commit=$GGML_COMMIT tag=${GGML_TAG:-none}"
SUPPORTED_PLATFORM="iPhoneOS"
[[ "$MODE" == "sim" ]] && SUPPORTED_PLATFORM="iPhoneSimulator"
# The table header the framework was built against: ios/embed-llama.sh refuses a
# staged framework whose header is not the one the app compiles against.
HEADER_DIGEST="$(shasum -a 256 "$REPO_ROOT/ios/native/pipette_llama.h" | cut -d' ' -f1)"
TAG_ENTRY=""
if [[ -n "$GGML_TAG" ]]; then
    TAG_ENTRY="    <key>PipetteLlamaTag</key>
    <string>$GGML_TAG</string>
"
fi
cat > "$FRAMEWORK/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$PIN</string>
    <key>CFBundleIdentifier</key>
    <string>ai.liquid.liquid-pipette.$PIN</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$PIN</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>16.4</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>$SUPPORTED_PLATFORM</string>
    </array>
    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>
    <key>DTPlatformName</key>
    <string>$SDK</string>
    <key>DTSDKName</key>
    <string>${SDK}16.4</string>
    <key>PipetteLlamaRepositoryUrl</key>
    <string>$LLAMA_URL</string>
${TAG_ENTRY}    <key>PipetteLlamaCommit</key>
    <string>$GGML_COMMIT</string>
    <key>PipetteLlamaHeaderDigest</key>
    <string>$HEADER_DIGEST</string>
</dict>
</plist>
EOF

# The Swift engine compiles against main's headers alone; every build's table
# matches them (the asserts and typed fill in ios/native/pipette_llama*).
if [[ -n "$VARIANT" ]]; then
    echo "==> Done: $FRAMEWORK"
    exit 0
fi

echo "==> Publishing headers + module map to $OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$LLAMA_INCLUDE/llama.h"        "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml.h"          "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-alloc.h"    "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-backend.h"  "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-metal.h"    "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-cpu.h"      "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-opt.h"      "$OUT_DIR/"
cp "$GGML_INCLUDE/gguf.h"          "$OUT_DIR/"
# Both mtmd headers are C-clean (C++ helpers sit behind `#ifdef __cplusplus`),
# and their includes of "ggml.h"/"llama.h" resolve because everything is flat.
cp "$MTMD_INCLUDE/mtmd.h"          "$OUT_DIR/"
cp "$MTMD_INCLUDE/mtmd-helper.h"   "$OUT_DIR/"
for header in pipette_vlspec pipette_chat pipette_llama; do
    cp "$NATIVE_DIR/$header.h"     "$OUT_DIR/"
done

# `[system]`: Native/PipetteLlama.c builds the module with the app's C warnings, which
# llama.cpp's own doc comments would trip.
cat > "$OUT_DIR/module.modulemap" <<'EOF'
module llama [system] {
    header "llama.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-opt.h"
    header "gguf.h"
    header "mtmd.h"
    header "mtmd-helper.h"
    header "pipette_vlspec.h"
    header "pipette_chat.h"
    export *
}
EOF

# llama.cpp's per-token cost depends on content, so iOS must tokenize the same
# prompt seed as the Rust CLIs and Android. Copied into Generated/ so the
# synchronized group bundles it.
PROMPT_SEED_SRC="$REPO_ROOT/crates/pipette-ops/src/prompt_seed.txt"
PROMPT_SEED_DIR="$REPO_ROOT/ios/Pipette/Pipette/Generated/PromptSeed"
if [[ ! -f "$PROMPT_SEED_SRC" ]]; then
    echo "error: shared prompt seed missing at $PROMPT_SEED_SRC" >&2
    exit 5
fi
echo "==> Publishing prompt seed corpus to $PROMPT_SEED_DIR"
mkdir -p "$PROMPT_SEED_DIR"
cp "$PROMPT_SEED_SRC" "$PROMPT_SEED_DIR/prompt_seed.txt"

echo "==> Done: $FRAMEWORK, and the module in $OUT_DIR"
