#!/usr/bin/env bash
#
# Build llama.cpp / ggml from the vendored sources into a single static library
# `libllama.a` plus a Clang module map, so the Pipette app can `import llama` and
# link the C symbols directly from OUR sources — no prebuilt xcframework, no Rust.
#
# Source of truth: the shared `vendor/llama.cpp` submodule (same one Android
# builds). To keep that submodule pristine, this script syncs a build COPY of it,
# applies the ggml-metal OOM null-check patch (ios/patches/) to the COPY, and
# drives llama.cpp's OWN CMake against the copy — exactly like upstream
# build-xcframework.sh. CMake owns the source list, the per-target compile flags,
# and the GGML_METAL_EMBED_LIBRARY metallib embed (nothing is hand-transcribed).
# The script then combines the per-target static libs into one libllama.a,
# publishes the headers + module map, and emits LlamaCppBuildInfo.swift.
#
# Device (iphoneos)        -> Metal ON  (GGML_METAL + GGML_METAL_EMBED_LIBRARY)
# Simulator (iphonesimulator) -> Metal OFF (CPU only)
#
# Outputs into ios/Pipette/Pipette/Generated/Llama/:
#   libllama.a, module.modulemap, the umbrella + ggml/llama headers, LlamaCppBuildInfo.swift.
#
# Usage:
#   ./ios/build-llama.sh           # auto-detect from $PLATFORM_NAME, else device
#   ./ios/build-llama.sh device
#   ./ios/build-llama.sh sim

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,30s/^# \{0,1\}//p' "$0"
    exit 0
fi

# Allow a CI fast-path: a pre-run (ci_pre_xcodebuild.sh) populates Generated/
# and the inner xcodebuild invocation is a no-op.
if [[ "${PIPETTE_SKIP_BUILD_LLAMA:-}" == "1" ]]; then
    echo "==> PIPETTE_SKIP_BUILD_LLAMA=1 — skipping (outputs assumed present)"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$REPO_ROOT/vendor/llama.cpp"        # the shared submodule (kept pristine)
OUT_DIR="$REPO_ROOT/ios/Pipette/Pipette/Generated/Llama"
PATCHES_DIR="$REPO_ROOT/ios/patches"   # numbered patch series (NNN-*.patch), applied in order

# Ensure the submodule is checked out (fresh clones / CI). We init recursively so
# any nested submodules llama.cpp may grow are picked up automatically — today it
# has none, so --recursive is a no-op, but it can't silently miss one later. The
# guard is recursive too (status --recursive prefixes uninitialized modules,
# including nested ones, with '-'), scoped to vendor/llama.cpp so an iOS build
# never pulls the Android-only kleidiai / sentry-native trees.
if [[ ! -f "$VENDOR/ggml/src/ggml.c" ]] \
   || git -C "$REPO_ROOT" submodule status --recursive vendor/llama.cpp | grep -q '^-'; then
    echo "==> initializing the vendored llama.cpp submodule"
    git -C "$REPO_ROOT" submodule update --init --recursive vendor/llama.cpp || true
fi
if [[ ! -f "$VENDOR/ggml/src/ggml.c" ]]; then
    echo "error: vendored llama.cpp sources missing at $VENDOR" >&2
    echo "       run: git submodule update --init --recursive vendor/llama.cpp" >&2
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

# Xcode Run Script phases run with a bare system PATH (/etc/paths + Xcode's own
# toolchain dirs). Homebrew's bin dir is NOT on it unless the user added it to
# /etc/paths, so a cmake/ninja that resolves fine in the terminal — where
# `brew shellenv` prepends it from the shell profile — is invisible here, and the
# phase fails with "cmake not found" on a machine that plainly has cmake. Append
# the standard Homebrew prefixes (Apple Silicon, then Intel) ourselves; appending
# rather than prepending keeps a deliberately-chosen cmake earlier in PATH winning.
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

BUILD_SRC="$REPO_ROOT/ios/.build-llama/src-$MODE"
BUILD_DIR="$REPO_ROOT/ios/.build-llama/cmake-$MODE"

# Commit for the in-app version readout (formerly the Rust `llamaCppCommit()`).
# Pin the abbreviation length (`--short=9`, not bare `--short`): the auto length
# is environment-dependent — it grows with the object db, so a bare short hash is
# 9 chars on a dev machine but 7 on a fresh CI checkout for the same commit. That
# mismatch breaks the committed-stamp freshness gate and makes the submitted
# runtime_version non-deterministic. A fixed 9 stays short, reproduces
# everywhere, and is unambiguous for llama.cpp.
GGML_COMMIT="$(git -C "$VENDOR" rev-parse --short=9 HEAD 2>/dev/null || echo unknown)"

# The upstream tag on the same pin, when there is one. A tag is remote metadata,
# not a property of the commit, and the submodule is a shallow clone with no tag
# objects — so this resolves from whichever source is available, never from the
# network, and never from `git describe` (which reports the *nearest* tag and
# would label a commit past b10216 as b10216).
GGML_TAG="$(git -C "$VENDOR" tag --points-at HEAD 2>/dev/null | grep -E '^b[0-9]+$' | head -1 || true)"

# Fallback: the tag recorded beside the pin, as `<tag> <full-sha>`. The sha makes
# staleness detectable offline — bump the submodule without updating this file
# and the recorded tag is rejected rather than stamped onto the wrong commit.
TAG_FILE="$REPO_ROOT/vendor/llama.cpp.tag"
if [[ -z "$GGML_TAG" && -f "$TAG_FILE" ]]; then
    read -r RECORDED_TAG RECORDED_SHA _ < "$TAG_FILE" || true
    HEAD_SHA="$(git -C "$VENDOR" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ ! "${RECORDED_TAG:-}" =~ ^b[0-9]+$ ]]; then
        echo "warning: $TAG_FILE: '${RECORDED_TAG:-}' is not a tag (bNNNN) — ignoring" >&2
    elif [[ "${RECORDED_SHA:-}" != "$HEAD_SHA" ]]; then
        echo "warning: $TAG_FILE records ${RECORDED_TAG} at ${RECORDED_SHA:-<none>}, but the" >&2
        echo "         submodule is at ${HEAD_SHA} — update it when bumping the pin. Reporting untagged." >&2
    else
        GGML_TAG="$RECORDED_TAG"
    fi
fi

# ---------------------------------------------------------------------------
# Sync a build COPY of the pristine submodule, then apply our carried patch
# series to the COPY — the submodule itself is never modified, and a clean
# checkout builds reproducibly. CMake compiles the copy in place.
#
# The series lives in ios/patches/ as numbered `NNN-*.patch` files, applied in
# numeric order (currently just the ggml-metal OOM null-check: upstream omits a
# check after ggml_metal_buffer_init, so a Metal OOM faults with EXC_BAD_ACCESS
# instead of returning NULL). The `rsync --delete` re-pristines the copy each
# run, so every patch is applied fresh; a patch that no longer applies (the code
# moved on a llama.cpp bump) loud-fails rather than silently shipping the unsafe
# path. Upstream a patch to retire it from the series.
# ---------------------------------------------------------------------------
echo "==> Syncing build copy: vendor/llama.cpp -> ios/.build-llama/src-$MODE"
mkdir -p "$BUILD_SRC"
rsync -a --delete --exclude='.git' "$VENDOR/" "$BUILD_SRC/"

# Use `patch` (not `git apply`): the copy lives inside the repo's gitignored
# tree, where `git apply` resolves paths against the repo root and silently
# no-ops. The rsync above re-pristines the copy, so each patch applies fresh.
shopt -s nullglob
for p in "$PATCHES_DIR"/[0-9][0-9][0-9]-*.patch; do
    echo "==> Applying patch: $(basename "$p")"
    ( cd "$BUILD_SRC" && patch -p1 < "$p" ) \
        || { echo "error: failed to apply $(basename "$p") — does it still match this llama.cpp commit?" >&2; exit 3; }
done
shopt -u nullglob

# Defense-in-depth: confirm the OOM null-check actually landed in the copy. A
# silent no-op apply would otherwise ship the crash-on-OOM path. (Generalize
# this check if the series grows beyond the one Metal patch.)
grep -qF 'if (res == NULL)' "$BUILD_SRC/ggml/src/ggml-metal/ggml-metal.cpp" \
    || { echo "error: OOM null-check missing from the build copy after patching" >&2; exit 3; }

GGML_INCLUDE="$BUILD_SRC/ggml/include"
LLAMA_INCLUDE="$BUILD_SRC/include"

# ---------------------------------------------------------------------------
# Configure + build llama.cpp's OWN CMake against the build copy (the repo-root
# Android CMake is NOT involved). CMake owns the source list, the per-target
# flags, and the GGML_METAL_EMBED_LIBRARY metallib embed.
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
    -DLLAMA_CURL=OFF \
    "${METAL_ARGS[@]}" \
    ${SCCACHE_ARGS[@]+"${SCCACHE_ARGS[@]}"}  # +"${..}" guard: expanding an empty array under `set -u` errors on bash 3.2 (macOS/Xcode Cloud)

# Build the static-lib targets we combine below. ggml-base is wired as a
# dependency of ggml, and ggml-metal of ggml, so building llama + ggml brings
# the rest; name them explicitly so a missing target fails loudly.
BUILD_TARGETS=(llama ggml ggml-base ggml-cpu)
[[ "$USE_METAL" == "1" ]] && BUILD_TARGETS+=(ggml-metal)
echo "==> Building: ${BUILD_TARGETS[*]}"
cmake --build "$BUILD_DIR" --target "${BUILD_TARGETS[@]}"

# ---------------------------------------------------------------------------
# Combine the per-target static libs into ONE libllama.a (models build-xcframework.sh's
# combine_static_libraries). Collect every .a CMake produced so we don't have to
# track which target emits which lib — libtool dedups object files.
# ---------------------------------------------------------------------------
echo "==> Combining static libs into libllama.a"
mkdir -p "$OUT_DIR"
ARCHIVES=()
while IFS= read -r -d '' a; do ARCHIVES+=("$a"); done \
    < <(find "$BUILD_DIR" -name '*.a' -print0)
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
    echo "error: no static libraries produced under $BUILD_DIR" >&2
    exit 4
fi
printf '    + %s\n' "${ARCHIVES[@]##*/}"
rm -f "$OUT_DIR/libllama.a"
"$LIBTOOL" -static -o "$OUT_DIR/libllama.a" "${ARCHIVES[@]}" 2>/dev/null

echo "==> Publishing headers + module map to $OUT_DIR"
cp "$LLAMA_INCLUDE/llama.h"        "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml.h"          "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-alloc.h"    "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-backend.h"  "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-metal.h"    "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-cpu.h"      "$OUT_DIR/"
cp "$GGML_INCLUDE/ggml-opt.h"      "$OUT_DIR/"
cp "$GGML_INCLUDE/gguf.h"          "$OUT_DIR/"

cat > "$OUT_DIR/module.modulemap" <<'EOF'
module llama {
    header "llama.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-opt.h"
    header "gguf.h"
    export *
}
EOF

# Emit what the engine was built from as a Swift value so the app can report its
# runtime version natively (formerly the Rust `llamaCppCommit()`). Lives under
# Generated/Llama so the synchronized group auto-compiles it.
if [[ -n "$GGML_TAG" ]]; then
    BUILD_CASE=".tagged(tag: \"$GGML_TAG\", commit: \"$GGML_COMMIT\")"
else
    BUILD_CASE=".untagged(commit: \"$GGML_COMMIT\")"
fi
echo "==> Writing LlamaCppBuildInfo.swift (commit=$GGML_COMMIT tag=${GGML_TAG:-none})"
cat > "$OUT_DIR/LlamaCppBuildInfo.swift" <<EOF
// Generated by ios/build-llama.sh — do not edit.
// What the in-process llama.cpp engine was built from.
// nonisolated so the nonisolated SubmissionRef can read it off the main actor.
nonisolated enum LlamaCppBuildInfo {
    /// Most pins are upstream releases, but any commit is buildable — pinning a
    /// cherry-picked fix leaves nothing to tag it with, so \`untagged\` is a real
    /// state rather than a defensive one.
    nonisolated enum Build {
        case tagged(tag: String, commit: String)
        case untagged(commit: String)

        /// Always present, and the exact identity: a tag can be moved or deleted
        /// upstream, a commit cannot.
        var commit: String {
            switch self {
            case let .tagged(_, commit), let .untagged(commit):
                return commit
            }
        }

        /// The upstream git tag on this pin, when it carries one.
        var tag: String? {
            switch self {
            case let .tagged(tag, _): return tag
            case .untagged: return nil
            }
        }
    }

    static let build = Build$BUILD_CASE

    static var commit: String { build.commit }

    /// What a submission reports as \`repository_version\`, and what results carry
    /// as \`runtime_version\`: the tag when the pin has one, so an iPhone names its
    /// engine the way every desktop runtime does (\`repository_version = "b10216"\`),
    /// and the commit otherwise — which is still exact, just not a name a plan
    /// can pin by.
    static var submissionVersion: String { build.tag ?? build.commit }
}
EOF

# Publish the shared prompt-seed corpus into the app bundle. llama.cpp's
# per-token cost is content-sensitive, so iOS must tokenize the SAME text as the
# Rust CLIs / Android to produce comparable numbers. Source of truth is
# crates/pipette-ops/src/prompt_seed.txt (Rust embeds it via include_str!); copy
# it into Generated/ so the synchronized group bundles it as an app resource.
PROMPT_SEED_SRC="$REPO_ROOT/crates/pipette-ops/src/prompt_seed.txt"
PROMPT_SEED_DIR="$REPO_ROOT/ios/Pipette/Pipette/Generated/PromptSeed"
if [[ ! -f "$PROMPT_SEED_SRC" ]]; then
    echo "error: shared prompt seed missing at $PROMPT_SEED_SRC" >&2
    exit 5
fi
echo "==> Publishing prompt seed corpus to $PROMPT_SEED_DIR"
mkdir -p "$PROMPT_SEED_DIR"
cp "$PROMPT_SEED_SRC" "$PROMPT_SEED_DIR/prompt_seed.txt"

echo "==> Done. llama.cpp static lib + module in $OUT_DIR:"
ls -la "$OUT_DIR"
