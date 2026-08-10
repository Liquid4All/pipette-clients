#!/usr/bin/env bash
#
# Build the Pipette Android app from the command line.
#
# Usage:
#   ./android/build.sh            # assemble the debug APK (:app:assembleDebug)
#   ./android/build.sh test       # compile + unit tests + assemble
#   ./android/build.sh --help     # this message
#   ./android/build.sh TASKS...   # any other args are forwarded to gradlew
#
# A fresh checkout needs a JDK, the Android SDK + NDK, CMake, Ninja, Rust, and
# the vendored submodules. This wrapper resolves what it safely can and fails
# fast with the exact fix for what it can't, so you avoid the usual scavenger
# hunt. Specifically it:
#   - finds a JDK (>= 17) to run Gradle and exports JAVA_HOME — it probes the
#     Homebrew location directly, so a Homebrew JDK works even un-symlinked;
#   - finds the Android SDK and exports ANDROID_HOME (no local.properties is
#     written — nothing persists silently);
#   - initializes any uninitialized vendored submodules;
#   - checks for the NDK, cmake, ninja, and cargo, printing the exact install
#     command for whatever is absent. It never installs anything, runs brew, or
#     accepts SDK licenses on your behalf.
#
# Gradle itself is fetched by ./gradlew; the aarch64-linux-android Rust target
# is added by the native build. See docs/pipette-android/build.md.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25s/^# \{0,1\}//p' "$0"
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRADLE_DIR="$REPO_ROOT/android/Pipette"

# Print an actionable error: first line is the problem, the rest are indented
# guidance, then exit non-zero.
die() {
    echo "error: $1" >&2
    shift
    local line
    for line in "$@"; do echo "  $line" >&2; done
    exit 1
}

# Pick the gradle tasks. `test` is a convenience for the full check; with no
# args we assemble the debug APK; anything else is forwarded verbatim.
case "${1:-}" in
    test)
        shift
        GRADLE_TASKS=(:app:compileDebugKotlin :app:testDebugUnitTest :app:assembleDebug "$@")
        ;;
    "")
        GRADLE_TASKS=(:app:assembleDebug)
        ;;
    *)
        GRADLE_TASKS=("$@")
        ;;
esac

# -----------------------------------------------------------------------
# JDK — Gradle is a JVM app, so a JDK must already exist to launch it. We probe
# common locations (including Homebrew's, which the macOS java registry misses
# unless symlinked) and pick the first that is >= 17.
# -----------------------------------------------------------------------
java_major() {  # $1 = java binary; prints the major version, empty if unknown
    local v
    v="$("$1" -version 2>&1 | awk -F'"' '/version/{print $2; exit}')" || true
    v="${v%%.*}"
    [[ "$v" =~ ^[0-9]+$ ]] && echo "$v"
}

resolve_jdk() {
    local cand jhome maj
    local -a candidates=()
    [[ -n "${JAVA_HOME:-}" ]] && candidates+=("$JAVA_HOME")
    if [[ -x /usr/libexec/java_home ]]; then
        cand="$(/usr/libexec/java_home 2>/dev/null || true)"
        [[ -n "$cand" ]] && candidates+=("$cand")
    fi
    for cand in /opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home \
                /opt/homebrew/opt/openjdk@*/libexec/openjdk.jdk/Contents/Home \
                /usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home \
                /usr/local/opt/openjdk@*/libexec/openjdk.jdk/Contents/Home \
                /Library/Java/JavaVirtualMachines/*/Contents/Home \
                /usr/lib/jvm/*; do
        [[ -d "$cand" ]] && candidates+=("$cand")
    done
    # PATH java, unless it's Apple's /usr/bin/java stub.
    if cand="$(command -v java 2>/dev/null)" && [[ "$cand" != /usr/bin/java ]]; then
        jhome="$(cd "$(dirname "$cand")/.." 2>/dev/null && pwd)" || true
        [[ -n "$jhome" ]] && candidates+=("$jhome")
    fi
    for cand in ${candidates[@]+"${candidates[@]}"}; do
        [[ -x "$cand/bin/java" ]] || continue
        maj="$(java_major "$cand/bin/java")"
        [[ -n "$maj" && "$maj" -ge 17 ]] && { echo "$cand"; return 0; }
    done
    return 1
}

if JDK="$(resolve_jdk)"; then
    export JAVA_HOME="$JDK"
    echo "==> JAVA_HOME=$JAVA_HOME"
else
    die "no JDK 17+ found to run Gradle." \
        "Gradle is a JVM app and needs a JDK already installed to launch it." \
        "Install one, e.g.:  brew install openjdk" \
        "Then re-run — this script probes the Homebrew location directly, so you" \
        "do not need the symlink Homebrew suggests. (Or set JAVA_HOME yourself.)"
fi

# -----------------------------------------------------------------------
# Android SDK — point Gradle at it via ANDROID_HOME (env > local.properties >
# common install locations). We export rather than write local.properties.
# -----------------------------------------------------------------------
resolve_sdk() {
    local cand
    local -a candidates=()
    [[ -n "${ANDROID_HOME:-}" ]] && candidates+=("$ANDROID_HOME")
    [[ -n "${ANDROID_SDK_ROOT:-}" ]] && candidates+=("$ANDROID_SDK_ROOT")
    if [[ -f "$GRADLE_DIR/local.properties" ]]; then
        cand="$(sed -n 's/^sdk\.dir=//p' "$GRADLE_DIR/local.properties" | head -1)"
        [[ -n "$cand" ]] && candidates+=("$cand")
    fi
    candidates+=(
        "$HOME/Library/Android/sdk"
        "/opt/homebrew/share/android-commandlinetools"
        "$HOME/Android/Sdk"
        "/usr/local/share/android-commandlinetools"
    )
    for cand in ${candidates[@]+"${candidates[@]}"}; do
        [[ -d "$cand/cmdline-tools" || -d "$cand/platform-tools" ]] && { echo "$cand"; return 0; }
    done
    return 1
}

if SDK="$(resolve_sdk)"; then
    export ANDROID_HOME="$SDK"
    export ANDROID_SDK_ROOT="$SDK"
    echo "==> ANDROID_HOME=$ANDROID_HOME"
else
    die "Android SDK not found." \
        "Install it (Android Studio, or 'brew install --cask android-commandlinetools')" \
        "then set ANDROID_HOME and re-run. See docs/pipette-android/build.md for the" \
        "sdkmanager package list (platform-tools, platforms;android-36, build-tools, ndk)."
fi

# NDK — required by the native build and NOT auto-downloaded by AGP. Honor an
# explicit ANDROID_NDK_HOME/ANDROID_NDK if it points at a real NDK; otherwise it
# must live under the SDK. A directory alone isn't enough — a usable NDK has a
# source.properties at its root, so we check for that to catch a partial or
# corrupt install here rather than deep in build-rust-android.sh.
ndk_is_valid() { [[ -n "$1" && -f "$1/source.properties" ]]; }

have_ndk=false
if ndk_is_valid "${ANDROID_NDK_HOME:-}" || ndk_is_valid "${ANDROID_NDK:-}"; then
    have_ndk=true
else
    for cand in "$ANDROID_HOME"/ndk/*/source.properties; do
        [[ -f "$cand" ]] && { have_ndk=true; break; }
    done
fi
if [[ "$have_ndk" != true ]]; then
    die "no usable Android NDK found under $ANDROID_HOME/ndk." \
        "(A directory may exist but lacks a source.properties — a partial install.)" \
        "AGP will not download it for you. Install it (accept licenses first):" \
        "  SDKM=\"\$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager\"" \
        "  yes | \"\$SDKM\" --licenses" \
        "  \"\$SDKM\" 'ndk;<version>'      # '\"\$SDKM\" --list | grep ndk' for options" \
        "Or point ANDROID_NDK_HOME at an existing NDK."
fi

# -----------------------------------------------------------------------
# CMake + Ninja — the native llama.cpp build (build-rust-android.sh) needs both
# on PATH. These are system tools, NOT the SDK's `cmake;` package.
# -----------------------------------------------------------------------
missing_tools=()
command -v cmake >/dev/null 2>&1 || missing_tools+=(cmake)
command -v ninja >/dev/null 2>&1 || missing_tools+=(ninja)
if (( ${#missing_tools[@]} )); then
    die "missing native build tool(s): ${missing_tools[*]}" \
        "build-rust-android.sh drives the llama.cpp/ggml build through CMake + Ninja." \
        "Install with:  brew install ${missing_tools[*]}" \
        "(Linux: e.g. 'sudo apt-get install ${missing_tools[*]}')"
fi

# -----------------------------------------------------------------------
# Rust — the native build invokes cargo/rustup (and adds the target itself).
# -----------------------------------------------------------------------
[[ -d "$HOME/.cargo/bin" ]] && PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null 2>&1 || die \
    "cargo not found on PATH." \
    "Install Rust from https://rustup.rs, then re-run."
command -v rustup >/dev/null 2>&1 || die \
    "rustup not found on PATH (required by build-rust-android.sh)." \
    "Install Rust from https://rustup.rs, then re-run."

# -----------------------------------------------------------------------
# Vendored submodules — build.rs / CMake compile their sources in-process.
# -----------------------------------------------------------------------
# `git submodule status --recursive` prefixes uninitialized submodules with '-',
# descending into nested submodules (some vendored submodules have their own).
# Init them all rather than naming individual ones, so new submodules are picked
# up automatically. Initialized submodules are left untouched.
if git -C "$REPO_ROOT" submodule status --recursive | grep -q '^-'; then
    echo "==> Initializing vendored submodules..."
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

# -----------------------------------------------------------------------
# Build.
# -----------------------------------------------------------------------
echo "==> Running: ./gradlew ${GRADLE_TASKS[*]}"
cd "$GRADLE_DIR"
exec ./gradlew "${GRADLE_TASKS[@]}"
