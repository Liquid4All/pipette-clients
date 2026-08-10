#!/usr/bin/env bash
#
# Build the Pipette iOS app from the command line.
#
# Usage:
#   ./ios/build.sh           # build for the simulator (default)
#   ./ios/build.sh sim       # build for the simulator (aarch64 sim slice)
#   ./ios/build.sh device    # build for a physical device (needs signing)
#   ./ios/build.sh --help     # this message
#
# Options:
#   --internal   compile in the private SoC die-temp read — internal builds
#                only, never shipped. Equivalent to PIPETTE_PRIVATE_THERMAL=1.
#   --debug-ui   compile in the Settings "Debugging" card. Already on in Debug
#                builds; this is how to get it in a local Release build.
#                Equivalent to PIPETTE_DEBUG_UI=1. On Xcode Cloud the Internal
#                Testing workflow sets that variable instead.
#
# Extra arguments after the mode are forwarded verbatim to xcodebuild, e.g.
#   ./ios/build.sh sim -derivedDataPath /tmp/pipette-dd
#
# Builds Release by default; pass your own -configuration to override, e.g.
#   ./ios/build.sh device -configuration Debug
#
# This wrapper owns the xcodebuild-layer concerns that build-llama.sh
# structurally cannot (build-llama.sh runs as the "Build llama.cpp" phase
# *inside* this xcodebuild invocation — it is invoked by xcodebuild, not the
# reverse):
#   - passes -skipPackagePluginValidation so the LicenseList build-tool plugin
#     runs without the interactive "Trust & Enable" prompt a headless build
#     can't answer, and -skipMacroValidation so the swift-syntax macros
#     (MLXHuggingFaceMacros, swift-transformers) build without the same
#     interactive macro-trust approval — a fresh checkout has never granted it;
#   - pre-flights the Metal toolchain (required by the mlx-swift dependency)
#     and prints an actionable install line instead of a 200-line wall of
#     CompileMetalFile errors;
#   - selects the right -destination for the chosen mode.
#
# The llama.cpp static library + `llama` module are produced automatically by
# the "Build llama.cpp" phase (ios/build-llama.sh) that xcodebuild runs for us;
# you do NOT need to run build-llama.sh first. See docs/pipette-ios/build.md.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,33s/^# \{0,1\}//p' "$0"
    exit 0
fi

# --internal and --debug-ui are ours; everything else belongs to xcodebuild, so pull
# them out rather than letting xcodebuild reject them. Before MODE is read, so a flag
# may precede the mode as well as follow it.
FORWARDED=()
for arg in "$@"; do
    case "$arg" in
        --internal) PIPETTE_PRIVATE_THERMAL=1 ;;
        --debug-ui) PIPETTE_DEBUG_UI=1 ;;
        *)          FORWARDED+=("$arg") ;;
    esac
done
set -- ${FORWARDED[@]+"${FORWARDED[@]}"}

MODE="${1:-sim}"
[[ $# -gt 0 ]] && shift   # remaining args ($@) are forwarded to xcodebuild

case "$MODE" in
    sim|device) ;;
    *)
        echo "error: unknown mode '$MODE' (expected: sim or device)" >&2
        echo "  Run '$0 --help' for usage." >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$REPO_ROOT/ios/Pipette/Pipette.xcodeproj"

# Pre-flight: Xcode itself. Without it `xcodebuild` is either absent or is the
# Command Line Tools shim (which can't build an .xcodeproj), so fail with an
# actionable message instead of a raw "command not found" or a cryptic
# "tool 'xcodebuild' requires Xcode" deep in the build.
if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "error: 'xcodebuild' not found — the full Xcode app is required (the" >&2
    echo "  Command Line Tools alone can't build the Pipette .xcodeproj)." >&2
    echo "  1. Install Xcode from the App Store." >&2
    echo "  2. Point the toolchain at it and finish first launch:" >&2
    echo "       sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    echo "       xcodebuild -runFirstLaunch" >&2
    exit 1
elif ! xcodebuild -version >/dev/null 2>&1; then
    echo "error: 'xcodebuild' is present but not usable — likely the Command Line" >&2
    echo "  Tools shim rather than the full Xcode app." >&2
    echo "  Install Xcode, then select it:" >&2
    echo "    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    echo "    xcodebuild -runFirstLaunch" >&2
    exit 1
fi

# Pre-flight: the Metal toolchain is an on-demand Xcode component (Xcode 16+).
# Without it the mlx-swift dependency fails every CompileMetalFile step with
# "cannot execute tool 'metal' due to missing Metal Toolchain". We only block
# when we POSITIVELY detect it's uninstalled — on older Xcode the subcommand
# doesn't exist (no match), so we fall through and let the build proceed.
if xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -q "Status: uninstalled"; then
    echo "error: the Metal toolchain is not installed." >&2
    echo "  mlx-swift's Metal shaders cannot compile without it. Install with:" >&2
    echo "    xcodebuild -downloadComponent MetalToolchain" >&2
    echo "  (or Xcode -> Settings -> Components), then re-run this script." >&2
    exit 1
fi

# Map the mode to an xcodebuild destination. Use a *generic* destination: a
# plain build doesn't need a specific booted simulator, and naming a concrete
# device ("iPhone 17 Pro") would break whenever that model isn't installed.
#   - ARCHS=arm64 + ONLY_ACTIVE_ARCH=YES: the llama.cpp static lib is built
#     arm64-only, so this avoids x86_64 link failures on the generic destination.
#   - -skipPackagePluginValidation / -skipMacroValidation: see header. Both
#     mirror what CI passes to xcodebuild, so a fresh/headless build matches CI.
COMMON_ARGS=(
    -project "$PROJECT"
    -scheme Pipette
    -skipPackagePluginValidation
    -skipMacroValidation
    ARCHS=arm64
    ONLY_ACTIVE_ARCH=YES
)

# The SoC die-temp read (`soc_temp`) is a private API, so it is OFF by default —
# the project defines nothing, and the CI / App Store archive never ships it (App
# Review rejects private APIs). Opt in for internal on-device benchmarking:
#   ./ios/build.sh device --internal
# See docs/pipette-ios/private-thermal-release-build.md. Single-quoted so the shell
# leaves `$(inherited)` for xcodebuild to expand (preserving lower-level defines).
# Swift conditions accumulate into ONE setting. Two `SWIFT_ACTIVE_COMPILATION_CONDITIONS=`
# arguments would not merge — the later one wins and the earlier condition is silently
# lost, so `--internal --debug-ui` would quietly drop the private-thermal read.
SWIFT_CONDITIONS=()

if [[ "${PIPETTE_PRIVATE_THERMAL:-}" == "1" ]]; then
    echo "==> PIPETTE_PRIVATE_THERMAL=1 — enabling the private SoC die-temp read (internal builds only, never ship)"
    COMMON_ARGS+=('GCC_PREPROCESSOR_DEFINITIONS=$(inherited) PIPETTE_PRIVATE_THERMAL=1')
    # Also expose it to Swift so the SoC collection path is compiled in (not just
    # the ObjC read) — Swift can't see GCC_PREPROCESSOR_DEFINITIONS.
    SWIFT_CONDITIONS+=(PIPETTE_PRIVATE_THERMAL)
fi

# Compiles in the Settings "Debugging" card for a local Release build, which is
# otherwise the one configuration that cannot show it (`#if DEBUG` is false there).
# On Xcode Cloud the same condition comes from the PIPETTE_DEBUG_UI environment
# variable via ci_scripts/ci_pre_xcodebuild.sh; this flag is the local equivalent.
if [[ "${PIPETTE_DEBUG_UI:-}" == "1" ]]; then
    echo "==> PIPETTE_DEBUG_UI=1 — compiling in the Settings debug card"
    SWIFT_CONDITIONS+=(PIPETTE_DEBUG_UI)
fi

# Single-quoted `$(inherited)` so the shell leaves it for xcodebuild to expand,
# preserving lower-level defines (notably DEBUG in the Debug configuration).
if [[ ${#SWIFT_CONDITIONS[@]} -gt 0 ]]; then
    COMMON_ARGS+=("SWIFT_ACTIVE_COMPILATION_CONDITIONS=\$(inherited) ${SWIFT_CONDITIONS[*]}")
fi

# Default to a Release build — it's what benchmark numbers need. The caller can
# still pick another configuration by passing their own `-configuration` (e.g.
# `-configuration Debug` for a faster, unoptimized build); we only inject the
# default when the forwarded args don't already set one.
for arg in "$@"; do
    if [[ "$arg" == "-configuration" ]]; then
        CONFIG_SET=1
        break
    fi
done
if [[ -z "${CONFIG_SET:-}" ]]; then
    COMMON_ARGS+=(-configuration Release)
fi

if [[ "$MODE" == "sim" ]]; then
    echo "==> Building Pipette for the iOS Simulator..."
    # CODE_SIGNING_ALLOWED=NO is safe for the simulator only — it yields an
    # arm64 .app that `simctl install` accepts. Device builds need a real
    # signing identity, so we don't pass it below.
    xcodebuild build \
        "${COMMON_ARGS[@]}" \
        -destination 'generic/platform=iOS Simulator' \
        CODE_SIGNING_ALLOWED=NO \
        "$@"
else
    echo "==> Building Pipette for a physical iOS device..."
    # Device builds require code signing; -allowProvisioningUpdates lets Xcode
    # resolve a signing identity / provisioning profile automatically. This
    # produces a signed .app in DerivedData but does NOT deploy it — install it
    # on a connected device with `xcrun devicectl` (see the device run loop in
    # docs/pipette-ios/build.md), or pass -derivedDataPath to locate the .app.
    xcodebuild build \
        "${COMMON_ARGS[@]}" \
        -destination 'generic/platform=iOS' \
        -allowProvisioningUpdates \
        "$@"
    echo "==> Built. To run it on a connected device, install with 'xcrun"
    echo "    devicectl device install app' — see docs/pipette-ios/build.md."
fi
