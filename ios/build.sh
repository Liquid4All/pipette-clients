#!/usr/bin/env bash
#
# Build the Pipette iOS app from the command line.
#
# Usage:
#   ./ios/build.sh           # build for the simulator (default)
#   ./ios/build.sh sim       # build for the simulator (aarch64 sim slice)
#   ./ios/build.sh device    # build for a physical device (needs signing)
#   ./ios/build.sh --help    # this message
#
# Options:
#   --internal   compile in the private SoC die-temp and kernel footprint-peak
#                reads and the reclaim-ring flush: internal builds only, never
#                shipped. Equivalent to PIPETTE_PRIVATE_THERMAL=1.
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
# Passes -skipPackagePluginValidation and -skipMacroValidation so the LicenseList
# plugin and the swift-syntax macros build without the trust prompts a headless
# build cannot answer, and checks for the Metal toolchain mlx-swift needs. The
# "Build llama.cpp" phase (ios/build-llama.sh) runs inside xcodebuild, so there
# is no need to run it first. See docs/pipette-ios/build.md.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,34s/^# \{0,1\}//p' "$0"
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

# Without the full Xcode app, `xcodebuild` is absent or the Command Line Tools
# shim, which cannot build an .xcodeproj.
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

# The Metal toolchain is an on-demand Xcode component; without it every
# mlx-swift CompileMetalFile step fails. Block only when it is reported
# uninstalled: an Xcode without the subcommand prints no match and builds on.
if xcodebuild -showComponent MetalToolchain 2>/dev/null | grep -q "Status: uninstalled"; then
    echo "error: the Metal toolchain is not installed." >&2
    echo "  mlx-swift's Metal shaders cannot compile without it. Install with:" >&2
    echo "    xcodebuild -downloadComponent MetalToolchain" >&2
    echo "  (or Xcode -> Settings -> Components), then re-run this script." >&2
    exit 1
fi

# Generic destinations below: a build needs no booted simulator, and a named
# model breaks wherever it is not installed. ARCHS=arm64 + ONLY_ACTIVE_ARCH=YES:
# the llama.cpp static lib is arm64-only, so x86_64 would fail to link.
COMMON_ARGS=(
    -project "$PROJECT"
    -scheme Pipette
    -skipPackagePluginValidation
    -skipMacroValidation
    ARCHS=arm64
    ONLY_ACTIVE_ARCH=YES
)

# The SoC die-temp read (`soc_temp`), the kernel footprint-peak read and the
# reclaim-ring flush are private APIs, so they are off by default and never in
# the App Store archive (App Review rejects private APIs). See
# docs/pipette-ios/private-thermal-release-build.md.
#
# Swift conditions accumulate into ONE setting: a second
# `SWIFT_ACTIVE_COMPILATION_CONDITIONS=` argument would replace the first, so
# `--internal --debug-ui` would silently drop the private-thermal condition.
SWIFT_CONDITIONS=()

if [[ "${PIPETTE_PRIVATE_THERMAL:-}" == "1" ]]; then
    echo "==> PIPETTE_PRIVATE_THERMAL=1 — enabling the private SoC die-temp and kernel footprint-peak reads and the reclaim-ring flush (internal builds only, never ship)"
    COMMON_ARGS+=('GCC_PREPROCESSOR_DEFINITIONS=$(inherited) PIPETTE_PRIVATE_THERMAL=1')
    # Swift cannot see GCC_PREPROCESSOR_DEFINITIONS.
    SWIFT_CONDITIONS+=(PIPETTE_PRIVATE_THERMAL)
    # The footprint-peak read and reclaim-ring flush for max_memory_usage.
    # Swift-only, so no GCC define.
    SWIFT_CONDITIONS+=(PIPETTE_PRIVATE_MEMORY)
fi

# `#if DEBUG` is false in Release, so this is how a local Release build gets the
# Settings "Debugging" card. Xcode Cloud gets it from
# ios/Pipette/ci_scripts/ci_pre_xcodebuild.sh.
if [[ "${PIPETTE_DEBUG_UI:-}" == "1" ]]; then
    echo "==> PIPETTE_DEBUG_UI=1 — compiling in the Settings debug card"
    SWIFT_CONDITIONS+=(PIPETTE_DEBUG_UI)
fi

# `$(inherited)` is quoted here and above so xcodebuild, not the shell, expands
# it, keeping lower-level defines (notably DEBUG in the Debug configuration).
if [[ ${#SWIFT_CONDITIONS[@]} -gt 0 ]]; then
    COMMON_ARGS+=("SWIFT_ACTIVE_COMPILATION_CONDITIONS=\$(inherited) ${SWIFT_CONDITIONS[*]}")
fi

# Release unless the caller passes -configuration: it is what benchmark numbers need.
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
    # Unsigned is fine for the simulator only: `simctl install` accepts it.
    xcodebuild build \
        "${COMMON_ARGS[@]}" \
        -destination 'generic/platform=iOS Simulator' \
        CODE_SIGNING_ALLOWED=NO \
        "$@"
else
    echo "==> Building Pipette for a physical iOS device..."
    # -allowProvisioningUpdates lets Xcode resolve a signing identity and
    # provisioning profile. This builds but does not install: use `xcrun
    # devicectl` (docs/pipette-ios/build.md).
    xcodebuild build \
        "${COMMON_ARGS[@]}" \
        -destination 'generic/platform=iOS' \
        -allowProvisioningUpdates \
        "$@"
    echo "==> Built. To run it on a connected device, install with 'xcrun"
    echo "    devicectl device install app' — see docs/pipette-ios/build.md."
fi
