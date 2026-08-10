#!/usr/bin/env bash
#
# Install the built Pipette app on every attached device.
#
# Usage:
#   ./ios/install.sh              # every connected device (default)
#   ./ios/install.sh device       # same
#   ./ios/install.sh sim          # every booted simulator
#   ./ios/install.sh --udid <id>  # just this one (repeatable)
#   ./ios/install.sh --help       # this message
#
# Installs what ./ios/build.sh already produced — it never builds. Pass
# --configuration to install a non-Release build:
#   ./ios/build.sh device -configuration Debug && ./ios/install.sh --configuration Debug
#
# `devicectl list devices` remembers every device this Mac has ever paired
# with, nearly all of them absent. Only those with a live transport are
# installable, so that is what this selects — plug in a phone and it is
# picked up, unplug it and it stops being a target, with no UDID to edit.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,20s/^# \{0,1\}//p' "$0"
    exit 0
fi

MODE=device
CONFIGURATION=Release
UDIDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        sim|device) MODE="$1"; shift ;;
        --udid) UDIDS+=("${2:?--udid needs a value}"); shift 2 ;;
        --configuration) CONFIGURATION="${2:?--configuration needs a value}"; shift 2 ;;
        *)
            echo "error: unknown argument '$1'" >&2
            echo "  Run '$0 --help' for usage." >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$REPO_ROOT/ios/Pipette/Pipette.xcodeproj"
BUNDLE_ID="ai.liquid.liquid-pipette"

if [[ "$MODE" == "sim" ]]; then
    DESTINATION='generic/platform=iOS Simulator'
else
    DESTINATION='generic/platform=iOS'
fi

# Ask xcodebuild where it puts the product rather than reconstructing the
# DerivedData path: the hash in it is not derivable, and a stale guess installs
# yesterday's build without saying so.
SETTINGS=$(xcodebuild -project "$PROJECT" -scheme Pipette \
    -configuration "$CONFIGURATION" -destination "$DESTINATION" \
    -skipPackagePluginValidation -skipMacroValidation \
    -showBuildSettings 2>/dev/null)
APP=$(printf '%s\n' "$SETTINGS" | awk -F' = ' '
    / BUILT_PRODUCTS_DIR/ { dir = $2 }
    / FULL_PRODUCT_NAME/  { name = $2 }
    END { if (dir && name) print dir "/" name }')

if [[ -z "$APP" || ! -d "$APP" ]]; then
    echo "error: no $CONFIGURATION build to install${APP:+ at $APP}" >&2
    echo "  Build it first:  ./ios/build.sh $MODE" >&2
    exit 1
fi
echo "==> Installing $APP"

if [[ ${#UDIDS[@]} -eq 0 ]]; then
    if [[ "$MODE" == "sim" ]]; then
        while read -r udid; do
            [[ -n "$udid" ]] && UDIDS+=("$udid")
        done < <(xcrun simctl list devices booted |
            sed -n 's/.*(\([0-9A-F-]\{36\}\)) (Booted).*/\1/p')
    else
        # A device with no `transportType` is one this Mac merely remembers.
        while read -r udid; do
            [[ -n "$udid" ]] && UDIDS+=("$udid")
        done < <(xcrun devicectl list devices --hide-headers \
            --hide-default-columns --columns identifier \
            --filter "connectionProperties.transportType != nil")
    fi
fi

if [[ ${#UDIDS[@]} -eq 0 ]]; then
    if [[ "$MODE" == "sim" ]]; then
        echo "error: no booted simulator — boot one with 'xcrun simctl boot <name>'" >&2
    else
        echo "error: no connected device — plug one in and unlock it" >&2
        echo "  Paired-but-absent devices are listed by 'xcrun devicectl list devices'." >&2
    fi
    exit 1
fi

FAILED=()
for udid in "${UDIDS[@]}"; do
    echo "==> $udid"
    if [[ "$MODE" == "sim" ]]; then
        xcrun simctl install "$udid" "$APP" || FAILED+=("$udid")
    else
        xcrun devicectl device install app --device "$udid" "$APP" || FAILED+=("$udid")
    fi
done

# Report every device before failing, so one locked phone doesn't hide the
# result for the rest of a fleet install.
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "error: install failed on ${#FAILED[@]} of ${#UDIDS[@]}: ${FAILED[*]}" >&2
    exit 1
fi

echo "==> Installed on ${#UDIDS[@]} device(s): ${UDIDS[*]}"
echo "    Run headless with:"
if [[ "$MODE" == "sim" ]]; then
    echo "      xcrun simctl launch --console-pty ${UDIDS[0]} $BUNDLE_ID headlessrun models list"
else
    echo "      xcrun devicectl device process launch --console --device ${UDIDS[0]} \\"
    echo "          --terminate-existing $BUNDLE_ID headlessrun models list"
fi
