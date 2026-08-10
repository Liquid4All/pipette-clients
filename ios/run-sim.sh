#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/Pipette"

DEVICE="${DEVICE:-iPhone 17 Pro}"
SCHEME="Pipette"
BUNDLE_ID="ai.liquid.liquid-pipette"
DERIVED="${DERIVED_DATA:-$SCRIPT_DIR/DerivedData}"

xcrun simctl boot "$DEVICE" 2>/dev/null || true
open -a Simulator

xcodebuild \
  -project Pipette.xcodeproj \
  -scheme "$SCHEME" \
  -configuration Debug \
  -destination "platform=iOS Simulator,name=$DEVICE" \
  -derivedDataPath "$DERIVED" \
  build

APP_PATH="$DERIVED/Build/Products/Debug-iphonesimulator/Pipette.app"
xcrun simctl install booted "$APP_PATH"
xcrun simctl launch booted "$BUNDLE_ID"
