#!/usr/bin/env bash
#
# Copy the llama.cpp frameworks ios/build-llama.sh staged into the app's
# Frameworks/, with their dSYMs, and sign them when the build signs. The app
# target's "Embed llama.cpp" build phase.
#
# PIPETTE_LLAMA_PINS names the builds the app carries: main, `llamacpp`, unless
# set. The internal archive adds every vendor/llamacpp-<variant> (build.yml,
# ios-archive); the App Store app carries main alone. A named build that was not
# staged, or was built against another revision of ios/native/pipette_llama.h,
# fails the phase.

set -euo pipefail

PINS="${PIPETTE_LLAMA_PINS:-llamacpp}"
STAGED="$SRCROOT/../.build-llama/frameworks/$PLATFORM_NAME"
DEST="$TARGET_BUILD_DIR/$FRAMEWORKS_FOLDER_PATH"
HEADER="$SRCROOT/../native/pipette_llama.h"
HEADER_DIGEST="$(shasum -a 256 "$HEADER" | cut -d' ' -f1)"

mkdir -p "$DEST"
# The app lists what Frameworks/ holds, so delete every build that an earlier run embedded.
rm -rf "$DEST"/llamacpp*.framework
read -r -a pins <<< "$PINS"
for pin in "${pins[@]}"; do
    if [[ ! -d "$STAGED/$pin.framework" ]]; then
        echo "error: $STAGED/$pin.framework is missing; build it with ios/build-llama.sh" >&2
        exit 1
    fi
    built_against="$(/usr/libexec/PlistBuddy -c 'Print :PipetteLlamaHeaderDigest' \
        "$STAGED/$pin.framework/Info.plist" 2>/dev/null || true)"
    if [[ "$built_against" != "$HEADER_DIGEST" ]]; then
        echo "error: $pin.framework was built against another $HEADER; rebuild it with ios/build-llama.sh" >&2
        exit 1
    fi
    echo "==> Embedding $pin.framework"
    cp -R "$STAGED/$pin.framework" "$DEST/"
    if [[ "${DEBUG_INFORMATION_FORMAT:-}" == "dwarf-with-dsym" ]]; then
        rm -rf "$DWARF_DSYM_FOLDER_PATH/$pin.framework.dSYM"
        cp -R "$STAGED/$pin.framework.dSYM" "$DWARF_DSYM_FOLDER_PATH/"
    fi
    # Before the app's own signature, which seals Frameworks/. CocoaPods signs
    # embedded frameworks the same way.
    if [[ "${CODE_SIGNING_ALLOWED:-NO}" == "YES" ]]; then
        codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" \
            --preserve-metadata=identifier,entitlements "$DEST/$pin.framework"
    fi
done
