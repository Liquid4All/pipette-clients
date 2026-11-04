#!/usr/bin/env bash
#
# Copy the tiny ONNX Runtime GenAI test model from the vendored submodule into the
# test bundle, so the engine tests also run on a device: `#filePath` names the
# build machine's filesystem, which a phone cannot see. ~3 MB of random-weight
# GPT-2, already in the onnxruntime-genai submodule.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="$REPO_ROOT/vendor/onnxruntime-genai/test/models/hf-internal-testing/tiny-random-gpt2-fp32"
# `TARGET_BUILD_DIR`, not `BUILT_PRODUCTS_DIR`: for a test bundle hosted in an app the
# two differ, and the hosted copy under `Pipette.app/PlugIns` is the one that gets
# installed. Writing to the other leaves the fixture on the build machine.
DEST="${TARGET_BUILD_DIR:?}/${UNLOCALIZED_RESOURCES_FOLDER_PATH:?}/OnnxGenaiFixture"

# Absent when the submodule is not checked out. The tests are gated on the fixture being
# readable and report as skipped, so a tree without it builds and tests fine.
if [[ ! -f "$FIXTURE/genai_config.json" ]]; then
    echo "note: $FIXTURE is not checked out; the ONNX engine tests will skip"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -R "$FIXTURE" "$DEST"
echo "==> Copied the ONNX GenAI test fixture into the test bundle"
