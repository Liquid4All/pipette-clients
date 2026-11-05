#!/usr/bin/env bash
#
# Xcode Cloud `ci_post_clone` hook, run before package resolution and the build.
# Sets up what the runner lacks:
#   1. CMake + Ninja for ci_pre_xcodebuild.sh's llama.cpp build, and uv for the
#      Python that drives build-onnx-genai.sh's onnxruntime build.
#   2. Package plugin and macro fingerprint trust. Xcode Cloud owns the
#      `xcodebuild archive` command, so it can't pass the
#      `-skipPackagePluginValidation` / `-skipMacroValidation` flags ios/build.sh
#      uses, and a headless runner can't answer the "Trust & Enable" prompt for
#      the LicenseList plugin and swift-syntax macros. These defaults are the
#      flags' equivalent.

set -euo pipefail

echo "==> ci_post_clone: installing native build toolchain (cmake, ninja, uv)"

brew install cmake ninja uv

echo "==> ci_post_clone: trusting SPM plugins + macros for headless archive"
# Apple's key really is spelled "Validatation" (double -ta-); it is NOT a typo here.
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidatation -bool YES
defaults write com.apple.dt.Xcode IDESkipMacroFingerprintValidation -bool YES

echo "==> ci_post_clone: done"
