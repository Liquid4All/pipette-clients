#!/usr/bin/env bash
#
# Xcode Cloud `ci_post_clone` hook, run before package resolution and the build.
# Sets up two things the runner needs but doesn't provide out of the box:
#   1. CMake + Ninja, which ci_pre_xcodebuild.sh uses to build vendored
#      llama.cpp (`cmake -G Ninja`).
#   2. Package plugin / macro fingerprint trust. Xcode Cloud generates its own
#      `xcodebuild archive` command, so it can't pass the `-skipPackagePlugin-
#      Validation` / `-skipMacroValidation` flags ios/build.sh uses locally.
#      Without them the build stops on the interactive "Trust & Enable" prompt
#      for the LicenseList build-tool plugin and the swift-syntax macros, which
#      a headless runner can't answer. These defaults are the flags' equivalent.

set -euo pipefail

echo "==> ci_post_clone: installing native build toolchain (cmake, ninja)"

brew install cmake ninja

echo "==> ci_post_clone: trusting SPM plugins + macros for headless archive"
# Apple's key really is spelled "Validatation" (double -ta-); it is NOT a typo here.
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidatation -bool YES
defaults write com.apple.dt.Xcode IDESkipMacroFingerprintValidation -bool YES

echo "==> ci_post_clone: done"
