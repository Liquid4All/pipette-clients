# Android toolchain wrapper.
#
# Delegates to the NDK's own android.toolchain.cmake, then derives the Rust
# cross-compile target + the env the cargo_wrapper must export so `cargo build`
# produces an arm64-v8a cdylib with the same clang the NDK uses for the C/C++.
#
# Ported from inference_engine's cmake/android.toolchain.cmake. The NDK path
# comes from ANDROID_NDK (cache) or $ENV{ANDROID_NDK}; the thin
# build-rust-android.sh resolves and exports it.
#
# CMake includes the toolchain file in the top-level configure scope, so the
# PIPETTE_CARGO_* variables set here are visible to cmake/cargo.cmake.

if(NOT ANDROID_NDK AND DEFINED ENV{ANDROID_NDK})
  set(ANDROID_NDK "$ENV{ANDROID_NDK}")
endif()
if(NOT ANDROID_NDK)
  message(FATAL_ERROR
    "ANDROID_NDK is not set. Build Android through android/Pipette/build-rust-android.sh, "
    "which resolves the NDK and passes it in.")
endif()

include("${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

# Map the NDK ABI to the Rust target triple.
if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(PIPETTE_CARGO_TARGET "aarch64-linux-android")
elseif(ANDROID_ABI STREQUAL "x86_64")
  set(PIPETTE_CARGO_TARGET "x86_64-linux-android")
else()
  message(FATAL_ERROR "Unsupported ANDROID_ABI '${ANDROID_ABI}' (pipette ships arm64-v8a).")
endif()

# cargo/cc read target-specific env vars with the triple's `-` replaced by `_`
# (e.g. CC_aarch64_linux_android); the linker var is uppercased. Shell can't
# export an identifier containing `-`, so the underscore form is mandatory.
string(REPLACE "-" "_" PIPETTE_CARGO_TARGET_USCORE "${PIPETTE_CARGO_TARGET}")
string(TOUPPER "${PIPETTE_CARGO_TARGET_USCORE}" PIPETTE_CARGO_TARGET_UPPER)

# API level for the clang path: digits of ANDROID_PLATFORM (e.g. "android-31" ->
# "31"). NOTE: we deliberately do NOT reuse CMAKE_SYSTEM_VERSION here — although
# the NDK toolchain's adjust_api_level() normalizes the level into it, that value
# is not settled when this toolchain block is evaluated (it reads back as "1"),
# which would yield a non-existent "...-android1-clang". ANDROID_PLATFORM is the
# cache var we set in the preset, so deriving from it is reliable.
string(REGEX REPLACE "[^0-9]+" "" _PIPETTE_API "${ANDROID_PLATFORM}")

# The NDK toolchain file sets ANDROID_TOOLCHAIN_ROOT to the prebuilt llvm dir.
set(_PIPETTE_NDK_BIN "${ANDROID_TOOLCHAIN_ROOT}/bin")
set(_PIPETTE_CC "${_PIPETTE_NDK_BIN}/${PIPETTE_CARGO_TARGET}${_PIPETTE_API}-clang")

# Env the cargo_wrapper sources before invoking cargo. rustc's linker reads
# CARGO_TARGET_*_LINKER; the cc crate reads CC_/CXX_/AR_ for any C/C++ in the
# dependency graph (pipette-android's own build.rs no longer compiles C++ —
# CMake does — but transitive build scripts may still need them). Names are the
# cargo target-specific env vars.
set(PIPETTE_CARGO_ENV "\
export AR_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_NDK_BIN}/llvm-ar\"
export CC_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_CC}\"
export CXX_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_NDK_BIN}/${PIPETTE_CARGO_TARGET}${_PIPETTE_API}-clang++\"
export CARGO_TARGET_${PIPETTE_CARGO_TARGET_UPPER}_LINKER=\"${_PIPETTE_CC}\"
export ANDROID_NDK_HOME=\"${ANDROID_NDK}\"
")
