# Android toolchain: wraps the NDK's android.toolchain.cmake and derives the Rust
# target plus the env cargo_wrapper exports, so the cdylib links with the NDK's
# clang. The toolchain file runs in the top-level scope, so the PIPETTE_CARGO_*
# variables set here reach cmake/cargo.cmake.

if(NOT ANDROID_NDK AND DEFINED ENV{ANDROID_NDK})
  set(ANDROID_NDK "$ENV{ANDROID_NDK}")
endif()
if(NOT ANDROID_NDK)
  message(FATAL_ERROR
    "ANDROID_NDK is not set. Build Android through android/Pipette/build-rust-android.sh, "
    "which resolves the NDK and passes it in.")
endif()

include("${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(PIPETTE_CARGO_TARGET "aarch64-linux-android")
elseif(ANDROID_ABI STREQUAL "x86_64")
  set(PIPETTE_CARGO_TARGET "x86_64-linux-android")
else()
  message(FATAL_ERROR "Unsupported ANDROID_ABI '${ANDROID_ABI}' (pipette ships arm64-v8a).")
endif()

# cargo/cc read target env vars with the triple's `-` as `_` (CC_aarch64_linux_android),
# uppercased for the linker var; a shell cannot export a name containing `-`.
string(REPLACE "-" "_" PIPETTE_CARGO_TARGET_USCORE "${PIPETTE_CARGO_TARGET}")
string(TOUPPER "${PIPETTE_CARGO_TARGET_USCORE}" PIPETTE_CARGO_TARGET_UPPER)

# API level from ANDROID_PLATFORM, not CMAKE_SYSTEM_VERSION: the NDK has not
# settled the latter yet at this point (it reads "1", giving a nonexistent
# "...-android1-clang").
string(REGEX REPLACE "[^0-9]+" "" _PIPETTE_API "${ANDROID_PLATFORM}")

set(_PIPETTE_NDK_BIN "${ANDROID_TOOLCHAIN_ROOT}/bin")
set(_PIPETTE_CC "${_PIPETTE_NDK_BIN}/${PIPETTE_CARGO_TARGET}${_PIPETTE_API}-clang")

# Pasted into cargo_wrapper. CC_/CXX_/AR_ serve the cc crate in transitive build
# scripts (pipette-android's build.rs compiles no C/C++; CMake does).
set(PIPETTE_CARGO_ENV "\
export AR_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_NDK_BIN}/llvm-ar\"
export CC_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_CC}\"
export CXX_${PIPETTE_CARGO_TARGET_USCORE}=\"${_PIPETTE_NDK_BIN}/${PIPETTE_CARGO_TARGET}${_PIPETTE_API}-clang++\"
export CARGO_TARGET_${PIPETTE_CARGO_TARGET_UPPER}_LINKER=\"${_PIPETTE_CC}\"
export ANDROID_NDK_HOME=\"${ANDROID_NDK}\"
")
