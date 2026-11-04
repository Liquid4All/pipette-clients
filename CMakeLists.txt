# Android native build: vendored llama.cpp, the `pipette_bridge` shim, the engine
# .so (Rust cdylib via cargo, or the C++ `pipette_android` target), staged into jniLibs.
#
# The GGML_*/LLAMA_* flags live in CMakePresets.json so they are in the cache
# before llama.cpp's option() calls run.

cmake_minimum_required(VERSION 3.25)
project(pipette_native C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(toolchain-helpers)

if(NOT ANDROID)
  message(FATAL_ERROR
    "pipette_native currently supports only the Android build. "
    "Configure with: cmake --preset android-arm64-v8a")
endif()

set(PIPETTE_ENGINE_BACKEND "rust" CACHE STRING "Engine backend: rust or cpp")

compute_backend_libs_suffix(PIPETTE_LINK_LIBS_SUFFIX)
set(PIPETTE_LINK_LIBS_PATH_FILE
    "${CMAKE_BINARY_DIR}/.pipette_link_libs.${PIPETTE_LINK_LIBS_SUFFIX}")

# Sentry matches an uploaded debug file to the crashing image by GNU build-id, so
# the shipped (stripped) lib and the uploaded (unstripped) copy must carry the same
# one. The note is SHF_ALLOC, so `llvm-strip --strip-unneeded` keeps it. Covers
# every subdirectory below; the Rust cdylib gets it via android/native's marker.
add_link_options(-Wl,--build-id=sha1)

# EXCLUDE_FROM_ALL so `cmake --build` skips llama.cpp's tool executables, which
# don't cross-compile for Android. What we need still builds through dependencies:
# the bridge links llama/mtmd/ggml/ggml-base, and ggml/src makes the CPU-variant
# backends dependencies of `ggml`.
add_subdirectory(vendor/llama.cpp "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# OFF builds the engine with no sentry-native, so a clone without the submodule
# still produces a working APK. The Gradle build decides the real value (from the
# DSN, or an explicit override; see android/Pipette/app/build.gradle.kts) and passes
# it down through android/Pipette/build-{rust,native}-android.sh. Defaulting ON would compile sentry-native,
# and require its submodule, for a build with no DSN that could never report.
# verifyReleaseSentryDsn in Gradle guards our release builds against losing it.
option(PIPETTE_ENABLE_CRASH_REPORTING
       "Build sentry-native crash capture into the :benchmark engine" OFF)

# Otherwise the build dies deep inside the missing subdirectory with a message
# that names neither remedy.
if(PIPETTE_ENABLE_CRASH_REPORTING
   AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/sentry-native/CMakeLists.txt")
  message(FATAL_ERROR
    "PIPETTE_ENABLE_CRASH_REPORTING is ON but vendor/sentry-native is not checked "
    "out. Either fetch it (--recursive is required: sentry-native carries its own "
    "nested submodule, external/libunwindstack-ndk, and without it the build fails "
    "later and less clearly):\n"
    "    git submodule update --init --recursive vendor/sentry-native\n"
    "or build without native crash capture:\n"
    "    cmake --preset android-arm64-v8a -DPIPETTE_ENABLE_CRASH_REPORTING=OFF")
endif()

if(PIPETTE_ENABLE_CRASH_REPORTING)
  # `inproc`, not crashpad: crashpad's separate handler process is the wrong model
  # for a short-lived isolated process. `SENTRY_TRANSPORT=none` because
  # android/native/crash_reporter.cpp writes envelopes to disk instead, so
  # `:benchmark` has no network thread to perturb the memory benchmark. The main
  # process uploads them via its JVM Sentry SDK.
  set(SENTRY_BACKEND inproc CACHE STRING "sentry-native crash backend" FORCE)
  set(SENTRY_TRANSPORT none CACHE STRING "sentry-native transport (custom disk transport)" FORCE)
  set(SENTRY_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(SENTRY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(SENTRY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(vendor/sentry-native "${CMAKE_BINARY_DIR}/sentry-native" EXCLUDE_FROM_ALL)

  # The Sentry Android SDK (main process) ships its own libsentry.so via prefab.
  # Ours would overwrite it in the jniLibs merge and pair their
  # libsentry-android.so with a mismatched sentry-native, so ours is renamed.
  # This relies on sentry-native setting no explicit SONAME (CMake then derives it
  # from OUTPUT_NAME); cmake/stage_android_jnilibs.cmake fails the build if the
  # staged lib's SONAME is not libpipette_sentry.so (when llvm-readelf is found).
  set_target_properties(sentry PROPERTIES OUTPUT_NAME pipette_sentry)
endif()

# pipette_bridge (shim + loader + crash reporter) + the .pipette_link_libs marker.
add_subdirectory(android/native)

if(PIPETTE_ENGINE_BACKEND STREQUAL "rust")
  include(cargo)

  # build.rs reads the marker at PIPETTE_LINK_LIBS_PATH and emits the rustc-link
  # directives.
  set(PIPETTE_ENGINE_SO
      "${CMAKE_SOURCE_DIR}/target/${PIPETTE_CARGO_TARGET}/release/libpipette_android.so")

  add_custom_target(pipette_android_cdylib ALL
    COMMAND ${CMAKE_COMMAND} -E env
        "PIPETTE_LINK_LIBS_PATH=${PIPETTE_LINK_LIBS_PATH_FILE}"
        "${PIPETTE_CARGO_COMMAND}" build --release -p pipette-android --target ${PIPETTE_CARGO_TARGET}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS pipette_bridge
    BYPRODUCTS "${PIPETTE_ENGINE_SO}"
    USES_TERMINAL
    VERBATIM
    COMMENT "cargo build pipette-android cdylib (links CMake llama.cpp + bridge)")
else()
  set(PIPETTE_ENGINE_SO "$<TARGET_FILE:pipette_android>")
endif()

# --- Stage jniLibs ------------------------------------------------------------
# Not in ALL. Building this target alone builds everything it stages, including
# the runtime-loaded CPU variants (via ggml's dependencies).
if(NOT DEFINED PIPETTE_JNILIBS_DIR)
  if(PIPETTE_ENGINE_BACKEND STREQUAL "rust")
    set(PIPETTE_JNILIBS_DIR
        "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/rustJniLibs/${ANDROID_ABI}")
  else()
    set(PIPETTE_JNILIBS_DIR
        "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/kotlinJniLibs/${ANDROID_ABI}")
  endif()
endif()
# Unstripped symbols for Sentry upload. A sibling of the jniLibs dir, not under it,
# so they are never packaged into the APK.
if(NOT DEFINED PIPETTE_SYMBOLS_DIR)
  if(PIPETTE_ENGINE_BACKEND STREQUAL "rust")
    set(PIPETTE_SYMBOLS_DIR
        "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/rustNativeSymbols/${ANDROID_ABI}")
  else()
    set(PIPETTE_SYMBOLS_DIR
        "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/kotlinNativeSymbols/${ANDROID_ABI}")
  endif()
endif()
set(PIPETTE_LIBCXX_SO
    "${ANDROID_TOOLCHAIN_ROOT}/sysroot/usr/lib/${PIPETTE_CARGO_TARGET}/libc++_shared.so")

# The genex fails without the `sentry` target, and the staging script treats a set
# SENTRY_SO as mandatory to stage.
set(PIPETTE_STAGE_SENTRY_ARG "")
if(PIPETTE_ENABLE_CRASH_REPORTING)
  set(PIPETTE_STAGE_SENTRY_ARG "-DSENTRY_SO=$<TARGET_FILE:sentry>")
endif()

add_custom_target(pipette_android_stage
  COMMAND ${CMAKE_COMMAND}
      "-DENGINE_SO=${PIPETTE_ENGINE_SO}"
      "-DLIBCXX_SO=${PIPETTE_LIBCXX_SO}"
      "-DGGML_BIN_DIR=$<TARGET_FILE_DIR:llama>"
      ${PIPETTE_STAGE_SENTRY_ARG}
      "-DDEST_DIR=${PIPETTE_JNILIBS_DIR}"
      "-DSYMBOLS_DIR=${PIPETTE_SYMBOLS_DIR}"
      "-DSTRIP=${ANDROID_TOOLCHAIN_ROOT}/bin/llvm-strip"
      "-DREADELF=${ANDROID_TOOLCHAIN_ROOT}/bin/llvm-readelf"
      -P "${CMAKE_SOURCE_DIR}/cmake/stage_android_jnilibs.cmake"
  VERBATIM
  COMMENT "Stage Android native libs into jniLibs/${ANDROID_ABI}")

if(PIPETTE_ENGINE_BACKEND STREQUAL "rust")
  add_dependencies(pipette_android_stage pipette_android_cdylib)
else()
  add_dependencies(pipette_android_stage pipette_android)
endif()
