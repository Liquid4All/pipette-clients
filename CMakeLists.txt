# pipette native build — CMake is the single source of truth (CMake-on-top).
#
# CMake configures the vendored llama.cpp, compiles the C-ABI shim + Android
# CPU-variant loader as `pipette_bridge`, emits a link-config marker that
# build.rs consumes, then drives `cargo` (via cmake/cargo.cmake's wrapper) to
# build the Rust cdylib and stages everything into jniLibs.
#
# Phase 1 = Android only. iOS (framework/XCFramework, Metal) lands in Phase 2.
#
# The GGML_*/LLAMA_* flags that must match the legacy build-rust-android.sh
# exactly (for code-identical kernels) live in CMakePresets.json so they are in
# the cache before llama.cpp's option() calls run.

cmake_minimum_required(VERSION 3.25)
project(pipette_native C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(toolchain-helpers)

if(NOT ANDROID)
  message(FATAL_ERROR
    "pipette_native currently supports only the Android build. "
    "Configure with: cmake --preset android-arm64-v8a")
endif()

compute_backend_libs_suffix(PIPETTE_LINK_LIBS_SUFFIX)
set(PIPETTE_LINK_LIBS_PATH_FILE
    "${CMAKE_BINARY_DIR}/.pipette_link_libs.${PIPETTE_LINK_LIBS_SUFFIX}")

# Emit a GNU build-id note in every native shared lib (M5). Sentry matches an
# uploaded debug file to the crashing image by this id; a deterministic build-id
# guarantees the shipped (stripped) lib and the uploaded (unstripped) copy carry
# the SAME id, rather than relying on the text-hash fallback matching. The note
# is SHF_ALLOC, so `llvm-strip --strip-unneeded` preserves it in the shipped lib.
# Applies to targets in this dir and all subdirectories added below (ggml, llama,
# mtmd, sentry); the Rust cdylib gets the same flag via native/CMakeLists' marker.
add_link_options(-Wl,--build-id=sha1)

# Vendored llama.cpp: llama, mtmd, ggml, ggml-base, common, and (with
# GGML_CPU_ALL_VARIANTS) the per-feature libggml-cpu-android_*.so backends.
# EXCLUDE_FROM_ALL so a default `cmake --build` does NOT build llama.cpp's tool
# executables (llama-bench/-quantize/etc.), which don't cross-compile for
# Android. The pieces we need still build: the cdylib links the bridge, which
# links llama/mtmd/ggml/ggml-base, and the CPU variants are wired as
# dependencies of `ggml` (add_dependencies(ggml ${backend}) in ggml/src), so
# building `ggml` brings them along.
add_subdirectory(vendor/llama.cpp "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# Native crash capture for `:benchmark` is opt-out. ON preserves the shipping
# behaviour; OFF builds the engine with no sentry-native at all, so a clone that
# never initialized the submodule (or simply does not want Sentry compiled in) can
# still produce a working APK. The rest of the app is unaffected either way: the
# JVM Sentry SDK is a normal Gradle dependency and already no-ops without a DSN.
#
# Defaults OFF. Whether crash reporting is wanted is decided one level up, by whether a
# Sentry DSN was supplied to the Gradle build (see android/Pipette/app/build.gradle.kts),
# which then passes an explicit value down through build-rust-android.sh. Defaulting ON
# here would mean a build with no DSN still compiles sentry-native, and still requires its
# submodule, for a reporter that could never report anything.
#
# Consequence worth knowing: a fork, or anyone doing native work, never needs
# vendor/sentry-native checked out at all. Silent loss of crash capture in OUR release
# builds is guarded separately at the Gradle layer, by verifyReleaseSentryDsn.
option(PIPETTE_ENABLE_CRASH_REPORTING
       "Build sentry-native crash capture into the :benchmark engine" OFF)

# Fail early and actionably when the flag is on but the submodule was never
# checked out. Without this the build dies deep inside the missing subdirectory
# with a message that names neither remedy.
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
  # sentry-native: in-process native crash capture for the `:benchmark` process.
  # `inproc` = an in-process signal handler (NOT crashpad, whose separate handler
  # binary + child process is the wrong model for a short-lived isolated process;
  # inproc is also sentry-native's Android default). `SENTRY_TRANSPORT=none`: we
  # install a custom transport that writes crash envelopes to disk (see
  # native/crash_reporter.cpp), so `:benchmark` does zero network — no libcurl, no
  # transport thread — and therefore doesn't perturb the memory benchmark. The
  # main process drains the envelope dir and uploads via its JVM Sentry SDK.
  set(SENTRY_BACKEND inproc CACHE STRING "sentry-native crash backend" FORCE)
  set(SENTRY_TRANSPORT none CACHE STRING "sentry-native transport (custom disk transport)" FORCE)
  set(SENTRY_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(SENTRY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(SENTRY_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(vendor/sentry-native "${CMAKE_BINARY_DIR}/sentry-native" EXCLUDE_FROM_ALL)

  # CRITICAL: rename our sentry-native shared lib so it does NOT collide with the
  # `libsentry.so` the Sentry Android SDK (io.sentry:sentry-android, used
  # by the MAIN process) already ships via prefab. Without this, our arm64
  # libsentry.so overwrites theirs in the jniLibs merge, pairing their
  # libsentry-android.so with a mismatched sentry-native version and risking a
  # main-process Sentry NDK break. Ours ships as libpipette_sentry.so alongside it.
  #
  # This rename only holds if sentry-native does NOT force its own DT_SONAME: CMake
  # derives the SONAME from OUTPUT_NAME, so the runtime linker resolves us as
  # libpipette_sentry.so rather than falling back to libsentry.so (which would load
  # the JVM SDK's copy and reintroduce the collision). Currently sentry-native sets
  # no explicit SONAME, so this works — empirically confirmed by the on-device
  # SIGSEGV -> envelope -> sent test. If a future submodule bump adds an explicit
  # SONAME/NO_SONAME to sentry's target, re-verify: check `readelf -d
  # libpipette_sentry.so | grep SONAME` reads libpipette_sentry.so, not libsentry.so.
  set_target_properties(sentry PROPERTIES OUTPUT_NAME pipette_sentry)
endif()

# pipette_bridge (shim + loader + crash reporter) + the .pipette_link_libs marker.
add_subdirectory(native)

include(cargo)

# --- Drive cargo: build the Rust cdylib, linking the CMake outputs -----------
# build.rs reads PIPETTE_LINK_LIBS_PATH (the marker) and emits the rustc-link
# directives. The cdylib links pipette_bridge (static) + the llama.cpp shared
# libs; the CPU-variant .so are runtime-loaded, not linked.
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

# --- Stage jniLibs ------------------------------------------------------------
# Explicit target (NOT in ALL): run AFTER a full `cmake --build` so the
# runtime-loaded CPU variants exist. build-rust-android.sh runs the two steps in
# order. Depends on the cdylib so a lone `--target pipette_android_stage` at
# least rebuilds the engine lib.
if(NOT DEFINED PIPETTE_JNILIBS_DIR)
  set(PIPETTE_JNILIBS_DIR
      "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/rustJniLibs/${ANDROID_ABI}")
endif()
# Unstripped native symbols for Sentry upload (M5). A sibling of the jniLibs dir
# (NOT under it, so the unstripped libs are never packaged into the APK).
if(NOT DEFINED PIPETTE_SYMBOLS_DIR)
  set(PIPETTE_SYMBOLS_DIR
      "${CMAKE_SOURCE_DIR}/android/Pipette/app/build/generated/rustNativeSymbols/${ANDROID_ABI}")
endif()
set(PIPETTE_LIBCXX_SO
    "${ANDROID_TOOLCHAIN_ROOT}/sysroot/usr/lib/${PIPETTE_CARGO_TARGET}/libc++_shared.so")

# SENTRY_SO is passed ONLY when crash reporting is built in: the genex would fail
# to evaluate without the target, and the staging script treats a set value as
# "the engine links this, so staging it is mandatory". Empty means "not built",
# which is what makes the script's fail-closed check conditional rather than gone.
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
add_dependencies(pipette_android_stage pipette_android_cdylib)
