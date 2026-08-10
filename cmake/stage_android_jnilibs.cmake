# Stage the Android native libs into jniLibs/<abi>/ with a fail-closed CPU-variant
# policy. Run as `cmake -P` from a post-build custom command. Ported verbatim in
# spirit from android/Pipette/build-rust-android.sh's stage step — the policy and
# guards are load-bearing, so keep them strict.
#
# Required -D args:
#   ENGINE_SO    : path to the built libpipette_android.so (cargo output)
#   LIBCXX_SO    : path to the NDK libc++_shared.so
#   GGML_BIN_DIR : dir holding libggml{,-base}.so libllama.so libmtmd.so + the
#                  per-variant libggml-cpu-android_*.so
#   DEST_DIR     : jniLibs/<abi> destination
#   SENTRY_SO    : path to libpipette_sentry.so. Set iff the engine was built with
#                  native crash capture (PIPETTE_ENABLE_CRASH_REPORTING=ON), in
#                  which case staging it is MANDATORY: the engine hard-links it
#                  (DT_NEEDED), so staging without it would ship an unloadable APK.
#                  Empty/unset means crash reporting was compiled out, so the lib
#                  does not exist and the engine carries no reference to it. The
#                  caller (root CMakeLists) owns that decision; this script only
#                  honours it, and stays fail-closed for the enabled case.
#   STRIP        : llvm-strip executable (optional; skipped if missing)
#   READELF      : llvm-readelf executable (optional; the sentry SONAME guard
#                  below is skipped if missing)
#   SYMBOLS_DIR  : dir to receive the UNSTRIPPED copies for Sentry symbol upload
#                  (optional; skipped if unset). The APK ships the stripped libs
#                  from DEST_DIR; Sentry matches these unstripped ones by build-id
#                  so `:benchmark` native crash backtraces symbolicate.
#
# Every ggml-cpu variant is classified against an explicit ship allowlist rather
# than staged-then-filtered, so the build FAILS rather than silently shipping or
# dropping anything if upstream ever renames or adds a variant.
#   SHIP_VARIANTS : the armv8.x and armv9.x backends we ship.
#   anything else : unknown -> hard error (confirm it loads, then add it).
#
# We ship the armv9 backends even though they produce incorrect LFM2A
# audio-encoder output on armv9 devices (Tensor G5 / Pixel 10): Pipette is a
# benchmark harness and measures every backend a device exposes. See
# docs/pipette-android/implementation.md for the upstream bug and full rationale.

# This runs via `cmake -P` (its own cmake process, no enclosing project), so it
# must set its own policy floor — without this, CMP0057 defaults to OLD and the
# `if(... IN_LIST ...)` below errors (seen on CI's cmake; local default masked it).
cmake_minimum_required(VERSION 3.25)

set(SHIP_VARIANTS "armv8.0_1;armv8.2_1;armv8.2_2;armv8.6_1;armv9.0_1;armv9.2_1;armv9.2_2")

# SENTRY_SO is deliberately NOT in this list: it is set only when the engine was built
# with native crash capture. See its entry in the header comment.
foreach(_required ENGINE_SO LIBCXX_SO GGML_BIN_DIR DEST_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "stage_android_jnilibs: -D${_required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${DEST_DIR}")
file(MAKE_DIRECTORY "${DEST_DIR}")

# Unstripped-symbol output for Sentry upload (M5). Cleaned + recreated so a
# rebuild never uploads a stale symbol file for a lib that no longer ships.
if(DEFINED SYMBOLS_DIR AND SYMBOLS_DIR)
  file(REMOVE_RECURSE "${SYMBOLS_DIR}")
  file(MAKE_DIRECTORY "${SYMBOLS_DIR}")
endif()

# stage(src): copy into DEST_DIR (+ the UNSTRIPPED copy into SYMBOLS_DIR) then
# strip --strip-unneeded the DEST copy (if STRIP given). The SYMBOLS_DIR copy is
# taken from the original `src`, which strip never touches, so it keeps its
# symbol table regardless of strip order.
function(stage src)
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "stage_android_jnilibs: expected lib missing: ${src}")
  endif()
  get_filename_component(_name "${src}" NAME)
  set(_dst "${DEST_DIR}/${_name}")
  file(COPY "${src}" DESTINATION "${DEST_DIR}")
  if(DEFINED SYMBOLS_DIR AND SYMBOLS_DIR)
    file(COPY "${src}" DESTINATION "${SYMBOLS_DIR}")
  endif()
  if(STRIP AND EXISTS "${STRIP}")
    execute_process(
      COMMAND "${STRIP}" --strip-unneeded "${_dst}"
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "stage_android_jnilibs: strip failed for ${_dst}")
    endif()
  endif()
endfunction()

# assert_soname(src expected): fail the build unless `src`'s DT_SONAME equals
# `expected`. Guards the sentry-native rename (root CMakeLists sets OUTPUT_NAME
# pipette_sentry): the collision-avoidance only holds if the DT_SONAME the linker
# embeds — and hence the engine's DT_NEEDED — is libpipette_sentry.so, NOT
# libsentry.so. CMake derives the SONAME from OUTPUT_NAME today, so this passes;
# if a future sentry-native submodule bump sets its own explicit SONAME the file
# would still be libpipette_sentry.so while its SONAME reverted to libsentry.so,
# silently reintroducing the collision with the Sentry Android SDK's libsentry.so.
# This catches that at build time instead of as an on-device Sentry NDK break.
function(assert_soname src expected)
  if(NOT (READELF AND EXISTS "${READELF}"))
    message(STATUS "stage_android_jnilibs: READELF unavailable — skipping "
                   "SONAME guard for ${expected} (rebuild with the NDK toolchain "
                   "to enable it)")
    return()
  endif()
  execute_process(
    COMMAND "${READELF}" -d "${src}"
    OUTPUT_VARIABLE _dyn
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "stage_android_jnilibs: readelf failed for ${src}")
  endif()
  if(NOT _dyn MATCHES "\\(SONAME\\)[^\n]*\\[([^]]+)\\]")
    message(FATAL_ERROR
      "stage_android_jnilibs: ${src} has no DT_SONAME — expected ${expected}. "
      "The sentry-native rename relies on a derived SONAME; investigate the "
      "submodule's link config.")
  endif()
  set(_soname "${CMAKE_MATCH_1}")
  if(NOT _soname STREQUAL expected)
    message(FATAL_ERROR
      "stage_android_jnilibs: DT_SONAME of ${src} is '${_soname}', expected "
      "'${expected}'. A sentry-native bump likely forced its own SONAME, which "
      "reintroduces the libsentry.so collision with the Sentry Android SDK. See "
      "the OUTPUT_NAME rename note in the root CMakeLists.txt.")
  endif()
endfunction()

# Core libs.
stage("${ENGINE_SO}")
stage("${LIBCXX_SO}")
foreach(_lib libggml-base.so libggml.so libllama.so libmtmd.so)
  stage("${GGML_BIN_DIR}/${_lib}")
endforeach()

# sentry-native (libpipette_sentry.so) — native crash capture for :benchmark.
#
# Staged iff the caller passed SENTRY_SO, which it does iff the engine was built with
# PIPETTE_ENABLE_CRASH_REPORTING=ON. In that case staging is MANDATORY, not
# best-effort: libpipette_android.so hard-links the lib (DT_NEEDED), so staging the
# engine without it would ship an APK that fails to load in BOTH processes. `stage()`
# therefore still hard-errors on a missing file, keeping that path fail-closed.
#
# When crash reporting was compiled out the lib does not exist and the engine carries
# no reference to it, so there is nothing to stage and nothing to guard.
if(DEFINED SENTRY_SO AND SENTRY_SO)
  stage("${SENTRY_SO}")
  # Guard the rename: its DT_SONAME must stay libpipette_sentry.so, else the
  # libsentry.so collision with the Sentry Android SDK silently returns.
  assert_soname("${SENTRY_SO}" "libpipette_sentry.so")
else()
  message(STATUS "stage_android_jnilibs: no SENTRY_SO, staging without native "
                 "crash capture (engine built with "
                 "PIPETTE_ENABLE_CRASH_REPORTING=OFF).")
endif()

# CPU-variant backends.
file(GLOB _variants "${GGML_BIN_DIR}/libggml-cpu-android_*.so")
set(_staged_variants 0)
foreach(_variant ${_variants})
  get_filename_component(_base "${_variant}" NAME)
  string(REGEX REPLACE "^libggml-cpu-android_(.+)\\.so$" "\\1" _tag "${_base}")
  if(_tag IN_LIST SHIP_VARIANTS)
    stage("${_variant}")
    math(EXPR _staged_variants "${_staged_variants} + 1")
  else()
    message(FATAL_ERROR
      "unknown ggml-cpu variant '${_base}' — not in SHIP_VARIANTS. Confirm it "
      "loads on-device, then add it to SHIP_VARIANTS in "
      "cmake/stage_android_jnilibs.cmake.")
  endif()
endforeach()

# Guard: at least one CPU variant must ship, or every on-device model load fails.
if(_staged_variants EQUAL 0)
  message(FATAL_ERROR
    "no CPU backend variant was staged — the APK would have no usable CPU "
    "backend and every model load would fail on-device.")
endif()

message(STATUS "Staged Android native libs in ${DEST_DIR} (${_staged_variants} CPU variant(s)).")
