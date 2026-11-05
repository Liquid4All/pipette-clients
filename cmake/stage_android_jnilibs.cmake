# Stage the Android native libs into jniLibs/<abi>/, fail-closed. Run as
# `cmake -P` by the root CMakeLists' pipette_android_stage target.
#
# -D args:
#   ENGINE_SO, LIBCXX_SO : libpipette_android.so and the NDK libc++_shared.so
#   GGML_BIN_DIR : llama.cpp libs plus the per-variant libggml-cpu-android_*.so
#   DEST_DIR     : jniLibs/<abi>
#   SENTRY_SO    : libpipette_sentry.so; set iff PIPETTE_ENABLE_CRASH_REPORTING=ON.
#                  Then it is mandatory: the engine has it as DT_NEEDED.
#   STRIP        : llvm-strip (optional)
#   READELF      : llvm-readelf (optional; without it the SONAME guard is skipped)
#   SYMBOLS_DIR  : receives unstripped copies for Sentry symbol upload (optional)
#
# ggml-cpu variants are checked against the SHIP_VARIANTS allowlist so an
# upstream rename or new variant fails the build instead of shipping or dropping
# silently. The armv9 variants ship despite bad LFM2A audio-encoder output on
# some armv9 devices; see docs/pipette-android/build.md.

# `cmake -P` has no project, so without this CMP0057 is OLD and IN_LIST errors.
cmake_minimum_required(VERSION 3.25)

set(SHIP_VARIANTS "armv8.0_1;armv8.2_1;armv8.2_2;armv8.6_1;armv9.0_1;armv9.2_1;armv9.2_2")

# SENTRY_SO is optional; see the header.
foreach(_required ENGINE_SO LIBCXX_SO GGML_BIN_DIR DEST_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "stage_android_jnilibs: -D${_required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${DEST_DIR}")
file(MAKE_DIRECTORY "${DEST_DIR}")

# Recreated so a rebuild never uploads symbols for a lib that no longer ships.
if(DEFINED SYMBOLS_DIR AND SYMBOLS_DIR)
  file(REMOVE_RECURSE "${SYMBOLS_DIR}")
  file(MAKE_DIRECTORY "${SYMBOLS_DIR}")
endif()

# Only the DEST_DIR copy is stripped; SYMBOLS_DIR gets the untouched `src`.
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

# Guards the sentry-native OUTPUT_NAME rename in the root CMakeLists: if a
# submodule bump sets an explicit SONAME, the file keeps its new name but the
# engine's DT_NEEDED reverts to libsentry.so and collides with the Sentry
# Android SDK's lib. Catch that here rather than on device.
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

stage("${ENGINE_SO}")
stage("${LIBCXX_SO}")
foreach(_lib libggml-base.so libggml.so libllama.so libmtmd.so)
  stage("${GGML_BIN_DIR}/${_lib}")
endforeach()

if(DEFINED SENTRY_SO AND SENTRY_SO)
  stage("${SENTRY_SO}")
  assert_soname("${SENTRY_SO}" "libpipette_sentry.so")
else()
  message(STATUS "stage_android_jnilibs: no SENTRY_SO, staging without native "
                 "crash capture (engine built with "
                 "PIPETTE_ENABLE_CRASH_REPORTING=OFF).")
endif()

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

if(_staged_variants EQUAL 0)
  message(FATAL_ERROR
    "no CPU backend variant was staged — the APK would have no usable CPU "
    "backend and every model load would fail on-device.")
endif()

message(STATUS "Staged Android native libs in ${DEST_DIR} (${_staged_variants} CPU variant(s)).")
