# pipette_bridge: the C-ABI shim (shared with iOS) + the Android CPU-variant
# loader, compiled by CMake. Replaces the cc::Build in crates/pipette-android/
# build.rs. Built as a static lib that the Rust cdylib links; emits the
# .pipette_link_libs.<suffix> marker that build.rs consumes.

add_library(pipette_bridge STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/llama_shim.cpp"
  # Native crash capture (sentry-native inproc + disk transport) for :benchmark.
  "${CMAKE_CURRENT_SOURCE_DIR}/crash_reporter.cpp"
  # The Android CPU-variant loader lives next to the crate (Android-only).
  "${CMAKE_CURRENT_SOURCE_DIR}/native_loader.cpp")

set_target_properties(pipette_bridge PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

# Match the legacy cc::Build flags. -O3/-DNDEBUG come from Release too, but set
# them explicitly so the bridge is identical regardless of how it's configured.
# stb_image.h / mtmd headers trip -Wcast-qual (upstream suppresses it likewise).
target_compile_options(pipette_bridge PRIVATE -O3 -DNDEBUG -Wno-cast-qual)

# ggml/include + llama include arrive transitively from the targets. The shim
# also pulls llama.cpp's bundled nlohmann/json (vendor/) and the mtmd headers
# (tools/mtmd), which are PRIVATE to common/mtmd and so not on the public
# interface: declare them here. `log` is used by the loader's __android_log_print.
target_include_directories(pipette_bridge PRIVATE
  "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/vendor"
  "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/tools/mtmd")

# The vendored llama.cpp shared libs the cdylib links (single-sourced here so
# the bridge's link deps, the marker's dylib= lines, and the watch= relink
# triggers can't drift apart. `log` + `c++_shared` are NDK system libs (no CMake
# target to link as a dependency or to watch).
set(_PIPETTE_LLAMA_LIBS llama mtmd ggml ggml-base)
target_link_libraries(pipette_bridge PUBLIC ${_PIPETTE_LLAMA_LIBS} log)

# `sentry` is the sentry-native shared lib, shipped as libpipette_sentry.so (the
# root CMakeLists renames it via OUTPUT_NAME to avoid colliding with the JVM
# SDK's libsentry.so); crash_reporter.cpp calls its C API and the cdylib loads
# it at runtime like the llama.cpp libs. Linked only when crash reporting is built
# in (PIPETTE_ENABLE_CRASH_REPORTING); the compile definition is what switches
# crash_reporter.cpp between its real implementation and its no-op stub, so the
# `ee_crash_reporter_init` symbol exists either way and the Rust FFI declaration
# in crates/pipette-android keeps linking untouched.
if(PIPETTE_ENABLE_CRASH_REPORTING)
  target_link_libraries(pipette_bridge PUBLIC sentry)
  target_compile_definitions(pipette_bridge PRIVATE PIPETTE_CRASH_REPORTING_ENABLED)
endif()

# === C++ Engine Target (kotlinEngine) =========================================
if(PIPETTE_ENGINE_BACKEND STREQUAL "cpp")
  add_library(pipette_android SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/jni.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jni_crash_reporter.cpp")

  set_target_properties(pipette_android PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

  target_compile_options(pipette_android PRIVATE -O3 -DNDEBUG)

  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}/vendor/llama.cpp" rev-parse --short=9 HEAD
    OUTPUT_VARIABLE LLAMA_CPP_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT LLAMA_CPP_COMMIT)
    set(LLAMA_CPP_COMMIT "unknown")
  endif()

  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}/vendor/llama.cpp" tag --points-at HEAD
    OUTPUT_VARIABLE LLAMA_CPP_TAG_POINTED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  set(LLAMA_CPP_TAG "")
  if(LLAMA_CPP_TAG_POINTED MATCHES "b[0-9]+")
    set(LLAMA_CPP_TAG "${CMAKE_MATCH_0}")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/vendor/llama.cpp.tag")
    file(READ "${CMAKE_SOURCE_DIR}/vendor/llama.cpp.tag" TAG_FILE_CONTENT)
    string(STRIP "${TAG_FILE_CONTENT}" TAG_FILE_CONTENT)
    if(TAG_FILE_CONTENT MATCHES "^(b[0-9]+)[ \t]+([0-9a-fA-F]+)")
      set(RECORDED_TAG "${CMAKE_MATCH_1}")
      set(RECORDED_SHA "${CMAKE_MATCH_2}")
      execute_process(
        COMMAND git -C "${CMAKE_SOURCE_DIR}/vendor/llama.cpp" rev-parse HEAD
        OUTPUT_VARIABLE HEAD_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(HEAD_SHA STREQUAL RECORDED_SHA)
        set(LLAMA_CPP_TAG "${RECORDED_TAG}")
      endif()
    endif()
  endif()

  target_compile_definitions(pipette_android PRIVATE
    LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT}"
    LLAMA_CPP_TAG="${LLAMA_CPP_TAG}")

  target_link_libraries(pipette_android PRIVATE
    pipette_bridge
    log)

  target_link_options(pipette_android PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,-z,common-page-size=16384
    -Wl,--build-id=sha1)
endif()

# === Link-config marker =======================================================
if(PIPETTE_ENGINE_BACKEND STREQUAL "rust")
  # CMake is the single source of truth for the cdylib's link interface. build.rs
  # translates each line into a cargo:rustc-link-* directive (see its match arms
  # for the per-key mapping). The per-variant libggml-cpu-android_*.so are
  # intentionally absent: they are dlopen'd + scored at runtime by the loader,
  # not linked.
  #
  # The dylib= / watch= lines are derived from _PIPETTE_LLAMA_LIBS so the marker
  # and the link deps above stay in lockstep; $<TARGET_FILE...> genexes are
  # expanded by file(GENERATE).
  set(_PIPETTE_MARKER_DYLIBS "")
  set(_PIPETTE_MARKER_WATCH "watch=$<TARGET_FILE:pipette_bridge>\n")
  foreach(_lib IN LISTS _PIPETTE_LLAMA_LIBS)
    string(APPEND _PIPETTE_MARKER_DYLIBS "dylib=${_lib}\n")
    string(APPEND _PIPETTE_MARKER_WATCH "watch=$<TARGET_FILE:${_lib}>\n")
  endforeach()
  string(APPEND _PIPETTE_MARKER_DYLIBS "dylib=log\ndylib=c++_shared\n")

  # sentry-native, renamed to libpipette_sentry.so (OUTPUT_NAME in the root
  # CMakeLists) so it can't collide with the JVM SDK's libsentry.so. Linked as a
  # dylib and staged into jniLibs like the llama.cpp libs; its build dir differs,
  # so add its own search + watch entries. The `-l` name must match the renamed
  # file (pipette_sentry), not the CMake target name (sentry).
  #
  # All three lines are omitted when crash reporting is off, so the cdylib carries no
  # DT_NEEDED on a lib that was never built. Leaving the `dylib=` line in would fail
  # the cargo link; leaving `search=`/`watch=` in would fail the genex, since the
  # `sentry` target does not exist.
  set(_PIPETTE_MARKER_SENTRY_SEARCH "")
  if(PIPETTE_ENABLE_CRASH_REPORTING)
    string(APPEND _PIPETTE_MARKER_DYLIBS "dylib=pipette_sentry\n")
    string(APPEND _PIPETTE_MARKER_WATCH "watch=$<TARGET_FILE:sentry>\n")
    set(_PIPETTE_MARKER_SENTRY_SEARCH "search=$<TARGET_FILE_DIR:sentry>\n")
  endif()

  file(GENERATE
    OUTPUT "${PIPETTE_LINK_LIBS_PATH_FILE}"
    CONTENT "search=$<TARGET_FILE_DIR:pipette_bridge>
search=$<TARGET_FILE_DIR:llama>
${_PIPETTE_MARKER_SENTRY_SEARCH}staticlib=pipette_bridge
${_PIPETTE_MARKER_DYLIBS}arg=-Wl,-z,max-page-size=16384
arg=-Wl,-z,common-page-size=16384
arg=-Wl,--build-id=sha1
${_PIPETTE_MARKER_WATCH}")
endif()
