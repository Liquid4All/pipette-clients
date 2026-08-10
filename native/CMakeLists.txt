# pipette_bridge — the C-ABI shim (shared with iOS) + the Android CPU-variant
# loader, compiled by CMake. Replaces the cc::Build in crates/pipette-android/
# build.rs. Built as a static lib that the Rust cdylib links; emits the
# .pipette_link_libs.<suffix> marker that build.rs consumes.

add_library(pipette_bridge STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/llama_shim.cpp"
  # Native crash capture (sentry-native inproc + disk transport) for :benchmark.
  "${CMAKE_CURRENT_SOURCE_DIR}/crash_reporter.cpp"
  # The Android CPU-variant loader lives next to the crate (Android-only).
  "${CMAKE_SOURCE_DIR}/crates/pipette-android/native_loader.cpp")

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
# interface — declare them here. `log` is used by the loader's __android_log_print.
target_include_directories(pipette_bridge PRIVATE
  "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/vendor"
  "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/tools/mtmd")

# The vendored llama.cpp shared libs the cdylib links — single-sourced here so
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

# --- Link-config marker -------------------------------------------------------
# CMake is the single source of truth for the cdylib's link interface. build.rs
# translates each line into a cargo:rustc-link-* directive (see its match arms
# for the per-key mapping). The per-variant libggml-cpu-android_*.so are
# intentionally absent — they are dlopen'd + scored at runtime by the loader,
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
