# Locate cargo/rustc and generate a per-preset `cargo_wrapper` shell script.
# The wrapper is the single place cargo is invoked; see cargo_wrapper.in's header
# for why (toolchain env export + forced --target).
#
# Ported from inference_engine's cmake/cargo.cmake (Unix path only — pipette has
# no Windows native build).

if(NOT CARGO_EXECUTABLE)
  find_program(CARGO_EXECUTABLE cargo)
  if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR "Failed to find `cargo` on PATH.")
  endif()
endif()
if(NOT RUSTC_EXECUTABLE)
  find_program(RUSTC_EXECUTABLE rustc)
  if(NOT RUSTC_EXECUTABLE)
    message(FATAL_ERROR "Failed to find `rustc` on PATH.")
  endif()
endif()

set(PIPETTE_CARGO_COMMAND "${CMAKE_BINARY_DIR}/cargo_wrapper")
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/cargo_wrapper.in"
  "${PIPETTE_CARGO_COMMAND}"
  @ONLY)
file(CHMOD "${PIPETTE_CARGO_COMMAND}"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

message(STATUS "pipette: generated cargo wrapper ${PIPETTE_CARGO_COMMAND} (target=${PIPETTE_CARGO_TARGET})")
