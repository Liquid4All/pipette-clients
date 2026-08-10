# Helpers shared across toolchains / the root CMakeLists.
#
# Ported from inference_engine's cmake/toolchain-helpers.cmake, trimmed to what
# the Android build needs today. The iOS-specific helpers (macOS SDK root, iOS
# cached probe values, LP64 sizeof pins) land in Phase 2 with the Apple toolchain.

#------------------------------------------------------------------------------
# Derive the per-flavour suffix used in the .pipette_link_libs.<suffix> marker
# file that build.rs consumes. Must be a macro so the OUTVAR write lands in the
# caller's scope. Kept here (not split across toolchain files) because the
# default branch has to run even when no toolchain file is loaded (host builds).
#------------------------------------------------------------------------------
macro(compute_backend_libs_suffix OUTVAR)
  if(ANDROID)
    set(${OUTVAR} "android-${ANDROID_ABI}")
  elseif(APPLE AND DEFINED PIPETTE_APPLE_FLAVOUR)
    # Phase 2: aarch64-apple-ios / aarch64-apple-ios-sim.
    set(${OUTVAR} "${PIPETTE_APPLE_FLAVOUR}")
  else()
    set(${OUTVAR} "default")
  endif()
endmacro()
