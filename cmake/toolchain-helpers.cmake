# Helpers for the root CMakeLists.

# Suffix of the .pipette_link_libs.<suffix> marker build.rs reads. A macro so
# OUTVAR lands in the caller's scope; lives here, not in a toolchain file,
# because host builds load no toolchain file.
macro(compute_backend_libs_suffix OUTVAR)
  if(ANDROID)
    set(${OUTVAR} "android-${ANDROID_ABI}")
  elseif(APPLE AND DEFINED PIPETTE_APPLE_FLAVOUR)
    # TODO: nothing sets PIPETTE_APPLE_FLAVOUR yet (no Apple toolchain file).
    set(${OUTVAR} "${PIPETTE_APPLE_FLAVOUR}")
  else()
    set(${OUTVAR} "default")
  endif()
endmacro()
