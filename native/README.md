# native/

Shared native (C/C++) source for the mobile native crates. **Not a Rust
crate**: no `Cargo.toml`, not a workspace member. It lives at the repo root
(outside `crates/`, which is exclusively Cargo crates).

## `llama_shim.cpp`

The C-ABI bridge over the llama.cpp / mtmd C API (`ee_llama_*`, the mtmd
wrappers, and the Android performance-core threadpool hooks).

- **Android**: compiled by CMake as the `pipette_bridge` static library
  (`native/CMakeLists.txt`, together with
  `crates/pipette-android/native_loader.cpp`), which links the CMake-built
  llama.cpp shared libs and emits the `.pipette_link_libs.*` marker that
  `crates/pipette-android/build.rs` consumes. CMake is the single source of
  truth for include dirs, C++ flags, the link interface, and the 16 KB
  page-size arg. See the repo-root `CMakeLists.txt` and `build-rust-android.sh`.
- **iOS**: no longer uses this `native/` C layer or Rust at all. The iOS app
  compiles llama.cpp directly into the app via `ios/build-llama.sh` and drives
  it from a native Swift engine (`LlamaCppEngine`). This shim/kernel is now
  Android-only.

Keep it free of platform-specific code except behind `#if defined(__ANDROID__)`
/ Apple guards, since the same file feeds both builds.
