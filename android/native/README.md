# android/native/

Native source for the Android engine: C/C++ files, plus `benchmarks.rs`, the
Rust measurement kernel that `crates/pipette-android` includes by `#[path]`.
The directory is not a Rust crate: it has no `Cargo.toml` and is not a
workspace member.

| File | Role |
|---|---|
| `llama_shim.cpp` | The C-ABI bridge over the llama.cpp / mtmd C API (`ee_llama_*`, the mtmd wrappers, and the Android performance-core threadpool hooks) |
| `native_loader.cpp` | The Android CPU-variant loader |
| `crash_reporter.cpp` | Native crash capture through sentry-native; a no-op stub unless the build turns crash reporting on |
| `jni.cpp`, `jni_crash_reporter.cpp` | The JNI layer of the default `kotlinEngine` flavor |
| `benchmarks.rs` | The measurement kernel of the `rustEngine` flavor, compiled as part of `pipette-android` |
| `CMakeLists.txt` | The CMake targets below |

## How it is built

- **`pipette_bridge`**: CMake compiles `llama_shim.cpp`, `crash_reporter.cpp`
  and `native_loader.cpp` into this static library
  (`android/native/CMakeLists.txt`). It links the CMake-built llama.cpp shared
  libraries.
  - Under `rustEngine`, CMake also emits the `.pipette_link_libs.*` marker that
    `crates/pipette-android/build.rs` reads, and the Rust cdylib links the
    bridge.
  - Under `kotlinEngine`, CMake links the bridge with `jni.cpp` and
    `jni_crash_reporter.cpp` into `libpipette_android.so`.
- CMake is the single source of truth for include dirs, C++ flags, the link
  interface, and the 16 KB page-size argument. See the repo-root
  `CMakeLists.txt` and `android/Pipette/build-native-android.sh`.

## iOS

iOS does not use this C layer or Rust. The iOS app compiles llama.cpp directly
into the app through `ios/build-llama.sh` and drives it from native Swift
(`ios/Pipette/Pipette/Runtimes/Llama/`, entry point `LlamaExecute`). This
directory is Android-only.
