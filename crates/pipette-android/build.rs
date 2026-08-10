use std::env;
use std::path::PathBuf;

use anyhow::Context;

// CMake is the single source of truth for the Android native build; this script
// only translates the `.pipette_link_libs.*` marker that the pipette_bridge
// CMake target emits into cargo:rustc-link directives. See native/CMakeLists.txt
// for the marker format.
//
// The per-variant `libggml-cpu-android_*.so` are NOT linked here — they are
// dlopen'd and scored at runtime by native_loader.cpp.
fn main() -> anyhow::Result<()> {
    let target = env::var("TARGET").unwrap_or_default();
    if !target.contains("linux-android") {
        return Ok(());
    }

    let marker = env::var("PIPETTE_LINK_LIBS_PATH").context(
        "PIPETTE_LINK_LIBS_PATH is not set. Android builds of pipette-android go through \
         CMake (android/Pipette/build-rust-android.sh, or `./gradlew :app:assembleDebug`), \
         which builds the llama.cpp shared libraries + the pipette_bridge shim/loader and \
         emits the link-config marker this script consumes. Run that rather than \
         `cargo build -p pipette-android` directly.",
    )?;
    println!("cargo:rerun-if-env-changed=PIPETTE_LINK_LIBS_PATH");
    println!("cargo:rerun-if-changed={marker}");

    let contents = std::fs::read_to_string(&marker)
        .with_context(|| format!("failed to read PIPETTE_LINK_LIBS_PATH ({marker})"))?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            anyhow::bail!("malformed marker line in {marker} (expected key=value): {line:?}");
        };
        match key {
            // `bin`/`native` dirs holding the shared libs + the static bridge.
            "search" => println!("cargo:rustc-link-search=native={val}"),
            // The bridge is static — its code is embedded into the cdylib.
            "staticlib" => println!("cargo:rustc-link-lib=static={val}"),
            // llama/mtmd/ggml/ggml-base (CMake) + log/c++_shared (NDK sysroot).
            "dylib" => println!("cargo:rustc-link-lib=dylib={val}"),
            // The 16 KB page-size alignment for the loadable .so (Android 15+).
            "arg" => println!("cargo:rustc-link-arg={val}"),
            // Force a relink when a linked artifact changes — critical for the
            // static bridge, whose object code the cdylib embeds at link time.
            "watch" => println!("cargo:rerun-if-changed={val}"),
            other => anyhow::bail!("unknown marker key {other:?} in {marker} line {line:?}"),
        }
    }

    // Stamp the vendored llama.cpp commit so Rust can surface it as the runtime
    // version. Relative path keeps cargo's CWD assumptions checkout-independent.
    let vendor_dir = PathBuf::from("../../vendor/llama.cpp");
    let ggml_commit = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&vendor_dir)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=LLAMA_CPP_COMMIT={ggml_commit}");

    Ok(())
}
