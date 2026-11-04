use std::env;
use std::path::Path;

use anyhow::Context;

// Its own file so `src/lib.rs` can include it under `cfg(test)`: a build script
// cannot host tests.
#[path = "tag.rs"]
mod tag;

/// Relative to the package root, which cargo uses as a build script's CWD.
const VENDOR_DIR: &str = "../../vendor/llama.cpp";
/// `<tag> <full-sha>` recorded beside the pin.
const TAG_FILE: &str = "../../vendor/llama.cpp.tag";

// CMake owns the Android native build; this only turns the `.pipette_link_libs.*`
// marker it emits (format: android/native/CMakeLists.txt) into cargo directives.
// The per-variant `libggml-cpu-android_*.so` are not linked: native_loader.cpp
// dlopens them at runtime.
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
            "search" => println!("cargo:rustc-link-search=native={val}"),
            "staticlib" => println!("cargo:rustc-link-lib=static={val}"),
            "dylib" => println!("cargo:rustc-link-lib=dylib={val}"),
            "arg" => println!("cargo:rustc-link-arg={val}"),
            // Forces a relink when a linked artifact changes; without it a
            // rebuilt static bridge would not reach the cdylib.
            "watch" => println!("cargo:rerun-if-changed={val}"),
            other => anyhow::bail!("unknown marker key {other:?} in {marker} line {line:?}"),
        }
    }

    // `--short=9`, not bare `--short`: git's auto length grows with the object
    // db, so one commit would read as two build ids (dev vs fresh CI checkout).
    // `ios/build-llama.sh` pins it the same way.
    let vendor_dir = Path::new(VENDOR_DIR);
    let ggml_commit = git_out(vendor_dir, &["rev-parse", "--short=9", "HEAD"])
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=LLAMA_CPP_COMMIT={ggml_commit}");

    // Empty when the pin carries no tag. Why the tag is reported:
    // `Capabilities.flags` (Kotlin).
    println!("cargo:rustc-env=LLAMA_CPP_TAG={}", vendored_tag(vendor_dir));

    // Watch the pin too, or bumping the submodule alone leaves a stale tag
    // stamped in. Ask git for the gitdir: `.git/modules/...` is elsewhere in a
    // worktree.
    println!("cargo:rerun-if-changed={TAG_FILE}");
    if let Some(git_dir) = git_out(vendor_dir, &["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
    }

    Ok(())
}

/// Trimmed stdout of `git <args>` in `dir`, or `None` when git fails or prints
/// nothing (`rev-parse` prints an empty line on some failures).
fn git_out(dir: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    let trimmed = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!trimmed.is_empty()).then_some(trimmed)
}

/// The `bNNNN` tag on the vendored pin, or an empty string when there is none.
///
/// The submodule is a shallow clone without tag objects (and `git describe`
/// would report the nearest tag anyway), so as in `ios/build-llama.sh`:
/// `git tag --points-at HEAD` first, then `vendor/llama.cpp.tag`, whose
/// recorded sha makes staleness detectable offline.
fn vendored_tag(vendor_dir: &Path) -> String {
    let pointed = git_out(vendor_dir, &["tag", "--points-at", "HEAD"]).and_then(|out| {
        out.lines()
            .map(str::trim)
            .find(|line| tag::is_llamacpp_tag(line))
            .map(str::to_owned)
    });
    if let Some(tag) = pointed {
        return tag;
    }

    let Ok(recorded) = std::fs::read_to_string(TAG_FILE) else {
        return String::new();
    };
    let head_sha = git_out(vendor_dir, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    match tag::recorded_tag(&recorded, &head_sha) {
        tag::Recorded::Tag(tag) => tag.to_owned(),
        tag::Recorded::Malformed => {
            println!("cargo:warning={TAG_FILE} is malformed; reporting untagged");
            String::new()
        }
        tag::Recorded::NotATag(value) => {
            println!("cargo:warning={TAG_FILE}: {value:?} is not a tag (bNNNN); ignoring");
            String::new()
        }
        tag::Recorded::Stale { tag, sha } => {
            println!(
                "cargo:warning={TAG_FILE} records {tag} at {sha}, but the submodule is at \
                 {head_sha}; update it when bumping the pin. Reporting untagged."
            );
            String::new()
        }
    }
}
