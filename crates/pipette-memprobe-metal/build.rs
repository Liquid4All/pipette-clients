use std::env;

fn main() -> anyhow::Result<()> {
    // Build the macOS Metal-memory shim. Gate on the *target* OS, not
    // the host — when cross-compiling (e.g. to Android from macOS),
    // `cfg(target_os = "macos")` would still evaluate true in the
    // build script because the build script runs on the host. Cargo
    // exposes the target's OS via `CARGO_CFG_TARGET_OS`.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    #[cfg(target_os = "macos")]
    if target_os == "macos" {
        macos_only::build_peakmtl_dylib()?;
        return Ok(());
    }
    // Non-macOS targets (or non-macOS hosts cross-compiling to anything):
    // emit a placeholder hash env var so `metal.rs`'s `env!` doesn't
    // fail. Cross-compiling *to* macOS from a non-macOS host is not
    // supported (the shim needs Apple frameworks, which require an
    // Apple host SDK).
    let _ = target_os;
    println!("cargo:rustc-env=PIPETTE_MEMPROBE_PEAKMTL_HASH=");

    Ok(())
}

// `cc`, `sha2`, and `hex` are macOS-only build-dependencies in
// Cargo.toml. The function that uses them must be cfg-gated to the
// macOS target *and* tied to the host platform that runs cargo: the
// `[target.'cfg(target_os = "macos")'.build-dependencies]` table only
// activates when the *target* is macOS, so we cannot reference these
// crates from code that compiles for a Linux/Windows target.
#[cfg(target_os = "macos")]
mod macos_only {
    use std::{env, fs, path::PathBuf};

    use anyhow::Context;

    pub(crate) fn build_peakmtl_dylib() -> anyhow::Result<()> {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
        let src = manifest_dir.join("peakmtl").join("peakmtl.m");
        println!("cargo:rerun-if-changed={}", src.display());

        let out_dir = PathBuf::from(env::var("OUT_DIR")?);
        let dylib = out_dir.join("peakmtl.dylib");

        // We use cc::Build for compiler discovery (xcrun-aware clang, correct
        // -arch / -mmacosx-version-min flags, SDK paths) but invoke the
        // resulting Tool ourselves with `-dynamiclib`, since `cc` only knows
        // how to produce static archives directly.
        let tool = cc::Build::new()
            .target(&env::var("TARGET")?)
            .host(&env::var("HOST")?)
            .opt_level(2)
            .cargo_metadata(false) // we don't link this into the Rust crate
            .get_compiler();

        let mut cmd = tool.to_command();
        cmd.arg("-dynamiclib")
            .arg("-framework")
            .arg("Metal")
            .arg("-framework")
            .arg("Foundation")
            .arg("-o")
            .arg(&dylib)
            .arg(&src);

        let status = cmd
            .status()
            .with_context(|| format!("failed to invoke {:?} for peakmtl.dylib", tool.path()))?;
        if !status.success() {
            anyhow::bail!(
                "{:?} failed building peakmtl.dylib (status {status}); cmd was {:?}",
                tool.path(),
                cmd
            );
        }

        // Compute a content hash for the freshly-built dylib and write it to
        // OUT_DIR so src/metal.rs can include it via env! and use it as a
        // cache-buster in the runtime extract path. Two builds with the same
        // bytes will produce the same hash; a tiny code edit that yields
        // identical byte length will produce a different hash. SHA-256
        // truncated to 16 hex chars (64 bits) is overkill for collision
        // resistance here but reuses a workspace dep instead of an
        // open-coded hash function.
        use sha2::{Digest, Sha256};
        let bytes = fs::read(&dylib)?;
        let digest = Sha256::digest(&bytes);
        let hash = &hex::encode(digest)[..16];
        let hash_path = out_dir.join("peakmtl.dylib.hash");
        fs::write(&hash_path, hash)?;
        println!("cargo:rustc-env=PIPETTE_MEMPROBE_PEAKMTL_HASH={hash}");
        Ok(())
    }
}
