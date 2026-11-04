use std::env;

fn main() -> anyhow::Result<()> {
    // A build script's own `cfg` is the host's, so the target OS comes from
    // `CARGO_CFG_TARGET_OS` (a macOS host building for Android must skip the shim).
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    #[cfg(target_os = "macos")]
    if target_os == "macos" {
        macos_only::build_peakmtl_dylib()?;
        return Ok(());
    }
    // An empty hash so `metal.rs`'s `env!` compiles. Building for macOS from
    // another host is unsupported: the shim needs the Apple SDK.
    let _ = target_os;
    println!("cargo:rustc-env=PIPETTE_MEMPROBE_PEAKMTL_HASH=");

    Ok(())
}

// `cc`, `sha2` and `hex` are macOS-only build-dependencies, so their user is
// gated the same way.
#[cfg(target_os = "macos")]
mod macos_only {
    use std::{env, fs, path::PathBuf};

    use anyhow::Context;
    use sha2::{Digest, Sha256};

    pub(crate) fn build_peakmtl_dylib() -> anyhow::Result<()> {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
        let src = manifest_dir.join("peakmtl").join("peakmtl.m");
        println!("cargo:rerun-if-changed={}", src.display());

        let out_dir = PathBuf::from(env::var("OUT_DIR")?);
        let dylib = out_dir.join("peakmtl.dylib");

        // `cc` finds the compiler and its arch, deployment-target and SDK flags,
        // but only builds static archives, so the dylib link is run by hand.
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
            .arg(&src)
            .stdout(std::io::stderr());

        #[expect(
            clippy::disallowed_methods,
            reason = "cargo reads a build script's stdout for directives, so the compiler's goes to stderr"
        )]
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

        // Keys `metal.rs`'s extract path, so an edit that keeps the byte length
        // still lands at a new path.
        let bytes = fs::read(&dylib)?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let hash = hex::encode(&digest[..8]);
        let hash_path = out_dir.join("peakmtl.dylib.hash");
        fs::write(&hash_path, &hash)?;
        println!("cargo:rustc-env=PIPETTE_MEMPROBE_PEAKMTL_HASH={hash}");
        Ok(())
    }
}
