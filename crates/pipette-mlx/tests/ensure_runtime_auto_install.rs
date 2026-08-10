//! End-to-end auto-provisioning test for the bundled-catalog path via the
//! shared runtime store (explicit artifacts ensure — same as CLI benchmarks).
//!
//! Marked `#[ignore]` because it shells out to `uv` to create a real
//! venv and download ~1 GB of mlx-lm + dependencies. Run explicitly:
//!
//!   cargo test -p pipette-mlx --test ensure_runtime_auto_install -- \
//!     --ignored --nocapture
//!
//! Requires `uv` on PATH and an Apple Silicon Mac (mlx-lm is arm64-only).

#![cfg(target_os = "macos")]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_artifacts::runtime::RuntimeArtifactStore;
use pipette_artifacts::{ensure_runtime, ArtifactsContext};
use pipette_http::HttpClient;
use pipette_mlx::catalog;
use pipette_plan_types::{Runtime, UvRuntimeSource};
use pipette_venv::resolve_uv;

struct TmpDir(PathBuf);

impl TmpDir {
    fn new(label: &str) -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "pipette-mlx-it-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        fs::create_dir_all(&dir).with_context(|| format!("create tmpdir {}", dir.display()))?;
        Ok(Self(dir))
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn test_ctx() -> anyhow::Result<ArtifactsContext> {
    Ok(ArtifactsContext::new(HttpClient::new("pipette-test")?))
}

fn mlx_python(bound: &Runtime) -> anyhow::Result<PathBuf> {
    let Runtime::MlxMacosPipette(rt) = bound else {
        anyhow::bail!("expected mlx runtime");
    };
    match &rt.source {
        UvRuntimeSource::AbsolutePreinstalled { dir } => {
            let python = Path::new(dir.as_ref()).join("bin").join("python");
            if !python.exists() {
                anyhow::bail!("MLX venv python missing at {}", python.display());
            }
            Ok(python)
        }
        other => anyhow::bail!("expected AbsolutePreinstalled after ensure, got {other:?}"),
    }
}

#[test]
#[ignore = "downloads ~1GB via uv; run with --ignored"]
fn auto_install_bundled_catalog_entry_end_to_end() -> anyhow::Result<()> {
    if resolve_uv(None).is_err() {
        eprintln!("skipping: `uv` not on PATH");
        return Ok(());
    }

    let tmp = TmpDir::new("auto-install")?;
    let data_dir = tmp.path();
    let catalog_ver = "0.31.3";
    let ctx = test_ctx()?;
    let declared = catalog::declared_from_catalog(catalog_ver)?;
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));

    let bound = ensure_runtime(&ctx, &store, &declared)?;
    let python = mlx_python(&bound)?;
    assert!(
        python.exists(),
        "venv python must exist after install: {python:?}"
    );
    let Runtime::MlxMacosPipette(runtime) = &bound else {
        anyhow::bail!("expected mlx runtime");
    };
    assert_eq!(runtime.version.as_ref(), catalog_ver);
    assert!(matches!(
        &runtime.source,
        UvRuntimeSource::AbsolutePreinstalled { .. }
    ));
    assert!(matches!(
        &declared,
        Runtime::MlxMacosPipette(rt)
            if matches!(
                &rt.source,
                UvRuntimeSource::PipRequirementsText { contents, .. }
                    if contents.as_ref().contains("mlx-lm==")
            )
    ));

    let listed = store.list()?;
    assert!(
        listed.iter().any(|m| matches!(
            &m.declared,
            Runtime::MlxMacosPipette(rt) if rt.version.as_ref() == catalog_ver
        )),
        "store list should include the installed MLX runtime"
    );

    let bound2 = ensure_runtime(&ctx, &store, &declared)?;
    assert_eq!(python, mlx_python(&bound2)?);
    let Runtime::MlxMacosPipette(runtime2) = bound2 else {
        anyhow::bail!("expected mlx runtime");
    };
    assert_eq!(runtime.version, runtime2.version);
    Ok(())
}

#[test]
#[ignore = "downloads ~1GB via uv; run with --ignored"]
fn reinstall_after_store_remove() -> anyhow::Result<()> {
    if resolve_uv(None).is_err() {
        eprintln!("skipping: `uv` not on PATH");
        return Ok(());
    }

    let tmp = TmpDir::new("reinstall")?;
    let data_dir = tmp.path();
    let catalog_ver = "0.31.3";
    let ctx = test_ctx()?;
    let declared = catalog::declared_from_catalog(catalog_ver)?;
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));

    let bound = ensure_runtime(&ctx, &store, &declared)?;
    assert!(mlx_python(&bound)?.exists());

    assert!(store.remove(&declared)?);

    let bound2 = ensure_runtime(&ctx, &store, &declared)?;
    assert!(mlx_python(&bound2)?.exists());
    let Runtime::MlxMacosPipette(restored) = bound2 else {
        anyhow::bail!("expected mlx runtime");
    };
    assert_eq!(restored.version.as_ref(), catalog_ver);
    Ok(())
}

#[test]
fn unknown_catalog_entry_errors_without_network() -> anyhow::Result<()> {
    let err = catalog::declared_from_catalog("not-a-real-catalog-entry")
        .err()
        .context("non-bundled name should error")?;
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not-a-real-catalog-entry"),
        "error should name the missing runtime: {msg}"
    );
    assert!(
        msg.contains("0.31.3"),
        "error should list bundled catalog entries for auto-install: {msg}"
    );
    Ok(())
}
