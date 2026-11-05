//! Integration tests for the uv runtime store: install, remove, and the cases
//! that never reach uv.
//!
//! Linux-only. The install test is ignored because it needs `uv` on PATH and
//! network access to PyPI. Run with
//! `cargo test -p pipette-artifacts --test uv_runtime_install -- --ignored --nocapture`.

#![cfg(target_os = "linux")]

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;

use pipette_artifacts::runtime::RuntimeArtifactStore;
use pipette_artifacts::{ArtifactsContext, ensure_runtime};
use pipette_http::HttpClient;
use pipette_plan_types::{UvBuild, UvPythonVersion, UvServerVersion};
use pipette_programs::ExternalPrograms;

struct TmpDir(PathBuf);

impl TmpDir {
    fn new(label: &str) -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "pipette-artifacts-uv-it-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4(),
        ));
        fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "needs uv + network; run with --ignored"]
fn uv_install_remove_removes_runtime_dir() -> anyhow::Result<()> {
    let programs = ExternalPrograms::from_path();
    if !programs.uv.is_resolved() {
        eprintln!("skipping: uv not on PATH");
        return Ok(());
    }
    let work = TmpDir::new("install-remove")?;
    let data_dir = work.path();
    let declared = pipette_plan_types::Runtime::UvSglang(pipette_plan_types::UvSglang {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: pipette_plan_types::UvRuntimeSource::PipRequirementsText {
            contents: pipette_plan_types::NonEmptyString::try_new("# empty\n".to_owned())?,
            install_flags: None,
        },
    });
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?, programs);
    ensure_runtime(&ctx, &store, &declared)?;
    let manifest = store
        .find(&declared)?
        .context("store has no manifest for the runtime it just installed")?;
    let entry = store.entry_dir_for(&manifest)?;
    assert!(entry.exists());
    assert!(pipette_venv::venv_python(&store.install_dir_for(&manifest)?).exists());

    assert!(store.remove(&declared)?);
    assert!(!entry.exists());
    Ok(())
}

#[test]
fn ensure_runtime_returns_existing_without_calling_uv() -> anyhow::Result<()> {
    use time::OffsetDateTime;

    use pipette_artifacts::runtime::RuntimeManifest;
    use pipette_artifacts::runtime::RuntimeStorageKey;
    use pipette_plan_types::{NonEmptyString, Runtime, UvRuntimeSource};

    let work = TmpDir::new("idempotency")?;
    let data_dir = work.path();
    let declared = pipette_plan_types::Runtime::UvVllm(pipette_plan_types::UvVllm {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("# empty\n".to_owned())?,
            install_flags: None,
        },
    });
    let key = RuntimeStorageKey::of(&declared)?;
    let entry = data_dir.join("runtimes").join(key.as_str());
    fs::create_dir_all(entry.join("blobs"))?;
    // Hand-built entry with an empty `blobs/`, so the recorded payload size is 0.
    let manifest = RuntimeManifest::new(declared.clone(), OffsetDateTime::now_utc(), 0)?;
    fs::write(
        entry.join("manifest.toml"),
        toml::to_string_pretty(&manifest)?,
    )?;

    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?, ExternalPrograms::from_path());
    let (Runtime::UvVllm(bound), _) = ensure_runtime(&ctx, &store, &declared)? else {
        anyhow::bail!("expected the declared uv_vllm variant back");
    };
    let UvRuntimeSource::AbsolutePreinstalled { dir } = bound.source else {
        anyhow::bail!("ensure must bind the runtime to an absolute install dir");
    };
    assert!(
        Path::new(dir.as_ref()).starts_with(&entry),
        "bound install dir {} is outside the seeded entry {}",
        dir.as_ref(),
        entry.display()
    );
    Ok(())
}

#[test]
fn ensure_runtime_rejects_preinstalled() -> anyhow::Result<()> {
    use pipette_plan_types::{RelativePath, UvRuntimeSource};

    let work = TmpDir::new("preinstalled")?;
    let data_dir = work.path();
    let declared = pipette_plan_types::Runtime::UvVllm(pipette_plan_types::UvVllm {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: UvRuntimeSource::RelativePreinstalled {
            dir: RelativePath::try_new("venv".to_owned())?,
        },
    });
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?, ExternalPrograms::from_path());
    let Err(err) = ensure_runtime(&ctx, &store, &declared) else {
        anyhow::bail!("preinstalled must not ensure");
    };
    let msg = format!("{err:#}");
    // Storage key rejects RelativePreinstalled as NotStorable before fetch runs.
    assert!(
        msg.contains("no storage key")
            || msg.contains("NotStorable")
            || msg.contains("not pullable")
            || msg.contains("effective-only"),
        "got: {msg}"
    );
    Ok(())
}
