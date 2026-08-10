//! Find-or-fetch models and runtimes into the artifact stores.

use std::path::{Path, PathBuf};

use pipette_plan_types::{Model, Runtime};
use pipette_venv::resolve_uv;

use crate::model::fetch::{declared_size_bytes, fetch_model};
use crate::model::{ModelArtifactStore, ModelStoreError};
use crate::progress::Reporter;
use crate::quota::{self, StoragePolicy, SweepPins};
use crate::runtime::docker::docker_image_pull;
use crate::runtime::llamacpp::install_llamacpp_archive;
use crate::runtime::mlx::install_mlx_runtime;
use crate::runtime::openvino::install_openvino_runtime;
use crate::runtime::uv::install_uv_runtime;
use crate::runtime::{RuntimeArtifactStore, RuntimeManifest, RuntimeStoreError};
use crate::ArtifactsContext;

/// Find-or-fetch `declared` into `store` and return the **host-bound** [`Model`].
///
/// Manifests stay inside the store; callers get a launchable plan-types instance.
///
/// Under a [`StoragePolicy`] the order is fetch → publish → sweep → return, so
/// peak disk is the quota plus this artifact. A cache hit publishes nothing and
/// therefore does not sweep; enforcement is at collection time.
pub fn ensure_model(
    ctx: &ArtifactsContext,
    store: &ModelArtifactStore,
    declared: &Model,
) -> Result<Model, ModelStoreError> {
    let http = &ctx.download_http_client;
    let stored = store.find(declared)?.is_some();
    let policy = collecting_policy(ctx, stored);
    // One size lookup serves both readers: the quota pre-flight refuses on it,
    // and a progress bar needs the same number for a denominator. Skipped when
    // nothing will be fetched — a cache hit costs no bytes, and asking a repo how
    // big they would have been is a network round trip for a number nobody reads.
    let size = (!stored && (policy.is_some() || ctx.progress.is_some()))
        .then(|| declared_size_bytes(http, declared))
        .flatten();
    if let Some(policy) = policy {
        quota::refuse_if_oversize(&declared.to_string(), size, policy.quota_bytes)?;
    }
    let mut reporter = Reporter::new(ctx.progress.clone(), declared.to_string(), size);
    let manifest = store.ensure(declared, |d, into| {
        fetch_model(http, d, into, &mut reporter).map_err(Into::into)
    })?;
    if let Some(policy) = policy {
        sweep(policy, policy.pins.clone().with_model(declared));
    }
    manifest.bind_under(store.models_dir()).map_err(Into::into)
}

/// Find-or-fetch `declared` into `store` and return the **host-bound** [`Runtime`].
///
/// Manifests stay inside the store; callers get a launchable plan-types instance.
/// Quota enforcement matches [`ensure_model`].
pub fn ensure_runtime(
    ctx: &ArtifactsContext,
    store: &RuntimeArtifactStore,
    declared: &Runtime,
) -> Result<Runtime, RuntimeStoreError> {
    let policy = collecting_policy(ctx, store.find(declared)?.is_some());
    if let Some(policy) = policy {
        quota::refuse_if_oversize(
            &declared.cli_ref(),
            runtime_size_bytes(declared),
            policy.quota_bytes,
        )?;
    }
    let manifest = store.ensure(declared, |d, blobs| install_runtime(ctx, d, blobs))?;
    if let Some(policy) = policy {
        sweep(policy, policy.pins.clone().with_runtime(declared));
    }
    manifest
        .bind_under(store.runtimes_dir())
        .map_err(Into::into)
}

/// The policy to enforce on this resolve, if any. Enforcement is at collection
/// time, so an artifact that is already stored publishes nothing and neither
/// pre-flights nor sweeps.
fn collecting_policy(ctx: &ArtifactsContext, already_stored: bool) -> Option<&StoragePolicy> {
    ctx.storage.as_ref().filter(|_| !already_stored)
}

/// Bring both stores back under the cap after a publish, never evicting `pins`.
///
/// Every removal is logged: no silent deletes. Nothing here can fail the
/// resolve — a run does not fail over disk bookkeeping.
fn sweep(policy: &StoragePolicy, pins: SweepPins) {
    let survey = quota::survey(&policy.models_dir, &policy.runtimes_dir);
    let plan = quota::plan(&survey, policy.quota_bytes, &pins);
    if let Some(over) = plan.still_over_by_bytes {
        log::warn!(
            "artifact storage is {over} bytes over the {} byte quota; every remaining entry is \
             pinned or frees nothing",
            policy.quota_bytes
        );
    }
    if plan.evictions.is_empty() {
        return;
    }
    let report = quota::apply_sweep(&plan);
    report.removed.iter().for_each(|entry| {
        log::info!(
            "reclaimed {} ({} bytes) to stay under the storage quota",
            entry.label,
            entry.size_bytes
        );
    });
    report.failed.iter().for_each(|(entry, reason)| {
        log::warn!("could not reclaim {}: {reason}", entry.path.display());
    });
}

/// Bytes `declared` will occupy once installed, when knowable before the install.
///
/// `None` means unknowable, and the post-publish sweep is then the only
/// enforcement. A llama.cpp release is `None` on purpose: the only figure
/// available up front is the archive's `Content-Length`, and the extracted
/// install is several times that — pre-flighting on it would pass an artifact
/// that cannot fit, which is the eviction cascade the refusal exists to prevent.
fn runtime_size_bytes(declared: &Runtime) -> Option<u64> {
    match declared {
        // The image lives in the docker daemon; the entry is manifest-only.
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => Some(0),
        // A uv/mlx venv is sized only once it is solved and built; anything else
        // has no installer to size.
        _ => None,
    }
}

/// What fetching `declared` would download, for a caller sizing a set of them
/// before any starts — the total a progress renderer divides by.
///
/// `Some(0)` for an artifact already in the store: nothing will be fetched, and a
/// zero term keeps a sum whole where a `None` would make it unknowable. `None`
/// means the size could not be established, which is not the same answer.
pub fn model_download_size(
    ctx: &ArtifactsContext,
    store: &ModelArtifactStore,
    declared: &Model,
) -> Result<Option<u64>, ModelStoreError> {
    if store.find(declared)?.is_some() {
        return Ok(Some(0));
    }
    Ok(declared_size_bytes(&ctx.download_http_client, declared))
}

/// [`model_download_size`] for a runtime. Only the kinds this process downloads
/// itself can answer: a uv venv is sized once solved, and a docker image is
/// pulled by the daemon, which reports its own progress.
pub fn runtime_download_size(
    store: &RuntimeArtifactStore,
    declared: &Runtime,
) -> Result<Option<u64>, RuntimeStoreError> {
    if store.find(declared)?.is_some() {
        return Ok(Some(0));
    }
    Ok(runtime_size_bytes(declared))
}

/// Absolute path of a tool basename under an installed runtime (`python`,
/// `llama-server`, …). `runtimes_dir` must be the same root used for ensure.
pub fn runtime_tool(
    runtimes_dir: &Path,
    manifest: &RuntimeManifest,
    tool: &str,
) -> anyhow::Result<PathBuf> {
    let path = manifest.resolve_tool(runtimes_dir, tool)?;
    if !path.exists() {
        anyhow::bail!(
            "runtime tool `{tool}` missing at {} (install may be corrupt)",
            path.display()
        );
    }
    Ok(path)
}

/// Path to `uv`, resolved once into `ctx.uv_executable` on first use.
fn uv_bin(ctx: &ArtifactsContext) -> anyhow::Result<&Path> {
    if let Some(p) = ctx.uv_executable.get() {
        return Ok(p.as_path());
    }
    let resolved = resolve_uv(None)?;
    let _ = ctx.uv_executable.set(resolved);
    ctx.uv_executable
        .get()
        .map(PathBuf::as_path)
        .ok_or_else(|| anyhow::anyhow!("uv path missing after resolve"))
}

/// Path to docker CLI, resolved once into `ctx.docker_executable` on first use.
fn resolve_docker_executable(ctx: &ArtifactsContext) -> anyhow::Result<&Path> {
    if let Some(p) = ctx.docker_executable.get() {
        return Ok(p.as_path());
    }
    let resolved = pipette_subprocess::which("docker")?;
    let _ = ctx.docker_executable.set(resolved);
    ctx.docker_executable
        .get()
        .map(PathBuf::as_path)
        .ok_or_else(|| anyhow::anyhow!("docker path missing after resolve"))
}

/// Path to host Python, resolved once into `ctx.python_executable`.
///
/// Prefers an already-set path; otherwise `python3` then `python` on `PATH`.
pub fn resolve_python_executable(ctx: &ArtifactsContext) -> anyhow::Result<&Path> {
    if let Some(p) = ctx.python_executable.get() {
        return Ok(p.as_path());
    }
    let resolved = pipette_subprocess::which("python3").or_else(|_| {
        pipette_subprocess::which("python").map_err(|_| {
            anyhow::anyhow!(
                "python3/python not found on PATH; install Python or set \
                 ArtifactsContext.python_executable"
            )
        })
    })?;
    let _ = ctx.python_executable.set(resolved);
    ctx.python_executable
        .get()
        .map(PathBuf::as_path)
        .ok_or_else(|| anyhow::anyhow!("python path missing after resolve"))
}

/// Install a runtime payload into store-owned `blobs_dir` (exhaustive kind match).
fn install_runtime(
    ctx: &ArtifactsContext,
    declared: &Runtime,
    blobs_dir: &Path,
) -> anyhow::Result<()> {
    match declared {
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => {
            docker_image_pull(resolve_docker_executable(ctx)?, declared)
        }
        Runtime::UvVllm(_) | Runtime::UvSglang(_) => {
            install_uv_runtime(uv_bin(ctx)?, declared, blobs_dir)
        }
        Runtime::MlxMacosPipette(_) => install_mlx_runtime(uv_bin(ctx)?, declared, blobs_dir),
        Runtime::UvOpenvino(_) => install_openvino_runtime(uv_bin(ctx)?, declared, blobs_dir),
        Runtime::LlamacppCliStockTools(_) => {
            // The archive is the one runtime install that streams bytes this
            // process can count; uv and docker report their own progress on
            // stderr, and a venv solve has no byte total to report at all.
            let mut reporter = Reporter::new(
                ctx.progress.clone(),
                declared.cli_ref(),
                runtime_size_bytes(declared),
            );
            install_llamacpp_archive(
                &ctx.download_http_client,
                declared,
                blobs_dir,
                &mut reporter,
            )
        }
        other => anyhow::bail!(
            "no built-in installer for runtime `{}`",
            other.headless_token()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pipette_http::HttpClient;
    use pipette_plan_types::{AbsolutePath, GgufText, GgufTextSource};

    use super::*;
    use crate::model::ModelStorageKey;

    fn test_ctx() -> anyhow::Result<ArtifactsContext> {
        Ok(ArtifactsContext::new(HttpClient::new("pipette-test")?))
    }

    /// A local gguf model authored at `root/<name>.gguf`, so `ensure_model`
    /// imports it by copy and never touches the network.
    fn local_model(root: &Path, name: &str) -> anyhow::Result<Model> {
        let src = root.join(format!("{name}.gguf"));
        fs::write(&src, b"weights")?;
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new(src.to_string_lossy().into_owned())?,
            },
        }))
    }

    fn entry_of(models_dir: &Path, model: &Model) -> anyhow::Result<PathBuf> {
        Ok(models_dir.join(ModelStorageKey::of(model)?.relative_dir()))
    }

    fn capped_ctx(root: &Path, quota_bytes: u64) -> anyhow::Result<ArtifactsContext> {
        Ok(test_ctx()?.with_storage(crate::quota::StoragePolicy::new(
            quota_bytes,
            root.join("models"),
            root.join("runtimes"),
        )))
    }

    /// Exactly what the pre-flight will measure for a local import, and — since
    /// `blobs_bytes` records the payload alone — also what the published entry
    /// measures. Quotas are derived from it so the tests don't guess at the
    /// filesystem's block size: a quota of one payload admits the fetch and
    /// leaves the store exactly full, so a test that needs an *overage* has to
    /// put bytes somewhere else.
    fn declared_bytes(ctx: &ArtifactsContext, model: &Model) -> u64 {
        crate::model::fetch::declared_size_bytes(&ctx.download_http_client, model).unwrap_or(0)
    }

    #[test]
    fn an_uncapped_context_never_sweeps() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let orphan = models_dir.join("no-manifest");
        fs::create_dir_all(&orphan)?;
        let store = ModelArtifactStore::new(models_dir.clone());

        ensure_model(&test_ctx()?, &store, &local_model(tmp.path(), "a")?)?;

        assert!(orphan.exists(), "an uncapped store reclaims nothing");
        Ok(())
    }

    #[test]
    fn ensure_sweeps_after_publish_and_keeps_the_freshly_fetched_entry() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let orphan = models_dir.join("no-manifest");
        fs::create_dir_all(&orphan)?;
        // Non-empty: a quota of one payload leaves the store exactly full, so the
        // overage that makes the sweep act has to come from the garbage itself —
        // and reclaiming an empty dir would free nothing anyway.
        fs::write(orphan.join("stale.bin"), [0u8; 8192])?;
        let store = ModelArtifactStore::new(models_dir.clone());
        let model = local_model(tmp.path(), "a")?;
        let ctx = capped_ctx(tmp.path(), declared_bytes(&test_ctx()?, &model))?;

        ensure_model(&ctx, &store, &model)?;

        assert!(!orphan.exists(), "garbage is reclaimed first");
        assert!(
            entry_of(&models_dir, &model)?.exists(),
            "the entry just fetched is pinned"
        );
        Ok(())
    }

    #[test]
    fn a_second_publish_evicts_the_least_recently_used_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let first = local_model(tmp.path(), "a")?;
        let second = local_model(tmp.path(), "b")?;
        let ctx = capped_ctx(tmp.path(), declared_bytes(&test_ctx()?, &first))?;

        ensure_model(&ctx, &store, &first)?;
        ensure_model(&ctx, &store, &second)?;

        assert!(!entry_of(&models_dir, &first)?.exists());
        assert!(entry_of(&models_dir, &second)?.exists());
        Ok(())
    }

    #[test]
    fn a_cache_hit_does_not_sweep() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let model = local_model(tmp.path(), "a")?;
        let ctx = capped_ctx(tmp.path(), declared_bytes(&test_ctx()?, &model))?;
        ensure_model(&ctx, &store, &model)?;

        // Enforcement is at collection time, so garbage that appears after the
        // publish survives a resolve that fetches nothing.
        let orphan = models_dir.join("no-manifest");
        fs::create_dir_all(&orphan)?;
        ensure_model(&ctx, &store, &model)?;

        assert!(orphan.exists());
        Ok(())
    }

    #[test]
    fn an_artifact_larger_than_the_whole_quota_is_refused_before_fetching() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let model = local_model(tmp.path(), "a")?;
        let one_byte_short = declared_bytes(&test_ctx()?, &model).saturating_sub(1);
        let ctx = capped_ctx(tmp.path(), one_byte_short)?;

        let err = ensure_model(&ctx, &store, &model)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a model over the whole quota must be refused"))?;

        assert!(matches!(err, ModelStoreError::Quota(_)));
        assert!(!entry_of(&models_dir, &model)?.exists());
        Ok(())
    }

    #[test]
    fn runtime_install_rejects_unpullable() -> anyhow::Result<()> {
        let ctx = test_ctx()?;
        let err = install_runtime(
            &ctx,
            &Runtime::AppleFoundation(Default::default()),
            Path::new("/tmp"),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected error"))?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no built-in") || msg.contains("apple") || msg.contains("installer"),
            "{msg}"
        );
        Ok(())
    }

    #[test]
    fn resolve_python_executable_prefers_preseeded_path() -> anyhow::Result<()> {
        let ctx = test_ctx()?;
        let seeded = PathBuf::from("/preseeded/python");
        ctx.python_executable
            .set(seeded.clone())
            .map_err(|_| anyhow::anyhow!("set python_executable"))?;
        assert_eq!(resolve_python_executable(&ctx)?, seeded.as_path());
        Ok(())
    }
}
