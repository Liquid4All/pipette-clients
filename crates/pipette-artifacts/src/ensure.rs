//! Find-or-fetch models and runtimes into the artifact stores.

use std::fs::File;
use std::path::{Path, PathBuf};

use pipette_plan_types::{Model, Runtime};

use crate::model::fetch::{declared_size_bytes, fetch_model};
use crate::model::{ModelArtifactStore, ModelStoreError};
use crate::progress::Reporter;
use crate::quota::{self, StoragePolicy, SweepPins};
use crate::runtime::docker::docker_image_pull;
use crate::runtime::llamacpp::install_llamacpp_archive;
use crate::runtime::mlx::install_mlx_runtime;
use crate::runtime::onnx_genai::install_onnx_genai_runtime;
use crate::runtime::openvino::install_openvino_runtime;
use crate::runtime::ovms::install_ovms_archive;
use crate::runtime::trt_edgellm::install_trt_edgellm_archive;
use crate::runtime::uv::install_uv_runtime;
use crate::runtime::{RuntimeArtifactStore, RuntimeManifest, RuntimeStoreError};
use crate::{ArtifactsContext, vault};

/// Find-or-fetch `declared` into `store` and return the **host-bound** [`Model`],
/// with a shared lock that keeps every process's sweep off the entry: hold it
/// while the model is in use. `None` where the store cannot be locked.
///
/// Under a [`StoragePolicy`] the order is fetch → publish → sweep → return, so
/// peak disk is the quota plus this artifact. A cache hit does not sweep.
pub fn ensure_model(
    ctx: &ArtifactsContext,
    store: &ModelArtifactStore,
    declared: &Model,
) -> Result<(Model, Option<File>), ModelStoreError> {
    let http = &ctx.download_http_client;
    let mut stored = store.find_intact(declared)?.is_some();
    // Named before the fetch: it is the phase of a run that can take minutes,
    // and a stall in it is only attributable if the log says it began.
    if stored {
        tracing::info!(model = %declared, "model in store");
    } else {
        tracing::info!(model = %declared, "model not in store; fetching");
    }
    let started = std::time::Instant::now();
    let mut policy = collecting_policy(ctx, stored);
    // One size lookup serves the quota pre-flight and the progress total, and is
    // skipped on a cache hit: it is a network round trip nobody would read.
    let size = if !stored && (policy.is_some() || ctx.progress.is_some()) {
        pre_fetch_size(ctx, declared)?
    } else {
        None
    };
    if let Some(policy) = policy {
        quota::refuse_if_oversize(&declared.to_string(), size, policy.quota_bytes)?;
    }
    let mut reporter = Reporter::new(ctx.progress.clone(), declared.to_string(), size);
    // Another process's sweep can remove the entry between the resolve and the
    // lock; resolving again fetches it back.
    let (manifest, in_use) = loop {
        let manifest = store.ensure(declared, |d, into| {
            fetch_model(
                http,
                &ctx.download_retry,
                ctx.vault.as_ref(),
                d,
                into,
                &mut reporter,
            )
            .map_err(Into::into)
        })?;
        if let Some(in_use) = store.hold(declared)? {
            break (manifest, in_use);
        }
        // The refetch publishes, so it sweeps like a miss.
        stored = false;
        policy = collecting_policy(ctx, stored);
    };
    if !stored {
        reporter.finish();
    }
    if let Some(policy) = policy {
        sweep(policy, policy.pins.clone().with_model(declared));
    }
    tracing::info!(model = %declared, elapsed = ?started.elapsed(), "model ready");
    Ok((manifest.bind_under(store.models_dir())?, in_use))
}

/// Find-or-fetch `declared` into `store` and return the **host-bound** [`Runtime`],
/// with the lock keeping its entry in place. Quota enforcement and locking match
/// [`ensure_model`].
pub fn ensure_runtime(
    ctx: &ArtifactsContext,
    store: &RuntimeArtifactStore,
    declared: &Runtime,
) -> Result<(Runtime, Option<File>), RuntimeStoreError> {
    let stored = store.find_intact(declared)?.is_some();
    if stored {
        tracing::info!(runtime = %declared.cli_ref(), "runtime in store");
    } else {
        tracing::info!(
            runtime = %declared.cli_ref(),
            "runtime not in store; installing"
        );
    }
    let started = std::time::Instant::now();
    let mut policy = collecting_policy(ctx, stored);
    if let Some(policy) = policy {
        quota::refuse_if_oversize(
            &declared.cli_ref(),
            runtime_size_bytes(declared),
            policy.quota_bytes,
        )?;
    }
    let (manifest, in_use) = loop {
        let manifest = store.ensure(declared, |d, blobs| install_runtime(ctx, d, blobs))?;
        if let Some(in_use) = store.hold(declared)? {
            break (manifest, in_use);
        }
        // The refetch publishes, so it sweeps like a miss.
        policy = collecting_policy(ctx, false);
    };
    if let Some(policy) = policy {
        sweep(policy, policy.pins.clone().with_runtime(declared));
    }
    tracing::info!(
        runtime = %declared.cli_ref(),
        elapsed = ?started.elapsed(),
        "runtime ready"
    );
    Ok((manifest.bind_under(store.runtimes_dir())?, in_use))
}

/// Enforcement is at collection time, so an artifact that is already stored
/// neither pre-flights nor sweeps.
fn collecting_policy(ctx: &ArtifactsContext, already_stored: bool) -> Option<&StoragePolicy> {
    ctx.storage.as_ref().filter(|_| !already_stored)
}

/// Bring both stores back under the cap after a publish, never evicting `pins`.
/// Every removal is logged, and nothing here can fail the resolve: a run does
/// not fail over disk bookkeeping.
fn sweep(policy: &StoragePolicy, pins: SweepPins) {
    let (plan, report) = quota::sweep(
        &policy.models_dir,
        &policy.runtimes_dir,
        policy.quota_bytes,
        pins,
    );
    if let Some(over) = plan.still_over_by_bytes {
        tracing::warn!(
            over_by_bytes = over,
            quota_bytes = policy.quota_bytes,
            "artifact storage is over its quota; every remaining entry is pinned, in use or frees nothing"
        );
    }
    report.removed.iter().for_each(|entry| {
        tracing::info!(
            entry = %entry.label,
            bytes = entry.size_bytes,
            "reclaimed to stay under the storage quota"
        );
    });
    report.failed.iter().for_each(|(entry, reason)| {
        tracing::warn!(path = %entry.path.display(), error = %reason, "could not reclaim");
    });
}

/// Bytes `declared` will occupy once fetched, `None` when unknowable.
///
/// [`declared_size_bytes`] errors only when the fetch cannot be planned, so the
/// error is kept: `None` would skip the quota pre-flight and run a fetch that
/// is going to fail anyway.
fn pre_fetch_size(
    ctx: &ArtifactsContext,
    declared: &Model,
) -> Result<Option<u64>, ModelStoreError> {
    declared_size_bytes(
        &ctx.download_http_client,
        &ctx.download_retry,
        ctx.vault.as_ref(),
        declared,
    )
    .map_err(|source| ModelStoreError::Fetch {
        model: declared.to_string(),
        source: source.into(),
    })
}

/// Bytes `declared` will occupy once installed, when knowable before the install.
///
/// `None` leaves the post-publish sweep as the only enforcement. A llama.cpp or
/// OVMS release is `None` on purpose: the only figure up front is the archive's
/// `Content-Length`, and the extracted install is several times that, so a
/// pre-flight on it would pass an artifact that cannot fit.
fn runtime_size_bytes(declared: &Runtime) -> Option<u64> {
    match declared {
        // The image lives in the docker daemon; the entry is manifest-only.
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => Some(0),
        // Archives as above; a uv/mlx venv is sized only once it is solved and
        // built; the rest have no installer to size.
        Runtime::LlamacppCliStockTools(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::AppleFoundation(_)
        | Runtime::AppleCoreAiMacosPipette(_) => None,
    }
}

/// What fetching `declared` would download, for a caller summing a progress
/// total before any fetch starts.
///
/// `Some(0)` for an artifact already in the store, so a sum stays whole; `None`
/// means the size could not be established.
pub fn model_download_size(
    ctx: &ArtifactsContext,
    store: &ModelArtifactStore,
    declared: &Model,
) -> Result<Option<u64>, ModelStoreError> {
    if store.find_intact(declared)?.is_some() {
        return Ok(Some(0));
    }
    pre_fetch_size(ctx, declared)
}

/// [`model_download_size`] for a runtime. `None` for every kind but docker
/// (whose entry is manifest-only): an archive's extracted size and a venv's are
/// not known before the install.
pub fn runtime_download_size(
    store: &RuntimeArtifactStore,
    declared: &Runtime,
) -> Result<Option<u64>, RuntimeStoreError> {
    if store.find_intact(declared)?.is_some() {
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

/// Install a runtime payload into store-owned `blobs_dir`.
fn install_runtime(
    ctx: &ArtifactsContext,
    declared: &Runtime,
    blobs_dir: &Path,
) -> anyhow::Result<()> {
    // The archives and the wheelhouse are the runtime installs that stream
    // bytes this process can count; uv and docker report their own progress
    // on stderr, and a venv solve has no byte total to report at all.
    let mut reporter = Reporter::new(
        ctx.progress.clone(),
        declared.cli_ref(),
        runtime_size_bytes(declared),
    );
    // Fetched before the venv it feeds, so a vault refusal costs no solve.
    let wheelhouse = vault::ensure_wheelhouse(
        ctx.vault.as_ref(),
        &ctx.download_http_client,
        &ctx.download_retry,
        declared,
        blobs_dir,
        &mut reporter,
    )?;
    let wheelhouse = wheelhouse.as_deref();
    match declared {
        Runtime::AppleCoreAiMacosPipette(_) => crate::runtime::coreai::install(blobs_dir),
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => {
            docker_image_pull(ctx.programs.docker.path()?, declared)
        }
        Runtime::UvVllm(_) | Runtime::UvSglang(_) => {
            install_uv_runtime(ctx.programs.uv.path()?, declared, blobs_dir, wheelhouse)
        }
        Runtime::MlxMacosPipette(_) => {
            install_mlx_runtime(ctx.programs.uv.path()?, declared, blobs_dir, wheelhouse)
        }
        Runtime::UvOpenvino(_) => {
            install_openvino_runtime(ctx.programs.uv.path()?, declared, blobs_dir, wheelhouse)
        }
        Runtime::UvOnnxGenai(_) => {
            install_onnx_genai_runtime(ctx.programs.uv.path()?, declared, blobs_dir, wheelhouse)
        }
        Runtime::LlamacppCliStockTools(_) => install_llamacpp_archive(
            &ctx.download_http_client,
            &ctx.download_retry,
            ctx.vault.as_ref(),
            declared,
            blobs_dir,
            &mut reporter,
        ),
        Runtime::Ovms(_) => install_ovms_archive(
            &ctx.download_http_client,
            &ctx.download_retry,
            ctx.vault.as_ref(),
            declared,
            blobs_dir,
            &mut reporter,
        ),
        Runtime::TrtEdgellm(_) => install_trt_edgellm_archive(
            &ctx.download_http_client,
            &ctx.download_retry,
            ctx.vault.as_ref(),
            declared,
            blobs_dir,
            &mut reporter,
        ),
        other @ (Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleFoundation(_)) => anyhow::bail!(
            "no built-in installer for runtime `{}`",
            other.headless_token()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pipette_http::HttpClient;
    use pipette_http::retry::RetryPolicy;
    use pipette_plan_types::{AbsolutePath, GgufText, GgufTextSource};
    use pipette_programs::ExternalPrograms;

    use super::*;
    use crate::entry::MANIFEST_NAME;
    use crate::model::ModelStorageKey;

    /// Programs that resolve to nothing, so whether this machine has `uv` does
    /// not decide what these tests do.
    fn test_ctx() -> anyhow::Result<ArtifactsContext> {
        Ok(ArtifactsContext::new(
            HttpClient::new("pipette-test")?,
            ExternalPrograms::at_missing_paths(),
        )
        .with_download_retry(RetryPolicy::IMMEDIATE))
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

    /// What the pre-flight measures for a local import, and what the published
    /// entry records as `blobs_bytes`. Quotas derive from it so the tests don't
    /// guess at the filesystem's block size: a quota of one payload leaves the
    /// store exactly full.
    fn declared_bytes(ctx: &ArtifactsContext, model: &Model) -> u64 {
        crate::model::fetch::declared_size_bytes(
            &ctx.download_http_client,
            &ctx.download_retry,
            ctx.vault.as_ref(),
            model,
        )
        .ok()
        .flatten()
        .unwrap_or(0)
    }

    fn vault_model() -> anyhow::Result<Model> {
        use crate::vault::test_support::{RESOURCE_ID, content_path};
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::VaultUrl {
                url: content_path(RESOURCE_ID)?,
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
        }))
    }

    /// A vault model is fetched as the context's device and published under
    /// its resource id; the next ensure is a cache hit that asks mgmt nothing.
    #[test]
    fn a_vault_model_is_ensured_as_the_device_and_cached() -> anyhow::Result<()> {
        use httpmock::MockServer;

        use crate::vault::test_support::{BODY, BODY_SHA, RESOURCE_ID, access, serve};

        let server = MockServer::start();
        let (record, content) = serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        let ctx = test_ctx()?.with_vault(access(&server)?);
        let declared = vault_model()?;

        let (bound, _) = ensure_model(&ctx, &store, &declared)?;
        let Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) = &bound
        else {
            anyhow::bail!("expected a bound gguf-text, got {bound}");
        };
        assert!(
            path.as_ref().ends_with(&format!("{RESOURCE_ID}.gguf")),
            "{path}"
        );
        assert_eq!(fs::read(path.as_ref())?, BODY);
        record.assert_calls(1);
        content.assert_calls(1);

        assert_eq!(ensure_model(&ctx, &store, &declared)?.0, bound);
        record.assert_calls(1);
        content.assert_calls(1);
        Ok(())
    }

    /// Without vault access the store is never touched, and the refusal names
    /// the registration the fetch needs.
    #[test]
    fn a_vault_model_without_access_is_refused_before_the_store() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        let declared = vault_model()?;

        let err = anyhow::Error::from(
            ensure_model(&test_ctx()?, &store, &declared)
                .err()
                .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?,
        );
        assert!(
            format!("{err:#}").contains("pipette auth register"),
            "{err:#}"
        );
        assert!(store.find(&declared)?.is_none());
        Ok(())
    }

    /// The quota pre-flight sizes a vault file off its record, so an artifact
    /// the whole quota cannot hold is refused before a byte is pulled.
    #[test]
    fn a_capped_context_sizes_a_vault_model_from_its_record() -> anyhow::Result<()> {
        use httpmock::MockServer;

        use crate::vault::test_support::{BODY, BODY_SHA, RESOURCE_ID, access, serve};

        let server = MockServer::start();
        let (_, content) = serve(&server, RESOURCE_ID, BODY_SHA, BODY);
        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        let ctx = capped_ctx(tmp.path(), BODY.len() as u64 - 1)?.with_vault(access(&server)?);

        let err = ensure_model(&ctx, &store, &vault_model()?)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected the quota to refuse"))?;
        assert!(matches!(err, ModelStoreError::Quota(_)), "{err}");
        content.assert_calls(0);
        Ok(())
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
        // Non-empty: the store is exactly full, so the overage that makes the
        // sweep act has to come from the garbage itself.
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

    /// The held lock stands for another process running the model; the sweep
    /// evicts the next least recently used in its place.
    #[test]
    fn a_sweep_leaves_a_model_in_use_until_it_is_released() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let running = local_model(tmp.path(), "a")?;
        let idle = local_model(tmp.path(), "b")?;
        let ctx = capped_ctx(tmp.path(), 2 * declared_bytes(&test_ctx()?, &running))?;

        let (_, in_use) = ensure_model(&ctx, &store, &running)?;
        ensure_model(&ctx, &store, &idle)?;
        ensure_model(&ctx, &store, &local_model(tmp.path(), "c")?)?;
        assert!(entry_of(&models_dir, &running)?.exists(), "evicted in use");
        assert!(!entry_of(&models_dir, &idle)?.exists(), "kept over quota");

        drop(in_use);
        ensure_model(&ctx, &store, &local_model(tmp.path(), "d")?)?;
        assert!(!entry_of(&models_dir, &running)?.exists());
        Ok(())
    }

    /// The pre-check treats a manifest that does not parse as "not stored"
    /// rather than failing before the store repairs the entry.
    #[test]
    fn a_zero_byte_manifest_is_refetched() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_dir = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_dir.clone());
        let model = local_model(tmp.path(), "a")?;
        let (bound, _) = ensure_model(&test_ctx()?, &store, &model)?;
        let manifest = entry_of(&models_dir, &model)?.join(MANIFEST_NAME);
        fs::write(&manifest, [0u8; 444])?;

        assert_eq!(ensure_model(&test_ctx()?, &store, &model)?.0, bound);
        assert!(store.find(&model)?.is_some(), "the manifest reads back");
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

        let orphan = models_dir.join("no-manifest");
        fs::create_dir_all(&orphan)?;
        ensure_model(&ctx, &store, &model)?;

        assert!(orphan.exists());
        Ok(())
    }

    /// A local copy moves without reporting bytes, so the landed model is
    /// reported whole, once; a cache hit reports nothing.
    #[test]
    fn a_fetched_model_is_reported_whole_once() -> anyhow::Result<()> {
        use std::sync::{Arc, Mutex};

        use crate::progress::{FetchProgress, ProgressSink};

        let tmp = tempfile::tempdir()?;
        let store = ModelArtifactStore::new(tmp.path().join("models"));
        let model = local_model(tmp.path(), "a")?;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&seen);
        let sink: ProgressSink = Arc::new(move |p: FetchProgress<'_>| {
            if let Ok(mut seen) = captured.lock() {
                seen.push((p.done_bytes, p.total_bytes));
            }
        });
        let ctx = test_ctx()?.with_progress(sink);

        ensure_model(&ctx, &store, &model)?;
        ensure_model(&ctx, &store, &model)?;

        let bytes = declared_bytes(&test_ctx()?, &model);
        let seen = seen.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(*seen, [(bytes, Some(bytes))]);
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

    /// A shell script stands in for `uv`, so these are Unix-only.
    #[cfg(unix)]
    mod venv_install {
        use std::os::unix::fs::PermissionsExt;

        use rstest::rstest;

        use pipette_plan_types::{
            NonEmptyString, OnnxExecutionProvider, UvOnnxGenai, UvOpenvino, UvPythonVersion,
            UvRuntimeSource, UvServerVersion,
        };
        use pipette_programs::ResolvedProgram;

        use super::*;

        /// `uv venv … <dir>` leaves the interpreter real uv would; every other
        /// subcommand succeeds.
        const FAKE_UV: &str = r#"#!/bin/sh
if [ "$1" = venv ]; then
  for dir; do :; done
  mkdir -p "$dir/bin" && touch "$dir/bin/python"
fi
"#;

        fn pip_source(requirements: &str) -> anyhow::Result<UvRuntimeSource> {
            Ok(UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new(requirements.to_owned())?,
                install_flags: None,
            })
        }

        fn openvino() -> anyhow::Result<Runtime> {
            Ok(Runtime::UvOpenvino(UvOpenvino {
                server_version: UvServerVersion::try_new("2026.4.1".to_owned())?,
                python_version: UvPythonVersion::try_new("3.11".to_owned())?,
                source: pip_source("openvino-genai==2026.4.1.0\n")?,
            }))
        }

        fn onnx_genai() -> anyhow::Result<Runtime> {
            Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
                server_version: UvServerVersion::try_new("0.16.0".to_owned())?,
                execution_provider: OnnxExecutionProvider::Cpu,
                python_version: UvPythonVersion::try_new("3.11".to_owned())?,
                source: pip_source("onnxruntime-genai==0.16.0\n")?,
            }))
        }

        #[rstest]
        #[case::openvino(openvino())]
        #[case::onnx_genai(onnx_genai())]
        fn a_uv_runtime_binds_to_the_venv_its_installer_created(
            #[case] declared: anyhow::Result<Runtime>,
        ) -> anyhow::Result<()> {
            let tmp = tempfile::tempdir()?;
            let uv = tmp.path().join("uv");
            fs::write(&uv, FAKE_UV)?;
            fs::set_permissions(&uv, fs::Permissions::from_mode(0o755))?;
            let ctx = ArtifactsContext::new(
                HttpClient::new("pipette-test")?,
                ExternalPrograms {
                    uv: ResolvedProgram::at(&uv),
                    ..ExternalPrograms::at_missing_paths()
                },
            );
            let store = RuntimeArtifactStore::new(tmp.path().join("runtimes"));

            let (bound, _) = ensure_runtime(&ctx, &store, &declared?)?;
            let Some(source) = bound.uv_source() else {
                anyhow::bail!("expected a uv runtime, got {bound}");
            };
            pipette_venv::require_bound_venv(source)?;
            Ok(())
        }
    }

    #[test]
    fn ensure_runtime_binds_core_ai_from_the_store() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().join("runtimes"));
        let declared = Runtime::AppleCoreAiMacosPipette(Default::default());
        assert_eq!(runtime_download_size(&store, &declared)?, None);
        store.ensure(&declared, |_, blobs| {
            std::fs::write(blobs.join("pipette-coreai-sidecar"), b"test")?;
            Ok(())
        })?;
        let (bound, _hold) = ensure_runtime(&test_ctx()?, &store, &declared)?;
        assert!(
            matches!(bound, Runtime::AppleCoreAiMacosPipette(rt) if matches!(rt.source, pipette_plan_types::AppleCoreAiMacosPipetteSource::AbsoluteDir { .. }))
        );
        assert_eq!(store.list()?.len(), 1);
        assert_eq!(runtime_download_size(&store, &declared)?, Some(0));
        Ok(())
    }
}
