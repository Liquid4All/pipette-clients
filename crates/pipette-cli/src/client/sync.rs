use std::collections::BTreeMap;

use anyhow::Context;
use serde::Serialize;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    types::{BenchmarkSummary, JobResponse, RemoteBenchmark},
    AuthIdentity, ConditionalResponse, EntityTag, IfNoneMatch, MgmtClient,
};
use pipette_plan_types::benchmark::BenchmarkSource;
use pipette_plan_types::BenchmarkType;

use crate::benchmarks::{BenchmarkStore, RemoteSyncState};
use crate::error::{Error, Result};
use crate::results::{
    move_result_dir, BenchmarkJobMetric, BenchmarkResultLocation, BenchmarkScoredResult,
    ResultsStore,
};
use crate::score_validation::dedupe_completion_ids;

const BENCHMARK_DETAIL_CONCURRENCY_CAP: usize = 16;

/// Submit a single pending result to the management server and move it to synced.
pub fn submit_pending_result(
    store: &ResultsStore,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    result_id: &str,
    payload: &impl Serialize,
) -> Result<String> {
    use pipette_mgmt_client::types::SubmitResponse;
    let src = store.result_dir(BenchmarkResultLocation::RemotePending, result_id);
    let marker = src.join("job_id");
    let job_id = if marker.exists() {
        std::fs::read_to_string(&marker)
            .with_context(|| format!("reading job_id marker at {}", marker.display()))?
            .trim()
            .to_string()
    } else {
        // Eval payloads must carry unique completion ids — the scores
        // server returns 400 with no per-sample diagnostics otherwise
        // (see issue #95). Repair the payload locally so a stale or buggy
        // on-disk result still submits successfully; the kept entry for
        // each id is the first occurrence, matching the checkpoint load
        // semantics.
        let mut payload_value = serde_json::to_value(payload)
            .context("failed to serialize submission payload for dedupe")?;
        require_descriptors(result_id, &payload_value)?;
        if let Some(report) = dedupe_completion_ids(&mut payload_value)
            .with_context(|| format!("pre-flight repair failed for result {result_id}"))?
        {
            log::warn!("result {result_id}: {}", report.summary());
        }
        let response: SubmitResponse = client.submit_result(http, auth, &payload_value)?;
        std::fs::write(&marker, &response.job_id)
            .with_context(|| format!("writing job_id marker at {}", marker.display()))?;
        response.job_id
    };
    let dst = store.result_dir(BenchmarkResultLocation::RemoteSynced, &job_id);
    move_result_dir(&src, &dst).with_context(|| {
        format!(
            "submitted as job {} but failed to move {} to {}",
            job_id,
            src.display(),
            dst.display()
        )
    })?;
    Ok(job_id)
}

/// Reject a pending payload that predates the model/runtime descriptor format.
/// The sync path forwards the on-disk payload verbatim, so a result recorded by
/// a binary before descriptors existed would submit with the retired
/// `model_name`/`model_quant`/`runtime_name`/… fields and no descriptors — the
/// server rejects it with no local diagnostic. Fail fast with an actionable
/// message instead. Freshly-built payloads always carry both descriptors, so
/// this only fires on stale on-disk results.
fn require_descriptors(result_id: &str, payload: &serde_json::Value) -> Result<()> {
    let present = |key: &str| {
        payload
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| !s.is_empty())
    };
    if present("model_descriptor") && present("runtime_descriptor") {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "result {result_id} was recorded before the model/runtime descriptor \
         format and can't be submitted; re-run the benchmark or discard the \
         pending result"
    )
    .into())
}

pub fn pull_remote_benchmarks(
    store: &BenchmarkStore,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
) -> Result<usize> {
    let previous_sync = store.get_sync_state()?;
    let cached_summaries = store.list_remote_index()?;
    let has_cached_index = !cached_summaries.is_empty() || previous_sync.is_some();
    let previous_benchmarks_etag = has_cached_index
        .then(|| {
            previous_sync
                .as_ref()
                .and_then(|s| s.benchmarks_etag.clone())
        })
        .flatten();
    let previous_benchmark_etags = previous_sync
        .as_ref()
        .map(|state| state.benchmark_etags.clone())
        .unwrap_or_default();

    let list_if_none_match = previous_benchmarks_etag.as_ref().map(IfNoneMatch::from);
    let summaries_response =
        client.list_benchmarks_conditional(http, auth, list_if_none_match.as_ref())?;
    let (summaries, benchmarks_etag) = match summaries_response {
        ConditionalResponse::Modified { value, etag } => (value, etag),
        ConditionalResponse::NotModified { etag } => {
            if !has_cached_index {
                return Err(Error::Other(anyhow::anyhow!(
                    "server returned 304 but cached benchmark index is unavailable"
                )));
            }
            (cached_summaries, etag.or(previous_benchmarks_etag))
        }
    };

    // Prune before the detail fetch, and against the *list* ids rather than the kept set below: a
    // benchmark whose detail merely failed or 304'd this round is still in the catalog, so deleting
    // its cached detail would force a re-fetch every sync. Mirrors where iOS prunes.
    let catalog_ids: Vec<String> = summaries
        .iter()
        .map(|summary| summary.benchmark_id.clone())
        .collect();
    store.prune_remote_details(&catalog_ids)?;

    let (kept_summaries, benchmark_etags) = pull_remote_benchmark_details(
        store,
        client,
        http,
        auth,
        &summaries,
        &previous_benchmark_etags,
    )?;
    // Index only domain-convertible benchmarks (unsupported server shapes dropped).
    store.put_remote_index(&kept_summaries)?;
    let sync_state = RemoteSyncState {
        pulled_at: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .context("failed to format the sync timestamp")?,
        benchmark_count: kept_summaries.len(),
        benchmarks_etag,
        benchmark_etags,
    };
    store.put_sync_state(&sync_state)?;
    Ok(kept_summaries.len())
}

/// Fetch details, convert to [`pipette_plan_types::benchmark::BenchmarkDefinition`], cache successes.
/// Returns index rows + etags for benchmarks we can operate on; skips the rest.
fn pull_remote_benchmark_details(
    store: &BenchmarkStore,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    summaries: &[BenchmarkSummary],
    previous_benchmark_etags: &BTreeMap<String, EntityTag>,
) -> Result<(Vec<BenchmarkSummary>, BTreeMap<String, EntityTag>)> {
    let requests: Vec<_> = summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| {
            let previous_etag =
                cached_benchmark_etag(store, &summary.benchmark_id, previous_benchmark_etags);
            BenchmarkDetailRequest {
                index,
                benchmark_id: summary.benchmark_id.clone(),
                previous_etag,
            }
        })
        .collect();
    let mut responses = fetch_remote_benchmark_details(
        client,
        http,
        auth,
        requests,
        benchmark_detail_concurrency(),
    )?;
    responses.sort_by_key(|response| response.index);

    let mut kept = Vec::new();
    let mut benchmark_etags = BTreeMap::new();
    for response in responses {
        let summary = &summaries[response.index];
        match response.response {
            ConditionalResponse::Modified { value, etag } => {
                let raw_type = value.benchmark_type.clone();
                match crate::benchmarks::benchmark_definition_from_remote(value) {
                    Ok(def) => {
                        match pipette_plan_types::BenchmarkId::try_new(
                            def.benchmark_id().to_string(),
                        ) {
                            Ok(_id) => {
                                store.put(BenchmarkSource::Remote, &def)?;
                                if let Some(etag) = etag {
                                    benchmark_etags.insert(response.benchmark_id.clone(), etag);
                                }
                                kept.push(summary.clone());
                            }
                            Err(e) => {
                                log::warn!(
                                    "skipping remote benchmark `{}`: invalid id ({e})",
                                    response.benchmark_id,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        log_skipped_benchmark(&response.benchmark_id, &raw_type, &e);
                    }
                }
            }
            ConditionalResponse::NotModified { etag } => {
                let id =
                    pipette_plan_types::BenchmarkId::try_new(response.benchmark_id.clone()).ok();
                let cached = id.as_ref().is_some_and(|id| store.has_remote_detail(id));
                if cached {
                    if let Some(etag) = etag.or(response.previous_etag) {
                        benchmark_etags.insert(response.benchmark_id.clone(), etag);
                    }
                    kept.push(summary.clone());
                } else {
                    log::warn!(
                        "skipping remote benchmark `{}`: 304 without a domain cache entry",
                        response.benchmark_id,
                    );
                }
            }
        }
    }
    Ok((kept, benchmark_etags))
}

/// Report a benchmark the client could not convert, splitting the two reasons it can happen
/// (PIP-248).
///
/// An unrecognized `benchmark_type` is expected: a newer server catalog reaching an older client is
/// version skew, not a fault, and logging it as an error trains readers to ignore the log. A type
/// the client *does* know that still fails to convert means server and client disagree about a
/// shape both implement, which is a bug on one side and worth an error.
///
/// Membership is decided by `BenchmarkType::from_str`, so it cannot drift from the enum. Mirrors
/// iOS `BenchmarkSync.logSkip` / `BenchmarkDefinition.knownTypes`.
fn log_skipped_benchmark(benchmark_id: &str, raw_type: &str, error: &impl std::fmt::Display) {
    if raw_type
        .parse::<pipette_plan_types::BenchmarkType>()
        .is_ok()
    {
        log::error!(
            "skipping remote benchmark `{benchmark_id}` ({raw_type}): schema mismatch ({error})"
        );
    } else {
        log::info!(
            "skipping remote benchmark `{benchmark_id}`: unrecognized benchmark_type `{raw_type}`"
        );
    }
}

fn cached_benchmark_etag(
    store: &BenchmarkStore,
    benchmark_id: &str,
    previous_benchmark_etags: &BTreeMap<String, EntityTag>,
) -> Option<EntityTag> {
    let id = pipette_plan_types::BenchmarkId::try_new(benchmark_id.to_owned()).ok()?;
    if !store.has_remote_detail(&id) {
        return None;
    }
    previous_benchmark_etags.get(benchmark_id).cloned()
}

fn benchmark_detail_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .clamp(1, BENCHMARK_DETAIL_CONCURRENCY_CAP)
}

struct BenchmarkDetailRequest {
    index: usize,
    benchmark_id: String,
    previous_etag: Option<EntityTag>,
}

struct BenchmarkDetailResponse {
    index: usize,
    benchmark_id: String,
    previous_etag: Option<EntityTag>,
    response: ConditionalResponse<RemoteBenchmark>,
}

/// Fetch each benchmark's detail concurrently, capped at `concurrency`.
///
/// The async `buffer_unordered` fan-out this replaced is now a thread
/// fan-out: the transport is synchronous and `Send + Sync`, so each
/// request blocks its own thread and a chunk of `concurrency` requests
/// runs at once. Order is irrelevant — the caller re-sorts by `index`.
fn fetch_remote_benchmark_details(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    requests: Vec<BenchmarkDetailRequest>,
    concurrency: usize,
) -> Result<Vec<BenchmarkDetailResponse>> {
    let concurrency = concurrency.max(1);
    let mut responses = Vec::with_capacity(requests.len());
    for chunk in requests.chunks(concurrency) {
        let chunk_results: Vec<Result<BenchmarkDetailResponse>> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|request| {
                    scope.spawn(move || {
                        let detail_if_none_match =
                            request.previous_etag.as_ref().map(IfNoneMatch::from);
                        let response = client.get_benchmark_conditional(
                            http,
                            auth,
                            &request.benchmark_id,
                            detail_if_none_match.as_ref(),
                        )?;
                        Ok(BenchmarkDetailResponse {
                            index: request.index,
                            benchmark_id: request.benchmark_id.clone(),
                            previous_etag: request.previous_etag.clone(),
                            response,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        Error::Other(anyhow::anyhow!("benchmark detail thread panicked"))
                    })?
                })
                .collect()
        });
        for result in chunk_results {
            responses.push(result?);
        }
    }
    Ok(responses)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::PrefillThroughput;

    use super::*;
    // Tests use anyhow's `Result` for free `?` on std::io / setup errors,
    // not the crate `Result` the parent module works in.

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> anyhow::Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "pipette-sync-{prefix}-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&path)?;
            Ok(Self { path })
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn require_descriptors_rejects_the_pre_descriptor_shape() -> anyhow::Result<()> {
        // A fresh payload carries both descriptors — accepted.
        let fresh = serde_json::json!({
            "model_descriptor": "{\"type\":\"mlx\"}",
            "runtime_descriptor": "{\"type\":\"mlx_macos_pipette\"}",
        });
        require_descriptors("r1", &fresh)?;

        // The retired shape (old identity fields, no descriptors) and a partial
        // payload with empty-string descriptors are both rejected with an
        // actionable message.
        let rejected = [
            serde_json::json!({ "model_name": "org/repo", "runtime_name": "mlx-lm" }),
            serde_json::json!({ "model_descriptor": "", "runtime_descriptor": "" }),
            serde_json::json!({ "model_descriptor": "{\"type\":\"mlx\"}" }),
        ];
        for payload in &rejected {
            match require_descriptors("r1", payload) {
                Ok(()) => anyhow::bail!("expected rejection for {payload}"),
                Err(e) => assert!(
                    format!("{e:#}").contains("descriptor format"),
                    "unexpected error: {e:#}"
                ),
            }
        }
        Ok(())
    }

    #[test]
    fn cached_benchmark_etag_requires_valid_detail_cache() -> anyhow::Result<()> {
        use crate::benchmarks::BenchmarkStore;

        let temp_dir = TempDir::new("etag-cache")?;
        let store = BenchmarkStore::new(&temp_dir.path);
        let mut previous_etags = BTreeMap::new();
        previous_etags.insert("b1".to_string(), EntityTag::try_new("\"detail\"")?);

        assert_eq!(cached_benchmark_etag(&store, "b1", &previous_etags), None);

        // Detail present unlocks the prior etag.
        store.put(
            BenchmarkSource::Remote,
            &pipette_plan_types::benchmark::BenchmarkDefinition::PrefillThroughput(
                PrefillThroughput {
                    benchmark_id: "b1".into(),
                    parameter_prefill_tokens: 1,
                },
            ),
        )?;
        assert_eq!(
            cached_benchmark_etag(&store, "b1", &previous_etags)
                .as_ref()
                .map(EntityTag::as_str),
            Some("\"detail\"")
        );
        Ok(())
    }

    #[test]
    fn benchmark_detail_concurrency_uses_system_limit_with_cap() {
        let concurrency = benchmark_detail_concurrency();
        assert!((1..=BENCHMARK_DETAIL_CONCURRENCY_CAP).contains(&concurrency));
    }
}

/// How a synced job should be treated by the score-refresh step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreRefreshState {
    /// Not an eval benchmark — it never needs server-side scoring.
    NotEval,
    /// An eval whose score is already saved on disk.
    AlreadyScored,
    /// An eval still awaiting a score from the server.
    NeedsRefresh,
}

pub fn score_refresh_state(store: &ResultsStore, job_id: &str) -> ScoreRefreshState {
    let payload = match store.load_payload(BenchmarkResultLocation::RemoteSynced, job_id) {
        Ok(v) => v,
        Err(err) => {
            log::info!("job {job_id}: cannot load payload, skipping refresh: {err:#}");
            return ScoreRefreshState::NotEval;
        }
    };
    let benchmark_id = payload
        .get("benchmark_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !matches!(
        BenchmarkType::from_id(benchmark_id),
        Some(BenchmarkType::Eval)
    ) {
        return ScoreRefreshState::NotEval;
    }
    // A job keeps its score once the server has processed it. If the metrics
    // file is already on disk the eval is done, so skip the refresh and never
    // re-fetch already-scored results.
    if store.has_valid_metrics(BenchmarkResultLocation::RemoteSynced, job_id) {
        log::info!("job {job_id}: already scored, skipping refresh");
        return ScoreRefreshState::AlreadyScored;
    }
    ScoreRefreshState::NeedsRefresh
}

pub fn refresh_scored_job(
    store: &ResultsStore,
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
    job_id: &str,
) -> Result<bool> {
    log::info!("refreshing job {}", job_id);
    let job: JobResponse = client.get_job(http, auth, job_id)?;
    if job.status != "processed" {
        log::info!("job {} not processed yet (status={})", job_id, job.status);
        return Ok(false);
    }
    let scored = serde_json::to_value(&BenchmarkScoredResult {
        scored_at: job.scored_at.or_else(|| {
            Some(
                OffsetDateTime::now_utc()
                    .format(&Rfc3339)
                    .unwrap_or_default(),
            )
        }),
        metrics: job.metrics.map(
            |items: Vec<pipette_mgmt_client::types::BenchmarkJobMetric>| {
                items
                    .into_iter()
                    .map(|item| BenchmarkJobMetric {
                        metric: item.metric,
                        value: item.value,
                        unit: item.unit,
                    })
                    .collect()
            },
        ),
    })
    .map_err(pipette_ops::Error::SerializeJson)?;
    store.save_metrics(BenchmarkResultLocation::RemoteSynced, job_id, &scored)?;
    log::info!("job {} processed", job_id);
    Ok(true)
}
