use anyhow::Context;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_http::HttpClient;
use pipette_mgmt_client::{
    ClientIdentity, ConditionalResponse, EntityTag, IfNoneMatch, JobStatus, ServerUrl, client_api,
    types::{BenchmarkSummary, JobResponse, RemoteBenchmark},
};
use pipette_plan_types::BenchmarkType;

use crate::benchmarks::{BenchmarkStore, RemoteSyncState};
use crate::error::{Error, Result};
use crate::results::{
    BenchmarkJobMetric, BenchmarkResultLocation, BenchmarkScoredResult, ResultsStore, check_job_id,
    move_result_dir, recorded_benchmark_type,
};
use crate::score_validation::dedupe_completion_ids;

const BENCHMARK_DETAIL_CONCURRENCY_CAP: usize = 16;

/// Submit a single pending result to the management server and move it to synced.
///
/// `None` when another process holds the result (a `benchmarks run --sync`
/// racing a `sync`) or has already moved it: posting it again would file a
/// second job for one run.
pub fn submit_pending_result(
    store: &ResultsStore,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    result_id: &str,
) -> Result<Option<String>> {
    use pipette_plan_types::JobAccepted;
    let src = store.result_dir(BenchmarkResultLocation::RemotePending, result_id);
    // Beside the dir, not in it, so the move never renames an open file.
    let lock_path = src.with_file_name(format!("{result_id}.lock"));
    let lock = std::fs::File::create(&lock_path)
        .with_context(|| format!("creating {}", lock_path.display()))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
        // A filesystem without locks: submit unlocked.
        Err(std::fs::TryLockError::Error(e)) if e.kind() == std::io::ErrorKind::Unsupported => {}
        Err(std::fs::TryLockError::Error(e)) => {
            return Err(anyhow::Error::new(e)
                .context(format!("locking {}", lock_path.display()))
                .into());
        }
    }
    // Every holder rechecks the source, so the lock file can go once it has moved.
    if !src.exists() {
        let _ = std::fs::remove_file(&lock_path);
        return Ok(None);
    }
    let marker = src.join("job_id");
    let job_id = if marker.exists() {
        let job_id = std::fs::read_to_string(&marker)
            .with_context(|| format!("reading job_id marker at {}", marker.display()))?;
        check_job_id(job_id.trim())?.to_string()
    } else {
        // Read under the lock: a result another process just moved is skipped above.
        let mut payload_value =
            store.load_payload(BenchmarkResultLocation::RemotePending, result_id)?;
        // The scores server answers duplicate completion ids with a bare 400,
        // so repair the payload locally and a stale on-disk result still submits.
        require_descriptors(result_id, &payload_value)?;
        if let Some(report) = dedupe_completion_ids(&mut payload_value)
            .with_context(|| format!("pre-flight repair failed for result {result_id}"))?
        {
            tracing::warn!(
                %result_id,
                summary = report.summary().as_str(),
                "repaired duplicate completion ids before submitting"
            );
        }
        let response: JobAccepted = client_api::submit_result(server, http, auth, &payload_value)?;
        check_job_id(&response.job_id)?;
        // Through a sibling, so a kill mid-write cannot leave an empty marker.
        let tmp = marker.with_extension("tmp");
        std::fs::write(&tmp, &response.job_id)
            .and_then(|()| std::fs::rename(&tmp, &marker))
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
    let _ = std::fs::remove_file(&lock_path);
    Ok(Some(job_id))
}

/// Reject an on-disk payload recorded before model/runtime descriptors existed:
/// it is forwarded verbatim and the server would refuse it with no local
/// diagnostic.
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

/// Pull the catalog into `store`. The routes are public, so an unregistered
/// workspace can do this much of a `sync`; a registered one signs so the server
/// can include what it keeps to this device's principal.
///
/// Returns the number of rows whose detail is fetched
/// ([`RemoteSyncState::benchmark_count`]).
pub fn pull_remote_benchmarks(
    store: &BenchmarkStore,
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
) -> Result<usize> {
    // Only a cache from *this* server is revalidated: `sync --server-url` can
    // aim an unregistered workspace at another host, whose 304 to the old
    // ETags would keep a catalog it never served. A mismatch drops the list
    // and detail ETags, not the cache: the pull below refills and prunes it.
    let previous_sync = store
        .get_sync_state()?
        .filter(|state| state.server_url.as_ref() == Some(server));
    let cached_rows = store.list_remote_index()?;
    let has_cached_index = !cached_rows.is_empty() || previous_sync.is_some();
    let previous_benchmarks_etag = has_cached_index
        .then(|| {
            previous_sync
                .as_ref()
                .and_then(|s| s.benchmarks_etag.clone())
        })
        .flatten();
    let revalidate_details = previous_sync.is_some();

    let list_if_none_match = previous_benchmarks_etag.as_ref().map(IfNoneMatch::from);
    let summaries_response =
        client_api::list_benchmarks_conditional(server, http, auth, list_if_none_match.as_ref())?;
    let (rows, benchmarks_etag) = match summaries_response {
        ConditionalResponse::Modified { value, etag } => (value, etag),
        ConditionalResponse::NotModified { etag } => {
            if !has_cached_index {
                return Err(Error::Other(anyhow::anyhow!(
                    "server returned 304 but cached benchmark index is unavailable"
                )));
            }
            (cached_rows, etag.or(previous_benchmarks_etag))
        }
    };

    let fetchable: Vec<&BenchmarkSummary> = rows.iter().filter(|row| fetches_detail(row)).collect();
    pull_remote_benchmark_details(store, server, http, auth, &fetchable, revalidate_details)?;
    // Every listed id (see `prune_remote_details`), and only after the detail
    // fetch, so an all-fetches-failed abort above leaves the store untouched.
    let catalog_ids: Vec<String> = rows.iter().map(|row| row.benchmark_id.clone()).collect();
    store.prune_remote_details(&catalog_ids)?;
    store.put_remote_index(&rows)?;
    let sync_state = RemoteSyncState {
        pulled_at: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .context("failed to format the sync timestamp")?,
        benchmark_count: fetchable.len(),
        server_url: Some(server.clone()),
        benchmarks_etag,
    };
    store.put_sync_state(&sync_state)?;
    Ok(fetchable.len())
}

/// A type this build knows and an id it accepts. A row lacks what only the detail carries (a
/// corpus URL, eval samples), so the rest is left to the fetched body's conversion.
fn fetches_detail(row: &BenchmarkSummary) -> bool {
    if let Err(e) = row.benchmark_type.parse::<BenchmarkType>() {
        log_skipped_benchmark(&row.benchmark_id, &row.benchmark_type, &e);
        return false;
    }
    if let Err(e) = pipette_plan_types::BenchmarkId::try_new(row.benchmark_id.clone()) {
        log_skipped_benchmark(&row.benchmark_id, &row.benchmark_type, &e);
        return false;
    }
    true
}

/// Caches each detail that converts, as the server sent it; a cached one is revalidated
/// against its ETag only when `revalidate`.
fn pull_remote_benchmark_details(
    store: &BenchmarkStore,
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
    rows: &[&BenchmarkSummary],
    revalidate: bool,
) -> Result<()> {
    let requests: Vec<_> = rows
        .iter()
        .map(|row| BenchmarkDetailRequest {
            previous_etag: revalidate
                .then(|| cached_benchmark_etag(store, &row.benchmark_id))
                .flatten(),
            benchmark_id: row.benchmark_id.clone(),
        })
        .collect();
    let responses = fetch_remote_benchmark_details(
        server,
        http,
        auth,
        requests,
        benchmark_detail_concurrency(),
    )?;

    responses
        .into_iter()
        .try_for_each(|response| -> Result<()> {
            let Some(conditional) = response.response else {
                // Fetch failed: the cached detail and its ETag stay as they
                // were, and the id stays in the list, so the prune keeps them.
                return Ok(());
            };
            match conditional {
                ConditionalResponse::Modified { value, etag } => {
                    // A definition holds a checked `BenchmarkId`, so an id the grammar
                    // refuses fails here, as a conversion.
                    let raw_type = value.benchmark_type.clone();
                    match crate::benchmarks::ConvertedRemote::try_from(value) {
                        Ok(converted) => store.put_remote(&converted, etag.as_ref())?,
                        Err(e) => {
                            log_skipped_benchmark(&response.benchmark_id, &raw_type, &e);
                        }
                    }
                }
                ConditionalResponse::NotModified { .. } => {
                    let id =
                        pipette_plan_types::BenchmarkId::try_new(response.benchmark_id.clone())
                            .ok();
                    let cached = id.as_ref().is_some_and(|id| store.has_remote_detail(id));
                    if !cached {
                        // If-None-Match goes out only for a cached detail, but
                        // one deleted mid-sync has no ETag either, so the next
                        // sync re-requests it unconditionally.
                        tracing::warn!(
                            benchmark_id = response.benchmark_id.as_str(),
                            "remote benchmark: 304 without a cached detail; \
                             will re-fetch on the next sync"
                        );
                    }
                }
            }
            Ok(())
        })
}

/// An unrecognized `benchmark_type` is version skew (a newer catalog reaching an older client),
/// so it logs at info; a known type that fails to convert is a bug on one side and logs an error.
///
/// The id and type are unchecked server text, so they go in as `&str` fields, which a text line
/// quotes and escapes.
fn log_skipped_benchmark(benchmark_id: &str, raw_type: &str, error: &impl std::fmt::Display) {
    if raw_type
        .parse::<pipette_plan_types::BenchmarkType>()
        .is_ok()
    {
        tracing::error!(
            benchmark_id,
            benchmark_type = raw_type,
            %error,
            "skipping remote benchmark: schema mismatch"
        );
    } else {
        tracing::info!(
            benchmark_id,
            benchmark_type = raw_type,
            "skipping remote benchmark: unrecognized benchmark_type"
        );
    }
}

/// The cached detail's ETag, whether a pull or a claim cached it; an older build's
/// `remote/<id>.json` never counts.
fn cached_benchmark_etag(store: &BenchmarkStore, benchmark_id: &str) -> Option<EntityTag> {
    let id = pipette_plan_types::BenchmarkId::try_new(benchmark_id.to_owned()).ok()?;
    store.remote_detail_etag(&id)
}

fn benchmark_detail_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
        .clamp(1, BENCHMARK_DETAIL_CONCURRENCY_CAP)
}

struct BenchmarkDetailRequest {
    benchmark_id: String,
    previous_etag: Option<EntityTag>,
}

struct BenchmarkDetailResponse {
    benchmark_id: String,
    /// `None` when the detail fetch failed.
    response: Option<ConditionalResponse<RemoteBenchmark>>,
}

/// Fetch each benchmark's detail on its own thread, `concurrency` at a time.
/// Responses come back in request order.
fn fetch_remote_benchmark_details(
    server: &ServerUrl,
    http: &HttpClient,
    auth: Option<&ClientIdentity>,
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
                        let response = client_api::get_benchmark_conditional(
                            server,
                            http,
                            auth,
                            &request.benchmark_id,
                            detail_if_none_match.as_ref(),
                        )?;
                        Ok(BenchmarkDetailResponse {
                            benchmark_id: request.benchmark_id.clone(),
                            response: Some(response),
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
        // One failed detail fetch must not abort the whole pull. The failed id
        // comes back with `response: None`; the list still names it, so the
        // next sync asks again and the prune keeps its cached detail.
        // `chunk_results` is in `chunk` order.
        responses.extend(chunk.iter().zip(chunk_results).map(|(request, result)| {
            result.unwrap_or_else(|e| {
                tracing::warn!(
                    benchmark_id = request.benchmark_id.as_str(),
                    error = %e,
                    "benchmark detail fetch failed (will retry next pull)"
                );
                BenchmarkDetailResponse {
                    benchmark_id: request.benchmark_id.clone(),
                    response: None,
                }
            })
        }));
    }
    // Every fetch failing means the server is down: abort before the caller
    // rewrites the index and sync state, so the cached catalog survives.
    let total = responses.len();
    if total > 0 && responses.iter().all(|r| r.response.is_none()) {
        return Err(Error::Other(anyhow::anyhow!(
            "all {total} benchmark detail fetches failed; keeping the cached catalog untouched"
        )));
    }
    Ok(responses)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{BenchmarkId, BenchmarkSource};

    use super::*;
    use crate::client::worker::protocol::tests::test_auth;
    use crate::test_support::{
        IMG80, cache_remote, cache_remote_with_etag, img80_wire, logged_text, remote_detail_path,
        remote_state_path, write_older_build_bookkeeping, write_public_2026_08_1_img80_cache,
    };

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
        let fresh = serde_json::json!({
            "model_descriptor": "{\"type\":\"mlx\"}",
            "runtime_descriptor": "{\"type\":\"mlx_macos_pipette\"}",
        });
        require_descriptors("r1", &fresh)?;

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

    /// An older build's `remote/<id>.json` sends no ETag: a 304 would keep it, whatever it lost.
    #[test]
    fn cached_benchmark_etag_requires_valid_detail_cache() -> anyhow::Result<()> {
        use crate::benchmarks::BenchmarkStore;

        let temp_dir = TempDir::new("etag-cache")?;
        let store = BenchmarkStore::new(&temp_dir.path);

        assert_eq!(cached_benchmark_etag(&store, "b1"), None);

        std::fs::create_dir_all(temp_dir.path.join("remote"))?;
        std::fs::write(temp_dir.path.join("remote/b1.json"), summary_json("b1"))?;
        assert_eq!(cached_benchmark_etag(&store, "b1"), None);

        cache_remote(&store, serde_json::from_str(&summary_json("b1"))?)?;
        assert_eq!(
            cached_benchmark_etag(&store, "b1"),
            None,
            "cached without one"
        );

        cache_remote_with_etag(
            &store,
            serde_json::from_str(&summary_json("b1"))?,
            Some("\"detail\""),
        )?;
        assert_eq!(
            cached_benchmark_etag(&store, "b1")
                .as_ref()
                .map(EntityTag::as_str),
            Some("\"detail\"")
        );
        Ok(())
    }

    #[test]
    fn a_detail_a_claim_rewrote_is_revalidated_against_the_claims_etag() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-claim-rewrote")?;
        let (store, server) = (&t.store, &t.server);
        let served: serde_json::Value = serde_json::from_str(&summary_json("b1"))?;
        t.list(serde_json::json!([served]).to_string(), None);
        let detail = detail_mocks(server, "b1", "\"b1-v1\"", &served);
        assert_eq!(t.pull()?, 1);
        detail.fetched.assert_calls(1);

        // What a claim leaves after the server changed b1 and before it changed it back.
        let changed = serde_json::json!({
            "benchmark_id": "b1",
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": 1024,
        });
        cache_remote_with_etag(store, changed, Some("\"b1-v2\""))?;
        let stale = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header("if-none-match", "\"b1-v2\"");
            then.status(200)
                .header("etag", "\"b1-v1\"")
                .body(served.to_string());
        });

        assert_eq!(t.pull()?, 1);
        stale.assert_calls(1);

        detail.revalidated.assert_calls(0);
        detail.fetched.assert_calls(1);
        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_detail_path(&t.dir.path, "b1"))?;
        assert_eq!(on_disk, served, "the claim's body was kept");

        assert_eq!(t.pull()?, 1);
        detail.revalidated.assert_calls(1);
        detail.fetched.assert_calls(1);
        Ok(())
    }

    #[test]
    fn benchmark_detail_concurrency_uses_system_limit_with_cap() {
        let concurrency = benchmark_detail_concurrency();
        assert!((1..=BENCHMARK_DETAIL_CONCURRENCY_CAP).contains(&concurrency));
    }

    fn summary_json(id: &str) -> String {
        format!(
            r#"{{"benchmark_id":"{id}","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}}"#
        )
    }

    fn pull(
        store: &crate::benchmarks::BenchmarkStore,
        server: &httpmock::MockServer,
    ) -> Result<usize> {
        let server_url = ServerUrl::try_new(server.base_url())?;
        let http =
            HttpClient::new("pipette-test/0").map_err(|e| Error::Other(anyhow::anyhow!("{e}")))?;
        pull_remote_benchmarks(store, &server_url, &http, None)
    }

    struct DetailMocks<'a> {
        revalidated: httpmock::Mock<'a>,
        fetched: httpmock::Mock<'a>,
    }

    fn detail_mocks<'a>(
        server: &'a httpmock::MockServer,
        id: &str,
        etag: &str,
        body: &serde_json::Value,
    ) -> DetailMocks<'a> {
        let path = format!("/benchmarks/{id}");
        DetailMocks {
            revalidated: server.mock(|when, then| {
                when.method(httpmock::Method::GET)
                    .path(&path)
                    .header("if-none-match", etag);
                then.status(304);
            }),
            fetched: server.mock(|when, then| {
                when.method(httpmock::Method::GET)
                    .path(&path)
                    .header_missing("if-none-match");
                then.status(200).header("etag", etag).body(body.to_string());
            }),
        }
    }

    struct PullFixture {
        dir: TempDir,
        store: crate::benchmarks::BenchmarkStore,
        server: httpmock::MockServer,
    }

    impl PullFixture {
        fn new(prefix: &str) -> anyhow::Result<Self> {
            let dir = TempDir::new(prefix)?;
            let store = crate::benchmarks::BenchmarkStore::new(&dir.path);
            Ok(Self {
                dir,
                store,
                server: httpmock::MockServer::start(),
            })
        }

        fn list(&self, body: impl Into<String>, etag: Option<&str>) -> httpmock::Mock<'_> {
            let body = body.into();
            self.server.mock(|when, then| {
                when.method(httpmock::Method::GET).path("/benchmarks");
                let then = then.status(200).body(body);
                if let Some(etag) = etag {
                    then.header("etag", etag);
                }
            })
        }

        fn detail(&self, id: &str, body: impl Into<String>) -> httpmock::Mock<'_> {
            let body = body.into();
            self.server.mock(|when, then| {
                when.method(httpmock::Method::GET)
                    .path(format!("/benchmarks/{id}"));
                then.status(200).body(body);
            })
        }

        fn pull(&self) -> Result<usize> {
            pull(&self.store, &self.server)
        }
    }

    /// The second sync's list answers `304`, and the failed id is still fetched.
    #[test]
    fn transient_detail_failure_recovers_on_next_sync() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-transient")?;
        let (store, server) = (&t.store, &t.server);

        let mut list = t.list(
            format!("[{},{}]", summary_json("b1"), summary_json("b2")),
            Some("\"v1\""),
        );
        t.detail("b1", summary_json("b1"));
        let mut b2_broken = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b2");
            then.status(500).body("transient");
        });

        assert_eq!(t.pull()?, 2, "both ids stay in the catalog");
        let index: Vec<String> = store
            .list_remote_index()?
            .into_iter()
            .map(|s| s.benchmark_id)
            .collect();
        assert!(index.contains(&"b2".to_string()), "failed id kept in index");
        let b2 = pipette_plan_types::BenchmarkId::try_new("b2".to_string())?;
        assert!(
            store.remote_detail_etag(&b2).is_none(),
            "no ETag stored for the failed fetch — must re-request next sync"
        );

        list.delete();
        b2_broken.delete();
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(304).header("etag", "\"v1\"");
        });
        let b2_ok = t.detail("b2", summary_json("b2"));

        assert_eq!(t.pull()?, 2);
        b2_ok.assert_calls(1);
        let id = pipette_plan_types::BenchmarkId::try_new("b2".to_string())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(
            store.has_remote_detail(&id),
            "the failed benchmark recovered through the 304 path"
        );
        Ok(())
    }

    #[test]
    fn a_text_eval_job_without_metrics_is_saved_as_before() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let temp_dir = TempDir::new("score-refresh-text")?;
        let results = ResultsStore::new(&temp_dir.path);
        results.save_payload(
            BenchmarkResultLocation::RemoteSynced,
            "job-1",
            &serde_json::json!({"benchmark_id": "eval_mmlu_2026.06.1"}),
        )?;
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/jobs/job-1");
            then.status(200)
                .body(r#"{"status":"processed","scored_at":"2026-08-12T00:00:00Z","metrics":[]}"#);
        });
        let server_url = ServerUrl::try_new(server.base_url())?;
        let http =
            HttpClient::new("pipette-test/0").map_err(|e| Error::Other(anyhow::anyhow!("{e}")))?;
        assert!(refresh_scored_job(
            &results,
            &server_url,
            &http,
            &test_auth()?,
            "job-1"
        )?);
        assert_eq!(
            score_refresh_state(&results, "job-1"),
            ScoreRefreshState::AlreadyScored
        );
        Ok(())
    }

    /// A vlm_eval or asr_eval saved without metrics would pass
    /// `has_valid_metrics`, and every later sync would skip it as AlreadyScored.
    #[rstest::rstest]
    #[case::vlm_eval("vlm_eval_mmstar_lfm2_5_vl_3b")]
    #[case::asr_eval("asr_eval_esb_ami_cleaned_voxtral")]
    fn processed_job_without_metrics_stays_refreshable(
        #[case] benchmark_id: &str,
    ) -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let temp_dir = TempDir::new("score-refresh")?;
        let results = ResultsStore::new(&temp_dir.path);
        results.save_payload(
            BenchmarkResultLocation::RemoteSynced,
            "job-1",
            &serde_json::json!({"benchmark_id": benchmark_id}),
        )?;
        assert_eq!(
            score_refresh_state(&results, "job-1"),
            ScoreRefreshState::NeedsRefresh
        );

        let server = MockServer::start();
        let server_url = ServerUrl::try_new(server.base_url())?;
        let http =
            HttpClient::new("pipette-test/0").map_err(|e| Error::Other(anyhow::anyhow!("{e}")))?;
        let auth = test_auth()?;

        for metrics_json in ["null", "[]"] {
            let mut no_metrics = server.mock(|when, then| {
                when.method(GET).path("/jobs/job-1");
                then.status(200).body(format!(
                    r#"{{"status":"processed","scored_at":"2026-08-12T00:00:00Z","metrics":{metrics_json}}}"#
                ));
            });
            assert!(
                !refresh_scored_job(&results, &server_url, &http, &auth, "job-1")?,
                "a metrics-less response (metrics={metrics_json}) must count as pending"
            );
            assert_eq!(
                score_refresh_state(&results, "job-1"),
                ScoreRefreshState::NeedsRefresh,
                "no scored file may be persisted for metrics={metrics_json}"
            );
            no_metrics.delete();
        }

        server.mock(|when, then| {
            when.method(GET).path("/jobs/job-1");
            then.status(200).body(
                r#"{"status":"processed","scored_at":"2026-08-12T00:00:00Z",
                    "metrics":[{"metric":"accuracy","value":0.5,"unit":"fraction"}]}"#,
            );
        });
        assert!(refresh_scored_job(
            &results,
            &server_url,
            &http,
            &auth,
            "job-1"
        )?);
        assert_eq!(
            score_refresh_state(&results, "job-1"),
            ScoreRefreshState::AlreadyScored
        );
        Ok(())
    }

    /// The id is server text: escaped, it cannot forge a log line.
    #[test]
    fn a_refused_detail_id_is_logged_escaped() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-refused-id")?;
        let (store, server) = (&t.store, &t.server);
        t.list(format!("[{}]", summary_json("b1")), None);
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            // JSON escapes: a newline, a zero-width space and a Cyrillic look-alike.
            then.status(200).body(summary_json(r"a\nb\u200B\u0430"));
        });

        let (pulled, log) = logged_text(|| t.pull())?;
        pulled?;
        let skipped: Vec<&str> = log
            .lines()
            .filter(|line| line.contains("skipping remote benchmark"))
            .collect();
        assert_eq!(skipped.len(), 1, "{log}");
        assert!(skipped[0].contains(r#"benchmark_id="b1""#), "{log}");
        assert!(
            skipped[0].contains(r#"benchmark "a\nb\u{200b}\u{430}" is not a known"#),
            "{log}"
        );
        assert!(
            store.list(BenchmarkSource::Remote)?.is_empty(),
            "nothing is stored"
        );
        Ok(())
    }

    /// Quoted, the server's text cannot start a log line of its own.
    #[rstest::rstest]
    #[case::schema_mismatch("prefill_throughput", r#"benchmark_type="prefill_throughput""#)]
    #[case::unknown_type("future\nmetric", r#"benchmark_type="future\nmetric""#)]
    fn a_skipped_benchmark_quotes_the_servers_text(
        #[case] raw_type: &str,
        #[case] logged_type: &str,
    ) -> anyhow::Result<()> {
        let ((), log) = logged_text(|| log_skipped_benchmark("a\nb", raw_type, &"bad"))?;
        assert_eq!(log.lines().count(), 1, "{log}");
        assert!(log.contains(r#"benchmark_id="a\nb""#), "{log}");
        assert!(log.contains(logged_type), "{log}");
        Ok(())
    }

    #[test]
    fn a_refused_row_id_is_logged_escaped_and_an_uncached_304_is_reported() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-uncached-304")?;
        let server = &t.server;
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            // A JSON-escaped newline in the first id.
            then.status(200).body(format!(
                "[{},{}]",
                summary_json(r"a\nb"),
                summary_json("b1")
            ));
        });
        let details = server.mock(|when, then| {
            when.method(GET).path_prefix("/benchmarks/");
            then.status(304);
        });

        let (pulled, log) = logged_text(|| t.pull())?;
        assert_eq!(pulled?, 1);
        details.assert_calls(1);
        let lines = |marker: &str| -> Vec<&str> {
            log.lines().filter(|line| line.contains(marker)).collect()
        };
        let skipped = lines("skipping remote benchmark");
        assert_eq!(skipped.len(), 1, "{log}");
        assert!(skipped[0].contains(r#"benchmark_id="a\nb""#), "{log}");
        let reported = lines("304 without a cached detail");
        assert_eq!(reported.len(), 1, "{log}");
        assert!(reported[0].contains(r#"benchmark_id="b1""#), "{log}");
        Ok(())
    }

    /// A pull aimed at a different server must not revalidate against the
    /// previous one's ETags, or a 304 would keep a catalog it never served.
    #[test]
    fn a_new_server_is_pulled_cold() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        use crate::benchmarks::BenchmarkStore;

        let temp_dir = TempDir::new("pull-new-server")?;
        let store = BenchmarkStore::new(&temp_dir.path);

        let first = MockServer::start();
        first.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200)
                .header("etag", "\"v1\"")
                .body(format!("[{}]", summary_json("b1")));
        });
        first.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(200)
                .header("etag", "\"d1\"")
                .body(summary_json("b1"));
        });
        assert_eq!(pull(&store, &first)?, 1);

        // The same ids and ETags, so a leaked `If-None-Match` would land on a
        // 304 rather than fail loudly.
        let second = MockServer::start();
        let conditional = second.mock(|when, then| {
            when.method(GET).header_exists("If-None-Match");
            then.status(304).header("etag", "\"v1\"");
        });
        let list = second.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200)
                .header("etag", "\"v1\"")
                .body(format!("[{}]", summary_json("b1")));
        });
        let detail = second.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(200)
                .header("etag", "\"d1\"")
                .body(summary_json("b1"));
        });

        assert_eq!(pull(&store, &second)?, 1);

        conditional.assert_calls(0);
        list.assert_calls(1);
        detail.assert_calls(1);
        let state = store
            .get_sync_state()?
            .ok_or_else(|| anyhow::anyhow!("sync state missing"))?;
        assert_eq!(
            state.server_url,
            Some(ServerUrl::try_new(second.base_url())?),
            "the cache now belongs to the server it was pulled from"
        );
        Ok(())
    }

    /// A full outage is an error, not "pulled 0".
    #[test]
    fn total_detail_outage_leaves_the_cached_catalog_untouched() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-outage")?;
        let (store, server) = (&t.store, &t.server);

        let mut list_ok = t.list(format!("[{}]", summary_json("b1")), Some("\"v0\""));
        let mut detail_ok = t.detail("b1", summary_json("b1"));
        assert_eq!(t.pull()?, 1);
        list_ok.delete();
        detail_ok.delete();

        // Outage: list changes (new ETag) but every detail fetch dies.
        t.list(format!("[{}]", summary_json("b1")), Some("\"v1\""));
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(500).body("down");
        });

        assert!(
            t.pull().is_err(),
            "an all-failed pull must be an error, not a clean 'pulled 0'"
        );
        let index: Vec<String> = store
            .list_remote_index()?
            .into_iter()
            .map(|s| s.benchmark_id)
            .collect();
        assert_eq!(index, vec!["b1".to_string()], "index untouched");
        let id = pipette_plan_types::BenchmarkId::try_new("b1".to_string())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(store.has_remote_detail(&id), "cached detail untouched");
        let state = store
            .get_sync_state()?
            .ok_or_else(|| anyhow::anyhow!("sync state missing"))?;
        assert_eq!(
            state.benchmarks_etag.as_ref().map(EntityTag::as_str),
            Some("\"v0\""),
            "list ETag not advanced past the failed pull"
        );
        Ok(())
    }

    /// With no index to keep, failing beats reporting an empty catalog.
    #[test]
    fn a_304_list_without_a_cached_index_is_an_error() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-cold-304")?;
        let (store, server) = (&t.store, &t.server);
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(304);
        });
        let err = t
            .pull()
            .err()
            .ok_or_else(|| anyhow::anyhow!("a cold 304 should fail"))?;
        assert!(
            err.to_string()
                .contains("cached benchmark index is unavailable"),
            "{err}"
        );
        assert!(store.list_remote_index()?.is_empty());
        Ok(())
    }

    /// The vanished detail file is met by a 304 anyway.
    #[test]
    fn a_revalidated_detail_is_kept_and_a_vanished_one_is_refetched() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-detail-304")?;
        let (store, server) = (&t.store, &t.server);
        t.list(format!("[{}]", summary_json("b1")), None);
        let mut fresh = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(200)
                .header("etag", "\"d1\"")
                .body(summary_json("b1"));
        });
        assert_eq!(t.pull()?, 1);
        fresh.delete();
        let recorded = std::fs::read_to_string(t.dir.path.join("remote/detail/b1.etag"))?;
        assert!(recorded.ends_with("\n\"d1\""), "{recorded}");

        let revalidated = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header("if-none-match", "\"d1\"");
            then.status(304);
        });
        assert_eq!(t.pull()?, 1);
        revalidated.assert();
        let id = pipette_plan_types::BenchmarkId::try_new("b1".to_string())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(store.has_remote_detail(&id), "the cached definition stays");
        assert_eq!(
            store
                .remote_detail_etag(&id)
                .as_ref()
                .map(EntityTag::as_str),
            Some("\"d1\""),
            "the detail ETag survives a revalidation"
        );

        std::fs::remove_file(remote_detail_path(&t.dir.path, "b1"))?;
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(304);
        });
        assert_eq!(t.pull()?, 1, "the id stays in the index");
        assert!(
            store.remote_detail_etag(&id).is_none(),
            "no ETag, so the next sync asks unconditionally"
        );
        Ok(())
    }

    /// An unknown type is never fetched; a known type's row is fetched and counted even when it
    /// lacks a parameter only the detail carries.
    #[test]
    fn an_unconvertible_benchmark_is_skipped_not_fatal() -> anyhow::Result<()> {
        let t = PullFixture::new("pull-skip")?;
        let store = &t.store;
        let unknown = r#"{"benchmark_id":"b_new","benchmark_type":"quantum_latency"}"#;
        let malformed = r#"{"benchmark_id":"b_bad","benchmark_type":"prefill_throughput"}"#;
        let partial_row = r#"{"benchmark_id":"b_partial","benchmark_type":"prefill_throughput"}"#;
        t.list(
            format!(
                "[{},{unknown},{malformed},{partial_row}]",
                summary_json("b1"),
            ),
            None,
        );
        t.detail("b1", summary_json("b1"));
        let fetched_unknown = t.detail("b_new", unknown);
        let fetched_malformed = t.detail("b_bad", malformed);
        let fetched_partial = t.detail("b_partial", summary_json("b_partial"));

        let (pulled, log) = logged_text(|| t.pull())?;

        assert_eq!(pulled?, 3, "every row of a known type");
        fetched_unknown.assert_calls(0);
        fetched_malformed.assert_calls(1);
        fetched_partial.assert_calls(1);
        let index: Vec<String> = store
            .list_remote_index()?
            .into_iter()
            .map(|s| s.benchmark_id)
            .collect();
        assert_eq!(index, ["b1", "b_new", "b_bad", "b_partial"]);
        assert!(remote_detail_path(&t.dir.path, "b_partial").exists());
        ["b_new", "b_bad"].iter().for_each(|skipped| {
            assert!(
                !remote_detail_path(&t.dir.path, skipped).exists(),
                "{skipped} was cached"
            );
        });
        let skipped = |reason: &str| log.lines().filter(|line| line.contains(reason)).count();
        assert_eq!(skipped("unrecognized benchmark_type"), 1, "{log}");
        assert_eq!(skipped("schema mismatch"), 1, "{log}");
        Ok(())
    }

    /// Every row, unknown types and keys included, so the list ETag validates
    /// exactly what is on disk.
    #[test]
    fn the_list_is_stored_as_the_server_sent_it() -> anyhow::Result<()> {
        let t = PullFixture::new("pull-list-verbatim")?;
        let store = &t.store;
        let list = serde_json::json!([
            {
                "benchmark_id": "b1",
                "benchmark_type": "prefill_throughput",
                "parameter_prefill_tokens": 512,
                "parameter_from_the_future": 0.25,
                "parameter_nested": {"a": [1, 2]},
                "org_id": "org_test",
            },
            {
                "benchmark_id": "b_new",
                "benchmark_type": "quantum_latency",
                "parameter_qubits": 9_007_199_254_740_993_u64,
            },
        ]);
        t.list(list.to_string(), Some("\"v1\""));
        t.detail("b1", summary_json("b1"));

        assert_eq!(t.pull()?, 1);

        let stored: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_state_path(&t.dir.path, "index"))?;
        assert_eq!(stored, list);
        let state = store
            .get_sync_state()?
            .ok_or_else(|| anyhow::anyhow!("sync state missing"))?;
        assert_eq!(
            state.benchmarks_etag.as_ref().map(EntityTag::as_str),
            Some("\"v1\"")
        );
        assert_eq!(state.benchmark_count, 1, "only the row of a known type");
        Ok(())
    }

    /// `b2` has no detail (the build that pulled could not read it) and `b_new`'s type is
    /// unknown here, its detail cached by a newer build.
    #[test]
    fn a_304_on_the_list_cannot_shrink_the_catalog() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-304-full-list")?;
        let (store, server) = (&t.store, &t.server);
        let b1: serde_json::Value = serde_json::from_str(&summary_json("b1"))?;
        let b2: serde_json::Value = serde_json::from_str(&summary_json("b2"))?;
        let newer = serde_json::json!({
            "benchmark_id": "b_new",
            "benchmark_type": "quantum_latency",
            "parameter_qubits": 9,
        });
        let rows = serde_json::json!([b1, b2, newer]);
        store.put_remote_index(&serde_json::from_value::<Vec<BenchmarkSummary>>(
            rows.clone(),
        )?)?;
        cache_remote_with_etag(store, b1.clone(), Some("\"b1-v1\""))?;
        std::fs::write(remote_detail_path(&t.dir.path, "b_new"), newer.to_string())?;
        store.put_sync_state(&RemoteSyncState {
            pulled_at: "2026-10-18T00:00:00Z".to_string(),
            benchmark_count: 1,
            server_url: Some(ServerUrl::try_new(server.base_url())?),
            benchmarks_etag: Some(EntityTag::try_new("\"list-v1\"")?),
        })?;

        let list = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks")
                .header("if-none-match", "\"list-v1\"");
            then.status(304).header("etag", "\"list-v1\"");
        });
        let b1_detail = detail_mocks(server, "b1", "\"b1-v1\"", &b1);
        let b2_detail = detail_mocks(server, "b2", "\"b2-v1\"", &b2);
        let newer_detail = t.detail("b_new", newer.to_string());

        assert_eq!(t.pull()?, 2);

        list.assert();
        b1_detail.revalidated.assert_calls(1);
        b1_detail.fetched.assert_calls(0);
        b2_detail.fetched.assert_calls(1);
        newer_detail.assert_calls(0);
        assert!(
            remote_detail_path(&t.dir.path, "b_new").exists(),
            "the prune dropped a listed id's detail"
        );
        let stored: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_state_path(&t.dir.path, "index"))?;
        assert_eq!(stored, rows);
        let listed: Vec<String> = store
            .list(BenchmarkSource::Remote)?
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect();
        assert_eq!(listed, ["b1", "b2"]);
        Ok(())
    }

    #[test]
    fn a_pulled_detail_keeps_keys_this_build_does_not_model() -> anyhow::Result<()> {
        let t = PullFixture::new("pull-verbatim")?;
        let sent = serde_json::json!({
            "benchmark_id": "b1",
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": 512,
            "parameter_batch_size": 4,
            "parameter_from_the_future": 0.25,
            "org_id": "org_test",
        });
        t.list(format!("[{}]", summary_json("b1")), None);
        t.detail("b1", sent.to_string());

        assert_eq!(t.pull()?, 1);

        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_detail_path(&t.dir.path, "b1"))?;
        assert_eq!(on_disk, sent);
        Ok(())
    }

    #[test]
    fn a_legacy_detail_is_refetched_not_revalidated() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-legacy")?;
        let (store, server) = (&t.store, &t.server);
        let legacy = write_public_2026_08_1_img80_cache(&t.dir.path)?;
        let sent = img80_wire();
        store.put_remote_index(&[serde_json::from_value(sent.clone())?])?;
        store.put_sync_state(&RemoteSyncState {
            pulled_at: "2026-10-18T00:00:00Z".to_string(),
            benchmark_count: 1,
            server_url: Some(ServerUrl::try_new(server.base_url())?),
            benchmarks_etag: Some(EntityTag::try_new("\"list-v1\"")?),
        })?;

        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(304).header("etag", "\"list-v1\"");
        });
        let detail = detail_mocks(server, IMG80, "\"img80-v1\"", &sent);

        assert_eq!(t.pull()?, 1);

        detail.revalidated.assert_calls(0);
        detail.fetched.assert_calls(1);
        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_detail_path(&t.dir.path, IMG80))?;
        assert_eq!(on_disk, sent);
        assert!(legacy.exists(), "the older build's file went");
        let reference = crate::benchmarks::SourcedBenchmarkId::new(
            BenchmarkSource::Remote,
            BenchmarkId::try_new(IMG80.to_owned())?,
        );
        let cached = store.get(&reference)?;
        let Some(pipette_plan_types::BenchmarkDefinition::VlThroughput(vl)) = &cached else {
            anyhow::bail!("expected the 80-image definition, got {cached:?}");
        };
        assert_eq!(vl.parameter_num_images.into_inner(), 80);
        Ok(())
    }

    /// None of their ETags is sent, even for an id with a `detail/` file.
    #[test]
    fn an_older_builds_list_and_sync_state_are_never_read() -> anyhow::Result<()> {
        use httpmock::prelude::*;

        let t = PullFixture::new("pull-older-bookkeeping")?;
        let (store, server) = (&t.store, &t.server);
        cache_remote(store, serde_json::from_str(&summary_json("b1"))?)?;
        // As an older build leaves them after pulling from this server, `b_old` still on its
        // list.
        let older =
            write_older_build_bookkeeping(&t.dir.path, &server.base_url(), &["b1", "b_old"])?;
        let changed = serde_json::json!({
            "benchmark_id": "b1",
            "benchmark_type": "prefill_throughput",
            "parameter_prefill_tokens": 1024,
        });

        let conditional = server.mock(|when, then| {
            when.method(GET).header_exists("if-none-match");
            then.status(304);
        });
        let list = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks")
                .header_missing("if-none-match");
            then.status(200)
                .header("etag", "\"list-v2\"")
                .body(format!("[{changed}]"));
        });
        let detail = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header_missing("if-none-match");
            then.status(200)
                .header("etag", "\"b1-v2\"")
                .body(changed.to_string());
        });

        assert_eq!(t.pull()?, 1);

        conditional.assert_calls(0);
        list.assert();
        detail.assert();
        let on_disk: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_detail_path(&t.dir.path, "b1"))?;
        assert_eq!(on_disk, changed, "the stale detail was kept");
        assert!(older.index.exists(), "the older build's index went");
        assert!(older.sync.exists(), "the older build's sync state went");
        let state_files: Vec<String> = std::fs::read_dir(t.dir.path.join("remote/state"))?
            .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<std::collections::BTreeSet<_>>>()?
            .into_iter()
            .collect();
        assert_eq!(state_files, ["index.json", "sync.json"]);
        let stored: serde_json::Value =
            pipette_ops::fs::read_json_file(&remote_state_path(&t.dir.path, "index"))?;
        assert_eq!(stored, serde_json::json!([changed]));
        let state = store
            .get_sync_state()?
            .ok_or_else(|| anyhow::anyhow!("sync state missing"))?;
        assert_eq!(
            state.benchmarks_etag.as_ref().map(EntityTag::as_str),
            Some("\"list-v2\"")
        );
        let b1 = pipette_plan_types::BenchmarkId::try_new("b1".to_string())?;
        assert_eq!(
            store
                .remote_detail_etag(&b1)
                .as_ref()
                .map(EntityTag::as_str),
            Some("\"b1-v2\"")
        );
        Ok(())
    }
}

/// How a synced job should be treated by the score-refresh step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreRefreshState {
    /// Not a server-scored type, or its payload is unreadable.
    NotServerScored,
    AlreadyScored,
    NeedsRefresh,
}

pub fn score_refresh_state(store: &ResultsStore, job_id: &str) -> ScoreRefreshState {
    let payload = match store.load_payload(BenchmarkResultLocation::RemoteSynced, job_id) {
        Ok(v) => v,
        Err(err) => {
            tracing::info!(
                %job_id,
                error = %format_args!("{err:#}"),
                "cannot load payload, skipping refresh"
            );
            return ScoreRefreshState::NotServerScored;
        }
    };
    // No type list here, so a new server-scored family cannot miss refresh.
    if !recorded_benchmark_type(&payload, "").is_some_and(BenchmarkType::is_scored_via_server) {
        return ScoreRefreshState::NotServerScored;
    }
    // A processed job's score never changes, so one on disk is final.
    if store.has_valid_metrics(BenchmarkResultLocation::RemoteSynced, job_id) {
        tracing::info!(%job_id, "already scored, skipping refresh");
        return ScoreRefreshState::AlreadyScored;
    }
    ScoreRefreshState::NeedsRefresh
}

fn stored_benchmark_type(store: &ResultsStore, job_id: &str) -> Option<BenchmarkType> {
    let payload = store
        .load_payload(BenchmarkResultLocation::RemoteSynced, job_id)
        .ok()?;
    recorded_benchmark_type(&payload, "")
}

pub fn refresh_scored_job(
    store: &ResultsStore,
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    job_id: &str,
) -> Result<bool> {
    tracing::info!(%job_id, "refreshing job");
    let job: JobResponse = client_api::get_job(server, http, auth, job_id)?;
    if job.status != JobStatus::Processed {
        tracing::info!(%job_id, job_status = %job.status, "job not processed yet");
        return Ok(false);
    }
    // Saved, a metrics-less vlm_eval or asr_eval would pass `has_valid_metrics`
    // and never be refreshed again. Other types are saved even without metrics.
    if job.metrics.as_ref().is_none_or(Vec::is_empty)
        && matches!(
            stored_benchmark_type(store, job_id),
            Some(BenchmarkType::VlmEval | BenchmarkType::AsrEval)
        )
    {
        tracing::warn!(
            "job {job_id} reports processed but carries no metrics; \
             not saving a score, will refresh again next sync"
        );
        return Ok(false);
    }
    let scored = serde_json::to_value(&BenchmarkScoredResult {
        scored_at: job.scored_at.map(|at| at.to_string()).or_else(|| {
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
    tracing::info!(%job_id, "job processed");
    Ok(true)
}

#[cfg(test)]
mod score_refresh_tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use pipette_mgmt_client::PrivateKey;
    use pipette_plan_types::ClientId;

    use super::*;

    const SYNCED: BenchmarkResultLocation = BenchmarkResultLocation::RemoteSynced;
    const PENDING: BenchmarkResultLocation = BenchmarkResultLocation::RemotePending;

    fn store() -> anyhow::Result<(tempfile::TempDir, ResultsStore)> {
        let tmp = tempfile::tempdir()?;
        let store = ResultsStore::new(tmp.path().join("results"));
        Ok((tmp, store))
    }

    fn auth() -> anyhow::Result<ClientIdentity> {
        Ok(ClientIdentity {
            client_id: ClientId::try_new("ev1_test")?,
            private_key_hex: PrivateKey::try_new("11".repeat(32))?,
        })
    }

    fn connect(server: &MockServer) -> anyhow::Result<(ServerUrl, HttpClient)> {
        Ok((
            ServerUrl::try_new(server.base_url())?,
            HttpClient::new("pipette-test/0")?,
        ))
    }

    #[rstest::rstest]
    #[case::timing(
        Some(json!({"benchmark_id": "prefill_throughput_512"})),
        None,
        ScoreRefreshState::NotServerScored
    )]
    #[case::eval_unscored(
        Some(json!({"benchmark_id": "eval_mmlu"})),
        None,
        ScoreRefreshState::NeedsRefresh
    )]
    #[case::eval_scored(
        Some(json!({"benchmark_id": "eval_mmlu"})),
        Some("{}"),
        ScoreRefreshState::AlreadyScored
    )]
    #[case::eval_corrupt_score(
        Some(json!({"benchmark_id": "eval_mmlu"})),
        Some("{not json"),
        ScoreRefreshState::NeedsRefresh
    )]
    #[case::recorded_eval(
        Some(json!({"benchmark_id": "quill_mmlu", "benchmark_type": "eval"})),
        None,
        ScoreRefreshState::NeedsRefresh
    )]
    #[case::no_payload(None, None, ScoreRefreshState::NotServerScored)]
    fn the_refresh_state_follows_type_and_metrics(
        #[case] payload: Option<serde_json::Value>,
        #[case] metrics: Option<&str>,
        #[case] expected: ScoreRefreshState,
    ) -> anyhow::Result<()> {
        let (_tmp, store) = store()?;
        if let Some(payload) = payload {
            store.save_payload(SYNCED, "job-1", &payload)?;
        }
        if let Some(metrics) = metrics {
            std::fs::write(store.metrics_path(SYNCED, "job-1"), metrics)?;
        }

        assert_eq!(score_refresh_state(&store, "job-1"), expected);
        Ok(())
    }

    fn refresh(body: &'static str) -> anyhow::Result<(tempfile::TempDir, ResultsStore, bool)> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/jobs/job-1");
            then.status(200).body(body);
        });
        let (tmp, store) = store()?;
        let (url, http) = connect(&server)?;
        let refreshed = refresh_scored_job(&store, &url, &http, &auth()?, "job-1")?;
        Ok((tmp, store, refreshed))
    }

    fn saved_score(store: &ResultsStore) -> anyhow::Result<BenchmarkScoredResult> {
        pipette_ops::fs::read_json_file(&store.metrics_path(SYNCED, "job-1"))
    }

    #[test]
    fn an_unprocessed_job_stores_no_score() -> anyhow::Result<()> {
        let (_tmp, store, refreshed) = refresh(r#"{"status":"scoring","metrics":null}"#)?;

        assert!(!refreshed);
        assert!(!store.metrics_path(SYNCED, "job-1").exists());
        Ok(())
    }

    #[test]
    fn a_processed_job_stores_its_metrics() -> anyhow::Result<()> {
        let (_tmp, store, refreshed) = refresh(
            r#"{"status":"processed","scored_at":"2026-09-01T00:00:00Z",
                "metrics":[{"metric":"accuracy","value":0.5,"unit":"ratio"}]}"#,
        )?;
        store.save_payload(SYNCED, "job-1", &json!({"benchmark_id": "eval_mmlu"}))?;

        assert!(refreshed);
        let saved = saved_score(&store)?;
        let metrics = saved.metrics.unwrap_or_default();
        assert_eq!(
            metrics
                .iter()
                .map(|m| m.metric.as_str())
                .collect::<Vec<_>>(),
            ["accuracy"]
        );
        assert!(
            saved
                .scored_at
                .is_some_and(|at| at.starts_with("2026-09-01"))
        );
        assert_eq!(
            score_refresh_state(&store, "job-1"),
            ScoreRefreshState::AlreadyScored
        );
        Ok(())
    }

    /// The listing must never show a scored result with no time.
    #[test]
    fn a_processed_job_without_a_timestamp_is_stamped_now() -> anyhow::Result<()> {
        let (_tmp, store, _) = refresh(r#"{"status":"processed","metrics":[]}"#)?;

        assert!(
            saved_score(&store)?
                .scored_at
                .is_some_and(|at| !at.is_empty())
        );
        Ok(())
    }

    fn pending_payload() -> serde_json::Value {
        json!({
            "benchmark_id": "prefill_throughput_512",
            "model_descriptor": "{\"type\":\"gguf_text\"}",
            "runtime_descriptor": "{\"type\":\"llamacpp_cli_stock_tools\"}",
        })
    }

    /// The job id is recorded before the move, so a failed move is finished
    /// without posting twice.
    #[test]
    fn a_recorded_job_id_finishes_the_move_without_resubmitting() -> anyhow::Result<()> {
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-other"}"#);
        });
        let (_tmp, store) = store()?;
        store.save_payload(PENDING, "r1", &pending_payload())?;
        std::fs::write(store.result_dir(PENDING, "r1").join("job_id"), "job-7\n")?;
        let (url, http) = connect(&server)?;

        let job_id = submit_pending_result(&store, &url, &http, &auth()?, "r1")?;

        assert_eq!(job_id.as_deref(), Some("job-7"));
        submit.assert_calls(0);
        assert!(!store.result_dir(PENDING, "r1").exists());
        assert_eq!(
            store.load_payload(SYNCED, "job-7")?["benchmark_id"],
            "prefill_throughput_512"
        );
        Ok(())
    }

    #[test]
    fn a_result_locked_by_another_submit_is_skipped() -> anyhow::Result<()> {
        let server = MockServer::start();
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"job-1"}"#);
        });
        let (_tmp, store) = store()?;
        store.save_payload(PENDING, "r1", &pending_payload())?;
        let held = std::fs::File::create(store.location_dir(PENDING).join("r1.lock"))?;
        held.try_lock()?;
        let (url, http) = connect(&server)?;

        let job_id = submit_pending_result(&store, &url, &http, &auth()?, "r1")?;

        assert_eq!(job_id, None);
        submit.assert_calls(0);
        assert!(store.result_dir(PENDING, "r1").exists());
        Ok(())
    }

    #[test]
    fn a_job_id_outside_synced_is_refused() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(202).body(r#"{"job_id":"../x"}"#);
        });
        let (_tmp, store) = store()?;
        store.save_payload(PENDING, "r1", &pending_payload())?;
        let (url, http) = connect(&server)?;

        assert!(submit_pending_result(&store, &url, &http, &auth()?, "r1").is_err());
        assert!(store.result_dir(PENDING, "r1").exists());
        assert!(!store.result_dir(PENDING, "r1").join("job_id").exists());
        assert!(!store.result_dir(SYNCED, "../x").exists());
        Ok(())
    }
}
