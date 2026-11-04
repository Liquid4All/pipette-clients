use clap::Args;

use pipette_http::HttpClient;
use pipette_mgmt_client::error::Error as MgmtError;
use pipette_mgmt_client::{ClientIdentity, ServerUrl};

use crate::client::sync::{
    ScoreRefreshState, pull_remote_benchmarks, refresh_scored_job, score_refresh_state,
    submit_pending_result,
};
use crate::console::Console;
use crate::error::{Error, Result};
use crate::identity::ClientRegistrationStore;
use crate::results::BenchmarkResultLocation;
use crate::workspace::PipetteWorkspace;

pub const SYNC_LONG_ABOUT: &str = "\
Synchronize remote benchmark state with the management server.

The command pulls benchmark definitions, submits pending results, and
refreshes synced jobs for scores. Pass a result ID to narrow the submit
and score steps to that one result; definitions are pulled either way.

What runs depends on whether this workspace holds a registration. The
benchmark catalog is a public read, so an unregistered workspace can still
pull it and see what there is to run before it registers to run any of it;
name the server with --server-url, since only `auth register` defaults one.
A registered workspace signs the pull, so the catalog it sees includes what
the server keeps to this client alone. Submitting results and refreshing
scores are authenticated, so those two steps are skipped, with a note, until
`auth register` (or `auth import`) puts an identity in place.";

pub const SYNC_AFTER_HELP: &str = "\
Examples:
  pipette sync                                         # registered
  pipette sync <result-id>
  pipette sync --server-url https://mgmt.example.com   # unregistered: pull only

Only results created from remote benchmarks are submitted upstream. Results from
local benchmark definitions stay local.";

/// Sync benchmarks and results with the management server
#[derive(Args, Debug)]
pub struct SyncArgs {
    /// Result ID to submit (omit to submit all pending)
    pub result_id: Option<String>,

    /// Management server to pull the catalog from. Required when this
    /// workspace holds no registration, since nothing else names one. A
    /// registered workspace uses its registered server and refuses a value
    /// that disagrees with it.
    #[arg(long)]
    pub server_url: Option<ServerUrl>,
}

impl SyncArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        console: &mut dyn Console,
    ) -> Result<()> {
        let registration = ws.registration();
        let results = ws.results();

        let identity = signing_identity_if_registered(&registration)?;

        // Ahead of resolving the server: for `sync <result-id>` without an
        // identity the missing registration is the blocker worth reporting,
        // not the missing pull destination it also implies.
        if identity.is_none() && self.result_id.is_some() {
            return Err(Error::SyncSubmitNeedsRegistration);
        }

        // Needed by the pull whether or not there is an identity to sign with.
        let server_url = self.resolve_server_url(&registration)?;

        tracing::info!(url = %server_url, "pulling remote benchmarks");
        let count = pull_remote_benchmarks(&ws.benchmarks(), &server_url, http, identity.as_ref())?;
        writeln!(console.out(), "pulled {count} benchmark(s)")?;

        let Some(auth) = identity.as_ref() else {
            writeln!(
                console.out(),
                "no signing identity: skipped submit and score \
                 (run `pipette auth register`, or `pipette auth import` to \
                 restore one, to submit results)"
            )?;
            return Ok(());
        };

        let (synced_job_ids, failed_submits): (Vec<String>, usize) = match &self.result_id {
            // One named result: its submit failure is the command's failure.
            Some(result_id) => {
                tracing::info!(%result_id, "submitting pending result");
                // An unknown id is a typo; one a racing submit moves after this is skipped.
                let src = results.result_dir(BenchmarkResultLocation::RemotePending, result_id);
                if !src.exists() {
                    return Err(anyhow::anyhow!("no pending result at {}", src.display()).into());
                }
                match submit_pending_result(&results, &server_url, http, auth, result_id)? {
                    Some(job_id) => {
                        tracing::info!(%result_id, %job_id, "submitted result");
                        (vec![job_id], 0)
                    }
                    None => {
                        tracing::info!(%result_id, "another process is submitting it; skipped");
                        (Vec::new(), 0)
                    }
                }
            }
            // Draining all of them: a failure with one result (an unreadable
            // payload included) is logged and counted, and the next is tried.
            // A result that fails stays in `pending` for the next run.
            None => results
                .list_ids(BenchmarkResultLocation::RemotePending)?
                .into_iter()
                .fold((Vec::new(), 0usize), |(mut synced, failed), result_id| {
                    tracing::info!(%result_id, "submitting pending result");
                    match submit_pending_result(&results, &server_url, http, auth, &result_id) {
                        Ok(Some(job_id)) => {
                            tracing::info!(%result_id, %job_id, "submitted result");
                            synced.push(job_id);
                            (synced, failed)
                        }
                        Ok(None) => {
                            tracing::info!(%result_id, "another process is submitting it; skipped");
                            (synced, failed)
                        }
                        Err(e) => {
                            tracing::info!(
                                %result_id,
                                error = %format_args!("{e:#}"),
                                "failed to submit result"
                            );
                            (synced, failed + 1)
                        }
                    }
                }),
        };
        writeln!(
            console.out(),
            "{}",
            submit_summary(synced_job_ids.len(), failed_submits)
        )?;

        let candidate_ids = if self.result_id.is_some() {
            synced_job_ids
        } else {
            results.list_ids(BenchmarkResultLocation::RemoteSynced)?
        };
        let (eval_jobs, already_scored) =
            candidate_ids
                .into_iter()
                .fold(
                    (Vec::new(), 0usize),
                    |(mut jobs, scored), id| match score_refresh_state(&results, &id) {
                        ScoreRefreshState::NeedsRefresh => {
                            jobs.push(id);
                            (jobs, scored)
                        }
                        ScoreRefreshState::AlreadyScored => (jobs, scored + 1),
                        ScoreRefreshState::NotServerScored => (jobs, scored),
                    },
                );
        let (scored, pending) =
            eval_jobs
                .iter()
                .fold(
                    (0usize, 0usize),
                    |(scored, pending), job_id| match refresh_scored_job(
                        &results,
                        &server_url,
                        http,
                        auth,
                        job_id,
                    ) {
                        Ok(true) => (scored + 1, pending),
                        Ok(false) => (scored, pending + 1),
                        Err(err) => {
                            if is_not_found(&err) {
                                tracing::info!(%job_id, "job not found on server, skipping");
                            } else {
                                tracing::info!(
                                    %job_id,
                                    error = %format_args!("{err:#}"),
                                    "job refresh failed"
                                );
                            }
                            (scored, pending)
                        }
                    },
                );
        writeln!(
            console.out(),
            "scored {} result(s) ({already_scored} already scored), {pending} pending",
            scored + already_scored
        )?;

        Ok(())
    }

    /// Where the catalog pull goes: the registered server, else `--server-url`,
    /// else [`Error::SyncServerUrlUnknown`]. The registration wins because its
    /// identity signs only for that server, and a flag that disagrees is an
    /// error (what the operator wants is `auth reset`). No built-in default, so
    /// no command quietly contacts a host the operator never named: only
    /// `auth register` picks one, and it records what it picked.
    fn resolve_server_url(&self, registration: &ClientRegistrationStore) -> Result<ServerUrl> {
        let registered = registration.get()?.map(|record| record.server_url);
        match (registered, self.server_url.clone()) {
            (Some(registered), Some(requested)) if requested != registered => {
                Err(Error::SyncServerUrlConflict {
                    requested,
                    registered,
                })
            }
            (Some(registered), _) => Ok(registered),
            (None, Some(requested)) => Ok(requested),
            (None, None) => Err(Error::SyncServerUrlUnknown),
        }
    }
}

/// The submit step's one-line summary. Names the failures because the drain
/// still exits `0`, so this is where an operator not reading the log learns
/// that some stayed behind.
fn submit_summary(synced: usize, failed: usize) -> String {
    if failed == 0 {
        format!("submitted {synced} result(s)")
    } else {
        format!("submitted {synced} result(s), {failed} failed (see log)")
    }
}

/// The signing identity, or `None` when this workspace holds none. A missing
/// registration and a missing key are one answer: neither can sign. A keyless
/// registration still names a server, which is why the URL is resolved
/// separately. Every other failure is a broken workspace and propagates.
fn signing_identity_if_registered(
    registration: &ClientRegistrationStore,
) -> Result<Option<ClientIdentity>> {
    match registration.signing_identity() {
        Ok(identity) => Ok(Some(identity)),
        Err(Error::RegistrationMissing(_) | Error::PrivateKeyMissing) => Ok(None),
        Err(other) => Err(other),
    }
}

fn is_not_found(err: &Error) -> bool {
    matches!(err, Error::Mgmt(MgmtError::HttpStatus { status, .. }) if *status == 404)
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;

    use super::*;
    use crate::test_support::CapturedConsole;
    use crate::test_support::TempWorkspace;

    const REGISTERED: &str = "https://registered.example.test";

    fn args(result_id: Option<&str>, server_url: Option<&str>) -> Result<SyncArgs> {
        Ok(SyncArgs {
            result_id: result_id.map(str::to_string),
            server_url: server_url.map(ServerUrl::try_new).transpose()?,
        })
    }

    /// Catches any catalog read that carries a signature; registered first so
    /// it outranks the plain mocks, and asserted at zero hits.
    fn signed_decoy(server: &MockServer) -> httpmock::Mock<'_> {
        server.mock(|when, then| {
            when.method(GET)
                .path_includes("/benchmarks")
                .header_exists("X-Signature");
            then.status(500).body("signed");
        })
    }

    fn http() -> anyhow::Result<HttpClient> {
        HttpClient::new("pipette-test/0").map_err(|e| anyhow::anyhow!("{e}"))
    }

    #[test]
    fn an_unregistered_workspace_takes_the_requested_server() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-requested-server")?;
        let url = args(None, Some("https://mgmt.example.test"))?
            .resolve_server_url(&temp.ws.registration())?;
        assert_eq!(url.to_string(), "https://mgmt.example.test");
        Ok(())
    }

    #[test]
    fn an_unregistered_workspace_with_no_flag_has_no_server() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-no-server")?;
        let error = args(None, None)?
            .resolve_server_url(&temp.ws.registration())
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(
            matches!(error, Error::SyncServerUrlUnknown),
            "unexpected error: {error}"
        );
        Ok(())
    }

    /// The registered server wins. `ServerUrl` normalizes on construction, so a
    /// flag differing only by a trailing slash is the same server, not a conflict.
    #[rstest::rstest]
    #[case::no_flag(None)]
    #[case::matching_flag(Some(REGISTERED))]
    #[case::trailing_slash(Some("https://registered.example.test/"))]
    fn a_registration_decides_the_server(#[case] flag: Option<&str>) -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-registered-server")?;
        temp.register(REGISTERED)?;
        let url = args(None, flag)?.resolve_server_url(&temp.ws.registration())?;
        assert_eq!(url.to_string(), REGISTERED);
        Ok(())
    }

    /// Pulling from one host while submitting to another is not a coherent
    /// run, so the disagreement is reported rather than resolved either way.
    #[test]
    fn a_conflicting_server_url_is_refused() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-server-conflict")?;
        temp.register(REGISTERED)?;
        let error = args(None, Some("https://elsewhere.example.test"))?
            .resolve_server_url(&temp.ws.registration())
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a conflict"))?;
        assert!(
            matches!(error, Error::SyncServerUrlConflict { .. }),
            "unexpected error: {error}"
        );
        // Both hosts are named, so the reader can tell which one to change.
        let message = error.to_string();
        assert!(message.contains("elsewhere.example.test"), "{message}");
        assert!(message.contains("registered.example.test"), "{message}");
        Ok(())
    }

    /// A submission payload the server would accept, so a test that seeds one
    /// reaches the submit step rather than tripping the stale-payload guard.
    fn pending_payload() -> serde_json::Value {
        serde_json::json!({
            "benchmark_id": "prefill_throughput_256",
            "model_descriptor": "{\"type\":\"mlx\"}",
            "runtime_descriptor": "{\"type\":\"mlx_macos_pipette\"}",
        })
    }

    /// The pull runs, the two authenticated steps are skipped, and the command
    /// exits `Ok`. A pending result is seeded so the decoy and the still-pending
    /// assertion tell "skipped because unregistered" from "nothing to submit".
    #[test]
    fn an_unregistered_sync_pulls_the_catalog_and_skips_the_rest() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-unregistered")?;
        let server = MockServer::start();
        let results = temp.ws.results();
        results.save_payload(
            BenchmarkResultLocation::RemotePending,
            "r-1",
            &pending_payload(),
        )?;

        let body = r#"{"benchmark_id":"b1","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#;
        let signed = signed_decoy(&server);
        let list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body(format!("[{body}]"));
        });
        let detail = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(200).body(body);
        });
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });

        args(None, Some(&server.base_url()))?.execute(
            &temp.ws,
            &http()?,
            &mut CapturedConsole::default(),
        )?;

        signed.assert_calls(0);
        list.assert();
        detail.assert();
        submit.assert_calls(0);
        let index: Vec<String> = temp
            .ws
            .benchmarks()
            .list_remote_index()?
            .into_iter()
            .map(|summary| summary.benchmark_id)
            .collect();
        assert_eq!(index, vec!["b1".to_string()]);
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemotePending)?,
            vec!["r-1".to_string()],
            "the pending result is untouched, not submitted"
        );
        Ok(())
    }

    /// Refused before the pull, so the command does not half-succeed; the mock
    /// server it was pointed at sees no request.
    #[test]
    fn an_unregistered_sync_refuses_a_result_id() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-unregistered-result")?;
        let server = MockServer::start();
        let list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });

        let error = args(Some("r-1"), Some(&server.base_url()))?
            .execute(&temp.ws, &http()?, &mut CapturedConsole::default())
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;

        assert!(
            matches!(error, Error::SyncSubmitNeedsRegistration),
            "unexpected error: {error}"
        );
        list.assert_calls(0);
        Ok(())
    }

    /// With neither a registration nor a server, the registration is the one
    /// reported: naming a server would not make the submit possible.
    #[test]
    fn a_result_id_reports_the_registration_not_the_missing_server() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-result-no-server")?;
        let error = args(Some("r-1"), None)?
            .execute(&temp.ws, &http()?, &mut CapturedConsole::default())
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a refusal"))?;
        assert!(
            matches!(error, Error::SyncSubmitNeedsRegistration),
            "unexpected error: {error}"
        );
        Ok(())
    }

    /// A non-eval benchmark id keeps the score step at `NotServerScored`, so
    /// the run needs no `/jobs` mock to reach the end.
    #[test]
    fn a_registered_sync_pulls_submits_and_scores() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-registered-full")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;

        let results = temp.ws.results();
        results.save_payload(
            BenchmarkResultLocation::RemotePending,
            "r-1",
            &pending_payload(),
        )?;

        let list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });

        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;

        list.assert();
        submit.assert();
        assert!(
            results
                .list_ids(BenchmarkResultLocation::RemotePending)?
                .is_empty(),
            "the pending result moved out of pending"
        );
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemoteSynced)?,
            vec!["job-1".to_string()],
            "and landed under the server-minted job id"
        );
        Ok(())
    }

    /// `r-1-bad` sorts ahead of `r-2-good`, so the good result reaches the
    /// server only if the drain continues past the bad one.
    #[test]
    fn a_corrupt_pending_payload_does_not_stop_the_drain() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-corrupt-payload")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;

        let results = temp.ws.results();
        let bad = results.payload_path(BenchmarkResultLocation::RemotePending, "r-1-bad");
        std::fs::create_dir_all(
            bad.parent()
                .ok_or_else(|| anyhow::anyhow!("payload path has no parent"))?,
        )?;
        std::fs::write(&bad, b"{ not json")?;
        results.save_payload(
            BenchmarkResultLocation::RemotePending,
            "r-2-good",
            &pending_payload(),
        )?;

        let list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });
        let submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });

        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;

        list.assert();
        submit.assert();
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemoteSynced)?,
            vec!["job-1".to_string()],
            "the good result submitted despite the corrupt one ahead of it"
        );
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemotePending)?,
            vec!["r-1-bad".to_string()],
            "and the corrupt one stays pending for a later run"
        );
        Ok(())
    }

    /// A tolerated failure has to reach the summary line, since the command
    /// still exits `0`; the zero case stays quiet.
    #[rstest::rstest]
    #[case::clean(3, 0, "submitted 3 result(s)")]
    #[case::some_failed(2, 1, "submitted 2 result(s), 1 failed (see log)")]
    #[case::all_failed(0, 2, "submitted 0 result(s), 2 failed (see log)")]
    fn the_submit_summary_names_failures(
        #[case] synced: usize,
        #[case] failed: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(submit_summary(synced, failed), expected);
    }

    /// Unlike the drain, a refused submit is not tolerated, and a result that
    /// is not pending is an error rather than a zero count.
    #[test]
    fn a_named_result_submits_alone_and_its_failure_is_the_commands() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-named-result")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;
        let results = temp.ws.results();
        ["r-1", "r-2"].into_iter().try_for_each(|id| {
            results.save_payload(
                BenchmarkResultLocation::RemotePending,
                id,
                &pending_payload(),
            )
        })?;
        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });
        let mut submit = server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });

        args(Some("r-1"), None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;
        submit.assert_calls(1);
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemotePending)?,
            vec!["r-2".to_string()],
            "only the named result was submitted"
        );

        let error = args(Some("r-missing"), None)?
            .execute(&temp.ws, &http()?, &mut CapturedConsole::default())
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unknown result should fail the command"))?;
        assert!(
            error.to_string().contains("r-missing"),
            "unexpected error: {error}"
        );

        submit.delete();
        server.mock(|when, then| {
            when.method(POST).path("/benchmarks");
            then.status(500).body("nope");
        });
        assert!(
            args(Some("r-2"), None)?
                .execute(&temp.ws, &http()?, &mut CapturedConsole::default())
                .is_err(),
            "a refused named submit is the command's failure"
        );
        assert_eq!(
            results.list_ids(BenchmarkResultLocation::RemotePending)?,
            vec!["r-2".to_string()],
            "and the result stays pending"
        );
        Ok(())
    }

    /// A synced eval result; `metrics` seeds the already-scored state.
    fn synced_eval(
        results: &crate::results::ResultsStore,
        job_id: &str,
        scored: bool,
    ) -> anyhow::Result<()> {
        results.save_payload(
            BenchmarkResultLocation::RemoteSynced,
            job_id,
            &serde_json::json!({
                "benchmark_id": "eval_ifbench",
                "model_descriptor": "{\"type\":\"mlx\"}",
                "runtime_descriptor": "{\"type\":\"mlx_macos_pipette\"}",
            }),
        )?;
        if scored {
            results.save_metrics(
                BenchmarkResultLocation::RemoteSynced,
                job_id,
                &serde_json::json!({"scored_at": "2026-01-01T00:00:00Z", "metrics": []}),
            )?;
        }
        Ok(())
    }

    /// The score step walks every synced eval job: a processed one lands its
    /// metrics on disk, an unprocessed one stays pending, one the server no
    /// longer knows is skipped, and one already scored is not re-fetched.
    #[test]
    fn the_score_step_refreshes_only_the_eval_jobs_that_need_it() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-score")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;
        let results = temp.ws.results();
        synced_eval(&results, "job-done", false)?;
        synced_eval(&results, "job-wait", false)?;
        synced_eval(&results, "job-gone", false)?;
        synced_eval(&results, "job-scored", true)?;
        results.save_payload(
            BenchmarkResultLocation::RemoteSynced,
            "job-prefill",
            &pending_payload(),
        )?;

        server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });
        let done = server.mock(|when, then| {
            when.method(GET).path("/jobs/job-done");
            then.status(200).body(
                r#"{"status":"processed","scored_at":"2026-01-02T00:00:00Z","metrics":[{"metric":"accuracy","value":0.5,"unit":"ratio"}]}"#,
            );
        });
        let wait = server.mock(|when, then| {
            when.method(GET).path("/jobs/job-wait");
            then.status(200)
                .body(r#"{"status":"scoring","metrics":null}"#);
        });
        let gone = server.mock(|when, then| {
            when.method(GET).path("/jobs/job-gone");
            then.status(404).body("no such job");
        });
        let scored = server.mock(|when, then| {
            when.method(GET).path("/jobs/job-scored");
            then.status(200)
                .body(r#"{"status":"processed","metrics":[]}"#);
        });
        let prefill = server.mock(|when, then| {
            when.method(GET).path("/jobs/job-prefill");
            then.status(200)
                .body(r#"{"status":"processed","metrics":[]}"#);
        });

        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;

        done.assert();
        wait.assert();
        gone.assert();
        scored.assert_calls(0);
        prefill.assert_calls(0);
        assert!(results.has_valid_metrics(BenchmarkResultLocation::RemoteSynced, "job-done"));
        assert!(!results.has_valid_metrics(BenchmarkResultLocation::RemoteSynced, "job-wait"));
        assert!(!results.has_valid_metrics(BenchmarkResultLocation::RemoteSynced, "job-gone"));
        Ok(())
    }

    #[test]
    fn only_a_404_reads_as_not_found() {
        let http_status = |status: u16| {
            Error::Mgmt(MgmtError::HttpStatus {
                method: "GET".to_owned(),
                url: "https://mgmt.example.test/jobs/j".to_owned(),
                status,
                body: String::new(),
            })
        };
        assert!(is_not_found(&http_status(404)));
        assert!(!is_not_found(&http_status(500)));
        assert!(!is_not_found(&Error::SyncServerUrlUnknown));
    }

    /// A registered workspace signs the pull as its own registration, list
    /// and detail alike: an unsigned one would quietly miss what the server
    /// keeps to this device.
    #[test]
    fn a_registered_sync_signs_the_catalog_pull() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-signed-pull")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;

        let body = r#"{"benchmark_id":"b1","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#;
        let list = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks")
                .header("X-Client-Id", "ev1_test")
                .header_exists("X-Signature");
            then.status(200).body(format!("[{body}]"));
        });
        let detail = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header("X-Client-Id", "ev1_test")
                .header_exists("X-Signature");
            then.status(200).body(body);
        });

        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;

        list.assert();
        detail.assert();
        Ok(())
    }

    /// The revalidating pull signs too. A second `sync` carries the ETags the
    /// first one stored, and the signature has to ride beside `If-None-Match`
    /// rather than be dropped on the cheaper path; the `304`s then keep the
    /// catalog the signed pull filled.
    #[test]
    fn a_registered_sync_signs_the_revalidating_pull() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-signed-revalidate")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;

        let body = r#"{"benchmark_id":"b1","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#;
        let mut list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200)
                .header("etag", "\"list-v1\"")
                .body(format!("[{body}]"));
        });
        let mut detail = server.mock(|when, then| {
            when.method(GET).path("/benchmarks/b1");
            then.status(200).header("etag", "\"b1-v1\"").body(body);
        });
        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;
        list.delete();
        detail.delete();

        let list = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks")
                .header("If-None-Match", "\"list-v1\"")
                .header("X-Client-Id", "ev1_test")
                .header_exists("X-Signature");
            then.status(304).header("etag", "\"list-v1\"");
        });
        let detail = server.mock(|when, then| {
            when.method(GET)
                .path("/benchmarks/b1")
                .header("If-None-Match", "\"b1-v1\"")
                .header("X-Client-Id", "ev1_test")
                .header_exists("X-Signature");
            then.status(304).header("etag", "\"b1-v1\"");
        });
        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;

        list.assert();
        detail.assert();
        let index: Vec<String> = temp
            .ws
            .benchmarks()
            .list_remote_index()?
            .into_iter()
            .map(|summary| summary.benchmark_id)
            .collect();
        assert_eq!(index, vec!["b1".to_string()], "the 304s kept the catalog");
        Ok(())
    }

    /// A registration whose key is gone cannot sign, but it still names a
    /// server, and the unsigned pull has to go there rather than be refused.
    #[test]
    fn a_registration_without_a_key_pulls_from_its_own_server() -> anyhow::Result<()> {
        let temp = TempWorkspace::new("sync-keyless")?;
        let server = MockServer::start();
        temp.register(&server.base_url())?;
        temp.ws.registration().delete_private_key()?;

        let signed = signed_decoy(&server);
        let list = server.mock(|when, then| {
            when.method(GET).path("/benchmarks");
            then.status(200).body("[]");
        });

        assert!(signing_identity_if_registered(&temp.ws.registration())?.is_none());
        args(None, None)?.execute(&temp.ws, &http()?, &mut CapturedConsole::default())?;
        signed.assert_calls(0);
        list.assert();
        Ok(())
    }
}
