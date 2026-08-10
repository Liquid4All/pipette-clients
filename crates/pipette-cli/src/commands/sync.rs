use clap::Args;

use pipette_http::HttpClient;
use pipette_mgmt_client::{error::Error as MgmtError, MgmtClient};

use crate::client::sync::{
    pull_remote_benchmarks, refresh_scored_job, score_refresh_state, submit_pending_result,
    ScoreRefreshState,
};
use crate::error::{Error, Result};
use crate::results::BenchmarkResultLocation;
use crate::workspace::PipetteWorkspace;

pub const SYNC_LONG_ABOUT: &str = "\
Synchronize remote benchmark state with the management server.

The command pulls benchmark definitions, submits pending results, and
refreshes synced jobs for scores. Pass a result ID to narrow the submit
and score steps to that one result; definitions are pulled either way.";

pub const SYNC_AFTER_HELP: &str = "\
Examples:
  pipette sync
  pipette sync <result-id>

Only results created from remote benchmarks are submitted upstream. Results from
local benchmark definitions stay local.";

/// Sync benchmarks and results with the management server
#[derive(Args, Debug)]
pub struct SyncArgs {
    /// Result ID to submit (omit to submit all pending)
    pub result_id: Option<String>,
}

impl SyncArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> Result<()> {
        let session = ws.mgmt_session()?;
        let results = ws.results();
        let auth = &session.auth;
        let server_url = &session.server_url;

        log::info!("pulling remote benchmarks from {server_url}");
        let client = MgmtClient::new(server_url)?;
        let count = pull_remote_benchmarks(&ws.benchmarks(), &client, http, auth)?;
        println!("pulled {count} benchmark(s)");

        let mut synced_job_ids: Vec<String> = Vec::new();
        if let Some(result_id) = &self.result_id {
            log::info!("submitting pending result {result_id}");
            let payload =
                results.load_payload(BenchmarkResultLocation::RemotePending, result_id)?;
            let job_id = submit_pending_result(&results, &client, http, auth, result_id, &payload)?;
            log::info!("submitted result {result_id} as job {job_id}");
            synced_job_ids.push(job_id);
        } else {
            for result_id in results.list_ids(BenchmarkResultLocation::RemotePending)? {
                log::info!("submitting pending result {result_id}");
                let payload =
                    results.load_payload(BenchmarkResultLocation::RemotePending, &result_id)?;
                match submit_pending_result(&results, &client, http, auth, &result_id, &payload) {
                    Ok(job_id) => {
                        log::info!("submitted result {result_id} as job {job_id}");
                        synced_job_ids.push(job_id);
                    }
                    Err(e) => log::info!("failed to submit {result_id}: {e:#}"),
                }
            }
        }
        println!("submitted {} result(s)", synced_job_ids.len());

        let candidate_ids = if self.result_id.is_some() {
            synced_job_ids
        } else {
            results.list_ids(BenchmarkResultLocation::RemoteSynced)?
        };
        let mut eval_jobs: Vec<String> = Vec::new();
        let mut already_scored = 0usize;
        for id in candidate_ids {
            match score_refresh_state(&results, &id) {
                ScoreRefreshState::NeedsRefresh => eval_jobs.push(id),
                ScoreRefreshState::AlreadyScored => already_scored += 1,
                ScoreRefreshState::NotEval => {}
            }
        }
        let (scored, pending) =
            eval_jobs
                .iter()
                .fold(
                    (0usize, 0usize),
                    |(scored, pending), job_id| match refresh_scored_job(
                        &results, &client, http, auth, job_id,
                    ) {
                        Ok(true) => (scored + 1, pending),
                        Ok(false) => (scored, pending + 1),
                        Err(err) => {
                            if is_not_found(&err) {
                                log::info!("job {job_id}: not found on server, skipping");
                            } else {
                                log::info!("job {job_id}: refresh failed: {err:#}");
                            }
                            (scored, pending)
                        }
                    },
                );
        println!(
            "scored {} result(s) ({already_scored} already scored), {pending} pending",
            scored + already_scored
        );

        Ok(())
    }
}

fn is_not_found(err: &Error) -> bool {
    matches!(err, Error::Mgmt(MgmtError::HttpStatus { status, .. }) if *status == 404)
}
