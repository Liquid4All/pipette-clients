//! A mid-run `llama-server` crash under `parallel` must restart the server
//! once per crash (every in-flight worker sees the same dead child, and each
//! spawning a replacement would leak servers), and must fail only the poison
//! prompt: the peers it took down are each sent again alone, since resume
//! treats a failed id as done. The fake server's rendezvous makes the crash
//! land with peers mid-request.

mod common;

use anyhow::Context;
use common::{
    Expect, assert_samples, completions_by_id, ignore_reps, labeled_temp_dir, no_corpus,
    no_readiness_gate, pool_request, sample_id,
};
use serde_json::json;

use pipette_llamacpp::run;
use pipette_plan_types::BenchmarkResultData;

const PARALLEL: u32 = 4;
/// Two full rendezvous batches: the first one crashes, the second must run
/// against the restarted server.
const TOTAL_SAMPLES: usize = 8;
/// Inside the first batch, so the crash lands while its peers are parked.
const CRASH_INDEX: usize = 1;

#[test]
fn parallel_eval_restarts_the_server_exactly_once_per_crash() -> anyhow::Result<()> {
    const CRASH_MARKER: &str = "PLEASE-CRASH-NOW";

    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_CRASH_PROMPT_CONTAINS", CRASH_MARKER) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS", PARALLEL.to_string()) };
    // Resends go one at a time and never reach the rendezvous target.
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS_TIMEOUT_MS", "1000") };
    // The crash fires before this delay, so the exiting process beats its
    // parked peers' replies.
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS_GRACE_MS", "250") };
    let pid_log = std::env::temp_dir().join(format!(
        "pipette-eval-parallel-crash-pids-{}-{}.txt",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log) };

    let root = labeled_temp_dir("parallel-crash")?;
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| {
            let content = if i == CRASH_INDEX {
                format!("sample {i} {CRASH_MARKER}")
            } else {
                format!("sample {i}")
            };
            json!({
                "id": sample_id(i),
                "messages": [{"role": "user", "content": content}],
            })
        })
        .collect();
    // An explicit total exercises the path that passes `--ctx-size` verbatim.
    let req = pool_request(
        root.path(),
        "parallel-crash-test",
        samples,
        PARALLEL,
        Some(4096 * PARALLEL),
    )?;

    let outcome = run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("parallel eval should recover from the mid-run crash")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    let by_id = completions_by_id(&completions, TOTAL_SAMPLES)?;

    let crasher_id = sample_id(CRASH_INDEX);
    assert_samples(
        &by_id,
        [CRASH_INDEX],
        Expect::Failed("crashed mid-completion"),
    )?;

    // The second batch ran against the restarted server.
    assert_samples(
        &by_id,
        PARALLEL as usize..TOTAL_SAMPLES,
        Expect::Answered("answer"),
    )?;

    let failed: Vec<&str> = completions
        .iter()
        .filter(|c| c.failed)
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(
        failed,
        vec![crasher_id.as_str()],
        "only the poison prompt may stay failed; failed={failed:?}",
    );
    assert_samples(
        &by_id,
        (0..PARALLEL as usize).filter(|&i| i != CRASH_INDEX),
        Expect::Answered("answer"),
    )?;

    // Three processes: the first, the replacement after the pooled crash, and
    // the one after the poison prompt, resent alone, crashed it again.
    let pids = std::fs::read_to_string(&pid_log).context("pid log should exist")?;
    let lines: Vec<&str> = pids.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        3,
        "each crash must restart llama-server exactly once; pid log: {pids:?}",
    );
    let unique: std::collections::HashSet<&str> = lines.iter().copied().collect();
    assert_eq!(
        unique.len(),
        lines.len(),
        "every restart must be a fresh process; pid log: {pids:?}",
    );

    let _ = std::fs::remove_file(&pid_log);
    Ok(())
}
