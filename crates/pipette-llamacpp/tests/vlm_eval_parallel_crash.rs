//! A mid-run `llama-server` crash under a vlm_eval pool must restart the
//! server once per crash, fail only the poison prompt (as in
//! `eval_parallel_crash.rs`), and bring the projector back: a VL server
//! restarted without `--mmproj` does not fail loudly, it answers every image
//! sample with a template error that scores as a wrong answer. So the argv of
//! every spawned process is checked, not just the count. The fake server's
//! rendezvous makes the crash land with peers mid-request.

mod common;

use anyhow::Context;
use common::{
    Expect, assert_samples, completions_by_id, ignore_reps, no_readiness_gate, sample_id,
    temp_store, vlm_pool_request, vlm_sample,
};

use pipette_llamacpp::run;
use pipette_plan_types::BenchmarkResultData;

const PARALLEL: u32 = 4;
/// Two full rendezvous batches: the first one crashes, the second must run
/// against the restarted server.
const TOTAL_SAMPLES: usize = 8;
/// Inside the first batch, so the crash lands while its peers are parked.
const CRASH_INDEX: usize = 1;

#[test]
fn parallel_vlm_eval_restarts_the_server_exactly_once_per_crash() -> anyhow::Result<()> {
    const CRASH_MARKER: &str = "PLEASE-CRASH-NOW";

    // The poison prompt stays failed, which `ensure_complete` would refuse;
    // that contract has its own tests.
    // SAFETY: every test in this binary sets the same value.
    unsafe { std::env::set_var("PIPETTE_ALLOW_INCOMPLETE_EVAL", "1") };

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
    let store = temp_store("vlm-parallel-crash")?;
    // Under the store's temp dir, so they go with the store even on a failure.
    let pid_log = store.root.join("pids.txt");
    let argv_log = store.root.join("argv.txt");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_ARGV_FILE", &argv_log) };
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| {
            let text = if i == CRASH_INDEX {
                format!("sample {i} {CRASH_MARKER}")
            } else {
                format!("sample {i}")
            };
            vlm_sample(i, &text)
        })
        .collect();
    // An explicit total exercises the path that passes `--ctx-size` verbatim.
    // Its per-slot share must clear the cell's calibrated need (~8208: image,
    // text, output and margins), not just max_tokens.
    let req = vlm_pool_request(
        &store.root,
        "vlm-parallel-crash-test",
        samples,
        PARALLEL,
        Some(16384 * PARALLEL),
    )?;

    let outcome = run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .context("parallel vlm_eval should recover from the mid-run crash")?;

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
    let distinct: std::collections::HashSet<&str> = lines.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        lines.len(),
        "every restart must be a fresh process; pid log: {pids:?}",
    );

    // Nothing above would catch a replacement without `--mmproj`.
    let argvs = std::fs::read_to_string(&argv_log).context("argv log should exist")?;
    let argv_lines: Vec<&str> = argvs.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        argv_lines.len(),
        lines.len(),
        "one argv line per process; argv log: {argvs:?}",
    );
    for argv in &argv_lines {
        assert!(
            argv.contains("--mmproj"),
            "a restarted VL server must still get the projector; argv: {argv:?}",
        );
        assert!(
            argv.contains(&format!("--parallel {PARALLEL}")),
            "a restarted server must reopen the same slots; argv: {argv:?}",
        );
    }

    Ok(())
}
