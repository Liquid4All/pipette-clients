//! A chat-completion error against a live llama-server must not stall the
//! vlm_eval pool. After a failure the client waits up to
//! `CRASH_OBSERVATION_WINDOW` (3s) to see whether the child died, and must do
//! so with the server slot released: failures against a live server are
//! routine under `parallel` (llama-server answers an undersized per-slot
//! context with a 4xx/5xx and keeps running, and one image can be most of a
//! slot's budget), and holding the slot would serialize every window.

mod common;

use std::time::{Duration, Instant};

use anyhow::Context;
use common::{
    Expect, assert_samples, completions_by_id, ignore_reps, no_readiness_gate, temp_store,
    vlm_pool_request, vlm_sample,
};

use pipette_llamacpp::run;
use pipette_plan_types::BenchmarkResultData;

const PARALLEL: u32 = 4;
/// Two rendezvous batches: the first fails wholesale, the second proves the
/// pool still overlaps afterwards.
const TOTAL_SAMPLES: usize = 8;

/// Serialized observation windows would cost `PARALLEL * 3s = 12s` plus the
/// run itself; overlapping ones cost a little over 3s. Anything under this
/// bound can only be the overlapping shape, with room left for a loaded
/// machine.
const SERIALIZED_WINDOW_BUDGET: Duration = Duration::from_secs(10);

#[test]
fn parallel_vlm_eval_does_not_serialize_on_transport_failures() -> anyhow::Result<()> {
    const DROP_MARKER: &str = "PLEASE-DROP-ME";

    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_DROP_ONCE_PROMPT_CONTAINS", DROP_MARKER) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS", PARALLEL.to_string()) };
    let store = temp_store("vlm-parallel-transport")?;
    // Under the store's temp dir, so it goes with the store even on a failure.
    let pid_log = store.root.join("pids.txt");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PID_FILE", &pid_log) };
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| {
            let text = if i < PARALLEL as usize {
                format!("sample {i} {DROP_MARKER}")
            } else {
                format!("sample {i}")
            };
            vlm_sample(i, &text)
        })
        .collect();
    let req = vlm_pool_request(
        &store.root,
        "vlm-parallel-transport-test",
        samples,
        PARALLEL,
        // The per-slot share must clear the cell's calibrated need (~8208).
        Some(16384 * PARALLEL),
    )?;

    let started = Instant::now();
    let outcome = run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .context("parallel vlm_eval should survive transport failures")?;
    let elapsed = started.elapsed();

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    let by_id = completions_by_id(&completions, TOTAL_SAMPLES)?;

    assert_samples(&by_id, 0..TOTAL_SAMPLES, Expect::Answered("answer"))?;

    let pids = std::fs::read_to_string(&pid_log).context("pid log should exist")?;
    let lines: Vec<&str> = pids.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        lines.len(),
        1,
        "a live server must not be restarted; pid log: {pids:?}",
    );

    assert!(
        elapsed < SERIALIZED_WINDOW_BUDGET,
        "the crash observation windows must overlap: {PARALLEL} simultaneous \
         transport failures took {elapsed:?}, budget {SERIALIZED_WINDOW_BUDGET:?}",
    );

    Ok(())
}
