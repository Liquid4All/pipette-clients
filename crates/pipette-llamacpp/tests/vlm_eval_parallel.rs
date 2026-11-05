//! The `parallel` runtime flag must overlap vlm_eval samples and keep resume
//! working. Deliberately mirrors `eval_parallel.rs`: both runners share one
//! pool, so a divergence between the files means the pool grew a text-only
//! assumption. Overlap is asserted through the fake server's rendezvous, not
//! timed.

mod common;

use anyhow::Context;
use common::{
    Expect, assert_samples, completions_by_id, ignore_reps, no_readiness_gate, sample_id,
    temp_store, vlm_pool_request, vlm_sample,
};

use pipette_llamacpp::run;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{BenchmarkEvalCompletion, BenchmarkResultData};

const PARALLEL: u32 = 4;
/// A multiple of `PARALLEL` once the pre-seeded ids are removed, so every
/// rendezvous batch fills and none has to wait out the server's timeout.
const TOTAL_SAMPLES: usize = 12;
const PRESEEDED: usize = 4;
/// Text of the rows written straight into the checkpoint; the fake server
/// only answers `answer`, so seeing this proves the loop never re-ran them.
const RESUMED_TEXT: &str = "PRESEEDED";

#[test]
fn parallel_vlm_eval_overlaps_samples_and_still_resumes() -> anyhow::Result<()> {
    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS", PARALLEL.to_string()) };
    let store = temp_store("vlm-parallel")?;
    let peak_file = store.root.join("peak-inflight.txt");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PEAK_INFLIGHT_FILE", &peak_file) };
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| vlm_sample(i, &format!("sample {i}")))
        .collect();
    // `ctx_size` unset: the calibrated per-slot budget is multiplied by the
    // slot count, which lets `parallel` be set on its own. The
    // crash/transport tests state a total.
    let req = vlm_pool_request(&store.root, "vlm-parallel-test", samples, PARALLEL, None)?;

    // Pre-seed part of the checkpoint without finalizing, as an interrupted
    // run leaves it. The digest covers `runtime_flags`, so open it against
    // the request the run will use.
    {
        let mut checkpoint = store.eval_completions().open(&req)?;
        (0..PRESEEDED).try_for_each(|i| {
            checkpoint.append(BenchmarkEvalCompletion {
                id: sample_id(i),
                completion: RESUMED_TEXT.into(),
                ..Default::default()
            })
        })?;
    }

    let outcome = run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .context("parallel vlm_eval run should succeed")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    let by_id = completions_by_id(&completions, TOTAL_SAMPLES)?;
    assert_samples(&by_id, 0..PRESEEDED, Expect::Answered(RESUMED_TEXT))?;
    assert_samples(&by_id, PRESEEDED..TOTAL_SAMPLES, Expect::Answered("answer"))?;
    assert!(
        completions.iter().all(|c| !c.failed),
        "no sample should have failed"
    );

    let peak: usize = std::fs::read_to_string(&peak_file)
        .context("fake server should have recorded a peak in-flight count")?
        .trim()
        .parse()
        .context("peak in-flight count should be a number")?;
    assert_eq!(
        peak, PARALLEL as usize,
        "expected {PARALLEL} concurrent chat completions, saw {peak}",
    );

    Ok(())
}
