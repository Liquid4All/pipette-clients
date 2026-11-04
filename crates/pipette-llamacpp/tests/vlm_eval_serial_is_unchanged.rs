//! An unset `parallel` keeps vlm_eval serial. The argv half is pinned by
//! `execute::vlm_eval`'s unit tests; this is the behavioural half. The fake
//! server parks each completion until two are in flight, which a pool reaches
//! at once and a serial client never does. The rendezvous timeout is short
//! because the serial path waits it out once per sample.

mod common;

use anyhow::Context;
use common::{
    Expect, assert_samples, completions_by_id, ignore_reps, no_readiness_gate, temp_store,
    vlm_pool_request, vlm_sample,
};

use pipette_llamacpp::run;
use pipette_plan_types::BenchmarkResultData;
use pipette_plan_types::VlmEvalLlamacppCliStockToolsGgufVision;
use pipette_plan_types::{RuntimeFlagRef, RuntimeFlags};

const TOTAL_SAMPLES: usize = 3;

#[test]
fn vlm_eval_without_the_knob_stays_one_sample_at_a_time() -> anyhow::Result<()> {
    // SAFETY: integration tests run in their own process per file; this
    // file holds a single test so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS", "2") };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS_TIMEOUT_MS", "200") };
    let store = temp_store("vlm-serial")?;
    // Inside the store's temp dir so it is removed even on a failure.
    let peak_file = store.root.join("peak-inflight.txt");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PEAK_INFLIGHT_FILE", &peak_file) };
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| vlm_sample(i, &format!("sample {i}")))
        .collect();
    // Built by the pool helper, then the knob unset, so it differs from the
    // parallel tests in exactly one field.
    let mut req = vlm_pool_request(&store.root, "vlm-serial-test", samples, 1, None)?;
    let RuntimeFlags::VlmEvalLlamacppCliStockToolsGgufVision(
        VlmEvalLlamacppCliStockToolsGgufVision { parallel, .. },
    ) = req
        .runtime_flags
        .as_mut()
        .context("the helper always sets runtime flags")?
    else {
        anyhow::bail!("expected vlm_eval llama.cpp flags");
    };
    *parallel = None;

    let outcome = run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .context("serial vlm_eval run should succeed")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    let by_id = completions_by_id(&completions, TOTAL_SAMPLES)?;
    assert_samples(&by_id, 0..TOTAL_SAMPLES, Expect::Answered("answer"))?;
    assert!(
        completions.iter().all(|c| !c.failed),
        "no sample should have failed"
    );

    // Rendered flags persist with the result, so a `parallel: 1` here would
    // make otherwise-identical runs compare as different configurations.
    let reported = outcome
        .runtime_flags
        .context("the run must report its resolved flags")?;
    assert_eq!(RuntimeFlagRef::from(reported).parallel, None);

    let peak: usize = std::fs::read_to_string(&peak_file)
        .context("fake server should have recorded a peak in-flight count")?
        .trim()
        .parse()
        .context("peak in-flight count should be a number")?;
    assert_eq!(
        peak, 1,
        "an unset `parallel` must keep one chat completion in flight, saw {peak}",
    );

    Ok(())
}
