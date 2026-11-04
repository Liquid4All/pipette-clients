//! The `parallel` runtime flag must overlap samples and keep resume working.
//! Both are checked in one run because they interact: a resumed sample is
//! never sent, so the pool must fill from the rest rather than stall.
//! Overlap is asserted through the fake server's rendezvous, not timed.

mod common;

use anyhow::Context;
use common::{
    Expect, assert_distinct_knobs, assert_samples, completions_by_id, ignore_reps,
    labeled_temp_dir, no_corpus, no_readiness_gate, pool_request, sample_id, with_distinct_knobs,
};
use serde_json::json;

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
fn parallel_eval_overlaps_samples_and_still_resumes() -> anyhow::Result<()> {
    // SAFETY: this file holds a single test, so no parallel test reads these vars.
    unsafe { std::env::set_var("FAKE_LLAMA_RENDEZVOUS", PARALLEL.to_string()) };
    let peak_file = std::env::temp_dir().join(format!(
        "pipette-eval-parallel-peak-{}-{}.txt",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_PEAK_INFLIGHT_FILE", &peak_file) };

    let root = labeled_temp_dir("parallel")?;
    let run_context = no_corpus(root.path());
    let samples: Vec<_> = (0..TOTAL_SAMPLES)
        .map(|i| {
            json!({
                "id": sample_id(i),
                "messages": [{"role": "user", "content": format!("sample {i}")}],
            })
        })
        .collect();
    // `ctx_size` unset: the per-cell default is multiplied by the slot count.
    // The crash/transport tests state a total.
    let req = with_distinct_knobs(pool_request(
        root.path(),
        "parallel-test",
        samples,
        PARALLEL,
        None,
    )?)?;

    // Pre-seed part of the checkpoint without finalizing, as an interrupted
    // run leaves it. The digest covers `runtime_flags`, so open it against
    // the request the run will use.
    {
        let mut checkpoint = run_context.eval_completions().open(&req)?;
        (0..PRESEEDED).try_for_each(|i| {
            checkpoint.append(BenchmarkEvalCompletion {
                id: sample_id(i),
                completion: RESUMED_TEXT.into(),
                ..Default::default()
            })
        })?;
    }

    let outcome = run(&req, &run_context, &no_readiness_gate, ignore_reps(), None)
        .context("parallel eval run should succeed")?;
    // Eval reports the cell's flags as authored, with no derived context.
    assert_eq!(outcome.runtime_flags, req.runtime_flags);
    assert_distinct_knobs(&req.benchmark, &outcome.command)?;
    // The server got one slot's default (8192 plus the 16 max tokens) per slot,
    // and llama.cpp's own load mode.
    let ctx = ((8192 + 16) * PARALLEL).to_string();
    anyhow::ensure!(
        outcome
            .command
            .windows(2)
            .any(|w| w == ["--ctx-size", ctx.as_str()]),
        "{:?}",
        outcome.command
    );
    anyhow::ensure!(
        !outcome
            .command
            .iter()
            .any(|token| token == "--load-mode" || token == "--no-mmap"),
        "{:?}",
        outcome.command
    );

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };

    let by_id = completions_by_id(&completions, TOTAL_SAMPLES)?;
    assert_samples(&by_id, 0..PRESEEDED, Expect::Answered(RESUMED_TEXT))?;
    assert_samples(&by_id, PRESEEDED..TOTAL_SAMPLES, Expect::Answered("answer"))?;

    let peak: usize = std::fs::read_to_string(&peak_file)
        .context("fake server should have recorded a peak in-flight count")?
        .trim()
        .parse()
        .context("peak in-flight count should be a number")?;
    assert_eq!(
        peak, PARALLEL as usize,
        "expected {PARALLEL} concurrent /completion requests, saw {peak}",
    );

    let _ = std::fs::remove_file(&peak_file);
    Ok(())
}
