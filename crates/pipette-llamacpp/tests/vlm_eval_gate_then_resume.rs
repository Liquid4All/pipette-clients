//! A vlm_eval cell that loses a row is refused, its checkpoint survives the
//! refusal, and re-running retries only that row and completes.
//!
//! The checkpoint is last-write-wins and the server always answers "answer",
//! so only the fake server's per-sample call count shows the resume did not
//! re-infer the rows that already answered.

mod common;

use anyhow::Context;
use common::{ignore_reps, no_readiness_gate, temp_store, vlm_pool_request, vlm_sample};

use pipette_llamacpp::run;
use pipette_plan_types::BenchmarkResultData;

#[test]
fn a_refused_cell_resumes_and_completes() -> anyhow::Result<()> {
    const DROP_MARKER: &str = "PLEASE-DROP-CONNECTION";

    // SAFETY: integration tests get a process per file; this file holds one test.
    unsafe { std::env::set_var("FAKE_LLAMA_DROP_PROMPT_CONTAINS", DROP_MARKER) };
    // SAFETY: as above.
    unsafe { std::env::remove_var("PIPETTE_ALLOW_INCOMPLETE_EVAL") };

    let store = temp_store("vlm-gate-then-resume")?;
    let call_log = store.root.join("calls.log");
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_CALL_LOG", &call_log) };
    let calls_mentioning = |needle: &str| -> usize {
        std::fs::read_to_string(&call_log)
            .map(|log| log.lines().filter(|l| l.contains(needle)).count())
            .unwrap_or(0)
    };

    let samples = vec![
        vlm_sample(0, "first sample"),
        vlm_sample(1, &format!("middle {DROP_MARKER}")),
        vlm_sample(2, "third sample"),
    ];
    let req = vlm_pool_request(&store.root, "vlm-gate-then-resume", samples, 1, None)?;

    let err = match run(&req, &store, &no_readiness_gate, ignore_reps(), None) {
        Ok(_) => anyhow::bail!("a cell that lost a row must not be submitted"),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("is incomplete") && err.contains("s-01"),
        "the refusal must name the failed row: {err}"
    );
    assert!(err.contains("run this cell again"), "{err}");

    let checkpoints: Vec<_> = std::fs::read_dir(store.root.join("evals"))?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .collect();
    assert_eq!(checkpoints.len(), 1, "the refusal must keep the checkpoint");
    let before = std::fs::read_to_string(checkpoints[0].path())?;
    assert!(
        before.contains("s-00") && before.contains("s-02"),
        "{before}"
    );
    // The failed row names this runner's endpoint, not the text eval's.
    assert!(
        before.contains("llama-server /v1/chat/completions failed"),
        "{before}"
    );
    let answered_before = (
        calls_mentioning("first sample"),
        calls_mentioning("third sample"),
    );
    assert_eq!(answered_before, (1, 1));

    // SAFETY: as above.
    unsafe { std::env::remove_var("FAKE_LLAMA_DROP_PROMPT_CONTAINS") };

    let outcome = run(&req, &store, &no_readiness_gate, ignore_reps(), None)
        .context("the resume should retry the failed row and complete the cell")?;
    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    assert_eq!(completions.len(), 3);
    assert!(
        completions
            .iter()
            .all(|c| !c.failed && c.completion == "answer")
    );
    assert_eq!(
        (
            calls_mentioning("first sample"),
            calls_mentioning("third sample"),
        ),
        answered_before,
        "the rows that answered must not be re-inferred on resume"
    );
    Ok(())
}
