//! End-to-end integration test for per-sample `stop_reason` (PIP-274/PIP-323):
//! drives `execute::run` with the fake `llama-server`
//! (`tests/bin/fake_llama_server.rs`) and asserts the eval loop classifies a
//! natural stop as `eos` and a sample that hits the `n_predict` cap as
//! `truncated`, carrying the server-reported token count through. The
//! `failure` path is covered by `eval_crash_recovery.rs`.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps,
    no_readiness_gate,
};
use serde_json::json;

use pipette_doomloop::plan::DoomloopOverrides;
use pipette_llamacpp::execute::run;
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::benchmark::{BenchmarkDefinition, EvalBenchmark};
use pipette_plan_types::result::{BenchmarkEvalCompletionStopReason, BenchmarkResultData};
use pipette_plan_types::BenchmarkFlags;

#[test]
fn eval_classifies_eos_and_truncated() -> anyhow::Result<()> {
    // The limit marker appears only in the second sample's prompt; the fake
    // server reports `stop_type: "limit"` for it (hit the n_predict cap) and
    // `eos` for everything else.
    const LIMIT_MARKER: &str = "PLEASE-HIT-THE-LIMIT";
    const MAX_TOKENS: u32 = 8;

    // SAFETY: integration tests run in their own process per file; this file
    // holds a single test so no parallel test reads this var.
    std::env::set_var("FAKE_LLAMA_LIMIT_PROMPT_CONTAINS", LIMIT_MARKER);

    let root = tempfile::tempdir()?;
    let evals = EvalCompletionsStore::new(root.path().join("evals"));
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: "stop-reason-test".into(),
        parameter_eval_id: "stop-reason-test".into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: MAX_TOKENS,
        parameter_mcq_choices: None,
        samples: Some(vec![
            json!({
                "id": "s-eos",
                "messages": [{"role": "user", "content": "a normal sample"}],
            }),
            json!({
                "id": "s-truncated",
                "messages": [{"role": "user", "content": format!("long one {LIMIT_MARKER}")}],
            }),
        ]),
    });

    let install = fake_runtime_install_dir(root.path())?;
    let runtime = fake_runtime(&install)?;
    let model = fake_model(&model_path)?;
    let benchmark_flags = BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
        http_timeout_seconds: Some(15),
        doomloop: DoomloopOverrides::default(),
    };
    let req = fake_run_request(runtime, model, benchmark, Some(benchmark_flags));

    let outcome = run(&req, &evals, &no_readiness_gate, &ignore_reps())
        .context("run should classify both samples")?;

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    assert_eq!(completions.len(), 2);
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    // Natural stop below the cap → eos, fewer tokens than the cap, no detail.
    let eos = by_id["s-eos"];
    assert!(!eos.failed);
    assert_eq!(eos.stop_reason, BenchmarkEvalCompletionStopReason::Eos);
    assert_eq!(eos.completion_tokens, Some(3));
    assert_eq!(eos.stop_detail, None);

    // Hit the n_predict cap → truncated, token count == cap.
    let truncated = by_id["s-truncated"];
    assert!(!truncated.failed);
    assert_eq!(
        truncated.stop_reason,
        BenchmarkEvalCompletionStopReason::Truncated
    );
    assert_eq!(truncated.completion_tokens, Some(MAX_TOKENS as u64));
    assert_eq!(truncated.stop_detail, None);

    Ok(())
}
