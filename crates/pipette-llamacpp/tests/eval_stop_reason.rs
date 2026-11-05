//! Per-sample `stop_reason`: a natural stop is `eos`, a sample that hits the
//! `n_predict` cap is `truncated`, each with the server's token count.
//! `failure` is covered by `eval_crash_recovery.rs`.

mod common;

use anyhow::Context;
use common::{
    fake_model, fake_run_request, fake_runtime, fake_runtime_install_dir, ignore_reps, no_corpus,
    no_readiness_gate,
};
use serde_json::json;

use pipette_llamacpp::run;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkEvalCompletionStopReason, BenchmarkFlags, BenchmarkId,
    BenchmarkResultData, EvalBenchmark, PositiveCount, PositiveSeconds,
};

#[test]
fn eval_classifies_eos_and_truncated() -> anyhow::Result<()> {
    const LIMIT_MARKER: &str = "PLEASE-HIT-THE-LIMIT";
    const MAX_TOKENS: u32 = 8;

    // SAFETY: this file holds a single test, so no parallel test reads this var.
    unsafe { std::env::set_var("FAKE_LLAMA_LIMIT_PROMPT_CONTAINS", LIMIT_MARKER) };

    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;

    let benchmark = BenchmarkDefinition::Eval(EvalBenchmark {
        benchmark_id: BenchmarkId::try_new("stop-reason-test".to_owned())?,
        parameter_eval_id: "stop-reason-test".into(),
        parameter_dataset_name: "local".into(),
        parameter_max_tokens: PositiveCount::try_from(MAX_TOKENS)?,
        parameter_max_context: None,
        parameter_mcq_choices: None,
        parameter_temperature: None,
        parameter_top_p: None,
        parameter_top_k: None,
        parameter_min_p: None,
        parameter_seed: None,
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
        http_timeout_seconds: Some(PositiveSeconds::try_new(15)?),
    };
    let req = fake_run_request(runtime, model, benchmark, Some(benchmark_flags));

    let outcome = run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("run should classify both samples")?;
    // A cell that authored no flags reports none.
    assert_eq!(outcome.runtime_flags, None);

    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    assert_eq!(completions.len(), 2);
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    let eos = by_id["s-eos"];
    assert!(!eos.failed);
    assert_eq!(eos.stop_reason, BenchmarkEvalCompletionStopReason::Eos);
    assert_eq!(eos.completion_tokens, Some(3));
    assert_eq!(eos.stop_detail, None);

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
