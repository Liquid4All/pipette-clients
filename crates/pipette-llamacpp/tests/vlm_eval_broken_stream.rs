//! A vlm_eval stream that breaks mid-sample fails that sample alone, and says why.
//! llama.cpp reports a mid-stream failure (ctx overflow, image decode) as an
//! SSE `error` event after the 200; a reader that only collected deltas would
//! score it as an empty successful completion. A stream that closes with no
//! `finish_reason` and no `[DONE]` must not count as a whole completion either.

mod common;

use std::sync::Once;

use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate,
};
use serde_json::json;

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkEvalCompletionStopReason, BenchmarkFlags, BenchmarkId,
    BenchmarkResultData, EnvForward, EvalSample, PositiveCount, PositiveSeconds, RuntimeFlags,
    VlmEvalBenchmark,
};

/// Must match the in-band error event in `tests/bin/fake_llama_server.rs`.
const SERVER_ERROR_MESSAGE: &str = "image exceeds context window";

/// A valid 1×1 PNG, so calibration reads a real image header.
const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// Waives `ensure_complete`: each case leaves a failed sample on purpose, and
/// refusing an incomplete cell has its own tests.
static WAIVE_INCOMPLETE: Once = Once::new();

#[rstest::rstest]
#[case::in_band_error(
    "FAKE_LLAMA_SSE_ERROR_PROMPT_CONTAINS",
    "PLEASE-FAIL-MID-STREAM",
    SERVER_ERROR_MESSAGE
)]
#[case::truncated(
    "FAKE_LLAMA_SSE_TRUNCATE_PROMPT_CONTAINS",
    "PLEASE-DROP-MID-STREAM",
    "ended before its terminal event"
)]
fn a_broken_stream_fails_only_its_sample(
    #[case] trigger: &str,
    #[case] marker: &str,
    #[case] reason_names: &str,
) -> anyhow::Result<()> {
    // SAFETY: the one write to the environment in this binary, made before any
    // case reads it or starts a process.
    WAIVE_INCOMPLETE
        .call_once(|| unsafe { std::env::set_var("PIPETTE_ALLOW_INCOMPLETE_EVAL", "1") });

    let root = tempfile::tempdir()?;
    let evals = no_corpus(root.path());
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;

    let sample = |id: &str, text: String| -> anyhow::Result<EvalSample> {
        Ok(serde_json::from_value(json!({
            "id": id,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": PNG}}
            ]}]
        }))?)
    };
    let benchmark = BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
        benchmark_id: BenchmarkId::try_new("vlm_eval_broken_stream_test".to_owned())?,
        parameter_lmms_task: "broken-stream-test".into(),
        parameter_dataset_version: "local".into(),
        parameter_sample_count: PositiveCount::try_new(2)?,
        parameter_max_tokens: PositiveCount::try_new(16)?,
        parameter_mcq_choices: None,
        parameter_temperature: None,
        parameter_top_p: None,
        parameter_top_k: None,
        parameter_min_p: None,
        parameter_presence_penalty: None,
        parameter_repeat_penalty: None,
        parameter_seed: None,
        parameter_seed_rule: None,
        parameter_prompt_profile: None,
        parameter_image_tokens: None,
        dataset: None,
        recipe: None,
        samples: Some(vec![
            sample("s-ok", "first sample".into())?,
            sample("s-broken", format!("second sample {marker}"))?,
        ]),
    });

    let runtime = fake_runtime(&fake_runtime_install_dir(root.path())?)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let benchmark_flags = BenchmarkFlags::VlmEvalLlamacppCliStockToolsGgufVision {
        http_timeout_seconds: Some(PositiveSeconds::try_new(15)?),
        readiness: None,
    };
    let mut req = fake_run_request(runtime, model, benchmark, Some(benchmark_flags));
    let mut flags = req.runtime_flags_ref()?;
    flags.envs = vec![EnvForward::try_new(format!("{trigger}={marker}"))?];
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);

    let outcome = pipette_llamacpp::run(&req, &evals, &no_readiness_gate, ignore_reps(), None)?;
    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    assert_eq!(completions.len(), 2, "both samples must be present");
    let by_id: std::collections::HashMap<_, _> =
        completions.iter().map(|c| (c.id.as_str(), c)).collect();

    let ok = by_id["s-ok"];
    assert!(!ok.failed, "the healthy sample should succeed");
    assert_eq!(ok.completion, "answer");
    assert_eq!(ok.stop_reason, BenchmarkEvalCompletionStopReason::Eos);

    let failed = by_id["s-broken"];
    assert!(
        failed.failed,
        "a broken stream must fail its sample, not record the partial text as a success"
    );
    assert_eq!(
        failed.completion, "",
        "a failed sample must not keep the partial content as a completion"
    );
    assert_eq!(
        failed.stop_reason,
        BenchmarkEvalCompletionStopReason::Failure
    );
    let reason = failed.failed_reason.as_deref().unwrap_or("");
    assert!(
        reason.contains(reason_names),
        "failed_reason must say {reason_names:?}; got {reason:?}",
    );
    assert!(
        reason.contains("server still alive"),
        "a broken stream leaves the server up (transport branch); got {reason:?}",
    );
    Ok(())
}
