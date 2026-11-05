//! A non-2xx from `/v1/chat/completions` fails the sample with the server's
//! response body in `failed_reason` (a ctx-overflow or image-decode 400 says
//! why; a bare status would not), and unlike vl_throughput the run survives.
//! Its own file so `FAKE_LLAMA_CHAT_400` cannot race a sibling.

mod common;

use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate,
};
use serde_json::json;

use pipette_plan_types::BenchmarkResultData;
use pipette_plan_types::{BenchmarkDefinition, ModelFlags, VlmEvalBenchmark};
use pipette_plan_types::{BenchmarkId, PositiveCount};

#[test]
fn vlm_eval_failed_reason_carries_server_error_body() -> anyhow::Result<()> {
    const ERR_BODY: &str = "image preprocessing failed: unsupported bitmap";

    // Waive `ensure_complete`: the failed sample is the point, and refusing
    // an incomplete cell has its own tests.
    // SAFETY: every test in this binary sets the same value.
    unsafe { std::env::set_var("PIPETTE_ALLOW_INCOMPLETE_EVAL", "1") };
    // SAFETY: one test per file, so no parallel test reads this var.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_400", ERR_BODY) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_THINKING_SWITCH", "1") };

    let root = tempfile::tempdir()?;
    let evals = no_corpus(root.path());
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;

    let benchmark = BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
        benchmark_id: BenchmarkId::try_new("vlm-eval-error-body-test".to_owned())?,
        parameter_lmms_task: "error-body-test".into(),
        parameter_dataset_version: "local".into(),
        parameter_sample_count: PositiveCount::try_new(1)?,
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
        samples: Some(vec![serde_json::from_value(json!({
            "id": "s-400",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is in the image?"},
                {"type": "image_url", "image_url": {"url":
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" }}
            ]}]
        }))?]),
    });

    let runtime = fake_runtime(&fake_runtime_install_dir(root.path())?)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let mut req = fake_run_request(runtime, model, benchmark, None);
    req.model_flags = Some(ModelFlags::VlmEvalGgufVision {
        enable_thinking: Some(false),
    });

    let outcome = pipette_llamacpp::run(&req, &evals, &no_readiness_gate, ignore_reps(), None)?;
    // With no runtime_flags at all, the model flag alone must reach argv on a
    // template with a thinking switch. Checked on the recorded command since
    // the fake server ignores flags it does not parse.
    for pair in [["--reasoning", "off"], ["--reasoning-budget", "0"]] {
        assert!(
            outcome.command.windows(2).any(|w| w == pair),
            "expected {pair:?} in the spawned argv, got {:?}",
            outcome.command
        );
    }
    let BenchmarkResultData::Eval { completions } = outcome.result_data else {
        anyhow::bail!("expected Eval result data");
    };
    assert_eq!(completions.len(), 1);
    let sample = &completions[0];
    assert!(sample.failed, "a 400 must record the sample as failed");
    let reason = sample.failed_reason.as_deref().unwrap_or("");
    assert!(
        reason.contains(ERR_BODY),
        "failed_reason must carry the server's response body; got {reason:?}",
    );
    Ok(())
}
