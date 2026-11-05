//! `vl_throughput` with `text_tokens > 0` grows its prompt to an exact token
//! count through `POST /tokenize` (`prompt_seed::build_prompt_text`). The other
//! VL tests use `text_tokens: 0`, which makes no `/tokenize` call.

mod common;

use anyhow::Context;
use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate,
};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkId, BenchmarkResultData, NonNegativeCount, PositiveCount,
    VlThroughputBenchmark,
};

#[test]
fn vl_throughput_builds_text_prompt_via_tokenize() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;

    let benchmark = BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-text-prompt-test".to_owned())?,
        parameter_image_width: PositiveCount::try_new(256)?,
        parameter_image_height: PositiveCount::try_new(256)?,
        parameter_text_tokens: NonNegativeCount::try_new(32)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    });

    let install = fake_runtime_install_dir(root.path())?;
    #[cfg(unix)]
    common::install_fake_mtmd_cli(&install, 10)?;
    let runtime = fake_runtime(&install)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let req = fake_run_request(runtime, model, benchmark, None);
    let sent = root.path().join("chat-bodies.jsonl");
    // SAFETY: this test binary runs this one test, and the server it spawns
    // reads the variable at start.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_BODY_FILE", &sent) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_TOKENIZE_BOS", "1") };

    // A result means the prompt builder converged on the fake's per-word count.
    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("vl_throughput with text_tokens>0 should build the prompt and complete")?;

    let BenchmarkResultData::VlThroughput {
        prompt_tokens,
        image_tokens,
        ..
    } = outcome.result_data
    else {
        anyhow::bail!("expected VlThroughput result data");
    };
    // The fake's prompt_n ignores the text: 5 + 10 per image.
    assert_eq!(prompt_tokens, 15);
    assert_eq!(image_tokens, cfg!(unix).then_some(10));
    // The filler is the message's text: counted without the BOS the fake's
    // tokenizer prepends for special tokens, it holds all 32 tokens, not 31.
    let body: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(&sent)?
            .lines()
            .last()
            .context("a chat request was sent")?,
    )?;
    let text: String = body["messages"][0]["content"]
        .as_array()
        .context("multimodal content")?
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect();
    assert_eq!(text.split_whitespace().count(), 32);
    Ok(())
}
