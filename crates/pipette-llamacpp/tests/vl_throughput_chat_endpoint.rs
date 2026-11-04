//! `vl_throughput` sends the image to `/v1/chat/completions` (the
//! vision-tower endpoint), not `/completion`, and reports `image_tokens`, with
//! `total_time_ms` on this client's clock rather than the server's timers.

mod common;

use anyhow::Context;
use common::{
    assert_distinct_knobs, assert_server_log, fake_run_request, fake_runtime,
    fake_runtime_install_dir, fake_vision_model, ignore_reps, no_corpus, no_readiness_gate,
    with_distinct_knobs,
};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkId, BenchmarkResultData, NonNegativeCount, PositiveCount,
    VlThroughputBenchmark,
};

#[test]
fn vl_throughput_routes_image_through_chat_endpoint() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let sent = root.path().join("chat-images.txt");
    // SAFETY: one test per file, so no parallel test reads this var.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_IMAGES_FILE", &sent) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_DELAY_MS", "20") };
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;

    let benchmark = BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-chat-endpoint-test".to_owned())?,
        parameter_image_width: PositiveCount::try_new(256)?,
        parameter_image_height: PositiveCount::try_new(256)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    });

    let install = fake_runtime_install_dir(root.path())?;
    #[cfg(unix)]
    common::install_fake_mtmd_cli(&install, 10)?;
    let runtime = fake_runtime(&install)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let req = with_distinct_knobs(fake_run_request(runtime, model, benchmark, None))?;

    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("vl_throughput should complete against the fake chat endpoint")?;
    assert_distinct_knobs(&req.benchmark, &outcome.command)?;

    // The fake `/completion` returns no `timings` or `finish_reason`, so a
    // result at all proves the chat endpoint was hit.
    let BenchmarkResultData::VlThroughput {
        prompt_tokens,
        image_tokens,
        prompt_ms,
        predicted_ms,
        total_time_ms,
        total_time_ms_stddev,
        ..
    } = outcome.result_data
    else {
        anyhow::bail!("expected VlThroughput result data");
    };
    // The fake's prompt_n is 5 + 10 per image.
    assert_eq!(prompt_tokens, 15);
    assert_eq!(image_tokens, cfg!(unix).then_some(10));
    // The server's 2 ms start at the first of the 4 tokens: 3 steps, scaled to 4.
    assert!((predicted_ms - 8.0 / 3.0).abs() < 1e-5, "{predicted_ms}");
    let total_time_ms = total_time_ms.context("a llama.cpp VL row times the whole request")?;
    // The fake's own timers say 1 ms and 2 ms; the client's clock saw the hold.
    assert!(
        total_time_ms >= 20.0,
        "{total_time_ms} vs {prompt_ms} + {predicted_ms}"
    );
    assert!(total_time_ms_stddev.is_some());
    // The warm-up and five reps, all with the image.
    let sent = std::fs::read_to_string(&sent)?;
    assert_eq!(sent.lines().collect::<Vec<_>>(), ["1"; 6]);
    assert_server_log(&outcome.runtime_output)
}
