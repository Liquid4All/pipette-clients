//! A non-2xx from `/v1/chat/completions` fails a VL cell with the response
//! body (a ctx-overflow or image-decode 400 carries the real message) and the
//! end of the server's log.

mod common;

use common::{
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate,
};

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkId, EnvForward, NonNegativeCount, PositiveCount, RunRequest,
    RuntimeFlags, VlMaxMemoryBenchmark, VlThroughputBenchmark,
};

const ERR_BODY: &str = "image preprocessing failed: unsupported bitmap";

fn vl_throughput() -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-error-body-test".to_owned())?,
        parameter_image_width: PositiveCount::try_new(256)?,
        parameter_image_height: PositiveCount::try_new(256)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: PositiveCount::try_new(4)?,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    }))
}

fn vl_max_memory() -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::VlMaxMemory(VlMaxMemoryBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-max-memory-failure".to_owned())?,
        parameter_image_width: PositiveCount::try_new(224)?,
        parameter_image_height: PositiveCount::try_new(224)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: None,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    }))
}

/// `req` with the fake told to refuse every chat request with [`ERR_BODY`],
/// through the cell's own `envs` rather than a variable every test shares.
fn refusing_chat(mut req: RunRequest) -> anyhow::Result<RunRequest> {
    let mut flags = req.runtime_flags_ref()?;
    flags.envs = vec![EnvForward::try_new(format!(
        "FAKE_LLAMA_CHAT_400={ERR_BODY}"
    ))?];
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    Ok(req)
}

#[rstest::rstest]
#[case::vl_throughput(vl_throughput())]
#[cfg_attr(
    not(any(target_os = "macos", target_os = "linux")),
    ignore = "gated to the hosts vl_max_memory_smoke runs on"
)]
#[case::vl_max_memory(vl_max_memory())]
fn a_refused_chat_request_carries_its_body_and_the_server_log(
    #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;
    let req = refusing_chat(fake_run_request(
        fake_runtime(&fake_runtime_install_dir(root.path())?)?,
        fake_vision_model(&model_path, &mmproj_path)?,
        benchmark?,
        None,
    ))?;

    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("a 400 from the chat endpoint must fail the run");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains(ERR_BODY),
        "error should carry the server body, got: {message}"
    );
    assert!(
        message.contains("llama-server was still running")
            && message.contains("fake-llama-server: listening"),
        "error should quote the server's log, got: {message}"
    );
    Ok(())
}
