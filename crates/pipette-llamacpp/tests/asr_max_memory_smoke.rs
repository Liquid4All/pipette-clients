//! `asr_max_memory` end to end against the fake `llama-server`: spawn,
//! readiness, the one-clip workload, the stop, and a real peak off the fake's
//! pid. Runs on macOS (`phys_footprint`) and Linux (`/proc`), the probe the
//! host-built fake runtime picks.
#![cfg(any(target_os = "macos", target_os = "linux"))]

mod common;

use anyhow::Context;
use common::{
    assert_distinct_knobs, assert_server_log, audio_projector, fake_run_request, fake_runtime,
    fake_runtime_install_dir, fake_vision_model, ignore_reps, no_corpus, no_readiness_gate,
    with_distinct_knobs,
};

use pipette_plan_types::{
    AsrDecodeTokens, AsrMaxMemoryBenchmark, AudioChannels, AudioDurationMillis, AudioSampleRate,
    BenchmarkDefinition, BenchmarkId, BenchmarkResultData, NonNegativeCount, RuntimeFlags,
};

#[test]
fn asr_max_memory_measures_a_real_peak_and_reports_derived_flags() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    let mmproj = audio_projector(Some("lfm2a"));
    std::fs::write(&mmproj_path, mmproj)?;

    let benchmark = BenchmarkDefinition::AsrMaxMemory(AsrMaxMemoryBenchmark {
        benchmark_id: BenchmarkId::try_new("asr-max-memory-smoke".to_owned())?,
        parameter_audio_duration_ms: AudioDurationMillis::try_new(2_500)?,
        parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
        parameter_audio_channels: AudioChannels::try_new(1)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: AsrDecodeTokens::try_new(8)?,
    });

    let runtime = fake_runtime(&fake_runtime_install_dir(root.path())?)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let req = with_distinct_knobs(fake_run_request(runtime, model, benchmark, None))?;

    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("asr_max_memory should complete against the fake server")?;
    assert_distinct_knobs(&req.benchmark, &outcome.command)?;

    let BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes,
        max_gpu_bytes,
        max_npu_bytes,
    } = outcome.result_data
    else {
        anyhow::bail!("expected MaxMemoryUsage result data");
    };
    assert!(
        max_host_bytes > 0,
        "the peak should be non-zero for a live process"
    );
    // No GPU pool on either OS: unified memory on macOS, CPU flavors on Linux.
    assert_eq!(max_gpu_bytes, None);
    assert_eq!(max_npu_bytes, None);
    // The observation covers the warm-up server too; on Linux the metric adds
    // the swap the observation carries beside it.
    let observed = outcome
        .memory
        .max_host_bytes
        .unwrap_or(0)
        .saturating_add(outcome.memory.max_swap_bytes.unwrap_or(0));
    assert!(observed >= max_host_bytes, "{:?}", outcome.memory);
    // Only `/proc` samples swap, and it reports `Some(0)` for a resident run.
    assert_eq!(
        outcome.memory.max_swap_bytes.is_some(),
        cfg!(target_os = "linux")
    );
    // The Linux row carries the `/proc` terms behind its swap-aware peak.
    let mut extras = outcome
        .client_extras
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    extras.sort_unstable();
    let expected: &[&str] = if cfg!(target_os = "linux") {
        &[
            "asr_beyond_model_audio_limit",
            "asr_context_sizing",
            "major_faults",
            "max_swap_kib",
            "peak_committed_kib",
            "peak_rss_kib",
        ]
    } else {
        &["asr_beyond_model_audio_limit", "asr_context_sizing"]
    };
    assert_eq!(extras, expected);
    // The smoke model's projector documents no audio limit.
    assert_eq!(
        outcome.client_extras.get("asr_beyond_model_audio_limit"),
        Some(&serde_json::Value::from(false))
    );
    // The fake runtime has no counting helper, so the context came from the
    // upper bound, and the row says so.
    assert_eq!(
        outcome.client_extras.get("asr_context_sizing"),
        Some(&serde_json::Value::from("upper_bound"))
    );
    assert_server_log(&outcome.runtime_output)?;

    let flags = outcome
        .runtime_flags
        .context("asr_max_memory should report its resolved runtime_flags")?;
    let RuntimeFlags::AsrMaxMemoryLlamacppCliStockToolsGgufVision(flags) = flags else {
        anyhow::bail!("expected the asr_max_memory runtime flags");
    };
    // The derived ctx_size is how the warehouse learns the server's `-c`. Here
    // it is the upper bound: at least the clip's audio, and far below the
    // pinned-default contexts.
    let ctx = flags
        .ctx_size
        .context("the derived ctx_size should be reported")?;
    assert!(
        (750..=8192).contains(&ctx),
        "a 2.5 s clip should size a small context, got {ctx:?}"
    );
    Ok(())
}
