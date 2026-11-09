use pipette_http::HttpClient;
use pipette_memprobe::Watch;
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_plan_types::{
    MaxMemoryUsageBenchmark, MlxMacosPipetteExtraRuntimeConfiguration, RunResponse,
};

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{completions, memory, server, tokenizer};

const DECODE_TOKENS: u32 = 1;

pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the tokenizer, the first process this cell starts.
    let watch = Watch::Peak.supported()?;
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();

    // EOS suppression forces the full decode token out so the peak covers real
    // generation, not an early stop.
    let (prompt, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        prefill_tokens,
        server::TEXT_TRUSTS_REMOTE_CODE,
    )?;
    let request = completions::completion_request(&prompt, DECODE_TOKENS, &eos_ids);
    let http = HttpClient::loopback("pipette", completions::TIMING_TIMEOUT)?;

    // The server's usage block confirms the shape.
    let generate = |url: &str| {
        let counts = pipette_completions::reply(&http, &completions::url(url), &request)?
            .counts_at_limit()?;
        measurement::expect_tokens("prompt_tokens", counts.prompt_tokens, prefill_tokens)?;
        measurement::expect_tokens("completion_tokens", counts.completion_tokens, DECODE_TOKENS)
    };
    let text_command = |port| server::text_command(venv_python, model_dir, port, client_config);

    let mut served = None;
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let max_host_bytes = memory::serve(
                served,
                server::TEXT_SERVER_MODULE,
                text_command,
                watch,
                generate,
            )?;
            Ok(MaxMemoryMeasurementReading {
                max_host_bytes,
                // Unified memory (Apple Silicon): the GPU allocation lives inside
                // the host footprint, so there is no separate pool to report.
                max_gpu_bytes: None,
                // No platform implements per-process NPU memory yet.
                max_npu_bytes: None,
            })
        },
    );
    memory::respond(result, &mut served)
}
