//! Peak memory of one OpenVINO VL cell: the text memory cell's warm-up and
//! measured servers (see `max_memory_usage`), each serving `pipette-llamacpp`'s
//! request (`docs/methodology/vl-max-memory.md`) on its VL routes.

use std::path::Path;
use std::time::Duration;

use anyhow::Context;
use base64::Engine;

use pipette_completions::llama_server;
use pipette_http::HttpClient;
use pipette_ops::measurement::{
    self, MeasurementObserver, MemoryMeasurementWarmup, VlMaxMemoryMeasurementReading,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    RunResponse, RuntimeFlags, UvOpenvinoExtraRuntimeConfiguration, VlMaxMemoryBenchmark,
    VlUvOpenvinoOpenvino,
};

use crate::memory::{serve, watch};
use crate::models::ResolvedOpenvinoVlModel;
use crate::response;
use crate::runtimes::ResolvedOpenvinoPython;
use crate::server;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &VlMaxMemoryBenchmark,
    runtime_flags: VlUvOpenvinoOpenvino,
    python: &ResolvedOpenvinoPython,
    model: &ResolvedOpenvinoVlModel,
    compile_cache: &Path,
    timeout: Duration,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOpenvinoExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let watch = watch(
        runtime_flags
            .device
            .as_ref()
            .context("resolved flags name no device")?,
    )?;
    let flags = RuntimeFlags::VlMaxMemoryUvOpenvinoOpenvino(runtime_flags);
    let width = benchmark.parameter_image_width.into_inner().unsigned_abs();
    let height = benchmark.parameter_image_height.into_inner().unsigned_abs();
    // A 0 would drop the image the cell measures; llama.cpp reads it as 1 too.
    let num_images = benchmark
        .parameter_num_images
        .into_inner()
        .unsigned_abs()
        .max(1);
    let text_tokens = benchmark.parameter_text_tokens.into_inner().unsigned_abs();
    // A cell without an output length decodes one token, as llama.cpp's.
    let decode_tokens = benchmark
        .parameter_decode_tokens
        .map_or(1, |tokens| tokens.into_inner().unsigned_abs());
    let image =
        base64::engine::general_purpose::STANDARD.encode(pipette_ops::png::gray(width, height)?);
    let script = server::script()?;
    let command = |port| {
        server::command(
            python,
            model,
            &flags,
            compile_cache,
            script.path(),
            port,
            client_config,
        )
    };
    let http = HttpClient::loopback("pipette", timeout)?;
    let mut served = None;
    // The `/proc` terms behind the swap-aware peak, for the row.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let mut extras = serde_json::Map::new();
    let result = measurement::vl_max_memory(
        benchmark,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let peak = serve(served, command, watch, observer, |url| {
                let text = llama_server::message_filler(&http, url, text_tokens)?;
                let generation =
                    llama_server::chat(&http, url, &text, &image, decode_tokens, num_images)?;
                measurement::decoded_in_full(&generation, decode_tokens)
            })?;
            // The peak counts swap: swap can hold part of the model mid-run.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            let peak = {
                extras = peak.footprint.extras();
                peak.counting_swap()
            };
            Ok(VlMaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu_bytes,
                max_npu_bytes: peak.npu_bytes,
            })
        },
    );
    let result_data = result.map_err(|err| match &mut served {
        Some((openvino_server, _)) => err.context(openvino_server.stop_with_tail()),
        None => err,
    })?;
    let response = served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (openvino_server, memory)| response.process(openvino_server).memory(*memory),
        )
        .command(response::command(python))
        .runtime_flags(flags);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let response = response.client_extras(extras);
    Ok(response.build())
}
