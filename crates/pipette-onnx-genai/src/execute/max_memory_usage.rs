//! Peak memory for one ONNX GenAI cell: the server process's peak across the
//! model load and one request at the cell's prompt size with one output token.
//!
//! A warm-up server runs the same request first and its reading is dropped:
//! the first import after an install writes the venv's Python bytecode, which
//! raised the macOS peak by 6 to 8 MB.
//!
//! A GPU provider's run must show device memory (`max_gpu_bytes`). macOS is
//! unified memory, so its host figure is the whole cost of a custom provider.
//! A provider whose device memory this OS cannot read (CUDA or DirectML on
//! macOS, a custom one on Linux) is refused before anything starts rather
//! than recorded as the host's alone.

use std::process::Command;
use std::time::Duration;

use anyhow::Context;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch, Watch};
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    MaxMemoryUsageBenchmark, MemoryObservation, OnnxExecutionProvider, RunResponse,
    UvOnnxGenaiExtraRuntimeConfiguration,
};
use pipette_subprocess::{Process, start_process};

use crate::completions;
use crate::models::ResolvedOnnxGenaiModel;
use crate::response;
use crate::runtimes::ResolvedOnnxGenaiInstall;
use crate::server;

const DECODE_TOKENS: u32 = 1;

pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    install: &ResolvedOnnxGenaiInstall,
    model: &ResolvedOnnxGenaiModel,
    timeout: Duration,
    observer: MeasurementObserver<'_>,
    client_config: Option<&UvOnnxGenaiExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let watch = watch(&install.execution_provider)?;
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
    let script = server::script()?;
    let command = |port| server::command(install, model, script.path(), port, client_config);
    let http = HttpClient::loopback("pipette", timeout)?;
    let mut served = None;
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| Ok(serve(served, command, watch, &http, prefill_tokens)?),
    );
    let result_data = result.map_err(|err| match &mut served {
        Some((onnx_server, _)) => err.context(onnx_server.stop_with_tail()),
        None => err,
    })?;
    Ok(served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (onnx_server, memory)| response.process(onnx_server).memory(*memory),
        )
        .command(response::command(&install.python))
        .build())
}

/// One server from its launch through one request at the cell's shape, left
/// in `served` for the entry to check its reading and the cell to record or
/// quote it. One an earlier attempt left there stops first; the observation
/// covers both.
fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    command: impl Fn(u16) -> anyhow::Result<Command>,
    watch: SupportedWatch,
    http: &HttpClient,
    prefill_tokens: u32,
) -> anyhow::Result<MaxMemoryMeasurementReading> {
    let earlier = match served.take() {
        Some((mut onnx_server, memory)) => {
            onnx_server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut onnx_server =
        start_process(server::SCRIPT_FILENAME, command(port)?, server::STOP_GRACE)?;
    // From the launch, so the load counts.
    let sampler = Sampler::start(onnx_server.pid(), watch)?;
    onnx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let measured = measure(http, &url, prefill_tokens);
    // While the server lives: Linux reads its `/proc` entry.
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((onnx_server, earlier.merge_max(observed)));
    measured?;
    let peak = peak?;
    Ok(MaxMemoryMeasurementReading {
        max_host_bytes: peak.host_bytes,
        max_gpu_bytes: peak.gpu_bytes,
        max_npu_bytes: peak.npu_bytes,
    })
}

fn measure(http: &HttpClient, url: &str, prefill_tokens: u32) -> anyhow::Result<()> {
    let prompt =
        pipette_completions::build_prompt_text(http, url, completions::MODEL, prefill_tokens)?;
    let counts = pipette_completions::reply(
        http,
        &completions::url(url),
        &TimingRequest::text(Some(completions::MODEL), &prompt, DECODE_TOKENS),
    )?
    .counts_at_limit()?;
    measurement::expect_tokens("prompt_tokens", counts.prompt_tokens, prefill_tokens)?;
    measurement::expect_tokens("completion_tokens", counts.completion_tokens, DECODE_TOKENS)?;
    Ok(())
}

/// `provider`'s watch, refused where this OS reads none of its device's memory.
fn watch(provider: &OnnxExecutionProvider) -> anyhow::Result<SupportedWatch> {
    provider_watch(provider).supported().with_context(|| {
        format!(
            "the {} provider's device memory cannot be read on this OS; run this memory cell \
             on the CPU provider",
            provider.as_str()
        )
    })
}

/// A custom provider picks its own device, so it requires none.
/// `ResolvedOnnxGenaiInstall::try_from` refuses Windows ML.
fn provider_watch(provider: &OnnxExecutionProvider) -> Watch {
    match provider {
        OnnxExecutionProvider::Cpu => Watch::Peak,
        OnnxExecutionProvider::Cuda | OnnxExecutionProvider::DirectMl => Watch::PeakWithGpu,
        OnnxExecutionProvider::WinMl | OnnxExecutionProvider::Custom(_) => {
            Watch::PeakWithAccelerators
        }
    }
}

#[cfg(test)]
mod tests {
    use pipette_memprobe::{UnsupportedError, WatchError};

    use super::*;

    #[rstest::rstest]
    #[case::cpu(OnnxExecutionProvider::Cpu, Watch::Peak)]
    #[case::cuda(OnnxExecutionProvider::Cuda, Watch::PeakWithGpu)]
    #[case::directml(OnnxExecutionProvider::DirectMl, Watch::PeakWithGpu)]
    #[case::custom(
        OnnxExecutionProvider::Custom("qnn".to_owned()),
        Watch::PeakWithAccelerators
    )]
    fn each_provider_has_its_watch(
        #[case] provider: OnnxExecutionProvider,
        #[case] expected: Watch,
    ) {
        assert_eq!(provider_watch(&provider), expected);
    }

    /// Where this OS reads the provider's device memory there is no refusal
    /// to check.
    #[rstest::rstest]
    #[case::cuda(OnnxExecutionProvider::Cuda)]
    #[case::directml(OnnxExecutionProvider::DirectMl)]
    #[case::custom(OnnxExecutionProvider::Custom("qnn".to_owned()))]
    fn a_refusal_names_the_provider(#[case] provider: OnnxExecutionProvider) {
        let Err(err) = watch(&provider) else {
            return;
        };
        let refused = UnsupportedError {
            watch: provider_watch(&provider),
        };
        assert_eq!(
            err.downcast_ref::<WatchError>(),
            Some(&refused.into()),
            "{err:#}"
        );
        let named = format!("the {} provider", provider.as_str());
        assert!(err.to_string().starts_with(&named), "{err:#}");
    }
}
