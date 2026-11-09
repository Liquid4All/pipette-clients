//! Peak memory for one OVMS cell: the server's peak resident memory across its
//! model load and one request at the cell's prompt length.
//!
//! Off the CPU, the device's memory comes from the OS, as `max_gpu_bytes` or
//! `max_npu_bytes`, and a run on the device must show some: on Windows what
//! WDDM bills the process on that adapter, on Linux what the GPU's DRM driver
//! bills it. A device whose memory this OS's sampler cannot read (an NPU or a
//! custom device on Linux) is refused before anything starts rather than
//! recorded as the host's alone.

use std::path::Path;
use std::process::Command;

use anyhow::Context;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch, Watch};
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BatchSize, MaxMemoryUsageBenchmark, MaxMemoryUsageOvmsOpenvino, MemoryObservation,
    OpenvinoDevice, OvmsExtraRuntimeConfiguration, PositiveSeconds, RunResponse, RuntimeFlags,
};
use pipette_subprocess::{Process, start_process};

use crate::completions;
use crate::models::ResolvedOpenvinoModel;
use crate::runtimes::ResolvedOvmsInstall;
use crate::server;

/// The first token comes off the prefill; the second takes one decode pass, so
/// its buffers count toward the peak.
const DECODE_TOKENS: u32 = 2;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    runtime_flags: MaxMemoryUsageOvmsOpenvino,
    install: &ResolvedOvmsInstall,
    model: &ResolvedOpenvinoModel,
    compile_cache: &Path,
    http_timeout: Option<PositiveSeconds>,
    observer: MeasurementObserver<'_>,
    client_config: Option<&OvmsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let watch = watch(
        runtime_flags
            .device
            .as_ref()
            .context("resolved flags name no device")?,
    )?;
    let flags = RuntimeFlags::MaxMemoryUsageOvmsOpenvino(runtime_flags);
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
    let http = HttpClient::loopback("pipette", measurement::http_timeout_for(http_timeout, 1))?;
    let command = |port| {
        server::command(
            install,
            model,
            &flags,
            BatchSize::one(),
            compile_cache,
            port,
            client_config,
        )
    };
    // On a discarded server, so tokenize neither breaks the measured server's
    // graphs nor lands in its peak.
    let sizing_port = pipette_subprocess::pick_server_port()?;
    let prompt = server::size_prompt(
        command(sizing_port)?,
        &server::url(sizing_port),
        &http,
        prefill_tokens,
        observer,
    )?;

    let mut served = None;
    // Sizing sends no completion, so a GPU's prompt-shape kernels compile in
    // the warm-up.
    let result = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            Ok(serve(
                served,
                command,
                watch,
                &http,
                &prompt,
                prefill_tokens,
            )?)
        },
    );
    let result_data = result.map_err(|err| match &mut served {
        Some((ovms, _)) => err.context(ovms.stop_with_tail()),
        None => err,
    })?;
    Ok(served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (ovms, memory)| response.process(ovms).memory(*memory),
        )
        .runtime_flags(flags)
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
    prompt: &str,
    prefill_tokens: u32,
) -> anyhow::Result<MaxMemoryMeasurementReading> {
    let earlier = match served.take() {
        Some((mut ovms, memory)) => {
            ovms.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut ovms = start_process(server::PROGRAM, command(port)?, server::STOP_GRACE)?;
    // From the launch, as on the timing cells, so the Linux swap term covers the
    // load too.
    let sampler = Sampler::start(ovms.pid(), watch)?;
    ovms.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let measured = measure(http, &url, prompt, prefill_tokens);
    // While the server lives: Linux reads its `/proc` entry.
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((ovms, earlier.merge_max(observed)));
    measured?;
    let peak = peak?;
    Ok(MaxMemoryMeasurementReading {
        max_host_bytes: peak.host_bytes,
        max_gpu_bytes: peak.gpu_bytes,
        max_npu_bytes: peak.npu_bytes,
    })
}

/// One request at the cell's shape, with `prompt` sized elsewhere.
fn measure(http: &HttpClient, url: &str, prompt: &str, prefill_tokens: u32) -> anyhow::Result<()> {
    let request = TimingRequest::text(Some(server::MODEL_NAME), prompt, DECODE_TOKENS);
    let counts =
        pipette_completions::reply(http, &completions::url(url), &request)?.counts_at_limit()?;
    measurement::expect_tokens("prompt_tokens", counts.prompt_tokens, prefill_tokens)?;
    measurement::expect_tokens("completion_tokens", counts.completion_tokens, DECODE_TOKENS)?;
    Ok(())
}

/// `device`'s watch, refused where this OS reads none of its memory, naming
/// the devices it does read.
fn watch(device: &OpenvinoDevice) -> anyhow::Result<SupportedWatch> {
    device_watch(device).supported().with_context(|| {
        let read: Vec<String> = OpenvinoDevice::known()
            .filter(|known| device_watch(known).supported().is_ok())
            .map(|known| known.as_str().to_owned())
            .collect();
        format!(
            "the {} device's memory cannot be read on this OS; run this memory cell on {}",
            crate::flags::device_property(device),
            read.join(" or ")
        )
    })
}

/// A custom device (`AUTO`, `HETERO:GPU,CPU`, `GPU.1`) may resolve anywhere,
/// so it requires none. `pipette-openvino` keeps the same rule.
fn device_watch(device: &OpenvinoDevice) -> Watch {
    match device {
        OpenvinoDevice::Cpu => Watch::Peak,
        OpenvinoDevice::Gpu => Watch::PeakWithGpu,
        OpenvinoDevice::Npu => Watch::PeakWithNpu,
        OpenvinoDevice::Custom(_) => Watch::PeakWithAccelerators,
    }
}

#[cfg(test)]
mod tests {
    use pipette_memprobe::{UnsupportedError, WatchError};

    use super::*;

    #[rstest::rstest]
    #[case::cpu(OpenvinoDevice::Cpu, Watch::Peak)]
    #[case::gpu(OpenvinoDevice::Gpu, Watch::PeakWithGpu)]
    #[case::npu(OpenvinoDevice::Npu, Watch::PeakWithNpu)]
    #[case::custom(OpenvinoDevice::Custom("GPU.1".to_owned()), Watch::PeakWithAccelerators)]
    fn each_device_has_its_watch(#[case] device: OpenvinoDevice, #[case] expected: Watch) {
        assert_eq!(device_watch(&device), expected);
    }

    /// Windows reads every device, so there it has nothing to refuse.
    #[test]
    fn a_refusal_names_the_device_and_the_devices_this_os_reads() {
        let Err(err) = watch(&OpenvinoDevice::Npu) else {
            return;
        };
        let refused = UnsupportedError {
            watch: Watch::PeakWithNpu,
        };
        assert_eq!(
            err.downcast_ref::<WatchError>(),
            Some(&refused.into()),
            "{err:#}"
        );
        let read = if cfg!(target_os = "linux") {
            "cpu or gpu"
        } else {
            "cpu"
        };
        let expected = format!(
            "the {} device's memory cannot be read on this OS; run this memory cell on {read}",
            crate::flags::device_property(&OpenvinoDevice::Npu)
        );
        assert_eq!(err.to_string(), expected);
    }

    /// A tokenize call would break the graph the completion then runs on
    /// (`docs/ovms.md`), so the measured server only gets the completion.
    #[test]
    fn the_measured_server_gets_no_tokenize_calls() -> anyhow::Result<()> {
        let server = httpmock::MockServer::start();
        let tokenize = server.mock(|when, then| {
            when.method("POST").path("/v3/tokenize");
            then.status(200)
                .json_body(serde_json::json!({ "tokens": [1, 2, 3, 4] }));
        });
        let completion = server.mock(|when, then| {
            when.method("POST")
                .path("/v3/completions")
                .json_body_includes(r#"{"prompt":"sized elsewhere","max_tokens":2}"#);
            then.status(200).json_body(serde_json::json!({
                "choices": [{"index": 0, "text": "ab", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 512, "completion_tokens": 2}
            }));
        });
        measure(
            &HttpClient::loopback("pipette", std::time::Duration::from_secs(5))?,
            &server.base_url(),
            "sized elsewhere",
            512,
        )?;
        assert_eq!((tokenize.calls(), completion.calls()), (0, 1));
        Ok(())
    }
}
