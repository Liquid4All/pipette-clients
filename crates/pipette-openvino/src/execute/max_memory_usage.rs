//! Peak memory for one OpenVINO cell: the server process's peak resident set
//! across its load and one request at the cell's prompt size with one output
//! token.
//!
//! A warm-up server runs the same request first: on a cold cache its load
//! compiles the model blob, and on a GPU its request compiles the kernels for
//! the cell's prompt shape, both into the cache, so the measured one loads
//! them. A compile's peak is the compiler's, not the model's
//! (`docs/openvino-measurement.md`). The observation covers the warm-up too,
//! since the cell started it.
//!
//! Off the CPU a run must show device memory (`max_gpu_bytes` or
//! `max_npu_bytes`). A device whose memory this OS cannot read (an NPU or a
//! custom device on Linux) is refused before anything starts rather than
//! recorded as the host's alone.

use std::path::Path;
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
    MaxMemoryUsageBenchmark, MaxMemoryUsageUvOpenvinoOpenvino, MemoryObservation, OpenvinoDevice,
    RunResponse, RuntimeFlags, UvOpenvinoExtraRuntimeConfiguration,
};
use pipette_subprocess::{Process, start_process};

use crate::completions;
use crate::models::ResolvedOpenvinoModel;
use crate::response;
use crate::runtimes::ResolvedOpenvinoPython;
use crate::server;

const DECODE_TOKENS: u32 = 1;

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    runtime_flags: MaxMemoryUsageUvOpenvinoOpenvino,
    python: &ResolvedOpenvinoPython,
    model: &ResolvedOpenvinoModel,
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
    let flags = RuntimeFlags::MaxMemoryUsageUvOpenvinoOpenvino(runtime_flags);
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
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
                prefill_tokens,
                observer,
            )?)
        },
    );
    let result_data = result.map_err(|err| match &mut served {
        Some((openvino_server, _)) => err.context(openvino_server.stop_with_tail()),
        None => err,
    })?;
    Ok(served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (openvino_server, memory)| response.process(openvino_server).memory(*memory),
        )
        .command(response::command(python))
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
    prefill_tokens: u32,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<MaxMemoryMeasurementReading> {
    let warm_up = served.is_none();
    let earlier = match served.take() {
        Some((mut openvino_server, memory)) => {
            openvino_server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut openvino_server =
        start_process(server::SCRIPT_FILENAME, command(port)?, server::STOP_GRACE)?;
    // Watched from the launch, so the load counts.
    let sampler = Sampler::start(openvino_server.pid(), watch)?;
    let ready = server::ready(&url)?;
    // Off the clock on the warm-up's server, which compiles a cold cache; the
    // measured one loads what it wrote.
    if warm_up {
        observer.outside(|| openvino_server.wait_ready(server::READY_TIMEOUT, ready))?;
    } else {
        openvino_server.wait_ready(server::READY_TIMEOUT, ready)?;
    }
    let measured = measure(http, &url, prefill_tokens);
    // Stopped while the server lives: Linux reads its `/proc` entry.
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((openvino_server, earlier.merge_max(observed)));
    measured?;
    let peak = peak?;
    Ok(MaxMemoryMeasurementReading {
        max_host_bytes: peak.host_bytes,
        max_gpu_bytes: peak.gpu_bytes,
        max_npu_bytes: peak.npu_bytes,
    })
}

/// One request at the cell's shape.
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
            crate::runtimes::device_property(device),
            read.join(" or ")
        )
    })
}

/// A custom device (`AUTO`, `HETERO:GPU,CPU`, `GPU.1`) may resolve anywhere,
/// so it requires none. `pipette-ovms` keeps the same rule.
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
            crate::runtimes::device_property(&OpenvinoDevice::Npu)
        );
        assert_eq!(err.to_string(), expected);
    }
}
