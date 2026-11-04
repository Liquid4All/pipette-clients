//! The run path for one cell (see `docs/architecture.md`): [`run_cell`]
//! resolves a [`ClientRunSpec`] into a [`RunRequest`] and dispatches it to an
//! engine. Recording and submitting are the caller's
//! (`record_and_maybe_submit_run` for the CLI, the claim loop for the worker).

use std::sync::{Mutex, MutexGuard};

use anyhow::Context;

use pipette_artifacts::quota::SweepPins;
use pipette_artifacts::{
    ArtifactsContext, ensure_model, ensure_runtime, model_download_size, runtime_download_size,
};

use pipette_device::{MemoryPressure, detect_device_readings, detect_power_state, detect_thermal};
use pipette_ops::eval_completions::{EvalRunDigest, run_request_digest};
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::run_context::RunContext;
use pipette_ops::thermal_series::ThermalSeries;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, BenchmarkFlagsError, BenchmarkSubmissionPayload,
    BenchmarkType, ClientRunSpec, DeclaredBound, ExtraRuntimeConfiguration, ModelFlags,
    ReadinessOverrides, RunRequest, RunResponse, RunThermal, Runtime, RuntimeType, SkinCelsius,
};

use pipette_programs::ExternalPrograms;

use crate::client_configuration::ClientConfiguration;
use crate::identity::DeviceLabels;
use crate::preflight;
use crate::progress::{CellProgress, ReadinessProgress};
use crate::rep_telemetry::{GateWaitRecorder, RepSample, RepTelemetry};
use crate::results::{BenchmarkResultExtras, finished_run_payload};
use crate::workspace::PipetteWorkspace;

/// Bind, run and describe one cell, without recording or submitting it.
///
/// `client_config` is the host's, not the cell's: the same spec under another
/// configuration is the same cell, so it stays out of the request. An eval, a
/// vlm_eval or an asr_eval also returns its checkpoint's digest, for
/// [`finalize_eval_checkpoint`] once the result is saved or accepted.
///
/// `held_touch` is the `--readiness-held-touch` flag. `None` leaves the mode to
/// `PIPETTE_READINESS_HELD_TOUCH` and the host's table.
///
/// `cancelled` reaches only the readiness gate: pulling, loading and measuring
/// have no cancel seam, so a caller with a stop signal waits those out.
/// `benchmarks run` passes [`pipette_wait::never_cancelled`]: a ^C there tears
/// the process down through the subprocess cleanup handler.
#[expect(
    clippy::too_many_arguments,
    reason = "the cell, what it runs against, and how it names the device"
)]
pub fn run_cell(
    spec: &ClientRunSpec,
    benchmark: BenchmarkDefinition,
    artifacts_ctx: &ArtifactsContext,
    ws: &PipetteWorkspace,
    labels: &DeviceLabels,
    client_config: &ClientConfiguration,
    held_touch: Option<pipette_readiness::HeldTouch>,
    cancelled: &dyn Fn() -> bool,
    programs: &ExternalPrograms,
) -> anyhow::Result<(
    BenchmarkSubmissionPayload,
    BenchmarkResultExtras,
    Option<EvalRunDigest>,
)> {
    // Ahead of prepare: its ensures can take minutes, and a stall there is only
    // attributable if the cell is already named.
    tracing::info!(
        ?spec,
        client_config = client_config.to_string().as_str(),
        "run"
    );
    // Before the readiness gate, which would otherwise wait on what a killed
    // run left busy.
    pipette_subprocess::reap_orphans();
    client_config.runtimes.admit(spec)?;
    // Before prepare, so a battery refusal does not wait on minutes of weights.
    refuse_on_battery(&benchmark)?;
    // mgmt checks a claimed cell at ingestion; a local run arrives unchecked.
    if let BenchmarkDefinition::Eval(eval) = &benchmark {
        eval.check_cell_context(RuntimeType::of(&spec.runtime), spec.runtime_flags.as_ref())?;
    }
    // Held until the cell returns, so other processes' sweeps leave the runtime
    // and the model alone while the engine runs from them.
    ensure_skin_on_android(spec.benchmark_flags.as_ref())?;
    let (req, _in_use) = prepare(spec, benchmark, artifacts_ctx, ws, programs)?;
    // Resolved here, not inside the gate: the same value has to drive the wait
    // and the record, or the result describes a run that did not happen.
    let readiness = pipette_readiness::resolve_readiness(
        readiness_max_wait(&req),
        readiness_thermal_gate(&req),
    )
    .with_configuration(client_config.readiness.clone())
    .with_skin_below_c(readiness_skin_below(&req)?)
    .with_held_touch(held_touch.or_else(env_held_touch));
    // Resolved once and spent twice, by the engine and the record, so the two
    // cannot describe different configurations.
    let entry = client_config.runtimes.for_runtime(&req.runtime.declared);
    let (outcome, telemetry) = dispatch_run(&req, ws, readiness, entry, cancelled, programs)?;
    let (payload, extras) = finished_run_payload(labels, &req, &outcome, entry, &telemetry)?;
    let eval_checkpoint = eval_checkpoint(&req)?;
    Ok((payload, extras, eval_checkpoint))
}

/// The runner's environment override, shared by `benchmarks run` and the
/// worker. A malformed value cannot change the mode or fail an unrelated run.
fn env_held_touch() -> Option<pipette_readiness::HeldTouch> {
    const ENV: &str = "PIPETTE_READINESS_HELD_TOUCH";
    let raw = std::env::var(ENV).ok()?;
    raw.parse()
        .inspect_err(|err| {
            tracing::warn!(
                var = ENV,
                error = %err,
                "ignoring an unparseable held-touch override; using the host configuration"
            );
        })
        .ok()
}

/// The checkpoint an accuracy eval's engine opened, under the identity every
/// engine opens it under: this request. Every accuracy eval keeps one, and one
/// left behind would hand its answers to the next run of the cell without
/// running it. `None` for a benchmark that keeps none.
pub(crate) fn eval_checkpoint(req: &RunRequest) -> anyhow::Result<Option<EvalRunDigest>> {
    matches!(
        req.benchmark,
        BenchmarkDefinition::Eval(_)
            | BenchmarkDefinition::VlmEval(_)
            | BenchmarkDefinition::AsrEval(_)
    )
    .then(|| run_request_digest(req))
    .transpose()
}

/// Retire an eval's checkpoint once its result is saved or accepted by the
/// server; until then it is what a lost result is rebuilt from. Best-effort: a
/// checkpoint left behind is resumed, whole, by the next run of the cell.
pub fn finalize_eval_checkpoint(ws: &PipetteWorkspace, digest: Option<&EvalRunDigest>) {
    let Some(digest) = digest else {
        return;
    };
    if let Err(e) = ws.eval_completions().finalize(digest) {
        tracing::warn!(
            %digest,
            error = %format_args!("{e:#}"),
            "eval checkpoint not finalized"
        );
    }
}

/// Ensure model + runtime into a [`RunRequest`] around a benchmark body the
/// caller resolved: only the caller knows which catalog it meant (a claim the
/// remote one, `local/<id>` on the CLI the other). Returned beside the locks
/// that keep both entries in the store while held.
fn prepare(
    spec: &ClientRunSpec,
    benchmark: BenchmarkDefinition,
    artifacts_ctx: &ArtifactsContext,
    ws: &PipetteWorkspace,
    programs: &ExternalPrograms,
) -> anyhow::Result<(RunRequest, [Option<std::fs::File>; 2])> {
    // The payload's id and the eval digest come from the body, so a mismatch
    // would record a cell that never ran.
    anyhow::ensure!(
        &spec.benchmark == benchmark.benchmark_id(),
        "spec names benchmark `{}` but the resolved body is `{}`",
        spec.benchmark,
        benchmark.benchmark_id()
    );
    require_desktop_runtime(&spec.runtime)?;
    let runtimes = ws.runtimes();
    let models = ws.models();

    validate_spec_flags(spec, &benchmark)?;

    // Ahead of any download: a catalog vlm_eval runs only the models its
    // recipe covers, in its thinking mode. The engine checks again before the
    // corpus, for a caller that skipped this.
    if let BenchmarkDefinition::VlmEval(vlm) = &benchmark
        && let Some(recipe) = &vlm.recipe
    {
        recipe
            .check(
                &spec.model,
                spec.model_flags
                    .as_ref()
                    .and_then(ModelFlags::enable_thinking),
            )
            .with_context(|| format!("vlm_eval `{}` is off its recipe", vlm.benchmark_id))?;
    }
    // A catalog asr_eval likewise, its cell's thinking off as its recipe's is.
    if let BenchmarkDefinition::AsrEval(asr) = &benchmark
        && let Some(recipe) = &asr.recipe
    {
        recipe
            .check_cell(
                BenchmarkType::AsrEval,
                &spec.model,
                spec.model_flags
                    .as_ref()
                    .and_then(ModelFlags::enable_thinking),
            )
            .with_context(|| format!("asr_eval `{}` is off its recipe", asr.benchmark_id))?;
    }

    // Ahead of ensure, so a broken driver fails before an image pull or a venv
    // build.
    preflight::assert_gpu_ready(&spec.runtime, programs)
        .with_context(|| format!("GPU preflight for runtime `{}`", spec.runtime))?;

    // The in-flight plan's pin set: the runtime's ensure must not evict the
    // model the model's ensure is about to resolve, and vice versa.
    let ctx = artifacts_ctx.with_pins(SweepPins::for_cell(&spec.model, &spec.runtime));
    // Sized before either starts, so the cell's total is whole from the first
    // byte. A cache hit sizes at zero without touching the network.
    let progress = CellProgress::new(&[
        runtime_download_size(&runtimes, &spec.runtime)
            .with_context(|| format!("sizing runtime `{}`", spec.runtime))?,
        model_download_size(&ctx, &models, &spec.model)
            .with_context(|| format!("sizing model `{}`", spec.model))?,
    ]);
    let ctx = ctx.with_progress(progress.sink());
    let (bound_runtime, runtime_in_use) = ensure_runtime(&ctx, &runtimes, &spec.runtime)
        .with_context(|| format!("ensuring runtime `{}`", spec.runtime))?;
    let (bound_model, model_in_use) = ensure_model(&ctx, &models, &spec.model)
        .with_context(|| format!("ensuring model `{}`", spec.model))?;

    let req = RunRequest {
        runtime: DeclaredBound {
            declared: spec.runtime.clone(),
            bound: bound_runtime,
        },
        model: DeclaredBound {
            declared: spec.model.clone(),
            bound: bound_model,
        },
        runtime_flags: spec.runtime_flags.clone(),
        model_flags: spec.model_flags.clone(),
        benchmark_flags: spec.benchmark_flags.clone(),
        benchmark,
    };
    Ok((req, [runtime_in_use, model_in_use]))
}

/// Refuse to start a corpus cell on battery; every other kind passes.
///
/// The server records the power state without rejecting, so this is the only
/// guard. Both arms gate, because either row ends up in a ratio, and a ratio
/// whose halves ran under different power states is not a measurement.
fn refuse_on_battery(benchmark: &BenchmarkDefinition) -> anyhow::Result<()> {
    if benchmark.benchmark_type() != BenchmarkType::VlSpecDecode {
        return Ok(());
    }
    let power = detect_power_state();
    match power_verdict(power.power_state) {
        PowerVerdict::Stable => Ok(()),
        PowerVerdict::Moving => anyhow::bail!(
            "`{}` measures two arms against each other, so it refuses to start \
             while the power state can move under it (device_power_state={:?}, \
             battery {:?}%). Run it on AC with the battery near full.",
            benchmark.benchmark_id(),
            power.power_state,
            power.battery_level
        ),
        PowerVerdict::Unknown => {
            tracing::warn!(
                benchmark_id = %benchmark.benchmark_id(),
                "could not read this host's power state; it must be on AC with \
                 the battery near full, or the two arms are not comparable"
            );
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PowerVerdict {
    /// The one state that holds still. A battery-less desktop reports it too.
    Stable,
    /// On battery, or charging: a fast-charging battery throttles the GPU even
    /// on AC.
    Moving,
    /// Refusing here would make the cell unrunnable on every host this client
    /// cannot read.
    Unknown,
}

fn power_verdict(state: Option<pipette_plan_types::DevicePowerState>) -> PowerVerdict {
    use pipette_plan_types::DevicePowerState;
    match state {
        Some(DevicePowerState::PluggedInNotCharging) => PowerVerdict::Stable,
        Some(DevicePowerState::Charging | DevicePowerState::NotCharging) => PowerVerdict::Moving,
        None => PowerVerdict::Unknown,
    }
}

fn validate_spec_flags(
    spec: &ClientRunSpec,
    benchmark: &BenchmarkDefinition,
) -> anyhow::Result<()> {
    let bt = benchmark.benchmark_type();
    if let Some(f) = &spec.model_flags
        && !f.matches(bt, &spec.model)
    {
        anyhow::bail!(
            "model_flags do not match resolved benchmark ({bt:?}) / model ({})",
            spec.model
        );
    }
    if let Some(f) = &spec.runtime_flags
        && !f.matches(bt, &spec.runtime, &spec.model)
    {
        anyhow::bail!(
            "runtime_flags do not match resolved cell (benchmark={bt:?}, runtime={}, model={})",
            spec.runtime,
            spec.model
        );
    }
    if let Some(f) = &spec.benchmark_flags
        && !f.matches(bt, &spec.runtime, &spec.model)
    {
        anyhow::bail!(
            "benchmark_flags do not match resolved cell (benchmark={bt:?}, runtime={}, model={})",
            spec.runtime,
            spec.model
        );
    }
    Ok(())
}

fn require_desktop_runtime(runtime: &Runtime) -> anyhow::Result<()> {
    match runtime {
        Runtime::LlamacppCliStockTools(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_) => Ok(()),
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleFoundation(_) => anyhow::bail!(
            "runtime `{runtime}` is not a desktop CLI runtime; \
             allowed: llamacpp_cli_stock_tools, mlx_macos_pipette, \
             docker_vllm, docker_sglang, uv_vllm, uv_sglang, uv_openvino, \
             uv_onnx_genai, ovms, trt_edgellm"
        ),
    }
}

/// A runtime this host's OS does not run. Retriable: a host on another OS can
/// take the job.
#[derive(Debug, thiserror::Error)]
#[error("runtime `{runtime}` runs on {}, and this host is {host_os}", runs_on.join(" and "))]
pub struct ForeignOsError {
    runtime: &'static str,
    runs_on: &'static [&'static str],
    host_os: String,
}

/// The OSes `runtime` runs on, spelled as [`std::env::consts::OS`] spells
/// them; `None` leaves the runtime to its own install or dispatch check.
fn runs_on(runtime: &Runtime) -> Option<&'static [&'static str]> {
    match runtime {
        Runtime::UvOpenvino(_) | Runtime::Ovms(_) => Some(&["linux", "windows"]),
        Runtime::TrtEdgellm(_) => Some(&["linux"]),
        Runtime::AppleCoreAiMacosPipette(_) => Some(&["macos"]),
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleFoundation(_) => None,
    }
}

/// Called where a spec is built, before anything installs: OpenVINO's install
/// is not gated, and OVMS's flavor check skips a custom build and an install
/// already in the store.
pub(crate) fn refuse_foreign_os(runtime: &Runtime, host_os: &str) -> Result<(), ForeignOsError> {
    match runs_on(runtime) {
        Some(runs_on) if !runs_on.contains(&host_os) => Err(ForeignOsError {
            runtime: runtime.headless_token(),
            runs_on,
            host_os: host_os.to_owned(),
        }),
        _ => Ok(()),
    }
}

/// The cell's `benchmark_flags` deadline; `None` leaves the platform default.
fn readiness_max_wait(req: &RunRequest) -> Option<std::time::Duration> {
    req.benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::readiness)
        .and_then(|r| r.max_wait_secs)
        .map(std::time::Duration::from_secs)
}

/// Refuse a cell flag that this host has no gate to apply.
pub(crate) fn ensure_skin_on_android(flags: Option<&BenchmarkFlags>) -> anyhow::Result<()> {
    let has_skin = flags
        .and_then(BenchmarkFlags::readiness)
        .and_then(|r| r.skin_below_c)
        .is_some();
    anyhow::ensure!(
        !has_skin || pipette_readiness::platform() == pipette_readiness::ReadinessPlatform::Android,
        "readiness.skin_below_c is supported only by the Android CLI gate"
    );
    Ok(())
}

/// Refuse a `--readiness-held-touch` that this host has no gate to apply.
pub(crate) fn ensure_held_touch_on_android(
    held_touch: Option<pipette_readiness::HeldTouch>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        held_touch.is_none()
            || pipette_readiness::platform() == pipette_readiness::ReadinessPlatform::Android,
        "--readiness-held-touch is supported only by the Android CLI gate"
    );
    Ok(())
}

fn readiness_skin_below(
    req: &RunRequest,
) -> anyhow::Result<Option<pipette_readiness::CeilingCelsius>> {
    req.benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::readiness)
        .and_then(|r| r.skin_below_c)
        .map(|c| pipette_readiness::CeilingCelsius::try_from(c.into_inner()))
        .transpose()
        .map_err(Into::into)
}

/// Three answers, not two: an absent knob defers to
/// `PIPETTE_READINESS_SKIP_THERMAL` (a fleet-wide waiver), while an authored
/// `skip_thermal = false` overrides a stale export in the runner's shell.
fn readiness_thermal_gate(req: &RunRequest) -> pipette_readiness::ThermalGate {
    match req
        .benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::readiness)
        .and_then(|r| r.skip_thermal)
    {
        Some(true) => pipette_readiness::ThermalGate::Skip,
        Some(false) => pipette_readiness::ThermalGate::Enforce,
        None => pipette_readiness::ThermalGate::Unset,
    }
}

/// A poisoned series fails the run rather than reporting a partial series.
fn lock(
    series: &Mutex<ThermalSeries<RepSample>>,
) -> anyhow::Result<MutexGuard<'_, ThermalSeries<RepSample>>> {
    series
        .lock()
        .map_err(|_| anyhow::anyhow!("the thermal series lock was poisoned by an earlier panic"))
}

/// The cell's benchmark flags as the run resolved them: the authored entry with
/// the readiness block filled in from what applied (the authored block says
/// `None` where the environment or a platform default decided).
///
/// `None` rather than an empty variant: the server NULLs an empty
/// `benchmark_flags` object anyway. An unset peak-memory cell resolves empty:
/// it gates on nothing and has only optional knobs.
fn resolved_flags(
    req: &RunRequest,
    readiness: pipette_readiness::ResolvedReadiness,
) -> anyhow::Result<Option<BenchmarkFlags>> {
    let mut r = req.benchmark_flags_ref()?;
    // Which cells gate is also encoded in whether their flag variant carries a
    // readiness field. Exhaustive so a new benchmark type has to answer here,
    // instead of reporting no readiness for a cell that waited.
    let gates = match req.benchmark.benchmark_type() {
        // vlm_eval gates for device protection on llama.cpp (the vision tower
        // runs thousands of images back-to-back), so that flag variant carries
        // readiness like the timing cells. On vLLM it runs ungated, as the text
        // eval there does, and its variant carries none.
        // asr_eval likewise, for the audio tower.
        BenchmarkType::VlmEval | BenchmarkType::AsrEval => {
            RuntimeType::of(&req.runtime.bound) != RuntimeType::UvVllm
        }
        BenchmarkType::PrefillThroughput
        | BenchmarkType::DecodeThroughput
        | BenchmarkType::EndToEndLatency
        | BenchmarkType::VlThroughput
        | BenchmarkType::AsrThroughput
        // Gates before the traversal and before every reference-batch sample.
        | BenchmarkType::VlSpecDecode => true,
        // vl_max_memory drives a server but takes one ungated measurement — a
        // peak is thermal-independent, so its flag variant carries no readiness.
        BenchmarkType::Eval
        | BenchmarkType::MaxMemoryUsage
        | BenchmarkType::VlMaxMemory
        | BenchmarkType::AsrMaxMemory => false,
    };
    if gates {
        r.readiness = Some(ReadinessOverrides {
            max_wait_secs: Some(readiness.max_wait_secs()),
            skip_thermal: Some(readiness.skip_thermal()),
            skin_below_c: readiness
                .skin_below_c()
                .map(|c| SkinCelsius::try_new(c.get()))
                .transpose()?,
        });
    }
    match BenchmarkFlags::try_from(r) {
        Ok(flags) if reports_nothing(&flags) => Ok(None),
        Ok(flags) => Ok(Some(flags)),
        // A cell the flag schema does not model (Apple, mobile) has nothing to
        // report.
        Err(BenchmarkFlagsError::NoSuchCombination { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Whether these flags would submit as an empty object.
fn reports_nothing(flags: &BenchmarkFlags) -> bool {
    flags
        .submission_value()
        .as_object()
        .is_some_and(serde_json::Map::is_empty)
}

/// Earlier cells' model files stay cached, and on a phone the next load
/// reclaiming them can swap it out
/// (docs/methodology/galaxy-s26-ultra-memory-pressure.md).
#[cfg(target_os = "android")]
fn drop_model_cache(ws: &PipetteWorkspace, memory: &MemoryPressure) {
    // Bracketed, so the log shows the state the drop found and what it freed.
    memory.log("before_drop");
    pipette_device::drop_cached_files(ws.models().models_dir());
    memory.log("after_drop");
}

/// Takes the resolved readiness rather than resolving it: resolving twice could
/// straddle an environment change, and the gate and the record must agree.
fn dispatch_run(
    req: &RunRequest,
    ws: &PipetteWorkspace,
    readiness: pipette_readiness::ResolvedReadiness,
    runtime_config: Option<&ExtraRuntimeConfiguration>,
    cancelled: &dyn Fn() -> bool,
    programs: &ExternalPrograms,
) -> anyhow::Result<(RunResponse, RepTelemetry)> {
    let readiness_progress = ReadinessProgress::new();
    let gate_wait_recorder = GateWaitRecorder::default();
    let report = |status: &pipette_readiness::ReadinessStatus| {
        readiness_progress.report(status);
        gate_wait_recorder.observe(status);
    };
    let polled = || gate_wait_recorder.polled();
    // The gate is where a stop signal lands: it is minutes long, and with no
    // child running the worker's handler absorbs a ^C into its flag. Without
    // this nobody reads that flag until the deadline.
    let callbacks = pipette_readiness::ReadinessCallbacks::new()
        .with_status(&report)
        .with_poll(&polled)
        .with_cancel(cancelled);
    let readiness_gate = || -> anyhow::Result<()> {
        gate_wait_recorder.begin();
        pipette_readiness::wait_until_ready_with(&readiness, &callbacks)?;
        gate_wait_recorder.pass();
        Ok(())
    };
    // The single pending slot assumes reps run one at a time. Probe before
    // taking the lock: `detect_thermal` shells out on some platforms, and
    // `detect_device_readings` takes a second on Windows.
    let series = Mutex::new(ThermalSeries::default());
    // The gate falls between reps: a rep's window holds its prepare and timed
    // run (a llama-bench rep's own load), the one before rep 1 the first gate
    // and the warm-up (a server's load).
    let memory = MemoryPressure::default();
    let observer = MeasurementObserver {
        on_before_rep_start: &|| {
            let sample = RepSample {
                thermal: detect_thermal(),
                device: detect_device_readings(),
                gate: gate_wait_recorder.take(),
            };
            // Last, as `after_rep` is first, so the rep's window leaves the probes out.
            memory.log("before_rep");
            lock(&series)?.start(sample)
        },
        on_after_rep_end: &|| {
            // First, before the probes that shell out.
            memory.log("after_rep");
            let sample = RepSample {
                thermal: detect_thermal(),
                device: detect_device_readings(),
                gate: None,
            };
            lock(&series)?.finish(sample)
        },
    };
    // After the model and runtime downloads, before the warm-up.
    #[cfg(target_os = "android")]
    drop_model_cache(ws, &memory);
    memory.log("before_cell");
    // The whole object, so a new field is logged too; `AuthToken` redacts
    // itself. Debug level: an eval's request carries its whole dataset.
    tracing::debug!(request = ?req, "dispatch");
    let mut response = match (&req.runtime.declared, &req.benchmark) {
        (Runtime::LlamacppCliStockTools(_), _) => pipette_llamacpp::run(
            req,
            ws,
            &readiness_gate,
            observer,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_llamacpp)
                .transpose()?,
        ),
        #[cfg(target_os = "macos")]
        (Runtime::MlxMacosPipette(_), _) => pipette_mlx::run(
            req,
            ws,
            &readiness_gate,
            observer,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_mlx)
                .transpose()?,
        ),
        #[cfg(not(target_os = "macos"))]
        (Runtime::MlxMacosPipette(_), _) => anyhow::bail!("the MLX runtime runs on macOS only"),

        #[cfg(target_os = "macos")]
        (Runtime::AppleCoreAiMacosPipette(_), _) => {
            pipette_coreai::run(req, &readiness_gate, &observer)
        }
        #[cfg(not(target_os = "macos"))]
        (Runtime::AppleCoreAiMacosPipette(_), _) => {
            anyhow::bail!("the Core AI runtime runs on macOS 27+ only")
        }

        (
            Runtime::DockerVllm(_)
            | Runtime::DockerSglang(_)
            | Runtime::UvVllm(_)
            | Runtime::UvSglang(_),
            _,
        ) => pipette_torch_oai::run(
            req,
            ws,
            &readiness_gate,
            observer,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_torch_oai)
                .transpose()?,
            programs,
        ),

        (Runtime::UvOpenvino(_), _) => pipette_openvino::run(
            req,
            &readiness_gate,
            observer,
            &ws.compile_cache(&req.runtime.declared)?,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_openvino)
                .transpose()?,
        ),

        (Runtime::UvOnnxGenai(_), _) => pipette_onnx_genai::run(
            req,
            &readiness_gate,
            observer,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_onnx_genai)
                .transpose()?,
        ),

        (Runtime::Ovms(_), _) => pipette_ovms::run(
            req,
            &readiness_gate,
            observer,
            &ws.compile_cache(&req.runtime.declared)?,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_ovms)
                .transpose()?,
        ),

        (Runtime::TrtEdgellm(_), _) => pipette_trt_edgellm::run(
            req,
            &readiness_gate,
            observer,
            &ws.compile_cache(&req.runtime.declared)?,
            runtime_config
                .map(ExtraRuntimeConfiguration::as_trt_edgellm)
                .transpose()?,
        ),

        (
            Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::AppleFoundation(_),
            _,
        ) => anyhow::bail!(
            "runtime `{}` is not a desktop CLI runtime",
            req.runtime.declared
        ),
    }?;

    // The engines mark the reps; the readings are this side's.
    let reps = lock(&series)?.take()?;
    let telemetry = RepTelemetry::from_reps(&reps);
    response.thermal = RunThermal::from_pairs(
        reps.into_iter()
            .map(|(start, end)| (start.thermal, end.thermal)),
    );
    response.benchmark_flags = resolved_flags(req, readiness)?;
    Ok((response, telemetry))
}

#[cfg(test)]
mod fixtures {
    use pipette_plan_types::{BenchmarkDefinition, DeclaredBound, Model, RunRequest, Runtime};

    pub(super) fn gguf_text() -> anyhow::Result<Model> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "gguf_text", "source": "huggingface",
            "org": "o", "repo_name": "r", "path": "m-Q4_0.gguf"
        }))?)
    }

    pub(super) fn llamacpp() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "llamacpp_cli_stock_tools", "source": "github_release",
            "repository_url": "github.com/ggml-org/llama.cpp",
            "version": "b5000", "flavor": "macos-arm64"
        }))?)
    }

    pub(super) fn openvino() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "uv_openvino", "server_version": "2026.4.1", "python_version": "3.11",
            "source": {
                "type": "pip_requirements_text",
                "contents": "openvino-genai==2026.4.1.0\n"
            }
        }))?)
    }

    /// No authored flags, so what the record carries came from resolution.
    pub(super) fn req(benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(llamacpp()?),
            model: DeclaredBound::already_bound(gguf_text()?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }
}

#[cfg(test)]
mod run_gate_tests {
    use pipette_plan_types::{DevicePowerState, Runtime};

    use super::*;

    fn onnx_genai() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "uv_onnx_genai", "server_version": "0.16.0",
            "execution_provider": "cpu", "python_version": "3.11",
            "source": {
                "type": "pip_requirements_text",
                "contents": "onnxruntime-genai==0.16.0\n"
            }
        }))?)
    }

    fn apple_foundation() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(
            serde_json::json!({"type": "apple_foundation"}),
        )?)
    }

    fn ovms() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "ovms", "source": "github_release",
            "repository_url": "github.com/openvinotoolkit/model_server",
            "repository_version": "v2026.4.1", "flavor": "windows-x64"
        }))?)
    }

    fn llamacpp_linux_cuda() -> anyhow::Result<Runtime> {
        Ok(serde_json::from_value(serde_json::json!({
            "type": "llamacpp_cli_stock_tools", "source": "remote_archive",
            "url": "https://example.com/llama-b10216-bin-linux-cuda-12.8-x64.tar.gz",
            "flavor": "linux-cuda-12-8"
        }))?)
    }

    #[rstest::rstest]
    #[case::llamacpp(super::fixtures::llamacpp, None)]
    #[case::onnx_genai(onnx_genai, None)]
    #[case::apple_foundation(apple_foundation, Some("not a desktop CLI runtime"))]
    #[case::ovms(ovms, None)]
    fn the_desktop_gate_admits_desktop_runtimes_only(
        #[case] runtime: fn() -> anyhow::Result<Runtime>,
        #[case] refusal: Option<&str>,
    ) -> anyhow::Result<()> {
        match (require_desktop_runtime(&runtime()?), refusal) {
            (Ok(()), None) => {}
            (Err(err), Some(reason)) => assert!(err.to_string().contains(reason), "{err}"),
            (Ok(()), Some(reason)) => anyhow::bail!("admitted, expected {reason:?}"),
            (Err(err), None) => anyhow::bail!("refused: {err}"),
        }
        Ok(())
    }

    use super::fixtures::openvino;

    /// Every runtime but OpenVINO and OVMS is left to its own checks, so a
    /// llama.cpp build for another OS fails at install or launch.
    #[rstest::rstest]
    #[case::openvino_linux(openvino, "linux", true)]
    #[case::openvino_windows(openvino, "windows", true)]
    #[case::openvino_macos(openvino, "macos", false)]
    #[case::openvino_android(openvino, "android", false)]
    #[case::openvino_ios(openvino, "ios", false)]
    #[case::ovms_linux(ovms, "linux", true)]
    #[case::ovms_windows(ovms, "windows", true)]
    #[case::ovms_macos(ovms, "macos", false)]
    #[case::ovms_android(ovms, "android", false)]
    #[case::ovms_ios(ovms, "ios", false)]
    #[case::llamacpp_macos(super::fixtures::llamacpp, "macos", true)]
    #[case::llamacpp_linux_cuda_macos(llamacpp_linux_cuda, "macos", true)]
    fn the_os_gate_admits_the_oses_a_runtime_runs_on(
        #[case] runtime: fn() -> anyhow::Result<Runtime>,
        #[case] host_os: &str,
        #[case] admitted: bool,
    ) -> anyhow::Result<()> {
        match (refuse_foreign_os(&runtime()?, host_os), admitted) {
            (Ok(()), true) => Ok(()),
            (Err(err), false) => {
                anyhow::ensure!(
                    err.to_string()
                        .ends_with(&format!("this host is {host_os}")),
                    "{err}"
                );
                Ok(())
            }
            (Err(err), true) => anyhow::bail!("refused on {host_os}: {err}"),
            (Ok(()), false) => anyhow::bail!("admitted on {host_os}"),
        }
    }

    #[test]
    fn the_os_refusal_names_the_runtime_and_both_oses() -> anyhow::Result<()> {
        let err = refuse_foreign_os(&openvino()?, "macos")
            .err()
            .context("macos must be refused")?;
        assert_eq!(
            err.to_string(),
            "runtime `uv_openvino` runs on linux and windows, and this host is macos"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case(Some(DevicePowerState::PluggedInNotCharging), PowerVerdict::Stable)]
    #[case(Some(DevicePowerState::Charging), PowerVerdict::Moving)]
    #[case(Some(DevicePowerState::NotCharging), PowerVerdict::Moving)]
    #[case(None, PowerVerdict::Unknown)]
    fn the_power_gate_admits_only_a_state_that_holds_still(
        #[case] state: Option<DevicePowerState>,
        #[case] expected: PowerVerdict,
    ) {
        assert_eq!(power_verdict(state), expected);
    }
}

#[cfg(test)]
mod resolved_flags_tests {
    use std::time::Duration;

    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkId, DeclaredBound, DecodeThroughputBenchmark,
        EndToEndLatencyBenchmark, EvalBenchmark, MaxMemoryUsageBenchmark, Model, NonNegativeCount,
        PositiveCount, PositiveSeconds, PrefillThroughputBenchmark, Runtime, VlMaxMemoryBenchmark,
    };

    use super::*;

    use super::fixtures::req;

    /// The text peak-memory cell: its one knob is a wall-clock deadline.
    fn max_memory_req() -> anyhow::Result<RunRequest> {
        req(BenchmarkDefinition::MaxMemoryUsage(
            MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new("max_memory_usage".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            },
        ))
    }

    /// The VL peak-memory cell: over HTTP, so a request timeout as well.
    fn vl_max_memory_req() -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_vision", "source": "huggingface",
            "org": "o", "repo_name": "r", "model": "a.gguf", "mmproj": "mm.gguf"
        }))?;
        Ok(RunRequest {
            model: DeclaredBound::already_bound(model),
            ..req(BenchmarkDefinition::VlMaxMemory(VlMaxMemoryBenchmark {
                benchmark_id: BenchmarkId::try_new("vl_max_memory".to_owned())?,
                parameter_image_width: PositiveCount::try_new(512)?,
                parameter_image_height: PositiveCount::try_new(512)?,
                parameter_text_tokens: NonNegativeCount::try_new(64)?,
                parameter_decode_tokens: None,
                parameter_num_images: NonNegativeCount::try_new(1)?,
            }))?
        })
    }

    /// The ASR peak-memory cell, on the same `gguf_vision` model.
    fn asr_max_memory_req() -> anyhow::Result<RunRequest> {
        use pipette_plan_types::{
            AsrDecodeTokens, AsrMaxMemoryBenchmark, AudioChannels, AudioDurationMillis,
            AudioSampleRate,
        };

        Ok(RunRequest {
            benchmark: BenchmarkDefinition::AsrMaxMemory(AsrMaxMemoryBenchmark {
                benchmark_id: BenchmarkId::try_new(
                    "asr_max_memory_10000ms_text0_decode256".to_owned(),
                )?,
                parameter_audio_duration_ms: AudioDurationMillis::try_new(10_000)?,
                parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
                parameter_audio_channels: AudioChannels::try_new(1)?,
                parameter_text_tokens: NonNegativeCount::try_new(0)?,
                parameter_decode_tokens: AsrDecodeTokens::try_new(256)?,
            }),
            ..vl_max_memory_req()?
        })
    }

    /// The same request on the OpenVINO seam: a uv-venv runtime driving an
    /// OpenVINO IR.
    fn openvino_req(benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "openvino", "source": "huggingface",
            "org": "o", "repo_name": "r", "prefix": "int4-sym-cw"
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(super::fixtures::openvino()?),
            model: DeclaredBound::already_bound(model),
            ..req(benchmark)?
        })
    }

    fn gated() -> pipette_readiness::ResolvedReadiness {
        pipette_readiness::resolve_readiness(
            Some(Duration::from_secs(300)),
            pipette_readiness::ThermalGate::Enforce,
        )
    }

    /// Across backends: a cell whose variant cannot hold the block would report
    /// a run that waited without saying under what.
    #[rstest]
    #[case::llamacpp_prefill(req(BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
        parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
        parameter_batch_size: BatchSize::one(),
    })))]
    #[case::openvino_prefill(openvino_req(BenchmarkDefinition::PrefillThroughput(
        PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            parameter_batch_size: BatchSize::one(),
        }
    )))]
    #[case::openvino_decode(openvino_req(BenchmarkDefinition::DecodeThroughput(
        DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_256_64".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            parameter_decode_tokens: PositiveCount::try_new(64)?,
            parameter_batch_size: BatchSize::one(),
        }
    )))]
    #[case::openvino_end_to_end(openvino_req(BenchmarkDefinition::EndToEndLatency(
        EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new("end_to_end_latency_256_64".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            parameter_decode_tokens: PositiveCount::try_new(64)?,
            parameter_batch_size: BatchSize::one(),
        }
    )))]
    fn a_gated_cell_reports_the_policy_it_applied(
        #[case] req: anyhow::Result<RunRequest>,
    ) -> anyhow::Result<()> {
        let flags = resolved_flags(&req?, gated())?.context("a gated cell must report flags")?;

        assert_eq!(
            flags.submission_value().get("readiness"),
            Some(&serde_json::json!({"max_wait_secs": 300, "skip_thermal": false}))
        );
        Ok(())
    }

    #[rstest]
    #[case::host(36, None, false, 36)]
    #[case::cell_overrides_host(36, Some(34), false, 34)]
    #[case::waived_still_records_bound(36, Some(34), true, 34)]
    fn android_result_records_the_resolved_skin_threshold(
        #[case] host_c: u32,
        #[case] cell_c: Option<u32>,
        #[case] skip: bool,
        #[case] expected: u32,
    ) -> anyhow::Result<()> {
        let config = pipette_readiness::ReadinessConfiguration::Android(
            pipette_readiness::AndroidReadinessConfiguration {
                skin_below_c: Some(pipette_readiness::CeilingCelsius::try_from(host_c)?),
                ..Default::default()
            },
        );
        let resolved = pipette_readiness::resolve_readiness(
            Some(Duration::from_secs(300)),
            if skip {
                pipette_readiness::ThermalGate::Skip
            } else {
                pipette_readiness::ThermalGate::Enforce
            },
        )
        .with_configuration(Some(config))
        .with_skin_below_c(
            cell_c
                .map(pipette_readiness::CeilingCelsius::try_from)
                .transpose()?,
        );
        let benchmark = BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
            parameter_batch_size: BatchSize::one(),
        });
        let flags = resolved_flags(&req(benchmark)?, resolved)?.context("gated flags")?;
        assert_eq!(
            flags.submission_value()["readiness"]["skin_below_c"],
            expected
        );
        assert_eq!(flags.submission_value()["readiness"]["skip_thermal"], skip);
        Ok(())
    }

    #[test]
    fn skin_flag_is_rejected_off_android() -> anyhow::Result<()> {
        let flags = BenchmarkFlags::PrefillThroughputLlamacppCliStockToolsGgufText {
            readiness: Some(ReadinessOverrides {
                max_wait_secs: None,
                skip_thermal: None,
                skin_below_c: Some(SkinCelsius::try_new(34)?),
            }),
        };
        let result = ensure_skin_on_android(Some(&flags));
        if pipette_readiness::platform() == pipette_readiness::ReadinessPlatform::Android {
            assert!(result.is_ok());
        } else {
            assert!(
                result
                    .err()
                    .context("non-Android host must reject skin flag")?
                    .to_string()
                    .contains("Android CLI gate")
            );
        }
        Ok(())
    }

    #[test]
    fn held_touch_flag_is_rejected_off_android() -> anyhow::Result<()> {
        ensure_held_touch_on_android(None)?;
        let result = ensure_held_touch_on_android(Some(pipette_readiness::HeldTouch::Fail));
        if pipette_readiness::platform() == pipette_readiness::ReadinessPlatform::Android {
            assert!(result.is_ok());
        } else {
            assert!(
                result
                    .err()
                    .context("non-Android host must reject the held-touch flag")?
                    .to_string()
                    .contains("Android CLI gate")
            );
        }
        Ok(())
    }

    /// Eval never calls the gate, so a readiness block would claim a wait that
    /// never happened.
    #[test]
    fn an_eval_cell_reports_no_readiness() -> anyhow::Result<()> {
        let req = req(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("ifbench".to_owned())?,
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "ifbench".into(),
            parameter_max_tokens: PositiveCount::try_new(64)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: None,
        }))?;

        let reported = resolved_flags(&req, gated())?
            .map(|f| f.submission_value())
            .and_then(|v| v.get("readiness").cloned());

        assert_eq!(reported, None);
        Ok(())
    }

    /// vlm_eval gates on llama.cpp and reports the policy; on vLLM it runs
    /// ungated, as the text eval there does, and a finished run records with
    /// no readiness rather than failing on a field its variant cannot hold.
    #[rstest]
    #[case::llamacpp(false, true)]
    #[case::uv_vllm(true, false)]
    fn a_vlm_eval_cell_reports_readiness_where_it_gates(
        #[case] vllm: bool,
        #[case] reports: bool,
    ) -> anyhow::Result<()> {
        let mut req = req(serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval",
            "benchmark_id": "vlm_eval_mmstar",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1,
            "parameter_max_tokens": 64,
            "samples": [{"id": "a", "messages": [{"role": "user", "content": "q"}]}],
        }))?)?;
        if vllm {
            req.runtime =
                DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                    "type": "uv_vllm", "server_version": "0.31.0", "build": "rocm723",
                    "python_version": "3.12",
                    "source": {"type": "absolute_preinstalled", "dir": "/venv"},
                }))?);
            req.model = DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                "type": "torch", "source": "huggingface", "org": "o", "repo_name": "r",
            }))?);
        } else {
            req.model = DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
                "type": "gguf_vision", "source": "huggingface", "org": "o", "repo_name": "r",
                "model": "m.gguf", "mmproj": "p.gguf",
            }))?);
        }
        let reported = resolved_flags(&req, gated())?
            .map(|f| f.submission_value())
            .and_then(|v| v.get("readiness").cloned());
        assert_eq!(reported.is_some(), reports, "{reported:?}");
        Ok(())
    }

    /// Nothing, rather than an error that fails the record.
    #[test]
    fn a_cell_outside_the_flag_schema_reports_nothing() -> anyhow::Result<()> {
        let apple: Runtime =
            serde_json::from_value(serde_json::json!({"type": "apple_foundation"}))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(apple),
            ..req(BenchmarkDefinition::PrefillThroughput(
                PrefillThroughputBenchmark {
                    benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
                    parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
                    parameter_batch_size: BatchSize::one(),
                },
            ))?
        };
        assert!(resolved_flags(&req, gated())?.is_none());
        Ok(())
    }

    /// Asserts the empty variant too: probing `readiness` alone would pass
    /// whether the cell reported nothing or flags without a gate.
    #[rstest]
    #[case::max_memory(max_memory_req())]
    #[case::vl_max_memory(vl_max_memory_req())]
    #[case::asr_max_memory(asr_max_memory_req())]
    fn a_cell_with_nothing_set_reports_nothing(
        #[case] req: anyhow::Result<RunRequest>,
    ) -> anyhow::Result<()> {
        let req = req?;

        let flags = BenchmarkFlags::try_from(req.benchmark_flags_ref()?)?;
        assert_eq!(
            flags.submission_value(),
            serde_json::json!({}),
            "the variant this cell resolves to must be the empty one",
        );
        assert!(resolved_flags(&req, gated())?.is_none());
        Ok(())
    }

    /// It still reports no readiness, because max-memory never gates.
    #[test]
    fn a_max_memory_cell_reports_its_authored_deadline() -> anyhow::Result<()> {
        let mut req = max_memory_req()?;
        req.benchmark_flags = Some(
            BenchmarkFlags::MaxMemoryUsageLlamacppCliStockToolsGgufText {
                bench_timeout_seconds: Some(PositiveSeconds::try_new(2400)?),
            },
        );

        let flags =
            resolved_flags(&req, gated())?.context("an authored deadline must reach the record")?;

        assert_eq!(flags.bench_timeout().map(u64::from), Some(2400));
        assert_eq!(flags.submission_value().get("readiness"), None);
        Ok(())
    }
}

#[cfg(test)]
mod spec_policy_tests {
    use std::time::Duration;

    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkId, ModelFlags, NonNegativeCount,
        PrefillThroughputBenchmark, RuntimeFlags,
    };

    use super::fixtures::{gguf_text, llamacpp, req};
    use super::*;

    fn prefill() -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(256)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn spec() -> anyhow::Result<ClientRunSpec> {
        Ok(ClientRunSpec {
            benchmark: BenchmarkId::try_new("prefill_throughput_256".to_owned())?,
            model: gguf_text()?,
            runtime: llamacpp()?,
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
        })
    }

    fn req_with(benchmark_flags: Option<BenchmarkFlags>) -> anyhow::Result<RunRequest> {
        Ok(RunRequest {
            benchmark_flags,
            ..req(prefill()?)?
        })
    }

    fn readiness(max_wait_secs: Option<u64>, skip_thermal: Option<bool>) -> Option<BenchmarkFlags> {
        Some(
            BenchmarkFlags::PrefillThroughputLlamacppCliStockToolsGgufText {
                readiness: Some(ReadinessOverrides {
                    max_wait_secs,
                    skip_thermal,
                    skin_below_c: None,
                }),
            },
        )
    }

    #[test]
    fn a_spec_with_no_flags_validates() -> anyhow::Result<()> {
        validate_spec_flags(&spec()?, &prefill()?)
    }

    /// The refusal names the group that disagreed.
    #[rstest]
    #[case::model(
        |s: &mut ClientRunSpec| s.model_flags = Some(ModelFlags::EvalGgufText { enable_thinking: Some(true), reasoning_effort: None }),
        "model_flags"
    )]
    #[case::runtime(
        |s: &mut ClientRunSpec| s.runtime_flags = Some(RuntimeFlags::PrefillThroughputLlamacppApkPipetteGgufText {
            number_gpu_layers: None, ctx_size: None, n_ubatch: None, load_mode: None,
        }),
        "runtime_flags"
    )]
    #[case::benchmark(
        |s: &mut ClientRunSpec| s.benchmark_flags = Some(BenchmarkFlags::DecodeThroughputLlamacppCliStockToolsGgufText { readiness: None }),
        "benchmark_flags"
    )]
    fn flags_for_another_cell_are_refused(
        #[case] author: fn(&mut ClientRunSpec),
        #[case] group: &str,
    ) -> anyhow::Result<()> {
        let mut spec = spec()?;
        author(&mut spec);

        let err = validate_spec_flags(&spec, &prefill()?)
            .err()
            .context("mismatched flags must be refused")?;

        assert!(err.to_string().starts_with(group), "{err}");
        Ok(())
    }

    /// An authored deadline reaches the gate; none leaves the platform default.
    #[rstest]
    #[case::unset(None, None)]
    #[case::readiness_without_a_deadline(readiness(None, Some(true)), None)]
    #[case::authored(readiness(Some(1800), None), Some(Duration::from_secs(1800)))]
    fn the_deadline_comes_from_the_cells_flags(
        #[case] flags: Option<BenchmarkFlags>,
        #[case] expected: Option<Duration>,
    ) -> anyhow::Result<()> {
        assert_eq!(readiness_max_wait(&req_with(flags)?), expected);
        Ok(())
    }

    #[rstest]
    #[case::no_flags(None, pipette_readiness::ThermalGate::Unset)]
    #[case::not_said(readiness(Some(60), None), pipette_readiness::ThermalGate::Unset)]
    #[case::waived(readiness(None, Some(true)), pipette_readiness::ThermalGate::Skip)]
    #[case::required(readiness(None, Some(false)), pipette_readiness::ThermalGate::Enforce)]
    fn the_thermal_gate_keeps_unset_apart_from_false(
        #[case] flags: Option<BenchmarkFlags>,
        #[case] expected: pipette_readiness::ThermalGate,
    ) -> anyhow::Result<()> {
        assert_eq!(readiness_thermal_gate(&req_with(flags)?), expected);
        Ok(())
    }

    #[test]
    fn a_poisoned_series_fails_instead_of_reporting() -> anyhow::Result<()> {
        let series = Mutex::new(ThermalSeries::default());
        assert!(lock(&series).is_ok());

        let _ = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _guard = series.lock();
                    std::panic::resume_unwind(Box::new("probe panicked"));
                })
                .join()
        });

        let err = lock(&series).err().context("a poisoned lock must fail")?;
        assert!(err.to_string().contains("poisoned"), "{err}");
        Ok(())
    }
}
