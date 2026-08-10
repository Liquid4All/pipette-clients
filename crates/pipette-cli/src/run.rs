//! Shared **run** path for one cell (see `docs/architecture.md`).
//!
//! - [`run_cell`] — resolve a [`ClientRunSpec`] into a [`RunRequest`], then
//!   dispatch it to an engine → the request plus what it measured (CLI and worker)
//! - [`finish_local`] — stdout summary + local record + optional submit (CLI only)
//!
//! Resolve and dispatch live together because a cell is one procedure: the
//! bound runtime [`prepare`] produces is the only thing [`dispatch_run`]
//! matches on.
//!
//! A **benchmark** is the catalog definition inside [`RunRequest`]; this module
//! does not list or sync benchmarks. Worker completes claims from that pair
//! itself (lease / success / failure).

use std::sync::{Mutex, MutexGuard};

use anyhow::Context;

use pipette_artifacts::quota::SweepPins;
use pipette_artifacts::{
    ensure_model, ensure_runtime, model_download_size, runtime_download_size, ArtifactsContext,
};

use pipette_device::detect_thermal;
use pipette_ops::readiness::RepObserver;
use pipette_ops::thermal_series::ThermalSeries;
use pipette_plan_types::benchmark::BenchmarkDefinition;
use pipette_plan_types::result::BenchmarkSubmissionPayload;
use pipette_plan_types::run::RunResponse;
use pipette_plan_types::run::{DeclaredBound, RunRequest};
use pipette_plan_types::thermal::RunThermal;
use pipette_plan_types::{
    BenchmarkFlagError, BenchmarkFlags, BenchmarkType, ClientRunSpec, ReadinessOverrides, Runtime,
};

use crate::progress::CellProgress;
use crate::results::{finished_run_payload, BenchmarkResultExtras};
use crate::workspace::PipetteWorkspace;

/// Bind, run, and describe one cell: `spec` + resolved body → request →
/// outcome → submission.
///
/// No local record and no submit: where a result goes differs between the CLI
/// and a claimed job, so that stays with the caller.
///
/// The request and the raw outcome do not leave: the request's bound halves are
/// spent once the engine returns, and everything the record and submit paths
/// need is in the payload.
pub fn run_cell(
    spec: &ClientRunSpec,
    benchmark: BenchmarkDefinition,
    artifacts_ctx: &ArtifactsContext,
    ws: &PipetteWorkspace,
) -> anyhow::Result<(BenchmarkSubmissionPayload, BenchmarkResultExtras)> {
    // Ahead of prepare: its ensures can pull an image or download weights for
    // minutes, and a stall or failure in there is only attributable if the cell
    // is already named.
    log::info!("run: {spec:?}");
    let req = prepare(spec, benchmark, artifacts_ctx, ws)?;
    // Resolved here, not inside the gate: the same value has to drive the wait
    // and the record, or the result describes a run that did not happen.
    let readiness = pipette_readiness::resolve_readiness(
        readiness_max_wait(&req),
        readiness_thermal_gate(&req),
    );
    let outcome = dispatch_run(&req, ws, readiness)?;
    finished_run_payload(&ws.identity(), &req, &outcome)
}

/// Ensure model + runtime into a [`RunRequest`] around an already-resolved
/// benchmark body.
///
/// Does **not** run a benchmark, and does **not** consult the benchmark catalog:
/// the caller resolved the body, because only the caller knows which catalog it
/// meant — a claim means the synced one, an explicit `local/<id>` on the CLI
/// means the other.
///
/// Steps:
/// 1. Check the body is the one the spec names, then reject on-device-only runtimes
/// 2. Validate flags against the resolved cell
/// 3. Check the host GPU stack the declared runtime needs
/// 4. Ensure runtime + model stores → **bound** [`Runtime`] / [`pipette_plan_types::Model`]
/// 5. Assemble the request (declared from spec, bound from stores)
fn prepare(
    spec: &ClientRunSpec,
    benchmark: BenchmarkDefinition,
    artifacts_ctx: &ArtifactsContext,
    ws: &PipetteWorkspace,
) -> anyhow::Result<RunRequest> {
    // The spec names the body and the caller resolved it, so the two have to
    // agree: the payload's id and the eval digest come from the body, so a
    // mismatch would record a cell that never ran.
    anyhow::ensure!(
        spec.benchmark.as_ref() == benchmark.benchmark_id(),
        "spec names benchmark `{}` but the resolved body is `{}`",
        spec.benchmark,
        benchmark.benchmark_id()
    );
    require_desktop_runtime(&spec.runtime)?;
    let runtimes = ws.runtimes();
    let models = ws.models();

    validate_spec_flags(spec, &benchmark)?;

    // Ahead of ensure: a broken driver should fail before an image pull or a
    // venv build, not after it.
    pipette_torch_oai::preflight::assert_gpu_ready(&spec.runtime)
        .with_context(|| format!("GPU preflight for runtime `{}`", spec.runtime))?;

    // The in-flight plan's pin set: the runtime's ensure must not evict the
    // model the model's ensure is about to resolve, and vice versa.
    let ctx = artifacts_ctx.with_pins(SweepPins::for_cell(&spec.model, &spec.runtime));
    // Sized before either starts, so the cell-level total is whole from the first
    // byte rather than growing as each artifact begins. A cache hit sizes at zero
    // without touching the network.
    let progress = CellProgress::new(&[
        runtime_download_size(&runtimes, &spec.runtime).unwrap_or(None),
        model_download_size(&ctx, &models, &spec.model).unwrap_or(None),
    ]);
    let ctx = ctx.with_progress(progress.sink());
    let bound_runtime = ensure_runtime(&ctx, &runtimes, &spec.runtime)
        .with_context(|| format!("ensuring runtime `{}`", spec.runtime))?;
    let bound_model = ensure_model(&ctx, &models, &spec.model)
        .with_context(|| format!("ensuring model `{}`", spec.model))?;

    Ok(RunRequest {
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
    })
}

fn validate_spec_flags(
    spec: &ClientRunSpec,
    benchmark: &BenchmarkDefinition,
) -> anyhow::Result<()> {
    let bt = benchmark.benchmark_type();
    if let Some(f) = &spec.model_flags {
        if !f.matches(bt, &spec.model) {
            anyhow::bail!(
                "model_flags do not match resolved benchmark ({bt:?}) / model ({})",
                spec.model
            );
        }
    }
    if let Some(f) = &spec.runtime_flags {
        if !f.matches(bt, &spec.runtime, &spec.model) {
            anyhow::bail!(
                "runtime_flags do not match resolved cell (benchmark={bt:?}, runtime={}, model={})",
                spec.runtime,
                spec.model
            );
        }
    }
    if let Some(f) = &spec.benchmark_flags {
        if !f.matches(bt, &spec.runtime, &spec.model) {
            anyhow::bail!(
                "benchmark_flags do not match resolved cell (benchmark={bt:?}, runtime={}, model={})",
                spec.runtime,
                spec.model
            );
        }
    }
    Ok(())
}

/// Desktop / CLI-runnable runtimes only (shared store + engine seams).
fn require_desktop_runtime(runtime: &Runtime) -> anyhow::Result<()> {
    match runtime {
        Runtime::LlamacppCliStockTools(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvVllm(_)
        | Runtime::UvSglang(_)
        | Runtime::UvOpenvino(_) => Ok(()),
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::AppleFoundation(_) => anyhow::bail!(
            "runtime `{runtime}` is not a desktop CLI runtime; \
             allowed: llamacpp_cli_stock_tools, mlx_macos_pipette, \
             docker_vllm, docker_sglang, uv_vllm, uv_sglang, uv_openvino"
        ),
    }
}

/// The readiness deadline for this cell: the plan's `benchmark_flags` override
/// if it set one, else the per-platform default the readiness crate picks.
fn readiness_max_wait(req: &RunRequest) -> Option<std::time::Duration> {
    req.benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::readiness)
        .and_then(|r| r.max_wait_secs)
        .map(std::time::Duration::from_secs)
}

/// What this cell says about the thermal criterion, preserving the difference
/// between "waive it", "require it", and "didn't say".
///
/// The three-way distinction is the point: an absent knob defers to
/// `PIPETTE_READINESS_SKIP_THERMAL` (so a fleet-wide waiver reaches this cell),
/// while an authored `skip_thermal = false` overrides it — a cell that said to
/// gate should not be silently ungated by a stale export in the shell that
/// launched the runner.
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

/// A poisoned series is a panic that already happened under the lock: the
/// readings are no longer trustworthy, and the hooks can say so rather than
/// papering over it, so the run fails instead of reporting a partial series.
fn lock(series: &Mutex<ThermalSeries>) -> anyhow::Result<MutexGuard<'_, ThermalSeries>> {
    series
        .lock()
        .map_err(|_| anyhow::anyhow!("the thermal series lock was poisoned by an earlier panic"))
}

/// The cell's benchmark flags as the run resolved them: the plan's entry with
/// the readiness block filled in from what actually applied.
///
/// The authored block cannot describe the run — `skip_thermal: None` on a host
/// where the environment waived the criterion, `max_wait_secs: None` where a
/// platform default supplied one.
///
/// A cell whose variant has no readiness field keeps `None`, and the conversion
/// back would refuse anything else — correct, because those cells do not gate.
fn resolved_flags(
    req: &RunRequest,
    readiness: pipette_readiness::ResolvedReadiness,
) -> anyhow::Result<Option<BenchmarkFlags>> {
    let mut r = req.benchmark_flags_ref()?;
    // Which cells gate is also encoded in whether their flag variant carries a
    // readiness field — two spellings of one fact. Exhaustive rather than
    // `matches!` so a new benchmark type has to answer here too, instead of
    // silently reporting no readiness for a cell that waited.
    let gates = match req.benchmark.benchmark_type() {
        BenchmarkType::PrefillThroughput
        | BenchmarkType::DecodeThroughput
        | BenchmarkType::EndToEndLatency
        | BenchmarkType::VlThroughput => true,
        BenchmarkType::Eval | BenchmarkType::MaxMemoryUsage => false,
    };
    if gates {
        r.readiness = Some(ReadinessOverrides {
            max_wait_secs: Some(readiness.max_wait_secs()),
            skip_thermal: Some(readiness.skip_thermal()),
        });
    }
    match BenchmarkFlags::try_from(r) {
        Ok(flags) => Ok(Some(flags)),
        // A cell the flag schema does not model (max-memory, Apple, mobile)
        // has nothing to report — not a failure.
        Err(BenchmarkFlagError::NoSuchCombination { .. }) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Takes the resolved policy rather than resolving it: resolving twice could
/// straddle an environment change, and the gate and the record must not
/// disagree about how this cell was measured.
fn dispatch_run(
    req: &RunRequest,
    ws: &PipetteWorkspace,
    readiness: pipette_readiness::ResolvedReadiness,
) -> anyhow::Result<RunResponse> {
    let readiness_gate = || -> anyhow::Result<()> {
        pipette_readiness::wait_until_ready(&readiness)?;
        Ok(())
    };
    // The single pending slot assumes an engine measures its reps one at a
    // time; concurrent reps would need a key per rep, not just a lock. Probe
    // before taking it — `detect_thermal` shells out on some platforms.
    let series = Mutex::new(ThermalSeries::default());
    let observer = RepObserver::new(
        || {
            let reading = detect_thermal();
            lock(&series)?.start(reading)
        },
        || {
            let reading = detect_thermal();
            lock(&series)?.finish(reading)
        },
    );
    // Handing over the object rather than a field list: a field added to
    // `RunRequest` is logged without anyone remembering to, and secrets stay
    // the types' own business (`AuthToken` redacts itself).
    log::info!("dispatch: {req:?}");
    let mut response = match (&req.runtime.declared, &req.benchmark) {
        (Runtime::LlamacppCliStockTools(_), _) => {
            pipette_llamacpp::run(req, &ws.eval_completions(), &readiness_gate, &observer)
        }
        #[cfg(target_os = "macos")]
        (Runtime::MlxMacosPipette(_), _) => {
            pipette_mlx::run(req, &ws.eval_completions(), &readiness_gate, &observer)
        }
        #[cfg(not(target_os = "macos"))]
        (Runtime::MlxMacosPipette(_), _) => anyhow::bail!("the MLX runtime runs on macOS only"),

        (
            Runtime::DockerVllm(_)
            | Runtime::DockerSglang(_)
            | Runtime::UvVllm(_)
            | Runtime::UvSglang(_),
            _,
        ) => pipette_torch_oai::run(req, &ws.eval_completions(), &readiness_gate, &observer),

        (Runtime::UvOpenvino(_), _) => pipette_openvino::run(
            req,
            &ws.eval_completions(),
            &readiness_gate,
            &observer,
            &ws.compile_cache(&req.runtime.declared)?,
        ),

        (
            Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::AppleFoundation(_),
            _,
        ) => anyhow::bail!(
            "runtime `{}` is not a desktop CLI runtime",
            req.runtime.declared
        ),
    }?;

    // The engines mark the reps; the readings are this side's, so attach them
    // to what goes back rather than making every engine carry them.
    response.thermal = RunThermal::from_pairs(lock(&series)?.take()?);
    response.benchmark_flags = resolved_flags(req, readiness)?;
    Ok(response)
}

#[cfg(test)]
mod resolved_flags_tests {
    use std::time::Duration;

    use pipette_plan_types::benchmark::{
        BenchmarkDefinition, EvalBenchmark, MaxMemoryUsage, PrefillThroughput,
    };
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{Model, Runtime};

    use super::*;

    /// A llama.cpp + GGUF cell running `benchmark`, with no authored flags —
    /// so what the record carries can only have come from resolution.
    fn req(benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        let model: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_text", "source": "huggingface",
            "org": "o", "repo_name": "r", "path": "m-Q4_0.gguf"
        }))?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "llamacpp_cli_stock_tools", "source": "github_release",
            "version": "b5000", "flavor": "macos-arm64"
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    fn gated() -> pipette_readiness::ResolvedReadiness {
        pipette_readiness::resolve_readiness(
            Some(Duration::from_secs(300)),
            pipette_readiness::ThermalGate::Enforce,
        )
    }

    /// A cell that gates reports what it gated on, even though it authored
    /// nothing — the resolved policy is the whole point of the field.
    #[test]
    fn a_gated_cell_reports_the_policy_it_applied() -> anyhow::Result<()> {
        let req = req(BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_256".into(),
            parameter_prefill_tokens: 256,
        }))?;

        let flags = resolved_flags(&req, gated())?.context("a gated cell must report flags")?;

        assert_eq!(
            flags.submission_value().get("readiness"),
            Some(&serde_json::json!({"max_wait_secs": 300, "skip_thermal": false}))
        );
        Ok(())
    }

    /// Eval never calls the gate, so reporting a readiness block would claim a
    /// wait that never happened. Its flag variant has no field to hold one,
    /// and this is what keeps the two facts agreeing.
    #[test]
    fn an_eval_cell_reports_no_readiness() -> anyhow::Result<()> {
        let req = req(BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "ifbench".into(),
            parameter_eval_id: "ifbench".into(),
            parameter_dataset_name: "ifbench".into(),
            parameter_max_tokens: 64,
            parameter_mcq_choices: None,
            samples: None,
        }))?;

        let reported = resolved_flags(&req, gated())?
            .map(|f| f.submission_value())
            .and_then(|v| v.get("readiness").cloned());

        assert_eq!(reported, None);
        Ok(())
    }

    /// Max-memory has no flag variant at all. That is nothing to report, not a
    /// failure — the run is otherwise fine.
    #[test]
    fn a_cell_with_no_flag_variant_reports_nothing() -> anyhow::Result<()> {
        let req = req(BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsage {
            benchmark_id: "max_memory_usage".into(),
            parameter_prefill_tokens: 256,
        }))?;

        assert!(resolved_flags(&req, gated())?.is_none());
        Ok(())
    }
}
