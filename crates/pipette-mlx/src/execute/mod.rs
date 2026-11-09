//! MLX execute: kind dispatch from a prepared [`RunRequest`].

mod decode_throughput;
mod end_to_end_latency;
mod eval;
mod max_memory_usage;
mod prefill_throughput;
mod vl_max_memory;
mod vl_spec_decode;
mod vl_throughput;

/// Why the timing cells refuse a batched entry: each sends one
/// `/v1/completions` per rep to a server launched with no slot count, so a
/// batched entry would be measured serially under a `_bs<n>` id that promises
/// a measurement under load.
const ONE_AT_A_TIME: &str = "`mlx_lm.server` here serves one request at a time";

use pipette_ops::eval_cell::EvalCell;
use pipette_ops::measurement::{self, MeasurementObserver};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_context::RunContext;
use pipette_plan_types::{
    BenchmarkDefinition, MlxMacosPipetteExtraRuntimeConfiguration, ModelType, RunRequest,
    RunResponse, RuntimeFlags, resolve_speculation,
};

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;

/// Run a prepared [`RunRequest`]; the CLI prepares it and records the
/// response. `run_context` lends the eval checkpoints and the corpus
/// benchmark's download; `client_config` is this client's configuration for
/// the runtime, applied whole to the server process.
pub fn run(
    req: &RunRequest,
    run_context: &dyn RunContext,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let RunRequest {
        runtime,
        model,
        runtime_flags,
        model_flags,
        benchmark_flags,
        benchmark: definition,
    } = req;
    // Checked only in the corpus arm: no other MLX kind reads runtime flags.
    let runtime_flag_ref = req.runtime_flags_ref();
    // Eval's SSE idle deadline. `vl_throughput` and `vl_spec_decode` read
    // `http_timeout` off `benchmark_flags`; the other cells take none.
    let idle_timeout = measurement::stream_idle_timeout(req);

    // The corpus benchmark's `speculative_enabled` is the one MLX flag, and
    // its runner reads it; every other cell has no flag cell to carry one.
    if runtime_flags.is_some()
        && definition.benchmark_type() != pipette_plan_types::BenchmarkType::VlSpecDecode
    {
        tracing::warn!(
            "runtime_flags are ignored for MLX runs (mlx_lm.server runs with fixed flags)"
        );
    }

    let venv_python = ResolvedMlxPython::try_from(&runtime.bound);

    match definition {
        BenchmarkDefinition::PrefillThroughput(benchmark) => {
            measurement::require_one_in_flight(definition, ONE_AT_A_TIME)?;
            prefill_throughput::run(
                benchmark,
                &venv_python?,
                &ResolvedMlxModel::try_from(&model.bound)?,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::DecodeThroughput(benchmark) => {
            measurement::require_one_in_flight(definition, ONE_AT_A_TIME)?;
            decode_throughput::run(
                benchmark,
                &venv_python?,
                &ResolvedMlxModel::try_from(&model.bound)?,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::EndToEndLatency(benchmark) => {
            measurement::require_one_in_flight(definition, ONE_AT_A_TIME)?;
            end_to_end_latency::run(
                benchmark,
                &venv_python?,
                &ResolvedMlxModel::try_from(&model.bound)?,
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => max_memory_usage::run(
            benchmark,
            &venv_python?,
            &ResolvedMlxModel::try_from(&model.bound)?,
            observer,
            client_config,
        ),
        BenchmarkDefinition::Eval(benchmark) => {
            // No MLX eval flag variant carries `parallel`: `mlx_lm.server`
            // serves one request at a time.
            let cell = EvalCell::new(benchmark, model_flags.as_ref(), None)?;
            let venv_python = venv_python?;
            let model_dir = ResolvedMlxModel::try_from(&model.bound)?;
            eval::run(
                benchmark,
                &cell,
                idle_timeout,
                // Opened last, as it writes the checkpoint, keyed by the whole
                // request's portable identity.
                run_context.eval_completions().open(req)?,
                &venv_python,
                &model_dir,
                client_config,
            )
        }
        BenchmarkDefinition::VlSpecDecode(benchmark) => {
            let lease = observer.outside(|| {
                pipette_vl_spec_decode_corpus::download(
                    run_context,
                    &benchmark.benchmark_id,
                    &benchmark.dataset,
                )
            })?;
            // Resolved against the model and written back, so the row records
            // the condition it was measured under rather than what the cell
            // left unset.
            let mut flags = runtime_flag_ref?;
            flags.speculative_enabled = Some(resolve_speculation(
                ModelType::of(&model.bound),
                flags.speculative_enabled,
            )?);
            vl_spec_decode::run(
                benchmark,
                benchmark_flags.as_ref(),
                RuntimeFlags::try_from(flags)?,
                &venv_python?,
                &model.bound,
                lease.corpus(),
                readiness_gate,
                observer,
                client_config,
            )
        }
        BenchmarkDefinition::VlThroughput(benchmark) => vl_throughput::run(
            benchmark,
            benchmark_flags.as_ref(),
            &venv_python?,
            &ResolvedMlxModel::try_from(&model.bound)?,
            readiness_gate,
            observer,
            client_config,
        ),
        BenchmarkDefinition::VlMaxMemory(benchmark) => vl_max_memory::run(
            benchmark,
            &venv_python?,
            &ResolvedMlxModel::try_from(&model.bound)?,
            observer,
            client_config,
        ),
        BenchmarkDefinition::AsrThroughput(_) | BenchmarkDefinition::AsrMaxMemory(_) => {
            anyhow::bail!("ASR benchmarks are not yet supported for MLX")
        }
        BenchmarkDefinition::VlmEval(_) => {
            anyhow::bail!("vlm_eval is not yet supported for MLX")
        }
        BenchmarkDefinition::AsrEval(_) => {
            anyhow::bail!("asr_eval is not yet supported for MLX")
        }
    }
}
