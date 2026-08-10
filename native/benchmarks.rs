//! Benchmark measurement kernel shared by the native iOS and Android clients.
//!
//! Implements the same 5 benchmark types as ee-cli/src/execute/mod.rs but
//! runs inference in-process via the llama.cpp C API instead of spawning
//! subprocesses.
//!
//! This file is not a crate of its own: it is textually included by
//! `pipette-android` via `#[path = ".../native/benchmarks.rs"]` and resolves
//! `crate::error::PipetteError`, `crate::llama`, `crate::ModelHandle`, and the
//! progress/readiness callback traits against that crate's modules. (The iOS
//! client shared this kernel too, until iOS moved to a native-Swift llama.cpp
//! engine.) Keep it free of
//! platform-specific `cfg` so the two compilations stay in lockstep.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use pipette_ops::measurement::{self, Rep, Stats};
use pipette_ops::readiness::RepObserver;
use pipette_ops::thermal_series::ThermalSeries;
use pipette_plan_types::result::{BenchmarkEvalCompletion, BenchmarkEvalCompletionStopReason};
use pipette_plan_types::thermal::{ThermalReading, ThermalTelemetry};

use crate::error::PipetteError;
use crate::llama;
use crate::{ProgressCallback, ReadinessCallback, ReadinessOutcome, ThermalSampler};

// ---------------------------------------------------------------------------
// Benchmark definition (subset matching ee-cli types)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BenchmarkDef {
    benchmark_id: String,
    benchmark_type: String,
    #[serde(default)]
    parameter_prefill_tokens: Option<u32>,
    #[serde(default)]
    parameter_decode_tokens: Option<u32>,
    #[serde(default)]
    parameter_max_tokens: Option<u32>,
    #[serde(default)]
    parameter_mcq_choices: Option<Vec<String>>,
    #[serde(default)]
    samples: Option<Vec<Value>>,
    // VL throughput parameters
    #[serde(default)]
    parameter_image_width: Option<u32>,
    #[serde(default)]
    parameter_image_height: Option<u32>,
    #[serde(default)]
    parameter_text_tokens: Option<u32>,
}

/// Whether an error serving one eval sample should end the whole cell, or be
/// recorded against that sample so the run continues.
///
/// The CLI records and continues unconditionally, because llama.cpp is known to
/// crash on particular prompts and a whole-cell abort loses every sample that
/// already succeeded. It can afford that: a dead llama-server gets restarted
/// before the next sample. This kernel runs in-process with no equivalent
/// recovery, so an error that means *the model is gone* would otherwise produce
/// one identical failure per remaining sample against a handle that can't work.
///
/// So split by kind, along the same line the CLI draws between a child that
/// exited and a transport error against a live server.
fn aborts_cell(err: &PipetteError) -> bool {
    matches!(
        err,
        // Not a failure at all: the user asked to stop, and a cancelled cell
        // must not report a warehouse full of failed samples.
        PipetteError::Cancelled { .. }
            // The model handle is unusable from here on.
            | PipetteError::OutOfMemory { .. }
            | PipetteError::ModelLoad { .. }
    )
}

// ---------------------------------------------------------------------------
// Context-size requirements
// ---------------------------------------------------------------------------

/// Eval prompt budget — enough to fit most prompts plus the longest plausible
/// completion. Mirrors the constant used in
/// `crates/pipette-llamacpp/src/execute/llama_server_common.rs`'s
/// `default_ctx_size`.
const EVAL_PROMPT_BUDGET: u32 = 8192;

/// Minimum llama.cpp context window needed to run `def`. Returns `None` for
/// types with no derivable lower bound. Validating up front lets us produce
/// a readable error instead of an opaque `llama_decode` code 1 mid-run.
fn required_ctx_size(def: &BenchmarkDef) -> Option<u32> {
    let prefill = def.parameter_prefill_tokens.unwrap_or(0);
    let decode = def.parameter_decode_tokens.unwrap_or(0);
    let max_tokens = def.parameter_max_tokens.unwrap_or(0);
    match def.benchmark_type.as_str() {
        "prefill_throughput" => Some(prefill),
        "max_memory_usage" => Some(prefill.saturating_add(1)),
        "decode_throughput" | "end_to_end_latency" => Some(prefill.saturating_add(decode)),
        "vl_throughput" => {
            // ~1 token per 14x14 patch of the input image.
            let w = def.parameter_image_width.unwrap_or(0);
            let h = def.parameter_image_height.unwrap_or(0);
            let image_tokens = (w / 14).saturating_mul(h / 14);
            let text = def.parameter_text_tokens.unwrap_or(0);
            Some(image_tokens.saturating_add(text).saturating_add(decode))
        }
        "eval" => Some(EVAL_PROMPT_BUDGET.saturating_add(max_tokens)),
        _ => None,
    }
}

fn check_ctx_size(def: &BenchmarkDef, context_size: u32) -> Result<(), PipetteError> {
    let Some(required) = required_ctx_size(def) else {
        return Ok(());
    };
    if context_size >= required {
        return Ok(());
    }
    Err(PipetteError::Benchmark {
        msg: format!(
            "context_size {} is too small for benchmark '{}' (needs at least {}; \
             prefill={}, decode={}, max_tokens={}). \
             Increase Context Size in the Job settings.",
            context_size,
            def.benchmark_id,
            required,
            def.parameter_prefill_tokens.unwrap_or(0),
            def.parameter_decode_tokens.unwrap_or(0),
            def.parameter_max_tokens.unwrap_or(0),
        ),
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Stamp the runtime's compute-thread count onto a benchmark result for
/// reproducibility (it's device- and cpuset-dependent). When a loaded `model` is
/// given, use its load-time count (authoritative — the cpuset can change between
/// load and this run); otherwise recompute (the fresh-load path, where load
/// immediately precedes the run). No-op if the result isn't a JSON object or the
/// count is unavailable (host).
fn with_thread_count(mut result: Value, model: Option<&crate::ModelHandle>) -> Value {
    let count = match model {
        Some(m) => llama::n_threads(m),
        None => llama::default_thread_count(),
    };
    if let (Value::Object(ref mut map), Some(n)) = (&mut result, count) {
        map.insert("runtime_thread_count".to_string(), json!(n));
    }
    result
}

// FFI boundary: params mirror the uniffi `run_benchmark` surface (model + run
// config + progress/readiness callbacks); grouping into a struct would churn the
// generated Swift API for no gain.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    benchmark_json: &str,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    mmproj_path: Option<&str>,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<String, PipetteError> {
    let def: BenchmarkDef =
        serde_json::from_str(benchmark_json).map_err(|e| PipetteError::Json {
            msg: format!("failed to parse benchmark definition: {e}"),
        })?;

    // Fail before model load — saves the user the GGUF read + Metal compile.
    check_ctx_size(&def, context_size)?;

    // Report start via progress callback (allows early cancellation check)
    if let Some(ref cb) = progress {
        if !cb.on_progress(0, 1, "Starting benchmark...".to_string()) {
            return Err(PipetteError::Cancelled {
                msg: "cancelled before start".to_string(),
            });
        }
    }

    let result = match def.benchmark_type.as_str() {
        "prefill_throughput" => run_prefill_throughput(
            &def,
            model_path,
            n_gpu_layers,
            context_size,
            n_ubatch,
            progress,
            readiness,
            thermal,
        ),
        "decode_throughput" => run_decode_throughput(
            &def,
            model_path,
            n_gpu_layers,
            context_size,
            n_ubatch,
            progress,
            readiness,
            thermal,
        ),
        "end_to_end_latency" => run_end_to_end_latency(
            &def,
            model_path,
            n_gpu_layers,
            context_size,
            n_ubatch,
            progress,
            readiness,
            thermal,
        ),
        "max_memory_usage" => {
            run_max_memory_usage(&def, model_path, n_gpu_layers, context_size, n_ubatch)
        }
        "eval" => run_eval(
            &def,
            model_path,
            n_gpu_layers,
            context_size,
            n_ubatch,
            progress,
            readiness,
        ),
        "vl_throughput" => {
            let mmproj = mmproj_path.ok_or_else(|| PipetteError::Benchmark {
                msg: "vl_throughput benchmark requires mmproj_path".to_string(),
            })?;
            run_vl_throughput(
                &def,
                model_path,
                n_gpu_layers,
                context_size,
                n_ubatch,
                mmproj,
                progress,
                readiness,
                thermal,
            )
        }
        _ => Err(PipetteError::Benchmark {
            msg: format!("unsupported benchmark type: {}", def.benchmark_type),
        }),
    }?;

    serde_json::to_string(&with_thread_count(result, None))
        .map_err(|e| PipetteError::Json { msg: e.to_string() })
}

/// Run a benchmark against a model that is already loaded. Used by the job
/// runner to avoid reloading the same model between consecutive cells.
///
/// `max_memory_usage` is intentionally unsupported here. Callers must fall
/// back to `run_benchmark` (which loads fresh) so the benchmark observes
/// model load plus inference in one isolated run.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark_on_model(
    benchmark_json: &str,
    model: &crate::ModelHandle,
    n_gpu_layers: u32,
    context_size: u32,
    mmproj_path: Option<&str>,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<String, PipetteError> {
    let def: BenchmarkDef =
        serde_json::from_str(benchmark_json).map_err(|e| PipetteError::Json {
            msg: format!("failed to parse benchmark definition: {e}"),
        })?;

    // `context_size` here is the value `load_model` was called with — the
    // actual KV-cache capacity available for this run.
    check_ctx_size(&def, context_size)?;

    if let Some(ref cb) = progress {
        if !cb.on_progress(0, 1, "Starting benchmark...".to_string()) {
            return Err(PipetteError::Cancelled {
                msg: "cancelled before start".to_string(),
            });
        }
    }

    // Every benchmark starts from a clean KV cache + sampler so state from
    // the previous cell cannot leak into this one's measurements.
    llama::reset_context(model)?;
    llama::sampler_reset(model);

    let result = match def.benchmark_type.as_str() {
        "prefill_throughput" => {
            run_prefill_throughput_impl(&def, model, context_size, progress, readiness, thermal)
        }
        "decode_throughput" => {
            run_decode_throughput_impl(&def, model, context_size, progress, readiness, thermal)
        }
        "end_to_end_latency" => {
            run_end_to_end_latency_impl(&def, model, context_size, progress, readiness, thermal)
        }
        "eval" => run_eval_impl(&def, model, progress, readiness),
        "vl_throughput" => {
            let mmproj = mmproj_path.ok_or_else(|| PipetteError::Benchmark {
                msg: "vl_throughput benchmark requires mmproj_path".to_string(),
            })?;
            run_vl_throughput_impl(
                &def,
                model,
                mmproj,
                n_gpu_layers,
                progress,
                readiness,
                thermal,
            )
        }
        "max_memory_usage" => Err(PipetteError::Benchmark {
            msg: "max_memory_usage requires a fresh model load; call run_benchmark instead"
                .to_string(),
        }),
        _ => Err(PipetteError::Benchmark {
            msg: format!("unsupported benchmark type: {}", def.benchmark_type),
        }),
    }?;

    serde_json::to_string(&with_thread_count(result, Some(model)))
        .map_err(|e| PipetteError::Json { msg: e.to_string() })
}

// ---------------------------------------------------------------------------
// Helper: load model, run closure, always unload even on error
// ---------------------------------------------------------------------------

fn with_model<F, T>(
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    f: F,
) -> Result<T, PipetteError>
where
    F: FnOnce(&crate::ModelHandle) -> Result<T, PipetteError>,
{
    // Fresh-load path: honor the configurable prefill micro-batch (n_ubatch=0 → 512).
    let model = llama::load_model(model_path, n_gpu_layers, context_size, n_ubatch)?;
    let result = f(&model);
    // Always unload the model, even if `f` failed, to avoid leaking multi-GB
    // C allocations on iOS where the OS will kill the app under memory pressure.
    let _ = llama::unload_model(&model);
    result
}

// ---------------------------------------------------------------------------
// Benchmark implementations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_prefill_throughput(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        run_prefill_throughput_impl(def, model, context_size, progress, readiness, thermal)
    })
}

// The measured-rep count for the perf benchmarks (prefill_throughput,
// decode_throughput, end_to_end_latency) is `pipette_ops::measurement::REPS`
// (5, matching llama-bench's default `--repetitions 5`), so this kernel and the
// desktop engines share one source of truth instead of a local constant. Each
// still runs one untimed warm-up pass first, whose shape comes from the same
// place for the same reason.
//
// On phones the warm-up staying light is load-bearing beyond the usual
// argument: it is untimed and ungated, so running the full sequence would bake
// heat in and burn time immediately before the cooldown-gated measurements.

/// Wait until the device is thermally ready for a measured rep, mirroring the
/// reference client's `wait_until_ready()`. Called before every measured rep;
/// the gate before a cell's first rep also serves as the between-cell cooldown.
/// A `None` callback is a no-op. Cancellation aborts the cell; a timeout or
/// unreadable sensor fails it, so we never measure under unknown thermal
/// conditions.
fn readiness_gate(readiness: &Option<Arc<dyn ReadinessCallback>>) -> Result<(), PipetteError> {
    let Some(cb) = readiness else { return Ok(()) };
    match cb.wait_until_ready() {
        ReadinessOutcome::Ready => Ok(()),
        ReadinessOutcome::Cancelled => Err(PipetteError::Cancelled {
            msg: "cancelled during thermal cooldown".to_string(),
        }),
        ReadinessOutcome::TimedOut { observed } => Err(PipetteError::Readiness {
            msg: format!("device failed to cool within the time budget: last seen {observed}"),
        }),
    }
}

/// Sample the current thermal telemetry into a per-rep [`ThermalReading`], or
/// `None` when no sampler is wired (so the caller pushes nothing). A wired
/// sampler always yields a reading — each field is itself `None` when that family
/// is unavailable (unsupported API, cold start, missing `DUMP` grant, or a
/// sampling failure): a reading is pushed for *every* measured rep so the series
/// stays index-aligned, and [`ThermalTelemetry::from_series`] reduces each family
/// (all-or-nothing for the scalar `headroom`/`status`, best-effort for the
/// `sensors` list — see `scalar_series` / `sensor_series`).
///
/// Sampling never fails the run — a `ThermalSampler` that errors reports `None`.
fn sample_thermal(thermal: &Option<Arc<dyn ThermalSampler>>) -> Option<ThermalReading> {
    let cb = thermal.as_ref()?;
    Some(ThermalReading {
        android_thermal_headroom: cb.headroom(),
        android_thermal_status: cb.status(),
        android_thermal_sensors: cb.sensors(),
        ..Default::default()
    })
}

/// Merge the populated per-rep thermal series (headroom, status, sensors) into a
/// benchmark's result object. [`ThermalTelemetry`] skips `None` families, so an
/// all-absent telemetry (no sampler, or a family unavailable on some rep)
/// contributes nothing. The fields land at the top level, siblings of the
/// benchmark metrics, matching the flattened submission schema (e.g.
/// `device_android_thermal_headroom_before/after`).
/// What a measured cell produced: the reduction, and the per-rep thermal
/// readings taken around it.
struct MeasuredCell {
    stats: Stats,
    thermal: Vec<(ThermalReading, ThermalReading)>,
}

/// Recover the typed error a hook reported. The harness's hooks are typed on
/// `anyhow` because they are shared with the desktop engines, so this kernel's
/// errors travel wrapped; `downcast` returns them intact, which is what keeps
/// `Cancelled` distinguishable from `Readiness` at the UniFFI boundary.
fn to_pipette_error(err: anyhow::Error) -> PipetteError {
    err.downcast::<PipetteError>()
        .unwrap_or_else(|err| PipetteError::Benchmark {
            msg: format!("{err:#}"),
        })
}

/// Run a cell's measured reps on the fleet-shared harness, so this kernel's rep
/// count, gating, per-rep bracketing and statistic are the ones every other
/// client uses rather than a parallel implementation of them.
///
/// `prepare` is the rep's untimed setup — the KV reset, and for decode the
/// re-prefill that re-establishes KV depth. It runs *after* the rep's thermal
/// reading, so however expensive it is (decode's re-prefill is a full prefill,
/// not a teardown) it cannot move what that reading describes: the device as
/// the readiness gate cleared. `work` is the timed region alone. `sample`
/// checks the rep and reads off its metric, untimed and before the next rep's
/// gate, so a bad rep stops the cell where it happened.
///
/// The two readings are paired by [`ThermalSeries`], so a rep that contributes
/// one and not the other fails the cell instead of shifting every later
/// reading onto the wrong iteration.
fn measure_cell<T>(
    label: &str,
    progress: &Option<Arc<dyn ProgressCallback>>,
    readiness: &Option<Arc<dyn ReadinessCallback>>,
    thermal: &Option<Arc<dyn ThermalSampler>>,
    prepare: impl Fn() -> Result<(), PipetteError>,
    mut work: impl FnMut() -> Result<T, PipetteError>,
    mut sample: impl FnMut(&Rep<T>) -> Result<f64, PipetteError>,
) -> Result<MeasuredCell, PipetteError> {
    // `RefCell`, not a lock: the harness drives the hooks one at a time, so the
    // borrows never overlap and there is no cross-thread access to guard.
    let series = RefCell::new(ThermalSeries::default());
    let gate = || readiness_gate(readiness).map_err(anyhow::Error::new);
    let observer = RepObserver::new(
        || match sample_thermal(thermal) {
            Some(reading) => series.borrow_mut().start(reading),
            None => Ok(()),
        },
        || match sample_thermal(thermal) {
            Some(reading) => series.borrow_mut().finish(reading),
            None => Ok(()),
        },
    );

    let measured = measurement::run(
        label,
        &gate,
        &observer,
        |_| prepare().map_err(anyhow::Error::new),
        |_| work().map_err(anyhow::Error::new),
        |idx, rep| {
            let metric = sample(rep).map_err(anyhow::Error::new)?;
            report_measurement(progress, idx, measurement::REPS).map_err(anyhow::Error::new)?;
            Ok(metric)
        },
    )
    .map_err(to_pipette_error)?;

    let thermal = series.borrow_mut().take().map_err(to_pipette_error)?;
    Ok(MeasuredCell {
        stats: measured.stats(),
        thermal,
    })
}

/// Merge a paired per-rep series, as [`measure_cell`] collects it.
fn merge_thermal_reps(result: &mut Value, reps: &[(ThermalReading, ThermalReading)]) {
    let (before, after): (Vec<_>, Vec<_>) = reps.iter().cloned().unzip();
    merge_thermal(result, &before, &after);
}

fn merge_thermal(result: &mut Value, before: &[ThermalReading], after: &[ThermalReading]) {
    let telemetry = ThermalTelemetry::from_series(before, after);
    // Telemetry must never fail a benchmark (same invariant JavaThermalSampler
    // upholds: a sampling failure yields `None`, never a cancellation). On the
    // practically-impossible serialize error, log and drop the telemetry rather
    // than discard an otherwise-complete measurement result.
    let fields = match serde_json::to_value(&telemetry) {
        Ok(fields) => fields,
        Err(e) => {
            log::warn!("dropping thermal telemetry: failed to serialize: {e}");
            return;
        }
    };
    if let (Value::Object(dst), Value::Object(src)) = (result, fields) {
        dst.extend(src);
    }
}

/// Report a measured rep to the UI over the progress channel as `"Measurement n/total"`.
/// A `None` callback is a no-op; a `false` return (cancellation requested) aborts the cell.
/// Only the throughput/latency benchmarks' *measured* reps call this — the untimed warm-up
/// pass runs before the measurement harness, so it is never mislabeled as a measurement.
fn report_measurement(
    progress: &Option<Arc<dyn ProgressCallback>>,
    idx: usize,
    total: usize,
) -> Result<(), PipetteError> {
    let Some(cb) = progress else { return Ok(()) };
    let n = idx + 1;
    if !cb.on_progress(n as u32, total as u32, format!("Measurement {n}/{total}")) {
        return Err(PipetteError::Cancelled {
            msg: format!("cancelled at measurement {n}/{total}"),
        });
    }
    Ok(())
}

/// Build a measurement prompt of `target_tokens` (capped to `context_size`)
/// from the shared `pipette_ops::prompt_seed` corpus, tokenized against this
/// model — so mobile measures the same corpus as the llama.cpp / MLX CLIs
/// rather than a degenerate `"hello"` repeat.
///
/// `add_special` must match the CLI path for the benchmark: e2e/prefill/decode
/// count BOS (the e2e CLI's `/tokenize` uses `add_special=true` to mirror
/// `/completion`'s reported `prompt_n`); `vl_throughput` uses the server
/// default (false).
fn seed_prompt_tokens(
    model: &crate::ModelHandle,
    target_tokens: u32,
    context_size: u32,
    add_special: bool,
) -> Result<Vec<i32>, PipetteError> {
    seed_prompt_tokens_with(target_tokens, context_size, add_special, |text, special| {
        llama::tokenize(model, text, special)
    })
}

/// Build the exact-size prompt **text** (untimed) against this model's
/// tokenizer, without tokenizing it into ids. The e2e benchmark uses this so the
/// final `llama::tokenize` of the text can run *inside* its timed window — see
/// [`seed_prompt_text_with`] for why.
fn seed_prompt_text(
    model: &crate::ModelHandle,
    target_tokens: u32,
    context_size: u32,
    add_special: bool,
) -> Result<String, PipetteError> {
    seed_prompt_text_with(target_tokens, context_size, add_special, |text, special| {
        llama::tokenize(model, text, special)
    })
}

/// Pure core of [`seed_prompt_text`]: build a prompt string that tokenizes to
/// exactly `min(target_tokens, context_size)` tokens under `tokenize`, without
/// producing the ids. For a target >= 1 `build_prompt_text` lands on the target
/// exactly (or errors), so callers that tokenize this text with the same
/// `add_special` get exactly the target count and need no truncation. The sole
/// exception is target == 0: `build_prompt_text` returns "" without consulting the
/// tokenizer, but an `add_special` tokenize still prepends a BOS, so the text then
/// tokenizes to 1, not 0.
///
/// The two seeding paths deliberately diverge there. [`seed_prompt_tokens_with`]
/// *keeps* that BOS: its callers (prefill / decode) hand the ids straight to
/// `llama::prefill`, and an empty batch is rejected outright by
/// `ee_llama_decode_batch`. The e2e runner instead tokenizes this *text* inside its
/// timed window and validates the resulting count against `prefill_tokens`, so a
/// 0-token prefill still fails that guard (1 != 0), matching the CLIs, which
/// likewise fail their `prompt_n == prompt_tokens` check when `prompt_tokens == 0`.
/// Parameterized over the tokenizer so it can be unit-tested without a model.
fn seed_prompt_text_with<F>(
    target_tokens: u32,
    context_size: u32,
    add_special: bool,
    mut tokenize: F,
) -> Result<String, PipetteError>
where
    F: FnMut(&str, bool) -> Result<Vec<i32>, PipetteError>,
{
    let target = std::cmp::min(target_tokens, context_size);
    pipette_ops::prompt_seed::build_prompt_text(target, |t| {
        tokenize(t, add_special)
            .map(|toks| toks.len())
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    })
    .map_err(|e| PipetteError::Benchmark {
        msg: format!("failed to build {target}-token prompt: {e}"),
    })
}

/// Pure core of [`seed_prompt_tokens`], parameterized over the tokenizer so it
/// can be unit-tested without a model.
fn seed_prompt_tokens_with<F>(
    target_tokens: u32,
    context_size: u32,
    add_special: bool,
    mut tokenize: F,
) -> Result<Vec<i32>, PipetteError>
where
    F: FnMut(&str, bool) -> Result<Vec<i32>, PipetteError>,
{
    let target = std::cmp::min(target_tokens, context_size);
    let text = seed_prompt_text_with(target_tokens, context_size, add_special, &mut tokenize)?;
    let mut tokens = tokenize(&text, add_special)?;
    // For target >= 1 `build_prompt_text` already lands on exactly `target` under the
    // same `add_special`, so this truncate is a no-op there (which is why the e2e text
    // path skips it).
    //
    // target == 0 is the generation-only shape (decode_throughput_0_32). Keep the BOS an
    // `add_special` tokenize prepends instead of clipping to empty: llama.cpp cannot decode
    // from an empty KV cache, and `ee_llama_decode_batch` rejects a zero-length batch with
    // -1 ("prefill failed with code -1"). llama-bench has the same constraint and handles it
    // the same way: it skips its prompt phase entirely when `n_prompt == 0` and opens
    // generation by decoding BOS (`test_gen`, tools/llama-bench/llama-bench.cpp). Keeping
    // the single BOS is what makes the mobile path match the CLI rather than diverge from it.
    tokens.truncate(std::cmp::max(
        target as usize,
        usize::from(!tokens.is_empty()),
    ));
    Ok(tokens)
}

fn run_prefill_throughput_impl(
    def: &BenchmarkDef,
    model: &crate::ModelHandle,
    context_size: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    let prefill_tokens = def.parameter_prefill_tokens.unwrap_or(512);
    let benchmark_id = def.benchmark_id.clone();

    let prompt_tokens = seed_prompt_tokens(model, prefill_tokens, context_size, true)?;
    let tokens = prompt_tokens.as_slice();

    // Warm-up at the measured shape, untimed and ungated.
    llama::reset_context(model)?;
    llama::sampler_reset(model);
    llama::prefill(model, tokens)?;
    log::info!("prefill_throughput: warm-up run ({}p)", tokens.len());

    let measured = measure_cell(
        "prefill_throughput",
        &progress,
        &readiness,
        &thermal,
        || {
            llama::reset_context(model)?;
            llama::sampler_reset(model);
            Ok(())
        },
        || llama::prefill(model, tokens),
        |rep| Ok(rep.elapsed_ms()),
    )?;
    let (mean_ms, stddev_ms) = (measured.stats.mean_ms, measured.stats.stddev_ms);
    log::info!("prefill_throughput: prefill_time_ms={mean_ms:.3} stddev={stddev_ms:.3}");

    let mut result = json!({
        "benchmark_id": benchmark_id,
        "prefill_time_ms": mean_ms,
        "prefill_time_ms_stddev": stddev_ms,
    });
    merge_thermal_reps(&mut result, &measured.thermal);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_decode_throughput(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        run_decode_throughput_impl(def, model, context_size, progress, readiness, thermal)
    })
}

fn run_decode_throughput_impl(
    def: &BenchmarkDef,
    model: &crate::ModelHandle,
    context_size: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    let prefill_tokens = def.parameter_prefill_tokens.unwrap_or(512);
    let decode_tokens = def.parameter_decode_tokens.unwrap_or(128);
    let benchmark_id = def.benchmark_id.clone();

    let prompt_tokens = seed_prompt_tokens(model, prefill_tokens, context_size, true)?;
    let tokens = prompt_tokens.as_slice();

    // Warm-up at the measured shape, untimed and ungated.
    llama::reset_context(model)?;
    llama::sampler_reset(model);
    llama::prefill(model, tokens)?;
    llama::decode_n_greedy_ignore_eog(model, decode_tokens)?;
    log::info!(
        "decode_throughput: warm-up run ({}p/{decode_tokens}g)",
        tokens.len()
    );

    let measured = measure_cell(
        "decode_throughput",
        &progress,
        &readiness,
        &thermal,
        || {
            llama::reset_context(model)?;
            llama::sampler_reset(model);
            // Untimed — re-establishes KV depth before the timed decode.
            llama::prefill(model, tokens)
        },
        // decode_n_greedy_ignore_eog always generates exactly `decode_tokens`:
        // decode_greedy has doomloop early-stopping that inflates throughput when
        // the dummy "hello " prompt produces repetitive output.
        || llama::decode_n_greedy_ignore_eog(model, decode_tokens),
        |rep| Ok(rep.elapsed_ms()),
    )?;
    let (mean_ms, stddev_ms) = (measured.stats.mean_ms, measured.stats.stddev_ms);
    log::info!("decode_throughput: decode_time_ms={mean_ms:.3} stddev={stddev_ms:.3}");

    let mut result = json!({
        "benchmark_id": benchmark_id,
        "decode_time_ms": mean_ms,
        "decode_time_ms_stddev": stddev_ms,
    });
    merge_thermal_reps(&mut result, &measured.thermal);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_end_to_end_latency(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        run_end_to_end_latency_impl(def, model, context_size, progress, readiness, thermal)
    })
}

fn run_end_to_end_latency_impl(
    def: &BenchmarkDef,
    model: &crate::ModelHandle,
    context_size: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    let prefill_tokens = def.parameter_prefill_tokens.unwrap_or(512);
    let decode_tokens = def.parameter_decode_tokens.unwrap_or(128);
    let benchmark_id = def.benchmark_id.clone();

    // True end-to-end latency: tokenization runs INSIDE the timed window, matching
    // the llama.cpp CLI (which posts a *string* to /completion, so the server
    // tokenizes in-band) and iOS's LlamaBenchmark. Build the prompt text once here,
    // untimed — as the CLI builds it via build_prompt_text outside its measured loop
    // — sized to exactly `prefill_tokens` under this model's tokenizer, so the per-rep
    // `llama::tokenize` lands on the same count. (Prefill / decode keep tokenization
    // untimed — they time a single kernel; max-memory measures peak allocation, not
    // time, so it has no timed window to keep pure.)
    let prompt_text = seed_prompt_text(model, prefill_tokens, context_size, true)?;
    // build_prompt_text lands on exactly this count under the same tokenizer, so the
    // in-window tokenize below is validated against it every rep — a wrong-sized prompt
    // (e.g. an add_special drift between build and in-window tokenize) fails the cell
    // loudly, mirroring the llama.cpp/MLX CLIs' `prompt_n == prompt_tokens` guard,
    // rather than silently reporting a bogus total_time_ms.
    let expected_prompt_tokens = std::cmp::min(prefill_tokens, context_size) as usize;

    // Warm-up at the measured shape, untimed and ungated.
    let warmup_prompt = seed_prompt_tokens(model, prefill_tokens, context_size, true)?;
    llama::reset_context(model)?;
    llama::sampler_reset(model);
    llama::prefill(model, warmup_prompt.as_slice())?;
    llama::decode_n_greedy_ignore_eog(model, decode_tokens)?;
    log::info!("end_to_end_latency: warm-up run ({prefill_tokens}p/{decode_tokens}g)");

    let measured = measure_cell(
        "end_to_end_latency",
        &progress,
        &readiness,
        &thermal,
        || {
            llama::reset_context(model)?;
            llama::sampler_reset(model);
            Ok(())
        },
        || {
            // Tokenize in-window (see prompt_text above), then prefill + decode.
            let tokens = llama::tokenize(model, &prompt_text, true)?;
            let prompt_len = tokens.len();
            llama::prefill(model, &tokens)?;
            // Ignore-EOG so the measured latency covers exactly `decode_tokens`
            // tokens, not a doomloop-truncated run (see run_decode_throughput_impl).
            llama::decode_n_greedy_ignore_eog(model, decode_tokens)?;
            Ok(prompt_len)
        },
        |rep| {
            if rep.value != expected_prompt_tokens {
                return Err(PipetteError::Benchmark {
                    msg: format!(
                        "e2e prompt tokenized to {} tokens in-window, expected \
                         {expected_prompt_tokens} — prompt-seed / tokenizer drift",
                        rep.value
                    ),
                });
            }
            Ok(rep.elapsed_ms())
        },
    )?;
    let (mean_ms, stddev_ms) = (measured.stats.mean_ms, measured.stats.stddev_ms);
    log::info!("end_to_end_latency: total_time_ms={mean_ms:.3} stddev={stddev_ms:.3}");

    let mut result = json!({
        "benchmark_id": benchmark_id,
        "total_time_ms": mean_ms,
        "total_time_ms_stddev": stddev_ms,
    });
    merge_thermal_reps(&mut result, &measured.thermal);
    Ok(result)
}

fn run_max_memory_usage(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
) -> Result<Value, PipetteError> {
    let prefill_tokens = def.parameter_prefill_tokens.unwrap_or(512);
    let benchmark_id = def.benchmark_id.clone();
    // iOS is a long-lived in-process runner, so phys_footprint behaves like a
    // sticky process-residency counter. currentAllocatedSize is the closest
    // per-run allocator counter we have so far, even when ngl=0. This assumes
    // llama_shim loads with mmap off; if mmap is re-enabled, Metal may
    // wrap file-backed GGUF pages with no-copy buffers that currentAllocatedSize
    // does not fully count.
    let metal_poller = llama::spawn_metal_allocation_poller();
    let mut live_metal_sample = 0;

    let result = with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        let tokens = seed_prompt_tokens(model, prefill_tokens, context_size, true)?;
        llama::prefill(model, &tokens)?;
        llama::decode_n_greedy_ignore_eog(model, 1)?;
        live_metal_sample = llama::metal_allocated_size_bytes();

        Ok(())
    });

    let max_ram_bytes = metal_poller.stop_and_join_with_sample(live_metal_sample)?;
    result?;

    Ok(json!({
        "benchmark_id": benchmark_id,
        "max_ram_bytes": max_ram_bytes
    }))
}

// ---------------------------------------------------------------------------
// VL throughput — multimodal vision+text measurement
// ---------------------------------------------------------------------------

/// Media marker that the llama.cpp mtmd library expects in the prompt string
/// to indicate where each image embedding should be spliced in. Matches
/// `crates/ee-cli/src/execute/vl_throughput.rs:21`.
const MEDIA_MARKER: &str = "<__media__>";

/// Number of measurement runs after the warmup. Matches the CLI's
/// `NUM_MEASUREMENT_RUNS` constant in `crates/ee-cli/src/execute/vl_throughput.rs:17`.
const VL_NUM_MEASUREMENT_RUNS: usize = 5;

#[allow(clippy::too_many_arguments)]
fn run_vl_throughput(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    mmproj_path: &str,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        run_vl_throughput_impl(
            def,
            model,
            mmproj_path,
            n_gpu_layers,
            progress,
            readiness,
            thermal,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_vl_throughput_impl(
    def: &BenchmarkDef,
    model: &crate::ModelHandle,
    mmproj_path: &str,
    n_gpu_layers: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    let use_gpu = n_gpu_layers > 0;
    let image_width = def
        .parameter_image_width
        .ok_or_else(|| PipetteError::Benchmark {
            msg: "vl_throughput missing parameter_image_width".to_string(),
        })?;
    let image_height = def
        .parameter_image_height
        .ok_or_else(|| PipetteError::Benchmark {
            msg: "vl_throughput missing parameter_image_height".to_string(),
        })?;
    let text_tokens = def
        .parameter_text_tokens
        .ok_or_else(|| PipetteError::Benchmark {
            msg: "vl_throughput missing parameter_text_tokens".to_string(),
        })?;
    let decode_tokens = def
        .parameter_decode_tokens
        .ok_or_else(|| PipetteError::Benchmark {
            msg: "vl_throughput missing parameter_decode_tokens".to_string(),
        })?;
    let benchmark_id = def.benchmark_id.clone();

    // Build an exact-token-count text prompt against this model's tokenizer.
    // The mtmd media marker is prepended separately so the tokenized length
    // reflects the text payload only. Shared with the CLIs via
    // ops::prompt_seed so mobile and llama-server measure the same corpus.
    let text_prompt = pipette_ops::prompt_seed::build_prompt_text(text_tokens, |text| {
        // add_special=false so the token count matches what the CLI measures
        // via llama-server's /tokenize (default add_special=false) — otherwise
        // iOS includes BOS and yields 1 fewer actual text token per target.
        llama::tokenize(model, text, false)
            .map(|tokens| tokens.len())
            .map_err(|err| anyhow::anyhow!(err.to_string()))
    })
    .map_err(|err| PipetteError::Benchmark {
        msg: format!(
            "failed to build text prompt of exactly {text_tokens} tokens for vl_throughput: {err}"
        ),
    })?;
    let prompt_with_marker = format!("{MEDIA_MARKER}{text_prompt}");

    let mtmd = llama::mtmd_init(model, mmproj_path, use_gpu)?;

    // Use a closure + explicit free so mtmd_free runs on all paths. We
    // can't use Drop because MtmdHandle doesn't own the model's lifetime.
    let result = run_vl_measurements(
        model,
        &mtmd,
        &prompt_with_marker,
        image_width,
        image_height,
        decode_tokens,
        &benchmark_id,
        progress,
        readiness,
        thermal,
    );
    llama::mtmd_free(mtmd);
    result
}

#[allow(clippy::too_many_arguments)]
fn run_vl_measurements(
    model: &crate::ModelHandle,
    mtmd: &crate::llama::MtmdHandle,
    prompt_with_marker: &str,
    width: u32,
    height: u32,
    decode_tokens: u32,
    benchmark_id: &str,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
    thermal: Option<Arc<dyn ThermalSampler>>,
) -> Result<Value, PipetteError> {
    let total_runs = 1 + VL_NUM_MEASUREMENT_RUNS;
    let mut prompt_ms_samples: Vec<f64> = Vec::with_capacity(VL_NUM_MEASUREMENT_RUNS);
    let mut predicted_ms_samples: Vec<f64> = Vec::with_capacity(VL_NUM_MEASUREMENT_RUNS);
    let mut prompt_tokens_observed: Option<usize> = None;
    // Per-rep thermal snapshots (headroom + status + sensors) for the measured runs (run_index > 0);
    // the warm-up (run_index 0) is ungated and excluded (see run_prefill_throughput_impl).
    let mut thermal_before: Vec<ThermalReading> = Vec::new();
    let mut thermal_after: Vec<ThermalReading> = Vec::new();

    for run_index in 0..total_runs {
        if let Some(ref cb) = progress {
            let label = if run_index == 0 {
                "VL warmup run".to_string()
            } else {
                format!("VL run {}/{}", run_index, VL_NUM_MEASUREMENT_RUNS)
            };
            if !cb.on_progress(run_index as u32, total_runs as u32, label) {
                return Err(PipetteError::Cancelled {
                    msg: format!("cancelled at vl run {run_index}/{total_runs}"),
                });
            }
        }

        // Cool down before each measured run. The warmup (run_index 0) stays
        // ungated so it primes the pipeline without burning the cooldown budget
        // right before the measurements — mirroring the throughput benchmarks,
        // whose warm-up pass is likewise untimed and ungated.
        if run_index > 0 {
            readiness_gate(&readiness)?;
            if let Some(reading) = sample_thermal(&thermal) {
                thermal_before.push(reading);
            }
        }

        llama::reset_context(model)?;
        llama::sampler_reset(model);

        let (chunks_ptr, n_tokens) =
            llama::mtmd_alloc_gray_chunks(mtmd, prompt_with_marker, width, height)?;

        let t0 = Instant::now();
        let eval_rc = llama::mtmd_eval_chunks(mtmd, model, chunks_ptr);
        let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        llama::mtmd_free_chunks(chunks_ptr);
        eval_rc?;

        let t1 = Instant::now();
        llama::decode_n_greedy_ignore_eog(model, decode_tokens)?;
        let predicted_ms = t1.elapsed().as_secs_f64() * 1000.0;

        if run_index == 0 {
            // Warmup: discard timings but capture the canonical prompt token
            // count. This is deterministic across runs for a fixed prompt +
            // image, so we record it once from the warmup run.
            prompt_tokens_observed = Some(n_tokens);
        } else {
            prompt_ms_samples.push(prompt_ms);
            predicted_ms_samples.push(predicted_ms);
            if let Some(reading) = sample_thermal(&thermal) {
                thermal_after.push(reading);
            }
        }
    }

    let prompt_tokens = prompt_tokens_observed.unwrap_or(0) as u32;
    let prompt_ms_mean = mean(&prompt_ms_samples);
    let predicted_ms_mean = mean(&predicted_ms_samples);
    let prompt_ms_stddev = stddev(&prompt_ms_samples, prompt_ms_mean);
    let predicted_ms_stddev = stddev(&predicted_ms_samples, predicted_ms_mean);

    let mut result = json!({
        "benchmark_id": benchmark_id,
        "prompt_tokens": prompt_tokens,
        "prompt_ms": prompt_ms_mean,
        "prompt_ms_stddev": prompt_ms_stddev,
        "predicted_ms": predicted_ms_mean,
        "predicted_ms_stddev": predicted_ms_stddev,
    });
    merge_thermal(&mut result, &thermal_before, &thermal_after);
    Ok(result)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[allow(clippy::too_many_arguments)]
fn run_eval(
    def: &BenchmarkDef,
    model_path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    n_ubatch: u32,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
) -> Result<Value, PipetteError> {
    with_model(model_path, n_gpu_layers, context_size, n_ubatch, |model| {
        run_eval_impl(def, model, progress, readiness)
    })
}

fn run_eval_impl(
    def: &BenchmarkDef,
    model: &crate::ModelHandle,
    progress: Option<Arc<dyn ProgressCallback>>,
    readiness: Option<Arc<dyn ReadinessCallback>>,
) -> Result<Value, PipetteError> {
    let samples = def
        .samples
        .as_deref()
        .ok_or_else(|| PipetteError::Benchmark {
            msg: "eval benchmark missing samples".to_string(),
        })?;
    let max_tokens = def.parameter_max_tokens.unwrap_or(256);
    let mcq_choices = def.parameter_mcq_choices.as_ref().filter(|c| !c.is_empty());
    let benchmark_id = def.benchmark_id.clone();

    let mcq_json = mcq_choices.map(|c| serde_json::to_string(c).unwrap_or_default());
    let total = samples.len() as u32;

    let sample_data: Vec<(String, String)> = samples
        .iter()
        .map(|sample| {
            let sample_id = sample
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let messages = sample
                .get("messages")
                .cloned()
                .unwrap_or(Value::Array(vec![]));
            let messages_json = serde_json::to_string(&messages).unwrap_or_default();
            (sample_id, messages_json)
        })
        .collect();

    let mut completions = Vec::with_capacity(sample_data.len());

    for (index, (sample_id, messages_json)) in sample_data.iter().enumerate() {
        // Wait until the device is thermally ready before every sample, not just
        // at cell entry — an eval suite runs many completions back-to-back and
        // would otherwise heat the device unchecked through the run (and bake
        // heat into any timing benchmark that follows it in the same job). The
        // gate is a no-op when the device is already cool. Mirrors the per-rep
        // gating in the throughput benchmarks.
        readiness_gate(&readiness)?;

        // Clear the KV cache before each sample so that every completion
        // starts with a fresh context (matching ee-cli which sends each
        // sample as an independent HTTP request to llama-server).
        llama::reset_context(model)?;

        let effective_max = if mcq_choices.is_some() { 1 } else { max_tokens };

        // A failed sample is recorded against itself rather than silently
        // submitted as an empty completion: `failed` excludes it from scoring
        // server-side, and `stop_reason: failure` says why the text is empty.
        // Only an error that leaves the model unusable ends the cell.
        match llama::chat_completion(model, messages_json, effective_max, mcq_json.as_deref()) {
            Ok(generation) => completions.push(BenchmarkEvalCompletion {
                id: sample_id.clone(),
                completion: generation.text,
                failed: false,
                failed_reason: None,
                stop_reason: generation.stop_reason,
                stop_detail: generation.stop_detail,
                completion_tokens: Some(generation.completion_tokens),
            }),
            Err(err) if aborts_cell(&err) => return Err(err),
            Err(err) => {
                let detail = err.to_string();
                log::warn!("eval sample {sample_id} failed, recording and continuing: {detail}");
                completions.push(BenchmarkEvalCompletion {
                    id: sample_id.clone(),
                    completion: String::new(),
                    failed: true,
                    // Dual-written: `failed_reason` is the legacy field
                    // consumers still read, `stop_detail` its generalization.
                    failed_reason: Some(detail.clone()),
                    stop_reason: BenchmarkEvalCompletionStopReason::Failure,
                    stop_detail: Some(detail),
                    // No count to report: the sample produced no completion.
                    completion_tokens: None,
                });
            }
        }

        let completed = (index + 1) as u32;
        if let Some(ref cb) = progress {
            let should_continue =
                cb.on_progress(completed, total, format!("Sample {completed}/{total}"));
            if !should_continue {
                return Err(PipetteError::Cancelled {
                    msg: format!("cancelled after {completed}/{total} samples"),
                });
            }
        }
    }

    Ok(json!({
        "benchmark_id": benchmark_id,
        "completions": completions
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Sentinel BOS (id 1) under `add_special`, mirroring how Gemma/LFM2
    /// prepend one — lets us test BOS handling without a model.
    fn fake_tokenize(text: &str, add_special: bool) -> Result<Vec<i32>, PipetteError> {
        let mut toks: Vec<i32> = text.bytes().map(i32::from).collect();
        if add_special {
            toks.insert(0, 1);
        }
        Ok(toks)
    }

    // Length == target (not target + 1) and BOS present iff add_special:
    // proves the same flag drives both the convergence count and the encode.
    #[rstest]
    #[case(true, true)]
    #[case(false, false)]
    fn add_special_controls_bos_within_target(
        #[case] add_special: bool,
        #[case] expect_bos: bool,
    ) -> Result<(), PipetteError> {
        let toks = seed_prompt_tokens_with(64, 4096, add_special, fake_tokenize)?;
        assert_eq!(toks.len(), 64);
        assert_eq!(toks[0] == 1, expect_bos);
        Ok(())
    }

    #[test]
    fn caps_target_to_context_size() -> Result<(), PipetteError> {
        let toks = seed_prompt_tokens_with(1000, 64, true, fake_tokenize)?;
        assert_eq!(toks.len(), 64);
        Ok(())
    }

    // The harness's hooks are typed on `anyhow`, so this kernel's errors make a
    // round trip through it. The UniFFI surface distinguishes a user
    // cancellation from a readiness failure, so that round trip has to be
    // lossless — a `.context()` added anywhere in the chain would break the
    // downcast and turn every cancellation into a generic benchmark failure.
    #[rstest]
    #[case::cancelled(PipetteError::Cancelled { msg: "stopped".into() })]
    #[case::readiness(PipetteError::Readiness { msg: "too hot".into() })]
    fn a_kernel_error_survives_the_harness(#[case] original: PipetteError) {
        let expected = std::mem::discriminant(&original);

        let recovered = to_pipette_error(anyhow::Error::new(original));

        assert_eq!(
            std::mem::discriminant(&recovered),
            expected,
            "came back as {recovered:?}"
        );
    }

    /// An error the harness itself raised — a misreported rep, say — has no
    /// kernel variant to recover, so it lands as a benchmark failure.
    #[test]
    fn a_harness_error_becomes_a_benchmark_failure() {
        let err = to_pipette_error(anyhow::anyhow!("a repetition ended that was never started"));

        assert!(matches!(err, PipetteError::Benchmark { .. }), "got {err:?}");
    }

    // e2e builds prompt *text* untimed, then tokenizes it inside the timed window.
    // That in-window tokenize must land on exactly `target` with no truncation —
    // proving the timed count matches what seed_prompt_tokens would have produced.
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn seed_prompt_text_tokenizes_to_exact_target(
        #[case] add_special: bool,
    ) -> Result<(), PipetteError> {
        let text = seed_prompt_text_with(64, 4096, add_special, fake_tokenize)?;
        assert_eq!(fake_tokenize(&text, add_special)?.len(), 64);
        Ok(())
    }

    // A 0-token prefill (decode_throughput_0_32) must still hand the engine one token: an
    // empty batch is rejected by ee_llama_decode_batch with -1, and llama-bench likewise
    // opens a generation-only run on BOS rather than on nothing.
    #[test]
    fn zero_target_keeps_bos_so_the_engine_has_something_to_decode() -> Result<(), PipetteError> {
        let toks = seed_prompt_tokens_with(0, 4096, true, fake_tokenize)?;
        assert_eq!(toks, vec![1]);
        Ok(())
    }

    // Without add_special there is no BOS to keep, so a 0 target really is empty. Callers on
    // that path (vl_throughput) must not ask for a 0-token prefill.
    #[test]
    fn zero_target_without_add_special_stays_empty() -> Result<(), PipetteError> {
        let toks = seed_prompt_tokens_with(0, 4096, false, fake_tokenize)?;
        assert!(toks.is_empty());
        Ok(())
    }

    /// A prompt that crashes inference costs its own sample, not the run. The
    /// CLI's reason for this is that llama.cpp is known to crash on particular
    /// prompts, and aborting would throw away every sample already measured.
    #[rstest]
    #[case::inference(PipetteError::Inference { msg: "decode failed with code 1".into() })]
    #[case::tokenize(PipetteError::Tokenize { msg: "bad token".into() })]
    #[case::benchmark(PipetteError::Benchmark { msg: "malformed sample".into() })]
    #[case::json(PipetteError::Json { msg: "trailing comma".into() })]
    fn a_recoverable_error_records_the_sample_and_continues(#[case] err: PipetteError) {
        assert!(!aborts_cell(&err), "{err} should not end the cell");
    }

    /// The exceptions. This kernel runs in-process and cannot restart anything,
    /// so once the model is unusable every remaining sample would fail the same
    /// way. A cancel is not a failure at all and must not be recorded as one.
    #[rstest]
    #[case::cancelled(PipetteError::Cancelled { msg: "user stopped the job".into() })]
    #[case::out_of_memory(PipetteError::OutOfMemory { msg: "alloc failed".into() })]
    #[case::model_load(PipetteError::ModelLoad { msg: "weights gone".into() })]
    fn an_unrecoverable_error_ends_the_cell(#[case] err: PipetteError) {
        assert!(aborts_cell(&err), "{err} should end the cell");
    }

    /// The wire shape mgmt receives for a clean sample. `stop_reason` is always
    /// present, and the optional fields elide rather than serializing as null,
    /// so a completion carries no `failed` key unless it actually failed.
    #[test]
    fn a_clean_completion_serializes_without_the_failure_fields() -> Result<(), PipetteError> {
        let completion = BenchmarkEvalCompletion {
            id: "s1".to_string(),
            completion: "answer".to_string(),
            failed: false,
            failed_reason: None,
            stop_reason: BenchmarkEvalCompletionStopReason::Eos,
            stop_detail: None,
            completion_tokens: Some(12),
        };

        assert_eq!(
            serde_json::to_value(&completion)?,
            json!({
                "id": "s1",
                "completion": "answer",
                "stop_reason": "eos",
                "completion_tokens": 12,
            })
        );
        Ok(())
    }

    /// A failed sample dual-writes the detail: `failed_reason` is the legacy
    /// field consumers still read, `stop_detail` its generalization. An empty
    /// completion paired with `failure` is what keeps it out of scoring.
    #[test]
    fn a_failed_sample_records_the_reason_in_both_fields() -> Result<(), PipetteError> {
        let completion = BenchmarkEvalCompletion {
            id: "s2".to_string(),
            completion: String::new(),
            failed: true,
            failed_reason: Some("inference error: decode failed with code 1".to_string()),
            stop_reason: BenchmarkEvalCompletionStopReason::Failure,
            stop_detail: Some("inference error: decode failed with code 1".to_string()),
            completion_tokens: None,
        };

        assert_eq!(
            serde_json::to_value(&completion)?,
            json!({
                "id": "s2",
                "completion": "",
                "failed": true,
                "failed_reason": "inference error: decode failed with code 1",
                "stop_reason": "failure",
                "stop_detail": "inference error: decode failed with code 1",
            })
        );
        Ok(())
    }
}
