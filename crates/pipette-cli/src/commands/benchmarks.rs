//! `benchmarks` — catalog ops (`list` / `show` / `init-local`) and **`run`**.
//!
//! - Catalog commands manage **benchmark** definitions in the workspace store.
//! - `benchmarks run` starts one **run** of a benchmark: build [`ClientRunSpec`],
//!   prepare [`pipette_plan_types::run::RunRequest`], then [`crate::run::run_cell`].
//!
//! See `docs/architecture.md` (“Benchmark vs run vs execute”).

use anyhow::Context;
use clap::{Args, Subcommand};
use tabled::Tabled;

use pipette_http::HttpClient;
use pipette_plan_types::benchmark::BenchmarkDefinition;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::{
    is_compatible, BenchmarkFlagRef, BenchmarkFlags, BenchmarkId, BenchmarkType, ClientRunSpec,
    Model, ModelFlagRef, ModelFlags, ModelType, ReadinessOverrides, Runtime, RuntimeFlags,
    RuntimeType,
};

use crate::artifact_ref::{resolve_model_arg, resolve_runtime_arg};
use crate::benchmarks::{seed_standard_local, SourcedBenchmarkId};
use crate::commands::print_table_or;
use crate::doomloop_cli::DoomloopCliArgs;
use crate::hf_auth::inject_env_hf_token;
use crate::results::{record_and_maybe_submit_run, RecordSubmitOutcome};
use crate::run::run_cell;
use crate::workspace::PipetteWorkspace;

/// List, inspect, and run benchmarks
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub command: BenchmarkCommand,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)] // BenchmarkRunArgs carries the full CLI surface; boxing it ripples through all call sites
pub enum BenchmarkCommand {
    /// List available benchmarks
    List(BenchmarkListArgs),
    /// Show a benchmark definition's id, type, source and parameters
    Show(BenchmarkShowArgs),
    /// Create the standard local benchmark definitions under benchmarks/local
    InitLocal,
    /// Execute a benchmark and store the result locally
    #[command(long_about = RUN_LONG_ABOUT, after_long_help = RUN_AFTER_HELP)]
    Run(BenchmarkRunArgs),
}

#[derive(Args, Debug)]
pub struct BenchmarkListArgs {
    /// Only list benchmarks of this type: prefill-throughput,
    /// decode-throughput, end-to-end-latency, max-memory-usage, eval,
    /// vl-throughput (the underscore spellings are accepted too)
    #[arg(long)]
    pub benchmark_type: Option<BenchmarkType>,
}

#[derive(Args, Debug)]
pub struct BenchmarkShowArgs {
    /// Benchmark reference: `<id>` or `remote/<id>` (synced catalog), or
    /// `local/<id>` for a definition only this machine has
    pub benchmark_ref: SourcedBenchmarkId,
}

/// One row of `benchmarks list` — the reference to pass to `run`/`show` plus
/// its kind. The catalog is runtime-agnostic, so this is identical across
/// clients.
#[derive(Tabled)]
struct BenchmarkRow {
    #[tabled(rename = "BENCHMARK")]
    benchmark_ref: String,
    #[tabled(rename = "TYPE")]
    benchmark_type: String,
}

impl BenchmarkListArgs {
    fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        print_table_or(&self.rows(ws)?, "No benchmarks.");
        Ok(())
    }

    /// The local + remote catalog rows honoring `--benchmark-type`, split out
    /// so the filtering/labeling is testable without capturing stdout.
    fn rows(&self, ws: &PipetteWorkspace) -> anyhow::Result<Vec<BenchmarkRow>> {
        use pipette_plan_types::benchmark::BenchmarkSource;

        Ok([BenchmarkSource::Local, BenchmarkSource::Remote]
            .into_iter()
            .map(|source| {
                Ok(ws
                    .benchmarks()
                    .list(source)?
                    .into_iter()
                    .filter(|(_, def)| {
                        self.benchmark_type
                            .is_none_or(|f| f == def.benchmark_type())
                    })
                    .map(|(id, def)| {
                        let benchmark_ref = SourcedBenchmarkId::new(source, id);
                        BenchmarkRow {
                            benchmark_ref: benchmark_ref.to_string(),
                            benchmark_type: def.benchmark_type().to_string(),
                        }
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}

impl BenchmarkShowArgs {
    fn execute(self, ws: &PipetteWorkspace) -> anyhow::Result<()> {
        let reference = self.benchmark_ref;
        let def = ws
            .benchmarks()
            .get(&reference)?
            .ok_or_else(|| anyhow::anyhow!("unknown benchmark reference `{reference}`"))?;
        print_benchmark(&def, reference.source());
        Ok(())
    }
}

fn print_benchmark(
    def: &pipette_plan_types::benchmark::BenchmarkDefinition,
    source: pipette_plan_types::benchmark::BenchmarkSource,
) {
    use pipette_plan_types::benchmark::BenchmarkDefinition;
    println!("id:      {}", def.benchmark_id());
    println!("type:    {}", def.benchmark_type());
    println!(
        "source:  {}",
        match source {
            pipette_plan_types::benchmark::BenchmarkSource::Local => "local",
            pipette_plan_types::benchmark::BenchmarkSource::Remote => "remote",
        }
    );
    match def {
        BenchmarkDefinition::PrefillThroughput(b) => {
            println!("prefill: {} tokens", b.parameter_prefill_tokens);
        }
        BenchmarkDefinition::DecodeThroughput(b) => {
            println!("prefill: {} tokens", b.parameter_prefill_tokens);
            println!("decode:  {} tokens", b.parameter_decode_tokens);
        }
        BenchmarkDefinition::EndToEndLatency(b) => {
            println!("prefill: {} tokens", b.parameter_prefill_tokens);
            println!("decode:  {} tokens", b.parameter_decode_tokens);
        }
        BenchmarkDefinition::MaxMemoryUsage(b) => {
            println!("prefill: {} tokens", b.parameter_prefill_tokens);
        }
        BenchmarkDefinition::Eval(b) => {
            println!("eval:    {}", b.parameter_eval_id);
            println!("dataset: {}", b.parameter_dataset_name);
            println!("max_tok: {}", b.parameter_max_tokens);
            if let Some(choices) = &b.parameter_mcq_choices {
                println!("mcq:     {}", choices.join(", "));
            }
            if let Some(samples) = &b.samples {
                println!("samples: {}", samples.len());
            }
        }
        BenchmarkDefinition::VlThroughput(b) => {
            println!(
                "image:   {}×{}",
                b.parameter_image_width, b.parameter_image_height
            );
            println!("text:    {} tokens", b.parameter_text_tokens);
            println!("decode:  {} tokens", b.parameter_decode_tokens);
        }
    }
}

const RUN_LONG_ABOUT: &str = "\
Execute one benchmark and store the result locally.

A run is one cell: a benchmark, a runtime, and a model. Name the runtime and the
model with a compact URI, a JSON object, or a `sha256=` digest of something
already in the local store; whatever is missing is fetched before the
measurement starts.

The cell decides which settings are legal. `--runtime-flags` takes the settings
for this cell only, so it carries no axis keys, while `--model-flags` keeps the
plan's axes. A setting the cell does not accept is an error, never a silent
no-op.";

const RUN_AFTER_HELP: &str = "\
Examples:
  # smallest thing that works: a local benchmark needs no server and no
  # registration. `local/` ids come from `benchmarks init-local`, so seed them once:
  pipette benchmarks init-local
  pipette benchmarks run --benchmark local/prefill_throughput_smoke \\
    --model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf' \\
    --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

  # tune the cell: settings only, no runtime_type/model_type/benchmark_type
  pipette benchmarks run --benchmark local/decode_throughput_512_100 \\
    --model 'gguf-text://repo=unsloth/Qwen3.5-0.8B-GGUF&path=Qwen3.5-0.8B-Q4_0.gguf' \\
    --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64' \\
    --runtime-flags '{\"threads\":8,\"number_gpu_layers\":99,\"flash_attention\":\"on\"}'

  # anything the typed settings don't cover goes through `raw`, verbatim
  pipette benchmarks run --benchmark local/prefill_throughput_512 \\
    --model 'gguf-text://repo=unsloth/Qwen3.5-0.8B-GGUF&path=Qwen3.5-0.8B-Q4_0.gguf' \\
    --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=linux-x64-cpu' \\
    --runtime-flags '{\"raw\":[\"--numa\",\"distribute\"]}'

  # vision: weights and mmproj in one URI
  pipette benchmarks run --benchmark local/vl_throughput_smoke \\
    --model 'gguf-vision://repo=ggml-org/gemma-3-4b-it-GGUF&model=gemma-3-4b-it-Q4_K_M.gguf&mmproj=mmproj-model-f16.gguf' \\
    --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

  # eval on MLX with thinking turned off (eval-only; --model-flags is the JSON form)
  pipette benchmarks run --benchmark local/eval_smoke \\
    --model 'mlx://repo=mlx-community/Qwen3.5-0.8B-4bit' \\
    --runtime 'mlx-macos-pipette://version=0.31.3' \\
    --model-enable-thinking false

  # the same thing spelled as JSON; this one DOES need its axes
  pipette benchmarks run --benchmark local/eval_smoke \\
    --model 'mlx://repo=mlx-community/Qwen3.5-0.8B-4bit' \\
    --runtime 'mlx-macos-pipette://version=0.31.3' \\
    --model-flags '{\"model_type\":\"mlx\",\"benchmark_type\":\"eval\",\"enable_thinking\":false}'

  # vLLM under docker
  pipette benchmarks run --benchmark local/end_to_end_latency_smoke \\
    --model 'torch://repo=Qwen/Qwen2.5-0.5B-Instruct' \\
    --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \\
    --runtime-flags '{\"dtype\":\"bfloat16\",\"max_model_len\":4096,\"gpus\":\"all\"}'

  # OpenVINO picks its device per cell, so the venv is shared across cpu/gpu/npu
  pipette benchmarks run --benchmark local/prefill_throughput_512 \\
    --model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw' \\
    --runtime 'uv-openvino://version=2026.2.1' \\
    --runtime-flags '{\"device\":\"cpu\"}'

  # re-run against what's already installed, by digest prefix
  pipette benchmarks run --benchmark local/decode_throughput_512_100 \\
    --model 'model://sha256=d86cc299' --runtime 'runtime://sha256=c07a4fd3'

CELL SETTINGS (--runtime-flags) -- which are legal depends on all three axes:
  llama.cpp x gguf_text
    prefill / decode / max-memory   threads, number_gpu_layers, mmap,
                                    flash_attention, raw
    end-to-end / eval               the above, plus ctx_size, no_cache
  llama.cpp x gguf_vision
    vl                              threads, number_gpu_layers, mmap,
                                    flash_attention, ctx_size, no_cache, raw
  docker-vllm x torch               tensor_parallel_size, dtype, max_model_len,
                                    prefix_caching, gpus, shm_size, ipc, envs, raw
  uv-vllm x torch                   the above, minus gpus / shm_size / ipc
  docker-sglang x torch             tensor_parallel_size, prefix_caching, gpus,
                                    shm_size, ipc, envs, raw
  uv-sglang x torch                 tensor_parallel_size, prefix_caching, envs, raw
  uv-openvino x openvino            device, max_prompt_len, min_response_len,
                                    generate_hint
  mlx-macos-pipette x mlx           none: any setting is refused ({} is a no-op)

`raw` takes verbatim argv tokens, and refuses any that alias a typed setting or
that the benchmark owns for that cell.

A `local/<id>` result stays on this machine. `--sync` submits only a result from
the synced catalog; `pipette sync` does the same afterwards.

Gated Hugging Face weights: set PIPETTE_HF_TOKEN.

Full notation reference: docs/pipette-cli/models-and-runtimes.md";

/// The `benchmarks run` flag surface. NOTE: the plan runner's
/// `RunnableCell::build_argv` still emits the legacy flat
/// `--model`/`--runtime` refs and `--mmproj`; reconciling it with these JSON/URI
/// refs (and dropping `--mmproj`) is PIP-344 step 5, when the runner is repointed
/// at this unified binary.
#[derive(Args, Debug)]
pub struct BenchmarkRunArgs {
    /// Benchmark reference: `<id>` or `remote/<id>` (synced catalog), or
    /// `local/<id>` for a definition only this machine has
    #[arg(long = "benchmark")]
    pub benchmark_ref: SourcedBenchmarkId,
    /// Model as the compact URI (`gguf-text://repo=org/r&path=Q4_K_M.gguf`,
    /// `gguf-vision://repo=org/r&model=m.gguf&mmproj=proj.gguf`,
    /// `mlx://repo=org/r`, `torch://repo=org/r`, `openvino://repo=org/r`), a
    /// JSON `Model` object, or `model://sha256=<prefix>` naming a model already
    /// in the local store (>= 8 hex chars from `pipette models list`).
    #[arg(long)]
    pub model: String,
    /// Runtime as the compact URI
    /// (`llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64`,
    /// `docker-vllm://image=…&tag=…`, `uv-vllm://server=…`), a JSON `Runtime`
    /// object, or `runtime://sha256=<prefix>` naming an installed runtime
    /// (>= 8 hex chars from `pipette runtimes list`).
    #[arg(long)]
    pub runtime: String,
    /// Runtime settings for this cell, as one JSON object —
    /// `'{"threads":8,"number_gpu_layers":99}'`. Cells that drive a command
    /// line also take `raw`, an array of verbatim argv tokens. An array of
    /// entries is rejected, and so are the `runtime_type` / `model_type` /
    /// `benchmark_type` axis keys a plan carries: the cell comes from
    /// `--benchmark`, `--runtime` and `--model`. Which settings a cell accepts
    /// depends on all three, and an unaccepted one is an error, not a no-op.
    #[arg(long)]
    pub runtime_flags: Option<String>,
    /// Model-generation settings, as one plan-form JSON object —
    /// `'{"model_type":"mlx","benchmark_type":"eval","enable_thinking":true}'`
    /// (a one-element array is also accepted). Unlike `--runtime-flags` this
    /// one *requires* its `model_type` / `benchmark_type` axes. Eval benchmarks
    /// only. Mutually exclusive with `--model-enable-thinking`.
    #[arg(long)]
    pub model_flags: Option<String>,
    /// HTTP timeout in seconds for the client's calls to the local server.
    /// Accepted on the cells that drive one over HTTP — eval, end-to-end-latency
    /// and vl — and rejected elsewhere.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub http_timeout_seconds: Option<u64>,
    /// Override the readiness wait ceiling (seconds) before the measurement.
    /// Valid only on the timing benchmarks that gate on device readiness
    /// (prefill / decode / end-to-end-latency / vl); on eval or max-memory the
    /// `(benchmark, runtime, model)` cell carries no readiness knob, so it is
    /// rejected. Set `PIPETTE_READINESS_MAX_WAIT_SECS` directly for an ungated
    /// override: it is read by the readiness gate itself, so unlike this flag it
    /// stays harmless on the cells that carry no readiness setting.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub readiness_max_wait_secs: Option<u64>,
    /// Waive the *thermal* readiness criterion for this cell, keeping the load
    /// criterion. Valid on the same timing benchmarks as
    /// `--readiness-max-wait-secs`. Changes the criteria rather than just the
    /// patience, so a cell run with it is not comparable to a gated one.
    #[arg(long)]
    pub readiness_skip_thermal: bool,
    /// Convenience for a single `enable_thinking` model flag: derives the axes
    /// from the cell, so it needs no JSON. Eval benchmarks only. Mutually
    /// exclusive with `--model-flags`.
    #[arg(long)]
    pub model_enable_thinking: Option<bool>,
    /// Submit the result to the management server immediately after the run.
    /// Applies only to a benchmark from the synced catalog — a `local/<id>` run
    /// always stays on disk, and passing this does not change that.
    #[arg(long)]
    pub sync: bool,
    // Last on purpose: this group carries a `next_help_heading`, which clap
    // applies to every argument declared after it.
    #[command(flatten)]
    pub doomloop: DoomloopCliArgs,
}

impl BenchmarkArgs {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        self.command.execute(ws, http)
    }
}

impl BenchmarkCommand {
    pub fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        match self {
            // The catalog is runtime-agnostic; init-local seeds the union of
            // every runtime's supported benchmark types.
            BenchmarkCommand::InitLocal => {
                let summary = seed_standard_local(&ws.benchmarks(), BenchmarkType::ALL)?;
                println!(
                    "Wrote standard local benchmarks: {} created, {} updated",
                    summary.created, summary.updated
                );
                Ok(())
            }
            BenchmarkCommand::List(args) => args.execute(ws),
            BenchmarkCommand::Show(args) => args.execute(ws),
            BenchmarkCommand::Run(args) => args.execute(ws, http),
        }
    }
}

impl BenchmarkRunArgs {
    fn execute(self, ws: &PipetteWorkspace, http: &HttpClient) -> anyhow::Result<()> {
        // Parse → run_cell → CLI finish (local record + optional submit).
        let sync = self.sync;
        // The catalog side is this client's business: it picks the filing location
        // and stops here — the run takes the resolved body, not the address.
        let location = self.benchmark_ref.source().into();
        let (spec, benchmark) = self.into_client_run_spec(ws)?;
        // `prepare` adds the cell's pins; the policy only carries the cap here.
        let artifacts = ws.artifacts(http);
        let (payload, extras) = run_cell(&spec, benchmark, &artifacts, ws)?;
        report_lines(&run_summary_lines(
            &payload.benchmark_id,
            payload.result.benchmark_type(),
            &payload.result,
        ));
        let done = record_and_maybe_submit_run(
            &ws.results(),
            &ws.identity(),
            &payload,
            &extras,
            location,
            sync,
            http,
        )?;
        print_record_done(ws, &done);
        Ok(())
    }

    /// Parse the runtime/model refs (canonical JSON `Runtime`/`Model`, or the
    /// compact URI), gate on `is_compatible`, resolve the benchmark ref once, and
    /// validate the run knobs into plan-form flag enums. Both refs are
    /// self-describing (the model tag names its format).
    ///
    /// Returns the resolved body alongside the spec, so the run path takes it
    /// rather than resolving the same id a second time.
    fn into_client_run_spec(
        self,
        ws: &PipetteWorkspace,
    ) -> anyhow::Result<(ClientRunSpec, BenchmarkDefinition)> {
        let runtime = resolve_runtime_arg(ws, &self.runtime)?;
        let mut model = resolve_model_arg(ws, &self.model)?;
        inject_env_hf_token(&mut model)?;
        if !is_compatible(&model, &runtime) {
            anyhow::bail!(
                "model `{}` is not compatible with runtime `{}`",
                self.model,
                self.runtime
            );
        }
        // Resolve the ref to its definition once. Its `.benchmark_type()` is the
        // authoritative type (not a naming-convention guess from the id string),
        // and the body is handed to the run so nothing resolves it twice.
        let run_ref = &self.benchmark_ref;
        let benchmark = ws
            .benchmarks()
            .get(run_ref)?
            .ok_or_else(|| anyhow::anyhow!("unknown benchmark reference `{run_ref}`"))?;
        // `--runtime-flags` carries the knobs; the cell they apply to is the one just
        // resolved from `--benchmark`, `--runtime` and `--model`, so it is derived here
        // rather than restated in the JSON. Parsed after the benchmark for that reason —
        // its `.benchmark_type()` is the authoritative axis. An entry that *does* name its
        // cell is checked against this one. The `raw` escape hatch rides inside the entry;
        // the owning runtime renders it at execution time.
        let runtime_flags = match self.runtime_flags.as_deref() {
            Some(json) => RuntimeFlags::from_cell_json(
                json,
                RuntimeType::of(&runtime),
                ModelType::of(&model),
                benchmark.benchmark_type(),
            )
            .context("parsing --runtime-flags JSON")?,
            None => None,
        };
        // Plan-form model flags: explicit `--model-flags` JSON (like runtime),
        // or the flat `--model-enable-thinking` convenience — not both.
        let model_flags = match (self.model_flags.as_deref(), self.model_enable_thinking) {
            (Some(_), Some(_)) => {
                anyhow::bail!("--model-flags and --model-enable-thinking are mutually exclusive")
            }
            (Some(json), None) => {
                let flags = parse_model_flags_json(json).context("parsing --model-flags JSON")?;
                if let Some(ref f) = flags {
                    if !f.matches(benchmark.benchmark_type(), &model) {
                        anyhow::bail!(
                            "--model-flags entry does not match this cell \
                             (benchmark={:?}, model={})",
                            benchmark.benchmark_type(),
                            model
                        );
                    }
                }
                flags
            }
            (None, Some(enable_thinking)) => Some(
                ModelFlags::try_from(ModelFlagRef {
                    model_type: ModelType::of(&model),
                    benchmark_type: benchmark.benchmark_type(),
                    enable_thinking: Some(enable_thinking),
                })
                .with_context(|| {
                    format!(
                        "--model-enable-thinking is not valid for this benchmark ({:?})",
                        benchmark.benchmark_type()
                    )
                })?,
            ),
            (None, None) => None,
        };
        let benchmark_flags = build_benchmark_flags(
            benchmark.benchmark_type(),
            &runtime,
            &model,
            self.http_timeout_seconds,
            self.doomloop.into_overrides(),
            self.readiness_max_wait_secs,
            self.readiness_skip_thermal,
        )?;
        let spec = ClientRunSpec {
            benchmark: BenchmarkId::try_new(benchmark.benchmark_id().to_string())
                .context("benchmark id")?,
            model,
            runtime,
            runtime_flags,
            model_flags,
            benchmark_flags,
        };
        Ok((spec, benchmark))
    }
}

/// Parse `--model-flags` JSON: a one-element `Vec<ModelFlags>` (plan wire) or a
/// bare `ModelFlags` object (claim shape). Reject multi-entry arrays.
fn parse_model_flags_json(json: &str) -> anyhow::Result<Option<ModelFlags>> {
    if let Ok(parsed) = serde_json::from_str::<Vec<ModelFlags>>(json) {
        if parsed.len() > 1 {
            anyhow::bail!(
                "--model-flags carries {} entries; a single run resolves to at most one",
                parsed.len(),
            );
        }
        return Ok(parsed.into_iter().next());
    }
    Ok(Some(
        serde_json::from_str::<ModelFlags>(json).context("expected ModelFlags object or array")?,
    ))
}

/// Build the eval-run `BenchmarkFlags` for a cell from its resolved benchmark
/// TYPE (authoritative) + model + the raw eval knobs. `None` when no knob is set.
fn build_benchmark_flags(
    benchmark_type: BenchmarkType,
    runtime: &Runtime,
    model: &Model,
    http_timeout_seconds: Option<u64>,
    doomloop: pipette_doomloop::plan::DoomloopOverrides,
    readiness_max_wait_secs: Option<u64>,
    readiness_skip_thermal: bool,
) -> anyhow::Result<Option<BenchmarkFlags>> {
    let readiness = (readiness_max_wait_secs.is_some() || readiness_skip_thermal).then(|| {
        ReadinessOverrides {
            max_wait_secs: readiness_max_wait_secs,
            // `None` rather than `Some(false)` when unset, so an untouched knob
            // stays absent from the serialized plan.
            skip_thermal: readiness_skip_thermal.then_some(true),
        }
    });
    let has_knob = http_timeout_seconds.is_some()
        || doomloop != pipette_doomloop::plan::DoomloopOverrides::default()
        || readiness.is_some();
    if !has_knob {
        return Ok(None);
    }
    Ok(Some(
        // `TryFrom` routes the (benchmark, runtime, model) triple to its variant
        // and rejects any knob that cell doesn't carry — so a readiness override
        // is accepted only on a readiness-carrying (timing/vl) cell, and errors
        // on eval/max-memory.
        BenchmarkFlags::try_from(BenchmarkFlagRef {
            runtime_type: RuntimeType::of(runtime),
            model_type: ModelType::of(model),
            benchmark_type,
            http_timeout_seconds,
            doomloop,
            readiness,
        })
        .with_context(|| {
            format!(
                "--http-timeout-seconds / --doomloop-* / --readiness-max-wait-secs \
                 are not valid for this benchmark ({benchmark_type:?})"
            )
        })?,
    ))
}

fn report_lines(lines: &[String]) {
    lines.iter().for_each(|line| println!("{line}"));
}

fn print_record_done(ws: &PipetteWorkspace, done: &RecordSubmitOutcome) {
    println!(
        "payload: {}",
        ws.results()
            .payload_path(done.location, &done.result_id)
            .display()
    );
    println!("recorded {} ({})", done.result_id, done.location.label());
    if let Some(job_id) = &done.job_id {
        println!("submitted as job {job_id}");
    }
}

/// Human-readable run summary lines (benchmark id/type + headline metrics).
fn run_summary_lines(
    benchmark_id: &str,
    benchmark_type: BenchmarkType,
    data: &BenchmarkResultData,
) -> Vec<String> {
    let mut lines = vec![
        format!("benchmark: {benchmark_id}"),
        format!("type: {benchmark_type}"),
    ];
    match data {
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms,
            prefill_time_ms_stddev,
        } => {
            lines.push(format!("prefill_time_ms: {prefill_time_ms:.6}"));
            if let Some(stddev) = prefill_time_ms_stddev {
                lines.push(format!("prefill_time_ms_stddev: {stddev:.6}"));
            }
        }
        BenchmarkResultData::DecodeThroughput {
            decode_time_ms,
            decode_time_ms_stddev,
        } => {
            lines.push(format!("decode_time_ms: {decode_time_ms:.6}"));
            if let Some(stddev) = decode_time_ms_stddev {
                lines.push(format!("decode_time_ms_stddev: {stddev:.6}"));
            }
        }
        BenchmarkResultData::EndToEndLatency {
            total_time_ms,
            total_time_ms_stddev,
        } => {
            lines.push(format!("total_time_ms: {total_time_ms:.6}"));
            if let Some(stddev) = total_time_ms_stddev {
                lines.push(format!("total_time_ms_stddev: {stddev:.6}"));
            }
        }
        BenchmarkResultData::MaxMemoryUsage {
            max_host_bytes,
            max_gpu_bytes,
            max_npu_bytes,
        } => {
            lines.push(format!("max_host_bytes: {max_host_bytes}"));
            if let Some(bytes) = max_gpu_bytes {
                lines.push(format!("max_gpu_bytes: {bytes}"));
            }
            if let Some(bytes) = max_npu_bytes {
                lines.push(format!("max_npu_bytes: {bytes}"));
            }
        }
        BenchmarkResultData::Eval { completions } => {
            lines.push(format!("completions: {}", completions.len()));
        }
        BenchmarkResultData::VlThroughput {
            prompt_tokens,
            prompt_ms,
            prompt_ms_stddev,
            predicted_ms,
            predicted_ms_stddev,
        } => {
            lines.push(format!("prompt_tokens: {prompt_tokens}"));
            lines.push(format!("prompt_ms: {prompt_ms:.3}"));
            if let Some(stddev) = prompt_ms_stddev {
                lines.push(format!("prompt_ms_stddev: {stddev:.3}"));
            }
            lines.push(format!("predicted_ms: {predicted_ms:.3}"));
            if let Some(stddev) = predicted_ms_stddev {
                lines.push(format!("predicted_ms_stddev: {stddev:.3}"));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};
    use rstest::rstest;

    use pipette_artifacts::ArtifactsContext;
    use pipette_plan_types::result::BenchmarkResultData;

    use super::*;
    use crate::model_uri::parse_model_arg;
    use crate::runtime_uri::parse_runtime_arg;
    use crate::workspace::test_support::TempWorkspace;

    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(flatten)]
        run: BenchmarkRunArgs,
    }

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test/0")?)
    }

    fn parse(argv: &[&str]) -> anyhow::Result<BenchmarkRunArgs> {
        let cli = TestCli::try_parse_from(std::iter::once("run").chain(argv.iter().copied()))?;
        Ok(cli.run)
    }

    /// Artifacts context whose `docker` / `uv` paths do not exist, so an install
    /// fails on spawn.
    ///
    /// Ensuring a real ref pair pulls a multi-gigabyte image or builds a venv. A
    /// dispatch test only needs to reach that arm — the real thing is covered by
    /// the `#[ignore]` install smokes under `crates/*/tests/`.
    ///
    /// Stubbing the two tools is enough only because `prepare` ensures the
    /// runtime before the model and both install arms resolve their tool before
    /// doing any work. Reorder either and the model fetch runs for real.
    fn stubbed_artifacts(tw: &TempWorkspace) -> anyhow::Result<ArtifactsContext> {
        let ctx = ArtifactsContext::new(test_http()?);
        let absent = tw.ws.root().join("no-such-tool");
        // A silently-ignored `set` would hand the test a real docker/uv path.
        // `set` returns the *rejected* value on failure, so report the occupant.
        ctx.docker_executable.set(absent.clone()).map_err(|_| {
            anyhow::anyhow!(
                "docker_executable pre-seeded to {:?}",
                ctx.docker_executable.get()
            )
        })?;
        ctx.uv_executable.set(absent).map_err(|_| {
            anyhow::anyhow!("uv_executable pre-seeded to {:?}", ctx.uv_executable.get())
        })?;
        Ok(ctx)
    }

    /// The spec names the body, so a caller handing over a different one is
    /// refused: the payload's id and the eval digest come from the body, so the
    /// run would otherwise record a cell it never executed.
    #[test]
    fn run_rejects_a_body_the_spec_does_not_name() -> anyhow::Result<()> {
        let tw = seeded_workspace("bench-body-mismatch")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "gguf-text://repo=o/r&path=m-Q4_0.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?;
        let (spec, _) = args.into_client_run_spec(&tw.ws)?;

        let other = tw
            .ws
            .benchmarks()
            .get(&"local/prefill_throughput_smoke".parse::<SourcedBenchmarkId>()?)?
            .ok_or_else(|| anyhow::anyhow!("a second benchmark should be seeded"))?;
        let err = run_cell(&spec, other, &stubbed_artifacts(&tw)?, &tw.ws)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a mismatched body should be refused"))?;

        let msg = format!("{err:#}");
        assert!(
            msg.contains("eval_smoke") && msg.contains("prefill_throughput_smoke"),
            "the error must name both ids; got: {msg}"
        );
        Ok(())
    }

    /// Drive a ref pair to the point where its runtime install would run, and
    /// assert it got there — i.e. it cleared the on-device bail in `prepare`
    /// rather than reaching an engine.
    fn assert_reaches_runtime_install(label: &str, argv: &[&str]) -> anyhow::Result<()> {
        let tw = seeded_workspace(label)?;
        let args = parse(argv)?;
        let (spec, benchmark) = args.into_client_run_spec(&tw.ws)?;
        let err = run_cell(&spec, benchmark, &stubbed_artifacts(&tw)?, &tw.ws)
            .err()
            .ok_or_else(|| anyhow::anyhow!("{label}: run should fail installing the runtime"))?;
        let msg = format!("{err:#}");
        anyhow::ensure!(
            msg.contains("ensuring runtime"),
            "{label}: expected a runtime-install failure, got: {msg}"
        );
        Ok(())
    }

    /// A fresh workspace seeded with the full standard local benchmark catalog.
    fn seeded_workspace(label: &str) -> anyhow::Result<TempWorkspace> {
        let tw = TempWorkspace::new(&format!("bench-{label}"))?;
        seed_standard_local(&tw.ws.benchmarks(), BenchmarkType::ALL)?;
        Ok(tw)
    }

    /// The summary leads with identity, then the type's headline metric — and
    /// an optional stddev only when present.
    #[test]
    fn run_summary_leads_with_identity_then_headline_metric() {
        let lines = run_summary_lines(
            "prefill_smoke",
            BenchmarkType::PrefillThroughput,
            &BenchmarkResultData::PrefillThroughput {
                prefill_time_ms: 12.5,
                prefill_time_ms_stddev: Some(0.25),
            },
        );
        assert_eq!(lines[0], "benchmark: prefill_smoke");
        assert_eq!(lines[1], "type: prefill_throughput");
        assert!(lines.iter().any(|l| l == "prefill_time_ms: 12.500000"));
        assert!(lines
            .iter()
            .any(|l| l == "prefill_time_ms_stddev: 0.250000"));
    }

    /// A missing optional stddev is omitted, not printed as `None`/`0`.
    #[test]
    fn run_summary_omits_absent_stddev() {
        let lines = run_summary_lines(
            "decode_smoke",
            BenchmarkType::DecodeThroughput,
            &BenchmarkResultData::DecodeThroughput {
                decode_time_ms: 3.0,
                decode_time_ms_stddev: None,
            },
        );
        assert!(lines.iter().any(|l| l == "decode_time_ms: 3.000000"));
        assert!(!lines.iter().any(|l| l.contains("stddev")));
    }

    /// Eval reports a completion count rather than a timing metric.
    #[test]
    fn run_summary_reports_eval_completion_count() {
        let lines = run_summary_lines(
            "ifbench",
            BenchmarkType::Eval,
            &BenchmarkResultData::Eval {
                completions: Vec::new(),
            },
        );
        assert!(lines.iter().any(|l| l == "completions: 0"));
    }

    /// `list` surfaces the seeded catalog as `local/<id>` rows.
    #[test]
    fn list_surfaces_seeded_local_catalog() -> anyhow::Result<()> {
        let tw = seeded_workspace("all")?;
        let rows = BenchmarkListArgs {
            benchmark_type: None,
        }
        .rows(&tw.ws)?;
        assert!(!rows.is_empty(), "seeded catalog should list benchmarks");
        assert!(rows.iter().all(|r| r.benchmark_ref.starts_with("local/")));
        Ok(())
    }

    /// `--benchmark-type` narrows the listing to a single kind.
    #[test]
    fn list_filters_by_benchmark_type() -> anyhow::Result<()> {
        let tw = seeded_workspace("filter")?;
        let rows = BenchmarkListArgs {
            benchmark_type: Some(BenchmarkType::DecodeThroughput),
        }
        .rows(&tw.ws)?;
        assert!(!rows.is_empty(), "catalog has decode_throughput cells");
        assert!(rows.iter().all(|r| r.benchmark_type == "decode_throughput"));
        Ok(())
    }

    #[test]
    fn parses_the_full_runner_emitted_surface() -> anyhow::Result<()> {
        let args = parse(&[
            "--benchmark",
            "remote/eval_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--runtime-flags",
            r#"[{"benchmark_type":"eval","runtime_type":"llamacpp_cli_stock_tools","model_type":"gguf_text","ctx_size":8192}]"#,
            "--http-timeout-seconds",
            "600",
            "--doomloop-exact-repeat-required",
            "3",
            "--model-flags",
            r#"[{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}]"#,
            "--sync",
        ])?;
        assert_eq!(args.benchmark_ref.to_string(), "remote/eval_smoke");
        assert_eq!(
            args.runtime,
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64"
        );
        assert_eq!(args.http_timeout_seconds, Some(600));
        assert!(args.model_flags.is_some());
        assert!(args.model_enable_thinking.is_none());
        assert!(args.sync);
        assert!(args.runtime_flags.is_some());
        Ok(())
    }

    /// A well-formed `(model, runtime)` pair parses and validates to a request —
    /// in both accepted ref forms. The URI case exercises the `scheme://` grammar;
    /// the JSON case exercises the `{`-routes-to-serde path, with the JSON produced
    /// by serializing the URI-parsed value so it can't drift from the serde shape.
    #[rstest::rstest]
    #[case::uri(false)]
    #[case::json(true)]
    fn valid_pair_parses_and_validates_to_a_request(#[case] as_json: bool) -> anyhow::Result<()> {
        let tw = seeded_workspace(if as_json {
            "valid-pair-json"
        } else {
            "valid-pair-uri"
        })?;
        let model_uri = "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf";
        let runtime_uri = "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64";
        let (model_ref, runtime_ref) = if as_json {
            (
                serde_json::to_string(&parse_model_arg(model_uri)?)?,
                serde_json::to_string(&parse_runtime_arg(runtime_uri)?)?,
            )
        } else {
            (model_uri.to_string(), runtime_uri.to_string())
        };
        if as_json {
            assert!(
                model_ref.starts_with('{') && runtime_ref.starts_with('{'),
                "expected JSON objects, got model={model_ref} runtime={runtime_ref}"
            );
        }
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            &model_ref,
            "--runtime",
            &runtime_ref,
        ])?;
        let (spec, _) = args.into_client_run_spec(&tw.ws)?;
        assert!(matches!(spec.runtime, Runtime::LlamacppCliStockTools(_)));
        assert!(matches!(spec.model, Model::GgufText(_)));
        Ok(())
    }

    /// The gating helper reconstructs the eval knobs into a typed
    /// `BenchmarkFlags` from the cell's authoritative benchmark TYPE.
    #[test]
    fn eval_run_reconstructs_benchmark_flags() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let doomloop = pipette_doomloop::plan::DoomloopOverrides {
            exact_repeat: Some(pipette_doomloop::plan::ExactRepeatOverrides {
                required: Some(3),
                ..Default::default()
            }),
            ..Default::default()
        };
        let bf = build_benchmark_flags(
            BenchmarkType::Eval,
            &runtime,
            &model,
            Some(600),
            doomloop,
            None,
            false,
        )?
        .ok_or_else(|| anyhow::anyhow!("benchmark_flags present for an eval run"))?;
        assert_eq!(bf.http_timeout(), Some(600));
        assert!(bf.doomloop().is_some(), "eval carries a doom-loop");
        assert_eq!(bf.axes().0, BenchmarkType::Eval);
        Ok(())
    }

    // `--http-timeout-seconds` needs a server-driven benchmark (eval/vl), so a
    // non-server type (prefill/decode) is rejected, never silently dropped.
    #[rstest::rstest]
    #[case(BenchmarkType::PrefillThroughput)]
    #[case(BenchmarkType::DecodeThroughput)]
    fn timeout_off_a_server_benchmark_is_rejected(
        #[case] benchmark_type: BenchmarkType,
    ) -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let err = build_benchmark_flags(
            benchmark_type,
            &runtime,
            &model,
            Some(600),
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            false,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            format!("{err:#}").contains("not valid for this benchmark"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// No eval knob set carries `None` regardless of benchmark type.
    #[test]
    fn no_knob_carries_no_benchmark_flags() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            false,
        )?;
        assert!(bf.is_none());
        Ok(())
    }

    /// `--http-timeout-seconds` is valid on vl-throughput (it drives
    /// `llama-server`), producing flags with a timeout and no doom-loop.
    #[test]
    fn timeout_on_vl_throughput_is_accepted() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg(
            "gguf-vision://repo=unsloth/gemma-4-E2B-it-GGUF&model=m.gguf&mmproj=mmproj.gguf",
        )?;
        let bf = build_benchmark_flags(
            BenchmarkType::VlThroughput,
            &runtime,
            &model,
            Some(600),
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            false,
        )?
        .ok_or_else(|| anyhow::anyhow!("vl-throughput accepts an http timeout"))?;
        assert_eq!(bf.http_timeout(), Some(600));
        assert!(bf.doomloop().is_none(), "vl runs no doom-loop");
        assert_eq!(bf.axes().0, BenchmarkType::VlThroughput);
        Ok(())
    }

    /// `--readiness-max-wait-secs` is honored on a timing cell — it crosses a
    /// readiness-carrying `(benchmark, runtime, model)` variant.
    #[test]
    fn readiness_on_timing_cell_is_accepted() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            Some(1800),
            false,
        )?
        .ok_or_else(|| anyhow::anyhow!("prefill accepts a readiness override"))?;
        assert_eq!(bf.readiness().and_then(|r| r.max_wait_secs), Some(1800));
        Ok(())
    }

    /// `--readiness-max-wait-secs` on eval is rejected — the conjunction doesn't
    /// cross a readiness-carrying cell (eval gates on server, not device).
    #[test]
    fn readiness_off_a_timing_benchmark_is_rejected() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let err = build_benchmark_flags(
            BenchmarkType::Eval,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            Some(1800),
            false,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            format!("{err:#}").contains("not valid for this benchmark"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// `--readiness-skip-thermal` on its own is enough to build a readiness
    /// override — it must not need `--readiness-max-wait-secs` alongside it,
    /// and it must not invent a deadline it wasn't given.
    #[test]
    fn readiness_skip_thermal_alone_is_accepted_on_a_timing_cell() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            true,
        )?
        .ok_or_else(|| anyhow::anyhow!("prefill accepts a thermal waiver on its own"))?;
        let readiness = bf
            .readiness()
            .ok_or_else(|| anyhow::anyhow!("expected a readiness override"))?;
        assert_eq!(readiness.skip_thermal, Some(true));
        assert_eq!(
            readiness.max_wait_secs, None,
            "waiving the thermal gate must not fabricate a deadline",
        );
        Ok(())
    }

    /// `false` is not an override. Passing it must leave the cell with no
    /// readiness knob at all, so an unused flag can't make a default-gated run
    /// look like one that opted into something.
    #[test]
    fn readiness_skip_thermal_false_records_no_override() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let flags = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            false,
        )?;
        assert!(
            flags.is_none_or(|bf| bf.readiness().is_none()),
            "an unset thermal waiver must not create a readiness override",
        );
        Ok(())
    }

    /// The waiver is rejected on the same benchmarks the deadline is: eval gates
    /// on the server, not the device, so there is no thermal criterion to waive.
    #[test]
    fn readiness_skip_thermal_off_a_timing_benchmark_is_rejected() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?;
        let model = parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?;
        let err = build_benchmark_flags(
            BenchmarkType::Eval,
            &runtime,
            &model,
            None,
            pipette_doomloop::plan::DoomloopOverrides::default(),
            None,
            true,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            format!("{err:#}").contains("not valid for this benchmark"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// Plan `ModelFlags` only exist on eval cells — non-eval +
    /// `--model-enable-thinking` is rejected at the CLI boundary (not stored
    /// then stripped on submit as the old flat ops flags did).
    #[test]
    fn model_enable_thinking_on_non_eval_is_rejected() -> anyhow::Result<()> {
        let tw = seeded_workspace("enable-thinking-non-eval")?;
        let args = parse(&[
            "--benchmark",
            "local/prefill_throughput_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--model-enable-thinking",
            "true",
        ])?;
        let err = args
            .into_client_run_spec(&tw.ws)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected non-eval enable_thinking to fail"))?;
        assert!(
            format!("{err:#}").contains("not valid for this benchmark"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// `--runtime-flags` is a JSON object of knobs, and nothing else.
    ///
    /// The cell comes from `--benchmark`, `--runtime` and `--model`. Frozen alongside the
    /// refusals below so the single-format rule cannot erode back into "accepts whatever
    /// happens to parse".
    #[test]
    fn runtime_flags_is_an_object_of_knobs() -> anyhow::Result<()> {
        let tw = seeded_workspace("rt-flags-ok")?;
        let (spec, _) = prefill_args(r#"{"threads":4}"#)?.into_client_run_spec(&tw.ws)?;

        let flags = spec
            .runtime_flags
            .ok_or_else(|| anyhow::anyhow!("expected flags"))?;
        assert!(
            flags.submission_value()["threads"] == 4,
            "got: {:?}",
            flags.submission_value()
        );
        Ok(())
    }

    /// An empty object is "no flags", not an error — the runner emits nothing at all in
    /// that case, but a hand-written `{}` should mean the same thing.
    #[test]
    fn an_empty_runtime_flags_object_is_no_flags() -> anyhow::Result<()> {
        let tw = seeded_workspace("rt-flags-empty")?;
        let (spec, _) = prefill_args("{}")?.into_client_run_spec(&tw.ws)?;

        assert!(spec.runtime_flags.is_none());
        Ok(())
    }

    /// What `--runtime-flags` refuses, and that the refusal names the offence. Each of
    /// these used to be expressible: an entry for another cell would have been applied to
    /// this one, and a knob the cell does not carry would have been dropped.
    #[rstest]
    #[case::any_array(r#"[{"threads":4}]"#, "object of knobs")]
    #[case::multiple_entries(r#"[{"threads":1},{"threads":2}]"#, "object of knobs")]
    #[case::carries_a_contradicting_axis(
        r#"{"benchmark_type":"eval","threads":4}"#,
        "must not carry"
    )]
    // Refused even when it agrees: the cell is derived, so restating it is never the
    // caller's to do, and "it happened to match" is not a rule anyone can rely on.
    #[case::carries_a_matching_axis(
        r#"{"runtime_type":"llamacpp_cli_stock_tools","threads":4}"#,
        "must not carry"
    )]
    #[case::knob_this_cell_does_not_carry(r#"{"max_model_len":4096}"#, "max_model_len")]
    #[case::malformed_json(r#"{"threads":"#, "parsing")]
    #[case::not_an_object(r#""threads=4""#, "object of knobs")]
    fn runtime_flags_refuses_and_names_the_offence(
        #[case] json: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let tw = seeded_workspace("rt-flags-bad")?;
        let Err(err) = prefill_args(json)?.into_client_run_spec(&tw.ws) else {
            anyhow::bail!("{json} should have been refused");
        };
        // `{:#}` walks the source chain: the naming happens in the error `context` wraps.
        let chain = format!("{err:#}");
        assert!(
            chain.contains(expected),
            "expected {expected:?}, got: {chain}"
        );
        Ok(())
    }

    /// The plan and the CLI agree on what a cell is.
    ///
    /// A plan authors `runtime_flags` keyed by `(benchmark, runtime, model)`; the runner
    /// resolves one entry for the cell and ships its knobs; the client re-derives the axes
    /// from the same three arguments. This walks that whole path — author, resolve, emit,
    /// parse — and asserts the client ends up holding exactly what the plan authored. It
    /// is the test that would fail if either side changed the wire alone.
    #[test]
    fn a_plans_cell_definition_survives_the_trip_to_the_client() -> anyhow::Result<()> {
        use pipette_plan_types::Plan;

        let plan = Plan::parse(
            r#"
plan_id    = "flags"
benchmarks = ["prefill_throughput_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "unsloth", repo_name = "gemma-4-E2B-it-GGUF", path = "g.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4 }]
"#,
        )?;
        let cells = plan.runnable_cells()?;
        let cell = cells
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no cells"))?;
        let wire = cell
            .runtime_flags_json()
            .ok_or_else(|| anyhow::anyhow!("plan authored flags but emitted none"))?;

        let tw = seeded_workspace("plan-cell")?;
        let (spec, _) = prefill_args(&wire)?.into_client_run_spec(&tw.ws)?;

        assert_eq!(spec.runtime_flags, cell.runtime_flags);
        Ok(())
    }

    /// A `prefill_throughput` run over the gguf-text × llama.cpp cell, varying only the
    /// flags — the pairing every `--runtime-flags` case above is resolved against.
    fn prefill_args(runtime_flags: &str) -> anyhow::Result<BenchmarkRunArgs> {
        parse(&[
            "--benchmark",
            "local/prefill_throughput_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--runtime-flags",
            runtime_flags,
        ])
    }

    #[test]
    fn runtime_flags_json_parses_with_raw_inside() -> anyhow::Result<()> {
        let tw = seeded_workspace("runtime-flags")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--runtime-flags",
            r#"{"ctx_size":8192,"raw":["--foo","bar"]}"#,
        ])?;
        let (spec, _) = args.into_client_run_spec(&tw.ws)?;
        // The `raw` escape hatch rides inside the one RuntimeFlags entry — there
        // is no separate raw channel.
        assert!(spec.runtime_flags.is_some());
        Ok(())
    }

    /// `--model-flags` accepts the plan's one-element array and the bare object a claim
    /// carries — the same two-way tolerance `--runtime-flags` has. Frozen here so the
    /// spellings a deployed client relies on cannot be dropped by accident.
    #[rstest]
    #[case::bare_object(
        r#"{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}"#
    )]
    #[case::one_element_array(
        r#"[{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}]"#
    )]
    fn model_flags_accepts_both_spellings(#[case] json: &str) -> anyhow::Result<()> {
        let tw = seeded_workspace("model-flags-ok")?;
        let (spec, _) = eval_args(&["--model-flags", json])?.into_client_run_spec(&tw.ws)?;

        assert_eq!(
            spec.model_flags.as_ref().and_then(|f| f.enable_thinking()),
            Some(true)
        );
        Ok(())
    }

    /// The refusals, each naming what it refused. A model-flags entry authored for another
    /// cell would otherwise be applied to this one, and two ways of asking for the same
    /// setting would silently pick a winner.
    #[rstest]
    #[case::multiple_entries(
        &["--model-flags", r#"[{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true},{"benchmark_type":"eval","model_type":"mlx","enable_thinking":false}]"#],
        "at most one"
    )]
    #[case::entry_for_another_cell(
        &["--model-flags", r#"{"benchmark_type":"eval","model_type":"mlx","enable_thinking":true}"#],
        "does not match this cell"
    )]
    #[case::both_spellings_at_once(
        &["--model-flags", r#"{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}"#,
          "--model-enable-thinking", "true"],
        "mutually exclusive"
    )]
    fn model_flags_refuses_and_names_the_offence(
        #[case] extra: &[&str],
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let tw = seeded_workspace("model-flags-bad")?;
        let Err(err) = eval_args(extra)?.into_client_run_spec(&tw.ws) else {
            anyhow::bail!("{extra:?} should have been refused");
        };

        let chain = format!("{err:#}");
        assert!(
            chain.contains(expected),
            "expected {expected:?}, got: {chain}"
        );
        Ok(())
    }

    /// An eval run over the gguf-text × llama.cpp cell — the pairing the model-flags cases
    /// above are resolved against, since only eval cells carry generation flags.
    fn eval_args(extra: &[&str]) -> anyhow::Result<BenchmarkRunArgs> {
        let mut argv = vec![
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ];
        argv.extend_from_slice(extra);
        parse(&argv)
    }

    #[test]
    fn model_flags_json_parses_into_spec() -> anyhow::Result<()> {
        let tw = seeded_workspace("model-flags")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--model-flags",
            r#"[{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}]"#,
        ])?;
        let (spec, _) = args.into_client_run_spec(&tw.ws)?;
        assert_eq!(
            spec.model_flags.as_ref().and_then(|f| f.enable_thinking()),
            Some(true)
        );
        Ok(())
    }

    #[test]
    fn model_incompatible_with_the_runtime_is_rejected() -> anyhow::Result<()> {
        // An MLX model on a llama.cpp runtime is a well-formed ref pair that fails
        // the `is_compatible` gate — rejected before benchmark resolution or any
        // download, so the workspace need not be seeded.
        let tw = TempWorkspace::new("bench-model-wrong")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "mlx://repo=meta-llama/Llama-3.2-1B",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?;
        assert!(args.into_client_run_spec(&tw.ws).is_err());
        Ok(())
    }

    /// An on-device runtime can't run from the CLI: it clears `is_compatible`
    /// and reaches dispatch, where its arm bails. Shown with JSON refs since
    /// on-device runtimes have no URI scheme.
    #[test]
    fn run_rejects_on_device_runtime() -> anyhow::Result<()> {
        // Seed the catalog so the benchmark resolves and the run reaches dispatch.
        let tw = seeded_workspace("run-dispatch")?;
        let err = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            r#"{"type":"apple_foundation_text"}"#,
            "--runtime",
            r#"{"type":"apple_foundation"}"#,
        ])?
        .execute(&tw.ws, &test_http()?)
        .err()
        .ok_or_else(|| anyhow::anyhow!("run should error for an on-device runtime"))?;
        assert!(
            format!("{err:#}").contains("not a desktop CLI runtime"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// A vLLM/SGLang ref pair clears the on-device bail and reaches artifact
    /// ensure, where the torch-oai install arm lives. It stops there — `prepare`
    /// fails before any engine runs.
    ///
    /// Runs against [`stubbed_artifacts`]: the docker arm of that ensure is
    /// `docker pull vllm/vllm-openai:v0.21.0`, a real registry pull.
    #[test]
    fn torch_oai_reaches_runtime_install() -> anyhow::Result<()> {
        let model = Model::Torch(pipette_plan_types::Torch {
            source: pipette_plan_types::ModelSource::AbsoluteDir {
                dir: pipette_plan_types::AbsolutePath::try_new("/models/torch-local".to_string())?,
            },
        });
        let runtime = Runtime::DockerVllm(pipette_plan_types::DockerVllm {
            image_name: pipette_plan_types::NonEmptyString::try_new(
                "vllm/vllm-openai".to_string(),
            )?,
            image_tag: pipette_plan_types::NonEmptyString::try_new("v0.21.0".to_string())?,
            flavor: pipette_plan_types::VllmFlavor::Cpu,
        });
        assert_reaches_runtime_install(
            "torch-dispatch",
            &[
                "--benchmark",
                "local/eval_smoke",
                "--model",
                &serde_json::to_string(&model)?,
                "--runtime",
                &serde_json::to_string(&runtime)?,
            ],
        )
    }

    /// Benchmark resolution now happens in `into_client_run_spec`, before dispatch:
    /// on a fresh (unseeded) workspace an absent benchmark fails there with
    /// "unknown benchmark reference", without any download or run.
    #[test]
    fn llama_arm_dispatches_into_run() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("bench-run-llama")?;
        let err = parse(&[
            "--benchmark",
            "remote/does-not-exist",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?
        .execute(&tw.ws, &test_http()?)
        .err()
        .ok_or_else(|| anyhow::anyhow!("run should fail resolving an absent benchmark"))?;
        assert!(
            format!("{err:#}").contains("unknown benchmark reference"),
            "expected benchmark-resolution failure from run, got: {err:#}"
        );
        Ok(())
    }

    /// An MLX ref pair resolves its catalog URI and reaches artifact ensure,
    /// where the MLX install arm lives. macOS-only — the MLX crate compiles away
    /// on other hosts.
    ///
    /// Runs against [`stubbed_artifacts`]: ensuring an MLX runtime builds a uv
    /// venv, which a unit test has no business doing.
    #[cfg(target_os = "macos")]
    #[test]
    fn mlx_reaches_runtime_install() -> anyhow::Result<()> {
        assert_reaches_runtime_install(
            "mlx-dispatch",
            &[
                "--benchmark",
                "local/eval_smoke",
                "--model",
                "mlx://repo=mlx-community/does-not-matter",
                "--runtime",
                // Bundled catalog version — the URI must resolve the full requirements text.
                "mlx-macos-pipette://version=0.31.3",
            ],
        )
    }

    /// Regression for PIP-409: the plan runner's `build_argv` must emit a
    /// `--model` the client can parse. It ships the model as JSON (projector
    /// inline for VL, no `--mmproj`); a flat `o/r:a.gguf` would be rejected by
    /// `parse_model_arg`.
    #[test]
    fn build_argv_model_parses_through_parse_model_arg() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["eval_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_vision", source = "huggingface", org = "o", repo_name = "r", model = "a.gguf", mmproj = "m.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
"#;
        let plan = pipette_plan_types::Plan::parse(toml_str)?;
        let transport = plan
            .transports
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("one transport"))?;
        let cells = plan.runnable_cells()?;
        let cell = cells
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("one cell"))?;
        let argv = cell.build_argv(transport)?;
        let model_idx = argv
            .iter()
            .position(|a| a == "--model")
            .ok_or_else(|| anyhow::anyhow!("--model present"))?;
        // The core PIP-409 assertion: this parses (a flat ref would `Err`), and
        // the projector rode inside the model ref — no `--mmproj`.
        let model = parse_model_arg(&argv[model_idx + 1])?;
        assert!(matches!(model, Model::GgufVision(_)));
        assert!(!argv.iter().any(|a| a == "--mmproj"));
        Ok(())
    }

    /// Every value the help block passes to `--<flag>`, quoted (`--model 'x'`)
    /// or bare (`--benchmark x`).
    ///
    /// Comment lines are skipped: the prose there names flags too, and a
    /// mention is not an example. The count is checked against the command
    /// lines so a reflowed example cannot silently stop being verified.
    fn help_values<'a>(help: &'a str, flag: &str) -> Vec<&'a str> {
        let needle = format!("--{flag} ");
        let commands = || {
            help.lines()
                .filter(|line| !line.trim_start().starts_with('#'))
        };
        let values: Vec<&str> = commands()
            .filter_map(|line| line.split_once(&needle))
            .map(|(_, rest)| match rest.strip_prefix('\'') {
                Some(quoted) => quoted.split_once('\'').map_or(quoted, |(value, _)| value),
                None => rest.split_whitespace().next().unwrap_or(rest),
            })
            .collect();
        assert_eq!(
            values.len(),
            commands()
                .map(|line| line.matches(&needle).count())
                .sum::<usize>(),
            "--{flag}: a value was not captured; is one split across lines?"
        );
        values
    }

    /// Every ref quoted in `benchmarks run --help` must parse, so the examples
    /// cannot drift from the grammar.
    #[test]
    fn after_help_refs_parse() -> anyhow::Result<()> {
        let models = help_values(RUN_AFTER_HELP, "model");
        let runtimes = help_values(RUN_AFTER_HELP, "runtime");
        assert!(
            models.len() >= 8,
            "expected >= 8 --model examples, found {}",
            models.len()
        );
        assert!(
            runtimes.len() >= 8,
            "expected >= 8 --runtime examples, found {}",
            runtimes.len()
        );

        // Digest refs resolve against an installed store, not the grammar.
        let failures: Vec<String> = models
            .iter()
            .filter(|u| !u.starts_with("model://"))
            .filter_map(|uri| {
                parse_model_arg(uri)
                    .err()
                    .map(|e| format!("--model `{uri}`: {e}"))
            })
            .chain(
                runtimes
                    .iter()
                    .filter(|u| !u.starts_with("runtime://"))
                    .filter_map(|uri| {
                        parse_runtime_arg(uri)
                            .err()
                            .map(|e| format!("--runtime `{uri}`: {e}"))
                    }),
            )
            .collect();
        assert!(
            failures.is_empty(),
            "help examples must parse:\n{}",
            failures.join("\n")
        );
        Ok(())
    }

    /// Every benchmark the help names must exist in the seeded local catalog.
    /// A fabricated id shipped in `--help` once; this is what catches the next.
    #[test]
    fn after_help_benchmark_ids_exist() -> anyhow::Result<()> {
        let tw = seeded_workspace("after-help-ids")?;
        let store = tw.ws.benchmarks();
        let refs = help_values(RUN_AFTER_HELP, "benchmark");
        assert!(
            refs.len() >= 6,
            "expected >= 6 --benchmark examples, found {}",
            refs.len()
        );

        let missing = refs
            .iter()
            .map(|r| Ok((*r, store.get(&r.parse::<SourcedBenchmarkId>()?)?.is_some())))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, found)| !found)
            .map(|(r, _)| r)
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "help names benchmarks that do not exist: {missing:?}"
        );
        Ok(())
    }

    /// `long_about` on an `Args` struct is silently ignored, so assert the
    /// rendered help actually carries it.
    #[test]
    fn run_long_about_renders() -> anyhow::Result<()> {
        let help = crate::commands::Cli::command()
            .find_subcommand_mut("benchmarks")
            .and_then(|b| b.find_subcommand_mut("run").map(|r| r.render_long_help()))
            .ok_or_else(|| anyhow::anyhow!("benchmarks run subcommand should exist"))?
            .to_string();
        assert!(
            help.contains("A run is one cell"),
            "benchmarks run --help must render RUN_LONG_ABOUT; got:\n{help}"
        );
        Ok(())
    }

    /// A probe value per settable `--runtime-flags` field, used to ask the real
    /// parser which ones a cell accepts. Kept honest by
    /// `probe_table_covers_every_settable_field`.
    const KNOB_PROBES: &[(&str, &str)] = &[
        ("threads", "4"),
        ("number_gpu_layers", "99"),
        ("mmap", "true"),
        ("flash_attention", "\"on\""),
        ("ctx_size", "4096"),
        ("n_ubatch", "512"),
        ("swa_full", "true"),
        ("no_cache", "true"),
        ("tensor_parallel_size", "2"),
        ("dtype", "\"bfloat16\""),
        ("max_model_len", "4096"),
        ("device", "\"cpu\""),
        ("max_prompt_len", "1024"),
        ("min_response_len", "128"),
        ("generate_hint", "\"best-perf\""),
        ("prefix_caching", "false"),
        ("gpus", "\"all\""),
        ("shm_size", "\"16g\""),
        ("ipc", "\"host\""),
        ("envs", "[\"NCCL_DEBUG\"]"),
        ("raw", "[\"--some-passthrough\"]"),
    ];

    /// The settings the real parser accepts for one cell.
    fn accepted_knobs(
        runtime: RuntimeType,
        model: ModelType,
        benchmark: BenchmarkType,
    ) -> Vec<&'static str> {
        KNOB_PROBES
            .iter()
            .filter(|(knob, probe)| {
                RuntimeFlags::from_cell_json(
                    &format!("{{\"{knob}\": {probe}}}"),
                    runtime,
                    model,
                    benchmark,
                )
                .is_ok()
            })
            .map(|(knob, _)| *knob)
            .collect()
    }

    /// Serde names every valid field when it rejects an unknown one, so the
    /// probe table can be checked against the struct itself.
    #[test]
    fn probe_table_covers_every_settable_field() -> anyhow::Result<()> {
        let err = RuntimeFlags::from_cell_json(
            "{\"__not_a_knob__\": 1}",
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
            BenchmarkType::PrefillThroughput,
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("an unknown field should be refused"))?;

        let message = format!("{err:#}");
        let (_, listed) = message
            .split_once("expected one of ")
            .ok_or_else(|| anyhow::anyhow!("serde should list the valid fields; got: {message}"))?;
        let mut settable: Vec<&str> = listed
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|f| !matches!(*f, "runtime_type" | "model_type" | "benchmark_type"))
            .collect();
        settable.sort_unstable();
        settable.dedup();

        let mut probed: Vec<&str> = KNOB_PROBES.iter().map(|(knob, _)| *knob).collect();
        probed.sort_unstable();
        assert_eq!(
            probed, settable,
            "KNOB_PROBES must cover exactly the settable fields"
        );
        Ok(())
    }

    /// The CELL SETTINGS block in `--help` must match what the parser accepts.
    ///
    /// The `documented` list is the load-bearing check: it is compared against
    /// the real parser, so a wrong row fails. The block is then searched for
    /// each name, which only catches a setting missing from the help *entirely*
    /// (a name dropped from one row still appears in another). Row-level
    /// precision would mean parsing the layout, which breaks on any reflow.
    #[rstest]
    #[case::llama_bench(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufText, BenchmarkType::PrefillThroughput,
        &["threads", "number_gpu_layers", "mmap", "flash_attention", "raw"]
    )]
    #[case::llama_server(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufText, BenchmarkType::Eval,
        &["threads", "number_gpu_layers", "mmap", "flash_attention", "ctx_size", "no_cache", "raw"]
    )]
    #[case::llama_vl(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufVision, BenchmarkType::VlThroughput,
        &["threads", "number_gpu_layers", "mmap", "flash_attention", "ctx_size", "no_cache", "raw"]
    )]
    #[case::docker_vllm(
        RuntimeType::DockerVllm, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "gpus", "shm_size",
          "ipc", "envs", "raw"]
    )]
    #[case::uv_vllm(
        RuntimeType::UvVllm, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "envs", "raw"]
    )]
    #[case::uv_sglang(
        RuntimeType::UvSglang, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "prefix_caching", "envs", "raw"]
    )]
    #[case::openvino(
        RuntimeType::UvOpenvino, ModelType::Openvino, BenchmarkType::PrefillThroughput,
        &["device", "max_prompt_len", "min_response_len", "generate_hint"]
    )]
    #[case::mlx(RuntimeType::MlxMacosPipette, ModelType::Mlx, BenchmarkType::Eval, &[])]
    fn help_cell_settings_match_the_parser(
        #[case] runtime: RuntimeType,
        #[case] model: ModelType,
        #[case] benchmark: BenchmarkType,
        #[case] documented: &[&str],
    ) -> anyhow::Result<()> {
        let mut accepted = accepted_knobs(runtime, model, benchmark);
        accepted.sort_unstable();
        let mut expected = documented.to_vec();
        expected.sort_unstable();
        assert_eq!(
            accepted, expected,
            "{runtime:?} x {model:?} on {benchmark:?}: help documents {expected:?}, parser accepts {accepted:?}"
        );

        let settings = RUN_AFTER_HELP
            .split_once("CELL SETTINGS")
            .map(|(_, tail)| tail)
            .ok_or_else(|| anyhow::anyhow!("--help must carry a CELL SETTINGS block"))?;
        let unlisted: Vec<_> = documented
            .iter()
            .filter(|knob| !settings.contains(**knob))
            .collect();
        assert!(
            unlisted.is_empty(),
            "CELL SETTINGS omits {unlisted:?} for {runtime:?} x {model:?}"
        );
        Ok(())
    }

    /// Whether a flag's JSON carries the `(benchmark, runtime, model)` axes.
    /// `--runtime-flags` derives them from the cell and rejects them; the
    /// plan-form `--model-flags` requires them.
    #[derive(Clone, Copy, Debug)]
    enum Axes {
        Forbidden,
        Required,
    }

    /// The flag JSON in the help must be well-formed and carry exactly the axis
    /// keys its flag accepts.
    #[rstest]
    #[case::runtime_flags(
        "runtime-flags",
        &["runtime_type", "model_type", "benchmark_type"],
        Axes::Forbidden,
        4
    )]
    #[case::model_flags("model-flags", &["model_type", "benchmark_type"], Axes::Required, 1)]
    fn after_help_flag_json_is_well_formed(
        #[case] flag: &str,
        #[case] axes: &[&str],
        #[case] rule: Axes,
        #[case] min_examples: usize,
    ) -> anyhow::Result<()> {
        let raws = help_values(RUN_AFTER_HELP, flag);
        assert!(
            raws.len() >= min_examples,
            "--{flag}: expected >= {min_examples} examples, found {}",
            raws.len()
        );

        raws.iter().try_for_each(|raw| {
            let value: serde_json::Value =
                serde_json::from_str(raw).with_context(|| format!("--{flag} `{raw}`"))?;
            let obj = value
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("--{flag} `{raw}` must be an object"))?;
            let offender = match rule {
                Axes::Forbidden => axes.iter().find(|axis| obj.contains_key(**axis)),
                Axes::Required => axes.iter().find(|axis| !obj.contains_key(**axis)),
            };
            match (offender, rule) {
                (None, _) => Ok(()),
                (Some(axis), Axes::Forbidden) => anyhow::bail!(
                    "--{flag} `{raw}` carries the axis key `{axis}`, which the CLI rejects"
                ),
                (Some(axis), Axes::Required) => {
                    anyhow::bail!("--{flag} `{raw}` is missing the required axis key `{axis}`")
                }
            }
        })
    }
}
