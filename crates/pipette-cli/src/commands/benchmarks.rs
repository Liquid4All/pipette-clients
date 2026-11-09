//! `benchmarks`: the catalog commands (`list`, `show`, `init-local`) and `run`,
//! which builds a [`ClientRunSpec`] and hands it to [`crate::run::run_cell`].
//! See `docs/architecture.md` ("Benchmark vs run vs execute").

use std::io::Write;

use anyhow::Context;
use clap::{Args, Subcommand};
use tabled::Tabled;

use pipette_http::HttpClient;
use pipette_ops::eval_completions::EvalRunDigest;
use pipette_plan_types::{
    AsrDecoding, AsrEvalBenchmark, AudioChannels, AudioDurationMillis, AudioSampleRate, BatchSize,
    BenchmarkDefinition, BenchmarkFlagRef, BenchmarkFlags, BenchmarkId, BenchmarkResultData,
    BenchmarkSubmissionPayload, BenchmarkType, ClientRunSpec, EvalBenchmark, Model, ModelFlagRef,
    ModelFlags, ModelType, NonNegativeCount, PositiveCount, PositiveSeconds, ReadinessOverrides,
    Runtime, RuntimeFlags, RuntimeType, SamplingTemperature, SeedRule, SkinCelsius,
};
use pipette_programs::ExternalPrograms;

use crate::artifact_ref::{resolve_model_arg, resolve_runtime_arg};
use crate::benchmarks::{SourcedBenchmarkId, seed_standard_local};
use crate::client_configuration::ClientConfiguration;
use crate::commands::write_table_or;
use crate::console::Console;
use crate::hf_auth::inject_env_hf_token;
use crate::results::{
    BenchmarkResultExtras, BenchmarkResultLocation, RecordSubmitOutcome,
    record_and_maybe_submit_run,
};
use crate::run::{finalize_eval_checkpoint, refuse_foreign_os, run_cell};
use crate::workspace::PipetteWorkspace;

/// List, inspect, and run benchmarks
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub command: BenchmarkCommand,
}

#[derive(Subcommand, Debug)]
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
    /// vlm-eval, vl-throughput, vl-max-memory, vl-spec-decode, asr-throughput,
    /// asr-max-memory (the underscore spellings are accepted too)
    #[arg(long)]
    pub benchmark_type: Option<BenchmarkType>,
}

#[derive(Args, Debug)]
pub struct BenchmarkShowArgs {
    /// Benchmark reference: `<id>` or `remote/<id>` (synced catalog), or
    /// `local/<id>` for a definition only this machine has
    pub benchmark_ref: SourcedBenchmarkId,
}

/// One row of `benchmarks list`: the reference to pass to `run`/`show`, and its
/// kind.
#[derive(Tabled)]
struct BenchmarkRow {
    #[tabled(rename = "BENCHMARK")]
    benchmark_ref: String,
    #[tabled(rename = "TYPE")]
    benchmark_type: String,
}

impl BenchmarkListArgs {
    fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
        write_table_or(console.out(), &self.rows(ws)?, "No benchmarks.")?;
        Ok(())
    }

    /// The catalog rows honoring `--benchmark-type`, split out so the filtering
    /// is testable without capturing stdout.
    fn rows(&self, ws: &PipetteWorkspace) -> anyhow::Result<Vec<BenchmarkRow>> {
        use pipette_plan_types::BenchmarkSource;

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

fn resolve_benchmark(
    ws: &PipetteWorkspace,
    reference: &SourcedBenchmarkId,
) -> anyhow::Result<BenchmarkDefinition> {
    ws.benchmarks()
        .get(reference)?
        .ok_or_else(|| anyhow::anyhow!("unknown benchmark reference `{reference}`"))
}

impl BenchmarkShowArgs {
    fn execute(self, ws: &PipetteWorkspace, console: &mut dyn Console) -> anyhow::Result<()> {
        let reference = self.benchmark_ref;
        let def = resolve_benchmark(ws, &reference)?;
        write_benchmark(console.out(), &def, reference.source())
    }
}

/// What `show` prints: the identity, then the parameters of the variant.
fn write_benchmark(
    out: &mut dyn Write,
    def: &pipette_plan_types::BenchmarkDefinition,
    source: pipette_plan_types::BenchmarkSource,
) -> anyhow::Result<()> {
    use pipette_plan_types::BenchmarkDefinition;
    writeln!(out, "id:      {}", def.benchmark_id())?;
    writeln!(out, "type:    {}", def.benchmark_type())?;
    writeln!(
        out,
        "source:  {}",
        match source {
            pipette_plan_types::BenchmarkSource::Local => "local",
            pipette_plan_types::BenchmarkSource::Remote => "remote",
        }
    )?;
    match def {
        BenchmarkDefinition::PrefillThroughput(b) => {
            writeln!(out, "prefill: {} tokens", b.parameter_prefill_tokens)?;
            write_batch_size(out, b.parameter_batch_size)?;
        }
        BenchmarkDefinition::DecodeThroughput(b) => {
            writeln!(out, "prefill: {} tokens", b.parameter_prefill_tokens)?;
            writeln!(out, "decode:  {} tokens", b.parameter_decode_tokens)?;
            write_batch_size(out, b.parameter_batch_size)?;
        }
        BenchmarkDefinition::EndToEndLatency(b) => {
            writeln!(out, "prefill: {} tokens", b.parameter_prefill_tokens)?;
            writeln!(out, "decode:  {} tokens", b.parameter_decode_tokens)?;
            write_batch_size(out, b.parameter_batch_size)?;
        }
        BenchmarkDefinition::MaxMemoryUsage(b) => {
            writeln!(out, "prefill: {} tokens", b.parameter_prefill_tokens)?;
        }
        BenchmarkDefinition::Eval(b) => {
            writeln!(out, "eval:    {}", b.parameter_eval_id)?;
            writeln!(out, "dataset: {}", b.parameter_dataset_name)?;
            writeln!(out, "max_tok: {}", b.parameter_max_tokens)?;
            if let Some(choices) = &b.parameter_mcq_choices {
                writeln!(out, "mcq:     {}", choices.join(", "))?;
            }
            write_eval_sampling(out, b)?;
            if let Some(samples) = &b.samples {
                writeln!(out, "samples: {}", samples.len())?;
            }
        }
        BenchmarkDefinition::AsrMaxMemory(b) => {
            write_asr_clip(
                out,
                b.parameter_audio_duration_ms,
                b.parameter_audio_sample_rate_hz,
                b.parameter_audio_channels,
                b.parameter_text_tokens,
            )?;
            writeln!(out, "decode:  {} tokens", b.parameter_decode_tokens)?;
        }
        BenchmarkDefinition::AsrThroughput(b) => {
            write_asr_clip(
                out,
                b.parameter_audio_duration_ms,
                b.parameter_audio_sample_rate_hz,
                b.parameter_audio_channels,
                b.parameter_text_tokens,
            )?;
            writeln!(out, "decode:  {} tokens", b.parameter_decode_tokens)?;
        }
        BenchmarkDefinition::VlThroughput(b) => write_vl_shape(
            out,
            b.parameter_image_width,
            b.parameter_image_height,
            b.parameter_text_tokens,
            b.parameter_num_images,
            Some(b.parameter_decode_tokens),
        )?,
        BenchmarkDefinition::VlMaxMemory(b) => write_vl_shape(
            out,
            b.parameter_image_width,
            b.parameter_image_height,
            b.parameter_text_tokens,
            b.parameter_num_images,
            b.parameter_decode_tokens,
        )?,
        BenchmarkDefinition::VlmEval(b) => {
            writeln!(out, "task:    {}", b.parameter_lmms_task)?;
            writeln!(out, "dataset: {}", b.parameter_dataset_version)?;
            writeln!(out, "count:   {}", b.parameter_sample_count)?;
            writeln!(out, "max_tok: {}", b.parameter_max_tokens)?;
            if let Some(choices) = &b.parameter_mcq_choices {
                writeln!(out, "mcq:     {}", choices.join(", "))?;
            }
            writeln!(out, "decode:  {}", vlm_eval_decoding(b))?;
            if let Some(profile) = &b.parameter_prompt_profile {
                writeln!(out, "profile: {}", profile.as_str())?;
            }
            if let Some(dataset) = &b.dataset {
                writeln!(out, "corpus:  {} ({} bytes)", dataset.format, dataset.bytes)?;
            }
            if let Some(samples) = &b.samples {
                writeln!(out, "samples: {}", samples.len())?;
            }
            if let Some(recipe) = &b.recipe {
                write_recipe(out, recipe)?;
            }
        }
        BenchmarkDefinition::AsrEval(b) => write_asr_eval(out, b)?,
        BenchmarkDefinition::VlSpecDecode(b) => {
            writeln!(
                out,
                "draft_k: {} tokens per step (with speculation on)",
                b.parameter_draft_length
            )?;
            writeln!(out, "max_tok: {}", b.parameter_max_tokens)?;
            writeln!(
                out,
                "batch:   {} sample(s) in flight",
                b.parameter_batch_size
            )?;
            write_temperature(out, b.parameter_temperature, None)?;
            write_corpus(out, &b.dataset)?;
        }
    }
    Ok(())
}

/// A catalog entry's recipe: whose settings it holds, the models it covers
/// and the sources the settings come from.
fn write_recipe(
    out: &mut dyn Write,
    recipe: &pipette_plan_types::VlmEvalRecipe,
) -> anyhow::Result<()> {
    let thinking = recipe
        .enable_thinking
        .map_or_else(|| "unset".to_owned(), |on| on.to_string());
    writeln!(
        out,
        "recipe:  {} (enable_thinking = {thinking})",
        recipe.family
    )?;
    recipe
        .models
        .iter()
        .try_for_each(|line| match &line.mmproj {
            Some(mmproj) => writeln!(out, "model:   {} with {mmproj}", line.repo),
            None => writeln!(out, "model:   {}", line.repo),
        })?;
    recipe
        .sources
        .iter()
        .try_for_each(|source| writeln!(out, "source:  {source}"))?;
    Ok(())
}

/// An asr_eval's dataset, profile, decoding, corpus and recipe. Its clips are
/// its corpus, downloaded at run time, so there is no sample count beyond the
/// one it declares.
fn write_asr_eval(out: &mut dyn Write, b: &AsrEvalBenchmark) -> anyhow::Result<()> {
    writeln!(out, "task:    {}", b.parameter_asr_task.as_str())?;
    writeln!(out, "dataset: {}", b.parameter_dataset_version)?;
    writeln!(out, "count:   {}", b.parameter_sample_count)?;
    writeln!(out, "profile: {}", b.parameter_prompt_profile)?;
    writeln!(out, "max_tok: {}", b.parameter_max_tokens)?;
    if b.beam_search() {
        // The penalties the run uses: each the entry's, else its card's
        // (MiniCPM-o's repetition 1.2), as the client and mgmt resolve them.
        let decoding = b
            .effective_decoding()
            .with_context(|| format!("{}: its decoding", b.benchmark_id))?;
        let AsrDecoding::Beams(beams) = decoding else {
            anyhow::bail!("{}: a beam entry resolved to sampling", b.benchmark_id);
        };
        writeln!(
            out,
            "beams:   {} (repetition penalty {}, length penalty {})",
            beams.num_beams.into_inner(),
            beams.repetition_penalty.into_inner(),
            beams.length_penalty.into_inner(),
        )?;
    } else {
        write_temperature(out, b.sampling_temperature(), None)?;
        if let Some(seed) = b.parameter_seed {
            writeln!(
                out,
                "seed:    {seed} (seed_rule={})",
                seed_rule_name(b.parameter_seed_rule)
            )?;
        }
    }
    if let Some(dataset) = &b.dataset {
        writeln!(out, "corpus:  {} ({} bytes)", dataset.format, dataset.bytes)?;
    }
    if let Some(recipe) = &b.recipe {
        write_recipe(out, recipe)?;
    }
    Ok(())
}

/// What an eval cell samples at. The temperature prints whatever its source,
/// so two ids can be compared; an unpinned cut does not, because none is sent.
fn write_eval_sampling(out: &mut dyn Write, benchmark: &EvalBenchmark) -> anyhow::Result<()> {
    let source = if benchmark.parameter_temperature.is_some() {
        "pinned"
    } else {
        "eval-id policy"
    };
    write_temperature(out, benchmark.sampling_temperature()?, Some(source))?;
    if let Some(top_p) = benchmark.parameter_top_p {
        writeln!(out, "top_p:   {top_p}")?;
    }
    if let Some(top_k) = benchmark.parameter_top_k {
        writeln!(out, "top_k:   {top_k}")?;
    }
    if let Some(min_p) = benchmark.parameter_min_p {
        writeln!(out, "min_p:   {min_p}")?;
    }
    if let Some(seed) = benchmark.parameter_seed {
        writeln!(out, "seed:    {seed} (offset by the #k repeat index)")?;
    }
    Ok(())
}

/// One `temp:` line for every kind that has one, so the key means the same
/// thing across entries. `source` says where an eval's came from; a corpus
/// entry has only the one. Greedy is the reference procedure, worth naming
/// beside a bare `0`.
fn write_temperature(
    out: &mut dyn Write,
    temperature: SamplingTemperature,
    source: Option<&str>,
) -> anyhow::Result<()> {
    let greedy = (temperature == SamplingTemperature::greedy()).then_some("greedy");
    let notes: Vec<&str> = source.into_iter().chain(greedy).collect();
    if notes.is_empty() {
        writeln!(out, "temp:    {temperature}")?;
    } else {
        writeln!(out, "temp:    {temperature} ({})", notes.join(", "))?;
    }
    Ok(())
}

/// The requests the cell keeps in flight, printed even at the reference size:
/// a reader comparing two entries should not have to know that an absent
/// field means one.
fn write_batch_size(out: &mut dyn Write, batch_size: BatchSize) -> anyhow::Result<()> {
    writeln!(out, "batch:   {batch_size} request(s) in flight")?;
    Ok(())
}

/// `greedy` at temperature 0 (or none set), then every knob the entry sets.
/// Unset knobs are sent neutral, so the line is the whole decode.
fn vlm_eval_decoding(b: &pipette_plan_types::VlmEvalBenchmark) -> String {
    let temperature = b.sampling_temperature().as_f64();
    let lead = if temperature > 0.0 {
        format!("temperature={temperature}")
    } else {
        "greedy".to_owned()
    };
    std::iter::once(Some(lead))
        .chain([
            b.parameter_top_p.map(|v| format!("top_p={}", v.as_f64())),
            b.parameter_top_k.map(|v| format!("top_k={}", i32::from(v))),
            b.parameter_min_p.map(|v| format!("min_p={}", v.as_f64())),
            b.parameter_presence_penalty
                .map(|v| format!("presence_penalty={}", v.as_f64())),
            b.parameter_repeat_penalty
                .map(|v| format!("repeat_penalty={}", v.as_f64())),
            b.parameter_seed.map(|v| {
                format!(
                    "seed={} seed_rule={}",
                    i64::from(v),
                    seed_rule_name(b.parameter_seed_rule)
                )
            }),
            b.parameter_image_tokens
                .map(|v| format!("image_tokens={}", v.into_inner())),
        ])
        .flatten()
        .collect::<Vec<_>>()
        .join(" ")
}

/// How a seeded vlm_eval or asr_eval derives each sample's seed, named even
/// when the entry names no rule: the two draw differently (a shared seed
/// biases a vLLM run), and a reader should not have to know that an absent
/// field means the repeat offset.
fn seed_rule_name(rule: Option<SeedRule>) -> &'static str {
    match rule {
        None => "repeat_offset",
        Some(SeedRule::PerSampleSha256) => "per_sample_sha256",
    }
}

fn write_corpus(out: &mut dyn Write, dataset: &pipette_plan_types::Dataset) -> anyhow::Result<()> {
    writeln!(out, "corpus:  {} ({})", dataset.format, dataset.adapter)?;
    writeln!(
        out,
        "subset:  {}",
        dataset.subset.as_deref().unwrap_or("(all)")
    )?;
    writeln!(out, "bytes:   {}", dataset.bytes)?;
    writeln!(out, "digest:  {}", dataset.digest)?;
    writeln!(out, "upstream:{}", dataset.upstream.reference())?;
    if let Some(license) = dataset.upstream.license() {
        writeln!(out, "license: {license}")?;
    }
    Ok(())
}

fn write_vl_shape(
    out: &mut dyn Write,
    width: PositiveCount,
    height: PositiveCount,
    text: NonNegativeCount,
    images: NonNegativeCount,
    decode: Option<PositiveCount>,
) -> anyhow::Result<()> {
    writeln!(out, "image:   {width}×{height}")?;
    writeln!(out, "text:    {text} tokens")?;
    if let Some(decode) = decode {
        writeln!(out, "decode:  {decode} tokens")?;
    }
    writeln!(out, "images:  {images}")?;
    Ok(())
}

/// The clip's format as well as its length: two entries of one length can
/// differ in format.
fn write_asr_clip(
    out: &mut dyn Write,
    duration: AudioDurationMillis,
    rate: AudioSampleRate,
    channels: AudioChannels,
    text: NonNegativeCount,
) -> anyhow::Result<()> {
    writeln!(out, "audio:   {duration} ms, {rate} Hz, {channels} ch")?;
    writeln!(out, "text:    {text} tokens")?;
    Ok(())
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
    --model 'openvino://repo=OpenVINO/LFM2.5-350M-int8-ov' \\
    --runtime 'uv-openvino://version=2026.4.1' \\
    --runtime-flags '{\"device\":\"cpu\"}'

  # re-run against what's already installed, by digest prefix
  pipette benchmarks run --benchmark local/decode_throughput_512_100 \\
    --model 'model://sha256=d86cc299' --runtime 'runtime://sha256=c07a4fd3'

CELL SETTINGS (--runtime-flags) -- which are legal depends on all three axes:
  llama.cpp x gguf_text
    prefill / decode / max-memory   threads, number_gpu_layers, mmap, load_mode,
                                    flash_attention, envs, raw
    end-to-end / eval               the above, plus ctx_size, no_cache
    eval                            and parallel, the llama-server slot count
  llama.cpp x gguf_vision
    vl / asr                        threads, number_gpu_layers, mmap, load_mode,
                                    flash_attention, ctx_size, no_cache, envs, raw
                                    (asr refuses a raw slot count, reasoning flag or
                                    chat-template flag)
    vlm_eval                       the above, plus parallel, the llama-server
                                    slot count (an unpinned ctx_size is
                                    calibrated per slot and multiplied by it);
                                    whether a model thinks is a model flag,
                                    --model-enable-thinking, as on eval
    asr_eval                        the vlm_eval settings; an unpinned
                                    ctx_size is sized per slot from the
                                    clips' durations, a beam entry runs one
                                    slot, and thinking is off: the model
                                    flags state enable_thinking = false
  docker-vllm x torch               tensor_parallel_size, dtype, max_model_len,
                                    prefix_caching, gpus, shm_size, ipc, envs, raw
    eval                            and parallel, the client's in-flight request
                                    count (the engine batches; nothing is argv),
                                    and repetition_detection, vLLM's own
                                    request field: {min_pattern_size,
                                    max_pattern_size, min_count}
  uv-vllm x torch                   the above, minus gpus / shm_size / ipc
    vlm_eval, asr_eval              and parallel, as on eval (no
                                    repetition_detection); raw and envs take
                                    only the speed, memory, device and logging
                                    settings the plan types list
  docker-sglang x torch            tensor_parallel_size, prefix_caching, gpus,
                                    shm_size, ipc, envs, raw
    eval                            and parallel, as above
  uv-sglang x torch                 tensor_parallel_size, prefix_caching, envs, raw
    eval                            and parallel, as above
  uv-openvino x openvino            device, max_prompt_len, min_response_len,
                                    generate_hint
  ovms x openvino                   the above, plus prefix_caching (unset means
                                    off; OVMS itself defaults it on)
  trt-edgellm x gguf_vision         precision (fp16, bf16, w8a16), envs, raw;
                                    vl_throughput and vl_max_memory only
  uv-onnx-genai x onnx_genai        none: any setting is refused ({} is a no-op)
  mlx-macos-pipette x mlx           none: any setting is refused ({} is a no-op)
  core-ai-macos-pipette x core_ai   none: any setting is refused ({} is a no-op)

`raw` takes verbatim argv tokens, and refuses any that alias a typed setting or
that the benchmark owns for that cell.

A `local/<id>` result stays on this machine. `--sync` submits only a result from
the synced catalog; `pipette sync` does the same afterwards.

Gated Hugging Face weights: set PIPETTE_HF_TOKEN.

Full notation reference: docs/pipette-cli/models-and-runtimes.md";

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
    /// one *requires* its `model_type` / `benchmark_type` axes. Eval and vlm_eval
    /// benchmarks only. Mutually exclusive with `--model-enable-thinking`.
    #[arg(long)]
    pub model_flags: Option<String>,
    /// Run-driving settings as one JSON object, a claim's `benchmark_flags`
    /// without its axis keys: `'{"readiness":{"max_wait_secs":900}}'`. A setting
    /// the cell does not carry is an error. Mutually exclusive with the five
    /// single-setting flags below, which are its shorthand.
    #[arg(
        long,
        conflicts_with_all = [
            "http_timeout_seconds",
            "attempt_timeout_seconds",
            "readiness_max_wait_secs",
            "readiness_skip_thermal",
            "readiness_skin_below_c",
        ]
    )]
    pub benchmark_flags: Option<String>,
    /// HTTP timeout in seconds for the client's calls to the local server.
    /// Accepted on the cells that drive one over HTTP — eval, end-to-end-latency,
    /// vl and asr — and rejected elsewhere.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub http_timeout_seconds: Option<u64>,
    /// Wall-clock limit in seconds on each attempt: one warm-up (model load
    /// included) or one measured rep, one sample on vl-spec-decode. An attempt
    /// past it is stopped and the run fails. Readiness waits, calibration,
    /// downloads and the steps that can compile do not count: a TensorRT engine
    /// build, and the whole start of the first server on Core AI, OpenVINO and
    /// OVMS. 300 s by default on the timing, throughput and peak-memory cells;
    /// vl-spec-decode has none, and an eval refuses it. Raise it for a device
    /// slow enough that the default stops a healthy rep.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub attempt_timeout_seconds: Option<u64>,
    /// Override the readiness wait ceiling (seconds) before the measurement.
    /// Valid only on the timing benchmarks that gate on device readiness
    /// (prefill / decode / end-to-end-latency / vl / asr-throughput); on eval or
    /// a peak-memory cell the `(benchmark, runtime, model)` cell carries no
    /// readiness knob, so it is rejected. Set `PIPETTE_READINESS_MAX_WAIT_SECS`
    /// directly for an ungated override: it is read by the readiness gate
    /// itself, so unlike this flag it stays harmless on the cells that carry no
    /// readiness setting.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub readiness_max_wait_secs: Option<u64>,
    /// Waive the *thermal* readiness criterion for this cell, keeping the load
    /// criterion. Valid on the same timing benchmarks as
    /// `--readiness-max-wait-secs`. Changes the criteria rather than just the
    /// patience, so a cell run with it is not comparable to a gated one.
    #[arg(long)]
    pub readiness_skip_thermal: bool,
    /// Android CLI gate: the hottest live skin sensor must read below this
    /// whole-degree Celsius threshold (1–110).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=110))]
    pub readiness_skin_below_c: Option<u32>,
    /// Android CLI gate: what to do if a touch is held on the touchscreen.
    /// `off` skips touch detection. `wait` holds the gate until the touch
    /// clears or the readiness deadline passes. `fail` fails the cell after
    /// about 9 s. This flag overrides
    /// `PIPETTE_READINESS_HELD_TOUCH` and `[readiness.android] held_touch`.
    /// The gate waits when no mode is configured. It is not a cell setting, so
    /// `--benchmark-flags` does not carry it. Other platforms reject it.
    #[arg(long, value_name = "off|wait|fail")]
    pub readiness_held_touch: Option<pipette_readiness::HeldTouch>,
    /// Convenience for a single `enable_thinking` model flag: derives the axes
    /// from the cell, so it needs no JSON. Eval, vlm_eval and asr_eval
    /// benchmarks only (an asr_eval cell runs only with `false`).
    /// Mutually exclusive with `--model-flags`.
    #[arg(long)]
    pub model_enable_thinking: Option<bool>,
    /// Submit the result to the management server immediately after the run.
    /// Applies only to a benchmark from the synced catalog — a `local/<id>` run
    /// always stays on disk, and passing this does not change that.
    #[arg(long)]
    pub sync: bool,
}

impl BenchmarkArgs {
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        self.command
            .execute(ws, http, client_config, programs, console)
    }
}

impl BenchmarkCommand {
    /// `client_config` and `programs` reach `run` alone: listing and seeding
    /// spawn nothing.
    pub fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        match self {
            // The catalog is runtime-agnostic; init-local seeds the union of
            // every runtime's supported benchmark types.
            BenchmarkCommand::InitLocal => {
                let summary = seed_standard_local(&ws.benchmarks(), BenchmarkType::ALL)?;
                writeln!(
                    console.out(),
                    "Wrote standard local benchmarks: {} created, {} updated",
                    summary.created,
                    summary.updated
                )?;
                Ok(())
            }
            BenchmarkCommand::List(args) => args.execute(ws, console),
            BenchmarkCommand::Show(args) => args.execute(ws, console),
            BenchmarkCommand::Run(args) => args.execute(ws, http, client_config, programs, console),
        }
    }
}

impl BenchmarkRunArgs {
    fn execute(
        self,
        ws: &PipetteWorkspace,
        http: &HttpClient,
        client_config: &ClientConfiguration,
        programs: &ExternalPrograms,
        console: &mut dyn Console,
    ) -> anyhow::Result<()> {
        let sync = self.sync;
        anyhow::ensure!(
            !sync || ws.eval_shard().is_none(),
            "--eval-shard records nothing, so --sync has nothing to send"
        );
        let held_touch = self.readiness_held_touch;
        crate::run::ensure_held_touch_on_android(held_touch)?;
        // The catalog side is this client's business: it picks the filing
        // location, and the run takes the resolved body, not the address.
        let location = self.benchmark_ref.source().into();
        let (spec, benchmark) = self.into_client_run_spec(ws)?;
        // The definition's type, not the result data's: `vl_max_memory` reuses
        // the `MaxMemoryUsage` result variant, so deriving the label from the
        // data would print `type: max_memory_usage` for a VL memory run.
        let benchmark_type = benchmark.benchmark_type();
        // `prepare` adds the cell's pins; the policy only carries the cap here.
        let artifacts = ws.artifacts(http, programs);
        let run = run_cell(
            &spec,
            benchmark,
            &artifacts,
            ws,
            &ws.device().get()?,
            client_config,
            held_touch,
            &pipette_wait::never_cancelled,
            programs,
        )?;
        record_finished_run(ws, http, console, benchmark_type, location, sync, run)
    }

    /// Resolve the refs and the benchmark, validate the run knobs into
    /// plan-form flags, and refuse a cell this host cannot run. Returns the
    /// resolved body so the run does not resolve the same id a second time.
    fn into_client_run_spec(
        self,
        ws: &PipetteWorkspace,
    ) -> anyhow::Result<(ClientRunSpec, BenchmarkDefinition)> {
        let runtime = resolve_runtime_arg(ws, &self.runtime)?;
        refuse_foreign_os(&runtime, std::env::consts::OS)?;
        let mut model = resolve_model_arg(ws, &self.model)?;
        inject_env_hf_token(&mut model)?;
        // Its `.benchmark_type()` is the authoritative type, not a guess from
        // the id string.
        let benchmark = resolve_benchmark(ws, &self.benchmark_ref)?;
        // `--runtime-flags` carries knobs only: the cell is the one just resolved
        // from `--benchmark`, `--runtime` and `--model`, so axis keys are refused
        // rather than restated. `raw` rides inside the entry, and the owning
        // runtime renders it at execution time.
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
                if let Some(ref f) = flags
                    && !f.matches(benchmark.benchmark_type(), &model)
                {
                    anyhow::bail!(
                        "--model-flags entry does not match this cell \
                             (benchmark={:?}, model={})",
                        benchmark.benchmark_type(),
                        model
                    );
                }
                flags
            }
            (None, Some(enable_thinking)) => Some(
                ModelFlags::try_from(ModelFlagRef {
                    model_type: ModelType::of(&model),
                    benchmark_type: benchmark.benchmark_type(),
                    enable_thinking: Some(enable_thinking),
                    reasoning_effort: None,
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
        let benchmark_flags = match self.benchmark_flags.as_deref() {
            Some(json) => {
                parse_benchmark_flags_json(json, benchmark.benchmark_type(), &runtime, &model)
                    .context("parsing --benchmark-flags JSON")?
            }
            None => build_benchmark_flags(
                benchmark.benchmark_type(),
                &runtime,
                &model,
                CellKnobs {
                    http_timeout_seconds: self.http_timeout_seconds,
                    attempt_timeout_seconds: self.attempt_timeout_seconds,
                    readiness_max_wait_secs: self.readiness_max_wait_secs,
                    readiness_skip_thermal: self.readiness_skip_thermal,
                    readiness_skin_below_c: self.readiness_skin_below_c,
                },
            )?,
        };
        crate::run::ensure_skin_on_android(benchmark_flags.as_ref())?;
        let spec = ClientRunSpec {
            benchmark: BenchmarkId::try_new(benchmark.benchmark_id().to_string())
                .context("benchmark id")?,
            model,
            runtime,
            runtime_flags,
            model_flags,
            benchmark_flags,
        };
        spec.validate()?;
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

/// Parse `--benchmark-flags` against the cell it resolved, as
/// [`RuntimeFlags::from_cell_json`] parses `--runtime-flags`: one JSON object of
/// knobs, the axes derived from the cell rather than restated, `{}` meaning none.
fn parse_benchmark_flags_json(
    json: &str,
    benchmark_type: BenchmarkType,
    runtime: &Runtime,
    model: &Model,
) -> anyhow::Result<Option<BenchmarkFlags>> {
    let serde_json::Value::Object(mut fields) = serde_json::from_str(json)? else {
        anyhow::bail!(r#"expected a JSON object of knobs, e.g. {{"http_timeout_seconds":600}}"#);
    };
    if fields.is_empty() {
        return Ok(None);
    }
    [
        (
            "runtime_type",
            serde_json::to_value(RuntimeType::of(runtime))?,
        ),
        ("model_type", serde_json::to_value(ModelType::of(model))?),
        ("benchmark_type", serde_json::to_value(benchmark_type)?),
    ]
    .into_iter()
    .try_for_each(|(key, derived)| {
        anyhow::ensure!(
            !fields.contains_key(key),
            "must not carry `{key}`: the cell comes from --benchmark, --runtime and --model"
        );
        fields.insert(key.to_owned(), derived);
        Ok(())
    })?;
    let reference: BenchmarkFlagRef = serde_json::from_value(serde_json::Value::Object(fields))?;
    // The shorthand's `range(1..)`, so the two forms accept the same waits.
    anyhow::ensure!(
        reference.readiness.as_ref().and_then(|r| r.max_wait_secs) != Some(0),
        "readiness.max_wait_secs must be at least 1, as --readiness-max-wait-secs requires"
    );
    let flags = BenchmarkFlags::try_from(reference)
        .with_context(|| format!("not valid for this benchmark ({benchmark_type:?})"))?;
    Ok(Some(flags))
}

/// The run-driving knobs as the command line carried them, before the cell's
/// [`BenchmarkFlags`] variant decides which of them it accepts.
#[derive(Debug, Default)]
struct CellKnobs {
    http_timeout_seconds: Option<u64>,
    attempt_timeout_seconds: Option<u64>,
    readiness_max_wait_secs: Option<u64>,
    readiness_skip_thermal: bool,
    readiness_skin_below_c: Option<u32>,
}

/// The run's `BenchmarkFlags` for a cell, or `None` when no knob is set.
fn build_benchmark_flags(
    benchmark_type: BenchmarkType,
    runtime: &Runtime,
    model: &Model,
    knobs: CellKnobs,
) -> anyhow::Result<Option<BenchmarkFlags>> {
    let CellKnobs {
        http_timeout_seconds,
        attempt_timeout_seconds,
        readiness_max_wait_secs,
        readiness_skip_thermal,
        readiness_skin_below_c,
    } = knobs;
    let readiness_skin_below_c = readiness_skin_below_c
        .map(SkinCelsius::try_new)
        .transpose()?;
    let readiness = (readiness_max_wait_secs.is_some()
        || readiness_skip_thermal
        || readiness_skin_below_c.is_some())
    .then(|| {
        ReadinessOverrides {
            max_wait_secs: readiness_max_wait_secs,
            // `None` rather than `Some(false)` when unset, so an untouched knob
            // stays absent from the serialized plan.
            skip_thermal: readiness_skip_thermal.then_some(true),
            skin_below_c: readiness_skin_below_c,
        }
    });
    let has_knob =
        http_timeout_seconds.is_some() || attempt_timeout_seconds.is_some() || readiness.is_some();
    if !has_knob {
        return Ok(None);
    }
    Ok(Some(
        // `TryFrom` routes the (benchmark, runtime, model) triple to its variant
        // and rejects any knob that cell doesn't carry: readiness only on a
        // cell that gates, the attempt limit nowhere on an eval.
        BenchmarkFlags::try_from(BenchmarkFlagRef {
            runtime_type: RuntimeType::of(runtime),
            model_type: ModelType::of(model),
            benchmark_type,
            http_timeout_seconds: http_timeout_seconds
                .map(PositiveSeconds::try_new)
                .transpose()?,
            attempt_timeout_seconds: attempt_timeout_seconds
                .map(PositiveSeconds::try_new)
                .transpose()?,
            _bench_timeout_obsolete: None,
            _doomloop_obsolete: None,
            readiness,
        })
        .with_context(|| {
            format!(
                "--http-timeout-seconds / --attempt-timeout-seconds / \
                 --readiness-max-wait-secs / --readiness-skin-below-c are not valid for this \
                 benchmark ({benchmark_type:?})"
            )
        })?,
    ))
}

fn write_lines(out: &mut dyn Write, lines: &[String]) -> anyhow::Result<()> {
    lines.iter().try_for_each(|line| writeln!(out, "{line}"))?;
    Ok(())
}

/// What `benchmarks run` keeps of a finished run. Split from `execute` so the
/// shard rule is testable without running a cell: a text eval has no
/// completeness check, so a shard's saved result could reach mgmt.
fn record_finished_run(
    ws: &PipetteWorkspace,
    http: &HttpClient,
    console: &mut dyn Console,
    benchmark_type: BenchmarkType,
    location: BenchmarkResultLocation,
    sync: bool,
    (payload, extras, eval_checkpoint): (
        BenchmarkSubmissionPayload,
        BenchmarkResultExtras,
        Option<EvalRunDigest>,
    ),
) -> anyhow::Result<()> {
    if let Some(shard) = ws.eval_shard() {
        writeln!(
            console.out(),
            "eval shard {shard} of `{}`: its answers are in state/evals; nothing saved, \
             submitted or finalized",
            payload.benchmark_id
        )?;
        return Ok(());
    }
    write_lines(
        console.out(),
        &run_summary_lines(&payload.benchmark_id, benchmark_type, &payload.result),
    )?;
    let done = record_and_maybe_submit_run(
        &ws.results(),
        &ws.registration(),
        &payload,
        &extras,
        location,
        sync,
        http,
    )?;
    // Saved on disk, so the checkpoint has done its job.
    finalize_eval_checkpoint(ws, eval_checkpoint.as_ref());
    write_record_done(console, ws, &done)
}

fn write_record_done(
    console: &mut dyn Console,
    ws: &PipetteWorkspace,
    done: &RecordSubmitOutcome,
) -> anyhow::Result<()> {
    writeln!(
        console.out(),
        "payload: {}",
        ws.results()
            .payload_path(done.location, &done.result_id)
            .display()
    )?;
    writeln!(
        console.out(),
        "recorded {} ({})",
        done.result_id,
        done.location.label()
    )?;
    if let Some(job_id) = &done.job_id {
        writeln!(console.out(), "submitted as job {job_id}")?;
    }
    Ok(())
}

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
            image_tokens,
            prompt_ms,
            prompt_ms_stddev,
            predicted_ms,
            predicted_ms_stddev,
            total_time_ms,
            total_time_ms_stddev,
        } => {
            lines.push(format!("prompt_tokens: {prompt_tokens}"));
            if let Some(image_tokens) = image_tokens {
                lines.push(format!("image_tokens: {image_tokens}"));
            }
            lines.push(format!("prompt_ms: {prompt_ms:.3}"));
            if let Some(stddev) = prompt_ms_stddev {
                lines.push(format!("prompt_ms_stddev: {stddev:.3}"));
            }
            lines.push(format!("predicted_ms: {predicted_ms:.3}"));
            if let Some(stddev) = predicted_ms_stddev {
                lines.push(format!("predicted_ms_stddev: {stddev:.3}"));
            }
            if let Some(total_time_ms) = total_time_ms {
                lines.push(format!("total_time_ms: {total_time_ms:.3}"));
            }
            if let Some(stddev) = total_time_ms_stddev {
                lines.push(format!("total_time_ms_stddev: {stddev:.3}"));
            }
        }
        BenchmarkResultData::AsrThroughput {
            asr_prompt_tokens,
            asr_audio_tokens,
            asr_prompt_ms,
            asr_prompt_ms_stddev,
            asr_predicted_ms,
            asr_predicted_ms_stddev,
            total_time_ms,
            total_time_ms_stddev,
        } => {
            lines.push(format!("asr_prompt_tokens: {asr_prompt_tokens}"));
            lines.push(format!("asr_audio_tokens: {asr_audio_tokens}"));
            lines.push(format!("asr_prompt_ms: {asr_prompt_ms:.3}"));
            if let Some(stddev) = asr_prompt_ms_stddev {
                lines.push(format!("asr_prompt_ms_stddev: {stddev:.3}"));
            }
            lines.push(format!("asr_predicted_ms: {asr_predicted_ms:.3}"));
            if let Some(stddev) = asr_predicted_ms_stddev {
                lines.push(format!("asr_predicted_ms_stddev: {stddev:.3}"));
            }
            if let Some(total_time_ms) = total_time_ms {
                lines.push(format!("total_time_ms: {total_time_ms:.3}"));
            }
            if let Some(stddev) = total_time_ms_stddev {
                lines.push(format!("total_time_ms_stddev: {stddev:.3}"));
            }
        }
        // Derived for the terminal only; the server derives the row's metrics
        // from the counters. A zero denominator omits the line. No speedup and
        // no identity rate: those need the other run's row, and this client
        // never has it.
        BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_turns,
            vl_spec_decode_predicted_ms,
            vl_spec_decode_generated_tokens,
            vl_spec_decode_drafted_tokens,
            vl_spec_decode_accepted_tokens,
            vl_spec_decode_turn_accept_length_sum,
            vl_spec_decode_prompt_ms,
            vl_spec_decode_ttft_ms,
            vl_spec_decode_e2e_ms,
            ..
        } => {
            lines.push(format!("turns: {vl_spec_decode_turns}"));
            lines.push(format!(
                "generated_tokens: {vl_spec_decode_generated_tokens}"
            ));
            // The drafter's lines exist only on a row with speculation on.
            // Target forwards are `generated - accepted`: every verification
            // step commits one token that is not an accepted draft. Saturating
            // and omitted at zero, so a contradictory row prints no accept
            // length rather than a flattering one.
            if let (Some(drafted), Some(accepted)) = (
                vl_spec_decode_drafted_tokens,
                vl_spec_decode_accepted_tokens,
            ) {
                push_ratio(
                    &mut lines,
                    "accept_length",
                    ratio(
                        *vl_spec_decode_generated_tokens as f64,
                        vl_spec_decode_generated_tokens.saturating_sub(*accepted) as f64,
                    ),
                );
                push_ratio(
                    &mut lines,
                    "draft_acceptance_rate",
                    ratio(*accepted as f64, *drafted as f64),
                );
            }
            if let Some(sum) = vl_spec_decode_turn_accept_length_sum {
                push_ratio(
                    &mut lines,
                    "accept_length_per_request",
                    ratio(*sum, f64::from(*vl_spec_decode_turns)),
                );
            }
            push_ratio(
                &mut lines,
                "decode_tok_per_s",
                ratio(
                    *vl_spec_decode_generated_tokens as f64 * 1000.0,
                    *vl_spec_decode_predicted_ms,
                ),
            );
            push_ratio(
                &mut lines,
                "prompt_ms_per_turn",
                ratio(*vl_spec_decode_prompt_ms, f64::from(*vl_spec_decode_turns)),
            );
            push_ratio(
                &mut lines,
                "ttft_ms_per_turn",
                ratio(*vl_spec_decode_ttft_ms, f64::from(*vl_spec_decode_turns)),
            );
            push_ratio(
                &mut lines,
                "e2e_ms_per_turn",
                ratio(*vl_spec_decode_e2e_ms, f64::from(*vl_spec_decode_turns)),
            );
        }
    }
    lines
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| numerator / denominator)
}

/// One derived ratio, or nothing when its denominator was zero — a missing
/// line reads as "not computable" where `0.000` would read as a measurement.
fn push_ratio(lines: &mut Vec<String>, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        lines.push(format!("{name}: {value:.3}"));
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};
    use rstest::{fixture, rstest};

    use pipette_artifacts::ArtifactsContext;
    use pipette_plan_types::{BenchmarkResultData, RunSpecError};
    use pipette_programs::ExternalPrograms;

    use super::*;
    use crate::identity::DeviceLabels;
    use crate::model_uri::parse_model_arg;
    use crate::runtime_uri::parse_runtime_arg;
    use crate::test_support::CapturedConsole;
    use crate::test_support::corpus_wire;
    use crate::test_support::{
        IMG80, TempWorkspace, cache_remote, expect_gate_on_this_host, img80_wire,
        write_public_2026_08_1_img80_cache,
    };

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

    /// Artifacts context whose `docker` and `uv` paths do not exist, so an
    /// install fails on spawn instead of pulling an image or building a venv
    /// (the `#[ignore]` install smokes under `crates/*/tests/` cover that).
    /// Enough only because `prepare` ensures the runtime before the model and
    /// both install arms take their program's path before doing any work.
    /// Reorder either and the model fetch runs for real.
    fn stubbed_artifacts(_tw: &TempWorkspace) -> anyhow::Result<ArtifactsContext> {
        Ok(
            ArtifactsContext::new(test_http()?, ExternalPrograms::at_missing_paths())
                .with_download_retry(pipette_http::retry::RetryPolicy::IMMEDIATE),
        )
    }

    /// The payload's id and the eval digest come from the body, so a body the
    /// spec does not name would record a cell that never ran.
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
        let err = run_cell(
            &spec,
            other,
            &stubbed_artifacts(&tw)?,
            &tw.ws,
            &DeviceLabels::default(),
            &ClientConfiguration::default(),
            None,
            &pipette_wait::never_cancelled,
            &ExternalPrograms::at_missing_paths(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("a mismatched body should be refused"))?;

        let msg = format!("{err:#}");
        assert!(
            msg.contains("eval_smoke") && msg.contains("prefill_throughput_smoke"),
            "the error must name both ids; got: {msg}"
        );
        Ok(())
    }

    /// A bounded eval on a cell that leaves its context unpinned is refused
    /// before the install: it would run unbounded and record the bound.
    #[test]
    fn run_refuses_a_context_bound_the_cell_does_not_pin() -> anyhow::Result<()> {
        let tw = seeded_workspace("bench-context-unpinned")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "gguf-text://repo=o/r&path=m-Q4_0.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?;
        let (spec, benchmark) = args.into_client_run_spec(&tw.ws)?;
        let BenchmarkDefinition::Eval(eval) = benchmark else {
            anyhow::bail!("eval_smoke should be an eval");
        };
        let bounded = BenchmarkDefinition::Eval(EvalBenchmark {
            parameter_max_context: Some(eval.parameter_max_tokens),
            ..eval
        });
        let err = run_cell(
            &spec,
            bounded,
            &stubbed_artifacts(&tw)?,
            &tw.ws,
            &DeviceLabels::default(),
            &ClientConfiguration::default(),
            None,
            &pipette_wait::never_cancelled,
            &ExternalPrograms::at_missing_paths(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("an unpinned bound should be refused"))?;

        let msg = format!("{err:#}");
        assert!(
            msg.contains("parameter_max_context") && !msg.contains("ensuring runtime"),
            "expected the context refusal ahead of the install; got: {msg}"
        );
        Ok(())
    }

    /// Drive a ref pair to the point where its runtime install would run, and
    /// assert it got there, past the on-device bail in `prepare`.
    fn assert_reaches_runtime_install(label: &str, argv: &[&str]) -> anyhow::Result<()> {
        let tw = seeded_workspace(label)?;
        let args = parse(argv)?;
        let (spec, benchmark) = args.into_client_run_spec(&tw.ws)?;
        let err = run_cell(
            &spec,
            benchmark,
            &stubbed_artifacts(&tw)?,
            &tw.ws,
            &DeviceLabels::default(),
            &ClientConfiguration::default(),
            None,
            &pipette_wait::never_cancelled,
            &ExternalPrograms::at_missing_paths(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("{label}: run should fail installing the runtime"))?;
        let msg = format!("{err:#}");
        anyhow::ensure!(
            msg.contains("ensuring runtime"),
            "{label}: expected a runtime-install failure, got: {msg}"
        );
        Ok(())
    }

    fn seeded_workspace(label: &str) -> anyhow::Result<TempWorkspace> {
        let tw = TempWorkspace::new(&format!("bench-{label}"))?;
        seed_standard_local(&tw.ws.benchmarks(), BenchmarkType::ALL)?;
        Ok(tw)
    }

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
        assert!(
            lines
                .iter()
                .any(|l| l == "prefill_time_ms_stddev: 0.250000")
        );
    }

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

    /// Each optional figure prints only when the engine measured it.
    #[rstest::rstest]
    #[case::decode(
        BenchmarkType::DecodeThroughput,
        BenchmarkResultData::DecodeThroughput { decode_time_ms: 3.0, decode_time_ms_stddev: Some(0.5) },
        &["decode_time_ms: 3.000000", "decode_time_ms_stddev: 0.500000"]
    )]
    #[case::end_to_end(
        BenchmarkType::EndToEndLatency,
        BenchmarkResultData::EndToEndLatency { total_time_ms: 940.0, total_time_ms_stddev: Some(11.0) },
        &["total_time_ms: 940.000000", "total_time_ms_stddev: 11.000000"]
    )]
    #[case::end_to_end_no_stddev(
        BenchmarkType::EndToEndLatency,
        BenchmarkResultData::EndToEndLatency { total_time_ms: 940.0, total_time_ms_stddev: None },
        &["total_time_ms: 940.000000"]
    )]
    #[case::max_memory_host_only(
        BenchmarkType::MaxMemoryUsage,
        BenchmarkResultData::MaxMemoryUsage { max_host_bytes: 8, max_gpu_bytes: None, max_npu_bytes: None },
        &["max_host_bytes: 8"]
    )]
    #[case::max_memory_all(
        BenchmarkType::MaxMemoryUsage,
        BenchmarkResultData::MaxMemoryUsage { max_host_bytes: 8, max_gpu_bytes: Some(4), max_npu_bytes: Some(2) },
        &["max_host_bytes: 8", "max_gpu_bytes: 4", "max_npu_bytes: 2"]
    )]
    #[case::vl_minimal(
        BenchmarkType::VlThroughput,
        BenchmarkResultData::VlThroughput {
            prompt_tokens: 75, image_tokens: None, prompt_ms: 352.3, prompt_ms_stddev: None,
            predicted_ms: 32.7, predicted_ms_stddev: None, total_time_ms: None,
            total_time_ms_stddev: None,
        },
        &["prompt_tokens: 75", "prompt_ms: 352.300", "predicted_ms: 32.700"]
    )]
    #[case::asr(
        BenchmarkType::AsrThroughput,
        BenchmarkResultData::AsrThroughput {
            asr_prompt_tokens: 137, asr_audio_tokens: 118, asr_prompt_ms: 83.3,
            asr_prompt_ms_stddev: Some(0.9), asr_predicted_ms: 129.4, asr_predicted_ms_stddev: None,
            total_time_ms: Some(230.0), total_time_ms_stddev: Some(2.0),
        },
        &[
            "asr_prompt_tokens: 137", "asr_audio_tokens: 118", "asr_prompt_ms: 83.300",
            "asr_prompt_ms_stddev: 0.900", "asr_predicted_ms: 129.400",
            "total_time_ms: 230.000", "total_time_ms_stddev: 2.000",
        ]
    )]
    #[case::vl_full(
        BenchmarkType::VlThroughput,
        BenchmarkResultData::VlThroughput {
            prompt_tokens: 75, image_tokens: Some(64), prompt_ms: 352.3, prompt_ms_stddev: Some(3.8),
            predicted_ms: 32.7, predicted_ms_stddev: Some(1.2), total_time_ms: Some(390.1),
            total_time_ms_stddev: Some(4.2),
        },
        &[
            "prompt_tokens: 75", "image_tokens: 64", "prompt_ms: 352.300",
            "prompt_ms_stddev: 3.800", "predicted_ms: 32.700", "predicted_ms_stddev: 1.200",
            "total_time_ms: 390.100", "total_time_ms_stddev: 4.200",
        ]
    )]
    fn run_summary_prints_what_the_engine_measured(
        #[case] benchmark_type: BenchmarkType,
        #[case] data: BenchmarkResultData,
        #[case] metrics: &[&str],
    ) {
        let lines = run_summary_lines("b", benchmark_type, &data);
        let expected: Vec<String> = metrics.iter().map(|m| (*m).to_owned()).collect();
        assert_eq!(lines.get(2..), Some(expected.as_slice()));
    }

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

    /// `vl_max_memory` reuses the `MaxMemoryUsage` result shape, so a label
    /// derived from the data would mislabel it.
    #[test]
    fn run_summary_labels_vl_max_memory_by_definition_type() {
        let lines = run_summary_lines(
            "vl_mem_smoke",
            BenchmarkType::VlMaxMemory,
            &BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: 1024,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
        );
        assert_eq!(lines[1], "type: vl_max_memory");
        assert!(lines.iter().any(|l| l == "max_host_bytes: 1024"));
    }

    /// `asr_max_memory` reuses the `MaxMemoryUsage` result shape too.
    #[test]
    fn run_summary_labels_asr_max_memory_by_definition_type() {
        let lines = run_summary_lines(
            "asr_mem_smoke",
            BenchmarkType::AsrMaxMemory,
            &BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: 2048,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
        );
        assert_eq!(lines[1], "type: asr_max_memory");
        assert!(lines.iter().any(|l| l == "max_host_bytes: 2048"));
    }

    /// One corpus row as the wire carries it. The clocks nobody asserts on are
    /// fixed per turn (prefill 9 ms, TTFT 11 ms); what a case varies is the
    /// drafter's counters and the two sums the derived lines divide.
    fn corpus_row(
        turns: u32,
        predicted_ms: f64,
        e2e_ms: f64,
        generated: u64,
        drafted: Option<u64>,
        accepted: Option<u64>,
        turn_accept_length_sum: Option<f64>,
    ) -> BenchmarkResultData {
        BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_turns: turns,
            vl_spec_decode_prompt_ms: 9.0 * f64::from(turns),
            vl_spec_decode_predicted_ms: predicted_ms,
            vl_spec_decode_ttft_ms: 11.0 * f64::from(turns),
            vl_spec_decode_decode_ms: predicted_ms * 1.05,
            vl_spec_decode_e2e_ms: e2e_ms,
            vl_spec_decode_generated_tokens: generated,
            vl_spec_decode_drafted_tokens: drafted,
            vl_spec_decode_accepted_tokens: accepted,
            vl_spec_decode_turn_accept_length_sum: turn_accept_length_sum,
        }
    }

    /// No speedup: the other cell's row is a different job. A row with
    /// speculation off shows nothing about a drafter, and a zero denominator
    /// prints nothing rather than a `0.000` that would read as measured.
    #[rstest]
    #[case::speculation_on(
        corpus_row(100, 10_000.0, 12_000.0, 1_400, Some(2_800), Some(1_050), Some(420.0)),
        &[
            "turns: 100",
            "accept_length: 4.000",
            "draft_acceptance_rate: 0.375",
            "accept_length_per_request: 4.200",
            "prompt_ms_per_turn: 9.000",
            "decode_tok_per_s: 140.000",
            "e2e_ms_per_turn: 120.000",
        ],
        &["speedup"]
    )]
    #[case::speculation_off(
        corpus_row(100, 20_000.0, 24_000.0, 1_400, None, None, None),
        &["decode_tok_per_s: 70.000", "prompt_ms_per_turn: 9.000"],
        &["accept_length", "draft_acceptance_rate", "speedup"]
    )]
    #[case::nothing_drafted(
        corpus_row(1, 1.0, 1.0, 1, Some(0), Some(0), Some(1.0)),
        &["accept_length: 1.000", "accept_length_per_request: 1.000"],
        &["draft_acceptance_rate"]
    )]
    fn run_summary_derives_a_corpus_rows_sanity_numbers(
        #[case] data: BenchmarkResultData,
        #[case] present: &[&str],
        #[case] absent: &[&str],
    ) {
        let lines = run_summary_lines(
            "vl_spec_decode_mmspec_test_gqa_k8",
            BenchmarkType::VlSpecDecode,
            &data,
        );
        assert_eq!(lines[1], "type: vl_spec_decode");
        present.iter().for_each(|want| {
            assert!(
                lines.iter().any(|l| l.as_str() == *want),
                "missing {want}: {lines:?}"
            );
        });
        absent.iter().for_each(|fragment| {
            assert!(
                !lines.iter().any(|l| l.contains(fragment)),
                "unexpected {fragment}: {lines:?}"
            );
        });
    }

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

    /// What `show` writes for one definition.
    fn shown(
        def: &BenchmarkDefinition,
        source: pipette_plan_types::BenchmarkSource,
    ) -> anyhow::Result<String> {
        let mut captured = CapturedConsole::default();
        write_benchmark(captured.out(), def, source)?;
        Ok(captured.out_text())
    }

    /// An asr_eval prints its dataset and profile, then its beams or its
    /// temperature (the two decodings never share an entry), then a catalog
    /// entry's corpus and recipe.
    #[rstest::rstest]
    #[case::greedy(
        serde_json::json!({}),
        &["task:", "dataset:", "count:", "profile:", "max_tok:", "temp:"],
        "temp:    0 (greedy)"
    )]
    // A seed prints with its rule, the repeat offset named when the entry
    // names none.
    #[case::seeded(
        serde_json::json!({"parameter_temperature": 1.0, "parameter_seed": 0}),
        &["task:", "dataset:", "count:", "profile:", "max_tok:", "temp:", "seed:"],
        "seed:    0 (seed_rule=repeat_offset)"
    )]
    #[case::seeded_per_sample(
        serde_json::json!({
            "parameter_temperature": 1.0,
            "parameter_seed": 0,
            "parameter_seed_rule": "per_sample_sha256"
        }),
        &["task:", "dataset:", "count:", "profile:", "max_tok:", "temp:", "seed:"],
        "seed:    0 (seed_rule=per_sample_sha256)"
    )]
    #[case::beams(
        serde_json::json!({
            "parameter_num_beams": 3,
            "parameter_beam_repetition_penalty": 1.2,
            "parameter_beam_length_penalty": 1.0
        }),
        &["task:", "dataset:", "count:", "profile:", "max_tok:", "beams:"],
        "beams:   3 (repetition penalty 1.2, length penalty 1)"
    )]
    // Unpinned penalties print as the run uses them: the card's.
    #[case::beams_at_the_cards_penalties(
        serde_json::json!({"parameter_num_beams": 3}),
        &["task:", "dataset:", "count:", "profile:", "max_tok:", "beams:"],
        "beams:   3 (repetition penalty 1.2, length penalty 1)"
    )]
    #[case::catalog_entry(
        serde_json::json!({
            "dataset": {
                "format": "asr-eval-corpus/1", "adapter": "evals-server@1",
                "digest": "1".repeat(64), "bytes": 2048,
                "upstream": {"source": "evals_server", "eval_id": "asr_esb_librispeech_test_clean",
                             "dataset_version": "2026.10.1", "prompt_profile": "minicpm_o",
                             "identity": "2".repeat(64)},
            },
            "recipe": {
                "family": "MiniCPM-o 4.5",
                "models": [{"repo": "openbmb/MiniCPM-o-4_5-gguf",
                            "mmproj": "audio/MiniCPM-o-4_5-audio-F16.gguf"}],
                "enable_thinking": false,
                "sources": ["https://huggingface.co/openbmb/MiniCPM-o-4_5"],
            },
        }),
        &[
            "task:", "dataset:", "count:", "profile:", "max_tok:", "temp:", "corpus:", "recipe:",
            "model:", "source:",
        ],
        "model:   openbmb/MiniCPM-o-4_5-gguf with audio/MiniCPM-o-4_5-audio-F16.gguf"
    )]
    fn asr_eval_is_shown_with_its_profile(
        #[case] decoding: serde_json::Value,
        #[case] keys: &[&str],
        #[case] line: &str,
    ) -> anyhow::Result<()> {
        let mut wire = serde_json::json!({
            "benchmark_id": "asr_eval_esb_librispeech_test_clean_minicpm_o",
            "benchmark_type": "asr_eval",
            "parameter_asr_task": "asr_esb_librispeech_test_clean",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 2620,
            "parameter_prompt_profile": "minicpm_o",
            "parameter_max_tokens": 128
        });
        if let (Some(wire), Some(decoding)) = (wire.as_object_mut(), decoding.as_object()) {
            wire.extend(decoding.clone());
        }
        let def: BenchmarkDefinition = serde_json::from_value(wire)?;
        let written = shown(&def, pipette_plan_types::BenchmarkSource::Remote)?;
        let got: Vec<&str> = written
            .lines()
            .skip(3)
            .map(|line| line.split_whitespace().next().unwrap_or_default())
            .collect();
        assert_eq!(got, keys, "{written}");
        assert!(written.contains("profile: minicpm_o"), "{written}");
        assert!(written.contains(line), "{written}");
        Ok(())
    }

    #[test]
    fn show_renders_every_seeded_definition_with_its_parameters() -> anyhow::Result<()> {
        use pipette_plan_types::BenchmarkSource;

        let tw = seeded_workspace("show")?;
        let seen: std::collections::BTreeSet<String> = tw
            .ws
            .benchmarks()
            .list(BenchmarkSource::Local)?
            .into_iter()
            .map(|(id, def)| -> anyhow::Result<String> {
                let written = shown(&def, BenchmarkSource::Local)?;
                let lines: Vec<&str> = written.lines().collect();
                assert_eq!(lines[0], format!("id:      {id}"));
                assert_eq!(lines[1], format!("type:    {}", def.benchmark_type()));
                assert_eq!(lines[2], "source:  local");
                let expected: &[&str] = match def.benchmark_type() {
                    BenchmarkType::PrefillThroughput => &["prefill:", "batch:"],
                    BenchmarkType::DecodeThroughput | BenchmarkType::EndToEndLatency => {
                        &["prefill:", "decode:", "batch:"]
                    }
                    BenchmarkType::MaxMemoryUsage => &["prefill:"],
                    BenchmarkType::Eval => &["eval:", "dataset:", "max_tok:", "temp:", "samples:"],
                    BenchmarkType::VlmEval => &[
                        "task:", "dataset:", "count:", "max_tok:", "decode:", "samples:",
                    ],
                    BenchmarkType::VlThroughput => &["image:", "text:", "decode:", "images:"],
                    BenchmarkType::VlMaxMemory => &["image:", "text:", "images:"],
                    BenchmarkType::VlSpecDecode => &["draft_k:", "corpus:"],
                    // Not seeded locally; `asr_eval_is_shown_with_its_profile` covers it.
                    BenchmarkType::AsrEval => &[],
                    BenchmarkType::AsrThroughput => &["audio:", "text:", "decode:"],
                    BenchmarkType::AsrMaxMemory => &["audio:", "text:", "decode:"],
                };
                let keys: Vec<&str> = lines[3..]
                    .iter()
                    .map(|line| line.split_whitespace().next().unwrap_or_default())
                    .collect();
                assert_eq!(keys, expected, "{id}: {written}");
                Ok(def.benchmark_type().to_string())
            })
            .collect::<anyhow::Result<_>>()?;
        assert!(
            seen.len() >= 6,
            "the seed covers the timing, memory, eval and VL kinds: {seen:?}"
        );
        Ok(())
    }

    #[test]
    fn show_renders_a_vlm_evals_decode_and_recipe() -> anyhow::Result<()> {
        let sampled: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval", "benchmark_id": "vlm_eval_mmstar_qwen3_5",
            "parameter_lmms_task": "mmstar", "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1500, "parameter_max_tokens": 32768,
            "parameter_temperature": 0.6, "parameter_top_p": 0.95, "parameter_top_k": 20,
            "parameter_presence_penalty": 1.5, "parameter_seed": 0,
            "parameter_prompt_profile": "qwen3_5",
            "recipe": {
                "family": "Qwen3.5",
                "models": [{"repo": "unsloth/Qwen3.5-0.8B-GGUF"}, {"repo": "unsloth/Qwen3.5-2B-GGUF"}],
                "enable_thinking": true,
                "sources": ["https://huggingface.co/Qwen/Qwen3.5-2B"],
            },
        }))?;
        let written = shown(&sampled, pipette_plan_types::BenchmarkSource::Remote)?;
        let lines: Vec<&str> = written.lines().collect();
        for expected in [
            "decode:  temperature=0.6 top_p=0.95 top_k=20 presence_penalty=1.5 seed=0 \
             seed_rule=repeat_offset",
            "profile: qwen3_5",
            "recipe:  Qwen3.5 (enable_thinking = true)",
            "model:   unsloth/Qwen3.5-0.8B-GGUF",
            "model:   unsloth/Qwen3.5-2B-GGUF",
            "source:  https://huggingface.co/Qwen/Qwen3.5-2B",
        ] {
            assert!(lines.contains(&expected), "{expected:?} in {written}");
        }
        let greedy: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval", "benchmark_id": "vlm_eval_mmstar_minicpm_v_4_7",
            "parameter_lmms_task": "mmstar", "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1500, "parameter_max_tokens": 8192,
            "parameter_repeat_penalty": 1.03, "parameter_top_k": 1, "parameter_seed": 7,
            "recipe": {
                "family": "MiniCPM-V 4.7",
                "models": [{"repo": "openbmb/MiniCPM-V-4.7-1B-gguf", "mmproj": "mmproj-model-f16.gguf"}],
                "sources": ["https://huggingface.co/openbmb/MiniCPM-V-4.7-1B"],
            },
        }))?;
        let written = shown(&greedy, pipette_plan_types::BenchmarkSource::Remote)?;
        let lines: Vec<&str> = written.lines().collect();
        for expected in [
            // Greedy, and still every knob the entry sets.
            "decode:  greedy top_k=1 repeat_penalty=1.03 seed=7 seed_rule=repeat_offset",
            "recipe:  MiniCPM-V 4.7 (enable_thinking = unset)",
            "model:   openbmb/MiniCPM-V-4.7-1B-gguf with mmproj-model-f16.gguf",
        ] {
            assert!(lines.contains(&expected), "{expected:?} in {written}");
        }
        // An entry on the per-sample rule says so: `seed=0` alone would read
        // the same as the repeat offset's.
        let mut wire = serde_json::to_value(&sampled)?;
        wire["parameter_seed_rule"] = serde_json::json!("per_sample_sha256");
        let per_sample: BenchmarkDefinition = serde_json::from_value(wire)?;
        let written = shown(&per_sample, pipette_plan_types::BenchmarkSource::Remote)?;
        let expected = "decode:  temperature=0.6 top_p=0.95 top_k=20 presence_penalty=1.5 seed=0 \
             seed_rule=per_sample_sha256";
        assert!(
            written.lines().any(|line| line == expected),
            "{expected:?} in {written}"
        );
        Ok(())
    }

    #[test]
    fn show_renders_a_vl_memory_cells_output_length() -> anyhow::Result<()> {
        let def = BenchmarkDefinition::VlMaxMemory(pipette_plan_types::VlMaxMemoryBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_max_memory_224x224_text0_img1_decode256")?,
            parameter_image_width: PositiveCount::try_new(224)?,
            parameter_image_height: PositiveCount::try_new(224)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: Some(PositiveCount::try_new(256)?),
            parameter_num_images: NonNegativeCount::try_new(1)?,
        });
        let written = shown(&def, pipette_plan_types::BenchmarkSource::Remote)?;
        assert!(
            written.lines().any(|l| l == "decode:  256 tokens"),
            "{written}"
        );
        Ok(())
    }

    /// The corpus benchmark is not seeded locally, so its rendering is pinned
    /// from the wire document the server serves.
    #[test]
    fn show_renders_a_corpus_benchmarks_pin() -> anyhow::Result<()> {
        use pipette_plan_types::BenchmarkSource;

        let def: BenchmarkDefinition =
            serde_json::from_value(corpus_wire("vl_spec_decode_mmspec_test_gqa_k8_bs8"))?;
        let written = shown(&def, BenchmarkSource::Remote)?;
        let lines: Vec<&str> = written.lines().collect();
        assert_eq!(lines[2], "source:  remote");
        assert!(lines.contains(&"draft_k: 8 tokens per step (with speculation on)"));
        assert!(lines.contains(&"batch:   8 sample(s) in flight"));
        assert!(lines.contains(&"temp:    0 (greedy)"));
        assert!(lines.contains(&"corpus:  vl-spec-decode-corpus/1 (mmspec@1)"));
        assert!(lines.contains(&"subset:  GQA"));
        assert!(lines.contains(&"license: Apache-2.0"));
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("upstream:github.com/killthefullmoon/MMSpec@")),
            "{written}"
        );
        Ok(())
    }

    /// Where the temperature came from is the whole difference between two
    /// ids a reader is comparing, so the rendering has to say.
    #[test]
    fn show_names_the_sampling_an_eval_pins() -> anyhow::Result<()> {
        use pipette_plan_types::BenchmarkSource;

        let pinned: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_math_500_2026.06.1_t1",
            "benchmark_type": "eval",
            "parameter_eval_id": "math_500",
            "parameter_dataset_name": "2026.06.1",
            "parameter_max_tokens": 8192,
            "parameter_temperature": 1.0,
            "parameter_top_p": 0.95,
            "parameter_top_k": 64,
            "parameter_min_p": 0.05,
            "parameter_seed": 7,
        }))?;
        let lines = shown(&pinned, BenchmarkSource::Remote)?;
        let lines: Vec<&str> = lines.lines().collect();
        assert!(lines.contains(&"temp:    1 (pinned)"), "{lines:?}");
        let greedy_pin: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_math_500_2026.06.1_t0",
            "benchmark_type": "eval",
            "parameter_eval_id": "math_500",
            "parameter_dataset_name": "2026.06.1",
            "parameter_max_tokens": 8192,
            "parameter_temperature": 0.0,
        }))?;
        assert!(
            shown(&greedy_pin, BenchmarkSource::Remote)?.contains("temp:    0 (pinned, greedy)"),
            "a pinned greedy eval must say both"
        );
        assert!(lines.contains(&"top_p:   0.95"), "{lines:?}");
        assert!(lines.contains(&"top_k:   64"), "{lines:?}");
        assert!(lines.contains(&"min_p:   0.05"), "{lines:?}");
        assert!(
            lines.contains(&"seed:    7 (offset by the #k repeat index)"),
            "{lines:?}"
        );

        // Nothing pinned claims no cut, because none is sent.
        let unpinned: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_math_500_2026.06.1",
            "benchmark_type": "eval",
            "parameter_eval_id": "math_500",
            "parameter_dataset_name": "2026.06.1",
            "parameter_max_tokens": 8192,
        }))?;
        let written = shown(&unpinned, BenchmarkSource::Remote)?;
        assert!(
            written.contains("temp:    0.6 (eval-id policy)"),
            "{written}"
        );
        ["top_p:", "top_k:", "min_p:", "seed:"]
            .iter()
            .for_each(|key| assert!(!written.contains(key), "{key} must not print: {written}"));
        Ok(())
    }

    #[test]
    fn show_names_mcq_choices_and_an_unsubset_corpus() -> anyhow::Result<()> {
        use pipette_plan_types::BenchmarkSource;

        let eval: BenchmarkDefinition = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_mcq",
            "benchmark_type": "eval",
            "parameter_eval_id": "mmlu",
            "parameter_dataset_name": "mmlu",
            "parameter_max_tokens": 8,
            "parameter_mcq_choices": ["A", "B", "C"],
        }))?;
        let written = shown(&eval, BenchmarkSource::Local)?;
        assert!(written.contains("\nmcq:     A, B, C\n"), "{written}");
        assert!(!written.contains("\nsamples:"), "{written}");

        let mut wire = corpus_wire("vl_spec_decode_all");
        let strip = |value: &mut serde_json::Value, key: &str| -> anyhow::Result<()> {
            value
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("`{key}` sits under a non-object"))?
                .remove(key);
            Ok(())
        };
        strip(&mut wire["dataset"], "subset")?;
        strip(&mut wire["dataset"]["upstream"], "license")?;
        strip(&mut wire, "parameter_batch_size")?;
        let corpus: BenchmarkDefinition = serde_json::from_value(wire)?;
        let written = shown(&corpus, BenchmarkSource::Local)?;
        assert!(written.contains("\nsubset:  (all)\n"), "{written}");
        assert!(written.contains("\nbatch:   1 sample(s) in flight\n"));
        assert!(!written.contains("\nlicense:"), "{written}");
        Ok(())
    }

    #[test]
    fn init_local_seeds_the_catalog_and_show_refuses_an_unknown_ref() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("bench-dispatch")?;
        let http = HttpClient::new("pipette-test")?;
        let config = ClientConfiguration::default();
        let programs = ExternalPrograms::at_missing_paths();
        let run = |command: BenchmarkCommand| {
            BenchmarkArgs { command }.execute(
                &tw.ws,
                &http,
                &config,
                &programs,
                &mut CapturedConsole::default(),
            )
        };
        let mut seeded = CapturedConsole::default();
        BenchmarkArgs {
            command: BenchmarkCommand::InitLocal,
        }
        .execute(&tw.ws, &http, &config, &programs, &mut seeded)?;
        assert!(
            seeded
                .out_text()
                .starts_with("Wrote standard local benchmarks: "),
            "got {}",
            seeded.out_text()
        );
        assert!(
            !tw.ws
                .benchmarks()
                .list(pipette_plan_types::BenchmarkSource::Local)?
                .is_empty(),
            "init-local seeded nothing"
        );
        let err = run(BenchmarkCommand::Show(BenchmarkShowArgs {
            benchmark_ref: "local/prefill_throughput_nowhere".parse()?,
        }))
        .err()
        .ok_or_else(|| anyhow::anyhow!("an unknown reference should be refused"))?;
        assert!(
            format!("{err:#}").contains("local/prefill_throughput_nowhere"),
            "got {err:#}"
        );
        Ok(())
    }

    fn write_2026_08_1_img80(tw: &TempWorkspace) -> anyhow::Result<()> {
        write_public_2026_08_1_img80_cache(&tw.ws.root().join("benchmarks"))?;
        Ok(())
    }

    #[test]
    fn list_and_show_read_a_synced_remote_detail() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("bench-remote-detail")?;
        let mut sent = img80_wire();
        sent["org_id"] = serde_json::json!("org_test");
        cache_remote(&tw.ws.benchmarks(), sent)?;

        let rows = BenchmarkListArgs {
            benchmark_type: None,
        }
        .rows(&tw.ws)?;
        let refs: Vec<&str> = rows.iter().map(|r| r.benchmark_ref.as_str()).collect();
        assert_eq!(refs, [format!("remote/{IMG80}")]);

        let mut console = CapturedConsole::default();
        BenchmarkShowArgs {
            benchmark_ref: format!("remote/{IMG80}").parse()?,
        }
        .execute(&tw.ws, &mut console)?;
        let written = console.out_text();
        assert!(written.contains("\nsource:  remote\n"), "{written}");
        assert!(written.contains("\nimages:  80\n"), "{written}");
        Ok(())
    }

    /// An older build's file may have lost a parameter, so it is never read.
    #[test]
    fn show_and_run_do_not_read_a_legacy_remote_file() -> anyhow::Result<()> {
        let tw = TempWorkspace::new("bench-legacy")?;
        write_2026_08_1_img80(&tw)?;
        let reference = format!("remote/{IMG80}");
        let unknown = format!("unknown benchmark reference `{reference}`");

        let shown = BenchmarkShowArgs {
            benchmark_ref: reference.parse()?,
        }
        .execute(&tw.ws, &mut CapturedConsole::default())
        .err()
        .ok_or_else(|| anyhow::anyhow!("show read an older build's file"))?;
        assert_eq!(format!("{shown:#}"), unknown);

        let run = parse(&[
            "--benchmark",
            &reference,
            "--model",
            "gguf-text://repo=o/r&path=m-Q4_0.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?
        .into_client_run_spec(&tw.ws)
        .err()
        .ok_or_else(|| anyhow::anyhow!("run read an older build's file"))?;
        assert_eq!(format!("{run:#}"), unknown);
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

    /// The JSON case is produced by serializing the URI-parsed value, so it
    /// can't drift from the serde shape.
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

    /// The llama.cpp CLI + GGUF-text cell the `build_benchmark_flags` tests
    /// below route through.
    fn gguf_text_cell() -> anyhow::Result<(Runtime, Model)> {
        Ok((
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?,
            parse_model_arg("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf")?,
        ))
    }

    /// The same runtime with a vision model, for the cells keyed on
    /// `gguf_vision`.
    fn gguf_vision_cell() -> anyhow::Result<(Runtime, Model)> {
        Ok((
            parse_runtime_arg("llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64")?,
            parse_model_arg(
                "gguf-vision://repo=unsloth/gemma-4-E2B-it-GGUF&model=g.gguf&mmproj=mm.gguf",
            )?,
        ))
    }

    /// The OpenVINO cell, for the readiness cases.
    fn openvino_cell() -> anyhow::Result<(Runtime, Model)> {
        Ok((
            parse_runtime_arg("uv-openvino://version=2026.4.1")?,
            parse_model_arg("openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw")?,
        ))
    }

    #[test]
    fn eval_run_reconstructs_benchmark_flags() -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let bf = build_benchmark_flags(
            BenchmarkType::Eval,
            &runtime,
            &model,
            CellKnobs {
                http_timeout_seconds: Some(600),
                ..Default::default()
            },
        )?
        .ok_or_else(|| anyhow::anyhow!("benchmark_flags present for an eval run"))?;
        assert_eq!(bf.http_timeout().map(u64::from), Some(600));
        assert_eq!(bf.axes().0, BenchmarkType::Eval);
        Ok(())
    }

    fn http(secs: u64) -> CellKnobs {
        CellKnobs {
            http_timeout_seconds: Some(secs),
            ..Default::default()
        }
    }

    fn attempt(secs: u64) -> CellKnobs {
        CellKnobs {
            attempt_timeout_seconds: Some(secs),
            ..Default::default()
        }
    }

    fn readiness_wait(secs: u64) -> CellKnobs {
        CellKnobs {
            readiness_max_wait_secs: Some(secs),
            ..Default::default()
        }
    }

    fn skip_thermal() -> CellKnobs {
        CellKnobs {
            readiness_skip_thermal: true,
            ..Default::default()
        }
    }

    /// One table rather than a test per knob: they all fail through the same
    /// `TryFrom` path. The two timeouts read as interchangeable and are not, so
    /// each gets a case on the other's cell.
    #[rstest::rstest]
    #[case::http_on_prefill(BenchmarkType::PrefillThroughput, http(600))]
    #[case::http_on_decode(BenchmarkType::DecodeThroughput, http(600))]
    #[case::http_on_max_memory(BenchmarkType::MaxMemoryUsage, http(600))]
    #[case::attempt_on_eval(BenchmarkType::Eval, attempt(2400))]
    #[case::readiness_on_eval(BenchmarkType::Eval, readiness_wait(1800))]
    #[case::thermal_on_eval(BenchmarkType::Eval, skip_thermal())]
    fn a_knob_the_cell_does_not_carry_is_rejected(
        #[case] benchmark_type: BenchmarkType,
        #[case] knobs: CellKnobs,
    ) -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let err = build_benchmark_flags(benchmark_type, &runtime, &model, knobs)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            format!("{err:#}").contains("not valid for this benchmark"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    #[test]
    fn no_knob_carries_no_benchmark_flags() -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            CellKnobs::default(),
        )?;
        assert!(bf.is_none());
        Ok(())
    }

    #[rstest::rstest]
    #[case::readiness(
        BenchmarkType::PrefillThroughput,
        r#"{"readiness":{"max_wait_secs":1800,"skip_thermal":true}}"#,
        CellKnobs { readiness_max_wait_secs: Some(1800), readiness_skip_thermal: true, ..Default::default() },
    )]
    #[case::skin(
        BenchmarkType::PrefillThroughput,
        r#"{"readiness":{"skin_below_c":34}}"#,
        CellKnobs { readiness_skin_below_c: Some(34), ..Default::default() },
    )]
    #[case::http(BenchmarkType::Eval, r#"{"http_timeout_seconds":600}"#, http(600))]
    #[case::attempt(
        BenchmarkType::MaxMemoryUsage,
        r#"{"attempt_timeout_seconds":2400}"#,
        attempt(2400)
    )]
    #[case::empty(BenchmarkType::PrefillThroughput, "{}", CellKnobs::default())]
    fn benchmark_flags_json_matches_its_shorthand(
        #[case] benchmark_type: BenchmarkType,
        #[case] json: &str,
        #[case] knobs: CellKnobs,
    ) -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        assert_eq!(
            parse_benchmark_flags_json(json, benchmark_type, &runtime, &model)?,
            build_benchmark_flags(benchmark_type, &runtime, &model, knobs)?,
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::array(
        BenchmarkType::Eval,
        r#"[{"http_timeout_seconds":600}]"#,
        "JSON object"
    )]
    #[case::axis(
        BenchmarkType::Eval,
        r#"{"benchmark_type":"eval","http_timeout_seconds":600}"#,
        "must not carry `benchmark_type`"
    )]
    #[case::unknown_knob(BenchmarkType::Eval, r#"{"cooldown":30}"#, "unknown field `cooldown`")]
    #[case::knob_not_on_cell(
        BenchmarkType::Eval,
        r#"{"readiness":{"max_wait_secs":1800}}"#,
        "not valid for this benchmark"
    )]
    #[case::scalar(BenchmarkType::Eval, r#""http_timeout_seconds=600""#, "JSON object")]
    #[case::malformed(
        BenchmarkType::Eval,
        r#"{"http_timeout_seconds":"#,
        "EOF while parsing"
    )]
    #[case::zero_wait(
        BenchmarkType::PrefillThroughput,
        r#"{"readiness":{"max_wait_secs":0}}"#,
        "at least 1"
    )]
    #[case::skin_out_of_range(
        BenchmarkType::PrefillThroughput,
        r#"{"readiness":{"skin_below_c":111}}"#,
        "SkinCelsius"
    )]
    fn benchmark_flags_json_is_refused(
        #[case] benchmark_type: BenchmarkType,
        #[case] json: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let err = parse_benchmark_flags_json(json, benchmark_type, &runtime, &model)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected an error"))?;
        assert!(
            format!("{err:#}").contains(needle),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::http(&["--http-timeout-seconds", "600"])]
    #[case::attempt(&["--attempt-timeout-seconds", "600"])]
    #[case::readiness_wait(&["--readiness-max-wait-secs", "600"])]
    #[case::skip_thermal(&["--readiness-skip-thermal"])]
    #[case::skin(&["--readiness-skin-below-c", "34"])]
    fn benchmark_flags_json_excludes_its_shorthand(
        #[case] shorthand: &[&str],
    ) -> anyhow::Result<()> {
        let mut argv = vec![
            "--benchmark",
            "remote/eval_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--benchmark-flags",
            r#"{"http_timeout_seconds":600}"#,
        ];
        let args = parse(&argv)?;
        assert!(args.benchmark_flags.is_some());
        argv.extend_from_slice(shorthand);
        assert!(parse(&argv).is_err(), "{shorthand:?} was accepted");
        Ok(())
    }

    #[fixture]
    fn benchmark_argv() -> Vec<&'static str> {
        vec![
            "--benchmark",
            "local/prefill_throughput_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ]
    }

    #[rstest]
    #[case::zero("0", false)]
    #[case::minimum("1", true)]
    #[case::maximum("110", true)]
    #[case::too_high("111", false)]
    #[case::fraction("34.5", false)]
    fn skin_cli_flag_accepts_only_whole_degrees_in_range(
        benchmark_argv: Vec<&str>,
        #[case] value: &str,
        #[case] accepted: bool,
    ) {
        let mut argv = benchmark_argv;
        argv.extend(["--readiness-skin-below-c", value]);
        let parsed = parse(&argv);
        assert_eq!(parsed.is_ok(), accepted);
    }

    #[rstest]
    #[case::wait("wait", Some(pipette_readiness::HeldTouch::Wait))]
    #[case::fail("fail", Some(pipette_readiness::HeldTouch::Fail))]
    #[case::off("off", Some(pipette_readiness::HeldTouch::Off))]
    #[case::unknown("skip", None)]
    fn held_touch_cli_flag_accepts_modes(
        benchmark_argv: Vec<&str>,
        #[case] value: &str,
        #[case] want: Option<pipette_readiness::HeldTouch>,
    ) {
        let mut argv = benchmark_argv;
        argv.extend(["--readiness-held-touch", value]);
        let parsed = parse(&argv);
        assert_eq!(parsed.ok().and_then(|args| args.readiness_held_touch), want);
    }

    #[rstest]
    fn held_touch_flag_does_not_conflict_with_benchmark_flags_json(
        benchmark_argv: Vec<&str>,
    ) -> anyhow::Result<()> {
        let mut argv = benchmark_argv;
        argv.extend([
            "--benchmark-flags",
            r#"{"readiness":{"max_wait_secs":900}}"#,
            "--readiness-held-touch",
            "fail",
        ]);
        let parsed = parse(&argv)?;
        assert_eq!(
            parsed.readiness_held_touch,
            Some(pipette_readiness::HeldTouch::Fail)
        );
        Ok(())
    }

    #[test]
    fn benchmark_flags_json_reaches_the_spec() -> anyhow::Result<()> {
        let tw = seeded_workspace("flags-json")?;
        let (spec, _) = parse(&[
            "--benchmark",
            "local/prefill_throughput_smoke",
            "--model",
            "gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
            "--benchmark-flags",
            r#"{"readiness":{"max_wait_secs":900}}"#,
        ])?
        .into_client_run_spec(&tw.ws)?;
        let readiness = spec
            .benchmark_flags
            .as_ref()
            .and_then(BenchmarkFlags::readiness);
        assert_eq!(readiness.and_then(|r| r.max_wait_secs), Some(900));
        Ok(())
    }

    /// vl-throughput drives `llama-server`, so it takes an HTTP timeout.
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
            CellKnobs {
                http_timeout_seconds: Some(600),
                ..Default::default()
            },
        )?
        .ok_or_else(|| anyhow::anyhow!("vl-throughput accepts an http timeout"))?;
        assert_eq!(bf.http_timeout().map(u64::from), Some(600));
        assert_eq!(bf.axes().0, BenchmarkType::VlThroughput);
        Ok(())
    }

    /// `--attempt-timeout-seconds` is a plan's to set on every timing,
    /// throughput and peak-memory cell.
    #[rstest::rstest]
    #[case::prefill(BenchmarkType::PrefillThroughput, gguf_text_cell())]
    #[case::end_to_end(BenchmarkType::EndToEndLatency, gguf_text_cell())]
    #[case::vl_throughput(BenchmarkType::VlThroughput, gguf_vision_cell())]
    #[case::text_memory(BenchmarkType::MaxMemoryUsage, gguf_text_cell())]
    #[case::vl_memory(BenchmarkType::VlMaxMemory, gguf_vision_cell())]
    #[case::asr_memory(BenchmarkType::AsrMaxMemory, gguf_vision_cell())]
    fn the_attempt_limit_is_accepted_on_a_timing_or_memory_cell(
        #[case] benchmark_type: BenchmarkType,
        #[case] cell: anyhow::Result<(Runtime, Model)>,
    ) -> anyhow::Result<()> {
        let (runtime, model) = cell?;
        let bf = build_benchmark_flags(benchmark_type, &runtime, &model, attempt(2400))?
            .ok_or_else(|| anyhow::anyhow!("the cell accepts an attempt limit"))?;
        assert_eq!(bf.attempt_timeout().map(u64::from), Some(2400));
        assert_eq!(bf.axes().0, benchmark_type);
        Ok(())
    }

    #[test]
    fn readiness_on_timing_cell_is_accepted() -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            CellKnobs {
                readiness_max_wait_secs: Some(1800),
                ..Default::default()
            },
        )?
        .ok_or_else(|| anyhow::anyhow!("prefill accepts a readiness override"))?;
        assert_eq!(bf.readiness().and_then(|r| r.max_wait_secs), Some(1800));
        Ok(())
    }

    /// OpenVINO's timing cells gate on the same criteria as llama.cpp's, so an
    /// override accepted elsewhere must be accepted here too.
    #[rstest::rstest]
    #[case::prefill(BenchmarkType::PrefillThroughput)]
    #[case::decode(BenchmarkType::DecodeThroughput)]
    #[case::end_to_end(BenchmarkType::EndToEndLatency)]
    fn readiness_on_an_openvino_timing_cell_is_accepted(
        #[case] benchmark_type: BenchmarkType,
    ) -> anyhow::Result<()> {
        let (runtime, model) = openvino_cell()?;
        let bf = build_benchmark_flags(
            benchmark_type,
            &runtime,
            &model,
            CellKnobs {
                readiness_max_wait_secs: Some(1800),
                readiness_skip_thermal: true,
                ..Default::default()
            },
        )?
        .ok_or_else(|| anyhow::anyhow!("an OpenVINO timing cell accepts a readiness override"))?;
        let readiness = bf
            .readiness()
            .ok_or_else(|| anyhow::anyhow!("expected a readiness override"))?;
        assert_eq!(readiness.max_wait_secs, Some(1800));
        assert_eq!(readiness.skip_thermal, Some(true));
        assert_eq!(bf.axes().0, benchmark_type);
        Ok(())
    }

    /// It must not need `--readiness-max-wait-secs` alongside it, and it must
    /// not invent a deadline it wasn't given.
    #[test]
    fn readiness_skip_thermal_alone_is_accepted_on_a_timing_cell() -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let bf = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            CellKnobs {
                readiness_skip_thermal: true,
                ..Default::default()
            },
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

    /// `false` is not an override: an unused flag must not make a default-gated
    /// run look like one that opted into something.
    #[test]
    fn readiness_skip_thermal_false_records_no_override() -> anyhow::Result<()> {
        let (runtime, model) = gguf_text_cell()?;
        let flags = build_benchmark_flags(
            BenchmarkType::PrefillThroughput,
            &runtime,
            &model,
            CellKnobs::default(),
        )?;
        assert!(
            flags.is_none_or(|bf| bf.readiness().is_none()),
            "an unset thermal waiver must not create a readiness override",
        );
        Ok(())
    }

    /// Plan `ModelFlags` exist only on eval and vlm_eval cells, so this is
    /// rejected at the CLI boundary.
    #[test]
    fn model_enable_thinking_on_a_timing_benchmark_is_rejected() -> anyhow::Result<()> {
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

    #[rstest]
    #[case::flat(&["--model-enable-thinking", "true"], None)]
    #[case::json(
        &["--model-flags", r#"{"benchmark_type":"vlm_eval","model_type":"gguf_vision","enable_thinking":true}"#],
        None
    )]
    #[case::entry_for_a_text_eval(
        &["--model-flags", r#"{"benchmark_type":"eval","model_type":"gguf_text","enable_thinking":true}"#],
        Some("does not match this cell")
    )]
    fn a_vlm_eval_takes_enable_thinking_as_a_model_flag(
        #[case] extra: &[&str],
        #[case] refusal: Option<&str>,
    ) -> anyhow::Result<()> {
        let tw = seeded_workspace("vlm-eval-thinking")?;
        let mut argv = vec![
            "--benchmark",
            "local/vlm_eval_smoke",
            "--model",
            "gguf-vision://repo=unsloth/gemma-4-E2B-it-GGUF&model=g.gguf&mmproj=mm.gguf",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ];
        argv.extend_from_slice(extra);
        let result = parse(&argv)?.into_client_run_spec(&tw.ws);

        match (result, refusal) {
            (Ok((spec, _)), None) => {
                let flags = spec
                    .model_flags
                    .ok_or_else(|| anyhow::anyhow!("expected model flags"))?;
                assert_eq!(
                    flags.axes(),
                    (BenchmarkType::VlmEval, ModelType::GgufVision)
                );
                assert_eq!(flags.enable_thinking(), Some(true));
            }
            (Err(err), Some(expected)) => {
                let chain = format!("{err:#}");
                assert!(
                    chain.contains(expected),
                    "expected {expected:?}, got: {chain}"
                );
            }
            (Ok(_), Some(_)) => anyhow::bail!("{extra:?} should have been refused"),
            (Err(err), None) => return Err(err),
        }
        Ok(())
    }

    /// The cell comes from `--benchmark`, `--runtime` and `--model`; the JSON
    /// carries only knobs.
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

    /// A hand-written `{}` means the same as no flag at all.
    #[test]
    fn an_empty_runtime_flags_object_is_no_flags() -> anyhow::Result<()> {
        let tw = seeded_workspace("rt-flags-empty")?;
        let (spec, _) = prefill_args("{}")?.into_client_run_spec(&tw.ws)?;

        assert!(spec.runtime_flags.is_none());
        Ok(())
    }

    /// An entry for another cell must not be applied to this one, and a knob
    /// the cell does not carry must not be dropped.
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

    /// Walks author, resolve, emit and parse, and asserts the client ends up
    /// holding exactly what the plan authored: the test that fails if either
    /// side changes the wire alone.
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
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
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

    /// A `prefill_throughput` run over the gguf-text x llama.cpp cell, varying
    /// only the flags.
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
        // `raw` rides inside the one RuntimeFlags entry; there is no separate
        // raw channel.
        assert!(spec.runtime_flags.is_some());
        Ok(())
    }

    /// The plan's one-element array and the bare object a claim carries: the
    /// spellings a deployed client relies on.
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

    /// An eval run over the gguf-text x llama.cpp cell: only eval and vlm_eval
    /// cells carry generation flags.
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

    /// An MLX model on a llama.cpp runtime is a well-formed ref pair that no
    /// host loads.
    #[test]
    fn model_incompatible_with_the_runtime_is_rejected() -> anyhow::Result<()> {
        let tw = seeded_workspace("model-wrong")?;
        let args = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "mlx://repo=meta-llama/Llama-3.2-1B",
            "--runtime",
            "llamacpp-cli-stock-tools://version=b9050&flavor=macos-arm64",
        ])?;
        let err = args
            .into_client_run_spec(&tw.ws)
            .err()
            .context("an incompatible pair must be refused")?;
        assert!(
            matches!(
                err.downcast_ref::<RunSpecError>(),
                Some(RunSpecError::Incompatible { .. })
            ),
            "{err:#}"
        );
        Ok(())
    }

    const OPENVINO_RUNTIME: &str = "uv-openvino://version=2026.4.1";
    const OVMS_RUNTIME: &str = "ovms://version=v2026.4.1&flavor=windows-x64";

    fn openvino_run_args(runtime: &str) -> anyhow::Result<BenchmarkRunArgs> {
        parse(&[
            "--benchmark",
            "local/prefill_throughput_smoke",
            "--model",
            "openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw",
            "--runtime",
            runtime,
        ])
    }

    /// OpenVINO and OVMS run on Linux and Windows only.
    #[rstest]
    #[case::openvino(OPENVINO_RUNTIME)]
    #[case::ovms(OVMS_RUNTIME)]
    fn openvino_and_ovms_follow_this_hosts_os(#[case] runtime: &str) -> anyhow::Result<()> {
        let tw = seeded_workspace("host-os")?;
        expect_gate_on_this_host(openvino_run_args(runtime)?.into_client_run_spec(&tw.ws))?;
        Ok(())
    }

    /// On a host the gate refuses, the run fails before anything installs:
    /// neither store is touched. Elsewhere the run would install for real.
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    #[rstest]
    #[case::openvino(OPENVINO_RUNTIME)]
    #[case::ovms(OVMS_RUNTIME)]
    fn a_refused_run_installs_nothing(#[case] runtime: &str) -> anyhow::Result<()> {
        let tw = seeded_workspace("host-os-run")?;
        let err = openvino_run_args(runtime)?
            .execute(
                &tw.ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &mut CapturedConsole::default(),
            )
            .err()
            .context("this host must be refused")?;
        assert!(
            err.downcast_ref::<crate::run::ForeignOsError>().is_some(),
            "{err:#}"
        );
        [
            tw.ws.runtimes().runtimes_dir().to_path_buf(),
            tw.ws.models().models_dir().to_path_buf(),
        ]
        .iter()
        .try_for_each(|dir| -> anyhow::Result<()> {
            let entries = match std::fs::read_dir(dir) {
                Ok(entries) => entries.count(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e.into()),
            };
            assert_eq!(entries, 0, "{} was touched", dir.display());
            Ok(())
        })
    }

    /// An on-device runtime can't run from the CLI: `prepare` refuses it before
    /// anything installs. Shown with JSON refs since on-device runtimes have no
    /// URI scheme.
    #[test]
    fn run_rejects_on_device_runtime() -> anyhow::Result<()> {
        let tw = seeded_workspace("run-dispatch")?;
        let err = parse(&[
            "--benchmark",
            "local/eval_smoke",
            "--model",
            r#"{"type":"apple_foundation_text"}"#,
            "--runtime",
            r#"{"type":"apple_foundation"}"#,
        ])?
        .execute(
            &tw.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
            &mut CapturedConsole::default(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("run should error for an on-device runtime"))?;
        assert!(
            format!("{err:#}").contains("not a desktop CLI runtime"),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    /// A shard refuses a cell that keeps no checkpoint, and `--sync`, before
    /// anything installs.
    #[rstest]
    #[case::a_cell_without_a_checkpoint(
        &["--benchmark", "local/prefill_throughput_smoke"],
        "--eval-shard splits only"
    )]
    #[case::sync(&["--benchmark", "local/eval_smoke", "--sync"], "--sync has nothing to send")]
    fn an_eval_shard_refuses_before_anything_installs(
        #[case] argv: &[&str],
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let tw = seeded_workspace("eval-shard-refused")?;
        let ws = tw.as_shard("0/2")?;
        let refs = [
            "--model",
            r#"{"type":"apple_foundation_text"}"#,
            "--runtime",
            r#"{"type":"apple_foundation"}"#,
        ];
        let err = parse(&[argv, &refs].concat())?
            .execute(
                &ws,
                &test_http()?,
                &ClientConfiguration::default(),
                &ExternalPrograms::at_missing_paths(),
                &mut CapturedConsole::default(),
            )
            .err()
            .context("a shard must refuse this run")?;
        assert!(format!("{err:#}").contains(needle), "{err:#}");
        [
            ws.runtimes().runtimes_dir().to_path_buf(),
            ws.models().models_dir().to_path_buf(),
        ]
        .iter()
        .try_for_each(|dir| -> anyhow::Result<()> {
            let entries = match std::fs::read_dir(dir) {
                Ok(entries) => entries.count(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
                Err(e) => return Err(e.into()),
            };
            assert_eq!(entries, 0, "{} was touched", dir.display());
            Ok(())
        })
    }

    /// A shard's run keeps only the rows in its part: nothing is saved or
    /// sent, even with `sync` asked for, and nothing is finalized. A text eval
    /// has no completeness check, so this is all that keeps a group's rows out
    /// of mgmt.
    #[test]
    fn a_shard_run_saves_syncs_and_finalizes_nothing() -> anyhow::Result<()> {
        use pipette_ops::eval_completions::run_request_digest;
        use pipette_ops::run_context::RunContext;
        use pipette_plan_types::{BenchmarkEvalCompletion, DeclaredBound, RunRequest};

        let tw = seeded_workspace("eval-shard-run")?;
        let server = httpmock::MockServer::start();
        tw.register(&server.base_url())?;
        let submit = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/benchmarks");
            then.status(200).body(r#"{"job_id":"job-1"}"#);
        });
        let ws = tw.as_shard("0/2")?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: tw
                .ws
                .benchmarks()
                .get(&"local/eval_smoke".parse::<SourcedBenchmarkId>()?)?
                .context("eval_smoke should be seeded")?,
        };
        let row = |id: &str| BenchmarkEvalCompletion {
            id: id.to_owned(),
            completion: "answer".to_owned(),
            ..Default::default()
        };
        tw.ws.eval_completions().open(&req)?.append(row("s0"))?;
        ws.eval_completions().open(&req)?.append(row("s1"))?;
        let evals = tw.ws.root().join("state/evals");
        let cell_files = || -> anyhow::Result<Vec<(std::path::PathBuf, Vec<u8>)>> {
            let mut files = std::fs::read_dir(&evals)?
                .map(|entry| -> anyhow::Result<_> {
                    let path = entry?.path();
                    let bytes = std::fs::read(&path)?;
                    Ok((path, bytes))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            files.sort();
            Ok(files)
        };
        let before = cell_files()?;
        // The descriptors let a sync reach the server.
        let payload: BenchmarkSubmissionPayload = serde_json::from_value(serde_json::json!({
            "benchmark_id": "eval_smoke",
            "device": {},
            "thermal": {},
            "submitted_at": "2026-07-30T00:00:00Z",
            "model_descriptor": r#"{"type":"apple_foundation_text"}"#,
            "runtime_descriptor": r#"{"type":"apple_foundation"}"#,
            "completions": [],
        }))?;
        let mut console = CapturedConsole::default();

        record_finished_run(
            &ws,
            &test_http()?,
            &mut console,
            BenchmarkType::Eval,
            BenchmarkResultLocation::RemotePending,
            true,
            (
                payload,
                BenchmarkResultExtras::default(),
                Some(run_request_digest(&req)?),
            ),
        )?;

        submit.assert_calls(0);
        [
            BenchmarkResultLocation::Local,
            BenchmarkResultLocation::RemotePending,
            BenchmarkResultLocation::RemoteSynced,
        ]
        .iter()
        .try_for_each(|&location| -> anyhow::Result<()> {
            assert!(ws.results().list_ids(location)?.is_empty(), "{location:?}");
            Ok(())
        })?;
        assert_eq!(cell_files()?, before, "a shard's run finalized its cell");
        let out = console.out_text();
        assert!(
            out.contains("nothing saved, submitted or finalized"),
            "{out}"
        );
        Ok(())
    }

    /// A vLLM ref pair clears the on-device bail and reaches artifact ensure,
    /// where it stops. [`stubbed_artifacts`] keeps it from a real
    /// `docker pull vllm/vllm-openai:v0.21.0`.
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

    /// An absent benchmark fails in `into_client_run_spec`, before any download
    /// or run.
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
        .execute(
            &tw.ws,
            &test_http()?,
            &ClientConfiguration::default(),
            &ExternalPrograms::at_missing_paths(),
            &mut CapturedConsole::default(),
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("run should fail resolving an absent benchmark"))?;
        assert!(
            format!("{err:#}").contains("unknown benchmark reference"),
            "expected benchmark-resolution failure from run, got: {err:#}"
        );
        Ok(())
    }

    /// An MLX ref pair resolves its catalog URI and reaches artifact ensure.
    /// [`stubbed_artifacts`] keeps it from building a uv venv.
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

    /// Regression for PIP-409: `build_argv` ships the model as JSON (projector
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
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
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
        let model = parse_model_arg(&argv[model_idx + 1])?;
        assert!(matches!(model, Model::GgufVision(_)));
        assert!(!argv.iter().any(|a| a == "--mmproj"));
        Ok(())
    }

    /// Every value the help block passes to `--<flag>`, quoted (`--model 'x'`)
    /// or bare (`--benchmark x`). Comment lines are skipped: a mention is not
    /// an example. The count is checked so a reflowed example cannot silently
    /// stop being verified.
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
        ("load_mode", "\"none\""),
        ("flash_attention", "\"on\""),
        ("ctx_size", "4096"),
        ("parallel", "8"),
        (
            "repetition_detection",
            "{\"min_pattern_size\": 2, \"max_pattern_size\": 128, \"min_count\": 10}",
        ),
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
        ("precision", "\"fp16\""),
        ("envs", "[\"NCCL_DEBUG\"]"),
        // A value, not a flag: every cell takes it, the vLLM vlm_eval cell's
        // allowlist included.
        ("raw", "[\"some-passthrough\"]"),
        ("speculative_enabled", "false"),
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

    /// The `documented` list is the load-bearing check, against the real
    /// parser. The block is then searched for each name, which only catches a
    /// setting missing from the help entirely; row-level precision would mean
    /// parsing the layout, which breaks on any reflow.
    #[rstest]
    #[case::llama_bench(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufText, BenchmarkType::PrefillThroughput,
        &["threads", "number_gpu_layers", "mmap", "load_mode", "flash_attention", "envs", "raw"]
    )]
    #[case::llama_server(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufText, BenchmarkType::Eval,
        &["threads", "number_gpu_layers", "mmap", "load_mode", "flash_attention", "ctx_size",
          "no_cache", "parallel", "envs", "raw"]
    )]
    #[case::llama_vl(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufVision, BenchmarkType::VlThroughput,
        &["threads", "number_gpu_layers", "mmap", "load_mode", "flash_attention", "ctx_size",
          "no_cache", "envs", "raw"]
    )]
    #[case::llama_vlm_eval(
        RuntimeType::LlamacppCliStockTools, ModelType::GgufVision, BenchmarkType::VlmEval,
        &["threads", "number_gpu_layers", "mmap", "load_mode", "flash_attention", "ctx_size",
          "no_cache", "parallel", "envs", "raw"]
    )]
    #[case::docker_vllm(
        RuntimeType::DockerVllm, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "gpus", "shm_size",
          "ipc", "parallel", "repetition_detection", "envs", "raw"]
    )]
    #[case::docker_vllm_prefill(
        RuntimeType::DockerVllm, ModelType::Torch, BenchmarkType::PrefillThroughput,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "gpus", "shm_size",
          "ipc", "envs", "raw"]
    )]
    #[case::uv_sglang_decode(
        RuntimeType::UvSglang, ModelType::Torch, BenchmarkType::DecodeThroughput,
        &["tensor_parallel_size", "prefix_caching", "envs", "raw"]
    )]
    #[case::uv_vllm(
        RuntimeType::UvVllm, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "parallel",
          "repetition_detection", "envs", "raw"]
    )]
    #[case::uv_vllm_vlm_eval(
        RuntimeType::UvVllm, ModelType::Torch, BenchmarkType::VlmEval,
        &["tensor_parallel_size", "dtype", "max_model_len", "prefix_caching", "parallel", "envs",
          "raw"]
    )]
    #[case::uv_sglang(
        RuntimeType::UvSglang, ModelType::Torch, BenchmarkType::Eval,
        &["tensor_parallel_size", "prefix_caching", "parallel", "envs", "raw"]
    )]
    #[case::openvino(
        RuntimeType::UvOpenvino, ModelType::Openvino, BenchmarkType::PrefillThroughput,
        &["device", "max_prompt_len", "min_response_len", "generate_hint"]
    )]
    #[case::ovms(
        RuntimeType::Ovms, ModelType::Openvino, BenchmarkType::DecodeThroughput,
        &["device", "max_prompt_len", "min_response_len", "generate_hint", "prefix_caching"]
    )]
    #[case::trt_edgellm(
        RuntimeType::TrtEdgellm, ModelType::GgufVision, BenchmarkType::VlThroughput,
        &["precision", "envs", "raw"]
    )]
    #[case::onnx_genai(
        RuntimeType::UvOnnxGenai, ModelType::OnnxGenai, BenchmarkType::PrefillThroughput, &[]
    )]
    #[case::mlx(RuntimeType::MlxMacosPipette, ModelType::Mlx, BenchmarkType::Eval, &[])]
    #[case::core_ai(
        RuntimeType::AppleCoreAiMacosPipette, ModelType::AppleCoreAi,
        BenchmarkType::PrefillThroughput, &[]
    )]
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
