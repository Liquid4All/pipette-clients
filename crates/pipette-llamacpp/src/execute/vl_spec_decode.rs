//! The corpus benchmark `vl_spec_decode` on `llama-server`: one VL model over
//! an image and text corpus, with speculation on or off. This is the llama.cpp
//! half: the server, the speculation argv, and the request envelope. Turns and
//! their reduction live in `pipette_vl_spec_decode_corpus::turn` and
//! `pipette_ops::vl_spec_decode`.
//!
//! Neither stock `llama-server` nor the pinned fork sends a step count, so
//! forwards are `draft_n / parameter_draft_length`: a block drafter proposes
//! the whole block every step.
//!
//! See `docs/methodology/vl-spec-decode.md`, which links the mgmt repo's
//! wire contract and pairing docs.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::Context;
use serde_json::json;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::{self, MeasurementObserver};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::vl_spec_decode::{Speculation, TimingsSource};
use pipette_plan_types::{
    BatchSize, BenchmarkFlagRef, LlamacppCliStockToolsExtraRuntimeConfiguration, Model,
    RunResponse, RuntimeFlags, SamplingTemperature, VlSpecDecodeBenchmark,
    VlSpecDecodeLlamacppCliStockTools,
};
use pipette_subprocess::{Process, start_process};
use pipette_vl_spec_decode_corpus::{VlSpecDecodeCorpus, turn};

use crate::models::{ResolvedGgufVisionModel, ResolvedGgufVisionSpeculativeModel};
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

/// Context each slot gets when the cell pins none. MMSpec `test`'s worst turn
/// measured 4,110 tokens of prompt plus output (images and multi-turn history
/// included) against a 2,048-token cap; twice that fits. llama-server divides
/// `--ctx-size` across slots, so the total is this times `parameter_batch_size`.
const DEFAULT_CTX_SIZE_PER_SLOT: u32 = 8192;

/// The spelling of "propose this many tokens per step", newest first. The fork
/// uses `--spec-*` and upstream `--draft-*`, so the binary is asked which it takes.
const DRAFT_LENGTH_FLAGS: &[&str] = &["--spec-draft-n-max", "--draft-max", "--draft-n", "--draft"];

/// The spelling of "load the drafter from this file", newest first.
const DRAFT_MODEL_FLAGS: &[&str] = &["--model-draft", "-md"];

/// How long `llama-server --help` gets to print it. It loads no model, so only
/// a wedged build runs this out.
const HELP_TIMEOUT: Duration = Duration::from_secs(60);

/// The model a corpus run loads, resolved with its speculation: one variant per
/// plan model kind and speculation it can run with.
pub enum LoadedModel {
    /// A `gguf_vision` model: its backbone and projector, never a drafter.
    Vision(ResolvedGgufVisionModel),
    /// A `gguf_vision_speculative` model with speculation off: the backbone and
    /// projector alone; a merged drafter stays in the file, resident and unused.
    SpeculativeOff(ResolvedGgufVisionSpeculativeModel),
    SpeculativeOn(ResolvedGgufVisionSpeculativeModel),
}

impl LoadedModel {
    /// `model` resolved for a run with speculation `enabled`; which files
    /// load depends on both, so this matches the plan's model kind itself.
    pub fn resolve(model: &Model, enabled: bool) -> anyhow::Result<Self> {
        Ok(match (enabled, model) {
            (true, model) => {
                Self::SpeculativeOn(ResolvedGgufVisionSpeculativeModel::try_from(model)?)
            }
            (false, model @ Model::GgufVisionSpeculative(_)) => {
                Self::SpeculativeOff(ResolvedGgufVisionSpeculativeModel::try_from(model)?)
            }
            (false, model) => Self::Vision(ResolvedGgufVisionModel::try_from(model)?),
        })
    }

    fn speculation(&self) -> Speculation {
        match self {
            Self::Vision(_) | Self::SpeculativeOff(_) => Speculation::Off,
            Self::SpeculativeOn(_) => Speculation::On,
        }
    }

    fn vision(&self) -> &ResolvedGgufVisionModel {
        match self {
            Self::Vision(vision) => vision,
            Self::SpeculativeOff(model) | Self::SpeculativeOn(model) => &model.vision,
        }
    }

    /// `runtime_flags` recorded as the variant for this model's kind.
    fn runtime_flags(&self, runtime_flags: VlSpecDecodeLlamacppCliStockTools) -> RuntimeFlags {
        match self {
            Self::Vision(_) => {
                RuntimeFlags::VlSpecDecodeLlamacppCliStockToolsGgufVision(runtime_flags)
            }
            Self::SpeculativeOff(_) | Self::SpeculativeOn(_) => {
                RuntimeFlags::VlSpecDecodeLlamacppCliStockToolsGgufVisionSpeculative(runtime_flags)
            }
        }
    }
}

/// `runtime_flags` carries `speculative_enabled` resolved against the model,
/// which `model` was resolved with.
#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, the corpus, and the seams its entry takes"
)]
pub fn run(
    benchmark: &VlSpecDecodeBenchmark,
    runtime_flags: VlSpecDecodeLlamacppCliStockTools,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &LoadedModel,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let slots = benchmark.parameter_batch_size.as_u32();
    let runtime_flags = VlSpecDecodeLlamacppCliStockTools {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: Some(match runtime_flags.ctx_size {
            Some(total) => total,
            None => server::ctx_across_slots(DEFAULT_CTX_SIZE_PER_SLOT, slots)?,
        }),
        ..runtime_flags
    };
    let speculation = model.speculation();
    let params = turn::Params::new(
        benchmark,
        benchmark_flags.http_timeout_seconds,
        benchmark_flags.bench_timeout_seconds,
        speculation,
    )?;
    let vision = model.vision();
    // On: the drafter's argv. Off: none; a merged drafter stays in the
    // backbone, resident and unused.
    let speculation_argv = match model {
        LoadedModel::SpeculativeOn(model) => {
            speculation_args(program, params.draft_length, model.draft.as_deref())?
        }
        LoadedModel::Vision(_) | LoadedModel::SpeculativeOff(_) => Vec::new(),
    };
    let launch = launch(
        &runtime_flags,
        program,
        &vision.backbone,
        &vision.mmproj,
        slots,
        &speculation_argv,
        client_config,
    );
    // The model load is budgeted as on every llama-server cell, not by the
    // turn's ten minutes: a server that hangs loading should fail the cell soon.
    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, slots);
    let client = HttpClient::loopback("pipette", params.http_timeout)?
        .client()
        .clone();
    let envelope = envelope(&params)?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, timeout)?;
    let command = launch.build(port)?;

    let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
    let sampler = Sampler::start(llama_server.pid(), SupportedWatch::OBSERVE)?;
    llama_server
        .wait_ready(timeout, ready)
        .context(speculation.label())?;
    let worked = (|| {
        if speculation == Speculation::On {
            check_draft_length(params.draft_length, drafter_block_size(&llama_server))
                .context(speculation.label())?;
        }
        if benchmark.parameter_batch_size == BatchSize::one() {
            return turn::measure(
                benchmark,
                &params,
                corpus,
                readiness_gate,
                observer,
                &|messages, max_tokens| envelope.stream(&client, &url, messages, max_tokens),
            );
        }
        // The entry sends a batch as one call; llama-server takes one
        // conversation per request.
        turn::run(
            &params,
            corpus,
            readiness_gate,
            observer,
            &|messages, max_tokens| envelope.send(&client, &url, messages, max_tokens),
        )
    })();
    // The sampler is finished before the stop, so the observation covers
    // the traversal and not the teardown.
    let (memory, _) = sampler.stop();
    let response =
        worked.map_err(|err| server::with_server_tail(err, llama_server.stop_with_tail()))?;
    Ok(response
        .process(&mut llama_server)
        .memory(memory)
        .runtime_flags(model.runtime_flags(runtime_flags))
        .build())
}

/// Refuse a draft length above the drafter's block: the server clamps it at load
/// with only a warning, and forwards here divide by `parameter_draft_length`. A
/// drafter that announced no block is only warned about: a missing log line says
/// nothing about what was drafted.
fn check_draft_length(draft_length: u32, drafter_block: Option<u32>) -> anyhow::Result<()> {
    match drafter_block {
        Some(block) if draft_length > block => anyhow::bail!(
            "the benchmark's draft length of {draft_length} is above the drafter's block \
             of {block}, which the server clamps at load: this run would report the \
             block's measurement under a draft length it never drafted at. Run a \
             benchmark whose `parameter_draft_length` is at most {block}"
        ),
        Some(_) => Ok(()),
        None => {
            tracing::warn!(
                draft_length,
                "the drafter announced no block size, so a draft length above its block \
                 cannot be refused here; the accept length assumes the server drafted \
                 draft_length tokens every step"
            );
            Ok(())
        }
    }
}

/// `cache_prompt: false` so a later turn's TTFT does not depend on what the
/// server still held, which differs between two processes and so two rows.
/// Greedy turns add `top_k: 1` so a fork with its own sampling defaults still
/// takes the argmax; a sampled turn omits it, or the pin would override the
/// temperature.
fn envelope(params: &turn::Params) -> anyhow::Result<turn::Envelope> {
    let source = match params.speculation {
        Speculation::On => TimingsSource::RuntimeFixedDraft(params.draft_length),
        Speculation::Off => TimingsSource::Runtime,
    };
    let extras = if params.temperature == SamplingTemperature::greedy() {
        json!({ "top_k": 1, "cache_prompt": false })
    } else {
        json!({ "cache_prompt": false })
    };
    turn::Envelope::new(extras, source, params.temperature)
}

/// The argv speculation adds over the cell's: the draft length, and a separate
/// drafter when the model has one. Everything else about speculation is the
/// cell's own `raw` and `envs`. The flag family is asked of the binary: a wrong
/// spelling dies in argument parsing after the model has loaded.
fn speculation_args(
    llama_server: &ResolvedLlamaServer,
    draft_length: u32,
    draft_path: Option<&Path>,
) -> anyhow::Result<Vec<String>> {
    let help = server_help(llama_server)?;
    let length_flag = accepted_flag(&help, DRAFT_LENGTH_FLAGS).with_context(|| {
        format!(
            "{} accepts none of {DRAFT_LENGTH_FLAGS:?}, so it cannot be told how many \
             tokens to draft; vl_spec_decode needs a llama-server build with \
             speculative decoding",
            llama_server.display()
        )
    })?;
    let mut args = vec![length_flag.to_string(), draft_length.to_string()];
    if let Some(draft) = draft_path {
        let model_flag = accepted_flag(&help, DRAFT_MODEL_FLAGS).with_context(|| {
            format!(
                "{} accepts none of {DRAFT_MODEL_FLAGS:?}, so it cannot load the \
                 separate drafter this model names",
                llama_server.display()
            )
        })?;
        args.push(model_flag.to_string());
        args.push(draft.display().to_string());
    }
    Ok(args)
}

/// `llama-server --help`, stdout and stderr together: builds differ on which
/// one they print it to.
fn server_help(llama_server: &Path) -> anyhow::Result<String> {
    let mut command = Command::new(llama_server);
    command.arg("--help");
    runtimes::apply_dylib_search_env(&mut command, llama_server);
    let failed = || format!("failed to run {} --help", llama_server.display());
    let mut process =
        start_process("llama-server", command, Duration::ZERO).with_context(failed)?;
    // A build that dies here would otherwise pass its error on as the help,
    // and the cell would fail saying the build lacks the draft flags.
    process.wait_success(HELP_TIMEOUT).with_context(failed)?;
    let output = process.output();
    Ok(format!(
        "{}{}",
        output.runtime_stdout.unwrap_or_default(),
        output.runtime_stderr.unwrap_or_default()
    ))
}

/// The first of `candidates` the help text mentions, followed by a non-flag
/// character, so `--draft` does not claim a build that only documents `--draft-max`.
fn accepted_flag<'a>(help: &str, candidates: &[&'a str]) -> Option<&'a str> {
    candidates.iter().copied().find(|flag| {
        help.split(flag).skip(1).any(|after| {
            after
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '-')
        })
    })
}

/// How long [`drafter_block_size`] waits for the load's line before the cell
/// runs without it: a caller that has just seen `/health` answer can be ahead
/// of the loader's own output.
const LOAD_LOG_WAIT: Duration = Duration::from_secs(2);

/// The block size (the most tokens it can propose in a round) a block drafter
/// announces as it loads:
///
/// ```text
/// common_speculative_impl_draft_dflash: - block_size=9, mask_token_id=4, n_extract=3
/// ```
fn parse_drafter_block_size(stderr: &str) -> Option<u32> {
    stderr.lines().find_map(|line| {
        line.split_once("- block_size=")?
            .1
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()
    })
}

/// [`parse_drafter_block_size`] over a running server, `None` once
/// [`LOAD_LOG_WAIT`] passes without the banner. Waited for, not read once: a
/// read too early would let a clamped launch through.
fn drafter_block_size(llama_server: &Process) -> Option<u32> {
    llama_server
        .wait_for_output(
            pipette_subprocess::Stream::Stderr,
            LOAD_LOG_WAIT,
            |stderr| parse_drafter_block_size(stderr).is_some(),
        )
        .and_then(|stderr| parse_drafter_block_size(&stderr))
}

/// The cell's server, every field named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
///
/// `speculative_enabled` picks the drafter argv `run` passes as
/// `speculation_argv`. `slots` is the benchmark's: at 1 it renders
/// `--parallel 1` only under `no_cache`.
fn launch(
    flags: &VlSpecDecodeLlamacppCliStockTools,
    program: &ResolvedLlamaServer,
    model: &Path,
    mmproj: &Path,
    slots: u32,
    speculation_argv: &[String],
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let VlSpecDecodeLlamacppCliStockTools {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        ctx_size,
        no_cache,
        envs,
        speculative_enabled: _,
        raw,
    } = flags.clone();
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::VlSpecDecode)
        .mmproj(mmproj)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .slots(Some(slots))
        .extra(speculation_argv.to_vec())
        .envs(envs)
        .client_config(client_config.cloned())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{LlamacppFlashAttention, LlamacppLoadMode};

    use super::*;
    use crate::runtimes::test_llama_server;

    fn server_args(
        flags: &VlSpecDecodeLlamacppCliStockTools,
        slots: u32,
        speculation_argv: &[String],
    ) -> Result<Vec<String>, server::ServerArgsError> {
        launch(
            flags,
            &test_llama_server(),
            Path::new("model.gguf"),
            Path::new("mmproj.gguf"),
            slots,
            speculation_argv,
            None,
        )
        .args()
    }

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn every_knob_set() -> VlSpecDecodeLlamacppCliStockTools {
        VlSpecDecodeLlamacppCliStockTools {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            ctx_size: Some(4096),
            no_cache: Some(true),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
            speculative_enabled: Some(true),
        }
    }

    #[test]
    fn the_server_takes_every_knob_the_slots_and_the_drafter() -> anyhow::Result<()> {
        assert_eq!(
            server_args(
                &every_knob_set(),
                2,
                &["--draft-max".to_owned(), "8".to_owned()]
            )?
            .join(" "),
            "--ctx-checkpoints 0 --ctx-size 4096 --draft-max 8 --flash-attn on --load-mode dio --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 2 --poll 0 --threads 3"
        );
        Ok(())
    }

    /// A regular VL corpus cell with the context `run` would have overlaid
    /// for `batch_size` slots and nothing else pinned.
    fn runtime_flags(batch_size: u32) -> VlSpecDecodeLlamacppCliStockTools {
        VlSpecDecodeLlamacppCliStockTools {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            ctx_size: Some(DEFAULT_CTX_SIZE_PER_SLOT * batch_size),
            no_cache: None,
            envs: Vec::new(),
            raw: Vec::new(),
            speculative_enabled: Some(false),
        }
    }

    /// At the reference batch size the slots render no `--parallel`: the flags
    /// are stored with the row, and a stray `--parallel 1` would split identical
    /// configurations. Above it each sample gets what a serial run gave one.
    #[rstest::rstest]
    #[case::reference(1, None, "8192")]
    #[case::batched(8, Some("8"), "65536")]
    fn the_slot_count_and_context_follow_the_batch_size(
        #[case] batch_size: u32,
        #[case] parallel: Option<&str>,
        #[case] ctx: &str,
    ) -> anyhow::Result<()> {
        let args = server_args(&runtime_flags(batch_size), batch_size, &[])?;
        let rendered = args
            .windows(2)
            .find(|w| w[0] == "--parallel")
            .map(|w| w[1].as_str());
        assert_eq!(rendered, parallel, "{args:?}");
        assert!(
            args.windows(2).any(|w| w == ["--ctx-size", ctx]),
            "{args:?}"
        );
        Ok(())
    }

    /// With speculation on the forwards are read as whole blocks of the draft length.
    #[rstest::rstest]
    #[case(Speculation::On, TimingsSource::RuntimeFixedDraft(8))]
    #[case(Speculation::Off, TimingsSource::Runtime)]
    fn the_envelope_is_greedy_uncached_and_reads_the_runtime(
        #[case] speculation: Speculation,
        #[case] source: TimingsSource,
    ) -> anyhow::Result<()> {
        let envelope = envelope(&turn::Params {
            speculation,
            max_tokens: 2048,
            batch_size: std::num::NonZeroUsize::MIN,
            draft_length: 8,
            temperature: SamplingTemperature::greedy(),
            http_timeout: std::time::Duration::from_secs(600),
            deadline: pipette_ops::vl_spec_decode::Deadline::starting_now(None),
        })?;
        let messages = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(
            envelope.body(messages.clone(), 2048),
            json!({
                "messages": messages,
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 2048,
                "cache_prompt": false,
                "stream": true,
            })
        );
        assert_eq!(envelope.source(), source);
        Ok(())
    }

    /// `top_k: 1` is how llama.cpp is told to take the argmax; leaving it on a
    /// sampled cell would pin the greedy token and make the pinned temperature
    /// a no-op, so a sampled envelope drops it and sends the temperature alone.
    #[test]
    fn a_sampled_envelope_drops_the_greedy_pin() -> anyhow::Result<()> {
        let envelope = envelope(&turn::Params {
            speculation: Speculation::Off,
            max_tokens: 2048,
            batch_size: std::num::NonZeroUsize::MIN,
            draft_length: 8,
            temperature: SamplingTemperature::try_new(0.8)?,
            http_timeout: std::time::Duration::from_secs(600),
            deadline: pipette_ops::vl_spec_decode::Deadline::starting_now(None),
        })?;
        let messages = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(
            envelope.body(messages.clone(), 2048),
            json!({
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 2048,
                "cache_prompt": false,
                "stream": true,
            })
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::at_the_block(8, Some(8), true)]
    #[case::below_the_block(7, Some(9), true)]
    #[case::above_the_block(10, Some(9), false)]
    #[case::no_banner(10, None, true)]
    fn a_draft_length_the_drafter_cannot_propose_is_refused(
        #[case] draft_length: u32,
        #[case] drafter_block: Option<u32>,
        #[case] accepted: bool,
    ) {
        match check_draft_length(draft_length, drafter_block) {
            Ok(()) => assert!(
                accepted,
                "{draft_length} over {drafter_block:?} should refuse"
            ),
            Err(e) => {
                assert!(
                    !accepted,
                    "{draft_length} over {drafter_block:?} refused: {e:#}"
                );
                let message = format!("{e:#}");
                // The operator has to know which two numbers disagree.
                assert!(message.contains(&draft_length.to_string()), "{message}");
                assert!(
                    drafter_block.is_none_or(|b| message.contains(&b.to_string())),
                    "{message}"
                );
            }
        }
    }

    /// The flag families are told apart by what the binary documents. A build
    /// that only has `--draft-max` must not be handed `--draft`.
    #[rstest::rstest]
    #[case(
        "  --spec-draft-n-max N     max drafted tokens\n",
        Some("--spec-draft-n-max")
    )]
    #[case(
        "  --draft-max, --draft N   number of tokens to draft\n",
        Some("--draft-max")
    )]
    #[case("  --draft N                tokens to draft\n", Some("--draft"))]
    #[case("  --threads N\n", None)]
    fn the_draft_length_flag_is_the_one_the_build_documents(
        #[case] help: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(accepted_flag(help, DRAFT_LENGTH_FLAGS), expected);
    }

    /// `--draft-maximum` is not `--draft-max`: the match has to end at a
    /// boundary or a longer flag would claim a build that never had it.
    #[test]
    fn a_longer_flag_does_not_match_a_shorter_candidate() {
        assert_eq!(
            accepted_flag("  --draft-maximum N\n", &["--draft-max"]),
            None
        );
    }

    /// The fork's banner, verbatim from `common/speculative.cpp`.
    const DRAFTER_BANNER: &str =
        "common_speculative_impl_draft_dflash: - block_size=9, mask_token_id=4, n_extract=3\n";

    #[rstest]
    #[case::the_banner(DRAFTER_BANNER, Some(9))]
    #[case::the_metadata_dump_is_not_the_banner(
        "llama_model_loader: - kv  23:  dflash.block_size u32 = 9\n",
        None
    )]
    #[case::a_drafter_with_no_banner("srv    load_model: the draft model is loaded\n", None)]
    fn the_drafters_block_is_read_off_its_banner(#[case] stderr: &str, #[case] want: Option<u32>) {
        assert_eq!(parse_drafter_block_size(stderr), want);
    }

    /// "No block" is the answer that lets a clamped launch through, so the
    /// banner is waited for rather than read once.
    #[cfg(unix)]
    #[test]
    fn the_drafters_banner_is_waited_for() -> anyhow::Result<()> {
        let mut command = Command::new("sh");
        command.arg("-c").arg(format!(
            "sleep 0.3; printf '{DRAFTER_BANNER}' >&2; exec sleep 30"
        ));
        let llama_server = start_process("sh", command, Duration::ZERO)?;
        assert_eq!(drafter_block_size(&llama_server), Some(9));
        Ok(())
    }

    /// The help is read from both streams; a build that dies fails with its
    /// log rather than passing that log on as the help.
    #[cfg(unix)]
    #[rstest]
    #[case::from_both_streams(
        "echo '  --draft-max N'; echo '  -md FNAME' >&2",
        Ok("  --draft-max N\n  -md FNAME\n")
    )]
    #[case::from_a_build_that_dies(
        "echo 'Library not loaded: libllama.dylib' >&2; exit 6",
        Err("--help: llama-server exited with exit status: 6\nstderr tail:\nLibrary not loaded")
    )]
    fn server_help_reads_the_help_or_fails_with_the_log(
        #[case] script: &str,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir()?;
        let program = dir.path().join("llama-server");
        std::fs::write(&program, format!("#!/bin/sh\n{script}\n"))?;
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755))?;
        match (server_help(&program), expected) {
            (Ok(help), Ok(want)) => assert_eq!(help, want),
            (Err(err), Err(want)) => assert!(format!("{err:#}").contains(want), "{err:#}"),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    /// A bound vision model of `kind` whose files exist under `dir`, with a
    /// separate drafter when `draft` is set.
    fn vision_model(dir: &std::path::Path, kind: &str, draft: bool) -> anyhow::Result<Model> {
        let file = |name: &str| -> anyhow::Result<std::path::PathBuf> {
            let path = dir.join(name);
            std::fs::write(&path, b"gguf")?;
            Ok(path)
        };
        let mut spec = json!({
            "type": kind,
            "source": "absolute_files",
            "model": file("model.gguf")?,
            "mmproj": file("mmproj.gguf")?,
        });
        if draft {
            spec["draft"] = json!(file("draft.gguf")?);
        }
        Ok(serde_json::from_value(spec)?)
    }

    /// Only speculation on loads the drafter; off, a speculative model loads
    /// its backbone and projector as a plain one does, and on is refused for a
    /// plain model, which has no drafter to load.
    #[rstest]
    #[case::speculative_on("gguf_vision_speculative", true, Some(Speculation::On))]
    #[case::speculative_off("gguf_vision_speculative", false, Some(Speculation::Off))]
    #[case::plain_off("gguf_vision", false, Some(Speculation::Off))]
    #[case::plain_on_is_refused("gguf_vision", true, None)]
    fn only_speculation_on_loads_the_drafter(
        #[case] kind: &str,
        #[case] enabled: bool,
        #[case] expected: Option<Speculation>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = vision_model(tmp.path(), kind, kind == "gguf_vision_speculative")?;
        match (LoadedModel::resolve(&model, enabled), expected) {
            (Ok(loaded), Some(speculation)) => {
                assert_eq!(loaded.speculation(), speculation);
                if let LoadedModel::SpeculativeOn(model) = &loaded {
                    assert_eq!(
                        model.draft.as_deref(),
                        Some(tmp.path().join("draft.gguf").as_path())
                    );
                }
            }
            (Err(_), None) => {}
            (Ok(_), None) => anyhow::bail!("{kind} with speculation {enabled} should be refused"),
            (Err(err), Some(_)) => return Err(err),
        }
        Ok(())
    }
}
