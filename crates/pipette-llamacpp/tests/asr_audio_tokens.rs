//! The exact audio-token count, end to end. Both ASR cells run against the
//! fake `llama-server` (20 prompt tokens per clip, 5 for the template, no audio
//! markers, as LFM2-Audio's recipe adds none) beside a fake `llama-mtmd-cli`
//! that reports a count of its own and logs every run
//! ([`common::install_fake_mtmd_cli_audio`]).
//!
//! The fakes read their knobs from the environment, which is process-wide, so
//! the scenarios take one lock and run one at a time.
#![cfg(unix)]

mod common;

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use common::{
    audio_projector, fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model,
    ignore_reps, install_fake_mtmd_cli_audio, no_corpus, no_readiness_gate, with_distinct_knobs,
};
use serde_json::Value;

use pipette_plan_types::{
    AsrDecodeTokens, AsrMaxMemoryBenchmark, AsrThroughputBenchmark, AudioChannels,
    AudioDurationMillis, AudioSampleRate, BenchmarkDefinition, BenchmarkId, BenchmarkResultData,
    EnvForward, NonNegativeCount, RuntimeFlags,
};

static ENV: Mutex<()> = Mutex::new(());

const FAKE_ENV: [&str; 8] = [
    "FAKE_LLAMA_AUDIO_TOKENS",
    "FAKE_LLAMA_CHAT_REPLY",
    "FAKE_LLAMA_COUNT_TEXT",
    "FAKE_LLAMA_ENFORCE_CTX",
    "FAKE_LLAMA_LATER_AUDIO_TOKENS",
    "FAKE_LLAMA_NEWLINE",
    "FAKE_LLAMA_PID_FILE",
    "FAKE_MTMD_AUDIO_TOKENS",
];

/// What the fake helper reports for the clip: the fake server's 20 audio
/// tokens, so the server confirms it.
const COUNTED: u32 = 20;

/// LFM2-Audio's ask: the system turn "Perform ASR.".
const ASK_CHARS: u32 = 12;

const DECODE_TOKENS: u32 = 4;

/// One runtime directory, which holds the helper and its cache, and the model
/// the cells load.
struct Fixture {
    root: tempfile::TempDir,
    install: PathBuf,
    helper_log: PathBuf,
    model: PathBuf,
    mmproj: PathBuf,
}

impl Fixture {
    fn new(helper_tokens: u32, helper_status: i32) -> anyhow::Result<Self> {
        Self::with_projector(helper_tokens, helper_status, "lfm2a")
    }

    fn with_projector(helper_tokens: u32, helper_status: i32, kind: &str) -> anyhow::Result<Self> {
        let root = tempfile::tempdir()?;
        let install = fake_runtime_install_dir(root.path())?;
        let helper_log = root.path().join("helper-runs.log");
        install_fake_mtmd_cli_audio(&install, &helper_log, helper_tokens, helper_status)?;
        let model = root.path().join("fake-model.gguf");
        std::fs::write(&model, b"fake-model-bytes")?;
        let mmproj = root.path().join("fake-mmproj.gguf");
        std::fs::write(&mmproj, audio_projector(Some(kind)))?;
        Ok(Self {
            root,
            install,
            helper_log,
            model,
            mmproj,
        })
    }

    fn cache(&self) -> PathBuf {
        self.install.join("audio_calibration.json")
    }

    fn cache_rows(&self) -> anyhow::Result<Vec<u64>> {
        if !self.cache().exists() {
            return Ok(Vec::new());
        }
        let cache: Value = serde_json::from_str(&std::fs::read_to_string(self.cache())?)?;
        Ok(cache
            .get("entries")
            .and_then(Value::as_object)
            .context("the cache keeps its rows under `entries`")?
            .values()
            .filter_map(Value::as_u64)
            .collect())
    }

    fn helper_runs(&self) -> anyhow::Result<Vec<HelperRun>> {
        if !self.helper_log.exists() {
            return Ok(Vec::new());
        }
        std::fs::read_to_string(&self.helper_log)?
            .lines()
            .map(HelperRun::parse)
            .collect()
    }

    /// Runs `benchmark` with the distinct knobs, a forward the helper must
    /// get, and `edit` on the flags.
    fn run(
        &self,
        benchmark: BenchmarkDefinition,
        edit: impl FnOnce(&mut pipette_plan_types::RuntimeFlagRef),
    ) -> anyhow::Result<Ran> {
        self.run_in(benchmark, &[], edit)
    }

    /// [`Self::run`] with the fakes' environment `env` (`FAKE_LLAMA_*`,
    /// `FAKE_MTMD_*`); every one it does not set is unset.
    fn run_in(
        &self,
        benchmark: BenchmarkDefinition,
        env: &[(&str, &str)],
        edit: impl FnOnce(&mut pipette_plan_types::RuntimeFlagRef),
    ) -> anyhow::Result<Ran> {
        self.run_gated(benchmark, env, edit, &no_readiness_gate)
    }

    /// [`Self::run_in`] behind `gate`, the readiness gate each rep waits on.
    fn run_gated(
        &self,
        benchmark: BenchmarkDefinition,
        env: &[(&str, &str)],
        edit: impl FnOnce(&mut pipette_plan_types::RuntimeFlagRef),
        gate: &dyn Fn() -> anyhow::Result<()>,
    ) -> anyhow::Result<Ran> {
        let _guard = ENV
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let bodies = self.root.path().join("chat-bodies.jsonl");
        let _ = std::fs::remove_file(&bodies);
        // SAFETY: the lock above serializes every reader and writer of this var.
        unsafe { std::env::set_var("FAKE_LLAMA_CHAT_BODY_FILE", &bodies) };
        for name in FAKE_ENV {
            match env.iter().find(|(set, _)| *set == name) {
                // SAFETY: as above.
                Some((_, value)) => unsafe { std::env::set_var(name, value) },
                // SAFETY: as above.
                None => unsafe { std::env::remove_var(name) },
            }
        }
        let req = with_distinct_knobs(fake_run_request(
            fake_runtime(&self.install)?,
            fake_vision_model(&self.model, &self.mmproj)?,
            benchmark,
            None,
        ))?;
        let mut flags = req.runtime_flags_ref()?;
        flags.envs = vec![EnvForward::try_new("PIPETTE_TEST_FORWARD=yes".to_owned())?];
        flags.raw = ["--main-gpu", "0", "-ub", "512"]
            .map(str::to_owned)
            .to_vec();
        edit(&mut flags);
        let req = pipette_plan_types::RunRequest {
            runtime_flags: Some(RuntimeFlags::try_from(flags)?),
            ..req
        };
        let outcome = pipette_llamacpp::run(
            &req,
            &no_corpus(self.root.path()),
            gate,
            ignore_reps(),
            None,
        )?;
        let bodies = if bodies.exists() {
            std::fs::read_to_string(&bodies)?
                .lines()
                .map(|line| serde_json::from_str(line).context("a chat body that is not JSON"))
                .collect::<anyhow::Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(Ran {
            bodies,
            extras: outcome.client_extras.clone().into_iter().collect(),
            ctx_size: ctx_size(outcome.runtime_flags.as_ref())?,
            data: outcome.result_data,
        })
    }
}

struct Ran {
    bodies: Vec<Value>,
    extras: serde_json::Map<String, Value>,
    ctx_size: Option<u32>,
    data: BenchmarkResultData,
}

impl Ran {
    fn extra(&self, key: &str) -> Option<&str> {
        self.extras.get(key).and_then(Value::as_str)
    }

    /// Whether the row flags its clip as past the model's documented limit.
    fn beyond_limit(&self) -> Option<bool> {
        self.extras
            .get("asr_beyond_model_audio_limit")
            .and_then(Value::as_bool)
    }
}

fn ctx_size(flags: Option<&RuntimeFlags>) -> anyhow::Result<Option<u32>> {
    if let Some(RuntimeFlags::AsrThroughputLlamacppCliStockToolsGgufVision(flags)) = flags {
        return Ok(flags.ctx_size);
    }
    if let Some(RuntimeFlags::AsrMaxMemoryLlamacppCliStockToolsGgufVision(flags)) = flags {
        return Ok(flags.ctx_size);
    }
    anyhow::bail!("not an ASR cell's flags: {flags:?}")
}

/// What one helper run logged.
struct HelperRun {
    clip: PathBuf,
    wav: bool,
    forward: String,
    args: Vec<String>,
}

impl HelperRun {
    fn parse(line: &str) -> anyhow::Result<Self> {
        let mut fields = line.split('\t');
        let mut field = |name: &str| {
            fields
                .next()
                .and_then(|f| f.strip_prefix(name))
                .map(str::to_owned)
                .with_context(|| format!("no {name} in {line:?}"))
        };
        let clip = PathBuf::from(field("clip=")?);
        let wav = field("wav=")? == "yes";
        let forward = field("forward=")?;
        let first = field("args=")?;
        let args = std::iter::once(first)
            .chain(fields.map(str::to_owned))
            .filter(|a| !a.is_empty())
            .collect();
        Ok(Self {
            clip,
            wav,
            forward,
            args,
        })
    }

    fn passes(&self, flag: &str, value: &str) -> bool {
        self.args.windows(2).any(|pair| pair == [flag, value])
    }
}

fn throughput() -> anyhow::Result<BenchmarkDefinition> {
    throughput_with_text(0)
}

fn throughput_with_text(text_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
    throughput_of(2_500, text_tokens)
}

fn throughput_of(duration_ms: i32, text_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::AsrThroughput(AsrThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("asr-audio-tokens-test".to_owned())?,
        parameter_audio_duration_ms: AudioDurationMillis::try_new(duration_ms)?,
        parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
        parameter_audio_channels: AudioChannels::try_new(1)?,
        parameter_text_tokens: NonNegativeCount::try_new(i32::try_from(text_tokens)?)?,
        parameter_decode_tokens: AsrDecodeTokens::try_new(i32::try_from(DECODE_TOKENS)?)?,
    }))
}

fn max_memory() -> anyhow::Result<BenchmarkDefinition> {
    max_memory_with_text(0)
}

fn max_memory_with_text(text_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
    max_memory_of(2_500, text_tokens)
}

fn max_memory_of(duration_ms: i32, text_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
    Ok(BenchmarkDefinition::AsrMaxMemory(AsrMaxMemoryBenchmark {
        benchmark_id: BenchmarkId::try_new("asr-audio-tokens-memory-test".to_owned())?,
        parameter_audio_duration_ms: AudioDurationMillis::try_new(duration_ms)?,
        parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
        parameter_audio_channels: AudioChannels::try_new(1)?,
        parameter_text_tokens: NonNegativeCount::try_new(i32::try_from(text_tokens)?)?,
        parameter_decode_tokens: AsrDecodeTokens::try_new(i32::try_from(DECODE_TOKENS)?)?,
    }))
}

const TEXT_TOKENS: u32 = 32;

/// The words of the ask ("Perform ASR."), which the fake counts as tokens
/// when it counts text.
const ASK_WORDS: u32 = 2;

/// The fake counting every word of the turns as a token, as `/tokenize` does.
const COUNT_TEXT: [(&str, &str); 1] = [("FAKE_LLAMA_COUNT_TEXT", "1")];

fn clips(body: &Value) -> usize {
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "input_audio")
        .count()
}

/// The helper counts the cell's own clip, placed like the server and bounded,
/// with the cell's forwards; the count sizes an exact context, the server
/// confirms it (the prompt with the clip less the one without it is the
/// count), and the row says both.
#[test]
fn a_counted_clip_sizes_an_exact_context_and_reports_its_count() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let ran = fixture.run(throughput()?, |_| {})?;

    let BenchmarkResultData::AsrThroughput {
        asr_prompt_tokens,
        asr_audio_tokens,
        ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    assert_eq!(asr_prompt_tokens, 25);
    assert_eq!(asr_audio_tokens, COUNTED);
    assert_eq!(ran.extra("asr_audio_tokens_source"), Some("calibrated"));
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    assert_eq!(ran.ctx_size, Some(COUNTED + ASK_CHARS + DECODE_TOKENS + 64));
    // The warm-up and five reps, then the text-only probe that confirms it.
    assert_eq!(
        ran.bodies.iter().map(clips).collect::<Vec<_>>(),
        [1, 1, 1, 1, 1, 1, 0]
    );

    let runs = fixture.helper_runs()?;
    let [run] = runs.as_slice() else {
        anyhow::bail!("expected one helper run, got {}", runs.len());
    };
    assert!(run.wav, "the clip was not a WAV file while the helper ran");
    assert!(!run.clip.exists(), "the clip outlived the helper");
    assert_eq!(run.clip.extension().and_then(|e| e.to_str()), Some("wav"));
    assert!(
        run.clip
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("asrcal-")),
        "{}",
        run.clip.display()
    );
    assert_eq!(run.forward, "yes");
    assert!(run.passes("--mmproj", &fixture.mmproj.display().to_string()));
    assert!(run.passes("--n-gpu-layers", "7"), "{:?}", run.args);
    assert!(run.passes("--main-gpu", "0"), "{:?}", run.args);
    // The bound (750) and the margin are under the floor.
    assert!(run.passes("--ctx-size", "4096"), "{:?}", run.args);
    for dropped in ["-ub", "--threads", "--flash-attn", "--reasoning"] {
        assert!(
            !run.args.iter().any(|a| a == dropped),
            "{dropped}: {:?}",
            run.args
        );
    }
    assert!(
        run.args.ends_with(&[
            "--jinja".to_owned(),
            "--verbose".to_owned(),
            "--audio".to_owned(),
            run.clip.display().to_string(),
        ]),
        "{:?}",
        run.args
    );
    assert_eq!(fixture.cache_rows()?, [u64::from(COUNTED)]);
    Ok(())
}

#[test]
fn a_second_cell_reuses_the_count() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    fixture.run(throughput()?, |_| {})?;
    let again = fixture.run(throughput()?, |_| {})?;
    assert_eq!(fixture.helper_runs()?.len(), 1);
    assert_eq!(again.extra("asr_audio_tokens_source"), Some("calibrated"));
    assert_eq!(again.extra("asr_context_sizing"), Some("exact"));
    Ok(())
}

/// A row no clip of the length can have is counted again, and replaced.
#[test]
fn an_impossible_cached_row_is_counted_again() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    fixture.run(throughput()?, |_| {})?;
    let stale =
        std::fs::read_to_string(fixture.cache())?.replace(&format!(": {COUNTED}"), ": 999999");
    std::fs::write(fixture.cache(), stale)?;
    assert_eq!(fixture.cache_rows()?, [999_999]);

    let again = fixture.run(throughput()?, |_| {})?;
    assert_eq!(fixture.helper_runs()?.len(), 2);
    assert_eq!(fixture.cache_rows()?, [u64::from(COUNTED)]);
    assert_eq!(
        again.ctx_size,
        Some(COUNTED + ASK_CHARS + DECODE_TOKENS + 64)
    );
    Ok(())
}

/// A helper that fails leaves no row, and the cell falls back: the context
/// from the bound, the count measured by the text-only probe.
#[test]
fn a_failing_helper_falls_back_to_measuring() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 1)?;
    let ran = fixture.run(throughput()?, |_| {})?;
    let BenchmarkResultData::AsrThroughput {
        asr_audio_tokens, ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    // With the clip (25) less the same turns without it (5).
    assert_eq!(asr_audio_tokens, 20);
    assert_eq!(ran.extra("asr_audio_tokens_source"), Some("measured"));
    assert_eq!(ran.extra("asr_context_sizing"), Some("upper_bound"));
    assert_eq!(ran.ctx_size, Some(750 + ASK_CHARS + DECODE_TOKENS + 64));
    assert_eq!(ran.bodies.iter().map(clips).next_back(), Some(0));
    assert_eq!(fixture.helper_runs()?.len(), 1);
    assert!(fixture.cache_rows()?.is_empty());
    Ok(())
}

/// A count the server does not confirm sized a context that was not exact,
/// whether an undercount the margin hides (18 of the server's 20, the context
/// still holds the prompt) or one past the whole prompt. Either cell is refused
/// rather than labelled `exact`, and the row dropped so the next run counts
/// again. With a count of 20 the same fixture is exact
/// (`a_counted_clip_sizes_an_exact_context_and_reports_its_count`).
#[rstest::rstest]
fn an_unconfirmed_count_fails_the_cell_and_is_dropped(
    #[values(18, 19, 21, 30)] helper_tokens: u32,
    #[values(throughput(), max_memory())] benchmark: anyhow::Result<BenchmarkDefinition>,
) -> anyhow::Result<()> {
    let fixture = Fixture::new(helper_tokens, 0)?;
    let err = fixture
        .run(benchmark?, |_| {})
        .err()
        .context("an unconfirmed count was published")?;
    assert!(
        format!("{err:#}").contains(&format!(
            "prefilled 20 more tokens with the clip than without it, where the calibrated count \
             of {helper_tokens} audio tokens"
        )),
        "{err:#}"
    );
    assert!(fixture.cache_rows()?.is_empty());
    Ok(())
}

/// Behind a pinned context, which the count did not size, an unconfirmed
/// count is measured instead and the row says `pinned`.
#[test]
fn a_pinned_cell_measures_an_unconfirmed_count() -> anyhow::Result<()> {
    let fixture = Fixture::new(18, 0)?;
    let ran = fixture.run(throughput()?, |flags| flags.ctx_size = Some(4096))?;
    let BenchmarkResultData::AsrThroughput {
        asr_audio_tokens, ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    assert_eq!(asr_audio_tokens, 20);
    assert_eq!(ran.extra("asr_audio_tokens_source"), Some("measured"));
    assert_eq!(ran.extra("asr_context_sizing"), Some("pinned"));
    assert_eq!(ran.ctx_size, Some(4096));
    Ok(())
}

/// A stale cached undercount whose context cannot hold the prompt is refused by
/// the server (its context-overflow 400) before any check after the run. The
/// cell fails and drops the row, naming it, so the next cell counts the clip
/// again and runs exact, rather than every cell on the key failing for good.
/// The memory cell fails on its warm-up.
#[rstest::rstest]
fn a_stale_undercount_that_overflows_is_dropped_and_counted_again(
    #[values(throughput(), max_memory())] benchmark: anyhow::Result<BenchmarkDefinition>,
) -> anyhow::Result<()> {
    let benchmark = benchmark?;
    let fixture = Fixture::new(COUNTED, 0)?;
    fixture.run(benchmark.clone(), |_| {})?;
    assert_eq!(fixture.cache_rows()?, [u64::from(COUNTED)]);
    // The server now prefills 200 audio tokens a clip, and so would the
    // helper; the cached 20 sizes a context of 20 + 12 + 4 + 64 for a
    // 205-token prompt.
    let grown = [
        ("FAKE_LLAMA_AUDIO_TOKENS", "200"),
        ("FAKE_MTMD_AUDIO_TOKENS", "200"),
        ("FAKE_LLAMA_ENFORCE_CTX", "1"),
    ];
    let err = fixture
        .run_in(benchmark.clone(), &grown, |_| {})
        .err()
        .context("a context too short for the prompt ran")?;
    let message = format!("{err:#}");
    assert!(
        message.contains("exceeds the available context size"),
        "{message}"
    );
    assert!(
        message.contains(&format!(
            "the context was sized from the count of {COUNTED} audio tokens in {}; its row is \
             dropped",
            fixture.cache().display()
        )),
        "{message}"
    );
    assert!(fixture.cache_rows()?.is_empty());
    assert_eq!(fixture.helper_runs()?.len(), 1);

    let again = fixture.run_in(benchmark, &grown, |_| {})?;
    assert_eq!(fixture.helper_runs()?.len(), 2);
    assert_eq!(fixture.cache_rows()?, [200]);
    assert_eq!(again.extra("asr_context_sizing"), Some("exact"));
    Ok(())
}

/// Gemma 4 runs a clip past the 30 s its card documents, in both cells, and
/// each row says the clip is beyond the model's documented audio limit; at or
/// under 30 s, and for a family that documents no limit, the row says it is
/// not. The fake server prefills the count and Gemma's two audio markers, so
/// every row is confirmed exact.
#[rstest::rstest]
#[case::gemma_the_long_rung("gemma4a", 119_900, true)]
#[case::gemma_unified_the_long_rung("gemma4ua", 119_900, true)]
#[case::gemma_just_over("gemma4a", 30_001, true)]
#[case::gemma_ladder_top("gemma4a", 29_900, false)]
#[case::lfm2_the_long_rung("lfm2a", 119_900, false)]
fn a_clip_past_the_documented_limit_runs_flagged(
    #[case] projector: &str,
    #[case] duration_ms: i32,
    #[case] beyond: bool,
) -> anyhow::Result<()> {
    let served = (COUNTED + markers(projector)).to_string();
    let env = [("FAKE_LLAMA_AUDIO_TOKENS", served.as_str())];
    let fixture = Fixture::with_projector(COUNTED, 0, projector)?;
    for benchmark in [
        throughput_of(duration_ms, 0)?,
        max_memory_of(duration_ms, 0)?,
    ] {
        let ran = fixture.run_in(benchmark, &env, |_| {})?;
        assert_eq!(ran.beyond_limit(), Some(beyond), "{:?}", ran.extras);
        assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    }
    Ok(())
}

/// The audio markers the fake server must prefill beside the count for
/// `projector`'s recipe to confirm it.
fn markers(projector: &str) -> u32 {
    match projector {
        "lfm2a" | "lfm3a" => 0,
        "voxtral" => 1,
        _ => 2,
    }
}

/// `parameter_text_tokens` is exactly the prompt the text adds, whatever the
/// tokenizer does with the newline that joins the text to the recipe's
/// instruction: a token of its own (Gemma 4's), merged into a neighbour, or
/// plain whitespace; for recipes with an instruction (Gemma 4, MiniCPM-o) and
/// one that sends the clip alone (LFM2). The memory cell, which confirms the
/// same, runs.
#[rstest::rstest]
fn the_text_is_exactly_its_tokens(
    #[values("gemma4a", "minicpmoa", "lfm2a")] projector: &str,
    #[values("word", "own", "merge")] newline: &str,
) -> anyhow::Result<()> {
    let served = (COUNTED + markers(projector)).to_string();
    let env = [
        ("FAKE_LLAMA_AUDIO_TOKENS", served.as_str()),
        ("FAKE_LLAMA_COUNT_TEXT", "1"),
        ("FAKE_LLAMA_NEWLINE", newline),
    ];
    let fixture = Fixture::with_projector(COUNTED, 0, projector)?;
    let prompt = |text_tokens| -> anyhow::Result<u32> {
        let ran = fixture.run_in(throughput_of(2_500, text_tokens)?, &env, |_| {})?;
        let BenchmarkResultData::AsrThroughput {
            asr_prompt_tokens, ..
        } = ran.data
        else {
            anyhow::bail!("expected AsrThroughput result data");
        };
        Ok(asr_prompt_tokens)
    };
    assert_eq!(prompt(TEXT_TOKENS)? - prompt(0)?, TEXT_TOKENS);
    let ran = fixture.run_in(max_memory_of(2_500, TEXT_TOKENS)?, &env, |_| {})?;
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    Ok(())
}

/// A server whose prompt counts the cell's text differently from its own
/// `/tokenize` (here the joining newline, in the prompt alone) fails both
/// cells rather than publish a row whose text is not the entry's.
#[rstest::rstest]
fn a_text_the_server_does_not_confirm_fails_the_cell(
    #[values(true, false)] throughput: bool,
) -> anyhow::Result<()> {
    let env = [
        ("FAKE_LLAMA_AUDIO_TOKENS", "22"),
        ("FAKE_LLAMA_COUNT_TEXT", "1"),
        ("FAKE_LLAMA_NEWLINE", "own_in_chat"),
    ];
    let fixture = Fixture::with_projector(COUNTED, 0, "gemma4a")?;
    let benchmark = if throughput {
        throughput_of(2_500, TEXT_TOKENS)?
    } else {
        max_memory_of(2_500, TEXT_TOKENS)?
    };
    let Err(err) = fixture.run_in(benchmark, &env, |_| {}) else {
        anyhow::bail!("a text the server does not confirm ran");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains(&format!("where the text is {TEXT_TOKENS} tokens")),
        "{message}"
    );
    // The refusal is the headline, the stopped server's tail the detail.
    assert!(
        err.to_string().contains("where the text is"),
        "headline: {err}"
    );
    assert!(message.contains("llama-server"), "{message}");
    Ok(())
}

/// The readiness gate timing out mid-cell, as the worker's gate does on a busy
/// host.
#[derive(Debug)]
struct GateTimedOut;

impl std::fmt::Display for GateTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("readiness wait timed out after 420s: last seen cpu=busy")
    }
}

impl std::error::Error for GateTimedOut {}

/// A readiness wait that times out mid-cell is the failure's headline, ahead
/// of the stopped server's tail, and still a cause a caller can find by type.
#[test]
fn a_readiness_timeout_heads_the_error() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let gate = || -> anyhow::Result<()> { Err(GateTimedOut.into()) };
    let Err(err) = fixture.run_gated(throughput()?, &[], |_| {}, &gate) else {
        anyhow::bail!("a cell whose readiness gate timed out ran");
    };
    assert!(
        err.to_string()
            .starts_with("readiness wait timed out after 420s"),
        "headline: {err}"
    );
    let message = format!("{err:#}");
    assert!(message.contains("llama-server"), "{message}");
    assert!(
        err.chain()
            .any(|cause| cause.downcast_ref::<GateTimedOut>().is_some()),
        "{message}"
    );
    Ok(())
}

/// A count of 20 under a recipe with a template of its own (LFM3) runs exact:
/// the server reports serving the recipe's template, and confirms the count.
#[test]
fn a_recipe_with_its_own_template_runs_exact() -> anyhow::Result<()> {
    let fixture = Fixture::with_projector(COUNTED, 0, "lfm3a")?;
    let ran = fixture.run(throughput()?, |_| {})?;
    assert_eq!(ran.extra("asr_audio_tokens_source"), Some("calibrated"));
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    Ok(())
}

/// The memory cell sizes its context for `parameter_decode_tokens` and decodes
/// exactly those, end-of-sequence ignored, as the throughput cell does, so the
/// peak holds the output's KV. The warm-up server confirms the count with a
/// text-only request after the clip's; the measured server sends the clip's
/// request alone, so its peak holds nothing else.
#[test]
fn the_memory_cell_decodes_its_tokens() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let ran = fixture.run(max_memory()?, |_| {})?;
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    assert_eq!(ran.ctx_size, Some(COUNTED + ASK_CHARS + DECODE_TOKENS + 64));
    let sent: Vec<(usize, Option<u64>, Option<bool>)> = ran
        .bodies
        .iter()
        .map(|body| {
            (
                clips(body),
                body["max_tokens"].as_u64(),
                body["ignore_eos"].as_bool(),
            )
        })
        .collect();
    let decode = Some(u64::from(DECODE_TOKENS));
    assert_eq!(
        sent,
        [
            (1, decode, Some(true)),
            (0, Some(1), Some(true)),
            (1, decode, Some(true))
        ]
    );
    Ok(())
}

/// A memory reply that did not decode the cell's N tokens fails the cell: one
/// that stopped on an end-of-sequence token before the limit, and one that
/// reached the limit with a completion count other than N. Either peak would
/// be labelled with an output length that never ran.
#[rstest::rstest]
#[case::early_stop("early_stop", "did not stop at the requested generation limit")]
#[case::one_token_short("short", "engine returned completion_tokens 3, expected 4")]
fn the_memory_cell_refuses_a_reply_short_of_its_tokens(
    #[case] reply: &str,
    #[case] expected: &str,
) -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let Err(err) = fixture.run_in(max_memory()?, &[("FAKE_LLAMA_CHAT_REPLY", reply)], |_| {})
    else {
        anyhow::bail!("a memory cell whose reply was short of its tokens ran");
    };
    let message = format!("{err:#}");
    assert!(message.contains(expected), "{message}");
    Ok(())
}

/// With text beside the clip, the count still sizes an exact context, now
/// with room for the text, and the server still confirms it: the text-only
/// request carries the same text, so the difference is the clip alone.
#[test]
fn a_counted_clip_with_text_sizes_an_exact_context_for_both() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let ran = fixture.run_in(throughput_with_text(TEXT_TOKENS)?, &COUNT_TEXT, |_| {})?;
    let BenchmarkResultData::AsrThroughput {
        asr_prompt_tokens,
        asr_audio_tokens,
        ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    assert_eq!(asr_prompt_tokens, 5 + COUNTED + ASK_WORDS + TEXT_TOKENS);
    assert_eq!(asr_audio_tokens, COUNTED);
    assert_eq!(ran.extra("asr_audio_tokens_source"), Some("calibrated"));
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    assert_eq!(
        ran.ctx_size,
        Some(COUNTED + TEXT_TOKENS + ASK_CHARS + DECODE_TOKENS + 64)
    );
    Ok(())
}

/// The memory cell holds the text in its context too, and confirms the count
/// against the same text.
#[test]
fn the_memory_cell_sizes_for_its_text() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let ran = fixture.run_in(max_memory_with_text(TEXT_TOKENS)?, &COUNT_TEXT, |_| {})?;
    assert_eq!(ran.extra("asr_context_sizing"), Some("exact"));
    assert_eq!(
        ran.ctx_size,
        Some(COUNTED + TEXT_TOKENS + ASK_CHARS + DECODE_TOKENS + 64)
    );
    let texts: Vec<usize> = ran
        .bodies
        .iter()
        .map(|body| {
            body["messages"][1]["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|part| part["text"].as_str())
                .map(|text| text.split_whitespace().count())
                .sum()
        })
        .collect();
    // The clip's request, the text-only one that confirms the count, the one
    // without the text that confirms the text, the measured clip's: the same
    // text on each but the third.
    let text = usize::try_from(TEXT_TOKENS)?;
    assert_eq!(texts, [text, text, 0, text]);
    Ok(())
}

/// How a memory cell's context was sized: from a count the warm-up server
/// confirms, from the upper bound when the helper could not count the clip, or
/// pinned by the cell, which never calibrates.
#[derive(Debug, Clone, Copy)]
enum Sizing {
    Exact,
    Uncalibrated,
    Pinned,
}

impl Sizing {
    /// Gemma 4's recipe (an instruction beside the text, two audio markers),
    /// with a helper that counts the server's 20 audio tokens or fails.
    fn fixture(self) -> anyhow::Result<Fixture> {
        let helper_status = i32::from(matches!(self, Self::Uncalibrated));
        Fixture::with_projector(COUNTED, helper_status, "gemma4a")
    }

    fn row(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Uncalibrated => "upper_bound",
            Self::Pinned => "pinned",
        }
    }

    fn run(self, fixture: &Fixture, text_tokens: u32, env: &[(&str, &str)]) -> anyhow::Result<Ran> {
        fixture.run_in(max_memory_of(2_500, text_tokens)?, env, |flags| {
            if matches!(self, Self::Pinned) {
                flags.ctx_size = Some(4096);
            }
        })
    }
}

/// The audio count and Gemma's two markers, as the fake server prefills them.
const GEMMA_SERVED: &str = "22";

/// A memory cell checks its text whether or not a count sized its context: an
/// uncalibrated (the helper failed) or pinned one whose server prefills the
/// text a token longer than `/tokenize` built it is refused, as an exact one
/// is. With the server agreeing (the negative control) the same cell runs: the
/// warm-up server sends the clip, the text-only request and the one without the
/// text, then the measured server the clip.
#[rstest::rstest]
fn an_unconfirmed_memory_cell_still_checks_its_text(
    #[values(Sizing::Uncalibrated, Sizing::Pinned)] sized: Sizing,
) -> anyhow::Result<()> {
    let fixture = sized.fixture()?;
    let env = |newline| {
        [
            ("FAKE_LLAMA_AUDIO_TOKENS", GEMMA_SERVED),
            ("FAKE_LLAMA_COUNT_TEXT", "1"),
            ("FAKE_LLAMA_NEWLINE", newline),
        ]
    };
    let Err(err) = sized.run(&fixture, TEXT_TOKENS, &env("own_in_chat")) else {
        anyhow::bail!("{sized:?}: a text the server does not confirm ran");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains(&format!("where the text is {TEXT_TOKENS} tokens")),
        "{sized:?}: {message}"
    );

    let ran = sized.run(&fixture, TEXT_TOKENS, &env("own"))?;
    assert_eq!(ran.extra("asr_context_sizing"), Some(sized.row()));
    assert_eq!(
        ran.bodies.iter().map(clips).collect::<Vec<_>>(),
        [1, 0, 0, 1]
    );
    Ok(())
}

/// The measured server must prefill the prompt the warm-up server checked, on
/// every path: one that prefills a token more of audio fails the cell,
/// whether the context was exact, from the bound or pinned, with or without
/// text. The same servers agreeing (the negative control) run.
#[rstest::rstest]
fn the_measured_server_repeats_the_checked_prompt_on_every_path(
    #[values(Sizing::Exact, Sizing::Uncalibrated, Sizing::Pinned)] sized: Sizing,
    #[values(0, TEXT_TOKENS)] text_tokens: u32,
) -> anyhow::Result<()> {
    let fixture = sized.fixture()?;
    let run = |later: &str, pids: &str| {
        let pids = fixture.root.path().join(pids).display().to_string();
        let env = [
            ("FAKE_LLAMA_AUDIO_TOKENS", GEMMA_SERVED),
            ("FAKE_LLAMA_COUNT_TEXT", "1"),
            ("FAKE_LLAMA_LATER_AUDIO_TOKENS", later),
            ("FAKE_LLAMA_PID_FILE", pids.as_str()),
        ];
        sized.run(&fixture, text_tokens, &env)
    };
    let Err(err) = run("23", "pids-differ") else {
        anyhow::bail!("{sized:?}: a measured server with another prompt ran");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("where the warm-up server, which checked the prompt, prefilled"),
        "{sized:?}: {message}"
    );

    let ran = run(GEMMA_SERVED, "pids-agree")?;
    assert_eq!(ran.extra("asr_context_sizing"), Some(sized.row()));
    Ok(())
}

/// The audio count is the clip's alone, so a cell with other text beside the
/// same clip reads it back rather than count again, and its server confirms
/// it against that cell's own text.
#[test]
fn the_count_serves_every_text_length() -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    fixture.run_in(throughput_with_text(0)?, &COUNT_TEXT, |_| {})?;
    let with_text = fixture.run_in(throughput_with_text(TEXT_TOKENS)?, &COUNT_TEXT, |_| {})?;
    assert_eq!(fixture.helper_runs()?.len(), 1);
    assert_eq!(fixture.cache_rows()?, [u64::from(COUNTED)]);
    assert_eq!(
        with_text.extra("asr_audio_tokens_source"),
        Some("calibrated")
    );
    assert_eq!(with_text.extra("asr_context_sizing"), Some("exact"));
    Ok(())
}

/// A raw flag the recipe refuses fails the cell before the helper or the
/// server loads the model.
#[rstest::rstest]
#[case::throughput_reasoning(throughput(), &["-rea", "on"])]
#[case::throughput_template(throughput(), &["--chat-template-file", "t.jinja"])]
#[case::memory_reasoning(max_memory(), &["--reasoning-budget", "-1"])]
#[case::memory_template(max_memory(), &["--no-jinja"])]
#[case::throughput_underscore_spelling(throughput(), &["--chat_template_kwargs={}"])]
fn a_refused_raw_flag_loads_nothing(
    #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
    #[case] raw: &[&str],
) -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let flag = raw[0];
    let raw: Vec<String> = raw.iter().map(|&t| t.to_owned()).collect();
    let err = fixture
        .run(benchmark?, |flags| flags.raw = raw)
        .err()
        .context("a refused flag ran")?;
    let message = format!("{err:#}");
    // plan-types reserves the thinking and template flags for both ASR cells,
    // so those stop at the flags' own check; the recipe refuses them too.
    let name = flag.split_once('=').map_or(flag, |(name, _)| name);
    assert!(
        message.contains(&format!("raw flag \"{flag}\" is reserved"))
            || message.contains(&format!("raw flag \"{name}\" is reserved"))
            || (message.contains(&format!("raw {} flag", name.replace('_', "-")))
                && message.contains("the prompt is the recipe's")),
        "{message}"
    );
    assert!(fixture.helper_runs()?.is_empty(), "the helper ran");
    assert!(!fixture.root.path().join("chat-bodies.jsonl").exists());
    Ok(())
}

/// An env forward of a template or reasoning knob fails the cell before the
/// helper or the server loads the model, as its flag does.
#[rstest::rstest]
#[case::throughput(throughput(), "LLAMA_ARG_CHAT_TEMPLATE=chatml")]
#[case::memory(max_memory(), "LLAMA_ARG_THINK_BUDGET=-1")]
fn a_refused_env_forward_loads_nothing(
    #[case] benchmark: anyhow::Result<BenchmarkDefinition>,
    #[case] forward: &str,
) -> anyhow::Result<()> {
    let fixture = Fixture::new(COUNTED, 0)?;
    let name = forward.split_once('=').map_or(forward, |(name, _)| name);
    let env = EnvForward::try_new(forward.to_owned())?;
    let err = fixture
        .run(benchmark?, |flags| flags.envs = vec![env])
        .err()
        .context("a refused env forward ran")?;
    let message = format!("{err:#}");
    // plan-types reserves these forwards for both ASR cells; the recipe
    // refuses them too.
    assert!(
        message.contains(&format!("env forward \"{name}\" is reserved"))
            || message.contains(&format!("the env forward {name} would")),
        "{message}"
    );
    assert!(fixture.helper_runs()?.is_empty(), "the helper ran");
    assert!(!fixture.root.path().join("chat-bodies.jsonl").exists());
    Ok(())
}
