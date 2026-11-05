//! `asr_throughput` sends the clip to `/v1/chat/completions` as an
//! `input_audio` part, asks the way the projector's audio family is asked to,
//! sizes the context from the clip, and reports the audio tokens, with
//! `total_time_ms` on this client's clock rather than the server's timers. The
//! fake `llama-server` counts 20 prompt tokens per clip and 5 for the template.
//!
//! The fake reads its knobs from the environment, which is process-wide, so
//! the scenarios take one lock and run one at a time.

mod common;

use std::path::Path;
use std::sync::Mutex;

use anyhow::Context;
use common::{
    assert_distinct_knobs, assert_server_log, audio_and_vision_projector, audio_projector,
    fake_run_request, fake_runtime, fake_runtime_install_dir, fake_vision_model, ignore_reps,
    no_corpus, no_readiness_gate, with_distinct_knobs,
};
use serde_json::Value;

use pipette_plan_types::{
    AsrDecodeTokens, AsrThroughputBenchmark, AudioChannels, AudioDurationMillis, AudioSampleRate,
    BenchmarkDefinition, BenchmarkId, BenchmarkResultData, NonNegativeCount, RuntimeFlags,
};

static ENV: Mutex<()> = Mutex::new(());

/// How long the fake holds each chat reply, which only the client's clock sees.
const CHAT_DELAY_MS: u32 = 20;

fn projector(kind: &str) -> Vec<u8> {
    audio_projector(Some(kind))
}

struct Ran {
    bodies: Vec<Value>,
    argv: Vec<String>,
    data: BenchmarkResultData,
    runtime_flags: Option<RuntimeFlags>,
}

fn run(mmproj: &[u8], duration_ms: u32, decode_tokens: u32) -> anyhow::Result<Ran> {
    run_with(mmproj, duration_ms, decode_tokens, None, 0)
}

/// [`run`], with `/props` reporting `props_template` when given, and
/// `text_tokens` of text beside the clip. With text, the fake counts a token a
/// word of the turns, and its `/tokenize` adds a BOS for special tokens.
fn run_with(
    mmproj: &[u8],
    duration_ms: u32,
    decode_tokens: u32,
    props_template: Option<&str>,
    text_tokens: u32,
) -> anyhow::Result<Ran> {
    let _guard = ENV
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for name in ["FAKE_LLAMA_COUNT_TEXT", "FAKE_LLAMA_TOKENIZE_BOS"] {
        if text_tokens > 0 {
            // SAFETY: the lock above serializes every reader and writer of this var.
            unsafe { std::env::set_var(name, "1") };
        } else {
            // SAFETY: as above.
            unsafe { std::env::remove_var(name) };
        }
    }
    let root = tempfile::tempdir()?;
    let bodies = root.path().join("chat-bodies.jsonl");
    // SAFETY: the lock above serializes every reader and writer of this var.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_BODY_FILE", &bodies) };
    // SAFETY: as above.
    unsafe { std::env::set_var("FAKE_LLAMA_CHAT_DELAY_MS", CHAT_DELAY_MS.to_string()) };
    match props_template {
        // SAFETY: as above.
        Some(template) => unsafe { std::env::set_var("FAKE_LLAMA_PROPS_TEMPLATE", template) },
        // SAFETY: as above.
        None => unsafe { std::env::remove_var("FAKE_LLAMA_PROPS_TEMPLATE") },
    }
    let model_path = root.path().join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.path().join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, mmproj)?;

    let benchmark = BenchmarkDefinition::AsrThroughput(AsrThroughputBenchmark {
        benchmark_id: BenchmarkId::try_new("asr-chat-endpoint-test".to_owned())?,
        parameter_audio_duration_ms: AudioDurationMillis::try_new(i32::try_from(duration_ms)?)?,
        parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
        parameter_audio_channels: AudioChannels::try_new(1)?,
        parameter_text_tokens: NonNegativeCount::try_new(i32::try_from(text_tokens)?)?,
        parameter_decode_tokens: AsrDecodeTokens::try_new(i32::try_from(decode_tokens)?)?,
    });
    let runtime = fake_runtime(&fake_runtime_install_dir(root.path())?)?;
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    let req = with_distinct_knobs(fake_run_request(runtime, model, benchmark, None))?;

    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("asr_throughput should complete against the fake chat endpoint")?;
    assert_distinct_knobs(&req.benchmark, &outcome.command)?;
    assert_server_log(&outcome.runtime_output)?;
    let bodies = if bodies.exists() {
        read_bodies(&bodies)?
    } else {
        Vec::new()
    };
    Ok(Ran {
        bodies,
        argv: outcome.command,
        data: outcome.result_data,
        runtime_flags: outcome.runtime_flags,
    })
}

fn read_bodies(path: &Path) -> anyhow::Result<Vec<Value>> {
    std::fs::read_to_string(path)?
        .lines()
        .map(|line| serde_json::from_str(line).context("a chat body that is not JSON"))
        .collect()
}

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

/// The warm-up and five reps send the clip; the probe that measures the audio
/// tokens (there is no `llama-mtmd-cli` in the fake runtime to count them
/// exactly) is the same turns with no clip.
fn assert_clip_then_probe(bodies: &[Value]) {
    assert_eq!(
        bodies.iter().map(clips).collect::<Vec<_>>(),
        [1, 1, 1, 1, 1, 1, 0]
    );
}

#[test]
fn the_clip_is_sent_as_audio_and_its_tokens_are_measured() -> anyhow::Result<()> {
    let ran = run(&projector("gemma4a"), 2_500, 4)?;
    let BenchmarkResultData::AsrThroughput {
        asr_prompt_tokens,
        asr_audio_tokens,
        asr_prompt_ms,
        asr_predicted_ms,
        total_time_ms,
        total_time_ms_stddev,
        ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    let total_time_ms = total_time_ms.context("a llama.cpp ASR row times the whole request")?;
    assert!(total_time_ms >= f64::from(CHAT_DELAY_MS), "{total_time_ms}");
    // The fake's own timers say a few ms; the client's clock saw the hold.
    assert!(
        total_time_ms > asr_prompt_ms + asr_predicted_ms,
        "{total_time_ms} vs {asr_prompt_ms} + {asr_predicted_ms}"
    );
    assert!(total_time_ms_stddev.is_some());
    // Fake server: 5 for the template, 20 for the one clip.
    assert_eq!(asr_prompt_tokens, 25);
    // With the clip (25) less the same turns without it (5).
    assert_eq!(asr_audio_tokens, 20);
    assert_clip_then_probe(&ran.bodies);

    let first = &ran.bodies[0];
    assert_eq!(first["max_tokens"], 4);
    assert_eq!(first["ignore_eos"], true);
    assert_eq!(first["cache_prompt"], false);
    let turns = first["messages"].as_array().context("messages")?;
    assert_eq!(turns.len(), 1, "no system turn for Gemma: {first}");
    let parts = turns[0]["content"].as_array().context("content")?;
    let ask = parts[0]["text"]
        .as_str()
        .context("Gemma's ask is a text part")?;
    let audio = &parts[1]["input_audio"];
    assert_eq!(audio["format"], "wav");
    // A WAV file: RIFF in base64.
    assert!(
        audio["data"]
            .as_str()
            .is_some_and(|d| d.starts_with("UklGR")),
        "{audio}"
    );
    // The probe carries the ask and no clip.
    assert_eq!(
        ran.bodies[6]["messages"][0]["content"],
        serde_json::json!([{ "type": "text", "text": ask }])
    );
    Ok(())
}

/// Without the helper the context is the upper bound (750 for a clip under
/// 30 s), the ask's characters, the decode length and the margin, for a recipe
/// with a system turn (LFM2) and one with the ask after the clip (Qwen3-Omni).
#[rstest::rstest]
#[case::lfm(audio_projector(Some("lfm2a")), 750 + 12)]
#[case::qwen3_omni(audio_and_vision_projector("qwen3a"), 750 + 39)]
fn the_context_is_sized_from_the_clip_when_it_cannot_be_counted(
    #[case] mmproj: Vec<u8>,
    #[case] audio_and_ask: u32,
) -> anyhow::Result<()> {
    let ran = run(&mmproj, 2_500, 4)?;
    let Some(RuntimeFlags::AsrThroughputLlamacppCliStockToolsGgufVision(flags)) = ran.runtime_flags
    else {
        anyhow::bail!(
            "expected the ASR cell's runtime flags: {:?}",
            ran.runtime_flags
        );
    };
    // The bound and the ask, 4 decoded, 64 of margin.
    assert_eq!(flags.ctx_size, Some(audio_and_ask + 4 + 64));
    Ok(())
}

#[test]
fn an_lfm_projector_gets_the_system_turn_and_the_clip_alone() -> anyhow::Result<()> {
    let ran = run(&projector("lfm2a"), 2_500, 4)?;
    assert_clip_then_probe(&ran.bodies);
    assert_eq!(
        ran.bodies[0]["messages"],
        serde_json::json!([
            { "role": "system", "content": "Perform ASR." },
            { "role": "user", "content": [
                { "type": "input_audio", "input_audio": {
                    "data": ran.bodies[0]["messages"][1]["content"][0]["input_audio"]["data"],
                    "format": "wav",
                }},
            ]},
        ])
    );
    // The baseline of a clip sent alone is an empty user turn.
    assert_eq!(ran.bodies[6]["messages"][1]["content"], "");
    Ok(())
}

/// Every ASR server runs with thinking off.
#[test]
fn a_gemma_projector_turns_reasoning_off_on_the_server() -> anyhow::Result<()> {
    let ran = run(&projector("gemma4a"), 2_500, 4)?;
    assert!(
        ran.argv
            .windows(2)
            .any(|pair| pair == ["--reasoning", "off"]),
        "{:?}",
        ran.argv
    );
    let first = &ran.bodies[0];
    let ask = first["messages"][0]["content"][0]["text"]
        .as_str()
        .context("Gemma's ask is a text part")?;
    assert!(
        ask.starts_with("Transcribe the following speech segment in English into English text."),
        "{ask}"
    );
    Ok(())
}

#[test]
fn an_lfm3_projector_runs_its_own_template() -> anyhow::Result<()> {
    let ran = run(&projector("lfm3a"), 2_500, 4)?;
    assert!(
        ran.argv.iter().any(|token| token == "--jinja"),
        "{:?}",
        ran.argv
    );
    let template = ran
        .argv
        .windows(2)
        .find(|pair| pair[0] == "--chat-template")
        .map(|pair| pair[1].as_str())
        .context("no --chat-template in the argv")?;
    assert!(template.starts_with("{{- bos_token -}}"), "{template}");
    assert!(!ran.argv.iter().any(|token| token == "--no-jinja"));
    assert_eq!(
        ran.bodies[0]["messages"][0],
        serde_json::json!({ "role": "system", "content": "Perform ASR in English." })
    );
    let parts = ran.bodies[0]["messages"][1]["content"]
        .as_array()
        .context("the user turn is a list of parts")?;
    let kinds: Vec<&str> = parts.iter().filter_map(|p| p["type"].as_str()).collect();
    assert_eq!(kinds, ["input_audio"], "{parts:?}");
    Ok(())
}

/// Voxtral gets its transcription template, the language and the clip alone,
/// as the reference evaluation asks it.
#[test]
fn a_voxtral_projector_runs_in_transcription_mode() -> anyhow::Result<()> {
    let ran = run(&projector("voxtral"), 2_500, 4)?;
    assert!(
        ran.argv.iter().any(|token| token == "--jinja"),
        "{:?}",
        ran.argv
    );
    let template = ran
        .argv
        .windows(2)
        .find(|pair| pair[0] == "--chat-template")
        .map(|pair| pair[1].as_str())
        .context("no --chat-template in the argv")?;
    assert!(template.contains("[TRANSCRIBE]"), "{template}");
    let kwargs = ran
        .argv
        .windows(2)
        .find(|pair| pair[0] == "--chat-template-kwargs")
        .map(|pair| pair[1].as_str())
        .context("no --chat-template-kwargs in the argv")?;
    assert_eq!(kwargs, r#"{"transcription":true,"language":"en"}"#);
    assert!(!ran.argv.iter().any(|token| token == "--no-jinja"));
    let parts = ran.bodies[0]["messages"][0]["content"]
        .as_array()
        .context("the user turn is a list of parts")?;
    let kinds: Vec<&str> = parts.iter().filter_map(|p| p["type"].as_str()).collect();
    assert_eq!(kinds, ["input_audio"], "{parts:?}");
    Ok(())
}

/// A projector the table has no entry for is refused, named, before anything
/// runs: no prompt for it was checked against a reference.
#[rstest::rstest]
#[case::unnamed(None)]
#[case::unknown(Some("ultravox"))]
fn a_projector_without_a_recipe_is_refused(#[case] kind: Option<&str>) -> anyhow::Result<()> {
    let err = run(&audio_projector(kind), 2_500, 4)
        .err()
        .context("an encoder without a recipe ran")?;
    assert!(
        format!("{err:#}").contains("no ASR recipe for the audio encoder"),
        "{err:#}"
    );
    Ok(())
}

/// Qwen3-ASR gets its transcription template, the language forced and the
/// clip alone, as its processor asks it.
#[test]
fn a_qwen3_asr_projector_runs_its_transcription_template() -> anyhow::Result<()> {
    let ran = run(&projector("qwen3a"), 2_500, 4)?;
    let flag = |name: &str| {
        ran.argv
            .windows(2)
            .find(|pair| pair[0] == name)
            .map(|pair| pair[1].clone())
    };
    let template = flag("--chat-template").context("no --chat-template in the argv")?;
    assert!(template.contains("<asr_text>"), "{template}");
    assert_eq!(
        flag("--chat-template-kwargs").as_deref(),
        Some(r#"{"language":"English"}"#)
    );
    assert!(ran.argv.iter().any(|token| token == "--jinja"));
    let turns = ran.bodies[0]["messages"].as_array().context("messages")?;
    assert_eq!(turns.len(), 1, "{turns:?}");
    let kinds: Vec<&str> = turns[0]["content"]
        .as_array()
        .context("the user turn is a list of parts")?
        .iter()
        .filter_map(|p| p["type"].as_str())
        .collect();
    assert_eq!(kinds, ["input_audio"]);
    Ok(())
}

/// Qwen3-Omni (`qwen3a` beside a vision encoder) is asked as its card asks:
/// the clip, then the English transcription prompt, on the GGUF's template.
#[test]
fn a_qwen3_omni_projector_is_asked_after_the_clip() -> anyhow::Result<()> {
    let ran = run(&audio_and_vision_projector("qwen3a"), 2_500, 4)?;
    assert!(!ran.argv.iter().any(|t| t.starts_with("--chat-template")));
    assert_eq!(
        ran.bodies[0]["messages"][0]["content"][1],
        serde_json::json!({ "type": "text", "text": "Transcribe the English audio into text." })
    );
    assert_eq!(
        ran.bodies[0]["messages"][0]["content"][0]["type"],
        "input_audio"
    );
    Ok(())
}

/// A server that does not report the recipe's template (one whose jinja
/// engine fell back to another) is refused before any clip is sent; the
/// template it was started with passes.
#[rstest::rstest]
#[case::fallen_back(Some("{{ chatml }}"), false)]
#[case::served(None, true)]
fn a_server_not_rendering_the_recipes_template_is_refused(
    #[case] reported: Option<&str>,
    #[case] accepted: bool,
) -> anyhow::Result<()> {
    match run_with(&projector("voxtral"), 2_500, 4, reported, 0) {
        Ok(ran) => assert!(accepted && !ran.bodies.is_empty()),
        Err(err) => assert!(
            !accepted && format!("{err:#}").contains("is not rendering the recipe's chat template"),
            "{err:#}"
        ),
    }
    Ok(())
}

/// The user turn's parts in order, `clip` for the audio.
fn user_parts(body: &Value) -> anyhow::Result<Vec<String>> {
    let turns = body["messages"].as_array().context("messages")?;
    let user = turns
        .iter()
        .find(|t| t["role"] == "user")
        .context("a user turn")?;
    Ok(match &user["content"] {
        Value::String(text) => vec![text.clone()],
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("input_audio") => "clip".to_owned(),
                _ => part["text"].as_str().unwrap_or_default().to_owned(),
            })
            .collect(),
        other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::Object(_)) => {
            anyhow::bail!("a user turn of {other}")
        }
    })
}

const TEXT_TOKENS: u32 = 32;

/// One rule for every family: the cell's text goes first in the ask's own
/// part, on the ask's side of the clip. The text is exactly the cell's count
/// under the server's tokenizer, which would add a BOS to a special-token
/// count; the probe carries the same text; and the context is sized for it.
#[rstest::rstest]
#[case::lfm2(audio_projector(Some("lfm2a")), Some("Perform ASR."), "", true)]
#[case::lfm3(
    audio_projector(Some("lfm3a")),
    Some("Perform ASR in English."),
    "",
    true
)]
#[case::qwen25_omni(
    audio_projector(Some("qwen2.5o")),
    Some("You are a speech recognition model."),
    "Transcribe the English audio into text without any punctuation marks.",
    false
)]
#[case::gemma4(
    audio_projector(Some("gemma4a")),
    None,
    "Transcribe the following speech segment in English into English text.",
    true
)]
#[case::minicpm_o(
    audio_projector(Some("minicpmoa")),
    None,
    "Please listen to the audio carefully and transcribe it with high precision.",
    true
)]
#[case::qwen3_asr(audio_projector(Some("qwen3a")), None, "", true)]
#[case::qwen3_omni(
    audio_and_vision_projector("qwen3a"),
    None,
    "Transcribe the English audio into text.",
    false
)]
#[case::voxtral(audio_projector(Some("voxtral")), None, "", false)]
fn the_cells_text_joins_each_familys_ask(
    #[case] mmproj: Vec<u8>,
    #[case] system: Option<&str>,
    #[case] ask_starts: &str,
    #[case] text_before_clip: bool,
) -> anyhow::Result<()> {
    let ran = run_with(&mmproj, 2_500, 4, None, TEXT_TOKENS)?;
    // The clip's requests, the probe with the text and no clip, then the
    // turns without the text either, which confirm the text's tokens.
    assert_eq!(
        ran.bodies.iter().map(clips).collect::<Vec<_>>(),
        [1, 1, 1, 1, 1, 1, 0, 0]
    );
    let first = &ran.bodies[0];
    match system {
        Some(system) => assert_eq!(
            first["messages"][0],
            serde_json::json!({ "role": "system", "content": system })
        ),
        None => assert_eq!(first["messages"].as_array().map(Vec::len), Some(1)),
    }
    let parts = user_parts(first)?;
    let [text, clip] = if text_before_clip {
        [&parts[0], &parts[1]]
    } else {
        [&parts[1], &parts[0]]
    };
    assert_eq!((parts.len(), clip.as_str()), (2, "clip"), "{parts:?}");
    let (cells_text, ask) = text.split_once('\n').unwrap_or((text.as_str(), ""));
    assert_eq!(
        cells_text.split_whitespace().count(),
        usize::try_from(TEXT_TOKENS)?,
        "{cells_text:?}"
    );
    assert!(ask.starts_with(ask_starts), "{ask:?}");
    assert_eq!(ask.is_empty(), ask_starts.is_empty(), "{text:?}");
    // The probe: the same text, no clip; then the ask alone.
    assert_eq!(user_parts(&ran.bodies[6])?, std::slice::from_ref(text));
    let bare = user_parts(&ran.bodies[7])?;
    assert_eq!(bare.concat(), ask, "{bare:?}");

    let BenchmarkResultData::AsrThroughput {
        asr_prompt_tokens,
        asr_audio_tokens,
        ..
    } = ran.data
    else {
        anyhow::bail!("expected AsrThroughput result data");
    };
    let words = |body: &Value| -> anyhow::Result<u32> {
        let system = body["messages"][0]["content"]
            .as_str()
            .filter(|_| system.is_some());
        let user: usize = user_parts(body)?
            .iter()
            .filter(|p| *p != "clip")
            .map(|p| p.split_whitespace().count())
            .sum();
        Ok(u32::try_from(
            user + system.map_or(0, |s| s.split_whitespace().count()),
        )?)
    };
    // The fake: 5 for the template, 20 for the clip, a token a word.
    assert_eq!(asr_prompt_tokens, 5 + 20 + words(first)?);
    assert_eq!(asr_audio_tokens, 20);

    let Some(RuntimeFlags::AsrThroughputLlamacppCliStockToolsGgufVision(flags)) = ran.runtime_flags
    else {
        anyhow::bail!("expected the ASR cell's runtime flags");
    };
    let ask_chars = system.map_or(0, str::len) + ask.len();
    // No helper: the bound (750), the text, the ask's characters, 4 decoded,
    // 64 of margin.
    assert_eq!(
        flags.ctx_size,
        Some(750 + TEXT_TOKENS + u32::try_from(ask_chars)? + 4 + 64)
    );
    Ok(())
}
