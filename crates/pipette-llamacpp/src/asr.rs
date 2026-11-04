//! Speech recognition on llama-server: the ask per audio family, the audio
//! tokens a clip becomes, and the context they need.
//!
//! **The count** is not a formula: the encoder's frame rate, 30 s windows and
//! pooling decide it. Upstream's own `llama-mtmd-cli` (in the runtime's archive,
//! so version-matched) prefills a clip of the same length with `-v`, and its
//! `decoding audio batch …, n_tokens_batch = N` lines sum to what the backbone
//! receives between the audio markers (the markers are text chunks). It depends
//! on the sample count alone (a tone, silence and speech of one length count
//! alike), so it is memoized per `(projector fingerprint, helper build, clip
//! format and samples, forwards)` for every cell and device.
//!
//! **The recipe.** A chat model transcribes only when asked as it was trained,
//! and the ask is part of the timed prefill. The projector's metadata names the
//! audio encoder, which picks the ask, so no plan spells it out; a `raw` flag or
//! env forward that would change the ask or turn thinking on is refused before
//! anything loads the model.
//!
//! The count is best-effort: any failure (helper absent, unparseable log,
//! unreadable cache) returns `None`, and the caller measures the count instead
//! (the prompt with the clip less the prompt without it).

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;

use pipette_ops::wav;
use pipette_plan_types::{
    AsrAudioTokensSource, AsrContextSizing, AudioDurationMillis, EnvForward, LlamacppCliStockTools,
};

use crate::gguf_meta;
use crate::mtmd::{self, HELPER_CTX_FLOOR, MtmdCommandBuilder, VL_CALIBRATION_TIMEOUT};
use crate::runtimes;

const AUDIO_BATCH_LINE: &str = "decoding audio batch";

const CACHE_FILE: &str = "audio_calibration.json";

/// Tokens on top of the exact audio and decode count, for the chat template's
/// scaffolding (role markers, the audio begin/end tokens). The ask is added
/// separately.
const CTX_MARGIN: u32 = 64;

/// Each refuses a cell before anything loads the model, except the last three,
/// which only the running server reveals.
#[derive(Debug, thiserror::Error)]
pub enum AsrRefusalError {
    #[error(
        "the projector holds no audio encoder (`clip.has_audio_encoder` is not set): an ASR \
         cell needs a model whose projector takes audio"
    )]
    NoAudioEncoder,

    #[error(
        "{label}: raw {flag} in --runtime-flags would divide the context sized for one clip \
         among several slots; an ASR cell runs one request at a time"
    )]
    RawParallel {
        label: &'static str,
        flag: &'static str,
    },

    #[error("{recipe}: ASR runs with thinking off on every model; drop the raw {flag} flag")]
    RawReasoning { recipe: &'static str, flag: String },

    #[error(
        "{recipe}: the prompt is the recipe's, and the raw {flag} flag would change what the \
         server renders and the benchmark times; drop it"
    )]
    RawPromptKnob { recipe: &'static str, flag: String },

    #[error(
        "{recipe}: the {origin} {name} would turn thinking on or change the prompt the recipe \
         renders; drop it"
    )]
    EnvKnob {
        recipe: &'static str,
        origin: &'static str,
        name: String,
    },

    #[error(
        "{recipe}: the clip is {duration_ms} ms at {sample_rate_hz} Hz, a whole number of the \
         encoder's {window_ms} ms windows; resampled to 16 kHz it can come out a sample or two \
         past the last window, which the encoder cannot encode, so the cell is refused rather \
         than left to the resampler's rounding. Send it at 16 kHz, or at a length that is not a \
         multiple of {window_ms} ms"
    )]
    ClipOnWindowEdge {
        recipe: &'static str,
        duration_ms: u32,
        sample_rate_hz: u32,
        window_ms: u32,
    },

    #[error(
        "no ASR recipe for the audio encoder {name:?}: a cell times the prompt the model's \
         public usage asks it with, and none has been checked for this encoder; add a recipe \
         (crates/pipette-llamacpp/src/asr.rs) rather than time an unchecked one"
    )]
    NoRecipe { name: String },

    #[error(
        "{recipe}: the server is not rendering the recipe's chat template (its /props reports \
         another), so it would not prefill the prompt the recipe times"
    )]
    TemplateNotServed { recipe: &'static str },

    #[error(
        "the server prefilled {with_text} prompt tokens with the cell's text and {without_text} \
         without it, where the text is {text_tokens} tokens: the row would not be the text the \
         entry names"
    )]
    TextUnconfirmed {
        text_tokens: u32,
        with_text: u32,
        without_text: u32,
    },

    #[error(
        "the server prefilled {measured} more tokens with the clip than without it, where the \
         calibrated count of {calibrated} audio tokens and the recipe's {markers} audio-marker \
         tokens make {expected}: the count is not the server's, so the context was not sized \
         exactly; its row in {cache} is dropped and the next run counts the clip again",
        expected = calibrated.saturating_add(*markers)
    )]
    CountUnconfirmed {
        calibrated: u32,
        markers: u32,
        measured: u32,
        cache: PathBuf,
    },
}

/// Every alias of a family the suite runs maps here, once; any other
/// (`gemma3na`, `qwen2a`, `ultravox`, `glma`, `granite_speech`,
/// `musicflamingo`, …) is [`Self::Other`], which no recipe asks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioEncoder {
    Voxtral,
    Lfm2Audio,
    Lfm3Audio,
    Qwen25Omni,
    /// `gemma4a`: Gemma 4 E2B's and E4B's encoder, which cuts a clip into 30 s
    /// windows.
    Gemma4,
    /// `gemma4ua`: Gemma 4 12B's encoder-free audio path, which frames the
    /// whole clip at once.
    Gemma4Unified,
    /// `qwen3a` in a projector of its own.
    Qwen3Asr,
    /// `qwen3a` beside a vision encoder: Qwen3-Omni's projector carries both
    /// towers.
    Qwen3Omni,
    MiniCpmO,
    /// Empty when the projector names none.
    Other(String),
}

impl AudioEncoder {
    pub fn from_name(name: &str, with_vision: bool) -> Self {
        match name {
            "voxtral" => Self::Voxtral,
            "lfm2a" => Self::Lfm2Audio,
            "lfm3a" => Self::Lfm3Audio,
            "qwen2.5o" | "qwen2.5oa" => Self::Qwen25Omni,
            "gemma4a" => Self::Gemma4,
            "gemma4ua" => Self::Gemma4Unified,
            "qwen3a" if with_vision => Self::Qwen3Omni,
            "qwen3a" => Self::Qwen3Asr,
            "minicpmoa" => Self::MiniCpmO,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn of(metadata: &gguf_meta::ProjectorMetadata) -> Result<Self, AsrRefusalError> {
        if !metadata.holds_audio_encoder() {
            return Err(AsrRefusalError::NoAudioEncoder);
        }
        Ok(Self::from_name(
            metadata.audio_encoder_name().unwrap_or(""),
            metadata.has_vision_encoder,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeKind {
    Voxtral,
    Lfm2Audio,
    Lfm3Audio,
    Qwen25Omni,
    Gemma4,
    MiniCpmO,
    Qwen3Asr,
    Qwen3Omni,
}

impl RecipeKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Voxtral => "voxtral",
            Self::Lfm2Audio => "lfm2-audio",
            Self::Lfm3Audio => "lfm3-audio",
            Self::Qwen25Omni => "qwen2.5-omni",
            Self::Gemma4 => "gemma4",
            Self::MiniCpmO => "minicpm-o",
            Self::Qwen3Asr => "qwen3-asr",
            Self::Qwen3Omni => "qwen3-omni",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentOrder {
    InstructionThenClip,
    ClipThenInstruction,
}

/// Thinking is not a kwarg: `--reasoning off` sets the template's
/// `enable_thinking` (`common/arg.cpp`), and the kwarg spelling is deprecated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
struct ChatTemplateKwargs {
    #[serde(skip_serializing_if = "Option::is_none")]
    use_tts_template: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcription: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerOption {
    /// Every ASR server runs with it ([`Recipe::server_args`]); no recipe lists
    /// it.
    ReasoningOff,
    ChatTemplateKwargs(ChatTemplateKwargs),
    ChatTemplate(&'static str),
}

impl ServerOption {
    fn argv(self) -> anyhow::Result<Vec<String>> {
        Ok(match self {
            Self::ReasoningOff => vec!["--reasoning".to_owned(), "off".to_owned()],
            Self::ChatTemplateKwargs(kwargs) => vec![
                "--chat-template-kwargs".to_owned(),
                serde_json::to_string(&kwargs)?,
            ],
            Self::ChatTemplate(template) => vec![
                "--jinja".to_owned(),
                "--chat-template".to_owned(),
                template.to_owned(),
            ],
        })
    }
}

/// How a family of audio models is asked to transcribe.
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    /// Named in the log, so a row can be traced to the ask that produced it.
    pub kind: RecipeKind,
    /// For a model whose ASR mode is entered through a system turn.
    pub system: Option<&'static str>,
    /// Empty sends the clip alone.
    pub instruction: &'static str,
    /// The side of the clip the instruction goes on, and with it any text the
    /// cell adds ([`Recipe::user_text`]); for a recipe that sends the clip
    /// alone, the side such text goes on.
    pub order: ContentOrder,
    server_options: &'static [ServerOption],
    /// The longest clip the model's documentation names. A longer one still
    /// runs, flagged ([`Recipe::beyond_documented_limit`]).
    documented_max_audio_ms: Option<u32>,
    /// The windows the encoder cuts a clip into, where a last window of a
    /// sample or two fails the encode ([`Recipe::check_duration`]).
    encoder_window_ms: Option<u32>,
    /// Prompt tokens a clip adds beside its audio tokens: the encoder's
    /// begin/end markers, with whatever the join around them tokenizes to
    /// (measured, see the `MARKERS_*` constants). The server's confirmation of
    /// a count, and so `exact`, needs it.
    audio_markers: u32,
}

/// Pipette's own, since the published GGUFs carry a text-only Devstral
/// template that renders neither the audio nor the `[TRANSCRIBE]` turn. Its
/// format and verification: `asr_templates/README.md`.
const VOXTRAL_CHAT_TEMPLATE: &str = include_str!("asr_templates/voxtral.jinja");

/// Voxtral's transcription mode, as its model card
/// (`mistralai/Voxtral-Mini-3B-2507`, "Transcription") asks it through
/// `TranscriptionRequest(language="en")`. Text a cell adds goes after the
/// clip, where the card's audio-instruct examples put a user's text.
const VOXTRAL: Recipe = Recipe {
    kind: RecipeKind::Voxtral,
    system: None,
    instruction: "",
    order: ContentOrder::ClipThenInstruction,
    server_options: &[
        ServerOption::ChatTemplate(VOXTRAL_CHAT_TEMPLATE),
        ServerOption::ChatTemplateKwargs(ChatTemplateKwargs {
            use_tts_template: None,
            transcription: Some(true),
            language: Some("en"),
        }),
    ],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_VOXTRAL,
};

/// LFM2-Audio's ASR mode is entered by the system prompt, with the clip alone,
/// as Liquid's public usage asks it (the `LiquidAI/LFM2.5-Audio-1.5B` card's
/// `liquid-audio` package, "ASR"; its GGUF card's `-sys "Perform ASR."`); a
/// user instruction beside the clip turns it back into a chat model.
const LFM2_AUDIO: Recipe = Recipe {
    kind: RecipeKind::Lfm2Audio,
    system: Some("Perform ASR."),
    instruction: "",
    order: ContentOrder::InstructionThenClip,
    server_options: &[],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_LFM2_AUDIO,
};

/// LFM3's native ASR prompt. Only text parts render: the server hands the
/// media over as its marker in a text part.
const LFM3_ASR_TEMPLATE: &str = concat!(
    "{{- bos_token -}}",
    "{%- for message in messages -%}",
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}",
    "{%- if message['content'] is string -%}{{- message['content'] -}}",
    "{%- else -%}{%- for part in message['content'] -%}",
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}",
    "{%- endfor -%}{%- endif -%}",
    "{{- '<|im_end|>\\n' -}}",
    "{%- endfor -%}",
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}",
);

/// LFM3-Audio's native ASR format, as the `LiquidAI/LFM3-Audio-2B` card gives it. It needs
/// [`LFM3_ASR_TEMPLATE`]: the GGUF's own template opens a `<think>` block by
/// default (the whole answer then lands in the reasoning and every transcript
/// comes back empty), and with thinking off renders an empty `<think></think>`
/// the ASR model was never trained with; the server's plain ChatML path drops
/// the BOS the native prompt starts with.
const LFM3_AUDIO: Recipe = Recipe {
    kind: RecipeKind::Lfm3Audio,
    system: Some("Perform ASR in English."),
    instruction: "",
    order: ContentOrder::InstructionThenClip,
    server_options: &[ServerOption::ChatTemplate(LFM3_ASR_TEMPLATE)],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_LFM3_AUDIO,
};

/// Qwen2.5-Omni summarizes instead of transcribing without the Qwen cookbook's
/// ASR prompt (`QwenLM/Qwen2.5-Omni`
/// `cookbooks/universal_audio_understanding.ipynb`).
const QWEN25_OMNI: Recipe = Recipe {
    kind: RecipeKind::Qwen25Omni,
    system: Some("You are a speech recognition model."),
    instruction: "Transcribe the English audio into text without any punctuation marks.",
    order: ContentOrder::ClipThenInstruction,
    server_options: &[],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_QWEN25_OMNI,
};

/// Gemma 4 is documented to take at most 30 s of audio ("Audio supports a
/// maximum length of 30 seconds", every Gemma 4 card; E2B's and E4B's
/// processors cut a clip to 750 tokens), where llama.cpp encodes the whole
/// clip.
const GEMMA4_MAX_AUDIO_MS: u32 = 30_000;

/// The windows llama.cpp's Gemma 4 E2B/E4B audio preprocessor (`gemma4a`)
/// cuts a clip into (`mtmd-audio.cpp`, 30 s of 16 kHz samples each). A last
/// window of a sample or two makes no mel frame, and the encode fails
/// ("invalid mel dimensions"). 12B's `gemma4ua` frames the whole clip in one
/// piece (`mtmd_audio_preprocessor_gemma4ua`), so its recipe has no windows
/// ([`GEMMA4_UNIFIED`]).
const GEMMA4_WINDOW_MS: u32 = 30_000;

/// Google's ASR prompt for a known language, before the clip, as the model card
/// (`google/gemma-4-E2B-it`, "Audio") gives it. Its HF template keeps thinking
/// off, but the server's default probe turns it on ([`Recipe::server_args`]).
const GEMMA4: Recipe = Recipe {
    kind: RecipeKind::Gemma4,
    system: None,
    instruction: "Transcribe the following speech segment in English into English text.\n\n\
                  Follow these specific instructions for formatting the answer:\n\
                  * Only output the transcription, with no newlines.\n\
                  * When transcribing numbers, write the digits, i.e. write 1.7 and not \
                  one point seven, and write 3 instead of three.",
    order: ContentOrder::InstructionThenClip,
    server_options: &[],
    documented_max_audio_ms: Some(GEMMA4_MAX_AUDIO_MS),
    encoder_window_ms: Some(GEMMA4_WINDOW_MS),
    audio_markers: MARKERS_GEMMA4,
};

/// Gemma 4 12B's encoder takes the clip whole, so no length is a window edge.
const GEMMA4_UNIFIED: Recipe = Recipe {
    encoder_window_ms: None,
    ..GEMMA4
};

/// MiniCPM-o 4.5, asked as OpenBMB's own evaluation asks a general English clip
/// (OmniEvalKit's `o_e_Kit/configs/generation_configs.json`, `fleurs_en`: the
/// prompt, a newline, the clip, no system turn), as the model card's `chat()`
/// example does. Its template renders that prompt only with the TTS template
/// on, which `chat()` turns on for any audio, and the think block closed, which
/// `--reasoning off` does. The answer ends at `<|tts_eos|>`, where llama.cpp
/// does not stop; the decode runs to its limit either way.
const MINICPM_O: Recipe = Recipe {
    kind: RecipeKind::MiniCpmO,
    system: None,
    instruction: "Please listen to the audio carefully and transcribe it with high precision.\n",
    order: ContentOrder::InstructionThenClip,
    server_options: &[ServerOption::ChatTemplateKwargs(ChatTemplateKwargs {
        use_tts_template: Some(true),
        transcription: None,
        language: None,
    })],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_MINICPM_O,
};

// The audio markers each recipe's prompt adds beside the clip's audio tokens:
// the server's prompt with the clip, less the same prompt without it, less the
// helper's count. Measured with the pipette CLI on the M4 Max (2.5 s, 10 s and
// 29.9 s tones, 16 kHz mono) on the builds the suite runs. libmtmd adds them as
// text once per clip, however many windows it spans (`mtmd.cpp`), so they do
// not depend on its length.

/// LFM2-Audio: no audio markers (`lfm2a` adds none); LFM2.5-Audio 1.5B Q8_0.
const MARKERS_LFM2_AUDIO: u32 = 0;
/// LFM3-Audio: none; LFM3 Q8_0.
const MARKERS_LFM3_AUDIO: u32 = 0;
/// Voxtral: `[BEGIN_AUDIO]`; Voxtral Mini 3B Q4_K_M.
const MARKERS_VOXTRAL: u32 = 1;
/// Qwen2.5-Omni: `<|audio_bos|>` and `<|audio_eos|>`; 3B Q8_0.
const MARKERS_QWEN25_OMNI: u32 = 2;
/// Gemma 4: `<|audio>` and `<audio|>`; E2B Q4_0.
const MARKERS_GEMMA4: u32 = 2;
/// MiniCPM-o 4.5: `<|audio_start|>` and `<|audio_end|>`; Q4_K_M.
const MARKERS_MINICPM_O: u32 = 2;
/// Qwen3-ASR: `<|audio_start|>` and `<|audio_end|>`; 1.7B Q8_0.
const MARKERS_QWEN3_ASR: u32 = 2;
/// Qwen3-Omni: the same two, as `qwen3a` adds them; 30B-A3B Q4_K_M with its
/// Q8_0 projector on stock b11256, at 2.5 s, 10 s, 29.9 s and 119.9 s, with
/// and without text beside the clip.
const MARKERS_QWEN3_OMNI: u32 = 2;

/// Qwen3-ASR's prompt as its Hugging Face processor renders an English request
/// (`Qwen3ASRProcessor.apply_transcription_request(audio, language="English")`
/// with the `Qwen/Qwen3-ASR-1.7B-hf` chat template, the entry point its model
/// card recommends). The assistant prefill `language English<asr_text>` makes
/// the model write only the transcript. The GGUF's own template is plain
/// ChatML, which renders neither the always-present system turn nor the
/// prefill.
const QWEN3_ASR_TEMPLATE: &str = concat!(
    "{{- '<|im_start|>system\\n' -}}",
    "{%- for message in messages -%}{%- if message['role'] == 'system' -%}",
    "{%- if message['content'] is string -%}{{- message['content'] -}}",
    "{%- else -%}{%- for part in message['content'] -%}",
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}",
    "{%- endfor -%}{%- endif -%}",
    "{%- endif -%}{%- endfor -%}",
    "{{- '<|im_end|>\\n<|im_start|>user\\n' -}}",
    "{%- for message in messages -%}{%- if message['role'] == 'user' -%}",
    "{%- if message['content'] is string -%}{{- message['content'] -}}",
    "{%- else -%}{%- for part in message['content'] -%}",
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}",
    "{%- endfor -%}{%- endif -%}",
    "{%- endif -%}{%- endfor -%}",
    "{{- '<|im_end|>\\n' -}}",
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}",
    "{%- if language is defined and language -%}",
    "{{- 'language ' + language + '<asr_text>' -}}",
    "{%- endif -%}{%- endif -%}",
);

const QWEN3_ASR: Recipe = Recipe {
    kind: RecipeKind::Qwen3Asr,
    system: None,
    instruction: "",
    order: ContentOrder::InstructionThenClip,
    server_options: &[
        ServerOption::ChatTemplate(QWEN3_ASR_TEMPLATE),
        ServerOption::ChatTemplateKwargs(ChatTemplateKwargs {
            use_tts_template: None,
            transcription: None,
            language: Some("English"),
        }),
    ],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_QWEN3_ASR,
};

/// As Qwen3-Omni's cookbook asks an English clip (`QwenLM/Qwen3-Omni`
/// `cookbooks/speech_recognition.ipynb`; the model card's default ASR prompt,
/// `Transcribe the <language> audio into text.`).
const QWEN3_OMNI: Recipe = Recipe {
    kind: RecipeKind::Qwen3Omni,
    system: None,
    instruction: "Transcribe the English audio into text.",
    order: ContentOrder::ClipThenInstruction,
    server_options: &[],
    documented_max_audio_ms: None,
    encoder_window_ms: None,
    audio_markers: MARKERS_QWEN3_OMNI,
};

impl Recipe {
    pub fn for_encoder(encoder: &AudioEncoder) -> Result<Self, AsrRefusalError> {
        Ok(match encoder {
            AudioEncoder::Voxtral => VOXTRAL,
            AudioEncoder::Lfm2Audio => LFM2_AUDIO,
            AudioEncoder::Lfm3Audio => LFM3_AUDIO,
            AudioEncoder::Qwen25Omni => QWEN25_OMNI,
            AudioEncoder::Gemma4 => GEMMA4,
            AudioEncoder::Gemma4Unified => GEMMA4_UNIFIED,
            AudioEncoder::Qwen3Asr => QWEN3_ASR,
            AudioEncoder::Qwen3Omni => QWEN3_OMNI,
            AudioEncoder::MiniCpmO => MINICPM_O,
            AudioEncoder::Other(name) => {
                return Err(AsrRefusalError::NoRecipe { name: name.clone() });
            }
        })
    }

    pub fn for_mmproj(mmproj: &Path) -> anyhow::Result<Self> {
        let metadata = gguf_meta::projector_metadata(mmproj)?;
        let recipe = AudioEncoder::of(&metadata)
            .and_then(|encoder| Self::for_encoder(&encoder))
            .with_context(|| format!("projector {}", mmproj.display()))?;
        tracing::info!(recipe = recipe.kind.name(), mmproj = %mmproj.display(), "asr recipe");
        Ok(recipe)
    }

    /// The flags the recipe adds, after refusing every `raw` flag, env forward,
    /// host-configuration env and inherited env that would change them. Loads
    /// nothing, so a cell calls it before the calibration and the server.
    ///
    /// Thinking is off for every ASR model: a transcript has nothing to reason
    /// about, no reference protocol reasons, and a thought spends the decode
    /// budget (for LFM3, the whole answer) outside the transcript. A model
    /// without a thinking mode ignores it. It needs llama.cpp b8287 or newer, where `--reasoning` first appears. A reasoning
    /// knob is refused in every spelling rather than outranked: llama.cpp keeps
    /// the last of two, and the order is the canonical sort's, not ours.
    ///
    /// The template flags are refused even where the recipe sets none: the ask
    /// is part of the timed prefill, so a plan's own template would time a
    /// different one under the same id. The host's envs reach the server and
    /// the helper too, and a CLI flag outranks an env only where the recipe
    /// sets one.
    pub fn server_args(
        &self,
        raw: &[String],
        envs: &[EnvForward],
        host_envs: &[EnvForward],
    ) -> anyhow::Result<Vec<String>> {
        let recipe = self.kind.name();
        // plan-types refuses the same raw flags and env forwards at load, by the
        // same rules; the host's and the inherited envs only this check sees.
        for flag in raw
            .iter()
            .filter_map(|token| LlamacppCliStockTools::flag_name(token))
        {
            if LlamacppCliStockTools::is_reasoning_flag(&flag) {
                return Err(AsrRefusalError::RawReasoning { recipe, flag }.into());
            }
            if LlamacppCliStockTools::RESERVED_TEMPLATE_DERIVED.contains(&flag.as_str()) {
                return Err(AsrRefusalError::RawPromptKnob { recipe, flag }.into());
            }
        }
        let named = |env: &EnvForward| {
            let forward: &str = env.as_ref();
            forward
                .split_once('=')
                .map_or(forward, |(name, _)| name)
                .to_owned()
        };
        let forwards = envs.iter().map(|env| ("env forward", named(env)));
        let host = host_envs
            .iter()
            .map(|env| ("host configuration's env", named(env)));
        // The server and the helper inherit pipette's own environment as well.
        let inherited = std::env::vars_os()
            .map(|(name, _)| ("worker's environment", name.to_string_lossy().into_owned()));
        refuse_env_knobs(recipe, forwards.chain(host).chain(inherited))?;
        let mut out = ServerOption::ReasoningOff.argv()?;
        for option in self.server_options {
            out.extend(option.argv()?);
        }
        Ok(out)
    }

    pub fn chat_template(&self) -> Option<&'static str> {
        self.server_options.iter().find_map(|option| match option {
            ServerOption::ChatTemplate(template) => Some(*template),
            ServerOption::ReasoningOff | ServerOption::ChatTemplateKwargs(_) => None,
        })
    }

    /// Refuses a server whose `/props` (`active`, asked only when the recipe
    /// serves a template) reports another template: a llama.cpp build whose
    /// jinja engine rejected the recipe's could fall back to its default and
    /// time another prompt under the recipe's name.
    pub fn check_served_template(
        &self,
        active: impl FnOnce() -> anyhow::Result<Option<String>>,
    ) -> anyhow::Result<()> {
        let Some(template) = self.chat_template() else {
            return Ok(());
        };
        // llama.cpp's jinja lexer drops one trailing newline from the source it
        // keeps, as jinja does by default, so `/props` reports it without one.
        let unterminated = |t: &str| t.strip_suffix('\n').unwrap_or(t).to_owned();
        if active()?.map(|active| unterminated(&active)) == Some(unterminated(template)) {
            return Ok(());
        }
        Err(AsrRefusalError::TemplateNotServed {
            recipe: self.kind.name(),
        }
        .into())
    }

    /// Refuses, before anything loads, a clip of whole encoder windows
    /// ([`GEMMA4_WINDOW_MS`]) at a rate other than 16 kHz: it can resample to
    /// just past the last edge (it did at 32 and 48 kHz, on every whole number
    /// of windows tried), so it is not left to the resampler's rounding. At
    /// 16 kHz the clip holds exactly its samples, and a tail of a whole
    /// millisecond (16 samples) encodes.
    pub fn check_duration(
        &self,
        duration_ms: u32,
        sample_rate_hz: u32,
    ) -> Result<(), AsrRefusalError> {
        match self.encoder_window_ms {
            Some(window_ms)
                if duration_ms.is_multiple_of(window_ms) && sample_rate_hz != wav::SAMPLE_RATE =>
            {
                Err(AsrRefusalError::ClipOnWindowEdge {
                    recipe: self.kind.name(),
                    duration_ms,
                    sample_rate_hz,
                    window_ms,
                })
            }
            _ => Ok(()),
        }
    }

    /// A clip past the documented limit still runs, its speed and memory a
    /// hypothetical scenario, and its row says so
    /// (`asr_beyond_model_audio_limit`): the model's transcription quality is
    /// not established there, and Gemma 4 E2B's and E4B's processors give no
    /// count past it.
    pub fn beyond_documented_limit(&self, duration_ms: u32) -> bool {
        let beyond = self
            .documented_max_audio_ms
            .is_some_and(|max| duration_ms > max);
        if beyond {
            tracing::warn!(
                recipe = self.kind.name(),
                duration_ms,
                limit_ms = self.documented_max_audio_ms,
                "the clip is longer than the model's documented audio limit; its speed and memory \
                 are measured, and the row is flagged"
            );
        }
        beyond
    }

    pub fn text_chars(&self) -> usize {
        self.system.map_or(0, str::len) + self.instruction.len()
    }

    /// The added text is the user's, so it joins the user turn before the
    /// recipe's ask, on the ask's side of the clip; the system turn is the
    /// model's mode switch and stays as written, and the ask stays next to the
    /// clip. The newline is the separator llama.cpp puts between two text
    /// parts (`common/chat.cpp`). One part, not two, so the server renders the
    /// same text with the clip and in the text-only request that confirms the
    /// audio count: llama.cpp joins the parts of a turn that holds media
    /// itself, and a template renders those of one that does not.
    pub fn user_text<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match (text.is_empty(), self.instruction.is_empty()) {
            (true, _) => self.instruction.into(),
            (false, true) => text.into(),
            (false, false) => format!("{text}\n{}", self.instruction).into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CtxInputs {
    /// From [`fallback_audio_tokens`].
    pub audio_bound: u32,
    /// `parameter_text_tokens`, exact: the text is built to it under the
    /// model's own tokenizer.
    pub text_tokens: u32,
    pub decode_tokens: u32,
    /// Bytes of system and instruction text: an upper bound on their tokens.
    pub text_chars: usize,
}

impl CtxInputs {
    /// The margin also covers the newline joining the text to the ask, and a
    /// token the tokenizer merges or splits where they meet.
    pub fn ctx_size(&self, calibrated: Option<u32>) -> u32 {
        calibrated
            .unwrap_or(self.audio_bound)
            .saturating_add(self.text_tokens)
            .saturating_add(u32::try_from(self.text_chars).unwrap_or(u32::MAX))
            .saturating_add(self.decode_tokens)
            .saturating_add(CTX_MARGIN)
    }
}

/// The shared seed passage, cut so the user text part ([`Recipe::user_text`])
/// is exactly `tokens` longer under the server's tokenizer (`count`) than a
/// cell without text sends.
///
/// The part is counted whole, not the text alone: the newline joining the text
/// to the instruction is a token of its own under some tokenizers (Gemma 4's)
/// and merges into a neighbour under others, so a text of exactly `tokens` on
/// its own can miss by one. `count` must add no special tokens: the part sits
/// inside the user turn, after whatever BOS the template writes, so a count
/// that adds a BOS builds it a token short. The server confirms the result
/// ([`confirm_text`]).
pub fn build_text(
    recipe: &Recipe,
    tokens: u32,
    mut count: impl FnMut(&str) -> anyhow::Result<usize>,
) -> anyhow::Result<String> {
    if tokens == 0 {
        return Ok(String::new());
    }
    let ask = recipe.user_text("");
    let without = if ask.is_empty() { 0 } else { count(&ask)? };
    pipette_ops::prompt_seed::build_prompt_text(tokens, |text| {
        Ok(count(&recipe.user_text(text))?.saturating_sub(without))
    })
    .with_context(|| format!("building the {tokens} tokens of text beside the clip"))
}

/// The server's prompt with the text must be exactly `text_tokens` longer than
/// without it, so `parameter_text_tokens` is all the cell's text adds.
pub fn confirm_text(
    text_tokens: u32,
    with_text: u32,
    without_text: u32,
) -> Result<(), AsrRefusalError> {
    if text_tokens == 0 || with_text.checked_sub(without_text) == Some(text_tokens) {
        return Ok(());
    }
    Err(AsrRefusalError::TextUnconfirmed {
        text_tokens,
        with_text,
        without_text,
    })
}

/// `prompt_of` sends a clip-free user turn and returns the prompt it
/// prefilled; `with_text` is the text-only request's.
pub fn confirm_served_text(
    recipe: &Recipe,
    text_tokens: u32,
    with_text: u32,
    prompt_of: impl FnOnce(&str) -> anyhow::Result<u32>,
) -> anyhow::Result<()> {
    if text_tokens == 0 {
        return Ok(());
    }
    let without_text = prompt_of(&recipe.user_text(""))?;
    Ok(confirm_text(text_tokens, with_text, without_text)?)
}

/// An upper bound on a clip's audio tokens, for a host with no helper.
///
/// The densest encoder in the suite emits 25 a second (Gemma, Qwen2.5-Omni),
/// and those that pad to 30 s windows emit the whole window (750 for
/// Qwen2.5-Omni on a build that does not trim it). Stock builds also add a
/// silent window at an exact multiple of 30 s (Whisper-family encoders,
/// upstream issue 27697), so the windows counted are `duration / 30 s + 1`,
/// not the ceiling. The larger bound wins, so no build's context is short; the
/// sparser encoders are over-sized by design, as this runs only without an
/// exact count.
pub fn fallback_audio_tokens(duration_ms: u32) -> u32 {
    const PER_SECOND: u32 = 25;
    const PER_WINDOW: u32 = 750;
    const WINDOW_MS: u32 = 30_000;
    let by_rate = duration_ms.div_ceil(1000).saturating_mul(PER_SECOND);
    let by_window = (duration_ms / WINDOW_MS + 1).saturating_mul(PER_WINDOW);
    by_rate.max(by_window)
}

/// The bound [`plausible`] holds every count to, plus the margin. Without a
/// context, llama.cpp gives the helper the model's training context
/// (131072 for Gemma 4), whose KV it allocates for a one-token prompt. Not in
/// the cache key: a context that holds the clip does not change its count.
fn helper_ctx_size(bound: u32) -> u32 {
    bound.saturating_add(CTX_MARGIN).max(HELPER_CTX_FLOOR)
}

/// A count with its cache row, so one the server does not confirm can be
/// dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calibrated {
    pub tokens: u32,
    markers: u32,
    cache: PathBuf,
    key: String,
}

impl Calibrated {
    /// `measured`, the prompt with the clip less the same prompt without it,
    /// is the audio and the markers and nothing else, so it must be exactly
    /// `tokens + markers`. Anything else, an undercount the margin hid
    /// included, drops the row.
    pub fn confirm(&self, measured: u32) -> Result<(), AsrRefusalError> {
        if self.tokens.checked_add(self.markers) == Some(measured) {
            return Ok(());
        }
        self.drop_row();
        Err(AsrRefusalError::CountUnconfirmed {
            calibrated: self.tokens,
            markers: self.markers,
            measured,
            cache: self.cache.clone(),
        })
    }

    /// Drops the row that sized a failed cell's context, so the next cell
    /// counts the clip again: a short context shows as the server refusing the
    /// prompt (HTTP 400) or stopping the decode at the context's end, which no
    /// check after the run sees. Dropping on any failure but a refusal costs one helper run
    /// when the row was right; keeping a wrong one fails every cell on its key.
    pub fn drop_on_failure(&self, err: anyhow::Error) -> anyhow::Error {
        // A refusal is the cell's, not the count's (an unconfirmed count has
        // dropped its row already).
        if err.downcast_ref::<AsrRefusalError>().is_some() {
            return err;
        }
        self.drop_row();
        err.context(format!(
            "the context was sized from the count of {} audio tokens in {}; its row is dropped, \
             so the next run counts the clip again",
            self.tokens,
            self.cache.display()
        ))
    }

    fn drop_row(&self) {
        mtmd::cache_evict(&self.cache, &self.key);
    }
}

/// The exact audio tokens the cell's clip becomes. `launch` is the cell's
/// helper, placed as its server is; this bounds its context. `None` if the
/// projector, cache or helper is unavailable, or the count is not
/// [`plausible`].
pub fn calibrate(
    recipe: &Recipe,
    duration: AudioDurationMillis,
    format: wav::ClipFormat,
    launch: &MtmdCommandBuilder,
) -> Option<Calibrated> {
    let bound = fallback_audio_tokens(duration.as_u32());
    let fingerprint = mtmd::mmproj_fingerprint(launch.mmproj())?;
    let cache = launch.runtime_dir()?.join(CACHE_FILE);
    // The helper's own build keys the count: a runtime directory whose binaries
    // were replaced must not reuse the counts its old build wrote.
    let build = helper_build(&launch.helper())?;
    let samples = wav::samples_in(duration, format.sample_rate_hz());
    let key = cache_key(
        &fingerprint,
        &build,
        format,
        samples,
        launch.forwarded_envs(),
    );

    // An impossible row is never believed: a stale or hand-edited count in the
    // millions would size, and allocate, a KV cache of that size before
    // anything measured it. It is measured again and the row replaced.
    let cached = mtmd::cache_lookup(&cache, &key).and_then(|t| plausible(bound, t));
    let tokens = match cached {
        Some(tokens) => tokens,
        None => {
            // The cell's own clip, so the helper resamples it as the server
            // will: a rate whose millisecond is not a whole number of samples
            // (44.1 kHz) counts the same samples it sends.
            let clip = write_clip(&wav::tone(duration, format).ok()?)?;
            let command = launch
                .clone()
                .ctx_size(Some(helper_ctx_size(bound)))
                .build_audio(&clip);
            let tokens = mtmd::run_counting_helper(
                command,
                &clip,
                AUDIO_BATCH_LINE,
                VL_CALIBRATION_TIMEOUT,
            )?;
            let tokens = plausible(bound, tokens)?;
            mtmd::cache_store(&cache, &key, tokens);
            tokens
        }
    };
    Some(Calibrated {
        tokens,
        markers: recipe.audio_markers,
        cache,
        key,
    })
}

/// Created exclusively under a random name, so no other user of the temp dir
/// can predict it or plant a link there; removed when the path drops.
fn write_clip(clip: &[u8]) -> Option<tempfile::TempPath> {
    let mut file = tempfile::Builder::new()
        .prefix("asrcal-")
        .suffix(".wav")
        .tempfile()
        .ok()?;
    file.write_all(clip).ok()?;
    file.flush().ok()?;
    // Closed before the helper opens it, which Windows needs.
    Some(file.into_temp_path())
}

/// A digest of the whole helper and of every `mtmd` library beside it or in
/// its package `lib/` ([`runtimes::package_lib_dir`]), where a dynamic build
/// keeps the code that counts. A patch rarely changes a binary's size
/// (segments are page-aligned) or its first pages, so nothing short of the
/// content tells two builds apart.
fn helper_build(helper: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    hash_file(&mut digest, helper)?;
    let bin_dir = helper.parent()?;
    let mut libraries: Vec<PathBuf> = Vec::new();
    for dir in std::iter::once(bin_dir.to_path_buf()).chain(runtimes::package_lib_dir(bin_dir)) {
        libraries.extend(
            std::fs::read_dir(dir)
                .ok()?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(is_mtmd_library)
                }),
        );
    }
    libraries.sort();
    for library in libraries {
        digest.update(library.file_name()?.to_string_lossy().as_bytes());
        hash_file(&mut digest, &library)?;
    }
    let digest: [u8; 32] = digest.finalize().into();
    Some(hex::encode(&digest[..8]))
}

/// A buffer at a time, so a large library is never held in memory whole
/// (`Sha256` is not `io::Write`).
fn hash_file(digest: &mut sha2::Sha256, path: &Path) -> Option<()> {
    use std::io::BufRead;

    use sha2::Digest;
    let mut file = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    loop {
        match file.fill_buf() {
            Ok([]) => return Some(()),
            Ok(chunk) => {
                digest.update(chunk);
                let n = chunk.len();
                file.consume(n);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
}

/// `libmtmd.so`, `libmtmd.so.0`, `libmtmd.dylib`, `libmtmd.0.dylib`, `mtmd.dll`.
fn is_mtmd_library(name: &str) -> bool {
    let mut parts = name.split('.');
    let stem = parts.next().unwrap_or_default();
    stem.strip_prefix("lib").unwrap_or(stem) == "mtmd"
        && parts.any(|ext| matches!(ext.to_ascii_lowercase().as_str(), "so" | "dylib" | "dll"))
}

fn plausible(bound: u32, tokens: u32) -> Option<u32> {
    if (1..=bound).contains(&tokens) {
        return Some(tokens);
    }
    tracing::warn!(
        audio_tokens = tokens,
        bound,
        "the calibrated audio count is not one a clip of this length can have; ignoring it"
    );
    None
}

/// The forwarded environment is hashed, so a value a cell forwards (a token,
/// say) never lands in the file in plaintext.
fn cache_key(
    fingerprint: &str,
    build: &str,
    format: wav::ClipFormat,
    samples: u64,
    envs: &[EnvForward],
) -> String {
    use sha2::{Digest, Sha256};
    let joined = envs
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join(" ");
    let env: [u8; 32] = Sha256::digest(joined.as_bytes()).into();
    format!(
        "{fingerprint}|build:{build}|audio:{}Hz:{}ch:{samples}|env:{}",
        format.sample_rate_hz(),
        format.channels(),
        hex::encode(&env[..8])
    )
}

/// The audio-only part of `prompt_tokens`.
///
/// Always measured: the prompt with the clip less the same prompt without it
/// (`text_only_prompt_tokens`), which is the clip's audio tokens and the
/// recipe's markers. A calibrated count is reported only when the server
/// confirms it ([`Calibrated::confirm`]). One it does not confirm sized a
/// context that was not exact, so the cell is refused; behind a pinned
/// context, which it did not size, the measured difference is reported
/// instead, as with no count. That difference includes the markers, so it is
/// the fallback, not the reference.
pub fn audio_tokens(
    calibrated: Option<&Calibrated>,
    ctx_pinned: bool,
    prompt_tokens: u32,
    text_only_prompt_tokens: impl FnOnce() -> anyhow::Result<u32>,
) -> anyhow::Result<(u32, AsrAudioTokensSource)> {
    let text_only = text_only_prompt_tokens()
        .context("the text-only request that measures the audio tokens failed")?;
    anyhow::ensure!(
        prompt_tokens > text_only,
        "the prompt took {prompt_tokens} tokens with its clip and {text_only} without: the clip \
         added none"
    );
    let measured = prompt_tokens - text_only;
    let (tokens, source) = match calibrated.map(|c| c.confirm(measured).map(|()| c.tokens)) {
        Some(Ok(tokens)) => (tokens, AsrAudioTokensSource::Calibrated),
        Some(Err(unconfirmed)) if !ctx_pinned => return Err(unconfirmed.into()),
        Some(Err(unconfirmed)) => {
            tracing::warn!(error = %unconfirmed, "reporting the measured audio tokens instead");
            (measured, AsrAudioTokensSource::Measured)
        }
        None => (measured, AsrAudioTokensSource::Measured),
    };
    tracing::info!(
        prompt_tokens,
        text_only_prompt_tokens = text_only,
        audio_tokens = tokens,
        audio_tokens_source = %source,
        "token counts"
    );
    Ok((tokens, source))
}

/// As the row's `client_extras` records it: a peak is the clip's own only when
/// `exact`, which a row says only when the server confirmed the count
/// ([`Calibrated::confirm`]). The upper bound ([`fallback_audio_tokens`])
/// holds KV for audio the clip lacks.
pub fn context_sizing(pinned: bool, calibrated: bool) -> AsrContextSizing {
    match (pinned, calibrated) {
        (true, _) => AsrContextSizing::Pinned,
        (false, true) => AsrContextSizing::Exact,
        (false, false) => AsrContextSizing::UpperBound,
    }
}

/// As [`crate::server::with_server_tail`].
pub fn with_server_tail(
    err: anyhow::Error,
    stopped: impl std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
) -> anyhow::Error {
    let cause = err.root_cause().to_string();
    err.context(stopped).context(cause)
}

/// By `(origin, name)`; the rule, in any case, is
/// [`LlamacppCliStockTools::asr_refuses_env`].
fn refuse_env_knobs(
    recipe: &'static str,
    names: impl IntoIterator<Item = (&'static str, String)>,
) -> Result<(), AsrRefusalError> {
    for (origin, name) in names {
        if LlamacppCliStockTools::asr_refuses_env(&name) {
            return Err(AsrRefusalError::EnvKnob {
                recipe,
                origin,
                name,
            });
        }
    }
    Ok(())
}

/// Refuses a `raw` slot count. The context is derived for one request, and
/// llama-server divides `--ctx-size` among its slots, so a second slot would
/// halve what the clip can use while the row still reports the whole.
pub fn refuse_raw_parallel(raw: &[String], label: &'static str) -> Result<(), AsrRefusalError> {
    match crate::server::raw_parallel(raw) {
        Some((flag, _)) => Err(AsrRefusalError::RawParallel { label, flag }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ResolvedGgufVisionModel;
    use crate::runtimes::test_llama_server;

    fn helper_launch(raw: &[&str]) -> MtmdCommandBuilder {
        let model = ResolvedGgufVisionModel {
            backbone: "model.gguf".into(),
            mmproj: "mmproj.gguf".into(),
        };
        MtmdCommandBuilder::new(test_llama_server(), &model.backbone, &model.mmproj)
            .number_gpu_layers(Some(99))
            .raw(self::raw(raw))
    }

    #[rstest::rstest]
    #[case::short_clip_at_the_floor(2_500, HELPER_CTX_FLOOR)]
    #[case::ten_minutes_by_the_bound(600_000, 15_750 + CTX_MARGIN)]
    fn the_helper_is_placed_and_bounded(#[case] duration_ms: u32, #[case] ctx: u32) {
        let command = helper_launch(&[
            "--device",
            "MTL0",
            "--temp",
            "0.7",
            "-ub",
            "64",
            "--image-max-tokens",
            "64",
        ])
        .ctx_size(Some(helper_ctx_size(fallback_audio_tokens(duration_ms))))
        .build_audio(Path::new("clip.wav"));
        let args: Vec<String> = command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let pair = |flag: &str, value: &str| args.windows(2).any(|w| w == [flag, value]);
        assert!(pair("--model", "model.gguf"), "{args:?}");
        assert!(pair("--mmproj", "mmproj.gguf"), "{args:?}");
        assert!(pair("--n-gpu-layers", "99"), "{args:?}");
        assert!(pair("--device", "MTL0"), "{args:?}");
        assert!(pair("--ctx-size", &ctx.to_string()), "{args:?}");
        for dropped in ["--temp", "-ub", "--image-max-tokens"] {
            assert!(!args.iter().any(|a| a == dropped), "{dropped}: {args:?}");
        }
        assert!(
            args.ends_with(&raw(&["--jinja", "--verbose", "--audio", "clip.wav"])),
            "{args:?}"
        );
        // `llama-mtmd-cli.exe` on Windows.
        assert_eq!(
            Path::new(command.get_program()).file_stem(),
            Some(std::ffi::OsStr::new("llama-mtmd-cli")),
            "{:?}",
            command.get_program()
        );
    }

    /// A byte past the first MiB, or a replaced `mtmd` library, is another
    /// build.
    #[test]
    fn the_helper_build_is_its_whole_content() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let helper = dir.path().join("llama-mtmd-cli");
        let mut bytes = vec![0_u8; (1 << 20) + 4096];
        std::fs::write(&helper, &bytes)?;
        let first = helper_build(&helper).context("a readable helper has a build")?;
        bytes[(1 << 20) + 100] = 1;
        std::fs::write(&helper, &bytes)?;
        let patched = helper_build(&helper).context("a readable helper has a build")?;
        assert_ne!(first, patched);
        std::fs::write(dir.path().join("libmtmd.dylib"), b"one")?;
        let with_library = helper_build(&helper).context("a build")?;
        assert_ne!(patched, with_library);
        std::fs::write(dir.path().join("libmtmd.dylib"), b"two")?;
        assert_ne!(with_library, helper_build(&helper).context("a build")?);
        // Another library is not the helper's.
        std::fs::write(dir.path().join("libggml.dylib"), b"x")?;
        std::fs::write(dir.path().join("libmtmd.dylib"), b"one")?;
        assert_eq!(with_library, helper_build(&helper).context("a build")?);
        assert_eq!(helper_build(&dir.path().join("absent")), None);
        Ok(())
    }

    #[test]
    fn the_helper_build_covers_the_package_lib() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir_all(dir.path().join("bin"))?;
        std::fs::create_dir_all(dir.path().join("lib"))?;
        let helper = dir.path().join("bin").join("llama-mtmd-cli");
        std::fs::write(&helper, b"helper")?;
        std::fs::write(dir.path().join("lib").join("libmtmd.so"), b"one")?;
        let first = helper_build(&helper).context("a build")?;
        std::fs::write(dir.path().join("lib").join("libmtmd.so"), b"two")?;
        assert_ne!(first, helper_build(&helper).context("a build")?);
        Ok(())
    }

    #[rstest::rstest]
    #[case::linux("libmtmd.so", true)]
    #[case::linux_versioned("libmtmd.so.0", true)]
    #[case::macos("libmtmd.dylib", true)]
    #[case::macos_versioned("libmtmd.0.dylib", true)]
    #[case::windows("mtmd.dll", true)]
    #[case::the_helper("llama-mtmd-cli", false)]
    #[case::another_library("libllama.dylib", false)]
    fn the_mtmd_library_is_recognized(#[case] name: &str, #[case] expected: bool) {
        assert_eq!(is_mtmd_library(name), expected);
    }

    #[test]
    fn the_clip_file_is_private_and_removed() -> anyhow::Result<()> {
        let first = write_clip(b"RIFF").context("a clip file")?;
        let second = write_clip(b"RIFF").context("a clip file")?;
        assert_ne!(first.to_path_buf(), second.to_path_buf());
        assert_eq!(std::fs::read(&first)?, b"RIFF");
        let path = first.to_path_buf();
        drop(first);
        assert!(!path.exists());
        Ok(())
    }

    /// Gemma 4's model card ASR prompt with `{LANGUAGE}` = English, spelled out
    /// here so the recipe is checked against the card, not against itself.
    const GOOGLE_ASR_PROMPT: &str = "Transcribe the following speech segment in English into \
        English text.\n\nFollow these specific instructions for formatting the answer:\n* Only \
        output the transcription, with no newlines.\n* When transcribing numbers, write the \
        digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.";

    #[rstest::rstest]
    #[case::lfm2("lfm2a", false, "lfm2-audio", Some("Perform ASR."), "")]
    #[case::lfm3("lfm3a", false, "lfm3-audio", Some("Perform ASR in English."), "")]
    #[case::omni(
        "qwen2.5o",
        true,
        "qwen2.5-omni",
        Some("You are a speech recognition model."),
        "Transcribe the English audio into text without any punctuation marks."
    )]
    #[case::omni_audio(
        "qwen2.5oa",
        false,
        "qwen2.5-omni",
        Some("You are a speech recognition model."),
        "Transcribe the English audio into text without any punctuation marks."
    )]
    #[case::voxtral("voxtral", false, "voxtral", None, "")]
    #[case::gemma4("gemma4a", true, "gemma4", None, GOOGLE_ASR_PROMPT)]
    #[case::gemma4_unified("gemma4ua", true, "gemma4", None, GOOGLE_ASR_PROMPT)]
    #[case::qwen3_asr("qwen3a", false, "qwen3-asr", None, "")]
    #[case::qwen3_omni(
        "qwen3a",
        true,
        "qwen3-omni",
        None,
        "Transcribe the English audio into text."
    )]
    #[case::minicpm_o(
        "minicpmoa",
        false,
        "minicpm-o",
        None,
        "Please listen to the audio carefully and transcribe it with high precision.\n"
    )]
    fn the_projector_picks_the_recipe(
        #[case] projector: &str,
        #[case] with_vision: bool,
        #[case] name: &str,
        #[case] system: Option<&str>,
        #[case] instruction: &str,
    ) -> anyhow::Result<()> {
        let recipe = Recipe::for_encoder(&AudioEncoder::from_name(projector, with_vision))?;
        assert_eq!(recipe.kind.name(), name);
        assert_eq!(recipe.system, system);
        assert_eq!(recipe.instruction, instruction);
        Ok(())
    }

    #[test]
    fn qwen3_omni_asks_after_the_clip() -> anyhow::Result<()> {
        let omni = Recipe::for_encoder(&AudioEncoder::Qwen3Omni)?;
        assert_eq!(omni.order, ContentOrder::ClipThenInstruction);
        assert_eq!(
            omni.server_args(&[], &[], &[])?,
            raw(&["--reasoning", "off"])
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::unknown_family("ultravox")]
    #[case::qwen2_audio("qwen2a")]
    #[case::unnamed("")]
    fn an_encoder_without_a_recipe_is_refused(#[case] projector: &str) {
        let got = Recipe::for_encoder(&AudioEncoder::from_name(projector, false));
        assert!(
            matches!(&got, Err(AsrRefusalError::NoRecipe { name }) if name == projector),
            "{got:?}"
        );
    }

    #[test]
    fn only_a_projector_with_an_audio_encoder_is_asked() -> anyhow::Result<()> {
        let image_only = gguf_meta::ProjectorMetadata {
            has_audio_encoder: false,
            has_vision_encoder: true,
            audio_projector_type: None,
            projector_type: Some("lfm2".to_owned()),
            openbmb_minicpmo_audio: false,
        };
        assert!(matches!(
            AudioEncoder::of(&image_only),
            Err(AsrRefusalError::NoAudioEncoder)
        ));
        let unnamed = gguf_meta::ProjectorMetadata {
            has_audio_encoder: true,
            ..Default::default()
        };
        assert_eq!(
            AudioEncoder::of(&unnamed)?,
            AudioEncoder::Other(String::new())
        );
        let omni = gguf_meta::ProjectorMetadata {
            has_audio_encoder: true,
            has_vision_encoder: true,
            audio_projector_type: Some("qwen2.5o".to_owned()),
            projector_type: Some("qwen2.5vl".to_owned()),
            openbmb_minicpmo_audio: false,
        };
        assert_eq!(AudioEncoder::of(&omni)?, AudioEncoder::Qwen25Omni);
        let qwen3 = |has_vision_encoder| gguf_meta::ProjectorMetadata {
            has_audio_encoder: true,
            has_vision_encoder,
            audio_projector_type: Some("qwen3a".to_owned()),
            projector_type: None,
            openbmb_minicpmo_audio: false,
        };
        assert_eq!(AudioEncoder::of(&qwen3(true))?, AudioEncoder::Qwen3Omni);
        assert_eq!(AudioEncoder::of(&qwen3(false))?, AudioEncoder::Qwen3Asr);
        Ok(())
    }

    #[rstest::rstest]
    #[case::openbmb_audio_gguf(gguf_meta::fixture::openbmb_audio(
        true,
        &["encoder.conv1.weight", "audio_projector.linear2.weight"],
    ))]
    #[case::converted_projector(gguf_meta::fixture::gguf(&[
        ("general.architecture", gguf_meta::fixture::TYPE_STRING, gguf_meta::fixture::string("clip")),
        ("clip.has_audio_encoder", gguf_meta::fixture::TYPE_BOOL, vec![1]),
        ("clip.audio.projector_type", gguf_meta::fixture::TYPE_STRING, gguf_meta::fixture::string("minicpmoa")),
    ]))]
    fn openbmbs_minicpmo_audio_gguf_runs_minicpmos_recipe(
        #[case] file: Vec<u8>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("MiniCPM-o-4_5-audio-F16.gguf");
        std::fs::write(&path, file)?;
        let encoder = AudioEncoder::of(&gguf_meta::projector_metadata(&path)?)?;
        assert_eq!(encoder, AudioEncoder::MiniCpmO);
        assert_eq!(Recipe::for_encoder(&encoder)?.kind, RecipeKind::MiniCpmO);
        Ok(())
    }

    fn raw(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|&t| t.to_owned()).collect()
    }

    fn recipe(projector: &str) -> anyhow::Result<Recipe> {
        Ok(Recipe::for_encoder(&AudioEncoder::from_name(
            projector, false,
        ))?)
    }

    #[rstest::rstest]
    #[case::voxtral(
        "voxtral",
        &[],
        &[
            "--jinja",
            "--chat-template",
            VOXTRAL_CHAT_TEMPLATE,
            "--chat-template-kwargs",
            r#"{"transcription":true,"language":"en"}"#,
        ]
    )]
    #[case::gemma_adds_only_thinking_off("gemma4a", &[], &[])]
    #[case::unrelated_raw_flag_changes_nothing("gemma4a", &["--threads", "4"], &[])]
    #[case::lfm3("lfm3a", &[], &["--jinja", "--chat-template", LFM3_ASR_TEMPLATE])]
    #[case::lfm_adds_none("lfm2a", &[], &[])]
    #[case::omni_adds_none("qwen2.5o", &[], &[])]
    #[case::minicpm_o(
        "minicpmoa",
        &[],
        &["--chat-template-kwargs", r#"{"use_tts_template":true}"#]
    )]
    #[case::qwen3_asr(
        "qwen3a",
        &[],
        &[
            "--jinja",
            "--chat-template",
            QWEN3_ASR_TEMPLATE,
            "--chat-template-kwargs",
            r#"{"language":"English"}"#,
        ]
    )]
    fn a_recipe_adds_its_flags(
        #[case] projector: &str,
        #[case] plan_raw: &[&str],
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let mut want = raw(&["--reasoning", "off"]);
        want.extend(raw(expected));
        assert_eq!(
            recipe(projector)?.server_args(&raw(plan_raw), &[], &[])?,
            want
        );
        Ok(())
    }

    fn refusal(got: anyhow::Result<Vec<String>>) -> Option<AsrRefusalError> {
        got.err()?.downcast::<AsrRefusalError>().ok()
    }

    /// Every spelling llama.cpp reads (`common/arg.cpp`).
    #[rstest::rstest]
    fn a_raw_reasoning_flag_is_refused(
        #[values("gemma4a", "minicpmoa", "lfm3a", "voxtral", "qwen3a")] projector: &str,
        #[values(
            &["--reasoning", "on"][..],
            &["--reasoning=auto"][..],
            &["-rea", "on"][..],
            &["-rea=on"][..],
            &["--reasoning-budget", "-1"][..],
            &["--reasoning_budget=-1"][..],
            &["--reasoning-effort", "high"][..],
            &["--reasoning-format", "none"][..],
            &["--no-reasoning-preserve"][..],
            &["--threads", "4", "-rea", "on"][..],
        )]
        plan_raw: &[&str],
    ) -> anyhow::Result<()> {
        let got = recipe(projector)?.server_args(&raw(plan_raw), &[], &[]);
        assert!(
            matches!(refusal(got), Some(AsrRefusalError::RawReasoning { .. })),
            "{plan_raw:?}"
        );
        Ok(())
    }

    #[rstest::rstest]
    fn a_raw_template_flag_is_refused(
        #[values("voxtral", "lfm3a", "minicpmoa", "gemma4a")] projector: &str,
        #[values(
            &["--jinja"][..],
            &["--no-jinja"][..],
            &["--chat-template", "chatml"][..],
            &["--chat-template-file", "t.jinja"][..],
            &["--chat_template_file=t.jinja"][..],
            &["--chat-template-kwargs", r#"{"enable_thinking":true}"#][..],
        )]
        plan_raw: &[&str],
    ) -> anyhow::Result<()> {
        let got = recipe(projector)?.server_args(&raw(plan_raw), &[], &[]);
        assert!(
            matches!(refusal(got), Some(AsrRefusalError::RawPromptKnob { .. })),
            "{plan_raw:?}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::reasoning("LLAMA_ARG_REASONING=on", true)]
    #[case::budget("LLAMA_ARG_THINK_BUDGET=-1", true)]
    #[case::format("LLAMA_ARG_THINK=none", true)]
    #[case::jinja("LLAMA_ARG_JINJA=0", true)]
    #[case::template("LLAMA_ARG_CHAT_TEMPLATE=chatml", true)]
    #[case::kwargs(r#"LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}"#, true)]
    #[case::bare_name("LLAMA_ARG_CHAT_TEMPLATE_FILE", true)]
    #[case::lower_case("llama_arg_chat_template=chatml", true)]
    #[case::unrelated("GGML_METAL_NDEBUG=1", false)]
    fn a_template_or_reasoning_env_forward_is_refused(
        #[case] forward: &str,
        #[case] refused: bool,
    ) -> anyhow::Result<()> {
        let envs = [EnvForward::try_new(forward.to_owned())?];
        let forwarded = recipe("gemma4a")?.server_args(&[], &envs, &[]);
        assert_eq!(
            matches!(
                refusal(forwarded),
                Some(AsrRefusalError::EnvKnob {
                    origin: "env forward",
                    ..
                })
            ),
            refused
        );
        let hosted = recipe("gemma4a")?.server_args(&[], &[], &envs);
        assert_eq!(
            matches!(
                refusal(hosted),
                Some(AsrRefusalError::EnvKnob {
                    origin: "host configuration's env",
                    ..
                })
            ),
            refused
        );
        Ok(())
    }

    #[test]
    fn an_inherited_template_env_is_refused() {
        let got = refuse_env_knobs(
            "gemma4",
            [
                ("worker's environment", "PATH".to_owned()),
                ("worker's environment", "LLAMA_ARG_CHAT_TEMPLATE".to_owned()),
            ],
        );
        assert!(matches!(
            got,
            Err(AsrRefusalError::EnvKnob {
                origin: "worker's environment",
                ..
            })
        ));
    }

    /// Rendered by minijinja; the same render was checked on the lfm3
    /// llama-server's `/apply-template`, whose jinja engine is llama.cpp's own.
    #[rstest::rstest]
    #[case::as_parts(serde_json::json!([{ "type": "text", "text": "<__media__>" }]))]
    #[case::as_a_string(serde_json::json!("<__media__>"))]
    fn the_lfm3_template_renders_the_native_prompt(
        #[case] user: serde_json::Value,
    ) -> anyhow::Result<()> {
        let mut env = minijinja::Environment::new();
        env.add_template("lfm3", LFM3_ASR_TEMPLATE)?;
        let rendered =
            env.get_template("lfm3")?
                .render(minijinja::value::Serde(serde_json::json!({
                    "bos_token": "<|startoftext|>",
                    "add_generation_prompt": true,
                    "messages": [
                        { "role": "system", "content": LFM3_AUDIO.system },
                        { "role": "user", "content": user },
                    ],
                })))?;
        assert_eq!(
            rendered,
            "<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n\
             <|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    /// What `Qwen3ASRProcessor.apply_transcription_request(audio,
    /// language="English")` renders with the `Qwen/Qwen3-ASR-1.7B-hf` template,
    /// spelled out, with the audio placeholder as the server's marker.
    #[rstest::rstest]
    #[case::with_the_clip(
        serde_json::json!([{ "type": "text", "text": "<__media__>" }]),
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n\
         <|im_start|>assistant\nlanguage English<asr_text>"
    )]
    #[case::as_a_string(
        serde_json::json!("<__media__>"),
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n\
         <|im_start|>assistant\nlanguage English<asr_text>"
    )]
    #[case::text_only(
        serde_json::json!(""),
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n\
         <|im_start|>assistant\nlanguage English<asr_text>"
    )]
    fn the_qwen3_asr_template_renders_the_processors_prompt(
        #[case] user: serde_json::Value,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let language = QWEN3_ASR
            .server_options
            .iter()
            .find_map(|option| match option {
                ServerOption::ChatTemplateKwargs(kwargs) => kwargs.language,
                ServerOption::ReasoningOff | ServerOption::ChatTemplate(_) => None,
            })
            .context("Qwen3-ASR's recipe names the language")?;
        let mut env = minijinja::Environment::new();
        env.add_template("qwen3-asr", QWEN3_ASR_TEMPLATE)?;
        let rendered = env
            .get_template("qwen3-asr")?
            .render(minijinja::value::Serde(serde_json::json!({
                "add_generation_prompt": true,
                "language": language,
                "messages": [{ "role": "user", "content": user }],
            })))?;
        assert_eq!(rendered, expected);
        Ok(())
    }

    #[rstest::rstest]
    #[case::served("voxtral", Some(VOXTRAL_CHAT_TEMPLATE), true)]
    // Voxtral's file ends in a newline, which `/props` drops.
    #[case::served_unterminated(
        "voxtral",
        VOXTRAL_CHAT_TEMPLATE.strip_suffix('\n'),
        true
    )]
    #[case::cut_short(
        "voxtral",
        VOXTRAL_CHAT_TEMPLATE.get(..VOXTRAL_CHAT_TEMPLATE.len() - 2),
        false
    )]
    #[case::fallen_back("voxtral", Some("{{ chatml }}"), false)]
    #[case::not_reported("lfm3a", None, false)]
    #[case::lfm3_served("lfm3a", Some(LFM3_ASR_TEMPLATE), true)]
    #[case::qwen3_asr_on_its_gguf_template("qwen3a", Some("{% for m in messages %}"), false)]
    #[case::no_template_of_its_own("gemma4a", Some("anything"), true)]
    fn the_server_must_render_the_recipes_template(
        #[case] projector: &str,
        #[case] active: Option<&str>,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let asked = std::cell::Cell::new(false);
        let recipe = recipe(projector)?;
        let got = recipe.check_served_template(|| {
            asked.set(true);
            Ok(active.map(str::to_owned))
        });
        assert_eq!(got.is_ok(), accepted, "{got:?}");
        assert!(
            got.as_ref().is_ok()
                || matches!(
                    got.as_ref()
                        .err()
                        .and_then(|e| e.downcast_ref::<AsrRefusalError>()),
                    Some(AsrRefusalError::TemplateNotServed { .. })
                ),
            "{got:?}"
        );
        // A server is asked only when the recipe has a template to hold it to.
        assert_eq!(asked.get(), recipe.chat_template().is_some());
        Ok(())
    }

    /// A stray edit would change every Voxtral prompt unreviewed.
    #[test]
    fn the_voxtral_template_is_the_reviewed_copy() {
        use sha2::Digest;
        assert_eq!(
            hex::encode(sha2::Sha256::digest(VOXTRAL_CHAT_TEMPLATE.as_bytes())),
            "3d7b2f9d649544f6fe3223620a84d91eb64bd9ff4a7467b0edc843c20b90f5de"
        );
    }

    #[test]
    fn the_voxtral_recipe_renders_the_transcription_turn() -> anyhow::Result<()> {
        let kwargs = VOXTRAL
            .server_options
            .iter()
            .find_map(|option| match option {
                ServerOption::ChatTemplateKwargs(kwargs) => Some(*kwargs),
                ServerOption::ReasoningOff | ServerOption::ChatTemplate(_) => None,
            })
            .context("Voxtral's recipe passes template kwargs")?;
        let mut env = minijinja::Environment::new();
        env.add_template("voxtral", VOXTRAL_CHAT_TEMPLATE)?;
        let rendered = env
            .get_template("voxtral")?
            .render(minijinja::value::Serde(serde_json::json!({
                "bos_token": "<s>",
                "transcription": kwargs.transcription,
                "language": kwargs.language,
                "messages": [
                    { "role": "user", "content": [{ "type": "text", "text": "<__media__>" }] },
                ],
            })))?;
        assert_eq!(rendered, "<s>[INST]<__media__>[/INST]lang:en[TRANSCRIBE]");
        Ok(())
    }

    /// As mistral-common's `encode_transcription` does for auto-detection.
    #[test]
    fn the_voxtral_template_auto_detects_without_a_language() -> anyhow::Result<()> {
        let mut env = minijinja::Environment::new();
        env.add_template("voxtral", VOXTRAL_CHAT_TEMPLATE)?;
        let rendered = env
            .get_template("voxtral")?
            .render(minijinja::value::Serde(serde_json::json!({
                "bos_token": "<s>",
                "transcription": true,
                "messages": [
                    { "role": "user", "content": [{ "type": "text", "text": "<__media__>" }] },
                ],
            })))?;
        assert_eq!(rendered, "<s>[INST]<__media__>[/INST][TRANSCRIBE]");
        Ok(())
    }

    #[rstest::rstest]
    #[case::gemma_short("gemma4a", 2_500, 16_000, true)]
    #[case::gemma_ladder_top("gemma4a", 29_900, 16_000, true)]
    #[case::gemma_exactly_30_s("gemma4a", 30_000, 16_000, true)]
    #[case::gemma_a_millisecond_over("gemma4a", 30_001, 16_000, true)]
    #[case::gemma_the_long_rung("gemma4a", 119_900, 16_000, true)]
    #[case::gemma_four_windows("gemma4a", 120_000, 16_000, true)]
    #[case::gemma_30_s_at_48_khz("gemma4a", 30_000, 48_000, false)]
    #[case::gemma_60_s_at_48_khz("gemma4a", 60_000, 48_000, false)]
    #[case::gemma_four_windows_at_32_khz("gemma4a", 120_000, 32_000, false)]
    #[case::gemma_the_long_rung_at_32_khz("gemma4a", 119_900, 32_000, true)]
    #[case::gemma_four_windows_at_44_1_khz("gemma4a", 120_000, 44_100, false)]
    #[case::gemma_just_under_at_48_khz("gemma4a", 29_999, 48_000, true)]
    #[case::gemma_off_the_edge_at_48_khz("gemma4a", 120_001, 48_000, true)]
    #[case::gemma_unified_30_s_at_48_khz("gemma4ua", 30_000, 48_000, true)]
    #[case::gemma_unified_on_the_edge("gemma4ua", 60_000, 48_000, true)]
    #[case::gemma_unified_four_windows_at_44_1_khz("gemma4ua", 120_000, 44_100, true)]
    #[case::gemma_unified_long("gemma4ua", 600_000, 16_000, true)]
    #[case::voxtral_long("voxtral", 120_000, 16_000, true)]
    #[case::minicpm_o_long("minicpmoa", 120_000, 48_000, true)]
    fn a_clip_on_a_window_edge_at_another_rate_is_refused(
        #[case] projector: &str,
        #[case] duration_ms: u32,
        #[case] rate: u32,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        match recipe(projector)?.check_duration(duration_ms, rate) {
            Ok(()) => assert!(accepted),
            Err(err) => assert!(
                !accepted && matches!(err, AsrRefusalError::ClipOnWindowEdge { .. }),
                "{err}"
            ),
        }
        Ok(())
    }

    #[rstest::rstest]
    #[case::gemma_ladder_top("gemma4a", 29_900, false)]
    #[case::gemma_at_the_limit("gemma4a", 30_000, false)]
    #[case::gemma_a_millisecond_over("gemma4a", 30_001, true)]
    #[case::gemma_the_long_rung("gemma4a", 119_900, true)]
    #[case::gemma_unified_long("gemma4ua", 119_900, true)]
    #[case::lfm2_long("lfm2a", 119_900, false)]
    #[case::voxtral_long("voxtral", 119_900, false)]
    #[case::qwen25_omni_long("qwen2.5o", 119_900, false)]
    #[case::qwen3_asr_long("qwen3a", 119_900, false)]
    #[case::minicpm_o_long("minicpmoa", 600_000, false)]
    fn a_clip_past_the_documented_limit_is_flagged(
        #[case] projector: &str,
        #[case] duration_ms: u32,
        #[case] beyond: bool,
    ) -> anyhow::Result<()> {
        assert_eq!(
            recipe(projector)?.beyond_documented_limit(duration_ms),
            beyond
        );
        Ok(())
    }

    /// Stock Qwen2.5-Omni pads to 750 a window and adds one at exact multiples
    /// of 30 s.
    #[rstest::rstest]
    #[case::one_ms(1, 750)]
    #[case::short(2_500, 750)]
    #[case::ten_seconds(9_400, 750)]
    #[case::just_under_a_window(29_900, 750)]
    #[case::exactly_a_window(30_000, 1_500)]
    #[case::just_over(30_100, 1_500)]
    #[case::just_under_two(59_999, 1_500)]
    #[case::exactly_two(60_000, 2_250)]
    #[case::the_long_rung(119_900, 3_000)]
    #[case::exactly_four(120_000, 3_750)]
    #[case::the_longest_clip(600_000, 15_750)]
    fn the_fallback_bounds_every_encoder(#[case] duration_ms: u32, #[case] tokens: u32) {
        assert_eq!(fallback_audio_tokens(duration_ms), tokens);
    }

    /// Stock b11256 counts measured for the clips the suite uses.
    #[rstest::rstest]
    #[case::omni_2_5s(2_500, 750)]
    #[case::omni_35_5s(35_494, 1_500)]
    #[case::omni_30s(30_000, 1_500)]
    #[case::omni_60s(60_000, 2_250)]
    #[case::gemma_29_9s(29_900, 748)]
    #[case::gemma_60s(60_000, 1_500)]
    #[case::qwen3_60s(60_000, 793)]
    // At exactly 120 s, stock Qwen2.5-Omni pads a fifth window and Qwen3-ASR a
    // whole extra second.
    #[case::omni_119_9s(119_900, 3_000)]
    #[case::gemma_119_9s(119_900, 2_998)]
    #[case::qwen3_119_9s(119_900, 1_560)]
    #[case::omni_120s(120_000, 3_750)]
    #[case::gemma_120s(120_000, 3_000)]
    #[case::qwen3_120s(120_000, 1_573)]
    fn the_fallback_is_never_below_a_measured_count(
        #[case] duration_ms: u32,
        #[case] measured: u32,
    ) {
        assert!(
            fallback_audio_tokens(duration_ms) >= measured,
            "{duration_ms} ms: {} < {measured}",
            fallback_audio_tokens(duration_ms)
        );
    }

    #[rstest::rstest]
    #[case::no_text(0)]
    #[case::text(1024)]
    fn the_context_covers_audio_text_ask_decode_and_margin(#[case] text_tokens: u32) {
        let inputs = CtxInputs {
            audio_bound: 750,
            text_tokens,
            decode_tokens: 128,
            text_chars: 100,
        };
        assert_eq!(
            inputs.ctx_size(Some(235)),
            235 + text_tokens + 100 + 128 + CTX_MARGIN
        );
        assert_eq!(
            inputs.ctx_size(None),
            750 + text_tokens + 100 + 128 + CTX_MARGIN
        );
    }

    #[test]
    fn the_context_does_not_overflow() {
        let inputs = CtxInputs {
            audio_bound: u32::MAX,
            text_tokens: u32::MAX,
            decode_tokens: u32::MAX,
            text_chars: usize::MAX,
        };
        assert_eq!(inputs.ctx_size(None), u32::MAX);
    }

    /// A fake tokenizer: a token a word, and a BOS with special tokens.
    fn word_tokens(text: &str, with_bos: bool) -> usize {
        text.split_whitespace().count() + usize::from(with_bos && !text.is_empty())
    }

    /// A newline between two words as whitespace, as a token of its own
    /// (Gemma 4's), or as glue that makes the words on either side one token.
    fn joined_tokens(text: &str, newline: &str) -> usize {
        let words = word_tokens(text, false);
        let between_words = text
            .as_bytes()
            .windows(3)
            .filter(|w| matches!(w, [a, b'\n', c] if !a.is_ascii_whitespace() && !c.is_ascii_whitespace()))
            .count();
        match newline {
            "own" => words + text.matches('\n').count(),
            "merge" => words.saturating_sub(between_words),
            _ => words,
        }
    }

    /// Whatever the tokenizer does with the newline that joins the text to the
    /// instruction.
    #[rstest::rstest]
    fn the_text_is_exactly_the_cells_tokens(
        #[values("lfm2a", "gemma4a", "minicpmoa")] projector: &str,
        #[values("word", "own", "merge")] newline: &str,
        #[values(0, 2, 1024)] tokens: u32,
    ) -> anyhow::Result<()> {
        let recipe = recipe(projector)?;
        let mut asked = 0;
        let text = build_text(&recipe, tokens, |part| {
            asked += 1;
            Ok(joined_tokens(part, newline))
        })?;
        let added = joined_tokens(&recipe.user_text(&text), newline)
            - joined_tokens(&recipe.user_text(""), newline);
        assert_eq!(added, usize::try_from(tokens)?);
        assert_eq!(asked == 0, tokens == 0);
        assert_eq!(text.is_empty(), tokens == 0);
        Ok(())
    }

    /// One token cannot be added where the joining newline is a token of its
    /// own: the builder fails rather than send another count.
    #[test]
    fn a_text_the_tokenizer_cannot_make_is_refused() -> anyhow::Result<()> {
        let got = build_text(&recipe("gemma4a")?, 1, |part| {
            Ok(joined_tokens(part, "own"))
        });
        assert!(got.is_err(), "{got:?}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::exact(32, 140, 108, true)]
    #[case::a_token_over(32, 141, 108, false)]
    #[case::a_token_under(32, 139, 108, false)]
    #[case::shorter_with_text(32, 100, 108, false)]
    #[case::no_text(0, 108, 108, true)]
    fn the_text_is_held_to_the_server(
        #[case] text_tokens: u32,
        #[case] with_text: u32,
        #[case] without_text: u32,
        #[case] confirmed: bool,
    ) {
        let got = confirm_text(text_tokens, with_text, without_text);
        assert_eq!(got.is_ok(), confirmed, "{got:?}");
    }

    #[rstest::rstest]
    #[case::lfm2("lfm2a", false, ContentOrder::InstructionThenClip)]
    #[case::lfm3("lfm3a", false, ContentOrder::InstructionThenClip)]
    #[case::qwen25_omni("qwen2.5o", true, ContentOrder::ClipThenInstruction)]
    #[case::gemma4("gemma4a", true, ContentOrder::InstructionThenClip)]
    #[case::minicpm_o("minicpmoa", false, ContentOrder::InstructionThenClip)]
    #[case::qwen3_asr("qwen3a", false, ContentOrder::InstructionThenClip)]
    #[case::qwen3_omni("qwen3a", true, ContentOrder::ClipThenInstruction)]
    #[case::voxtral("voxtral", false, ContentOrder::ClipThenInstruction)]
    fn the_cells_text_joins_the_ask(
        #[case] projector: &str,
        #[case] with_vision: bool,
        #[case] order: ContentOrder,
    ) -> anyhow::Result<()> {
        let recipe = Recipe::for_encoder(&AudioEncoder::from_name(projector, with_vision))?;
        assert_eq!(recipe.order, order);
        assert_eq!(recipe.user_text(""), recipe.instruction);
        let joined = recipe.user_text("the cell's text");
        if recipe.instruction.is_empty() {
            assert_eq!(joined, "the cell's text");
        } else {
            assert_eq!(joined, format!("the cell's text\n{}", recipe.instruction));
        }
        Ok(())
    }

    fn calibrated(tokens: u32, markers: u32, cache: &Path) -> Calibrated {
        Calibrated {
            tokens,
            markers,
            cache: cache.to_path_buf(),
            key: "row".to_owned(),
        }
    }

    /// Every case measured a 300-token prompt, and the recipe adds 2 markers.
    #[rstest::rstest]
    #[case::confirmed(Some(288), false, Ok(10), Ok((288, AsrAudioTokensSource::Calibrated)))]
    #[case::two_under(Some(286), false, Ok(10), Err("make 288"))]
    #[case::one_over(Some(289), false, Ok(10), Err("make 291"))]
    #[case::a_plausible_undercount(Some(18), false, Ok(278), Err("prefilled 22 more tokens"))]
    #[case::one_below_the_prompt(Some(299), false, Ok(10), Err("make 301"))]
    #[case::past_the_prompt(Some(400), false, Ok(100), Err("make 402"))]
    #[case::unconfirmed_behind_a_pin(Some(286), true, Ok(10), Ok((290, AsrAudioTokensSource::Measured)))]
    #[case::uncalibrated(None, false, Ok(100), Ok((200, AsrAudioTokensSource::Measured)))]
    #[case::failed_probe(Some(288), false, Err("connection refused"), Err("connection refused"))]
    #[case::nothing_added(None, false, Ok(300), Err("the clip added none"))]
    fn audio_tokens_are_confirmed_or_measured(
        #[case] count: Option<u32>,
        #[case] ctx_pinned: bool,
        #[case] text_only: Result<u32, &str>,
        #[case] expected: Result<(u32, AsrAudioTokensSource), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let cache = dir.path().join(CACHE_FILE);
        mtmd::cache_store(&cache, "row", 1);
        let count = count.map(|tokens| calibrated(tokens, 2, &cache));
        let probed = std::cell::Cell::new(false);
        let got = audio_tokens(count.as_ref(), ctx_pinned, 300, || {
            probed.set(true);
            text_only.map_err(|e| anyhow::anyhow!("{e}"))
        });
        let confirmed = matches!(got, Ok((_, AsrAudioTokensSource::Calibrated)));
        match (got, expected) {
            (Ok(got), Ok(expected)) => assert_eq!(got, expected),
            (Err(err), Err(expected)) => {
                assert!(format!("{err:#}").contains(expected), "{err:#}");
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        // The probe always runs.
        assert!(probed.get());
        // An unconfirmed count's row is dropped; a confirmed one's kept, and
        // a probe that failed confirmed nothing, so it leaves the row alone.
        let kept = mtmd::cache_lookup(&cache, "row").is_some();
        assert_eq!(
            kept,
            count.is_none() || confirmed || text_only.is_err(),
            "{count:?}"
        );
        Ok(())
    }

    #[test]
    fn a_failed_run_drops_the_row_that_sized_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let cache = dir.path().join(CACHE_FILE);
        mtmd::cache_store(&cache, "row", 18);
        let count = calibrated(18, 2, &cache);
        let err = count.drop_on_failure(anyhow::anyhow!(
            "request (205 tokens) exceeds the available context size (109 tokens)"
        ));
        assert!(
            format!("{err:#}").contains("its row is dropped, so the next run counts the clip"),
            "{err:#}"
        );
        assert_eq!(mtmd::cache_lookup(&cache, "row"), None);

        mtmd::cache_store(&cache, "row", 18);
        let refused =
            count.drop_on_failure(AsrRefusalError::TemplateNotServed { recipe: "voxtral" }.into());
        assert!(!format!("{refused:#}").contains("its row is dropped"));
        assert_eq!(mtmd::cache_lookup(&cache, "row"), Some(18));

        let unconfirmed = count
            .confirm(25)
            .err()
            .context("25 is not 18 and 2 markers")?;
        assert_eq!(mtmd::cache_lookup(&cache, "row"), None);
        let err = count.drop_on_failure(unconfirmed.into());
        assert!(
            !format!("{err:#}").contains("its row is dropped"),
            "{err:#}"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::exact(235, Some(235))]
    #[case::at_the_bound(750, Some(750))]
    #[case::past_the_bound(751, None)]
    #[case::stale_millions(999_999, None)]
    #[case::zero(0, None)]
    fn an_impossible_count_is_not_believed(#[case] tokens: u32, #[case] expected: Option<u32>) {
        assert_eq!(plausible(750, tokens), expected);
    }

    /// Pinned so a change to one is a reviewed change to its rows' `exact`.
    #[rstest::rstest]
    #[case::lfm2("lfm2a", false, 0)]
    #[case::lfm3("lfm3a", false, 0)]
    #[case::voxtral("voxtral", false, 1)]
    #[case::qwen25_omni("qwen2.5o", true, 2)]
    #[case::qwen25_omni_audio("qwen2.5oa", false, 2)]
    #[case::gemma4("gemma4a", true, 2)]
    #[case::gemma4_unified("gemma4ua", true, 2)]
    #[case::minicpm_o("minicpmoa", false, 2)]
    #[case::qwen3_asr("qwen3a", false, 2)]
    #[case::qwen3_omni("qwen3a", true, 2)]
    fn every_recipe_pins_its_audio_markers(
        #[case] projector: &str,
        #[case] with_vision: bool,
        #[case] markers: u32,
    ) -> anyhow::Result<()> {
        let recipe = Recipe::for_encoder(&AudioEncoder::from_name(projector, with_vision))?;
        assert_eq!(recipe.audio_markers, markers);
        Ok(())
    }

    #[test]
    fn the_cache_key_names_everything_the_count_depends_on() -> anyhow::Result<()> {
        use pipette_plan_types::{AudioChannels, AudioSampleRate};
        let mono = wav::ClipFormat::ENCODER;
        let stereo = wav::ClipFormat::of(
            AudioSampleRate::try_new(16_000)?,
            AudioChannels::try_new(2)?,
        );
        let cd = wav::ClipFormat::of(
            AudioSampleRate::try_new(44_100)?,
            AudioChannels::try_new(1)?,
        );
        let base = cache_key("1234:abcd", "77:helper", mono, 150_400, &[]);
        assert!(
            base.starts_with("1234:abcd|build:77:helper|audio:16000Hz:1ch:150400|env:"),
            "{base}"
        );
        for other in [
            cache_key("9999:ffff", "77:helper", mono, 150_400, &[]),
            cache_key("1234:abcd", "78:helper", mono, 150_400, &[]),
            cache_key("1234:abcd", "77:helper", stereo, 150_400, &[]),
            cache_key("1234:abcd", "77:helper", cd, 150_400, &[]),
            cache_key("1234:abcd", "77:helper", mono, 480_000, &[]),
        ] {
            assert_ne!(base, other);
        }
        let secret = EnvForward::try_new("HF_TOKEN=hf_sentinel_secret".to_owned())?;
        let keyed = cache_key("1234:abcd", "77:helper", mono, 150_400, &[secret]);
        assert_ne!(base, keyed);
        assert!(!keyed.contains("hf_sentinel_secret"), "{keyed}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::long(&["--parallel", "2"], true)]
    #[case::equals(&["--parallel=2"], true)]
    #[case::short(&["-np", "2"], true)]
    #[case::unrelated(&["--threads", "4"], false)]
    fn a_raw_slot_count_is_refused(#[case] tokens: &[&str], #[case] refused: bool) {
        let got = refuse_raw_parallel(&raw(tokens), "asr_throughput");
        assert_eq!(
            matches!(got, Err(AsrRefusalError::RawParallel { .. })),
            refused,
            "{got:?}"
        );
    }

    #[rstest::rstest]
    #[case::pinned(true, true, AsrContextSizing::Pinned)]
    #[case::pinned_uncounted(true, false, AsrContextSizing::Pinned)]
    #[case::exact(false, true, AsrContextSizing::Exact)]
    #[case::bound(false, false, AsrContextSizing::UpperBound)]
    fn the_row_says_how_its_context_was_sized(
        #[case] pinned: bool,
        #[case] calibrated: bool,
        #[case] expected: AsrContextSizing,
    ) {
        assert_eq!(context_sizing(pinned, calibrated), expected);
    }

    #[rstest::rstest]
    #[case::one_chunk("I decoding audio batch 1/1, n_tokens_batch = 39\n", Some(39))]
    #[case::qwen3_chunks(
        "I decoding audio batch 1/1, n_tokens_batch = 104\nI decoding audio batch 1/1, n_tokens_batch = 26\n",
        Some(130)
    )]
    #[case::image_lines_are_not_audio("I decoding image batch 1/1, n_tokens_batch = 64\n", None)]
    #[case::mixed(
        "I decoding image batch 1/1, n_tokens_batch = 64\nI decoding audio batch 1/1, n_tokens_batch = 39\n",
        Some(39)
    )]
    #[case::garbled_line_is_no_count(
        "I decoding audio batch 1/1, n_tokens_batch = 104\nI decoding audio batch 1/1, n_tokens_batch = x\n",
        None
    )]
    #[case::no_batches("nothing\n", None)]
    fn the_helper_log_sums_the_audio_batches(#[case] log: &str, #[case] expected: Option<u32>) {
        assert_eq!(mtmd::parse_batch_tokens(log, AUDIO_BATCH_LINE), expected);
    }
}
