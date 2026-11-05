use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;
use reqwest::Method;
use serde::Deserialize;
use serde_json::{Value, json};

use pipette_completions::llama_server::{ServerKeys, post_chat};
pub use pipette_completions::llama_server::{
    chat, health, message_filler, tokenize, tokenize_message_text, url, with_server_tail,
};
use pipette_completions::{
    AudioFormat, ChatMessage, ChatMessages, ChatRole, ContentPart, InputAudio, MessageContent,
    TimingRequest,
};
use pipette_http::HttpClient;
use pipette_ops::generation::Generation;
use pipette_plan_types::{
    EngineExtraRuntimeConfiguration, EnvForward, LlamacppCliStockToolsExtraRuntimeConfiguration,
    LlamacppFlashAttention, LlamacppLoadMode,
};

use crate::asr::ContentOrder;
use crate::runtimes::{
    CommonKnobs, ResolvedLlamaServer, apply_launch_env, canonicalize_flag_order, launch_env,
};
use crate::thinking_controls;

pub fn apply_template(http: &HttpClient, url: &str, body: Value) -> anyhow::Result<String> {
    let apply: Value = http
        .json_request(
            Method::POST,
            &format!("{url}/apply-template"),
            None,
            Some(body),
        )
        .context("POST /apply-template")?;
    apply
        .get("prompt")
        .and_then(Value::as_str)
        .map(str::to_string)
        .context("/apply-template response missing prompt")
}

/// Whether the running server's chat template has a thinking switch: a probe
/// conversation, with or without a system prompt, renders differently with
/// `enable_thinking` off than on, as llama.cpp itself tests. The request's
/// kwarg overrides the server's default, so its reasoning flags do not change
/// the answer. Only Jinja reads the kwarg, which is why
/// [`ServerCommandBuilder::check_thinking_controls`] refuses whatever could
/// turn Jinja off.
///
/// A template that reads `enable_thinking` only beside tools, an assistant turn
/// or an image passes as one without a switch.
pub fn thinking_switch(http: &HttpClient, url: &str) -> anyhow::Result<bool> {
    let conversations = [
        json!([{"role": "user", "content": "Hi"}]),
        json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ]),
    ];
    for messages in &conversations {
        let switch = match renders_a_switch(http, url, messages)? {
            Some(switch) => switch,
            // Once more: a date that turned over between two renders settles.
            None => renders_a_switch(http, url, messages)?.context(
                "the chat template rendered the same probe differently with thinking on both \
                 times, twice, so whether it has a thinking switch cannot be told; it likely \
                 prints something that changes per render",
            )?,
        };
        if switch {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Rendered on, off, then on again: `None` when the two `on` renders disagree,
/// as a template that prints the date (gpt-oss) can, so the difference says
/// nothing about a switch.
fn renders_a_switch(
    http: &HttpClient,
    url: &str,
    messages: &Value,
) -> anyhow::Result<Option<bool>> {
    let render = |enable_thinking: bool| {
        apply_template(
            http,
            url,
            json!({
                "messages": messages,
                "add_generation_prompt": true,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }),
        )
    };
    let on = render(true)?;
    let off = render(false)?;
    Ok((render(true)? == on).then(|| off != on))
}

pub fn no_thinking_switch(model: &Path, asked: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{} has no thinking switch in its chat template, so {asked} cannot be honoured; \
         prompt-switched thinking (e.g. InternVL3.5) is the internvl_r1 prompt profile",
        model.display()
    )
}

/// [`chat`] with one audio clip in place of the images, timed and refused the
/// same way. An empty `instruction` sends the clip alone.
pub fn audio_chat(
    http: &HttpClient,
    url: &str,
    system: Option<&str>,
    instruction: &str,
    order: ContentOrder,
    clip: Option<&str>,
    decode_tokens: u32,
) -> anyhow::Result<Generation> {
    post_chat(
        http,
        url,
        &build_audio_chat_body(system, instruction, order, clip, decode_tokens)?,
    )
}

/// As `/props` reports it, so an ASR cell can check the server renders its
/// recipe's template.
pub fn chat_template(http: &HttpClient, url: &str) -> anyhow::Result<Option<String>> {
    #[derive(Deserialize)]
    struct Props {
        #[serde(default)]
        chat_template: Option<String>,
    }
    let props: Props = http
        .json_request(Method::GET, &format!("{url}/props"), None, None)
        .context("GET /props")?;
    Ok(props.chat_template)
}

/// `clip` is the WAV file base64-encoded; `None` is the baseline probe, the
/// same turns with no audio, which measures the audio tokens when they cannot
/// be counted directly.
fn build_audio_chat_body(
    system: Option<&str>,
    instruction: &str,
    order: ContentOrder,
    clip: Option<&str>,
    decode_tokens: u32,
) -> anyhow::Result<Value> {
    let text = (!instruction.is_empty()).then_some(ContentPart::Text { text: instruction });
    let audio = clip.map(|data| ContentPart::InputAudio {
        input_audio: InputAudio {
            data,
            format: AudioFormat::Wav,
        },
    });
    let parts: Vec<ContentPart<'_>> = match order {
        ContentOrder::ClipThenInstruction => audio.into_iter().chain(text).collect(),
        ContentOrder::InstructionThenClip => text.into_iter().chain(audio).collect(),
    };
    // The baseline of a clip sent alone has no parts; an empty string still
    // renders through the chat template.
    let content = if parts.is_empty() {
        MessageContent::Text("")
    } else {
        MessageContent::Parts(parts)
    };
    let user = ChatMessage {
        role: ChatRole::User,
        content,
    };
    let messages = ChatMessages {
        messages: match system {
            Some(system) => vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: MessageContent::Text(system),
                },
                user,
            ],
            None => vec![user],
        },
    };
    let request = TimingRequest::new(
        None,
        messages,
        decode_tokens,
        ServerKeys {
            ignore_eos: true,
            cache_prompt: false,
        },
    );
    serde_json::to_value(request).context("failed to encode the chat request")
}

/// The context a server cell derives for `slots` slots of `per_slot_ctx` each, because
/// llama-server divides `--ctx-size` across them and each request of a batch should get
/// the window a serial run gave the only request. A cell that pins `ctx_size` states the
/// total itself and owns the division, as an eval cell with `parallel` does.
pub fn ctx_across_slots(per_slot_ctx: u32, slots: u32) -> anyhow::Result<u32> {
    per_slot_ctx.checked_mul(slots).with_context(|| {
        format!(
            "{slots} slots x {per_slot_ctx} context per slot overflows; pin ctx_size on the cell"
        )
    })
}

/// Whether `raw` sets the option one of `aliases` spells, its name read as
/// llama.cpp reads it (`_` for `-` in a long name). A value glued on with `=`
/// counts too: llama.cpp refuses that spelling, so the launch fails on it
/// rather than on a second value of ours.
fn raw_sets(raw: &[String], aliases: &[&str]) -> bool {
    raw.iter().any(|token| {
        let name = token
            .split_once('=')
            .map_or(token.as_str(), |(name, _)| name);
        aliases.contains(&crate::runtimes::canonical_flag_name(name).as_str())
    })
}

/// Largest `--ctx-size` llama-server parses: past `i32::MAX`, `std::stoi` dies
/// in argument parsing with an unhelpful message.
const MAX_CTX_SIZE: u32 = i32::MAX as u32;
/// llama.cpp's spellings of the slot count, to spot one in `raw`.
const PARALLEL_ALIASES: &[&str] = &["--parallel", "-np"];
/// llama.cpp's spellings of the checkpoint count, to spot one in `raw`.
const CTX_CHECKPOINTS_ALIASES: &[&str] = &["--ctx-checkpoints", "-ctxcp", "--swa-checkpoints"];
/// llama.cpp's spellings of the host prompt cache's size, to spot one in `raw`.
const CACHE_RAM_ALIASES: &[&str] = &["--cache-ram", "-cram"];
/// llama.cpp's switch for reusing a slot's previous prompt, to spot it in `raw`.
const CACHE_PROMPT_SWITCHES: &[&str] = &["--cache-prompt", "--no-cache-prompt"];
/// The server cell a [`ServerCommandBuilder`] launches, named in its errors and
/// log lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::Display)]
#[strum(serialize_all = "snake_case")]
pub enum LoggingMarker {
    AsrEval,
    AsrMaxMemory,
    AsrThroughput,
    EndToEndLatency,
    Eval,
    VlMaxMemory,
    VlSpecDecode,
    VlThroughput,
    VlmEval,
}

impl LoggingMarker {
    fn is_asr(self) -> bool {
        match self {
            Self::AsrMaxMemory | Self::AsrThroughput => true,
            Self::AsrEval
            | Self::EndToEndLatency
            | Self::Eval
            | Self::VlMaxMemory
            | Self::VlSpecDecode
            | Self::VlThroughput
            | Self::VlmEval => false,
        }
    }
}

/// A launch [`ServerCommandBuilder`] refuses to render.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ServerArgsError {
    #[error(
        "{logging_marker}: ctx-size {ctx} exceeds the largest context llama-server can parse \
         ({MAX_CTX_SIZE}); pin a smaller ctx_size"
    )]
    CtxTooLarge {
        logging_marker: LoggingMarker,
        ctx: u32,
    },
    #[error(
        "{logging_marker}: parallel={slots} conflicts with raw {flag} in --runtime-flags; two slot \
         flags would leave llama-server honoring one while we sized the context for the \
         other. Drop the raw flag"
    )]
    RawSlotConflict {
        logging_marker: LoggingMarker,
        slots: u32,
        flag: &'static str,
    },
    /// Plan-types refuses zero, but [`ServerCommandBuilder::slots`] takes a bare `u32`.
    #[error("{logging_marker}: parallel must be at least 1, got 0")]
    ZeroSlots { logging_marker: LoggingMarker },
    #[error("{logging_marker}: an ASR server runs one slot, got parallel={slots}")]
    AsrSlots {
        logging_marker: LoggingMarker,
        slots: u32,
    },
    #[error(
        "{logging_marker}: more than one slot flag in raw --runtime-flags; the flags are \
         rendered in canonical order, so which one llama-server applies need not be the one \
         written last, and the context would be sized for the other. Keep one"
    )]
    RepeatedSlotFlag { logging_marker: LoggingMarker },
    #[error(
        "{logging_marker}: raw {flag} in --runtime-flags renders the chat template without \
         Jinja, the only renderer that reads enable_thinking, so whether the model thinks could \
         be neither told nor set. Drop the raw flag"
    )]
    NoJinja {
        logging_marker: LoggingMarker,
        flag: &'static str,
    },
    #[error(
        "{logging_marker}: raw {flag} in --runtime-flags sets how llama-server thinks past the \
         model flags, which own it on this cell: `enable_thinking` (eval and vlm_eval; false \
         on asr_eval, which runs with `--reasoning off`) and `reasoning_effort` (eval). Drop the \
         raw flag"
    )]
    RawReasoningFlag {
        logging_marker: LoggingMarker,
        flag: &'static str,
    },
    #[error(
        "{logging_marker}: raw {flag} in --runtime-flags takes the model from somewhere other \
         than the path pipette launches, and a Hugging Face repo with a preset.ini turns \
         llama-server into a router running that file's options, which the recorded command \
         would not show. Drop the raw flag"
    )]
    ModelSourceFlag {
        logging_marker: LoggingMarker,
        flag: &'static str,
    },
    #[error(
        "{logging_marker}: {name} in the cell's or the client's envs sets how llama-server \
         thinks, renders its chat template or finds its model outside its argv, where the \
         recorded command would not show it. Drop it: an eval or vlm_eval cell sets thinking \
         with the `enable_thinking` model flag"
    )]
    HiddenEnv {
        logging_marker: LoggingMarker,
        name: &'static str,
    },
    #[error(
        "{logging_marker}: {name} in the client's envs sets a llama.cpp option, which a vlm_eval's \
         catalog entry owns; llama-server would read it before its argv, where the recorded \
         command would not show it. Drop it"
    )]
    LlamaArgEnv {
        logging_marker: LoggingMarker,
        name: String,
    },
    #[error(
        "{logging_marker}: {} sets `{key}`, which decides how llama-server thinks, renders its \
         chat template or finds its model, or (on a vlm_eval) a setting its recipe owns. \
         llama.cpp applies that file to every launch, before env and argv, and the recorded \
         command would not show it. Remove the key from the file",
        path.display()
    )]
    HiddenConfig {
        logging_marker: LoggingMarker,
        path: PathBuf,
        key: String,
    },
    #[error(
        "{logging_marker}: {name} in the client's envs sets a llama.cpp option, which an \
         asr_eval's catalog entry and prompt profile own; llama-server would read it before its \
         argv, where the recorded command would not show it. Drop it"
    )]
    AsrOwnedEnv {
        logging_marker: LoggingMarker,
        name: String,
    },
    #[error(
        "{logging_marker}: {} sets `{key}`, a setting an asr_eval's catalog entry and prompt \
         profile own (its decode, chat template, metadata overrides, beam search, slot split or \
         output cap, or any option through its variable name). llama.cpp applies that file to \
         every launch, before env and argv, and the recorded command would not show it. Remove \
         the key from the file",
        path.display()
    )]
    AsrOwnedConfig {
        logging_marker: LoggingMarker,
        path: PathBuf,
        key: String,
    },
    #[error(
        "{logging_marker}: llama.cpp would apply {} to every launch, and it could not be read to \
         check it for hidden controls: {error}",
        path.display()
    )]
    UnreadableConfig {
        logging_marker: LoggingMarker,
        path: PathBuf,
        error: String,
    },
}

/// One llama-server launch. [`Self::build`] takes the port, so one builder starts
/// each server of a cell, and each restart, with the same flags. The knobs arrive
/// resolved: the slot count, derived context and load mode already overlaid
/// ([`ctx_across_slots`], [`crate::runtimes::get_load_mode`]). Converting the
/// plan's entry already refused a reserved flag in `raw`.
#[derive(Debug, Clone)]
pub struct ServerCommandBuilder {
    program: ResolvedLlamaServer,
    model: PathBuf,
    logging_marker: LoggingMarker,
    mmproj: Option<PathBuf>,
    threads: Option<u32>,
    number_gpu_layers: Option<u32>,
    mmap: Option<bool>,
    load_mode: Option<LlamacppLoadMode>,
    flash_attention: Option<LlamacppFlashAttention>,
    raw: Vec<String>,
    ctx_size: Option<u32>,
    no_cache: Option<bool>,
    prompt_caches: bool,
    slots: Option<u32>,
    one_slot_spelled: bool,
    extra: Vec<String>,
    envs: Vec<EnvForward>,
    client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
}

impl ServerCommandBuilder {
    pub fn new(
        program: ResolvedLlamaServer,
        model: impl Into<PathBuf>,
        logging_marker: LoggingMarker,
    ) -> Self {
        Self {
            program,
            model: model.into(),
            logging_marker,
            mmproj: None,
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            raw: Vec::new(),
            ctx_size: None,
            no_cache: None,
            prompt_caches: true,
            slots: None,
            one_slot_spelled: false,
            extra: Vec::new(),
            envs: Vec::new(),
            client_config: None,
        }
    }

    pub fn mmproj(mut self, mmproj: impl Into<PathBuf>) -> Self {
        self.mmproj = Some(mmproj.into());
        self
    }

    pub fn threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    pub fn number_gpu_layers(mut self, number_gpu_layers: Option<u32>) -> Self {
        self.number_gpu_layers = number_gpu_layers;
        self
    }

    pub fn mmap(mut self, mmap: Option<bool>) -> Self {
        self.mmap = mmap;
        self
    }

    pub fn load_mode(mut self, load_mode: Option<LlamacppLoadMode>) -> Self {
        self.load_mode = load_mode;
        self
    }

    pub fn flash_attention(mut self, flash_attention: Option<LlamacppFlashAttention>) -> Self {
        self.flash_attention = flash_attention;
        self
    }

    pub fn raw(mut self, raw: Vec<String>) -> Self {
        self.raw = raw;
        self
    }

    /// Renders verbatim: the total llama-server divides across the slots, and
    /// the benchmark sized it so.
    pub fn ctx_size(mut self, ctx_size: Option<u32>) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    /// `true` renders the single-request server: `--no-cache-prompt`,
    /// `--ctx-checkpoints 0` and, at one slot, `--parallel 1`, each unless `raw`
    /// sets it in any spelling.
    pub fn no_cache(mut self, no_cache: Option<bool>) -> Self {
        self.no_cache = no_cache;
        self
    }

    /// `--no-cache-prompt`, `--ctx-checkpoints 0` and `--cache-ram 0`, each
    /// unless `raw` sets it, at the cell's own slot count. Left on, a request
    /// that lands on a slot by LRU, as each new sample of an eval does, first
    /// copies that slot's whole state to host memory, and every idle slot's on
    /// each new task, while the server loop waits; a hybrid or sliding-window
    /// model also snapshots its state on every prompt. `cache_prompt: false` on
    /// a request stops none of that.
    pub fn without_prompt_caches(mut self) -> Self {
        self.prompt_caches = false;
        self
    }

    /// `None` where nothing types a slot count, so a `raw` `--parallel` passes
    /// through, with a warning.
    pub fn slots(mut self, slots: Option<u32>) -> Self {
        self.slots = slots;
        self
    }

    /// `true` renders `--parallel 1` at one slot, for a server that needs the
    /// count spelled: the `--beam-search` of pipette's ASR build of llama.cpp
    /// refuses to start beside llama-server's default of four.
    pub fn one_slot_spelled(mut self, one_slot_spelled: bool) -> Self {
        self.one_slot_spelled = one_slot_spelled;
        self
    }

    /// Argv the benchmark adds beside `raw`: a speculative cell's drafter, a
    /// vlm_eval cell's [`thinking`] flags.
    pub fn extra(mut self, extra: Vec<String>) -> Self {
        self.extra = extra;
        self
    }

    pub fn envs(mut self, envs: Vec<EnvForward>) -> Self {
        self.envs = envs;
        self
    }

    pub fn client_config(
        mut self,
        client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
    ) -> Self {
        self.client_config = client_config;
        self
    }

    /// Everything but the model, projector, host and port, in canonical order.
    pub fn args(&self) -> Result<Vec<String>, ServerArgsError> {
        self.check_raw_and_envs()?;
        let Self {
            threads,
            number_gpu_layers,
            mmap,
            load_mode,
            flash_attention,
            raw,
            ctx_size,
            no_cache,
            prompt_caches,
            slots,
            one_slot_spelled,
            extra,
            logging_marker,
            ..
        } = self;
        let no_cache = *no_cache == Some(true);
        let mut tokens = CommonKnobs {
            threads: *threads,
            number_gpu_layers: *number_gpu_layers,
            mmap: *mmap,
            load_mode: *load_mode,
            flash_attention: *flash_attention,
            raw,
        }
        .common_arg_tokens();
        // With the prompt cache off the checkpoints are never restored.
        if no_cache || !*prompt_caches {
            if !raw_sets(raw, CACHE_PROMPT_SWITCHES) {
                tokens.push("--no-cache-prompt".to_owned());
            }
            if !raw_sets(raw, CTX_CHECKPOINTS_ALIASES) {
                tokens.extend(["--ctx-checkpoints".to_owned(), "0".to_owned()]);
            }
        }
        if !*prompt_caches && !raw_sets(raw, CACHE_RAM_ALIASES) {
            tokens.extend(["--cache-ram".to_owned(), "0".to_owned()]);
        }
        tokens.extend(raw.iter().cloned());
        tokens.extend(extra.iter().cloned());

        let slots = resolve_slots(&tokens, *slots, *logging_marker)?;
        let mut out = vec!["--no-warmup".to_owned()];
        // Unset, llama-server opens four slots, each with its own recurrent and
        // sliding-window state; an explicit 1 holds from b7433. An ASR server
        // pins one whatever its cache: its context is sized for one clip, and
        // four slots would each hold a quarter of it.
        let pin_one = no_cache || *one_slot_spelled || logging_marker.is_asr();
        if slots > 1 || (pin_one && raw_parallel(&tokens).is_none()) {
            out.extend(["--parallel".to_owned(), slots.to_string()]);
        }
        if let Some(ctx) = *ctx_size {
            if ctx > MAX_CTX_SIZE {
                return Err(ServerArgsError::CtxTooLarge {
                    logging_marker: *logging_marker,
                    ctx,
                });
            }
            if slots > 1 {
                // One slot's share decides whether a sample's prompt plus
                // generation fits; llama-server echoes it as `n_ctx_slot`.
                tracing::info!(
                    benchmark_type = %logging_marker,
                    ctx_size = ctx,
                    slots,
                    per_slot = ctx / slots,
                    "--ctx-size divided across slots"
                );
            }
            out.extend(["--ctx-size".to_owned(), ctx.to_string()]);
        }
        out.extend(tokens);
        Ok(canonicalize_flag_order(&out))
    }

    /// The server listening on `port` with [`Self::args`], once
    /// [`Self::check_thinking_controls`] has found nothing to refuse.
    pub fn build(&self, port: u16) -> Result<Command, ServerArgsError> {
        self.check_thinking_controls()?;
        let args = self.args()?;
        let mut command = Command::new(self.program.as_os_str());
        command.arg("--model").arg(&self.model);
        if let Some(mmproj) = &self.mmproj {
            command.arg("--mmproj").arg(mmproj);
        }
        command.arg("--host").arg("127.0.0.1");
        command.arg("--port").arg(port.to_string());
        command.args(args);
        // A vlm_eval's or an asr_eval's recipe owns every option, so its
        // server inherits none from pipette's environment.
        if self.recipe_keys() != thinking_controls::RecipeKeys::None {
            for name in thinking_controls::HIDDEN_ENVS {
                command.env_remove(name);
            }
            thinking_controls::remove_inherited_llama_args(&mut command);
        }
        if self.recipe_keys() == thinking_controls::RecipeKeys::AsrEval {
            command.env_remove(thinking_controls::TEMPLATE_KWARGS_ENV);
        }
        apply_launch_env(
            &mut command,
            &self.program,
            &self.envs,
            self.client_config.as_ref(),
        );
        Ok(command)
    }

    /// On a cell whose catalog recipe owns the model and its thinking mode (a
    /// vlm_eval, or an asr_eval, whose entry and prompt profile own them),
    /// refuse every control over either that the typed knobs do not own or the
    /// recorded command would not show: [`Self::args`]'s checks of `raw` and the
    /// envs, then a guarded key in a config file llama.cpp applies to every
    /// launch. It reads nothing but those files, so vlm_eval and the ASR cells
    /// run it before they calibrate; [`Self::build`] runs it for every launch.
    ///
    /// An asr_throughput or asr_max_memory cell's recipe owns its template and
    /// thinking, and refuses the raw
    /// flags and envs that would change them itself
    /// ([`crate::asr::Recipe::server_args`]), so it gets the config files'
    /// check alone, against the keys every cell's guard names (thinking, the
    /// template, Jinja, the model's source), not the samplers, drafters and
    /// slots a vlm_eval's recipe owns. Every other cell passes.
    pub fn check_thinking_controls(&self) -> Result<(), ServerArgsError> {
        if self.recipe_keys() == thinking_controls::RecipeKeys::None
            && !self.logging_marker.is_asr()
        {
            return Ok(());
        }
        self.check_raw_and_envs()?;
        let var =
            |name: &str| launch_env(name, &self.program, &self.envs, self.client_config.as_ref());
        thinking_controls::config_files(var, thinking_controls::account_home)
            .into_iter()
            .try_for_each(|path| {
                // An asr_eval's thinking and model-source keys keep their own
                // error; the rest of what its recipe owns gets asr_eval's.
                let checked = match self.recipe_keys() {
                    thinking_controls::RecipeKeys::AsrEval => thinking_controls::config_control(
                        &path,
                        thinking_controls::RecipeKeys::None,
                    )
                    .and_then(|hidden| match hidden {
                        Some(key) => Ok(Some((key, false))),
                        None => thinking_controls::config_control(
                            &path,
                            thinking_controls::RecipeKeys::AsrEval,
                        )
                        .map(|owned| owned.map(|key| (key, true))),
                    }),
                    recipe @ (thinking_controls::RecipeKeys::None
                    | thinking_controls::RecipeKeys::VlmEval) => {
                        thinking_controls::config_control(&path, recipe)
                            .map(|key| key.map(|key| (key, false)))
                    }
                };
                match checked {
                    Ok(None) => Ok(()),
                    Ok(Some((key, false))) => Err(ServerArgsError::HiddenConfig {
                        logging_marker: self.logging_marker,
                        path,
                        key,
                    }),
                    Ok(Some((key, true))) => Err(ServerArgsError::AsrOwnedConfig {
                        logging_marker: self.logging_marker,
                        path,
                        key,
                    }),
                    Err(err) => Err(ServerArgsError::UnreadableConfig {
                        logging_marker: self.logging_marker,
                        path,
                        error: err.to_string(),
                    }),
                }
            })
    }

    /// The checks that read nothing, on a vlm_eval or asr_eval cell only:
    /// `--no-jinja`, the reasoning flags and a model source in `raw`, a guarded
    /// variable in the cell's or the client's envs (llama-server reads them
    /// before its argv), and any llama.cpp option variable in the client's
    /// configuration (an asr_eval's own refusal names what its entry owns).
    fn check_raw_and_envs(&self) -> Result<(), ServerArgsError> {
        let logging_marker = self.logging_marker;
        if self.recipe_keys() == thinking_controls::RecipeKeys::None {
            return Ok(());
        }
        if let Some(flag) =
            thinking_controls::raw_flag(&self.raw, thinking_controls::MODEL_SOURCE_FLAGS)
        {
            return Err(ServerArgsError::ModelSourceFlag {
                logging_marker,
                flag,
            });
        }
        if let Some(flag) = thinking_controls::raw_flag(&self.raw, &[thinking_controls::NO_JINJA]) {
            return Err(ServerArgsError::NoJinja {
                logging_marker,
                flag,
            });
        }
        if let Some(flag) =
            thinking_controls::raw_flag(&self.raw, thinking_controls::REASONING_FLAGS)
        {
            return Err(ServerArgsError::RawReasoningFlag {
                logging_marker,
                flag,
            });
        }
        let client_envs = self
            .client_config
            .as_ref()
            .map_or(&[][..], EngineExtraRuntimeConfiguration::envs);
        if let Some(name) = thinking_controls::hidden_env(self.envs.iter().chain(client_envs)) {
            return Err(ServerArgsError::HiddenEnv {
                logging_marker,
                name,
            });
        }
        // Plan-types refuses one in a vlm_eval cell's envs; the client's own
        // configuration is checked here.
        if logging_marker == LoggingMarker::VlmEval
            && let Some(name) =
                thinking_controls::llama_arg_env(self.envs.iter().chain(client_envs))
        {
            return Err(ServerArgsError::LlamaArgEnv {
                logging_marker,
                name,
            });
        }
        // Likewise on an asr_eval cell, whose entry and prompt profile own them
        // and the template kwargs' older variable.
        if logging_marker == LoggingMarker::AsrEval
            && let Some(name) = thinking_controls::llama_arg_env(
                self.envs.iter().chain(client_envs),
            )
            .or_else(|| thinking_controls::template_kwargs_env(self.envs.iter().chain(client_envs)))
        {
            return Err(ServerArgsError::AsrOwnedEnv {
                logging_marker,
                name,
            });
        }
        Ok(())
    }

    /// Whose catalog recipe this launch runs under, which decides what its
    /// config files and inherited environment may set: a vlm_eval's or an
    /// asr_eval's; no other cell's.
    fn recipe_keys(&self) -> thinking_controls::RecipeKeys {
        match self.logging_marker {
            LoggingMarker::VlmEval => thinking_controls::RecipeKeys::VlmEval,
            LoggingMarker::AsrEval => thinking_controls::RecipeKeys::AsrEval,
            LoggingMarker::AsrMaxMemory
            | LoggingMarker::AsrThroughput
            | LoggingMarker::EndToEndLatency
            | LoggingMarker::Eval
            | LoggingMarker::VlMaxMemory
            | LoggingMarker::VlThroughput
            | LoggingMarker::VlSpecDecode => thinking_controls::RecipeKeys::None,
        }
    }
}

/// The argv a vlm_eval cell's `enable_thinking` adds, given whether the model's
/// chat template has a thinking switch ([`thinking_switch`]) and whether the
/// cell's prompt profile turns thinking on (InternVL3.5's `internvl_r1`).
///
/// A template with a switch is switched explicitly, because llama-server's
/// default (`--reasoning auto`) enables thinking on Qwen3.5 and Gemma 4; `false`
/// adds the sampler-level hard stop `--reasoning-budget 0` beside
/// `--reasoning off`, and an unstated mode is refused, so every row says which
/// mode it scored. A template with no switch gets no flag: `false` is refused
/// there, and `true` is honoured only when the prompt profile thinks.
pub fn thinking(
    enable_thinking: Option<bool>,
    thinking_switch: bool,
    prompt_thinks: bool,
) -> Result<Vec<String>, ThinkingRefused> {
    let argv: &[&str] = match (enable_thinking, thinking_switch) {
        (Some(true), true) => &["--reasoning", "on"],
        (Some(false), true) => &["--reasoning", "off", "--reasoning-budget", "0"],
        (None, true) => return Err(ThinkingRefused::Unstated),
        (None, false) => &[],
        (Some(true), false) if prompt_thinks => &[],
        (Some(true), false) => return Err(ThinkingRefused::NoSwitch),
        (Some(false), false) => return Err(ThinkingRefused::NoMode),
    };
    Ok(argv.iter().map(|&token| token.to_owned()).collect())
}

/// Why [`thinking`] refused a cell's declared mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingRefused {
    /// The template has a switch and the cell did not say which way.
    Unstated,
    /// Thinking asked of a template with no switch and no thinking prompt.
    NoSwitch,
    /// `false` asked of a model with no reasoning mode to turn off.
    NoMode,
}

impl ThinkingRefused {
    pub fn error(self, model: &Path) -> anyhow::Error {
        match self {
            Self::NoSwitch => no_thinking_switch(model, "enable_thinking = true"),
            Self::Unstated => anyhow::anyhow!(
                "{} has a thinking switch in its chat template, so the cell must say which mode \
                 it scores: set its model flags' enable_thinking (true runs its reasoning, false \
                 turns it off)",
                model.display()
            ),
            Self::NoMode => anyhow::anyhow!(
                "{} has no thinking switch in its chat template, so it has no reasoning mode to \
                 turn off: leave enable_thinking unset and it runs as its template renders",
                model.display()
            ),
        }
    }
}

/// Slot count to render, refusing the combinations where the context computed
/// here would not be the one llama-server honors. With no typed knob a `raw`
/// `--parallel` passes through, so plans that set one keep working, with a
/// warning: llama-server divides a context sized for one slot across all of them.
fn resolve_slots(
    tokens: &[String],
    slots: Option<u32>,
    logging_marker: LoggingMarker,
) -> Result<u32, ServerArgsError> {
    // vlm_eval sizes every slot for its largest sample, so it refuses the
    // ambiguity; another cell takes the first flag found.
    if logging_marker == LoggingMarker::VlmEval && slot_flags(tokens).nth(1).is_some() {
        return Err(ServerArgsError::RepeatedSlotFlag { logging_marker });
    }
    let raw = raw_parallel(tokens).map(|(name, _)| name);
    // An ASR server always runs one slot, so a raw slot flag conflicts with it.
    let slots = slots.or_else(|| logging_marker.is_asr().then_some(1));
    let Some(n) = slots else {
        if let Some(name) = raw {
            tracing::warn!(
                benchmark_type = %logging_marker,
                flag = %name,
                "a raw slot flag in --runtime-flags splits --ctx-size across slots, so each \
                 gets a share of a context sized for one request; pin a ctx_size that covers \
                 them all or add --kv-unified, or set the typed `parallel` knob where the \
                 cell has one"
            );
        }
        return Ok(1);
    };
    if let Some(flag) = raw {
        return Err(ServerArgsError::RawSlotConflict {
            logging_marker,
            slots: n,
            flag,
        });
    }
    if n == 0 {
        return Err(ServerArgsError::ZeroSlots { logging_marker });
    }
    if logging_marker.is_asr() && n != 1 {
        return Err(ServerArgsError::AsrSlots {
            logging_marker,
            slots: n,
        });
    }
    Ok(n)
}

/// The first slot flag in `tokens` (`--parallel N`, `-np N`, and the `=` and
/// glued forms a plan may carry although llama-server refuses them): its
/// spelling, and its count when it parses. [`ServerCommandBuilder::args`]
/// passes a `raw` one through unscaled, so a caller that checks the context
/// per slot needs the count.
pub fn raw_parallel(tokens: &[String]) -> Option<(&'static str, Option<u32>)> {
    slot_flags(tokens).next()
}

/// Every slot flag in `tokens`, in order, as [`raw_parallel`] reads one.
fn slot_flags(tokens: &[String]) -> impl Iterator<Item = (&'static str, Option<u32>)> + '_ {
    tokens.iter().enumerate().filter_map(|(i, token)| {
        PARALLEL_ALIASES.iter().find_map(|&name| {
            let rest = token.strip_prefix(name)?;
            let count = if rest.is_empty() {
                tokens.get(i + 1).map(String::as_str)
            } else if let Some(count) = rest.strip_prefix('=') {
                Some(count)
            } else if !name.starts_with("--") && rest.starts_with(|c: char| c.is_ascii_digit()) {
                Some(rest)
            } else {
                return None;
            };
            Some((name, count.and_then(|count| count.parse().ok())))
        })
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::runtimes::test_llama_server;

    /// `cache_prompt` is off and generation runs to its limit in every shape, or
    /// the timings would not be the workload's.
    #[rstest]
    #[case::instruction_and_clip(
        None, "Transcribe this audio.", ContentOrder::InstructionThenClip, Some("QUJD"),
        serde_json::json!([{ "role": "user", "content": [
            { "type": "text", "text": "Transcribe this audio." },
            { "type": "input_audio", "input_audio": { "data": "QUJD", "format": "wav" } },
        ]}])
    )]
    #[case::clip_then_instruction(
        None, "Transcribe this audio.", ContentOrder::ClipThenInstruction, Some("QUJD"),
        serde_json::json!([{ "role": "user", "content": [
            { "type": "input_audio", "input_audio": { "data": "QUJD", "format": "wav" } },
            { "type": "text", "text": "Transcribe this audio." },
        ]}])
    )]
    #[case::system_and_clip_alone(
        Some("Perform ASR."), "", ContentOrder::InstructionThenClip, Some("QUJD"),
        serde_json::json!([
            { "role": "system", "content": "Perform ASR." },
            { "role": "user", "content": [
                { "type": "input_audio", "input_audio": { "data": "QUJD", "format": "wav" } },
            ]},
        ])
    )]
    #[case::baseline_with_instruction(
        None, "Transcribe this audio.", ContentOrder::InstructionThenClip, None,
        serde_json::json!([{ "role": "user", "content": [
            { "type": "text", "text": "Transcribe this audio." },
        ]}])
    )]
    #[case::baseline_of_a_clip_alone(
        Some("Perform ASR."), "", ContentOrder::InstructionThenClip, None,
        serde_json::json!([
            { "role": "system", "content": "Perform ASR." },
            { "role": "user", "content": "" },
        ])
    )]
    fn the_asr_request_carries_the_recipes_turns(
        #[case] system: Option<&str>,
        #[case] instruction: &str,
        #[case] order: ContentOrder,
        #[case] clip: Option<&str>,
        #[case] messages: Value,
    ) -> anyhow::Result<()> {
        let body = build_audio_chat_body(system, instruction, order, clip, 128)?;
        assert_eq!(body["messages"], messages);
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(body["cache_prompt"], false);
        assert_eq!(body["temperature"], 0.0);
        Ok(())
    }

    /// A builder for a text cell that sets nothing yet.
    fn launch() -> ServerCommandBuilder {
        ServerCommandBuilder::new(test_llama_server(), "model.gguf", LoggingMarker::Eval)
    }

    fn render(
        ctx_size: u32,
        mmap: Option<bool>,
        load_mode: Option<LlamacppLoadMode>,
        no_cache: Option<bool>,
        raw: &[&str],
        slots: Option<u32>,
    ) -> Result<Vec<String>, ServerArgsError> {
        let raw: Vec<String> = raw.iter().map(|s| (*s).to_string()).collect();
        launch()
            .mmap(mmap)
            .load_mode(load_mode)
            .raw(raw)
            .ctx_size(Some(ctx_size))
            .no_cache(no_cache)
            .slots(slots)
            .args()
    }

    #[rstest]
    #[case::asr_eval(LoggingMarker::AsrEval, "asr_eval")]
    #[case::asr_max_memory(LoggingMarker::AsrMaxMemory, "asr_max_memory")]
    #[case::asr_throughput(LoggingMarker::AsrThroughput, "asr_throughput")]
    #[case::end_to_end_latency(LoggingMarker::EndToEndLatency, "end_to_end_latency")]
    #[case::eval(LoggingMarker::Eval, "eval")]
    #[case::vl_max_memory(LoggingMarker::VlMaxMemory, "vl_max_memory")]
    #[case::vl_spec_decode(LoggingMarker::VlSpecDecode, "vl_spec_decode")]
    #[case::vl_throughput(LoggingMarker::VlThroughput, "vl_throughput")]
    #[case::vlm_eval(LoggingMarker::VlmEval, "vlm_eval")]
    fn a_marker_renders_its_benchmark_type(#[case] marker: LoggingMarker, #[case] expected: &str) {
        assert_eq!(marker.to_string(), expected);
    }

    #[rstest]
    #[case::switch_on(Some(true), true, false, Ok(&["--no-warmup", "--reasoning", "on"][..]))]
    #[case::switch_off(
        Some(false),
        true,
        false,
        Ok(&["--no-warmup", "--reasoning", "off", "--reasoning-budget", "0"][..])
    )]
    #[case::switch_unstated(None, true, false, Err(ThinkingRefused::Unstated))]
    #[case::no_switch_unstated(None, false, false, Ok(&["--no-warmup"][..]))]
    #[case::no_switch_thinking_prompt(Some(true), false, true, Ok(&["--no-warmup"][..]))]
    #[case::no_switch_on(Some(true), false, false, Err(ThinkingRefused::NoSwitch))]
    #[case::no_switch_off(Some(false), false, false, Err(ThinkingRefused::NoMode))]
    fn the_declared_mode_renders_its_argv(
        #[case] enable_thinking: Option<bool>,
        #[case] thinking_switch: bool,
        #[case] prompt_thinks: bool,
        #[case] expected: Result<&[&str], ThinkingRefused>,
    ) -> anyhow::Result<()> {
        match (
            thinking(enable_thinking, thinking_switch, prompt_thinks),
            expected,
        ) {
            (Ok(extra), Ok(expected)) => assert_eq!(
                ServerCommandBuilder::new(
                    test_llama_server(),
                    "model.gguf",
                    LoggingMarker::VlmEval
                )
                .extra(extra)
                .args()?,
                expected
            ),
            (Err(got), Err(expected)) => assert_eq!(got, expected),
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_cells_cause_heads_its_error() {
        #[derive(Debug, thiserror::Error)]
        #[error("readiness wait timed out after 420s")]
        struct TimedOutError;
        let err = with_server_tail(
            anyhow::Error::new(TimedOutError).context("vl_throughput rep 1"),
            "llama-server was still running\nstderr tail:\nslot 0 ready",
        );
        let shown = format!("{err:#}");
        assert!(
            shown.starts_with("readiness wait timed out after 420s"),
            "{shown}"
        );
        assert!(shown.contains("llama-server was still running"), "{shown}");
        assert!(
            err.downcast_ref::<TimedOutError>().is_some(),
            "the cause keeps its type"
        );
    }

    /// The truncation check divides by the count: a spelling that read as `None`
    /// would skip the check.
    #[rstest]
    #[case::long(&["--parallel", "8"], Some(8))]
    #[case::long_equals(&["--parallel=8"], Some(8))]
    #[case::short(&["-np", "8"], Some(8))]
    #[case::short_glued(&["-np8"], Some(8))]
    #[case::short_equals(&["-np=8"], Some(8))]
    #[case::among_other_flags(&["-fa", "on", "--parallel", "16", "--verbose"], Some(16))]
    #[case::absent(&["--verbose"], None)]
    #[case::empty(&[], None)]
    // A value we cannot read is llama-server's to reject; guessing would be
    // worse than declining to check.
    #[case::unparseable(&["--parallel", "lots"], None)]
    #[case::missing_value(&["--parallel"], None)]
    #[case::repeated_first_found(&["--parallel", "1", "--parallel", "4"], Some(1))]
    #[case::repeated_mixed_spellings(&["-np", "4", "--parallel=1"], Some(4))]
    fn raw_parallel_reads_every_spelling(
        #[case] raw: &[&str],
        #[case] want: Option<u32>,
    ) -> anyhow::Result<()> {
        let raw: Vec<String> = raw.iter().map(|s| (*s).to_string()).collect();
        let got = raw_parallel(&raw).and_then(|(_, count)| count);
        anyhow::ensure!(got == want, "{raw:?} should read as {want:?}, got {got:?}");
        Ok(())
    }

    /// `--ctx-size` and the mmap spellings are closed to `raw`, so the typed
    /// field is the only source.
    #[rstest]
    #[case::pinned_in_ram(
        render(8448, Some(false), None, None, &[], None),
        &["--ctx-size", "8448", "--no-mmap", "--no-warmup"]
    )]
    #[case::mmap_left_on(
        render(8196, None, None, None, &[], None),
        &["--ctx-size", "8196", "--no-warmup"]
    )]
    // `mmap = true` is the escape hatch from the overlay, and it has to render
    // *nothing*: it is the only authoring that reproduces a reference protocol
    // whose server takes no load flag, and the only one a pre-`b10105` runtime
    // can launch, since every `load_mode` value renders `--load-mode`.
    #[case::mmap_asked_for(
        render(8196, Some(true), None, None, &[], None),
        &["--ctx-size", "8196", "--no-warmup"]
    )]
    // The enum renders as upstream's flag, including the modes the boolean could not spell.
    #[case::load_mode_alone(
        render(8448, None, Some(LlamacppLoadMode::DirectIo), None, &[], None),
        &["--ctx-size", "8448", "--load-mode", "dio", "--no-warmup"]
    )]
    // Both set: the enum wins and `--no-mmap` is left out, so the server is never handed
    // the deprecated-plus-new combination it warns about.
    #[case::load_mode_beats_the_boolean(
        render(8448, Some(false), Some(LlamacppLoadMode::Mmap), None, &[], None),
        &["--ctx-size", "8448", "--load-mode", "mmap", "--no-warmup"]
    )]
    // The default a cell with no load setting gets: no deprecated `--no-mmap`.
    #[case::the_default_load_mode(
        render(8448, None, Some(LlamacppLoadMode::Read), None, &[], None),
        &["--ctx-size", "8448", "--load-mode", "none", "--no-warmup"]
    )]
    // CommonKnobs that share a type, set apart so two cannot trade places.
    #[case::every_knob_set(
        launch()
            .threads(Some(3))
            .number_gpu_layers(Some(7))
            .mmap(Some(false))
            .flash_attention(Some(LlamacppFlashAttention::On))
            .ctx_size(Some(4096))
            .no_cache(Some(true))
            .args(),
        &[
            "--ctx-checkpoints", "0", "--ctx-size", "4096", "--flash-attn", "on",
            "--n-gpu-layers", "7", "--no-cache-prompt", "--no-mmap", "--no-warmup",
            "--parallel", "1", "--threads", "3",
        ]
    )]
    // What the benchmark adds: its slot count and its own argv, beside `raw`.
    #[case::slots_and_extra(
        launch()
            .raw(vec![String::from("--verbose")])
            .ctx_size(Some(4096))
            .slots(Some(4))
            .extra(vec![String::from("--model-draft"), String::from("draft.gguf")])
            .args(),
        &[
            "--ctx-size", "4096", "--model-draft", "draft.gguf", "--no-warmup",
            "--parallel", "4", "--verbose",
        ]
    )]
    fn args_render_the_typed_fields(
        #[case] argv: Result<Vec<String>, ServerArgsError>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        assert_eq!(argv?, expected);
        Ok(())
    }

    #[rstest]
    #[case::on(true, &[], None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::off(
        false, &[], None,
        &["--cache-ram", "0", "--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup"]
    )]
    #[case::off_slots(
        false, &[], Some(3),
        &["--cache-ram", "0", "--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "3"]
    )]
    #[case::off_raw_ram(
        false, &["--cache-ram", "512"], None,
        &["--cache-ram", "512", "--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup"]
    )]
    #[case::off_raw_ram_equals(
        false, &["--cache-ram=512"], None,
        &["--cache-ram=512", "--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup"]
    )]
    #[case::off_raw_ram_short(
        false, &["-cram", "512"], None,
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "-cram", "512"]
    )]
    #[case::off_raw_checkpoints(
        false, &["-ctxcp", "8"], None,
        &["--cache-ram", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "-ctxcp", "8"]
    )]
    #[case::off_raw_cache_prompt(
        false, &["--cache-prompt"], None,
        &["--cache-prompt", "--cache-ram", "0", "--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-warmup"]
    )]
    #[case::off_raw_underscores(
        false, &["--cache_prompt", "--cache_ram", "512", "--ctx_checkpoints", "8"], None,
        &["--cache_prompt", "--cache_ram", "512", "--ctx-size", "4096", "--ctx_checkpoints", "8", "--no-warmup"]
    )]
    fn without_prompt_caches_turns_each_off_unless_raw_sets_it(
        #[case] caches: bool,
        #[case] raw: &[&str],
        #[case] slots: Option<u32>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let builder = launch()
            .raw(raw.iter().map(|t| (*t).to_owned()).collect())
            .ctx_size(Some(4096))
            .slots(slots);
        let builder = if caches {
            builder
        } else {
            builder.without_prompt_caches()
        };
        assert_eq!(builder.args()?, expected);
        Ok(())
    }

    /// `no_cache` serves one request: no prompt cache, no checkpoints, one
    /// slot. `raw`, applied after it, replaces a default in any spelling, and a
    /// typed slot count renders as itself.
    #[rstest]
    #[case::cache_unset(None, &[], None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::cache_on(Some(false), &[], None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::cache_on_one_slot(Some(false), &[], Some(1), &["--ctx-size", "4096", "--no-warmup"])]
    #[case::cache_on_with_slots(Some(false), &[], Some(4), &["--ctx-size", "4096", "--no-warmup", "--parallel", "4"])]
    #[case::no_cache(
        Some(true), &[], None,
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1"]
    )]
    #[case::raw_checkpoints_long(
        Some(true), &["--ctx-checkpoints", "8"], None,
        &["--ctx-checkpoints", "8", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1"]
    )]
    #[case::raw_checkpoints_equals(
        Some(true), &["--ctx-checkpoints=8"], None,
        &["--ctx-checkpoints=8", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1"]
    )]
    #[case::raw_checkpoints_short(
        Some(true), &["-ctxcp", "8"], None,
        &["--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1", "-ctxcp", "8"]
    )]
    #[case::raw_checkpoints_old_name(
        Some(true), &["--swa-checkpoints", "8"], None,
        &["--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1", "--swa-checkpoints", "8"]
    )]
    #[case::raw_parallel_long(
        Some(true), &["--parallel", "2"], None,
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "2"]
    )]
    #[case::raw_parallel_short(
        Some(true), &["-np", "2"], None,
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "-np", "2"]
    )]
    #[case::raw_parallel_glued(
        Some(true), &["-np2"], None,
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "-np2"]
    )]
    #[case::typed_one_slot(
        Some(true), &[], Some(1),
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "1"]
    )]
    #[case::typed_slots(
        Some(true), &[], Some(4),
        &["--ctx-checkpoints", "0", "--ctx-size", "4096", "--no-cache-prompt", "--no-warmup", "--parallel", "4"]
    )]
    fn no_cache_serves_one_request(
        #[case] no_cache: Option<bool>,
        #[case] raw: &[&str],
        #[case] slots: Option<u32>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        assert_eq!(render(4096, None, None, no_cache, raw, slots)?, expected);
        Ok(())
    }

    /// The other cells with the cache left on still render no slot count.
    #[rstest]
    #[case::asr_throughput_cache_unset(LoggingMarker::AsrThroughput, None, None, &["--ctx-size", "4096", "--no-warmup", "--parallel", "1"])]
    #[case::asr_max_memory_cache_unset(LoggingMarker::AsrMaxMemory, None, None, &["--ctx-size", "4096", "--no-warmup", "--parallel", "1"])]
    #[case::asr_cache_on(LoggingMarker::AsrThroughput, Some(false), None, &["--ctx-size", "4096", "--no-warmup", "--parallel", "1"])]
    #[case::asr_one_typed_slot(LoggingMarker::AsrMaxMemory, None, Some(1), &["--ctx-size", "4096", "--no-warmup", "--parallel", "1"])]
    #[case::vl_throughput_cache_unset(LoggingMarker::VlThroughput, None, None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::vl_max_memory_cache_unset(LoggingMarker::VlMaxMemory, None, None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::eval_cache_unset(LoggingMarker::Eval, None, None, &["--ctx-size", "4096", "--no-warmup"])]
    #[case::end_to_end_cache_on(LoggingMarker::EndToEndLatency, Some(false), None, &["--ctx-size", "4096", "--no-warmup"])]
    fn an_asr_server_runs_one_slot_whatever_its_cache(
        #[case] marker: LoggingMarker,
        #[case] no_cache: Option<bool>,
        #[case] slots: Option<u32>,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let args = ServerCommandBuilder::new(test_llama_server(), "model.gguf", marker)
            .ctx_size(Some(4096))
            .no_cache(no_cache)
            .slots(slots)
            .args()?;
        assert_eq!(args, expected);
        Ok(())
    }

    #[rstest]
    #[case::asr_throughput(LoggingMarker::AsrThroughput, 2)]
    #[case::asr_max_memory(LoggingMarker::AsrMaxMemory, 4)]
    fn an_asr_server_refuses_more_than_one_typed_slot(
        #[case] marker: LoggingMarker,
        #[case] slots: u32,
    ) -> anyhow::Result<()> {
        let err = ServerCommandBuilder::new(test_llama_server(), "model.gguf", marker)
            .ctx_size(Some(4096))
            .slots(Some(slots))
            .args()
            .err()
            .context("expected Err")?;
        assert_eq!(
            err,
            ServerArgsError::AsrSlots {
                logging_marker: marker,
                slots
            }
        );
        Ok(())
    }

    #[rstest]
    #[case::asr_throughput(LoggingMarker::AsrThroughput)]
    #[case::asr_max_memory(LoggingMarker::AsrMaxMemory)]
    fn an_asr_server_refuses_a_raw_slot_flag(#[case] marker: LoggingMarker) -> anyhow::Result<()> {
        let err = ServerCommandBuilder::new(test_llama_server(), "model.gguf", marker)
            .ctx_size(Some(4096))
            .raw(vec!["--parallel".to_owned(), "2".to_owned()])
            .args()
            .err()
            .context("expected Err")?;
        assert_eq!(
            err,
            ServerArgsError::RawSlotConflict {
                logging_marker: marker,
                slots: 1,
                flag: "--parallel",
            }
        );
        Ok(())
    }

    /// The context renders verbatim whatever the slot count.
    #[rstest]
    #[case::no_slots(None, 8196, &[], &["--ctx-size", "8196", "--no-warmup"])]
    #[case::one_slot_renders_no_parallel(Some(1), 8196, &[], &["--ctx-size", "8196", "--no-warmup"])]
    #[case::four_slots(Some(4), 65536, &[], &["--ctx-size", "65536", "--no-warmup", "--parallel", "4"])]
    #[case::sixty_four_slots(Some(64), 589_824, &[], &["--ctx-size", "589824", "--no-warmup", "--parallel", "64"])]
    #[case::raw_parallel_alone_passes_through(None, 8196, &["--parallel", "8"], &["--ctx-size", "8196", "--no-warmup", "--parallel", "8"])]
    fn args_render_the_slots(
        #[case] slots: Option<u32>,
        #[case] ctx_size: u32,
        #[case] raw: &[&str],
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        assert_eq!(render(ctx_size, None, None, None, raw, slots)?, expected);
        Ok(())
    }

    #[rstest]
    #[case::two_raw_slot_flags(
        &["-np", "4", "--parallel", "1"],
        &["--ctx-size", "4096", "--no-warmup", "--parallel", "1", "-np", "4"]
    )]
    #[case::one_raw_slot_flag_twice(
        &["--parallel", "2", "--parallel", "2"],
        &["--ctx-size", "4096", "--no-warmup", "--parallel", "2", "--parallel", "2"]
    )]
    fn repeated_slot_flags_are_refused_on_vlm_eval_only(
        #[case] raw: &[&str],
        #[case] elsewhere: &[&str],
    ) -> anyhow::Result<()> {
        let builder = |marker| {
            ServerCommandBuilder::new(test_llama_server(), "model.gguf", marker)
                .raw(raw.iter().map(|t| (*t).to_owned()).collect())
                .ctx_size(Some(4096))
        };
        assert_eq!(
            builder(LoggingMarker::VlmEval).args().err(),
            Some(ServerArgsError::RepeatedSlotFlag {
                logging_marker: LoggingMarker::VlmEval
            })
        );
        assert_eq!(builder(LoggingMarker::Eval).args()?, elsewhere);
        Ok(())
    }

    fn conflict(flag: &'static str) -> ServerArgsError {
        ServerArgsError::RawSlotConflict {
            logging_marker: LoggingMarker::Eval,
            slots: 4,
            flag,
        }
    }

    /// The combinations where the context we compute would not be the context
    /// llama-server honors, plus the slot-count bounds. `RuntimeFlags` refuses
    /// the rest.
    #[rstest]
    #[case::zero_slots(4096, &[], Some(0), ServerArgsError::ZeroSlots { logging_marker: LoggingMarker::Eval })]
    #[case::raw_parallel_conflicts(4096, &["--parallel", "8"], Some(4), conflict("--parallel"))]
    #[case::raw_parallel_equals_conflicts(4096, &["--parallel=8"], Some(4), conflict("--parallel"))]
    #[case::raw_np_conflicts(4096, &["-np", "8"], Some(4), conflict("-np"))]
    #[case::raw_np_glued_conflicts(4096, &["-np8"], Some(4), conflict("-np"))]
    #[case::ctx_too_large(
        3_000_000_000,
        &[],
        Some(4),
        ServerArgsError::CtxTooLarge { logging_marker: LoggingMarker::Eval, ctx: 3_000_000_000 },
    )]
    fn args_rejections(
        #[case] ctx_size: u32,
        #[case] raw: &[&str],
        #[case] slots: Option<u32>,
        #[case] expected: ServerArgsError,
    ) -> anyhow::Result<()> {
        let err = render(ctx_size, None, None, None, raw, slots)
            .err()
            .context("expected Err")?;
        assert_eq!(err, expected);
        Ok(())
    }

    fn thinking_launch(
        marker: LoggingMarker,
        raw: &[&str],
        envs: &[&str],
        client_envs: &[&str],
    ) -> anyhow::Result<ServerCommandBuilder> {
        let forwards = |envs: &[&str]| {
            envs.iter()
                .map(|env| EnvForward::try_new((*env).to_owned()))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(
            ServerCommandBuilder::new(test_llama_server(), "model.gguf", marker)
                .raw(raw.iter().map(|s| (*s).to_owned()).collect())
                .envs(forwards(envs)?)
                .client_config(Some(forwards(client_envs)?.into())),
        )
    }

    #[rstest]
    #[case::no_jinja(LoggingMarker::VlmEval, &["--no-jinja"], &[], &[],
        Some(ServerArgsError::NoJinja { logging_marker: LoggingMarker::VlmEval, flag: "--no-jinja" }))]
    #[case::no_jinja_underscored(LoggingMarker::VlmEval, &["--no_jinja"], &[], &[],
        Some(ServerArgsError::NoJinja { logging_marker: LoggingMarker::VlmEval, flag: "--no-jinja" }))]
    #[case::reasoning_on_vlm_eval(LoggingMarker::VlmEval, &["--reasoning", "on"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::VlmEval, flag: "--reasoning" }))]
    #[case::short_reasoning(LoggingMarker::VlmEval, &["-rea", "on"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::VlmEval, flag: "-rea" }))]
    #[case::budget_underscored(LoggingMarker::VlmEval, &["--reasoning_budget", "0"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::VlmEval, flag: "--reasoning-budget" }))]
    #[case::kwargs_underscored(LoggingMarker::VlmEval, &["--chat_template_kwargs", "{}"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::VlmEval, flag: "--chat-template-kwargs" }))]
    #[case::effort_underscored(LoggingMarker::VlmEval, &["--reasoning_effort", "low"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::VlmEval, flag: "--reasoning-effort" }))]
    #[case::no_jinja_elsewhere(LoggingMarker::VlThroughput, &["--no-jinja"], &[], &[], None)]
    #[case::kwargs_elsewhere(LoggingMarker::VlThroughput, &["--chat-template-kwargs", "{}"], &[], &[], None)]
    #[case::effort_elsewhere(LoggingMarker::EndToEndLatency, &["--reasoning-effort", "low"], &[], &[], None)]
    #[case::other_raw(LoggingMarker::VlmEval, &["--jinja", "--verbose"], &[], &[], None)]
    // A model taken from elsewhere, or a router's preset.
    #[case::hf_repo(LoggingMarker::VlmEval, &["-hf", "org/presets"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "-hf" }))]
    #[case::hf_repo_underscored(LoggingMarker::VlmEval, &["--hf_repo", "org/presets"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "--hf-repo" }))]
    #[case::docker_repo(LoggingMarker::VlmEval, &["-dr", "gemma3"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "-dr" }))]
    #[case::models_preset(LoggingMarker::VlmEval, &["--models-preset", "p.ini"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "--models-preset" }))]
    #[case::builtin_preset(LoggingMarker::VlmEval, &["--fim-qwen-7b-default"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "--fim-qwen-7b-default" }))]
    #[case::projector_url(LoggingMarker::VlmEval, &["--mmproj_url", "https://x/p.gguf"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "--mmproj-url" }))]
    #[case::projector(LoggingMarker::VlmEval, &["-mm", "other.gguf"], &[], &[],
        Some(ServerArgsError::ModelSourceFlag { logging_marker: LoggingMarker::VlmEval, flag: "-mm" }))]
    #[case::projector_placement(LoggingMarker::VlmEval, &["--mmproj-offload", "--no-mmproj"], &[], &[], None)]
    #[case::cell_env(LoggingMarker::VlmEval, &[], &["LLAMA_ARG_REASONING=on"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_REASONING" }))]
    #[case::lowercase_env(LoggingMarker::VlmEval, &[], &["llama_arg_reasoning=on"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_REASONING" }))]
    #[case::inherited_env(LoggingMarker::VlmEval, &[], &["LLAMA_ARG_NO_JINJA"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_NO_JINJA" }))]
    #[case::client_env(LoggingMarker::VlmEval, &[], &[], &["Llama_Arg_Think_Budget=0"],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_THINK_BUDGET" }))]
    #[case::hf_repo_env(LoggingMarker::VlmEval, &[], &["llama_arg_hf_repo=org/presets"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_HF_REPO" }))]
    #[case::projector_url_env(LoggingMarker::VlmEval, &[], &[], &["LLAMA_ARG_MMPROJ_URL=https://x/p.gguf"],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_MMPROJ_URL" }))]
    #[case::template_env(LoggingMarker::VlmEval, &[], &["LLAMA_ARG_CHAT_TEMPLATE=chatml"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_CHAT_TEMPLATE" }))]
    #[case::skip_chat_parsing_env(LoggingMarker::VlmEval, &[], &[], &["LLAMA_ARG_SKIP_CHAT_PARSING=1"],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_SKIP_CHAT_PARSING" }))]
    #[case::prefill_assistant_env(LoggingMarker::VlmEval, &[], &["LLAMA_ARG_NO_PREFILL_ASSISTANT=1"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::VlmEval, name: "LLAMA_ARG_NO_PREFILL_ASSISTANT" }))]
    #[case::other_envs(LoggingMarker::VlmEval, &[], &["GGML_METAL_NDEBUG=1"], &["HOME"], None)]
    // asr_eval runs with thinking off, so it refuses what the eval cells do.
    #[case::no_jinja_on_asr_eval(LoggingMarker::AsrEval, &["--no-jinja"], &[], &[],
        Some(ServerArgsError::NoJinja { logging_marker: LoggingMarker::AsrEval, flag: "--no-jinja" }))]
    #[case::reasoning_on_asr_eval(LoggingMarker::AsrEval, &["-rea", "on"], &[], &[],
        Some(ServerArgsError::RawReasoningFlag { logging_marker: LoggingMarker::AsrEval, flag: "-rea" }))]
    #[case::thinking_env_on_asr_eval(LoggingMarker::AsrEval, &[], &["LLAMA_ARG_THINK_BUDGET=-1"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_THINK_BUDGET" }))]
    // A template variable is hidden on every cell, so it keeps that error.
    #[case::template_file_env(LoggingMarker::AsrEval, &[], &["LLAMA_ARG_CHAT_TEMPLATE_FILE=/tmp/t.jinja"], &[],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_CHAT_TEMPLATE_FILE" }))]
    #[case::template_env(LoggingMarker::AsrEval, &[], &[], &["llama_arg_chat_template=chatml"],
        Some(ServerArgsError::HiddenEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_CHAT_TEMPLATE" }))]
    // And any other llama.cpp option variable, set by the cell or the client:
    // its entry and prompt profile own the decode, beams and slots.
    #[case::parallel_env(LoggingMarker::AsrEval, &[], &["LLAMA_ARG_N_PARALLEL=4"], &[],
        Some(ServerArgsError::AsrOwnedEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_N_PARALLEL".to_owned() }))]
    #[case::beam_env(LoggingMarker::AsrEval, &[], &["LLAMA_ARG_BEAM_SEARCH"], &[],
        Some(ServerArgsError::AsrOwnedEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_BEAM_SEARCH".to_owned() }))]
    #[case::sampler_env(LoggingMarker::AsrEval, &[], &[], &["LLAMA_ARG_TOP_K=40"],
        Some(ServerArgsError::AsrOwnedEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_ARG_TOP_K".to_owned() }))]
    // The template kwargs' older variable, which builds before b9360 read.
    #[case::template_kwargs_env(LoggingMarker::AsrEval, &[], &["LLAMA_CHAT_TEMPLATE_KWARGS={}"], &[],
        Some(ServerArgsError::AsrOwnedEnv { logging_marker: LoggingMarker::AsrEval, name: "LLAMA_CHAT_TEMPLATE_KWARGS".to_owned() }))]
    #[case::client_template_kwargs_env(LoggingMarker::AsrEval, &[], &[], &["llama_chat_template_kwargs"],
        Some(ServerArgsError::AsrOwnedEnv { logging_marker: LoggingMarker::AsrEval, name: "llama_chat_template_kwargs".to_owned() }))]
    #[case::option_env_elsewhere(LoggingMarker::VlThroughput, &[], &["LLAMA_ARG_N_PARALLEL=4", "LLAMA_ARG_BEAM_SEARCH"], &[], None)]
    fn thinking_controls_outside_the_typed_knobs_are_refused(
        #[case] marker: LoggingMarker,
        #[case] raw: &[&str],
        #[case] envs: &[&str],
        #[case] client_envs: &[&str],
        #[case] expected: Option<ServerArgsError>,
    ) -> anyhow::Result<()> {
        let got = thinking_launch(marker, raw, envs, client_envs)?.args();
        assert_eq!(got.err(), expected);
        Ok(())
    }

    #[rstest]
    #[case::no_jinja_on_eval(LoggingMarker::Eval, &["--no-jinja"], &[], &[])]
    #[case::kwargs_underscored(LoggingMarker::Eval, &["--chat_template_kwargs", "{}"], &[], &[])]
    #[case::effort_underscored(LoggingMarker::Eval, &["--reasoning_effort", "low"], &[], &[])]
    #[case::hf_repo_elsewhere(LoggingMarker::VlThroughput, &["-hf", "org/presets"], &[], &[])]
    #[case::hf_repo_underscored(LoggingMarker::EndToEndLatency, &["--hf_repo", "org/presets"], &[], &[])]
    #[case::docker_repo(LoggingMarker::Eval, &["-dr", "gemma3"], &[], &[])]
    #[case::models_preset(LoggingMarker::VlMaxMemory, &["--models-preset", "p.ini"], &[], &[])]
    #[case::builtin_preset(LoggingMarker::Eval, &["--fim-qwen-7b-default"], &[], &[])]
    #[case::projector(LoggingMarker::VlThroughput, &["-mm", "other.gguf"], &[], &[])]
    #[case::cell_env(LoggingMarker::Eval, &[], &["LLAMA_ARG_REASONING=on"], &[])]
    #[case::inherited_env_elsewhere(LoggingMarker::VlThroughput, &[], &["LLAMA_ARG_NO_JINJA"], &[])]
    #[case::client_env(LoggingMarker::VlSpecDecode, &[], &[], &["Llama_Arg_Think_Budget=0"])]
    #[case::hf_repo_env(LoggingMarker::VlThroughput, &[], &["llama_arg_hf_repo=org/presets"], &[])]
    #[case::projector_url_env(LoggingMarker::VlMaxMemory, &[], &[], &["LLAMA_ARG_MMPROJ_URL=https://x/p.gguf"])]
    #[case::template_env(LoggingMarker::Eval, &[], &["LLAMA_ARG_CHAT_TEMPLATE=chatml"], &[])]
    #[case::skip_chat_parsing_env(LoggingMarker::Eval, &[], &[], &["LLAMA_ARG_SKIP_CHAT_PARSING=1"])]
    #[case::prefill_assistant_env(LoggingMarker::VlThroughput, &[], &["LLAMA_ARG_NO_PREFILL_ASSISTANT=1"], &[])]
    // How the 24B plans turn an eval's thinking off.
    #[case::eval_template_kwargs(LoggingMarker::Eval, &["--backend-sampling", "--chat-template-kwargs", "{\"reasoning\":false}"], &[], &[])]
    #[case::eval_reasoning_off(LoggingMarker::Eval, &["--reasoning", "off"], &[], &[])]
    #[case::e2e_template_env(LoggingMarker::EndToEndLatency, &[], &["LLAMA_ARG_CHAT_TEMPLATE=chatml"], &[])]
    fn other_cells_launch_as_before(
        #[case] marker: LoggingMarker,
        #[case] raw: &[&str],
        #[case] envs: &[&str],
        #[case] client_envs: &[&str],
    ) -> anyhow::Result<()> {
        let launch = thinking_launch(marker, raw, envs, client_envs)?;
        launch.check_thinking_controls()?;
        let argv = launch.args()?;
        for token in raw {
            assert!(argv.iter().any(|t| t == token), "{token} in {argv:?}");
        }
        let command = launch.build(8080)?;
        let set: Vec<(String, Option<String>)> = command
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert!(
            set.iter().all(|(_, v)| v.is_some()),
            "nothing removed: {set:?}"
        );
        for forward in envs.iter().chain(client_envs) {
            if let Some((name, value)) = forward.split_once('=') {
                assert!(
                    set.iter()
                        .any(|(k, v)| k == name && v.as_deref() == Some(value)),
                    "{forward} reaches the server: {set:?}"
                );
            }
        }
        Ok(())
    }

    /// A vlm_eval or ASR cell is refused, the file found where the launch's own
    /// environment says: a forwarded `XDG_CONFIG_HOME`, or `HOME` without one,
    /// the cell's or the client's. Unix only: Windows reads `%APPDATA%`.
    #[cfg(unix)]
    #[rstest]
    #[case::cell_xdg(
        LoggingMarker::VlmEval,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "reasoning_budget = 0",
        Some("reasoning_budget")
    )]
    #[case::client_home(
        LoggingMarker::VlmEval,
        true,
        "HOME={}",
        ".config",
        "[*]\nLLAMA_ARG_JINJA = false",
        Some("LLAMA_ARG_JINJA")
    )]
    #[case::hf_repo(
        LoggingMarker::VlmEval,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "hf-repo = org/presets",
        Some("hf-repo")
    )]
    #[case::builtin_preset(
        LoggingMarker::VlmEval,
        true,
        "XDG_CONFIG_HOME={}",
        "",
        "[*]\ngpt-oss-120b-default = true",
        Some("gpt-oss-120b-default")
    )]
    #[case::elsewhere(
        LoggingMarker::VlThroughput,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "reasoning_budget = 0",
        None
    )]
    #[case::text_eval(
        LoggingMarker::Eval,
        true,
        "HOME={}",
        ".config",
        "[*]\nLLAMA_ARG_JINJA = false",
        None
    )]
    #[case::other_keys(
        LoggingMarker::VlmEval,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "threads = 4",
        None
    )]
    // The keys a recipe owns are the other cells' to set.
    #[case::template_file_elsewhere(
        LoggingMarker::VlThroughput,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "parallel = 4\nLLAMA_ARG_BEAM_SEARCH = 3",
        None
    )]
    // A template key is hidden on every cell, asr_eval's included.
    #[case::template_file_on_asr_eval(
        LoggingMarker::AsrEval,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "[*]\nchat_template_file = t.jinja",
        Some("chat_template_file")
    )]
    #[case::template_on_asr_eval(
        LoggingMarker::AsrEval,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "--chat-template = chatml",
        Some("--chat-template")
    )]
    // An ASR cell's recipe owns its template and thinking, not its samplers.
    #[case::asr_template(
        LoggingMarker::AsrThroughput,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "chat-template = chatml",
        Some("chat-template")
    )]
    #[case::asr_reasoning(
        LoggingMarker::AsrMaxMemory,
        true,
        "HOME={}",
        ".config",
        "[*]\nLLAMA_ARG_REASONING = on",
        Some("LLAMA_ARG_REASONING")
    )]
    #[case::asr_sampler(
        LoggingMarker::AsrThroughput,
        false,
        "XDG_CONFIG_HOME={}",
        "",
        "temp = 0.7",
        None
    )]
    fn a_config_file_that_sets_a_control_is_refused(
        #[case] marker: LoggingMarker,
        #[case] from_client: bool,
        #[case] forward: &str,
        #[case] config_dir: &str,
        #[case] text: &str,
        #[case] key: Option<&str>,
    ) -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let dir = root.path().join(config_dir).join("llama.cpp");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.ini");
        std::fs::write(&path, text)?;
        let forward = forward.replace("{}", &root.path().display().to_string());
        // An empty `XDG_CONFIG_HOME` is unset to llama.cpp, so `HOME` decides.
        let envs = [forward.as_str(), "XDG_CONFIG_HOME="];
        let envs = if forward.starts_with("HOME") {
            &envs[..]
        } else {
            &envs[..1]
        };
        let launch = if from_client {
            thinking_launch(marker, &[], &[], envs)?
        } else {
            thinking_launch(marker, &[], envs, &[])?
        };
        let expected = key.map(|key| ServerArgsError::HiddenConfig {
            logging_marker: marker,
            path: path.clone(),
            key: key.to_owned(),
        });
        assert_eq!(launch.check_thinking_controls().err(), expected);
        assert_eq!(launch.build(8080).err().as_ref(), expected.as_ref());
        Ok(())
    }

    /// On asr_eval a config file's key for anything its recipe owns (a vlm_eval's
    /// recipe keys, its beams and slots, any option variable) is refused, in every
    /// spelling and section, under its own error; a thinking or template key keeps
    /// the hidden-key error.
    #[cfg(unix)]
    #[rstest]
    #[case::override_kv("override-kv = tokenizer.ggml.eot_token_id=int:2", Some("override-kv"))]
    #[case::parallel("np = 4", Some("np"))]
    #[case::beams("LLAMA_ARG_BEAM_SEARCH = 3", Some("LLAMA_ARG_BEAM_SEARCH"))]
    #[case::per_slot("kv-unified-per-slot = 1024", Some("kv-unified-per-slot"))]
    #[case::samplers("samplers = top_k;temperature", Some("samplers"))]
    #[case::sampling_seq("sampling_seq = kt", Some("sampling_seq"))]
    #[case::predict("[model]\nn-predict = 16", Some("n-predict"))]
    #[case::sampler("top_k = 40", Some("top_k"))]
    #[case::option_variable("LLAMA_ARG_CTX_SIZE = 4096", Some("LLAMA_ARG_CTX_SIZE"))]
    #[case::other("threads = 4\nctx-size = 4096", None)]
    fn an_asr_eval_config_key_is_refused(
        #[case] text: &str,
        #[case] key: Option<&str>,
    ) -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let dir = root.path().join("llama.cpp");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.ini");
        std::fs::write(&path, text)?;
        let forward = format!("XDG_CONFIG_HOME={}", root.path().display());
        let launch = thinking_launch(LoggingMarker::AsrEval, &[], &[forward.as_str()], &[])?;
        let expected = key.map(|key| ServerArgsError::AsrOwnedConfig {
            logging_marker: LoggingMarker::AsrEval,
            path: path.clone(),
            key: key.to_owned(),
        });
        assert_eq!(launch.check_thinking_controls().err(), expected);
        std::fs::write(&path, "reasoning = on")?;
        assert_eq!(
            launch.check_thinking_controls().err(),
            Some(ServerArgsError::HiddenConfig {
                logging_marker: LoggingMarker::AsrEval,
                path,
                key: "reasoning".to_owned(),
            })
        );
        Ok(())
    }

    #[test]
    fn a_vlm_evals_server_takes_no_llama_option_from_the_environment() -> anyhow::Result<()> {
        let config = tempfile::tempdir()?;
        let xdg = format!("XDG_CONFIG_HOME={}", config.path().display());
        let refused = thinking_launch(
            LoggingMarker::VlmEval,
            &[],
            &[xdg.as_str()],
            &["LLAMA_ARG_TOP_K=40"],
        )?
        .check_thinking_controls()
        .err();
        assert_eq!(
            refused,
            Some(ServerArgsError::LlamaArgEnv {
                logging_marker: LoggingMarker::VlmEval,
                name: "LLAMA_ARG_TOP_K".to_owned(),
            })
        );
        thinking_launch(
            LoggingMarker::VlThroughput,
            &[],
            &[xdg.as_str()],
            &["LLAMA_ARG_TOP_K=40"],
        )?
        .check_thinking_controls()?;
        // An asr_eval's entry and prompt profile own them likewise.
        assert_eq!(
            thinking_launch(
                LoggingMarker::AsrEval,
                &[],
                &[xdg.as_str()],
                &["LLAMA_ARG_TOP_K=40"],
            )?
            .check_thinking_controls()
            .err(),
            Some(ServerArgsError::AsrOwnedEnv {
                logging_marker: LoggingMarker::AsrEval,
                name: "LLAMA_ARG_TOP_K".to_owned(),
            })
        );
        Ok(())
    }

    /// Only on vlm_eval: another cell's server inherits its environment untouched.
    #[test]
    fn the_server_inherits_no_thinking_environment() -> anyhow::Result<()> {
        let config = tempfile::tempdir()?;
        let xdg = format!("XDG_CONFIG_HOME={}", config.path().display());
        let command =
            thinking_launch(LoggingMarker::VlmEval, &[], &[xdg.as_str()], &[])?.build(8080)?;
        let removed: Vec<_> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .filter_map(|(name, _)| name.to_str())
            .collect();
        for name in thinking_controls::HIDDEN_ENVS {
            assert!(removed.contains(name), "{name} in {removed:?}");
        }
        let elsewhere =
            thinking_launch(LoggingMarker::VlThroughput, &[], &[xdg.as_str()], &[])?.build(8080)?;
        let removed_elsewhere: Vec<_> = elsewhere
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .filter_map(|(name, _)| name.to_str())
            .collect();
        assert!(removed_elsewhere.is_empty(), "{removed_elsewhere:?}");
        Ok(())
    }

    /// The derived context is one slot's budget times the slots. A product that
    /// does not fit is refused by name rather than wrapped into a context
    /// llama-server would silently accept.
    #[rstest]
    #[case::reference(768, 1, Some(768))]
    #[case::batched(768, 4, Some(3072))]
    #[case::overflow_is_refused(u32::MAX, 2, None)]
    fn the_context_scales_with_the_slots(
        #[case] per_slot: u32,
        #[case] slots: u32,
        #[case] expected: Option<u32>,
    ) -> anyhow::Result<()> {
        match (ctx_across_slots(per_slot, slots), expected) {
            (Ok(ctx), Some(want)) => assert_eq!(ctx, want),
            (Err(err), None) => assert!(format!("{err:#}").contains("overflows"), "{err:#}"),
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    mod audio_chat_timing {
        use std::{
            io::{BufRead, BufReader, Read, Write},
            net::TcpListener,
            thread,
            time::Duration,
        };

        use super::*;

        /// What one write of this crate's log costs on a slow `-v` terminal.
        const SLOW_LOG: Duration = Duration::from_millis(200);

        struct SlowLog;

        impl tracing::Subscriber for SlowLog {
            fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
                // `post_chat` logs from the crate it moved to.
                metadata.target().starts_with(env!("CARGO_CRATE_NAME"))
                    || metadata.target().starts_with("pipette_completions")
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, _: &tracing::Event<'_>) {
                std::thread::sleep(SLOW_LOG);
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }

        /// Answers one request after reading it whole.
        fn serve_once(
            status: &'static str,
            body: &'static str,
        ) -> anyhow::Result<(String, thread::JoinHandle<std::io::Result<()>>)> {
            let listener =
                TcpListener::bind(("127.0.0.1", 0)).context("failed to bind test listener")?;
            let url = format!("http://{}", listener.local_addr()?);
            let server = thread::spawn(move || -> std::io::Result<()> {
                let (mut stream, _) = listener.accept()?;
                let mut reader = BufReader::new(stream.try_clone()?);
                let length = (&mut reader)
                    .lines()
                    .map_while(Result::ok)
                    .take_while(|line| !line.is_empty())
                    .filter_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .and_then(|n| n.parse::<u64>().ok())
                    })
                    .last()
                    .unwrap_or(0);
                std::io::copy(&mut (&mut reader).take(length), &mut std::io::sink())?;
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            });
            Ok((url, server))
        }

        fn join(server: thread::JoinHandle<std::io::Result<()>>) -> anyhow::Result<()> {
            server
                .join()
                .map_err(|_| anyhow::anyhow!("the test server panicked"))??;
            Ok(())
        }

        /// The debug line is written after the reply's arrival is stamped, so a
        /// slow log stays out of the whole request's time.
        #[test]
        fn a_slow_debug_log_stays_out_of_the_request() -> anyhow::Result<()> {
            const REPLY: &str = r#"{"choices":[{"finish_reason":"length"}],"timings":{"prompt_n":10,"prompt_ms":1.0,"predicted_n":4,"predicted_ms":2.0}}"#;
            let (url, server) = serve_once("200 OK", REPLY)?;
            let http = HttpClient::loopback("pipette", Duration::from_secs(5))?;
            let generation = tracing::subscriber::with_default(SlowLog, || {
                audio_chat(
                    &http,
                    &url,
                    None,
                    "Transcribe this audio.",
                    ContentOrder::InstructionThenClip,
                    Some("QUJD"),
                    4,
                )
            })?;
            join(server)?;
            anyhow::ensure!(
                generation.end_ms < SLOW_LOG.as_secs_f64() * 1000.0,
                "{generation:?}"
            );
            Ok(())
        }

        /// A clip past a pinned context fails with llama-server's own message.
        #[test]
        fn a_refusal_keeps_the_servers_message() -> anyhow::Result<()> {
            const REFUSAL: &str = r#"{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error"}}"#;
            let (url, server) = serve_once("400 Bad Request", REFUSAL)?;
            let http = HttpClient::loopback("pipette", Duration::from_secs(5))?;
            let err = audio_chat(
                &http,
                &url,
                None,
                "Transcribe this audio.",
                ContentOrder::InstructionThenClip,
                Some("QUJD"),
                4,
            )
            .err()
            .context("a refused request was timed")?;
            join(server)?;
            let text = format!("{err:#}");
            assert!(text.contains("exceed_context_size_error"), "{text}");
            Ok(())
        }
    }
}
