//! The corpus benchmark's shared contract: `vl_spec_decode`, one VL model over
//! a real image + text corpus with its drafter on or off. One traversal, one
//! row; comparing a pair of rows is a join elsewhere.
//!
//! Rows from different engines are comparable only if counted and reduced the
//! same way, so that lives here (and the traversal in
//! `pipette_vl_spec_decode_corpus::turn`, kept out of this crate because
//! `pipette-android` must not link its file and HTTP code). Each engine adds
//! only its request envelope.
//!
//! See `docs/methodology/vl-spec-decode.md`.

use std::io::BufRead;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_plan_types::{BenchmarkResultData, Sha256};

/// Attempts per turn before the traversal gives up.
///
/// A greedy, cache-free turn gives the same answer on a repeat and the clock
/// restarts per attempt, so a retry costs nothing; without one, a single blip
/// loses a traversal that may have an hour behind it. Matches the reference
/// MMSpec driver.
pub const TURN_ATTEMPTS: u32 = 3;

/// Whether a traversal's drafter is on, as
/// [`resolve_speculation`](pipette_plan_types::resolve_speculation) resolved
/// the cell's `speculative_enabled`. A run with it off that drafted, or on that
/// drafted nothing, is refused as the other condition under the wrong name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speculation {
    Off,
    On,
}

impl Speculation {
    pub fn from_enabled(enabled: bool) -> Self {
        if enabled { Self::On } else { Self::Off }
    }

    /// For log lines: the benchmark and the condition it ran under.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "vl_spec_decode (speculation off)",
            Self::On => "vl_spec_decode (speculation on)",
        }
    }
}

/// What one turn produced, in the terms the wire counts.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRecord {
    /// The runtime's own prefill time, the reference harness's TTFT.
    pub prompt_ms: f64,
    /// The runtime's own generation time.
    pub predicted_ms: f64,
    /// Client clock: send to first token.
    pub ttft_ms: f64,
    /// Client clock: first token to last token.
    pub decode_ms: f64,
    /// Client clock: send to the end of the stream. Its excess over
    /// `ttft + decode` is overhead the runtime never sees.
    pub e2e_ms: f64,
    pub generated_tokens: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    /// Decode-time target forwards as the runtime counted them, the prefill's
    /// excluded: the accept length's denominator. Not on the wire. Zero on a
    /// one-token reply or a build that counts none.
    pub counted_forwards: u64,
    /// Prompt tokens the runtime answered from its cache (`cache_n`). Not on
    /// the wire: [`Self::validate`] requires zero.
    pub cached_tokens: u64,
    /// The model stopped on its own end-of-sequence (`finish_reason` of
    /// `stop`). Not on the wire.
    pub ended_on_eos: bool,
    /// Whether the runtime stopped at a length limit (`finish_reason` of
    /// `length`): `max_tokens`, or the context. Not on the wire.
    pub ended_on_length: bool,
    /// The reply as the model emitted it, for the greedy-identity comparison.
    pub text: String,
}

impl TurnRecord {
    /// Nothing ties `accepted` to `generated`: the runtimes count accepted
    /// drafts before the stop token discards the rest of the run, so
    /// `generated - accepted` is no forward count. A drafting turn must carry
    /// the runtime's own, and is refused by name without one.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.generated_tokens > 0,
            "the turn emitted no tokens, so it measured nothing"
        );
        anyhow::ensure!(
            self.accepted_tokens <= self.drafted_tokens,
            "the runtime reported {} accepted of {} drafted tokens",
            self.accepted_tokens,
            self.drafted_tokens
        );
        anyhow::ensure!(
            self.drafted_tokens == 0 || self.counted_forwards > 0,
            "the runtime drafted {} tokens and reported no forward count: the accept \
             length needs `target_n_steps` from the llama.cpp fork, `draft_rounds` from \
             mlx_vlm.server or `spec_verify_ct` from SGLang, and this build sent none",
            self.drafted_tokens
        );
        // Every forward commits a token, so more forwards than tokens is a
        // count shared with other requests (`mlx_vlm.server` at batch > 1).
        anyhow::ensure!(
            self.counted_forwards <= self.decode_tokens(),
            "the runtime counted {} decode forwards for {} decode tokens: the count is \
             not this request's own",
            self.counted_forwards,
            self.decode_tokens()
        );
        // Every turn prefills its whole conversation, as the reference
        // harness's `cache_prompt: false` does: a reused prefix would make a
        // later turn's TTFT depend on what the server still held.
        anyhow::ensure!(
            self.cached_tokens == 0,
            "the runtime answered {} prompt tokens from its cache: prompt caching \
             must be off for every turn",
            self.cached_tokens
        );
        Ok(())
    }

    /// `llama-server` stops a reply that runs out of context with `length`, as
    /// it stops one at `max_tokens`, and SGLang clips `max_tokens` to what
    /// fits, so a `length` stop short of the `max_tokens` the turn was sent
    /// with is a truncated reply.
    pub fn check_length(&self, max_tokens: u32) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.ended_on_length || self.generated_tokens >= u64::from(max_tokens),
            "the runtime stopped the reply for its length at {} tokens, short of the \
             {max_tokens} max_tokens it was sent with: the context cut it. Size the \
             context for the corpus's longest conversation",
            self.generated_tokens
        );
        Ok(())
    }

    /// Tokens committed per decode forward, the prefill out of both terms
    /// (SGLang's own `spec_accept_length` keeps the prefill's token, so it
    /// reads higher). A one-token reply reads zero. Only meaningful after
    /// [`Self::validate`].
    pub fn accept_length(&self) -> f64 {
        if self.counted_forwards == 0 {
            return 0.0;
        }
        self.decode_tokens() as f64 / self.counted_forwards as f64
    }

    /// Every token but the first, which the prefill sampled before any draft
    /// existed. All three runtimes count that one.
    pub fn decode_tokens(&self) -> u64 {
        self.generated_tokens.saturating_sub(1)
    }
}

/// One traversal, summed.
#[derive(Debug, Default, Clone, PartialEq)]
struct Totals {
    turns: u64,
    prompt_ms: f64,
    predicted_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    e2e_ms: f64,
    generated_tokens: u64,
    decode_tokens: u64,
    drafted_tokens: u64,
    accepted_tokens: u64,
}

impl Totals {
    /// Nothing here branches on purpose: a second implementation can disagree
    /// with this one only about what it observed, never about what it did
    /// with an observation.
    fn add(&mut self, turn: &TurnRecord) {
        self.turns += 1;
        self.prompt_ms += turn.prompt_ms;
        self.predicted_ms += turn.predicted_ms;
        self.ttft_ms += turn.ttft_ms;
        self.decode_ms += turn.decode_ms;
        self.e2e_ms += turn.e2e_ms;
        self.generated_tokens += turn.generated_tokens;
        self.decode_tokens += turn.decode_tokens();
        self.drafted_tokens += turn.drafted_tokens;
        self.accepted_tokens += turn.accepted_tokens;
    }

    fn of(turns: &[TurnRecord]) -> Self {
        turns.iter().fold(Self::default(), |mut totals, turn| {
            totals.add(turn);
            totals
        })
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// One streamed reply: the text, the runtime's timings, and the three
/// client-clock spans around it.
#[derive(Debug)]
pub struct StreamedTurn {
    pub content: String,
    pub timings: ChatTimings,
    /// Off the terminal chunk: `stop` for the model's own end-of-sequence,
    /// `length` for `max_tokens` or the context.
    pub finish_reason: Option<String>,
    pub ttft: Duration,
    pub decode: Duration,
    pub e2e: Duration,
}

impl StreamedTurn {
    pub fn into_record(self) -> TurnRecord {
        TurnRecord {
            prompt_ms: self.timings.prompt_ms,
            predicted_ms: self.timings.predicted_ms,
            ttft_ms: millis(self.ttft),
            decode_ms: millis(self.decode),
            e2e_ms: millis(self.e2e),
            generated_tokens: u64::from(self.timings.predicted_n),
            drafted_tokens: u64::from(self.timings.draft_n),
            accepted_tokens: u64::from(self.timings.draft_n_accepted),
            counted_forwards: u64::from(self.timings.counted_forwards),
            cached_tokens: u64::from(self.timings.cache_n),
            ended_on_eos: self.finish_reason.as_deref() == Some("stop"),
            ended_on_length: self.finish_reason.as_deref() == Some("length"),
            text: self.content,
        }
    }
}

/// A runtime's per-request `timings`, once it is complete.
///
/// `mlx_vlm.server` mirrors `llama-server`'s field names, so one reader serves
/// both. SGLang reports the counts in its own shape and no clock, see
/// [`TimingsSource::ClientClock`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTimings {
    pub prompt_ms: f64,
    pub predicted_n: u32,
    pub predicted_ms: f64,
    pub draft_n: u32,
    pub draft_n_accepted: u32,
    /// Decode-time target forwards the runtime counted, see
    /// [`TurnRecord::counted_forwards`]; zero where the build counts none.
    pub counted_forwards: u32,
    /// Prompt tokens served from the runtime's cache. Required from the
    /// runtime servers ([`MaybeTimings::complete`]); on SGLang absence means
    /// zero ([`StreamUsage`]).
    pub cache_n: u32,
}

/// Where a stream's [`ChatTimings`] come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingsSource {
    /// The runtime's own `timings` object (`llama-server`, `mlx_vlm.server`):
    /// its prefill and generation clock, its token and draft counts.
    Runtime,
    /// [`Self::Runtime`] from a `llama-server` with a block drafter at this
    /// draft length: it sends no step count, so forwards are recovered, see
    /// [`ChatTimings::recover_forwards`].
    RuntimeFixedDraft(u32),
    /// The OpenAI `usage` block plus SGLang's `sglext.spec_tokens_details`
    /// for the counts, and the client's stamps for the times (`prompt_ms` to
    /// the first token, `predicted_ms` first to last), since SGLang reports no
    /// per-request clock. They compare within SGLang only, not with a
    /// llama.cpp or MLX row's runtime clock.
    ClientClock,
}

impl ChatTimings {
    /// Forwards for a turn with no count: a block drafter proposes
    /// `draft_length` tokens every step, so `draft_n / draft_length`. Exact
    /// on a reply that ended on its own. Near `max_tokens` the server shrinks
    /// the draft, so a cut reply reads the ceiling, a step or two short, and
    /// is logged. A finished reply with a partial block has no reading: the
    /// drafter proposes no fixed block, or the draft length exceeds it.
    fn recover_forwards(mut self, draft_length: u32, ended_on_eos: bool) -> anyhow::Result<Self> {
        if self.counted_forwards > 0 || self.draft_n == 0 || draft_length == 0 {
            return Ok(self);
        }
        if !self.draft_n.is_multiple_of(draft_length) {
            anyhow::ensure!(
                !ended_on_eos,
                "the runtime reported no forward count and its {} drafted tokens are not \
                 whole blocks of {draft_length}: this drafter does not propose a fixed \
                 block every step, so the accept length needs a build that counts its \
                 verification steps",
                self.draft_n
            );
            tracing::warn!(
                drafted_tokens = self.draft_n,
                draft_length,
                "a reply cut at max_tokens drafted tokens in partial blocks: its forward count \
                 is the ceiling, short by the steps the budget shrank"
            );
        }
        self.counted_forwards = self.draft_n.div_ceil(draft_length);
        Ok(self)
    }
}

/// A `timings` object as it arrives, which may be a partial.
///
/// `mlx_vlm.server` puts a progress-only `timings` on every delta and the
/// complete one on the final usage chunk, so every field is optional and
/// [`Self::complete`] drops a partial instead of failing the stream.
///
/// The speculation counters are `Option` because `mlx_vlm.server` sends them
/// as explicit `null` with no drafter, which a `#[serde(default)]` `u32`
/// rejects. They default to zero; `cache_n` does not, since a turn is refused
/// on it and both servers always send it.
#[derive(Debug, Deserialize)]
struct MaybeTimings {
    prompt_ms: Option<f64>,
    predicted_n: Option<u32>,
    predicted_ms: Option<f64>,
    #[serde(default)]
    draft_n: Option<u32>,
    #[serde(default)]
    draft_n_accepted: Option<u32>,
    /// `mlx_vlm.server`'s verification rounds; absent on `llama-server`.
    #[serde(default)]
    draft_rounds: Option<u32>,
    /// `llama-server`'s count of every target sampling step, the prefill's
    /// included. Only builds that expose it send it.
    #[serde(default)]
    target_n_steps: Option<u32>,
    /// `llama-server`'s verification steps, which upstream logs rather than
    /// sends. One short of the forwards per decode step that drafted nothing.
    #[serde(default)]
    draft_n_verif_steps: Option<u32>,
    #[serde(default)]
    cache_n: Option<u32>,
}

impl MaybeTimings {
    fn complete(self) -> Option<ChatTimings> {
        Some(ChatTimings {
            prompt_ms: self.prompt_ms?,
            predicted_n: self.predicted_n?,
            predicted_ms: self.predicted_ms?,
            draft_n: self.draft_n.unwrap_or(0),
            draft_n_accepted: self.draft_n_accepted.unwrap_or(0),
            // `target_n_steps` includes the prefill's forward; the other two
            // count decode-time verifications already.
            counted_forwards: self
                .target_n_steps
                .map(|steps| steps.saturating_sub(1))
                .or(self.draft_n_verif_steps)
                .or(self.draft_rounds)
                .unwrap_or(0),
            cache_n: self.cache_n?,
        })
    }
}

/// The OpenAI `usage` block on the trailing chunk of an SGLang stream
/// (`stream_options.include_usage`). Every field optional: the other servers
/// put a `usage` of their own shape on the same chunk.
///
/// Absent `prompt_tokens_details` reads as zero, as SGLang omits it rather
/// than send a zero. It reports cached tokens only with
/// `--enable-cache-report`, so the corpus launch passes that flag.
#[derive(Debug, Clone, Deserialize)]
struct StreamUsage {
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

/// SGLang's per-request speculation counters, under `sglext` on the chunk
/// that carries the `finish_reason` (`return_spec_tokens_details`). Absent
/// with speculation off, where zero drafts is the truth.
#[derive(Debug, Clone, Deserialize)]
struct SglExt {
    #[serde(default)]
    spec_tokens_details: Option<SpecTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
struct SpecTokensDetails {
    #[serde(default)]
    spec_num_proposed_drafts: u32,
    #[serde(default)]
    spec_num_correct_drafts: u32,
    #[serde(default)]
    spec_verify_ct: u32,
}

const NO_TIMINGS: &str = "the /v1/chat/completions stream carried no complete timings (the two \
     clocks, the token count and `cache_n`); the corpus benchmarks need the \
     runtime's per-request timings to count tokens. On `mlx_vlm.server` that \
     needs `stream_options.include_usage`";

/// Read one `/v1/chat/completions` SSE stream to its end, stamping `now()` at
/// the first token, the last token, and the end of the stream.
///
/// `started` is when the request was sent; `now` is injected for tests. A
/// token is a delta with content or the terminal chunk, so an EOS-only reply
/// has both stamps there. The lines come off [`crate::sse::payloads`], so a
/// body that ends before `[DONE]` is a cut stream rather than a short answer.
pub fn read_chat_stream(
    reader: impl BufRead,
    started: Instant,
    mut now: impl FnMut() -> Instant,
    source: TimingsSource,
) -> anyhow::Result<StreamedTurn> {
    let mut content = String::new();
    let mut timings = None;
    let mut usage: Option<StreamUsage> = None;
    let mut spec: Option<SpecTokensDetails> = None;
    let mut first_token: Option<Instant> = None;
    let mut last_token: Option<Instant> = None;
    let mut finish_reason: Option<String> = None;
    crate::sse::payloads(reader.lines())
        .map(|data| data.context("reading the /v1/chat/completions stream"))
        .try_for_each(|data| -> anyhow::Result<()> {
            let data = data?;
            let chunk: ChatChunk = serde_json::from_str(&data)
                .with_context(|| format!("failed to parse /v1/chat/completions chunk: {data}"))?;
            if let Some(error) = chunk.error {
                anyhow::bail!("/v1/chat/completions failed mid-stream: {error}");
            }
            let choice = chunk.choices.first();
            let delta = choice
                .and_then(|c| c.delta.content.as_deref())
                .unwrap_or_default();
            let finish = choice.and_then(|c| c.finish_reason.as_deref());
            if !delta.is_empty() || finish.is_some() {
                let at = now();
                first_token.get_or_insert(at);
                last_token = Some(at);
            }
            content.push_str(delta);
            if let Some(reason) = finish {
                finish_reason = Some(reason.to_owned());
            }
            // A partial `timings` is progress, not the reading: keep the last
            // complete one and ignore the rest.
            if let Some(complete) = chunk.timings.and_then(MaybeTimings::complete) {
                timings = Some(complete);
            }
            if let Some(chunk_usage) = chunk.usage {
                usage = Some(chunk_usage);
            }
            if let Some(details) = chunk.sglext.and_then(|ext| ext.spec_tokens_details) {
                spec = Some(details);
            }
            Ok(())
        })?;
    let finished = now();
    let first = first_token.context("the /v1/chat/completions stream ended without a token")?;
    let last = last_token.unwrap_or(first);
    let ttft = first.duration_since(started);
    let decode = last.duration_since(first);
    let timings = match source {
        TimingsSource::Runtime => timings.context(NO_TIMINGS)?,
        TimingsSource::RuntimeFixedDraft(draft_length) => timings
            .context(NO_TIMINGS)?
            .recover_forwards(draft_length, finish_reason.as_deref() == Some("stop"))?,
        TimingsSource::ClientClock => {
            let usage = usage
                .and_then(|usage| {
                    Some((
                        usage.completion_tokens?,
                        usage
                            .prompt_tokens_details
                            .map_or(0, |details| details.cached_tokens),
                    ))
                })
                .context(
                    "the /v1/chat/completions stream carried no usage block with a \
                     completion_tokens count; the corpus benchmarks need the token \
                     counts, which on SGLang arrive with `stream_options.include_usage`",
                )?;
            let spec = spec.unwrap_or(SpecTokensDetails {
                spec_num_proposed_drafts: 0,
                spec_num_correct_drafts: 0,
                spec_verify_ct: 0,
            });
            let (completion_tokens, cached_tokens) = usage;
            ChatTimings {
                prompt_ms: millis(ttft),
                predicted_n: completion_tokens,
                predicted_ms: millis(decode),
                draft_n: spec.spec_num_proposed_drafts,
                draft_n_accepted: spec.spec_num_correct_drafts,
                counted_forwards: spec.spec_verify_ct,
                cache_n: cached_tokens,
            }
        }
    };
    Ok(StreamedTurn {
        content,
        timings,
        finish_reason,
        ttft,
        decode,
        e2e: finished.duration_since(started),
    })
}

/// One `chat.completion.chunk` off the stream. An error after the headers
/// arrives as a chunk with `error` instead of `choices`.
#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    timings: Option<MaybeTimings>,
    #[serde(default)]
    usage: Option<StreamUsage>,
    #[serde(default)]
    sglext: Option<SglExt>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Reduce one traversal to the counters the wire carries, refusing anything
/// the server would reject or divide by, and anything that says the row is
/// not the condition it claims to be.
///
/// Checked here rather than left to the gateway: a `400` at the end of an
/// hour-long run throws the measurement away.
pub fn counters(
    speculation: Speculation,
    turns: &[TurnRecord],
) -> anyhow::Result<BenchmarkResultData> {
    let totals = Totals::of(turns);
    anyhow::ensure!(totals.turns > 0, "the traversal covered no turns");
    anyhow::ensure!(
        totals.predicted_ms > 0.0,
        "the traversal reported no generation time ({:.3}ms)",
        totals.predicted_ms
    );
    anyhow::ensure!(
        totals.generated_tokens > 0,
        "the traversal emitted no tokens"
    );
    anyhow::ensure!(
        totals.e2e_ms > 0.0,
        "the traversal took no time on the client clock ({:.3}ms)",
        totals.e2e_ms
    );
    let tok_s = totals.generated_tokens as f64 / totals.predicted_ms * 1000.0;
    let (drafted, accepted, turn_accept_length_sum) = match speculation {
        Speculation::Off => {
            anyhow::ensure!(
                totals.drafted_tokens == 0 && totals.accepted_tokens == 0,
                "the run drafted {} tokens ({} accepted) with speculation off: a \
                 speculative measurement under the baseline's name",
                totals.drafted_tokens,
                totals.accepted_tokens
            );
            tracing::info!(
                cell = speculation.label(),
                turns = totals.turns,
                generated_tokens = totals.generated_tokens,
                tok_s,
                "totals"
            );
            (None, None, None)
        }
        Speculation::On => {
            // A turn may draft nothing, a traversal may not: that is an
            // unwired drafter, which would otherwise be stored as a bad one.
            anyhow::ensure!(
                totals.drafted_tokens > 0,
                "the traversal drafted no tokens: the drafter is not wired up. Check \
                 the server log for the drafter loading, and that the cell names the \
                 speculation settings this build needs"
            );
            // Per turn, then summed: over `turns` the request-mean convention,
            // which the pooled ratio below is not. Validating every turn here
            // is what keeps the per-turn denominator positive.
            let turn_accept_length_sum = turns
                .iter()
                .map(|turn| turn.validate().map(|()| turn.accept_length()))
                .sum::<anyhow::Result<f64>>()?;
            // The service divides by `generated - accepted` unguarded, which a
            // traversal of one-word replies can drive to zero.
            anyhow::ensure!(
                totals.generated_tokens > totals.accepted_tokens,
                "the traversal accepted every one of its {} generated tokens: the pooled \
                 accept length the service derives from the counters has no forward \
                 count to divide by",
                totals.generated_tokens
            );
            let decode_forwards: u64 = turns.iter().map(|t| t.counted_forwards).sum();
            let accept_length = totals.decode_tokens as f64 / decode_forwards as f64;
            tracing::info!(
                cell = speculation.label(),
                turns = totals.turns,
                generated_tokens = totals.generated_tokens,
                decode_forwards,
                accept_length_pooled = accept_length,
                accept_length_per_request = turn_accept_length_sum / totals.turns as f64,
                tok_s,
                "totals"
            );
            (
                Some(totals.drafted_tokens),
                Some(totals.accepted_tokens),
                Some(turn_accept_length_sum),
            )
        }
    };
    Ok(BenchmarkResultData::VlSpecDecode {
        vl_spec_decode_turns: u32::try_from(totals.turns).context("turn count exceeds u32")?,
        vl_spec_decode_prompt_ms: totals.prompt_ms,
        vl_spec_decode_predicted_ms: totals.predicted_ms,
        vl_spec_decode_ttft_ms: totals.ttft_ms,
        vl_spec_decode_decode_ms: totals.decode_ms,
        vl_spec_decode_e2e_ms: totals.e2e_ms,
        vl_spec_decode_generated_tokens: totals.generated_tokens,
        vl_spec_decode_drafted_tokens: drafted,
        vl_spec_decode_accepted_tokens: accepted,
        vl_spec_decode_turn_accept_length_sum: turn_accept_length_sum,
    })
}

/// sha256 over the traversal's replies, in order: per turn, the reply's own
/// sha256, a tab, its token count, a newline.
///
/// Lets two rows be compared for greedy identity without the text leaving the
/// device. Per turn so turn boundaries count, and so the per-turn record
/// (which carries the same hashes) says which turn diverged.
pub fn output_digest(turns: &[TurnRecord]) -> anyhow::Result<Sha256> {
    let hasher = turns
        .iter()
        .try_fold(Sha256Hasher::new(), |mut hasher, turn| {
            hasher.update(text_sha256(&turn.text)?.as_ref().as_bytes());
            hasher.update(b"\t");
            hasher.update(turn.generated_tokens.to_string().as_bytes());
            hasher.update(b"\n");
            anyhow::Ok(hasher)
        })?;
    sha256_of(hasher)
}

pub fn text_sha256(text: &str) -> anyhow::Result<Sha256> {
    let mut hasher = Sha256Hasher::new();
    hasher.update(text.as_bytes());
    sha256_of(hasher)
}

/// Fallible only in the type: 32 bytes always render as 64 lowercase hex.
fn sha256_of(hasher: Sha256Hasher) -> anyhow::Result<Sha256> {
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(Sha256::try_new(hex)?)
}

/// How far a traversal has got: samples and turns done, and the two sums
/// the progress line derives an accept length from.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub samples: usize,
    pub turns: usize,
    pub generated_tokens: u64,
    pub decode_tokens: u64,
    pub decode_forwards: u64,
}

impl Progress {
    pub fn add_sample(&mut self, turns: &[TurnRecord]) {
        self.samples += 1;
        self.turns += turns.len();
        self.generated_tokens += turns.iter().map(|t| t.generated_tokens).sum::<u64>();
        self.decode_tokens += turns.iter().map(TurnRecord::decode_tokens).sum::<u64>();
        self.decode_forwards += turns.iter().map(|t| t.counted_forwards).sum::<u64>();
    }
}

/// Every hundredth sample, so an hour-long traversal says how it is going
/// without writing a line per turn.
pub fn log_progress(speculation: Speculation, progress: Progress, total: usize) {
    let done = progress.samples;
    if !done.is_multiple_of(100) && done != total {
        return;
    }
    let generated = progress.generated_tokens;
    let forwards = progress.decode_forwards;
    match speculation {
        Speculation::Off => tracing::info!(
            cell = speculation.label(),
            completed = done,
            total,
            turns = progress.turns,
            generated_tokens = generated,
            "progress"
        ),
        Speculation::On => {
            let accept_length =
                (forwards > 0).then(|| progress.decode_tokens as f64 / forwards as f64);
            tracing::info!(
                cell = speculation.label(),
                completed = done,
                total,
                turns = progress.turns,
                accept_length,
                "progress"
            );
        }
    }
}

/// Why a turn failed, and whether repeating it could plausibly do better.
pub struct TurnFailure {
    pub error: anyhow::Error,
    pub retriable: bool,
}

/// Run `attempt` until it succeeds, fails unretriably, or runs out of
/// attempts. `sleep` is injected for tests; the backoff is the reference
/// driver's, five seconds times the attempt.
pub fn with_retries(
    attempt: &mut dyn FnMut(u32) -> std::result::Result<TurnRecord, TurnFailure>,
    sleep: &dyn Fn(Duration),
) -> anyhow::Result<TurnRecord> {
    (1..=TURN_ATTEMPTS)
        .find_map(|n| match attempt(n) {
            Ok(record) => Some(Ok(record)),
            Err(failure) if failure.retriable && n < TURN_ATTEMPTS => {
                let backoff = Duration::from_secs(5 * u64::from(n));
                tracing::warn!(
                    attempt = n,
                    attempts = TURN_ATTEMPTS,
                    retry_in = ?backoff,
                    error = %format_args!("{:#}", failure.error),
                    "turn attempt failed; retrying"
                );
                sleep(backoff);
                None
            }
            Err(failure) => Some(Err(failure.error)),
        })
        .unwrap_or_else(|| Err(anyhow::anyhow!("turn exhausted its attempts")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn turn(generated: u64, drafted: u64, accepted: u64, text: &str) -> TurnRecord {
        TurnRecord {
            prompt_ms: 10.0,
            predicted_ms: 100.0,
            ttft_ms: 12.0,
            decode_ms: 105.0,
            e2e_ms: 130.0,
            generated_tokens: generated,
            drafted_tokens: drafted,
            accepted_tokens: accepted,
            counted_forwards: 0,
            cached_tokens: 0,
            ended_on_eos: false,
            ended_on_length: false,
            text: text.to_string(),
        }
    }

    fn counted(
        generated: u64,
        drafted: u64,
        accepted: u64,
        counted_forwards: u64,
        text: &str,
    ) -> TurnRecord {
        TurnRecord {
            counted_forwards,
            ..turn(generated, drafted, accepted, text)
        }
    }

    /// `accepted` may exceed `generated`: the runtimes count drafts their stop
    /// token then discarded.
    #[rstest::rstest]
    #[case::no_speculation(8, 0, 0, 0)]
    #[case::no_speculation_on_a_counting_build(8, 0, 0, 7)]
    #[case::all_accepted(9, 8, 8, 1)]
    #[case::partly_accepted(12, 16, 8, 3)]
    #[case::one_word_reply_overcounted(2, 7, 6, 1)]
    #[case::accepted_past_generated(7, 12, 9, 3)]
    #[case::one_token_reply(1, 0, 0, 0)]
    fn a_truthful_turn_is_countable(
        #[case] generated: u64,
        #[case] drafted: u64,
        #[case] accepted: u64,
        #[case] counted_forwards: u64,
    ) {
        let record = TurnRecord {
            counted_forwards,
            ..turn(generated, drafted, accepted, "x")
        };
        assert!(record.validate().is_ok(), "{:?}", record.validate().err());
    }

    /// The numerator is `generated - 1` on every engine, so accepted drafts
    /// the stop token discarded never move it.
    #[rstest::rstest]
    #[case::sglang_one_step(2, 2, 2, 1, 1.0)]
    #[case::sglang_two_steps(15, 18, 14, 2, 7.0)]
    #[case::mlx_one_word_reply(2, 7, 2, 1, 1.0)]
    #[case::mlx_one_word_reply_overcounted(2, 7, 6, 1, 1.0)]
    #[case::mlx_three_rounds(10, 21, 8, 3, 3.0)]
    #[case::mlx_the_fleet_failure(4, 16, 5, 1, 3.0)]
    #[case::mlx_cut_at_max_tokens(9, 14, 10, 2, 4.0)]
    #[case::llamacpp_fork_two_steps(10, 16, 8, 2, 4.5)]
    #[case::one_token_reply(1, 0, 0, 0, 0.0)]
    fn a_turn_divides_by_the_forwards_the_runtime_counted(
        #[case] generated: u64,
        #[case] drafted: u64,
        #[case] accepted: u64,
        #[case] counted_forwards: u64,
        #[case] expected: f64,
    ) -> anyhow::Result<()> {
        let record = TurnRecord {
            counted_forwards,
            ..turn(generated, drafted, accepted, "x")
        };
        record.validate()?;
        assert!(
            (record.accept_length() - expected).abs() < 1e-9,
            "{}",
            record.accept_length()
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::drafted_without_a_count(12, 16, 8, 0, "reported no forward count")]
    #[case::more_forwards_than_tokens(4, 14, 3, 5, "not this request's own")]
    #[case::accepted_past_drafted(12, 4, 8, 3, "8 accepted of 4 drafted")]
    #[case::emitted_nothing(0, 0, 0, 0, "measured nothing")]
    fn a_contradictory_turn_is_refused(
        #[case] generated: u64,
        #[case] drafted: u64,
        #[case] accepted: u64,
        #[case] counted_forwards: u64,
        #[case] expected: &str,
    ) {
        let message = TurnRecord {
            counted_forwards,
            ..turn(generated, drafted, accepted, "x")
        }
        .validate()
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(message.contains(expected), "{message}");
    }

    #[test]
    fn a_turn_served_from_the_prompt_cache_is_refused() {
        let message = TurnRecord {
            cached_tokens: 312,
            ..turn(8, 0, 0, "x")
        }
        .validate()
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(
            message.contains("312 prompt tokens from its cache"),
            "{message}"
        );
    }

    #[test]
    fn a_speculative_traversal_pools_its_counters() -> anyhow::Result<()> {
        let turns = vec![counted(12, 16, 8, 3, "a"), counted(9, 8, 8, 1, "b")];
        let BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_turns,
            vl_spec_decode_prompt_ms,
            vl_spec_decode_predicted_ms,
            vl_spec_decode_ttft_ms,
            vl_spec_decode_decode_ms,
            vl_spec_decode_e2e_ms,
            vl_spec_decode_generated_tokens,
            vl_spec_decode_drafted_tokens,
            vl_spec_decode_accepted_tokens,
            vl_spec_decode_turn_accept_length_sum,
        } = counters(Speculation::On, &turns)?
        else {
            anyhow::bail!("expected a vl_spec_decode result");
        };
        assert_eq!(vl_spec_decode_turns, 2);
        assert_eq!(vl_spec_decode_generated_tokens, 21);
        assert_eq!(vl_spec_decode_drafted_tokens, Some(24));
        assert_eq!(vl_spec_decode_accepted_tokens, Some(16));
        // The service's pooled accept length is 21 / (21 - 16) = 4.2.
        assert_eq!(vl_spec_decode_generated_tokens - 16, 5);
        // The per-request one is 11 / 3 and 8 / 1, a mean of 5.83: hence the
        // separate sum.
        let sum = vl_spec_decode_turn_accept_length_sum.context("an accept length sum")?;
        assert!((sum - (11.0 / 3.0 + 8.0)).abs() < 1e-9, "{sum}");
        assert_eq!(vl_spec_decode_prompt_ms, 20.0);
        assert_eq!(vl_spec_decode_predicted_ms, 200.0);
        // End-to-end keeps the overhead the two phases miss: 260, not 234.
        assert_eq!(vl_spec_decode_ttft_ms, 24.0);
        assert_eq!(vl_spec_decode_decode_ms, 210.0);
        assert_eq!(vl_spec_decode_e2e_ms, 260.0);
        Ok(())
    }

    /// A fully accepted turn adds nothing to the service's denominator, so a
    /// traversal of only those is refused rather than sent.
    #[test]
    fn a_traversal_of_fully_accepted_turns_keeps_a_pooled_denominator() -> anyhow::Result<()> {
        let eos = |generated: u64, text: &str| counted(generated, generated, generated, 1, text);
        let turns = vec![counted(12, 16, 8, 3, "a"), eos(2, "b")];
        let BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_generated_tokens,
            vl_spec_decode_accepted_tokens,
            vl_spec_decode_turn_accept_length_sum,
            ..
        } = counters(Speculation::On, &turns)?
        else {
            anyhow::bail!("expected a vl_spec_decode result");
        };
        // Per request 11 / 3 and 1 / 1; on the wire 14 / (14 - 10).
        let sum = vl_spec_decode_turn_accept_length_sum.context("an accept length sum")?;
        assert!((sum - (11.0 / 3.0 + 1.0)).abs() < 1e-9, "{sum}");
        assert_eq!(vl_spec_decode_generated_tokens, 14);
        assert_eq!(vl_spec_decode_accepted_tokens, Some(10));

        let message = counters(Speculation::On, &[eos(2, "b"), eos(3, "c")])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(message.contains("accepted every one of its 5"), "{message}");
        Ok(())
    }

    /// Absent rather than zero: a zero would put a `0 / 0` in the server's rate.
    #[test]
    fn a_traversal_with_speculation_off_carries_no_drafter_counters() -> anyhow::Result<()> {
        let turns = vec![turn(12, 0, 0, "a"), turn(9, 0, 0, "b")];
        let BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_turns,
            vl_spec_decode_predicted_ms,
            vl_spec_decode_generated_tokens,
            vl_spec_decode_drafted_tokens,
            vl_spec_decode_accepted_tokens,
            vl_spec_decode_turn_accept_length_sum,
            ..
        } = counters(Speculation::Off, &turns)?
        else {
            anyhow::bail!("expected a vl_spec_decode result");
        };
        assert_eq!(vl_spec_decode_turns, 2);
        assert_eq!(vl_spec_decode_generated_tokens, 21);
        assert_eq!(vl_spec_decode_predicted_ms, 200.0);
        assert_eq!(vl_spec_decode_drafted_tokens, None);
        assert_eq!(vl_spec_decode_turn_accept_length_sum, None);
        assert_eq!(vl_spec_decode_accepted_tokens, None);
        Ok(())
    }

    #[rstest::rstest]
    #[case::never_drafted(Speculation::On, vec![turn(8, 0, 0, "x")], "not wired up")]
    #[case::drafted_with_speculation_off(
        Speculation::Off,
        vec![turn(8, 4, 4, "x")],
        "with speculation off"
    )]
    #[case::no_turns(Speculation::Off, vec![], "covered no turns")]
    #[case::no_wall_clock(
        Speculation::Off,
        vec![TurnRecord { e2e_ms: 0.0, ..turn(8, 0, 0, "x") }],
        "took no time on the client clock"
    )]
    fn a_traversal_that_measured_nothing_is_refused(
        #[case] speculation: Speculation,
        #[case] turns: Vec<TurnRecord>,
        #[case] expected: &str,
    ) {
        let message = counters(speculation, &turns)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(message.contains(expected), "{message}");
    }

    #[test]
    fn the_output_digest_follows_text_and_length_per_turn() -> anyhow::Result<()> {
        let same = output_digest(&[turn(12, 0, 0, "a"), turn(9, 0, 0, "b")])?;
        assert_eq!(
            same,
            output_digest(&[turn(12, 16, 8, "a"), turn(9, 8, 8, "b")])?,
            "drafting does not change what was said"
        );
        assert_ne!(
            same,
            output_digest(&[turn(12, 0, 0, "a"), turn(9, 0, 0, "c")])?
        );
        assert_ne!(
            same,
            output_digest(&[turn(12, 0, 0, "a"), turn(10, 0, 0, "b")])?
        );
        assert_ne!(
            output_digest(&[turn(12, 0, 0, "ab"), turn(9, 0, 0, "")])?,
            output_digest(&[turn(12, 0, 0, "a"), turn(9, 0, 0, "b")])?,
            "turn boundaries are part of the digest"
        );
        Ok(())
    }

    #[test]
    fn a_retriable_turn_is_tried_again() -> anyhow::Result<()> {
        let mut seen = Vec::new();
        let record = with_retries(
            &mut |n| {
                seen.push(n);
                if n < 3 {
                    return Err(TurnFailure {
                        error: anyhow::anyhow!("connection reset"),
                        retriable: true,
                    });
                }
                Ok(turn(8, 4, 4, "ok"))
            },
            &|_| {},
        )?;
        assert_eq!(seen, [1, 2, 3], "every attempt up to the success");
        assert_eq!(record.text, "ok");
        Ok(())
    }

    #[test]
    fn an_unretriable_turn_fails_at_once() {
        let mut attempts = 0;
        let error = with_retries(
            &mut |_| {
                attempts += 1;
                Err(TurnFailure {
                    error: anyhow::anyhow!("/v1/chat/completions failed: HTTP 400 Bad Request"),
                    retriable: false,
                })
            },
            &|_| {},
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert_eq!(attempts, 1, "a 4xx is not worth a second send");
        assert!(error.contains("400"), "{error}");
    }

    #[test]
    fn retries_give_up_and_report_the_last_failure() {
        let mut attempts = 0u32;
        let error = with_retries(
            &mut |_| {
                attempts += 1;
                Err(TurnFailure {
                    error: anyhow::anyhow!("HTTP 503 Service Unavailable"),
                    retriable: true,
                })
            },
            &|_| {},
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert_eq!(attempts, TURN_ATTEMPTS);
        assert!(error.contains("503"), "{error}");
    }

    fn stepping_clock(step_ms: u64) -> (Instant, impl FnMut() -> Instant) {
        let started = Instant::now();
        let mut ticks = 0u64;
        (started, move || {
            ticks += 1;
            started + Duration::from_millis(step_ms * ticks)
        })
    }

    fn sse(events: &[serde_json::Value]) -> String {
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .chain(std::iter::once("data: [DONE]\n\n".to_string()))
            .collect()
    }

    /// SSE frames from raw lines, with no `[DONE]` appended.
    fn sse_lines(lines: &[&str]) -> String {
        lines.iter().map(|line| format!("{line}\n\n")).collect()
    }

    const TIMINGS: &str = r#"{"prompt_ms": 50.0, "predicted_n": 3, "predicted_ms": 90.0, "draft_n": 4, "draft_n_accepted": 2, "cache_n": 0}"#;

    /// The role-only delta is not a token, and end-to-end runs one step past
    /// the last token.
    #[test]
    fn a_streamed_turn_is_stamped_at_its_tokens() -> anyhow::Result<()> {
        let timings: serde_json::Value = serde_json::from_str(TIMINGS)?;
        let input = sse(&[
            json!({"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": timings}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(turn.content, "Hello");
        assert_eq!(turn.timings.predicted_n, 3);
        assert_eq!(turn.timings.draft_n_accepted, 2);
        assert_eq!(turn.ttft, Duration::from_millis(10));
        assert_eq!(turn.decode, Duration::from_millis(20));
        assert_eq!(turn.e2e, Duration::from_millis(40));
        assert!(turn.e2e >= turn.ttft + turn.decode);
        Ok(())
    }

    #[rstest::rstest]
    #[case::end_of_sequence(json!("stop"), (true, false))]
    #[case::truncated(json!("length"), (false, true))]
    #[case::unnamed(json!(null), (false, false))]
    fn the_finish_reason_reaches_the_record(
        #[case] finish_reason: serde_json::Value,
        #[case] expected: (bool, bool),
    ) -> anyhow::Result<()> {
        let timings: serde_json::Value = serde_json::from_str(TIMINGS)?;
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "Yes"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {}, "finish_reason": finish_reason}],
                   "timings": timings}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(
            turn.finish_reason,
            finish_reason.as_str().map(str::to_owned)
        );
        let record = turn.into_record();
        assert_eq!((record.ended_on_eos, record.ended_on_length), expected);
        Ok(())
    }

    #[test]
    fn an_eos_only_turn_stamps_the_terminal_chunk() -> anyhow::Result<()> {
        let timings: serde_json::Value = serde_json::from_str(TIMINGS)?;
        let input = sse(&[
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": timings}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(turn.content, "");
        assert_eq!(turn.ttft, Duration::from_millis(10));
        assert_eq!(turn.decode, Duration::ZERO);
        assert_eq!(turn.e2e, Duration::from_millis(20));
        Ok(())
    }

    /// A cut body would otherwise be hashed into the digest as a whole answer.
    #[rstest::rstest]
    #[case::no_timings(
        sse(&[json!({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]})]),
        "carried no complete timings"
    )]
    #[case::error_chunk(
        sse(&[json!({"error": {"code": 500, "message": "context exceeded"}})]),
        "context exceeded"
    )]
    #[case::no_cache_count(
        sse(&[json!({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}],
                     "timings": {"prompt_ms": 50.0, "predicted_n": 3,
                                 "predicted_ms": 90.0}})]),
        "carried no complete timings"
    )]
    #[case::no_token(sse(&[]), "ended without a token")]
    #[case::cut_stream(
        sse_lines(&[
            &format!("data: {}", json!({"choices": [{"delta": {"content": "Hel"},
                                                     "finish_reason": "stop"}],
                                        "timings": {"prompt_ms": 50.0, "predicted_n": 3,
                                                    "predicted_ms": 90.0, "draft_n": 4,
                                                    "draft_n_accepted": 2,
                                                    "cache_n": 0}})),
            &format!("data: {}", json!({"choices": [{"delta": {"content": "lo"},
                                                     "finish_reason": null}]})),
        ]),
        "ended before its terminal event or [DONE]"
    )]
    fn an_uncountable_stream_is_refused(#[case] input: String, #[case] expected: &str) {
        let (started, clock) = stepping_clock(10);
        // `{:#}`, as the retry log renders it: the whole chain, not just the
        // outermost context.
        let message = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default();
        assert!(message.contains(expected), "{message}");
    }

    /// `mlx_vlm.server`'s shape: the complete `timings` comes last, on a
    /// choiceless usage chunk.
    #[test]
    fn a_partial_timings_chunk_is_progress_not_the_reading() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": null}],
                   "timings": {"predicted_per_second": 91.2}}),
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "timings": {"predicted_per_second": 91.2}}),
            json!({"choices": [], "usage": {"total_tokens": 43},
                   "timings": {"prompt_n": 40, "cache_n": 0, "predicted_n": 3,
                               "prompt_ms": 50.0, "predicted_ms": 90.0,
                               "draft_n": 4, "draft_n_accepted": 2}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(turn.content, "hi");
        assert_eq!(turn.timings.predicted_n, 3);
        assert_eq!(turn.timings.draft_n_accepted, 2);
        assert_eq!(turn.timings.cache_n, 0);
        // The usage chunk carries no token, so it must not move the last stamp.
        assert_eq!(turn.decode, Duration::from_millis(10));
        Ok(())
    }

    /// With no runtime clock, the two times are the client's own stamps.
    #[test]
    fn an_sglang_stream_is_read_from_usage_and_sglext() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": "length"}],
                   "sglext": {"spec_tokens_details": {
                       "spec_accept_rate": 0.19, "spec_accept_length": 2.72,
                       "spec_num_correct_drafts": 84, "spec_num_proposed_drafts": 423,
                       "spec_verify_ct": 47, "spec_correct_drafts_histogram": [16, 9]}}}),
            json!({"choices": [], "usage": {"prompt_tokens": 1302, "completion_tokens": 128,
                   "total_tokens": 1430,
                   "prompt_tokens_details": {"cached_tokens": 5, "image_tokens": 1280}}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::ClientClock,
        )?;
        assert_eq!(turn.content, "Hello");
        assert_eq!(turn.timings.predicted_n, 128);
        assert_eq!(turn.timings.draft_n, 423);
        assert_eq!(turn.timings.draft_n_accepted, 84);
        assert_eq!(turn.timings.counted_forwards, 47);
        assert_eq!(turn.timings.cache_n, 5);
        assert_eq!(turn.ttft, Duration::from_millis(10));
        assert_eq!(turn.decode, Duration::from_millis(10));
        assert_eq!(turn.timings.prompt_ms, 10.0);
        assert_eq!(turn.timings.predicted_ms, 10.0);
        Ok(())
    }

    #[test]
    fn an_sglang_stream_without_sglext_reads_as_no_drafts() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1,
                   "total_tokens": 10}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::ClientClock,
        )?;
        assert_eq!(turn.timings.predicted_n, 1);
        assert_eq!(
            (
                turn.timings.draft_n,
                turn.timings.draft_n_accepted,
                turn.timings.counted_forwards
            ),
            (0, 0, 0)
        );
        Ok(())
    }

    #[test]
    fn an_empty_data_event_is_skipped() -> anyhow::Result<()> {
        let timings: serde_json::Value = serde_json::from_str(TIMINGS)?;
        let first = json!({"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]});
        let last = json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
                          "timings": timings});
        let input = sse_lines(&[
            &format!("data: {first}"),
            "data:",
            &format!("data: {last}"),
            "data: [DONE]",
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(turn.content, "Hello");
        Ok(())
    }

    #[test]
    fn an_sglang_stream_without_usage_is_refused() {
        let input =
            sse(&[json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})]);
        let (started, clock) = stepping_clock(10);
        let err = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::ClientClock,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(err.contains("include_usage"), "{err}");
    }

    /// `mlx_vlm.server` with speculation off sends the counters as `null`.
    #[test]
    fn null_speculation_counters_read_as_zero() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}),
            json!({"choices": [],
                   "timings": {"prompt_n": 1794, "cache_n": 0, "predicted_n": 130,
                               "prompt_ms": 736.0, "predicted_ms": 1268.7,
                               "draft_kind": null, "draft_rounds": null,
                               "draft_n": null, "draft_n_accepted": null}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let turn = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?;
        assert_eq!(turn.timings.predicted_n, 130);
        assert_eq!(
            (
                turn.timings.draft_n,
                turn.timings.draft_n_accepted,
                turn.timings.counted_forwards
            ),
            (0, 0, 0)
        );
        Ok(())
    }

    #[test]
    fn mlx_draft_rounds_are_the_forward_count() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "2,3"}, "finish_reason": "stop"}]}),
            json!({"choices": [],
                   "timings": {"prompt_n": 1374, "cache_n": 0, "predicted_n": 4,
                               "prompt_ms": 744.0, "predicted_ms": 38.0,
                               "draft_kind": "dflash", "draft_rounds": 1,
                               "draft_n": 8, "draft_n_accepted": 5}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let record = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?
        .into_record();
        assert_eq!(record.counted_forwards, 1);
        assert_eq!(record.decode_tokens(), 3);
        record.validate()?;
        assert!((record.accept_length() - 3.0).abs() < 1e-9);
        Ok(())
    }

    /// A probed reply on the fork build.
    #[test]
    fn llamacpp_target_steps_are_the_forward_count() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "Hi! How can I help you today?"},
                                "finish_reason": "stop"}]}),
            json!({"choices": [],
                   "timings": {"prompt_n": 21, "cache_n": 0, "predicted_n": 10,
                               "target_n_steps": 3, "prompt_ms": 96.0,
                               "predicted_ms": 61.0, "draft_n": 16,
                               "draft_n_accepted": 8, "draft_n_verif_steps": 2}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let record = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?
        .into_record();
        // Three sampling steps, one of them the prefill's.
        assert_eq!(record.counted_forwards, 2);
        assert_eq!(record.decode_tokens(), 9);
        record.validate()?;
        assert!((record.accept_length() - 4.5).abs() < 1e-9);
        Ok(())
    }

    /// The same probed reply with `target_n_steps` withheld, as upstream sends it.
    #[test]
    fn llamacpp_verification_steps_are_read_where_the_step_total_is_absent() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "Hi! How can I help you today?"},
                                "finish_reason": "stop"}]}),
            json!({"choices": [],
                   "timings": {"prompt_n": 21, "cache_n": 0, "predicted_n": 10,
                               "prompt_ms": 96.0, "predicted_ms": 61.0, "draft_n": 16,
                               "draft_n_accepted": 8, "draft_n_verif_steps": 2}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let record = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?
        .into_record();
        assert_eq!(record.counted_forwards, 2);
        record.validate()?;
        assert!((record.accept_length() - 4.5).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn a_drafting_turn_with_no_count_and_no_draft_length_is_refused() -> anyhow::Result<()> {
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "Hi! How can I help you today?"},
                                "finish_reason": "stop"}]}),
            json!({"choices": [],
                   "timings": {"prompt_n": 21, "cache_n": 0, "predicted_n": 10,
                               "prompt_ms": 96.0, "predicted_ms": 61.0,
                               "draft_n": 16, "draft_n_accepted": 8}}),
        ]);
        let (started, clock) = stepping_clock(10);
        let record = read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::Runtime,
        )?
        .into_record();
        assert_eq!(record.counted_forwards, 0);
        let message = record
            .validate()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(message.contains("reported no forward count"), "{message}");
        assert!(message.contains("target_n_steps"), "{message}");
        Ok(())
    }

    /// A `llama-server` turn with no step count unless `steps` adds one.
    fn llamacpp_turn(
        predicted_n: u32,
        draft_n: u32,
        finish_reason: &str,
        steps: Option<u32>,
        draft_length: u32,
    ) -> anyhow::Result<TurnRecord> {
        let mut timings = json!({"prompt_n": 21, "cache_n": 0, "predicted_n": predicted_n,
                                 "prompt_ms": 96.0, "predicted_ms": 61.0,
                                 "draft_n": draft_n, "draft_n_accepted": draft_n.min(8)});
        if let Some(steps) = steps {
            timings["target_n_steps"] = json!(steps);
        }
        let input = sse(&[
            json!({"choices": [{"delta": {"content": "x"}, "finish_reason": finish_reason}]}),
            json!({"choices": [], "timings": timings}),
        ]);
        let (started, clock) = stepping_clock(10);
        Ok(read_chat_stream(
            std::io::Cursor::new(input),
            started,
            clock,
            TimingsSource::RuntimeFixedDraft(draft_length),
        )?
        .into_record())
    }

    /// Measured turns from the fork build, which reports both: exact when the
    /// reply ended on its own, the ceiling when `max_tokens` cut it (123
    /// drafted over 18 real steps reads 16).
    #[rstest::rstest]
    #[case::ended_on_its_own(10, 16, "stop", None, 2)]
    #[case::one_word_reply(2, 8, "stop", None, 1)]
    #[case::cut_at_max_tokens(37, 123, "length", None, 16)]
    #[case::the_servers_own_count_wins(37, 123, "length", Some(20), 19)]
    #[case::one_token_reply(1, 0, "stop", None, 0)]
    fn llamacpp_forwards_are_recovered_from_whole_draft_blocks(
        #[case] predicted_n: u32,
        #[case] draft_n: u32,
        #[case] finish_reason: &str,
        #[case] steps: Option<u32>,
        #[case] expected: u64,
    ) -> anyhow::Result<()> {
        let record = llamacpp_turn(predicted_n, draft_n, finish_reason, steps, 8)?;
        record.validate()?;
        assert_eq!(record.counted_forwards, expected);
        Ok(())
    }

    #[test]
    fn a_partial_draft_block_on_a_finished_reply_is_refused() {
        let message = llamacpp_turn(10, 18, "stop", None, 10)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(message.contains("not whole blocks of 10"), "{message}");
    }
}
