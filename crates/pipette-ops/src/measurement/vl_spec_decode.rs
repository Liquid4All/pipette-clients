//! VL spec decode, the DSpark benchmark: one traversal of an image-plus-text
//! corpus with the drafter on or off, summed to the counters the wire carries
//! (`docs/methodology/vl-spec-decode.md`).

use std::num::NonZeroUsize;
use std::time::{Duration, Instant, TryFromFloatSecsError};

use anyhow::Context;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256 as Sha256Hasher};

use pipette_plan_types::{
    BenchmarkResultData, PositiveCount, Sha256, TurnObservation, VlSpecDecodeBenchmark,
};

use super::MeasurementObserver;
use crate::readiness::ReadinessGate;
use crate::vl_spec_decode::StreamedTurn;

/// Attempts per batched call, the first included: a blip late in an hour-long
/// traversal should not cost the row. Matches the reference MMSpec driver.
const ATTEMPTS: u32 = 3;

/// An entry a VL spec-decode cell cannot run, or a reading or traversal it
/// cannot use.
#[derive(Debug, thiserror::Error)]
pub enum VlSpecDecodeMeasurementError {
    #[error("{marker} has no corpus samples to traverse")]
    NoSamples { marker: String },

    #[error("engine answered {actual} turns, expected {expected}")]
    TurnCount { actual: usize, expected: usize },

    #[error("invalid {field}: {ms}")]
    InvalidTime {
        field: &'static str,
        ms: f64,
        #[source]
        source: TryFromFloatSecsError,
    },

    #[error("engine returned generated_tokens {tokens}, not a positive count")]
    InvalidTokens { tokens: u32 },

    #[error(
        "engine drafted {drafted} tokens with speculation off: a speculative measurement \
         under the baseline's name"
    )]
    DraftedWithSpeculationOff { drafted: u32 },

    #[error("engine returned {accepted} accepted of {drafted} drafted tokens")]
    AcceptedExceedDrafted { accepted: u32, drafted: u32 },

    #[error(
        "engine drafted {drafted} tokens and counted no decode forwards: the accept length \
         needs `target_n_steps` from the llama.cpp fork, `draft_rounds` from mlx_vlm.server \
         or `spec_verify_ct` from SGLang"
    )]
    NoForwardCount { drafted: u32 },

    #[error(
        "engine counted {forwards} decode forwards for {decode_tokens} decode tokens: the \
         count is not this request's own"
    )]
    ForwardsExceedDecodeTokens { forwards: u32, decode_tokens: u64 },

    #[error("engine answered {cached} prompt tokens from its cache: prompt caching must be off")]
    PromptCached { cached: u32 },

    #[error("engine returned e2e {e2e:?}, below ttft {ttft:?} plus decode {decode:?}")]
    EndToEndBelowPhases {
        ttft: Duration,
        decode: Duration,
        e2e: Duration,
    },

    #[error(
        "engine stopped the reply for its length at {generated} tokens, short of the \
         {max_tokens} max_tokens it was sent with: the context cut it. Size the context \
         for the corpus's longest conversation"
    )]
    CutBelowMaxTokens { generated: u64, max_tokens: u32 },

    #[error("the traversal reported no generation time")]
    NoGenerationTime,

    #[error("the traversal took no time on the client clock")]
    NoClientTime,

    #[error("the traversal drafted no tokens with speculation on: the drafter is not wired up")]
    NothingDrafted,

    #[error(
        "the traversal accepted every one of its {generated} generated tokens: the pooled \
         accept length has no forward count to divide by"
    )]
    EveryTokenAccepted { generated: u64 },
}

/// What one turn ran, as the engine reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct VlSpecDecodeMeasurementReading {
    /// The runtime's own prefill time: the reference harness's TTFT.
    pub prompt: Duration,
    /// The runtime's own generation time, which a one-token reply can report
    /// as zero.
    pub predicted: Duration,
    /// Client clock: send to first token.
    pub ttft: Duration,
    /// Client clock: first token to last.
    pub decode: Duration,
    /// Client clock: send to the end of the stream.
    pub e2e: Duration,
    /// Tokens emitted, the one the prefill sampled included.
    pub generated_tokens: PositiveCount,
    pub drafted_tokens: u32,
    pub accepted_tokens: u32,
    /// Decode-time target forwards the runtime counted, the accept length's
    /// denominator; zero where it counts none.
    pub decode_forwards: u32,
    /// `cache_n`: a turn is refused unless it is zero.
    pub cached_tokens: u32,
    /// Stopped on the model's own end-of-sequence rather than a length limit.
    pub ended_on_eos: bool,
    /// Stopped at a length limit (`finish_reason` of `length`), which is
    /// `max_tokens` or the context.
    pub ended_on_length: bool,
    /// The reply, which the next turn carries and the digest hashes.
    pub text: String,
}

impl VlSpecDecodeMeasurementReading {
    /// A reading from one streamed reply
    /// ([`read_chat_stream`](crate::vl_spec_decode::read_chat_stream)).
    pub fn from_stream(turn: StreamedTurn) -> Result<Self, VlSpecDecodeMeasurementError> {
        let timings = turn.timings;
        let tokens = timings.predicted_n;
        Ok(Self {
            prompt: runtime_time("prompt_ms", timings.prompt_ms)?,
            predicted: runtime_time("predicted_ms", timings.predicted_ms)?,
            ttft: turn.ttft,
            decode: turn.decode,
            e2e: turn.e2e,
            generated_tokens: PositiveCount::try_from(tokens)
                .map_err(|_| VlSpecDecodeMeasurementError::InvalidTokens { tokens })?,
            drafted_tokens: timings.draft_n,
            accepted_tokens: timings.draft_n_accepted,
            decode_forwards: timings.counted_forwards,
            cached_tokens: timings.cache_n,
            ended_on_eos: turn.finish_reason.as_deref() == Some("stop"),
            ended_on_length: turn.finish_reason.as_deref() == Some("length"),
            text: turn.content,
        })
    }
}

/// Whether the traversal's drafter is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlSpecDecodeMeasurementSpeculation {
    Off,
    On,
}

/// Why an attempt failed, and whether sending it again could do better.
#[derive(Debug)]
pub struct VlSpecDecodeMeasurementAttemptFailure {
    pub error: anyhow::Error,
    pub retriable: bool,
}

/// One corpus sample: a conversation of `turns` user turns.
#[derive(Debug, Clone, Copy)]
pub struct VlSpecDecodeMeasurementSample<'a> {
    pub id: &'a str,
    pub turns: NonZeroUsize,
}

/// One turn of the batch an attempt sends.
#[derive(Debug, Clone, Copy)]
pub struct VlSpecDecodeMeasurementTurn<'a> {
    /// Index into the samples.
    pub sample: usize,
    /// Index into that sample's turns.
    pub turn: usize,
    /// The sample's earlier turns, whose replies the conversation carries.
    pub earlier: &'a [VlSpecDecodeMeasurementReading],
}

/// What a traversal reports: the wire's counters, and the digest, per-turn
/// record and summary the run carries beside them.
#[derive(Debug, Clone, PartialEq)]
pub struct VlSpecDecodeMeasurementRow {
    pub result_data: BenchmarkResultData,
    /// Per turn, the reply's sha256, a tab, its token count and a newline.
    pub output_digest: Sha256,
    /// `RunResponse::turns`, in file order at any batch size.
    pub turns: Vec<TurnObservation>,
    /// `RunResponse::client_extras`.
    pub summary: Map<String, Value>,
}

/// Measure a VL spec-decode cell: an untimed warm-up sending the corpus's
/// first turn once per sample of the first batch, then the samples in file
/// order, `parameter_batch_size` at a time, each sample's turns sent in order
/// with the earlier replies carried forward, each checked and summed.
///
/// An attempt carries the next turn of every sample in the batch still in its
/// conversation, for one batched call, and returns a reading per turn in order.
/// It must cap each turn at `parameter_max_tokens`: a `length` stop below that
/// is refused. A batch is not refilled as its conversations end, so its later
/// rounds, and a short last batch, carry fewer than `parameter_batch_size`
/// turns.
///
/// `readiness_gate` runs before each batch, never between its turns. An
/// attempt that fails retriably is sent again, three attempts in all, `sleep`
/// apart: `std::thread::sleep` outside tests.
///
/// `ctx` is lent to every attempt, the warm-up's included, for what the engine
/// records while they run: the server it sends to. Lend a reference to read it
/// back afterwards, a failed run included.
///
/// `attempt_timeout_seconds` is the caller's: it begins each batch's attempt in
/// `readiness_gate`.
#[expect(
    clippy::too_many_arguments,
    reason = "the seams every entry takes, plus the corpus and the retry sleep"
)]
pub fn vl_spec_decode<C>(
    benchmark: &VlSpecDecodeBenchmark,
    speculation: VlSpecDecodeMeasurementSpeculation,
    samples: &[VlSpecDecodeMeasurementSample<'_>],
    readiness_gate: ReadinessGate<'_>,
    observer: MeasurementObserver<'_>,
    sleep: &dyn Fn(Duration),
    mut ctx: C,
    mut attempt: impl FnMut(
        &mut C,
        &[VlSpecDecodeMeasurementTurn<'_>],
    ) -> Result<
        Vec<VlSpecDecodeMeasurementReading>,
        VlSpecDecodeMeasurementAttemptFailure,
    >,
) -> anyhow::Result<VlSpecDecodeMeasurementRow> {
    // Names the cell in every log line and error. TODO: add the job id.
    let marker = match speculation {
        VlSpecDecodeMeasurementSpeculation::Off => {
            format!("{} speculation off", benchmark.benchmark_id)
        }
        VlSpecDecodeMeasurementSpeculation::On => {
            format!("{} speculation on", benchmark.benchmark_id)
        }
    };

    let batch_size = benchmark.parameter_batch_size.in_flight()?;
    if samples.is_empty() {
        return Err(VlSpecDecodeMeasurementError::NoSamples { marker }.into());
    }

    let warm_up = format!("{marker} warm-up");
    tracing::info!(cell = warm_up.as_str(), "running");
    let first = VlSpecDecodeMeasurementTurn {
        sample: 0,
        turn: 0,
        earlier: &[],
    };
    let first_batch = vec![first; batch_size.get().min(samples.len())];
    send(&warm_up, sleep, &mut ctx, &mut attempt, &first_batch).context(warm_up)?;

    readiness_gate().with_context(|| marker.clone())?;
    (observer.on_before_rep_start)().with_context(|| marker.clone())?;
    let started = Instant::now();
    let mut progress = Progress::default();
    let readings = samples.chunks(batch_size.get()).enumerate().try_fold(
        Vec::with_capacity(samples.len()),
        |mut readings, (batch, group)| {
            let start = batch * batch_size.get();
            let group = group
                .iter()
                .enumerate()
                .map(|(offset, &sample)| Member {
                    index: start + offset,
                    label: format!(
                        "sample {} ({}/{})",
                        sample.id,
                        start + offset + 1,
                        samples.len()
                    ),
                    sample,
                })
                .collect::<Vec<_>>();
            let label = match group.as_slice() {
                [only] => format!("{marker} {}", only.label),
                _ => format!(
                    "{marker} samples {}-{}/{}",
                    start + 1,
                    start + group.len(),
                    samples.len()
                ),
            };
            // The gate before the traversal covers the first batch.
            if batch > 0 {
                readiness_gate().context(label.clone())?;
            }
            let conversations = run_batch(
                &marker,
                &label,
                sleep,
                &mut ctx,
                &mut attempt,
                speculation,
                benchmark.parameter_max_tokens.into_inner().unsigned_abs(),
                &group,
            )
            .context(label)?;
            conversations.iter().for_each(|conversation| {
                progress.add(conversation);
                progress.log(&marker, speculation, samples.len());
            });
            readings.extend(conversations);
            anyhow::Ok(readings)
        },
    )?;
    let wall = started.elapsed();
    (observer.on_after_rep_end)().with_context(|| marker.clone())?;

    reduce(&marker, speculation, batch_size, samples, &readings, wall).context(marker)
}

struct Member<'a> {
    index: usize,
    label: String,
    sample: VlSpecDecodeMeasurementSample<'a>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the entry's seams, plus the condition and the cap each reading is checked against"
)]
fn run_batch<C>(
    marker: &str,
    label: &str,
    sleep: &dyn Fn(Duration),
    ctx: &mut C,
    attempt: &mut impl FnMut(
        &mut C,
        &[VlSpecDecodeMeasurementTurn<'_>],
    ) -> Result<
        Vec<VlSpecDecodeMeasurementReading>,
        VlSpecDecodeMeasurementAttemptFailure,
    >,
    speculation: VlSpecDecodeMeasurementSpeculation,
    max_tokens: u32,
    group: &[Member<'_>],
) -> anyhow::Result<Vec<Vec<VlSpecDecodeMeasurementReading>>> {
    let rounds = group
        .iter()
        .map(|member| member.sample.turns.get())
        .max()
        .unwrap_or_default();
    let conversations = group
        .iter()
        .map(|member| Vec::with_capacity(member.sample.turns.get()))
        .collect::<Vec<_>>();
    (0..rounds).try_fold(conversations, |mut conversations, turn| {
        let at = format!("turn {}/{rounds}", turn + 1);
        let turns = group
            .iter()
            .zip(&conversations)
            .filter(|(member, _)| member.sample.turns.get() > turn)
            .map(|(member, earlier)| VlSpecDecodeMeasurementTurn {
                sample: member.index,
                turn,
                earlier,
            })
            .collect::<Vec<_>>();
        let expected = turns.len();
        let got =
            send(&format!("{label} {at}"), sleep, ctx, attempt, &turns).context(at.clone())?;
        if got.len() != expected {
            return Err(VlSpecDecodeMeasurementError::TurnCount {
                actual: got.len(),
                expected,
            })
            .context(at);
        }
        group
            .iter()
            .zip(conversations.iter_mut())
            .filter(|(member, _)| member.sample.turns.get() > turn)
            .zip(got)
            .try_for_each(|((member, conversation), reading)| {
                let at = format!("turn {}/{}", turn + 1, member.sample.turns);
                // Before the check, so a rejected reading is in the log too.
                tracing::debug!(
                    cell = marker,
                    sample = member.label.as_str(),
                    sample_id = %member.sample.id,
                    turn = turn + 1,
                    turns = member.sample.turns.get(),
                    prompt_ms = millis(reading.prompt),
                    predicted_ms = millis(reading.predicted),
                    ttft_ms = millis(reading.ttft),
                    decode_ms = millis(reading.decode),
                    e2e_ms = millis(reading.e2e),
                    generated_tokens = reading.generated_tokens.into_inner(),
                    drafted_tokens = reading.drafted_tokens,
                    accepted_tokens = reading.accepted_tokens,
                    decode_forwards = reading.decode_forwards,
                    cached_tokens = reading.cached_tokens,
                    "turn reading"
                );
                let checked = check(&reading, speculation, max_tokens).context(at);
                // One sample is the batch's own label already.
                match group {
                    [_] => checked?,
                    _ => checked.context(member.label.clone())?,
                }
                conversation.push(reading);
                anyhow::Ok(())
            })?;
        anyhow::Ok(conversations)
    })
}

/// One batched call, sent again after a retriable failure with the reference
/// driver's backoff of five seconds times the attempt.
fn send<C>(
    label: &str,
    sleep: &dyn Fn(Duration),
    ctx: &mut C,
    attempt: &mut impl FnMut(
        &mut C,
        &[VlSpecDecodeMeasurementTurn<'_>],
    ) -> Result<
        Vec<VlSpecDecodeMeasurementReading>,
        VlSpecDecodeMeasurementAttemptFailure,
    >,
    turns: &[VlSpecDecodeMeasurementTurn<'_>],
) -> anyhow::Result<Vec<VlSpecDecodeMeasurementReading>> {
    let mut n = 1;
    loop {
        match attempt(ctx, turns) {
            Ok(readings) => return Ok(readings),
            Err(failure) if failure.retriable && n < ATTEMPTS => {
                let backoff = Duration::from_secs(5 * u64::from(n));
                tracing::warn!(
                    cell = label,
                    attempt = n,
                    attempts = ATTEMPTS,
                    retry_in = ?backoff,
                    error = %format_args!("{:#}", failure.error),
                    "attempt failed; retrying"
                );
                sleep(backoff);
                n += 1;
            }
            Err(failure) => return Err(failure.error),
        }
    }
}

/// A turn whose counters agree with each other, the condition and the
/// `max_tokens` it was sent with. Never retried: a contradiction or a cut
/// recurs on every send.
fn check(
    reading: &VlSpecDecodeMeasurementReading,
    speculation: VlSpecDecodeMeasurementSpeculation,
    max_tokens: u32,
) -> Result<(), VlSpecDecodeMeasurementError> {
    let drafted = reading.drafted_tokens;
    // Per turn rather than over the traversal, so a baseline that drafts stops
    // at its first turn instead of its last.
    if speculation == VlSpecDecodeMeasurementSpeculation::Off && drafted > 0 {
        return Err(VlSpecDecodeMeasurementError::DraftedWithSpeculationOff { drafted });
    }
    if reading.accepted_tokens > drafted {
        return Err(VlSpecDecodeMeasurementError::AcceptedExceedDrafted {
            accepted: reading.accepted_tokens,
            drafted,
        });
    }
    // `generated - accepted` is no forward count: the runtimes count drafts
    // the stop token discarded.
    if drafted > 0 && reading.decode_forwards == 0 {
        return Err(VlSpecDecodeMeasurementError::NoForwardCount { drafted });
    }
    // Every forward commits a token, so more forwards than tokens is a count
    // shared with other requests.
    let decode_tokens = decode_tokens(reading);
    if u64::from(reading.decode_forwards) > decode_tokens {
        return Err(VlSpecDecodeMeasurementError::ForwardsExceedDecodeTokens {
            forwards: reading.decode_forwards,
            decode_tokens,
        });
    }
    if reading.cached_tokens > 0 {
        return Err(VlSpecDecodeMeasurementError::PromptCached {
            cached: reading.cached_tokens,
        });
    }
    // The service refuses a row whose e2e is below the sum of its phases.
    if reading.e2e < reading.ttft.saturating_add(reading.decode) {
        return Err(VlSpecDecodeMeasurementError::EndToEndBelowPhases {
            ttft: reading.ttft,
            decode: reading.decode,
            e2e: reading.e2e,
        });
    }
    // `llama-server` stops a reply that runs out of context with `length`, as
    // it stops one at `max_tokens`, and SGLang clips `max_tokens` to what fits.
    let generated = count(reading.generated_tokens);
    if reading.ended_on_length && generated < u64::from(max_tokens) {
        return Err(VlSpecDecodeMeasurementError::CutBelowMaxTokens {
            generated,
            max_tokens,
        });
    }
    Ok(())
}

/// The traversal's row: the per-turn record, the digest over it, and the sums,
/// refused where the service would divide by zero or the condition drafted
/// nothing.
fn reduce(
    marker: &str,
    speculation: VlSpecDecodeMeasurementSpeculation,
    batch_size: NonZeroUsize,
    samples: &[VlSpecDecodeMeasurementSample<'_>],
    readings: &[Vec<VlSpecDecodeMeasurementReading>],
    wall: Duration,
) -> anyhow::Result<VlSpecDecodeMeasurementRow> {
    let turns = samples
        .iter()
        .zip(readings)
        .flat_map(|(sample, readings)| {
            readings
                .iter()
                .enumerate()
                .map(|(turn, reading)| observation(sample.id, turn, reading))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let totals = readings
        .iter()
        .flatten()
        .fold(Totals::default(), |totals, reading| totals.add(reading));

    if totals.predicted_ms <= 0.0 {
        return Err(VlSpecDecodeMeasurementError::NoGenerationTime.into());
    }
    if totals.e2e_ms <= 0.0 {
        return Err(VlSpecDecodeMeasurementError::NoClientTime.into());
    }
    let tok_s = totals.generated_tokens as f64 / totals.predicted_ms * 1000.0;
    let wall_ms = millis(wall);
    let (drafted, accepted, turn_accept_length_sum) = match speculation {
        VlSpecDecodeMeasurementSpeculation::Off => {
            tracing::info!(
                cell = marker,
                turns = totals.turns,
                generated_tokens = totals.generated_tokens,
                tok_s,
                wall_ms,
                "totals"
            );
            (None, None, None)
        }
        VlSpecDecodeMeasurementSpeculation::On => {
            // A single turn may draft nothing; a whole traversal may not.
            if totals.drafted_tokens == 0 {
                return Err(VlSpecDecodeMeasurementError::NothingDrafted.into());
            }
            // The service divides by `generated - accepted` unguarded.
            if totals.generated_tokens <= totals.accepted_tokens {
                return Err(VlSpecDecodeMeasurementError::EveryTokenAccepted {
                    generated: totals.generated_tokens,
                }
                .into());
            }
            tracing::info!(
                cell = marker,
                turns = totals.turns,
                generated_tokens = totals.generated_tokens,
                decode_forwards = totals.decode_forwards,
                accept_length_pooled =
                    totals.decode_tokens as f64 / totals.decode_forwards.max(1) as f64,
                accept_length_per_request = totals.turn_accept_length_sum / totals.turns as f64,
                tok_s,
                wall_ms,
                "totals"
            );
            (
                Some(totals.drafted_tokens),
                Some(totals.accepted_tokens),
                Some(totals.turn_accept_length_sum),
            )
        }
    };

    let summary = summary(speculation, batch_size, &totals, wall_ms);
    Ok(VlSpecDecodeMeasurementRow {
        result_data: BenchmarkResultData::VlSpecDecode {
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
        },
        output_digest: output_digest(&turns)?,
        turns,
        summary,
    })
}

/// The row's `client_extras`, as `pipette_vl_spec_decode_corpus::turn` reports
/// a traversal.
fn summary(
    speculation: VlSpecDecodeMeasurementSpeculation,
    batch_size: NonZeroUsize,
    totals: &Totals,
    wall_ms: f64,
) -> Map<String, Value> {
    let generated = totals.generated_tokens;
    let aggregate_tok_s = if wall_ms > 0.0 {
        generated as f64 / wall_ms * 1000.0
    } else {
        0.0
    };
    let mut out = Map::from_iter([
        ("turns".to_owned(), totals.turns.into()),
        ("batch_size".to_owned(), batch_size.get().into()),
        ("generated_tokens".to_owned(), generated.into()),
        ("wall_ms".to_owned(), wall_ms.into()),
        ("aggregate_tok_s".to_owned(), aggregate_tok_s.into()),
    ]);
    if speculation == VlSpecDecodeMeasurementSpeculation::On {
        let accepted = totals.accepted_tokens;
        let forwards = totals.decode_forwards;
        out.extend([
            ("decode_forwards".to_owned(), forwards.into()),
            ("accepted_tokens".to_owned(), accepted.into()),
            (
                "accept_length_pooled".to_owned(),
                (totals.decode_tokens as f64 / forwards.max(1) as f64).into(),
            ),
            (
                "accept_length_per_request".to_owned(),
                (totals.turn_accept_length_sum / totals.turns.max(1) as f64).into(),
            ),
            (
                "accept_length_from_counters".to_owned(),
                (generated as f64 / generated.saturating_sub(accepted).max(1) as f64).into(),
            ),
        ]);
    }
    out
}

#[derive(Debug, Default)]
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
    decode_forwards: u64,
    turn_accept_length_sum: f64,
}

impl Totals {
    fn add(mut self, reading: &VlSpecDecodeMeasurementReading) -> Self {
        self.turns += 1;
        self.prompt_ms += millis(reading.prompt);
        self.predicted_ms += millis(reading.predicted);
        self.ttft_ms += millis(reading.ttft);
        self.decode_ms += millis(reading.decode);
        self.e2e_ms += millis(reading.e2e);
        self.generated_tokens += count(reading.generated_tokens);
        self.decode_tokens += decode_tokens(reading);
        self.drafted_tokens += u64::from(reading.drafted_tokens);
        self.accepted_tokens += u64::from(reading.accepted_tokens);
        self.decode_forwards += u64::from(reading.decode_forwards);
        self.turn_accept_length_sum += accept_length(reading);
        self
    }
}

/// How far the traversal has got, logged every hundredth sample so an
/// hour-long run says how it is going without a line per turn.
#[derive(Debug, Default)]
struct Progress {
    samples: usize,
    turns: usize,
    decode_tokens: u64,
    decode_forwards: u64,
}

impl Progress {
    fn add(&mut self, readings: &[VlSpecDecodeMeasurementReading]) {
        self.samples += 1;
        self.turns += readings.len();
        self.decode_tokens += readings.iter().map(decode_tokens).sum::<u64>();
        self.decode_forwards += readings
            .iter()
            .map(|reading| u64::from(reading.decode_forwards))
            .sum::<u64>();
    }

    fn log(&self, marker: &str, speculation: VlSpecDecodeMeasurementSpeculation, total: usize) {
        let done = self.samples;
        if !done.is_multiple_of(100) && done != total {
            return;
        }
        match speculation {
            VlSpecDecodeMeasurementSpeculation::Off => tracing::info!(
                cell = marker,
                completed = done,
                total,
                turns = self.turns,
                "progress"
            ),
            VlSpecDecodeMeasurementSpeculation::On => tracing::info!(
                cell = marker,
                completed = done,
                total,
                turns = self.turns,
                accept_length = self.decode_tokens as f64 / self.decode_forwards.max(1) as f64,
                "progress"
            ),
        }
    }
}

fn observation(
    sample_id: &str,
    turn: usize,
    reading: &VlSpecDecodeMeasurementReading,
) -> anyhow::Result<TurnObservation> {
    Ok(TurnObservation {
        sample_id: sample_id.to_owned(),
        turn: u32::try_from(turn).context("turn index exceeds u32")?,
        text_sha256: text_sha256(&reading.text)?,
        generated_tokens: count(reading.generated_tokens),
        drafted_tokens: u64::from(reading.drafted_tokens),
        accepted_tokens: u64::from(reading.accepted_tokens),
        counted_forwards: u64::from(reading.decode_forwards),
        ended_on_eos: reading.ended_on_eos,
        prompt_ms: millis(reading.prompt),
        predicted_ms: millis(reading.predicted),
        ttft_ms: millis(reading.ttft),
        decode_ms: millis(reading.decode),
        e2e_ms: millis(reading.e2e),
    })
}

/// sha256 over the per-turn record's reply hashes and token counts, so two
/// rows compare for greedy identity without either's text leaving the device.
fn output_digest(turns: &[TurnObservation]) -> anyhow::Result<Sha256> {
    let hasher = turns.iter().fold(Sha256Hasher::new(), |mut hasher, turn| {
        hasher.update(turn.text_sha256.as_ref().as_bytes());
        hasher.update(b"\t");
        hasher.update(turn.generated_tokens.to_string().as_bytes());
        hasher.update(b"\n");
        hasher
    });
    sha256_of(hasher)
}

fn text_sha256(text: &str) -> anyhow::Result<Sha256> {
    let mut hasher = Sha256Hasher::new();
    hasher.update(text.as_bytes());
    sha256_of(hasher)
}

/// Fallible only in the type: 32 bytes always render as 64 hex characters.
fn sha256_of(hasher: Sha256Hasher) -> anyhow::Result<Sha256> {
    let hex = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(Sha256::try_new(hex)?)
}

/// Tokens committed per decode forward, the prefill out of both terms; a
/// one-token reply ran no decode loop and reads zero.
fn accept_length(reading: &VlSpecDecodeMeasurementReading) -> f64 {
    if reading.decode_forwards == 0 {
        return 0.0;
    }
    decode_tokens(reading) as f64 / f64::from(reading.decode_forwards)
}

/// Every token but the first, which the prefill sampled before any draft
/// existed.
fn decode_tokens(reading: &VlSpecDecodeMeasurementReading) -> u64 {
    count(reading.generated_tokens).saturating_sub(1)
}

/// A count on the wire. `PositiveCount` is never negative, so this is exact.
fn count(tokens: PositiveCount) -> u64 {
    u64::from(tokens.into_inner().unsigned_abs())
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// A runtime's `timings` time, refused where it is no duration.
fn runtime_time(field: &'static str, ms: f64) -> Result<Duration, VlSpecDecodeMeasurementError> {
    Duration::try_from_secs_f64(ms / 1000.0)
        .map_err(|source| VlSpecDecodeMeasurementError::InvalidTime { field, ms, source })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::vl_spec_decode::{ChatTimings, Speculation, TurnRecord, counters};

    const ID: &str = "vl_spec_decode_mmspec_test_gqa_k8";

    fn body(batch_size: u32) -> anyhow::Result<VlSpecDecodeBenchmark> {
        Ok(serde_json::from_value(json!({
            "benchmark_id": ID,
            "parameter_draft_length": 8,
            "parameter_max_tokens": 2048,
            "parameter_batch_size": batch_size,
            "dataset": {
                "format": "vl-spec-decode-corpus/1",
                "adapter": "mmspec@1",
                "digest": "1862eedc832710f6c4776472d00d5ee4aa0a16d85d6a9c433398109f90239810",
                "bytes": 15_955_456,
                "upstream": {
                    "source": "git",
                    "url": "github.com/acme/corpus",
                    "revision": "0123456789abcdef0123456789abcdef01234567",
                    "identity": "2923603292c7dd463ad01a119f57965817778fc3",
                    "digest": "2222222222222222222222222222222222222222222222222222222222222222",
                },
            },
        }))?)
    }

    /// A one-turn sample and a two-turn conversation.
    fn samples() -> anyhow::Result<[VlSpecDecodeMeasurementSample<'static>; 2]> {
        Ok([
            VlSpecDecodeMeasurementSample {
                id: "s1",
                turns: NonZeroUsize::try_from(1)?,
            },
            VlSpecDecodeMeasurementSample {
                id: "s2",
                turns: NonZeroUsize::try_from(2)?,
            },
        ])
    }

    fn reading(
        generated: i32,
        drafted: u32,
        accepted: u32,
        forwards: u32,
        text: &str,
    ) -> anyhow::Result<VlSpecDecodeMeasurementReading> {
        Ok(VlSpecDecodeMeasurementReading {
            prompt: Duration::from_millis(10),
            predicted: Duration::from_millis(100),
            ttft: Duration::from_millis(12),
            decode: Duration::from_millis(105),
            e2e: Duration::from_millis(130),
            generated_tokens: PositiveCount::try_new(generated)?,
            drafted_tokens: drafted,
            accepted_tokens: accepted,
            decode_forwards: forwards,
            cached_tokens: 0,
            ended_on_eos: true,
            ended_on_length: false,
            text: text.to_owned(),
        })
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver::QUIET
    }

    /// The entry behind an open gate, with no observer and no sleep.
    fn unobserved(
        benchmark: &VlSpecDecodeBenchmark,
        speculation: VlSpecDecodeMeasurementSpeculation,
        samples: &[VlSpecDecodeMeasurementSample<'_>],
        mut attempt: impl FnMut(
            VlSpecDecodeMeasurementTurn<'_>,
        ) -> Result<
            VlSpecDecodeMeasurementReading,
            VlSpecDecodeMeasurementAttemptFailure,
        >,
    ) -> anyhow::Result<VlSpecDecodeMeasurementRow> {
        vl_spec_decode(
            benchmark,
            speculation,
            samples,
            &|| Ok(()),
            no_observer(),
            &|_| {},
            (),
            |(), turns| turns.iter().map(|&turn| attempt(turn)).collect(),
        )
    }

    fn sent(turn: &VlSpecDecodeMeasurementTurn<'_>) -> String {
        let earlier = turn
            .earlier
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join(",");
        format!("{}.{} [{earlier}]", turn.sample, turn.turn)
    }

    fn fatal(error: anyhow::Error) -> VlSpecDecodeMeasurementAttemptFailure {
        VlSpecDecodeMeasurementAttemptFailure {
            error,
            retriable: false,
        }
    }

    /// The same turn as the traversal in `crate::vl_spec_decode` records it.
    fn record(reading: &VlSpecDecodeMeasurementReading) -> TurnRecord {
        TurnRecord {
            prompt_ms: millis(reading.prompt),
            predicted_ms: millis(reading.predicted),
            ttft_ms: millis(reading.ttft),
            decode_ms: millis(reading.decode),
            e2e_ms: millis(reading.e2e),
            generated_tokens: count(reading.generated_tokens),
            drafted_tokens: u64::from(reading.drafted_tokens),
            accepted_tokens: u64::from(reading.accepted_tokens),
            counted_forwards: u64::from(reading.decode_forwards),
            cached_tokens: u64::from(reading.cached_tokens),
            ended_on_eos: reading.ended_on_eos,
            ended_on_length: reading.ended_on_length,
            text: reading.text.clone(),
        }
    }

    fn streamed(
        prompt_ms: f64,
        predicted_ms: f64,
        predicted_n: u32,
        finish_reason: &str,
    ) -> StreamedTurn {
        StreamedTurn {
            content: "Green".to_owned(),
            timings: ChatTimings {
                prompt_ms,
                predicted_n,
                predicted_ms,
                draft_n: 7,
                draft_n_accepted: 6,
                counted_forwards: 1,
                cache_n: 0,
            },
            finish_reason: Some(finish_reason.to_owned()),
            ttft: Duration::from_millis(12),
            decode: Duration::from_millis(3),
            e2e: Duration::from_millis(20),
        }
    }

    /// A stream's counts and spans carry over, a zero generation time
    /// included, a `stop` ending on the model's own end-of-sequence and a
    /// `length` at a length limit; a runtime time that is no duration, or a
    /// count that is no positive `i32`, is refused by field.
    #[rstest]
    #[case::one_token_reply(10.0, 0.0, 1, "stop", Ok((true, false)))]
    #[case::cut_at_max_tokens(10.0, 0.0, 1, "length", Ok((false, true)))]
    #[case::negative_prompt(-1.0, 40.0, 2, "stop", Err("prompt_ms"))]
    #[case::nan_predicted(10.0, f64::NAN, 2, "stop", Err("predicted_ms"))]
    #[case::infinite_prompt(f64::INFINITY, 40.0, 2, "stop", Err("prompt_ms"))]
    #[case::no_tokens(10.0, 40.0, 0, "stop", Err("generated_tokens"))]
    #[case::past_i32(10.0, 40.0, u32::MAX, "stop", Err("generated_tokens"))]
    fn a_stream_becomes_a_reading(
        #[case] prompt_ms: f64,
        #[case] predicted_ms: f64,
        #[case] predicted_n: u32,
        #[case] finish_reason: &str,
        #[case] expected: Result<(bool, bool), &str>,
    ) -> anyhow::Result<()> {
        let got = VlSpecDecodeMeasurementReading::from_stream(streamed(
            prompt_ms,
            predicted_ms,
            predicted_n,
            finish_reason,
        ));
        match (got, expected) {
            (Ok(got), Ok((ended_on_eos, ended_on_length))) => assert_eq!(
                got,
                VlSpecDecodeMeasurementReading {
                    prompt: Duration::from_millis(10),
                    predicted: Duration::ZERO,
                    decode: Duration::from_millis(3),
                    e2e: Duration::from_millis(20),
                    ended_on_eos,
                    ended_on_length,
                    ..reading(1, 7, 6, 1, "Green")?
                }
            ),
            (Err(VlSpecDecodeMeasurementError::InvalidTime { field, .. }), Err(want)) => {
                assert_eq!(field, want);
            }
            (Err(VlSpecDecodeMeasurementError::InvalidTokens { tokens }), Err(want)) => {
                assert_eq!(("generated_tokens", tokens), (want, predicted_n));
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    /// The row is what `vl_spec_decode::counters` and `output_digest` make of
    /// the same turns, with the per-turn record in traversal order. A turn
    /// with speculation on may draft nothing, and a one-token reply reads an
    /// accept length of zero.
    #[rstest]
    #[case::off(VlSpecDecodeMeasurementSpeculation::Off, [
        reading(5, 0, 0, 0, "a"),
        reading(3, 0, 0, 0, "b"),
        reading(1, 0, 0, 0, "c"),
    ])]
    #[case::on(VlSpecDecodeMeasurementSpeculation::On, [
        reading(9, 7, 6, 2, "a"),
        reading(5, 14, 3, 2, "b"),
        reading(1, 0, 0, 0, "c"),
    ])]
    fn a_traversal_becomes_the_row(
        #[case] speculation: VlSpecDecodeMeasurementSpeculation,
        #[case] readings: [anyhow::Result<VlSpecDecodeMeasurementReading>; 3],
    ) -> anyhow::Result<()> {
        let readings = readings.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
        let by_turn = [(0, 0), (1, 0), (1, 1)];
        let row = unobserved(&body(1)?, speculation, &samples()?, |turn| {
            by_turn
                .iter()
                .position(|&at| at == (turn.sample, turn.turn))
                .and_then(|i| readings.get(i).cloned())
                .context("no reading for this turn")
                .map_err(fatal)
        })?;

        let records = readings.iter().map(record).collect::<Vec<_>>();
        let legacy =
            Speculation::from_enabled(speculation == VlSpecDecodeMeasurementSpeculation::On);
        assert_eq!(row.result_data, counters(legacy, &records)?);
        assert_eq!(
            row.output_digest,
            crate::vl_spec_decode::output_digest(&records)?
        );
        assert_eq!(
            row.turns
                .iter()
                .map(|t| (t.sample_id.as_str(), t.turn, t.generated_tokens))
                .collect::<Vec<_>>(),
            [
                ("s1", 0, records[0].generated_tokens),
                ("s2", 0, records[1].generated_tokens),
                ("s2", 1, 1)
            ]
        );
        Ok(())
    }

    /// A turn whose counters contradict each other, the condition or the
    /// `max_tokens` it was sent with stops the traversal at that turn, and is
    /// not sent again.
    #[rstest]
    #[case::drafted_off(VlSpecDecodeMeasurementSpeculation::Off, reading(5, 7, 3, 2, "c"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::DraftedWithSpeculationOff { drafted: 7 }))]
    #[case::accepted_past_drafted(VlSpecDecodeMeasurementSpeculation::On, reading(5, 3, 4, 2, "c"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::AcceptedExceedDrafted { .. }))]
    #[case::no_forward_count(VlSpecDecodeMeasurementSpeculation::On, reading(5, 7, 3, 0, "c"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::NoForwardCount { .. }))]
    #[case::forwards_past_tokens(VlSpecDecodeMeasurementSpeculation::On, reading(5, 7, 3, 5, "c"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::ForwardsExceedDecodeTokens { forwards: 5, decode_tokens: 4 }))]
    #[case::cached(VlSpecDecodeMeasurementSpeculation::On, reading(5, 7, 3, 2, "c").map(|r| VlSpecDecodeMeasurementReading { cached_tokens: 10, ..r }), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::PromptCached { cached: 10 }))]
    #[case::e2e_below_phases(VlSpecDecodeMeasurementSpeculation::On, reading(5, 7, 3, 2, "c").map(|r| VlSpecDecodeMeasurementReading { e2e: Duration::from_millis(116), ..r }), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::EndToEndBelowPhases { .. }))]
    #[case::cut_below_max_tokens(VlSpecDecodeMeasurementSpeculation::On, reading(5, 7, 3, 2, "c").map(|r| VlSpecDecodeMeasurementReading { ended_on_eos: false, ended_on_length: true, ..r }), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::CutBelowMaxTokens { generated: 5, max_tokens: 2048 }))]
    fn a_turn_the_entry_cannot_use_is_refused(
        #[case] speculation: VlSpecDecodeMeasurementSpeculation,
        #[case] last: anyhow::Result<VlSpecDecodeMeasurementReading>,
        #[case] is_expected: fn(&VlSpecDecodeMeasurementError) -> bool,
    ) -> anyhow::Result<()> {
        let last = last?;
        let (good, condition) = match speculation {
            VlSpecDecodeMeasurementSpeculation::Off => (reading(5, 0, 0, 0, "a")?, "off"),
            VlSpecDecodeMeasurementSpeculation::On => (reading(5, 7, 3, 2, "a")?, "on"),
        };
        let mut attempts = 0;
        let err = unobserved(&body(1)?, speculation, &samples()?, |turn| {
            attempts += 1;
            Ok(if (turn.sample, turn.turn) == (1, 1) {
                last.clone()
            } else {
                good.clone()
            })
        })
        .err()
        .context("a refused turn was measured")?;
        assert!(
            format!("{err:#}").starts_with(&format!(
                "{ID} speculation {condition} sample s2 (2/2): turn 2/2: "
            )),
            "{err:#}"
        );
        assert!(
            err.downcast_ref::<VlSpecDecodeMeasurementError>()
                .is_some_and(is_expected),
            "{err:#}"
        );
        // The warm-up and the three turns, once each.
        assert_eq!(attempts, 4);
        Ok(())
    }

    /// A traversal the service would divide by zero on, or one with
    /// speculation on that never drafted, is refused whole.
    #[rstest]
    #[case::no_generation_time(VlSpecDecodeMeasurementSpeculation::Off, reading(5, 0, 0, 0, "a").map(|r| VlSpecDecodeMeasurementReading { predicted: Duration::ZERO, ..r }), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::NoGenerationTime))]
    #[case::no_client_time(VlSpecDecodeMeasurementSpeculation::Off, reading(5, 0, 0, 0, "a").map(|r| VlSpecDecodeMeasurementReading { ttft: Duration::ZERO, decode: Duration::ZERO, e2e: Duration::ZERO, ..r }), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::NoClientTime))]
    #[case::nothing_drafted(VlSpecDecodeMeasurementSpeculation::On, reading(5, 0, 0, 0, "a"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::NothingDrafted))]
    #[case::every_token_accepted(VlSpecDecodeMeasurementSpeculation::On, reading(2, 7, 6, 1, "a"), |e: &VlSpecDecodeMeasurementError| matches!(e, VlSpecDecodeMeasurementError::EveryTokenAccepted { generated: 6 }))]
    fn a_traversal_the_entry_cannot_use_is_refused(
        #[case] speculation: VlSpecDecodeMeasurementSpeculation,
        #[case] every_turn: anyhow::Result<VlSpecDecodeMeasurementReading>,
        #[case] is_expected: fn(&VlSpecDecodeMeasurementError) -> bool,
    ) -> anyhow::Result<()> {
        let every_turn = every_turn?;
        let err = unobserved(&body(1)?, speculation, &samples()?, |_| {
            Ok(every_turn.clone())
        })
        .err()
        .context("a refused traversal was measured")?;
        assert!(
            err.downcast_ref::<VlSpecDecodeMeasurementError>()
                .is_some_and(is_expected),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn a_corpus_with_no_samples_is_refused() -> anyhow::Result<()> {
        let err = unobserved(
            &body(1)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &[],
            |_| Err(fatal(anyhow::anyhow!("sent a turn"))),
        )
        .err()
        .context("an empty corpus was measured")?;
        assert!(
            matches!(
                err.downcast_ref::<VlSpecDecodeMeasurementError>(),
                Some(VlSpecDecodeMeasurementError::NoSamples { .. })
            ),
            "{err:#}"
        );
        Ok(())
    }

    /// A retriable failure is sent again after 5 s times the attempt, up to
    /// three attempts; an unretriable one is not.
    #[rstest]
    #[case::recovers(&[true, true], true, &[5, 10])]
    #[case::exhausted(&[true, true, true], false, &[5, 10])]
    #[case::unretriable(&[false], false, &[])]
    fn a_retriable_failure_is_sent_again(
        #[case] failures: &[bool],
        #[case] recovers: bool,
        #[case] backoff_secs: &[u64],
    ) -> anyhow::Result<()> {
        let slept = RefCell::new(Vec::new());
        let sleep = |d: Duration| slept.borrow_mut().push(d.as_secs());
        let mut failures = failures.iter();
        let got = vl_spec_decode(
            &body(1)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &samples()?,
            &|| Ok(()),
            no_observer(),
            &sleep,
            (),
            |(), turns| match failures.next() {
                Some(&retriable) => Err(VlSpecDecodeMeasurementAttemptFailure {
                    error: anyhow::anyhow!("HTTP 503"),
                    retriable,
                }),
                None => turns
                    .iter()
                    .map(|_| reading(5, 0, 0, 0, "a").map_err(fatal))
                    .collect(),
            },
        );
        match (got, recovers) {
            (Ok(_), true) => {}
            (Err(err), false) => assert_eq!(
                format!("{err:#}"),
                format!("{ID} speculation off warm-up: HTTP 503")
            ),
            (got, recovers) => anyhow::bail!("recovers {recovers}, got {got:?}"),
        }
        assert_eq!(*slept.borrow(), backoff_secs);
        Ok(())
    }

    /// A round retried inside a conversation is sent again whole, with the
    /// same earlier replies, and only its successful attempt is counted.
    #[rstest]
    #[case::reference(1, 4, &["0.0 []", "0.0 []", "1.0 []", "1.1 [r1.0]", "1.1 [r1.0]"])]
    #[case::pair(2, 2, &["0.0 [] | 0.0 []", "0.0 [] | 1.0 []", "0.0 [] | 1.0 []", "1.1 [r1.0]"])]
    fn a_retried_turn_carries_the_same_conversation(
        #[case] batch_size: u32,
        #[case] failing_call: usize,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let mut calls = Vec::new();
        let row = vl_spec_decode(
            &body(batch_size)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &samples()?,
            &|| Ok(()),
            no_observer(),
            &|_| {},
            (),
            |(), turns| {
                calls.push(turns.iter().map(sent).collect::<Vec<_>>().join(" | "));
                if calls.len() == failing_call {
                    return Err(VlSpecDecodeMeasurementAttemptFailure {
                        error: anyhow::anyhow!("HTTP 503"),
                        retriable: true,
                    });
                }
                turns
                    .iter()
                    .map(|turn| {
                        reading(5, 0, 0, 0, &format!("r{}.{}", turn.sample, turn.turn))
                            .map_err(fatal)
                    })
                    .collect()
            },
        )?;
        assert_eq!(calls, expected);
        assert_eq!(row.turns.len(), 3);
        Ok(())
    }

    /// The context is lent to the warm-up and to each turn sent, and what
    /// they recorded is still the caller's after a turn fails.
    #[test]
    fn the_context_outlives_a_failed_turn() -> anyhow::Result<()> {
        let mut recorded = Vec::new();
        let got = vl_spec_decode(
            &body(1)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &samples()?,
            &|| Ok(()),
            no_observer(),
            &|_| {},
            &mut recorded,
            |recorded, turns| {
                turns
                    .iter()
                    .map(|turn| {
                        recorded.push(format!("{}.{}", turn.sample, turn.turn));
                        if recorded.len() == 3 {
                            return Err(fatal(anyhow::anyhow!("the server went away")));
                        }
                        reading(5, 0, 0, 0, "a").map_err(fatal)
                    })
                    .collect()
            },
        );
        let err = got.err().context("the third call should fail")?;
        assert!(
            format!("{err:#}").contains("the server went away"),
            "{err:#}"
        );
        assert_eq!(recorded, ["0.0", "0.0", "1.0"]);
        Ok(())
    }

    /// The warm-up on the corpus's first turn, the gate, the start, then each
    /// sample gated and its turns sent with the earlier replies, and the end.
    /// A failing step abandons the cell there, named by the marker and, inside
    /// the traversal, by the sample and turn.
    #[rstest]
    #[case::none(None, "", &["attempt 0.0 []", "gate", "start", "attempt 0.0 []", "gate", "attempt 1.0 []", "attempt 1.1 [r1.0]", "end"])]
    #[case::warm_up(Some(("attempt", 1)), " warm-up: ", &["attempt 0.0 []"])]
    #[case::first_gate(Some(("gate", 1)), ": ", &["attempt 0.0 []", "gate"])]
    #[case::start(Some(("start", 1)), ": ", &["attempt 0.0 []", "gate", "start"])]
    #[case::second_gate(Some(("gate", 2)), " sample s2 (2/2): ", &["attempt 0.0 []", "gate", "start", "attempt 0.0 []", "gate"])]
    #[case::turn(Some(("attempt", 4)), " sample s2 (2/2): turn 2/2: ", &["attempt 0.0 []", "gate", "start", "attempt 0.0 []", "gate", "attempt 1.0 []", "attempt 1.1 [r1.0]"])]
    #[case::end(Some(("end", 1)), ": ", &["attempt 0.0 []", "gate", "start", "attempt 0.0 []", "gate", "attempt 1.0 []", "attempt 1.1 [r1.0]", "end"])]
    fn the_steps_run_in_order_until_one_fails(
        #[case] failing: Option<(&str, usize)>,
        #[case] at: &str,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let events = RefCell::new(Vec::<String>::new());
        let step = |event: String| {
            let name = event.split(' ').next().unwrap_or_default().to_owned();
            let mut events = events.borrow_mut();
            events.push(event);
            let nth = events
                .iter()
                .filter(|e| e.split(' ').next() == Some(name.as_str()))
                .count();
            anyhow::ensure!(failing != Some((name.as_str(), nth)), "{name} failed");
            Ok(())
        };
        let observer = MeasurementObserver {
            on_before_rep_start: &|| step("start".to_owned()),
            on_after_rep_end: &|| step("end".to_owned()),
            ..MeasurementObserver::QUIET
        };
        let got = vl_spec_decode(
            &body(1)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &samples()?,
            &|| step("gate".to_owned()),
            observer,
            &|_| {},
            (),
            |(), turns| {
                turns
                    .iter()
                    .map(|turn| {
                        step(format!("attempt {}", sent(turn))).map_err(fatal)?;
                        reading(5, 0, 0, 0, &format!("r{}.{}", turn.sample, turn.turn))
                            .map_err(fatal)
                    })
                    .collect()
            },
        );
        match (got, failing) {
            (Ok(_), None) => {}
            (Err(err), Some((name, _))) => assert_eq!(
                format!("{err:#}"),
                format!("{ID} speculation off{at}{name} failed")
            ),
            (got, failing) => anyhow::bail!("failing {failing:?}, got {got:?}"),
        }
        assert_eq!(*events.borrow(), expected);
        Ok(())
    }

    /// Three samples, the middle one a two-turn conversation.
    fn three_samples() -> anyhow::Result<[VlSpecDecodeMeasurementSample<'static>; 3]> {
        Ok([
            VlSpecDecodeMeasurementSample {
                id: "s1",
                turns: NonZeroUsize::try_from(1)?,
            },
            VlSpecDecodeMeasurementSample {
                id: "s2",
                turns: NonZeroUsize::try_from(2)?,
            },
            VlSpecDecodeMeasurementSample {
                id: "s3",
                turns: NonZeroUsize::try_from(1)?,
            },
        ])
    }

    /// A round of turns per attempt, a sample leaving once its turns are done;
    /// the warm-up repeats the corpus's first turn once per sample of the
    /// first batch, and the gate runs before each batch.
    #[rstest]
    #[case::reference(1, &["0.0 []", "gate", "start", "0.0 []", "gate", "1.0 []", "1.1 [r1.0]", "gate", "2.0 []", "end"])]
    #[case::pair(2, &["0.0 [] | 0.0 []", "gate", "start", "0.0 [] | 1.0 []", "1.1 [r1.0]", "gate", "2.0 []", "end"])]
    #[case::whole_corpus(4, &["0.0 [] | 0.0 [] | 0.0 []", "gate", "start", "0.0 [] | 1.0 [] | 2.0 []", "1.1 [r1.0]", "end"])]
    fn a_batch_sends_its_samples_together(
        #[case] batch_size: u32,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let events = RefCell::new(Vec::<String>::new());
        let observer = MeasurementObserver {
            on_before_rep_start: &|| {
                events.borrow_mut().push("start".to_owned());
                Ok(())
            },
            on_after_rep_end: &|| {
                events.borrow_mut().push("end".to_owned());
                Ok(())
            },
            ..MeasurementObserver::QUIET
        };
        vl_spec_decode(
            &body(batch_size)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &three_samples()?,
            &|| {
                events.borrow_mut().push("gate".to_owned());
                Ok(())
            },
            observer,
            &|_| {},
            (),
            |(), turns| {
                events
                    .borrow_mut()
                    .push(turns.iter().map(sent).collect::<Vec<_>>().join(" | "));
                turns
                    .iter()
                    .map(|turn| {
                        reading(5, 0, 0, 0, &format!("r{}.{}", turn.sample, turn.turn))
                            .map_err(fatal)
                    })
                    .collect()
            },
        )?;
        assert_eq!(*events.borrow(), expected);
        Ok(())
    }

    /// Batching changes how turns are sent, not the row: only the summary's
    /// batch size and wall clock differ.
    #[test]
    fn a_batch_reduces_to_the_serial_row() -> anyhow::Result<()> {
        let row = |batch_size| -> anyhow::Result<VlSpecDecodeMeasurementRow> {
            unobserved(
                &body(batch_size)?,
                VlSpecDecodeMeasurementSpeculation::On,
                &three_samples()?,
                |turn| {
                    let generated =
                        i32::try_from(3 + turn.sample + turn.turn).map_err(|e| fatal(e.into()))?;
                    reading(
                        generated,
                        7,
                        3,
                        2,
                        &format!("r{}.{}", turn.sample, turn.turn),
                    )
                    .map_err(fatal)
                },
            )
        };
        let without_timing = |mut row: VlSpecDecodeMeasurementRow| {
            row.summary.retain(|key, _| {
                !matches!(key.as_str(), "batch_size" | "wall_ms" | "aggregate_tok_s")
            });
            row
        };
        let (batched, serial) = (row(2)?, row(1)?);
        assert_eq!(batched.summary.get("batch_size"), Some(&Value::from(2)));
        assert_eq!(serial.summary.get("batch_size"), Some(&Value::from(1)));
        assert_eq!(without_timing(batched), without_timing(serial));
        Ok(())
    }

    /// A short answer fails the round, named by batch; a refused turn is named
    /// by its sample too.
    #[rstest]
    #[case::short(
        None,
        "{ID} speculation off samples 1-2/3: turn 1/2: engine answered 1 turns, expected 2"
    )]
    #[case::refused(
        Some(1),
        "{ID} speculation off samples 1-2/3: sample s2 (2/3): turn 1/2: engine drafted 7 tokens with speculation off: a speculative measurement under the baseline's name"
    )]
    fn a_batch_failure_names_the_batch_and_sample(
        #[case] drafting: Option<usize>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = vl_spec_decode(
            &body(2)?,
            VlSpecDecodeMeasurementSpeculation::Off,
            &three_samples()?,
            &|| Ok(()),
            no_observer(),
            &|_| {},
            (),
            |(), turns| {
                let mut readings = turns
                    .iter()
                    .map(|turn| {
                        let drafted = if Some(turn.sample) == drafting { 7 } else { 0 };
                        reading(5, drafted, 0, 0, "a").map_err(fatal)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // Past the warm-up, a short answer drops the last reading.
                if drafting.is_none() && turns.iter().any(|turn| turn.sample > 0) {
                    readings.pop();
                }
                Ok(readings)
            },
        )
        .err()
        .context("a failed batch was measured")?;
        assert_eq!(format!("{err:#}"), expected.replace("{ID}", ID));
        Ok(())
    }
}
