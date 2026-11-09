//! The corpus traversal shared by every engine: the conversation each turn
//! sends, the send-and-read around it, and the reduction to a row.
//!
//! Shared because two rows are comparable only if both runs saw the same
//! conversation, and a copy per engine makes that an invariant nothing
//! enforces. An engine contributes its request [`Envelope`] and nothing else
//! about the turn. [`Params`] and [`run`] are here too, so every engine
//! refuses, times out and reports a failure the same way.

use std::cell::Cell;
use std::io::BufReader;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};
use std::thread::ScopedJoinHandle;
use std::time::{Duration, Instant};

use anyhow::Context;
use base64::Engine;
use reqwest::blocking::Client;
use serde_json::{Map, Value, json};

use pipette_ops::measurement::{
    self, MeasurementObserver, VlSpecDecodeMeasurementAttemptFailure,
    VlSpecDecodeMeasurementReading, VlSpecDecodeMeasurementRow, VlSpecDecodeMeasurementSample,
    VlSpecDecodeMeasurementSpeculation,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::vl_spec_decode::{
    Progress, Speculation, StreamedTurn, TimingsSource, TurnFailure, TurnRecord, counters,
    log_progress, output_digest, read_chat_stream, text_sha256, with_retries,
};
use pipette_plan_types::{
    PositiveSeconds, SamplingTemperature, Sha256, TurnObservation, VlSpecDecodeBenchmark,
};

use crate::{VlSpecDecodeCorpus, VlSpecDecodeCorpusPart, VlSpecDecodeCorpusSample};

/// The only endpoint that runs an image through the vision tower, and the
/// one every engine's server answers on.
pub const ENDPOINT: &str = "/v1/chat/completions";

/// Per-turn HTTP timeout per slot when the cell pins none: a vision prefill
/// plus up to `parameter_max_tokens` decoded with speculation off, so ten
/// minutes, not a text cell's two.
pub const DEFAULT_TURN_TIMEOUT_SECS: u64 = 600;

/// One send of `messages` through an engine's request envelope, capped at
/// `max_tokens`.
///
/// `Sync` because a batched traversal sends from every slot at once.
pub type Send<'a> =
    &'a (dyn Fn(Vec<Value>, u32) -> std::result::Result<TurnRecord, TurnFailure> + Sync);

/// What the benchmark and the cell's flags fix about a traversal, the same
/// on every engine.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub speculation: Speculation,
    pub max_tokens: u32,
    pub batch_size: NonZeroUsize,
    /// At least 1: the benchmark's type refuses 0.
    pub draft_length: u32,
    pub temperature: SamplingTemperature,
    /// One turn's budget, see [`turn_timeout`].
    pub http_timeout: Duration,
}

impl Params {
    /// Refuses parameters that cannot run.
    pub fn new(
        benchmark: &VlSpecDecodeBenchmark,
        http_timeout: Option<PositiveSeconds>,
        speculation: Speculation,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            speculation,
            max_tokens: benchmark.parameter_max_tokens.into_inner().unsigned_abs(),
            batch_size: benchmark.parameter_batch_size.in_flight()?,
            draft_length: benchmark.parameter_draft_length.into_inner().unsigned_abs(),
            temperature: benchmark.parameter_temperature,
            http_timeout: turn_timeout(http_timeout, benchmark.parameter_batch_size.as_u32()),
        })
    }
}

/// Each sample is one attempt for `attempt_timeout_seconds`.
fn begin_sample(on_attempt_start: &(dyn Fn(&str) + Sync), speculation: Speculation, number: usize) {
    on_attempt_start(&format!("{}, sample {number}", speculation.label()));
}

/// The shared HTTP-timeout rule at [`DEFAULT_TURN_TIMEOUT_SECS`].
fn turn_timeout(pinned: Option<PositiveSeconds>, slots: u32) -> Duration {
    pipette_ops::measurement::http_timeout_with_default(pinned, slots, DEFAULT_TURN_TIMEOUT_SECS)
}

/// An engine's half of a turn: the request fields only its server
/// understands, and where its counters are read from.
#[derive(Debug, Clone)]
pub struct Envelope {
    extras: Map<String, Value>,
    source: TimingsSource,
    temperature: SamplingTemperature,
}

impl Envelope {
    /// `extras` must be a JSON object; it is merged into every turn's body.
    pub fn new(
        extras: Value,
        source: TimingsSource,
        temperature: SamplingTemperature,
    ) -> anyhow::Result<Self> {
        let Value::Object(extras) = extras else {
            anyhow::bail!("an engine's request extras must be a JSON object, got {extras}");
        };
        Ok(Self {
            extras,
            source,
            temperature,
        })
    }

    pub fn source(&self) -> TimingsSource {
        self.source
    }

    /// The engine's extras under the shared protocol fields, which win a key
    /// both name. Streamed because TTFT on the client's clock needs the first
    /// token seen arriving.
    pub fn body(&self, messages: Vec<Value>, max_tokens: u32) -> Value {
        let mut body = self.extras.clone();
        body.extend([
            ("messages".to_owned(), Value::Array(messages)),
            ("temperature".to_owned(), json!(self.temperature.as_f64())),
            ("max_tokens".to_owned(), json!(max_tokens)),
            ("stream".to_owned(), json!(true)),
        ]);
        Value::Object(body)
    }

    /// One attempt at a turn against the server at `base_url`.
    pub fn send(
        &self,
        client: &Client,
        base_url: &str,
        messages: Vec<Value>,
        max_tokens: u32,
    ) -> std::result::Result<TurnRecord, TurnFailure> {
        one_attempt(
            client,
            base_url,
            ENDPOINT,
            &self.body(messages, max_tokens),
            self.source,
        )
    }

    /// [`Self::send`] without the record check, for a caller that checks the
    /// stream itself.
    pub fn stream(
        &self,
        client: &Client,
        base_url: &str,
        messages: Vec<Value>,
        max_tokens: u32,
    ) -> std::result::Result<StreamedTurn, TurnFailure> {
        stream_attempt(
            client,
            base_url,
            ENDPOINT,
            &self.body(messages, max_tokens),
            self.source,
        )
    }
}

/// Traverse the corpus against a server that is up and reduce to the row, the
/// corpus's digest included. The server, its output and the run's provenance
/// are the caller's to record on the response.
pub fn run(
    params: &Params,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    send: Send<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    let response = traverse(params, corpus, readiness_gate, observer, send)
        .with_context(|| format!("{} failed", params.speculation.label()))?
        .reduce(params.speculation, params.batch_size)?;
    Ok(response.dataset_digest(corpus.digest().clone()))
}

/// One send of a turn's messages, capped at `max_tokens`, read back unchecked
/// for the typed entry to check.
pub type Stream<'a> = &'a dyn Fn(Vec<Value>, u32) -> std::result::Result<StreamedTurn, TurnFailure>;

/// [`run`] for a batch of one through the typed [`measurement::vl_spec_decode()`]
/// entry, which checks and sums each turn. Each sample's attempt starts in the
/// gate, which the entry calls before each sample, as [`run`] does.
pub fn measure(
    benchmark: &VlSpecDecodeBenchmark,
    params: &Params,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    stream: Stream<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    let row = measure_row(benchmark, params, corpus, readiness_gate, observer, stream)
        .with_context(|| format!("{} failed", params.speculation.label()))?;
    Ok(RunResponseBuilder::new(row.result_data)
        .traversal(row.output_digest, row.turns)
        .dataset_digest(corpus.digest().clone())
        .client_extras(row.summary))
}

fn measure_row(
    benchmark: &VlSpecDecodeBenchmark,
    params: &Params,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    stream: Stream<'_>,
) -> anyhow::Result<VlSpecDecodeMeasurementRow> {
    let batch_size = benchmark.parameter_batch_size.in_flight()?;
    anyhow::ensure!(
        batch_size.get() == 1,
        "turn::measure sends one conversation per call; a batch of {batch_size} runs through \
         turn::run"
    );
    let samples = corpus
        .samples()
        .iter()
        .map(|sample| {
            Ok(VlSpecDecodeMeasurementSample {
                id: &sample.id,
                turns: NonZeroUsize::new(sample.turns.len())
                    .with_context(|| format!("corpus sample {} holds no turns", sample.id))?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let gated = Cell::new(0);
    let gate = || {
        gated.set(gated.get() + 1);
        readiness_gate()?;
        begin_sample(observer.on_attempt_start, params.speculation, gated.get());
        Ok(())
    };
    // The entry starts the rep after sample 1's gate, which would rename it.
    let on_before_rep_start = || {
        (observer.on_before_rep_start)()?;
        begin_sample(observer.on_attempt_start, params.speculation, gated.get());
        Ok(())
    };
    let observer = MeasurementObserver {
        on_before_rep_start: &on_before_rep_start,
        ..observer
    };
    let speculation = match params.speculation {
        Speculation::Off => VlSpecDecodeMeasurementSpeculation::Off,
        Speculation::On => VlSpecDecodeMeasurementSpeculation::On,
    };
    let fatal = |error: anyhow::Error| VlSpecDecodeMeasurementAttemptFailure {
        error,
        retriable: false,
    };
    measurement::vl_spec_decode(
        benchmark,
        speculation,
        &samples,
        &gate,
        observer,
        &std::thread::sleep,
        (),
        |(), turns| {
            let [turn] = turns else {
                return Err(fatal(anyhow::anyhow!(
                    "the entry sent {} turns to a cell that sends one",
                    turns.len()
                )));
            };
            let sample = corpus
                .samples()
                .get(turn.sample)
                .context("the entry sent a sample the corpus does not hold")
                .map_err(fatal)?;
            let replies = turn
                .earlier
                .iter()
                .map(|reading| reading.text.as_str())
                .collect::<Vec<_>>();
            let messages = chat_messages(corpus, sample, turn.turn, &replies).map_err(fatal)?;
            // The cap the entry checks a `length` stop against.
            let streamed = stream(
                messages,
                benchmark.parameter_max_tokens.into_inner().unsigned_abs(),
            )
            .map_err(|sent| VlSpecDecodeMeasurementAttemptFailure {
                error: sent.error,
                retriable: sent.retriable,
            })?;
            let reading = VlSpecDecodeMeasurementReading::from_stream(streamed)
                .map_err(|e| fatal(e.into()))?;
            Ok(vec![reading])
        },
    )
}

pub(crate) struct Traversal {
    pub turns: Vec<TurnRecord>,
    pub observations: Vec<TurnObservation>,
    pub output_digest: Sha256,
    /// Gate cleared to last turn landed. The aggregate token rate divides by
    /// this, since per-turn sums cannot say how much the turns overlapped.
    pub wall: Duration,
}

impl Traversal {
    /// The row, with [`Self::summary`] as its client extras. The engine adds
    /// its server and provenance.
    pub(crate) fn reduce(
        self,
        speculation: Speculation,
        batch_size: NonZeroUsize,
    ) -> anyhow::Result<RunResponseBuilder> {
        let result_data = counters(speculation, &self.turns)?;
        let summary = self.summary(speculation, batch_size);
        tracing::info!(
            cell = speculation.label(),
            summary = %serde_json::Value::Object(summary.clone()),
            "traversal summary"
        );
        Ok(RunResponseBuilder::new(result_data)
            .traversal(self.output_digest, self.observations)
            .client_extras(summary))
    }

    /// What no result field carries: the wall clock and aggregate token
    /// rate, and with speculation on the forward count beside three accept
    /// lengths: pooled, per request, and the service's own from the two
    /// counters, which reads high by the discarded drafts.
    fn summary(&self, speculation: Speculation, batch_size: NonZeroUsize) -> Map<String, Value> {
        let generated: u64 = self.turns.iter().map(|t| t.generated_tokens).sum();
        let wall_ms = self.wall.as_secs_f64() * 1000.0;
        let aggregate_tok_s = if wall_ms > 0.0 {
            generated as f64 / wall_ms * 1000.0
        } else {
            0.0
        };
        let mut out = Map::from_iter([
            ("turns".to_owned(), self.turns.len().into()),
            ("batch_size".to_owned(), batch_size.get().into()),
            ("generated_tokens".to_owned(), generated.into()),
            ("wall_ms".to_owned(), wall_ms.into()),
            ("aggregate_tok_s".to_owned(), aggregate_tok_s.into()),
        ]);
        if speculation == Speculation::On {
            let accepted: u64 = self.turns.iter().map(|t| t.accepted_tokens).sum();
            let decode: u64 = self.turns.iter().map(TurnRecord::decode_tokens).sum();
            let forwards: u64 = self.turns.iter().map(|t| t.counted_forwards).sum();
            let per_request = self
                .turns
                .iter()
                .map(TurnRecord::accept_length)
                .sum::<f64>()
                / self.turns.len().max(1) as f64;
            out.extend([
                ("decode_forwards".to_owned(), forwards.into()),
                ("accepted_tokens".to_owned(), accepted.into()),
                (
                    "accept_length_pooled".to_owned(),
                    (decode as f64 / forwards.max(1) as f64).into(),
                ),
                ("accept_length_per_request".to_owned(), per_request.into()),
                (
                    "accept_length_from_counters".to_owned(),
                    (generated as f64 / generated.saturating_sub(accepted).max(1) as f64).into(),
                ),
            ]);
        }
        out
    }
}

/// Walk the whole corpus through a server that is already up, with
/// `batch_size` samples in flight.
///
/// The warm-up runs one untimed turn per slot on the corpus's own first turn,
/// so kernel selection and the vision tower's first-shape cost are paid on a
/// real sample's shapes, by every slot. A sample's turns run back to back: a
/// multi-turn sample is one conversation.
///
/// At `batch_size = 1` the readiness gate runs before every sample, holding
/// each to the same host state. With more in flight it runs once: the host is
/// never idle between samples, so a per-sample gate would only stall the
/// other slots and measure the stall.
///
/// The observer brackets the whole traversal, the unit two thermal readings
/// can describe.
pub(crate) fn traverse(
    params: &Params,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    send: Send<'_>,
) -> anyhow::Result<Traversal> {
    let Params {
        speculation,
        max_tokens,
        batch_size,
        ..
    } = *params;
    let first = corpus
        .samples()
        .first()
        .context("the corpus holds no samples")?;
    let slots = batch_size.get();
    tracing::info!(
        cell = speculation.label(),
        samples = corpus.samples().len(),
        turns = corpus.turns(),
        max_tokens,
        batch_size = batch_size.get(),
        "traversing the corpus"
    );
    tracing::info!(cell = speculation.label(), slots, "warm-up run");
    warm_up(corpus, first, max_tokens, slots, send)?;

    readiness_gate()?;
    (observer.on_before_rep_start)()?;
    let started = Instant::now();
    let samples = if slots == 1 {
        sequential(
            speculation,
            corpus,
            max_tokens,
            readiness_gate,
            observer.on_attempt_start,
            send,
        )?
    } else {
        batched(
            speculation,
            corpus,
            max_tokens,
            slots,
            observer.on_attempt_start,
            send,
        )?
    };
    let wall = started.elapsed();
    (observer.on_after_rep_end)()?;

    let (turns, observations) = corpus.samples().iter().zip(samples).try_fold(
        (
            Vec::with_capacity(corpus.turns()),
            Vec::with_capacity(corpus.turns()),
        ),
        |(mut turns, mut observations), (sample, sample_turns)| {
            observations.extend(
                sample_turns
                    .iter()
                    .enumerate()
                    .map(|(turn, record)| observation(&sample.id, turn, record))
                    .collect::<anyhow::Result<Vec<_>>>()?,
            );
            turns.extend(sample_turns);
            anyhow::Ok((turns, observations))
        },
    )?;
    Ok(Traversal {
        output_digest: output_digest(&turns)?,
        turns,
        observations,
        wall,
    })
}

/// Concurrent so a batched server pays its per-slot first-shape cost now,
/// not on the first measured sample each slot takes.
fn warm_up(
    corpus: &VlSpecDecodeCorpus,
    first: &VlSpecDecodeCorpusSample,
    max_tokens: u32,
    slots: usize,
    send: Send<'_>,
) -> anyhow::Result<()> {
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..slots)
            .map(|_| {
                scope.spawn(move || one_turn(corpus, first, 0, &[], max_tokens, send).map(drop))
            })
            .collect();
        handles.into_iter().map(join).collect::<Vec<_>>()
    });
    outcomes
        .into_iter()
        .collect::<anyhow::Result<()>>()
        .context("warm-up turn failed")
}

/// One sample at a time, gated before each after the first (the traversal's
/// gate covers that one).
fn sequential(
    speculation: Speculation,
    corpus: &VlSpecDecodeCorpus,
    max_tokens: u32,
    readiness_gate: ReadinessGate,
    on_attempt_start: &(dyn Fn(&str) + Sync),
    send: Send<'_>,
) -> anyhow::Result<Vec<Vec<TurnRecord>>> {
    let total = corpus.samples().len();
    let mut progress = Progress::default();
    corpus
        .samples()
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            if index > 0 {
                readiness_gate()?;
            }
            begin_sample(on_attempt_start, speculation, index + 1);
            let turns = one_sample(speculation, corpus, sample, index, max_tokens, send)?;
            progress.add_sample(&turns);
            log_progress(speculation, progress, total);
            Ok(turns)
        })
        .collect()
}

/// `slots` samples in flight, each slot taking the next in file order. No
/// gate between samples (see [`traverse`]).
///
/// A failed sample stops the traversal: the other slots finish the sample
/// they hold and take no more. Results come back in file order.
fn batched(
    speculation: Speculation,
    corpus: &VlSpecDecodeCorpus,
    max_tokens: u32,
    slots: usize,
    on_attempt_start: &(dyn Fn(&str) + Sync),
    send: Send<'_>,
) -> anyhow::Result<Vec<Vec<TurnRecord>>> {
    let total = corpus.samples().len();
    let shared = Mutex::new(Shared {
        next: 0,
        stopped: false,
        done: (0..total).map(|_| None).collect(),
        progress: Progress::default(),
    });
    let lock = || shared.lock().unwrap_or_else(PoisonError::into_inner);
    let run_slot = || -> anyhow::Result<()> {
        loop {
            let index = {
                let mut shared = lock();
                if shared.stopped {
                    return Ok(());
                }
                shared.next += 1;
                shared.next - 1
            };
            let Some(sample) = corpus.samples().get(index) else {
                return Ok(());
            };
            // shortcut: the slots share one clock, so a stuck sample is caught
            // only once the other slots stop starting samples; give each slot
            // its own clock if batched traversals need an exact per-sample limit.
            begin_sample(on_attempt_start, speculation, index + 1);
            let outcome = one_sample(speculation, corpus, sample, index, max_tokens, send);
            let mut shared = lock();
            let turns = match outcome {
                Ok(turns) => turns,
                Err(e) => {
                    shared.stopped = true;
                    return Err(e);
                }
            };
            shared.progress.add_sample(&turns);
            log_progress(speculation, shared.progress, total);
            if let Some(slot) = shared.done.get_mut(index) {
                *slot = Some(turns);
            }
        }
    };
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..slots).map(|_| scope.spawn(run_slot)).collect();
        handles.into_iter().map(join).collect::<Vec<_>>()
    });
    outcomes.into_iter().collect::<anyhow::Result<()>>()?;
    let shared = shared.into_inner().unwrap_or_else(PoisonError::into_inner);
    shared
        .done
        .into_iter()
        .enumerate()
        .map(|(index, turns)| {
            // Unreachable once every slot returned `Ok`: a slot only
            // returns without an error when the counter passed the end,
            // and it stores before it takes again.
            turns.with_context(|| format!("sample {} was never traversed", index + 1))
        })
        .collect()
}

/// The slots' one lock, so a take and its stop check are one step.
struct Shared {
    next: usize,
    stopped: bool,
    done: Vec<Option<Vec<TurnRecord>>>,
    progress: Progress,
}

/// A slot's outcome, with a panic (a dependency's: this crate denies its own)
/// reported as an error rather than propagated.
fn join(handle: ScopedJoinHandle<'_, anyhow::Result<()>>) -> anyhow::Result<()> {
    handle
        .join()
        .unwrap_or_else(|_| Err(anyhow::anyhow!("a traversal slot panicked")))
}

/// No readiness wait in here: the conversation is the unit.
fn one_sample(
    speculation: Speculation,
    corpus: &VlSpecDecodeCorpus,
    sample: &VlSpecDecodeCorpusSample,
    index: usize,
    max_tokens: u32,
    send: Send<'_>,
) -> anyhow::Result<Vec<TurnRecord>> {
    (0..sample.turns.len())
        .try_fold(Vec::new(), |mut turns, turn| {
            let record = one_turn(corpus, sample, turn, &turns, max_tokens, send)
                .with_context(|| format!("turn {}/{}", turn + 1, sample.turns.len()))?;
            turns.push(record);
            anyhow::Ok(turns)
        })
        .with_context(|| {
            format!(
                "{}: sample {} ({}/{}) failed",
                speculation.label(),
                sample.id,
                index + 1,
                corpus.samples().len()
            )
        })
}

fn one_turn(
    corpus: &VlSpecDecodeCorpus,
    sample: &VlSpecDecodeCorpusSample,
    index: usize,
    earlier: &[TurnRecord],
    max_tokens: u32,
    send: Send<'_>,
) -> anyhow::Result<TurnRecord> {
    let replies = earlier
        .iter()
        .map(|record| record.text.as_str())
        .collect::<Vec<_>>();
    let messages = chat_messages(corpus, sample, index, &replies)?;
    with_retries(
        &mut |_| {
            let record = send(messages.clone(), max_tokens)?;
            // Unretriable, like `validate` in `one_attempt`: the same
            // conversation runs into the same cut.
            record
                .check_length(max_tokens)
                .map_err(|error| TurnFailure {
                    error,
                    retriable: false,
                })?;
            Ok(record)
        },
        &std::thread::sleep,
    )
}

fn observation(
    sample_id: &str,
    turn: usize,
    record: &TurnRecord,
) -> anyhow::Result<TurnObservation> {
    Ok(TurnObservation {
        sample_id: sample_id.to_owned(),
        turn: u32::try_from(turn).context("turn index exceeds u32")?,
        text_sha256: text_sha256(&record.text)?,
        generated_tokens: record.generated_tokens,
        drafted_tokens: record.drafted_tokens,
        accepted_tokens: record.accepted_tokens,
        counted_forwards: record.counted_forwards,
        ended_on_eos: record.ended_on_eos,
        prompt_ms: record.prompt_ms,
        predicted_ms: record.predicted_ms,
        ttft_ms: record.ttft_ms,
        decode_ms: record.decode_ms,
        e2e_ms: record.e2e_ms,
    })
}

/// The messages for turn `index` of `sample`: the system prompt if not
/// blank, then every turn through this one, each followed by its reply.
///
/// `replies` are the earlier turns'; the current turn has none yet, hence
/// `get` rather than an index.
pub(crate) fn chat_messages(
    corpus: &VlSpecDecodeCorpus,
    sample: &VlSpecDecodeCorpusSample,
    index: usize,
    replies: &[&str],
) -> anyhow::Result<Vec<Value>> {
    let mut messages: Vec<Value> = Vec::with_capacity(index * 2 + 2);
    if !sample.system.trim().is_empty() {
        messages.push(json!({ "role": "system", "content": sample.system }));
    }
    sample
        .turns
        .iter()
        .take(index + 1)
        .enumerate()
        .try_for_each(|(turn, parts)| {
            messages.push(json!({ "role": "user", "content": content_parts(corpus, parts)? }));
            if let Some(reply) = replies.get(turn) {
                messages.push(json!({ "role": "assistant", "content": reply }));
            }
            anyhow::Ok(())
        })?;
    Ok(messages)
}

/// One user turn's parts as OpenAI-style multimodal content, images inlined
/// as base64 data URLs. Encoded here, before the send starts its clock; the
/// measured span still carries the larger upload.
fn content_parts(
    corpus: &VlSpecDecodeCorpus,
    parts: &[VlSpecDecodeCorpusPart],
) -> anyhow::Result<Value> {
    let content = parts
        .iter()
        .map(|part| match part {
            VlSpecDecodeCorpusPart::Text { text } => Ok(json!({ "type": "text", "text": text })),
            VlSpecDecodeCorpusPart::Image { path, mime, .. } => {
                let bytes = std::fs::read(corpus.media_path(path)?)
                    .with_context(|| format!("failed to read corpus media {path}"))?;
                let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
                Ok(json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{mime};base64,{encoded}") }
                }))
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!content.is_empty(), "a corpus turn holds no content parts");
    Ok(Value::Array(content))
}

fn one_attempt(
    client: &Client,
    base_url: &str,
    endpoint: &str,
    body: &Value,
    source: TimingsSource,
) -> std::result::Result<TurnRecord, TurnFailure> {
    let record = stream_attempt(client, base_url, endpoint, body, source)?.into_record();
    // Unretriable: counters that contradict each other will do so again on
    // the next send.
    record.validate().map_err(|error| TurnFailure {
        error,
        retriable: false,
    })?;
    Ok(record)
}

/// Unchecked. The clock starts here, so a failed attempt leaves nothing in
/// the sums.
fn stream_attempt(
    client: &Client,
    base_url: &str,
    endpoint: &str,
    body: &Value,
    source: TimingsSource,
) -> std::result::Result<StreamedTurn, TurnFailure> {
    let started = Instant::now();
    let response = pipette_http::send(client.post(format!("{base_url}{endpoint}")).json(body))
        .map_err(|error| TurnFailure {
            // A dropped connection or a 5xx may pass on a retry; a 4xx is
            // this request being wrong and would fail the same way again.
            retriable: if let pipette_http::Error::Status { status, .. } = &error {
                status.is_server_error()
            } else {
                true
            },
            error: anyhow::Error::new(error).context(format!("failed to call {endpoint}")),
        })?;
    // A stream that stopped early or arrived malformed is a transport
    // symptom on this side of a 200, so it is retried like one.
    read_chat_stream(BufReader::new(response), started, Instant::now, source).map_err(|error| {
        TurnFailure {
            error,
            retriable: true,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex, PoisonError};
    use std::time::Duration;

    use httpmock::MockServer;
    use serde_json::{Value, json};

    use pipette_ops::measurement::MeasurementObserver;
    use pipette_ops::vl_spec_decode::{
        ChatTimings, Speculation, StreamedTurn, TimingsSource, TurnFailure, TurnRecord,
        output_digest,
    };
    use pipette_plan_types::{
        BenchmarkFlags, BenchmarkResultData, PositiveSeconds, SamplingTemperature,
        VlSpecDecodeBenchmark,
    };

    use super::{
        Envelope, Params, Traversal, chat_messages, measure, run, stream_attempt, traverse,
    };
    use crate::{VlSpecDecodeCorpus, VlSpecDecodeCorpusSample};

    const UPSTREAM: &str = r#"{
        "source": "git",
        "url": "github.com/acme/corpus",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "identity": "2923603292c7dd463ad01a119f57965817778fc3",
        "digest": "2222222222222222222222222222222222222222222222222222222222222222"
    }"#;

    /// Three samples: one image+text turn under a system prompt, one two-turn
    /// text conversation with none, and one whose system prompt is whitespace.
    fn corpus(dest: &std::path::Path) -> anyhow::Result<VlSpecDecodeCorpus> {
        let manifest = format!(
            r#"{{"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1","upstream":{UPSTREAM},"samples":3,"turns":4,"subsets":{{"GQA":{{"samples":3,"turns":4}}}}}}"#
        );
        let samples = concat!(
            r#"{"id":"img","subset":"GQA","system":"be brief","turns":[[{"type":"image","path":"media/aa.png","mime":"image/png","bytes":3},{"type":"text","text":"what is this?"}]]}"#,
            "\n",
            r#"{"id":"multi","subset":"GQA","system":"","turns":[[{"type":"text","text":"hello"}],[{"type":"text","text":"and again"}]]}"#,
            "\n",
            r#"{"id":"blank","subset":"GQA","system":"   \n ","turns":[[{"type":"text","text":"hi"}]]}"#,
            "\n",
        );
        let mut builder = tar::Builder::new(Vec::new());
        let mut append = |path: &str, body: &[u8]| -> anyhow::Result<()> {
            let mut header = tar::Header::new_gnu();
            header.set_size(body.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, path, body)?;
            Ok(())
        };
        append("manifest.json", manifest.as_bytes())?;
        append("samples.jsonl", samples.as_bytes())?;
        append("media/aa.png", b"png")?;
        let tar = builder.into_inner()?;
        let digest = pipette_plan_types::Sha256::try_new(hex::encode(
            <sha2::Sha256 as sha2::Digest>::digest(&tar),
        ))?;
        Ok(crate::verify_and_open(&tar, &digest, dest)?.0)
    }

    fn sample<'a>(
        corpus: &'a VlSpecDecodeCorpus,
        id: &str,
    ) -> anyhow::Result<&'a VlSpecDecodeCorpusSample> {
        corpus
            .samples()
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| anyhow::anyhow!("no sample {id}"))
    }

    fn record(text: &str) -> TurnRecord {
        TurnRecord {
            prompt_ms: 1.0,
            predicted_ms: 1.0,
            ttft_ms: 1.0,
            decode_ms: 1.0,
            e2e_ms: 1.0,
            generated_tokens: 1,
            drafted_tokens: 0,
            accepted_tokens: 0,
            counted_forwards: 0,
            cached_tokens: 0,
            ended_on_eos: true,
            ended_on_length: false,
            text: text.to_owned(),
        }
    }

    /// Every engine sends this, so a change here changes what every row measured.
    #[test]
    fn an_image_turn_renders_as_a_data_url_beside_its_text() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let messages = chat_messages(&corpus, sample(&corpus, "img")?, 0, &[])?;

        assert_eq!(
            messages,
            vec![
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": [
                    // "png" base64s to "cG5n".
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,cG5n"}},
                    {"type": "text", "text": "what is this?"},
                ]}),
            ]
        );
        Ok(())
    }

    /// Turn `n` carries turns `0..n` and their replies but no reply of its own,
    /// which would measure the wrong prefill.
    #[test]
    fn a_later_turn_carries_the_conversation_before_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let multi = sample(&corpus, "multi")?;

        let first = chat_messages(&corpus, multi, 0, &[])?;
        assert_eq!(
            first,
            vec![json!({"role": "user", "content": [{"type": "text", "text": "hello"}]})],
            "no system prompt, and no reply yet"
        );

        let second = chat_messages(&corpus, multi, 1, &["the first reply"])?;
        assert_eq!(
            second,
            vec![
                json!({"role": "user", "content": [{"type": "text", "text": "hello"}]}),
                json!({"role": "assistant", "content": "the first reply"}),
                json!({"role": "user", "content": [{"type": "text", "text": "and again"}]}),
            ]
        );
        Ok(())
    }

    /// Sent, it would put an empty turn before every sample on an engine that
    /// does not trim, and not on one that does.
    #[test]
    fn a_blank_system_prompt_is_not_sent() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let messages = chat_messages(&corpus, sample(&corpus, "blank")?, 0, &[])?;
        assert!(
            !messages
                .iter()
                .any(|m| m["role"] == Value::String("system".to_owned())),
            "got {messages:?}"
        );
        Ok(())
    }

    /// A stand-in `send` that counts how many turns are in flight at once.
    ///
    /// The first `slots` calls (the warm-up) meet at a barrier, so the peak is
    /// proven by construction rather than by a sleep a loaded runner could
    /// outlast.
    struct FakeEngine {
        slots: usize,
        calls: AtomicUsize,
        warm_up: Barrier,
        in_flight: AtomicUsize,
        peak: AtomicUsize,
        /// Each request's latest user text, in the order the engine saw them.
        seen: Mutex<Vec<String>>,
    }

    impl FakeEngine {
        fn new(slots: usize) -> Self {
            Self {
                slots,
                calls: AtomicUsize::new(0),
                warm_up: Barrier::new(slots),
                in_flight: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                seen: Mutex::new(Vec::new()),
            }
        }

        fn send(&self, messages: Vec<Value>, _: u32) -> Result<TurnRecord, TurnFailure> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            if self.calls.fetch_add(1, Ordering::SeqCst) < self.slots {
                self.warm_up.wait();
            }
            let last_user = messages
                .iter()
                .rev()
                .find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_array())
                .and_then(|parts| parts.iter().find_map(|p| p["text"].as_str()))
                .unwrap_or_default()
                .to_owned();
            self.seen
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(last_user.clone());
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(record(&format!("reply to {last_user}")))
        }

        fn peak(&self) -> usize {
            self.peak.load(Ordering::SeqCst)
        }
    }

    fn params(speculation: Speculation, batch_size: usize) -> anyhow::Result<Params> {
        Ok(Params {
            speculation,
            max_tokens: 64,
            batch_size: NonZeroUsize::new(batch_size)
                .ok_or_else(|| anyhow::anyhow!("batch size 0"))?,
            draft_length: 8,
            temperature: SamplingTemperature::greedy(),
            http_timeout: Duration::from_secs(1),
        })
    }

    fn benchmark(
        max_tokens: u32,
        draft_length: u32,
        batch_size: u32,
    ) -> anyhow::Result<VlSpecDecodeBenchmark> {
        Ok(serde_json::from_value(json!({
            "benchmark_id": "vl_spec_decode_mmspec_test_gqa_k8",
            "parameter_draft_length": draft_length,
            "parameter_max_tokens": max_tokens,
            "parameter_batch_size": batch_size,
            "dataset": {
                "format": "vl-spec-decode-corpus/1",
                "adapter": "mmspec@1",
                "digest": "1862eedc832710f6c4776472d00d5ee4aa0a16d85d6a9c433398109f90239810",
                "bytes": 15_955_456,
                "upstream": serde_json::from_str::<Value>(UPSTREAM)?,
            },
        }))?)
    }

    #[rstest::rstest]
    #[case::bad_request(Some(400), false)]
    #[case::unavailable(Some(503), true)]
    #[case::no_server(None, true)]
    fn only_a_4xx_is_not_retried(
        #[case] status: Option<u16>,
        #[case] retriable: bool,
    ) -> anyhow::Result<()> {
        let server = MockServer::start();
        let base_url = match status {
            Some(status) => {
                server.mock(|_, then| {
                    then.status(status).body("{\"error\":\"refused\"}");
                });
                server.base_url()
            }
            None => {
                let port = std::net::TcpListener::bind("127.0.0.1:0")?
                    .local_addr()?
                    .port();
                format!("http://127.0.0.1:{port}")
            }
        };
        let http = pipette_http::HttpClient::builder("pipette-test/0")
            .no_proxy()
            .build()?;
        let Err(failure) = stream_attempt(
            http.client(),
            &base_url,
            "/v1/chat/completions",
            &json!({}),
            TimingsSource::Runtime,
        ) else {
            anyhow::bail!("{status:?} must fail the attempt");
        };
        assert_eq!(failure.retriable, retriable, "{:#}", failure.error);
        Ok(())
    }

    #[rstest::rstest]
    #[case::pinned(Some(30), 8, 30)]
    #[case::reference(None, 1, 600)]
    #[case::batched(None, 8, 4800)]
    fn the_turn_timeout_is_pinned_or_ten_minutes_per_slot(
        #[case] pinned: Option<u64>,
        #[case] batch_size: u32,
        #[case] expected_secs: u64,
    ) -> anyhow::Result<()> {
        let flags = BenchmarkFlags::VlSpecDecodeMlxMacosPipetteMlx {
            http_timeout_seconds: pinned.map(PositiveSeconds::try_new).transpose()?,
            attempt_timeout_seconds: None,
            readiness: None,
        };
        let params = Params::new(
            &benchmark(2048, 8, batch_size)?,
            flags.http_timeout(),
            Speculation::On,
        )?;
        assert_eq!(params.http_timeout, Duration::from_secs(expected_secs));
        assert_eq!(params.batch_size.get(), usize::try_from(batch_size)?);
        Ok(())
    }

    #[test]
    fn a_cell_without_benchmark_flags_gets_the_default_timeout() -> anyhow::Result<()> {
        let params = Params::new(&benchmark(2048, 8, 2)?, None, Speculation::Off)?;
        assert_eq!(params.http_timeout, Duration::from_secs(1200));
        Ok(())
    }

    /// An engine's extras cannot override a protocol field.
    #[rstest::rstest]
    #[case::llamacpp(
        json!({"top_k": 1, "cache_prompt": false}),
        json!({"top_k": 1, "cache_prompt": false})
    )]
    #[case::sglang(
        json!({"model": "m", "stream_options": {"include_usage": true},
               "return_spec_tokens_details": true}),
        json!({"model": "m", "stream_options": {"include_usage": true},
               "return_spec_tokens_details": true})
    )]
    #[case::none(json!({}), json!({}))]
    #[case::restated(json!({"temperature": 0.7, "stream": false}), json!({}))]
    fn the_body_is_the_protocol_plus_the_engines_extras(
        #[case] extras: Value,
        #[case] kept: Value,
    ) -> anyhow::Result<()> {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let body = Envelope::new(
            extras,
            TimingsSource::Runtime,
            SamplingTemperature::greedy(),
        )?
        .body(messages.clone(), 7);

        let mut expected = json!({
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 7,
            "stream": true,
        });
        if let (Some(expected), Some(kept)) = (expected.as_object_mut(), kept.as_object()) {
            expected.extend(kept.clone());
        }
        assert_eq!(body, expected);
        Ok(())
    }

    /// Greedy is the default and the only one the identity gate holds for,
    /// but a pinned temperature must not be flattened to it.
    #[rstest::rstest]
    #[case::greedy(0.0)]
    #[case::sampled(0.8)]
    #[case::hottest(2.0)]
    fn the_turn_body_carries_the_benchmarks_temperature(#[case] pinned: f64) -> anyhow::Result<()> {
        let envelope = Envelope::new(
            json!({}),
            TimingsSource::Runtime,
            SamplingTemperature::try_new(pinned)?,
        )?;
        let body = envelope.body(vec![json!({"role": "user", "content": "hi"})], 7);
        assert_eq!(body["temperature"], json!(pinned));
        Ok(())
    }

    #[test]
    fn extras_that_are_not_an_object_are_refused() {
        let refused = Envelope::new(
            json!(["top_k", 1]),
            TimingsSource::Runtime,
            SamplingTemperature::greedy(),
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(refused.contains("must be a JSON object"), "{refused}");
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver::QUIET
    }

    /// Unretriable, so the test does not wait out retries.
    fn server_gone(_: Vec<Value>, _: u32) -> Result<TurnRecord, TurnFailure> {
        Err(TurnFailure {
            error: anyhow::anyhow!("the server went away"),
            retriable: false,
        })
    }

    #[test]
    fn a_failed_traversal_is_reported_under_its_condition() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;

        let outcome = run(
            &params(Speculation::Off, 1)?,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &server_gone,
        );

        let Err(err) = outcome else {
            anyhow::bail!("a failed traversal must fail the run");
        };
        assert_eq!(err.to_string(), "vl_spec_decode (speculation off) failed");
        assert!(
            format!("{err:#}").contains("the server went away"),
            "{err:#}"
        );
        Ok(())
    }

    #[test]
    fn a_traversal_that_succeeds_carries_the_corpus_digest() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let engine = FakeEngine::new(1);

        let response = run(
            &params(Speculation::Off, 1)?,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|messages, max_tokens| engine.send(messages, max_tokens),
        )?
        .build();

        assert_eq!(response.turns.len(), 4);
        assert_eq!(response.dataset_digest.as_ref(), Some(corpus.digest()));
        Ok(())
    }

    /// A reply whose text is the request it answered, so a turn built
    /// differently shows in the digests.
    fn echo(
        drafted: u32,
        accepted: u32,
        forwards: u32,
    ) -> impl Fn(Vec<Value>, u32) -> Result<StreamedTurn, TurnFailure> + Sync {
        move |messages, _| {
            Ok(StreamedTurn {
                content: format!("reply to {}", Value::Array(messages)),
                timings: ChatTimings {
                    prompt_ms: 1.0,
                    predicted_n: 3,
                    predicted_ms: 2.0,
                    draft_n: drafted,
                    draft_n_accepted: accepted,
                    counted_forwards: forwards,
                    cache_n: 0,
                },
                finish_reason: Some("stop".to_owned()),
                ttft: Duration::from_millis(1),
                decode: Duration::from_millis(2),
                e2e: Duration::from_millis(4),
            })
        }
    }

    /// Everything matches but the wall clock, which each measured itself.
    #[rstest::rstest]
    #[case::off(Speculation::Off, 0, 0, 0)]
    #[case::on(Speculation::On, 4, 1, 1)]
    fn measure_gives_the_row_run_gives(
        #[case] speculation: Speculation,
        #[case] drafted: u32,
        #[case] accepted: u32,
        #[case] forwards: u32,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let params = params(speculation, 1)?;
        let echo = echo(drafted, accepted, forwards);
        let legacy = run(
            &params,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|messages, max_tokens| echo(messages, max_tokens).map(StreamedTurn::into_record),
        )?
        .build();
        let entry = measure(
            &benchmark(64, 8, 1)?,
            &params,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &echo,
        )?
        .build();

        let without_wall = |mut extras: serde_json::Map<String, Value>| {
            extras.remove("wall_ms");
            extras.remove("aggregate_tok_s");
            extras
        };
        assert_eq!(entry.result_data, legacy.result_data);
        assert_eq!(entry.output_digest, legacy.output_digest);
        assert_eq!(entry.dataset_digest, legacy.dataset_digest);
        assert_eq!(entry.turns, legacy.turns);
        assert_eq!(
            without_wall(entry.client_extras),
            without_wall(legacy.client_extras)
        );
        Ok(())
    }

    #[test]
    fn measure_refuses_a_batch() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let outcome = measure(
            &benchmark(64, 8, 2)?,
            &params(Speculation::Off, 2)?,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|_, _| {
                Err(TurnFailure {
                    error: anyhow::anyhow!("sent a turn"),
                    retriable: false,
                })
            },
        );
        let Err(err) = outcome else {
            anyhow::bail!("a batched cell was measured");
        };
        assert!(
            format!("{err:#}").contains("a batch of 2 runs through turn::run"),
            "{err:#}"
        );
        Ok(())
    }

    /// Each sample sent is an attempt of its own, sample 1 included, though
    /// the rep's start after its gate begins one too, as the CLI's does.
    #[test]
    fn each_measured_sample_is_its_own_attempt() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let current = Mutex::new(String::new());
        let begin = |what: &str| {
            *current.lock().unwrap_or_else(PoisonError::into_inner) = what.to_owned();
        };
        let observer = MeasurementObserver {
            on_before_rep_start: &|| {
                begin("rep 1");
                Ok(())
            },
            on_attempt_start: &begin,
            ..MeasurementObserver::QUIET
        };
        let sent = Mutex::new(Vec::new());
        let echo = echo(0, 0, 0);
        let _ = measure(
            &benchmark(64, 8, 1)?,
            &params(Speculation::Off, 1)?,
            &corpus,
            &|| Ok(()),
            observer,
            &|messages, max_tokens| {
                let attempt = current
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone();
                sent.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(attempt);
                echo(messages, max_tokens)
            },
        )?;
        let sample = |n: usize| format!("vl_spec_decode (speculation off), sample {n}");
        let sent = sent.into_inner().unwrap_or_else(PoisonError::into_inner);
        // The warm-up's turn, then sample 1's, sample 2's two and sample 3's.
        let expected = [String::new(), sample(1), sample(2), sample(2), sample(3)];
        anyhow::ensure!(sent == expected, "{sent:?}");
        Ok(())
    }

    /// One drafting turn in a source's own stream shape, stopping on
    /// `finish_reason` after `generated` tokens.
    fn stream_body(source: TimingsSource, finish_reason: &str, generated: u32) -> String {
        let events = match source {
            // As `mlx_vlm.server` streams it.
            TimingsSource::Runtime => vec![
                json!({"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {}, "finish_reason": finish_reason}]}),
                json!({"choices": [],
                       "timings": {"prompt_n": 40, "cache_n": 0, "predicted_n": generated,
                                   "prompt_ms": 50.0, "predicted_ms": 90.0,
                                   "draft_rounds": 2, "draft_n": 14, "draft_n_accepted": 6}}),
            ],
            // As `llama-server` streams it: no step count, so the forwards are
            // recovered.
            TimingsSource::RuntimeFixedDraft(_) => vec![
                json!({"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {}, "finish_reason": finish_reason}],
                       "timings": {"prompt_n": 21, "cache_n": 0, "predicted_n": generated,
                                   "prompt_ms": 96.0, "predicted_ms": 61.0,
                                   "draft_n": 16, "draft_n_accepted": 6}}),
            ],
            // As SGLang streams it. Two content chunks keep the client-clock
            // generation time above zero.
            TimingsSource::ClientClock => vec![
                json!({"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}),
                json!({"choices": [{"delta": {"content": "y"}, "finish_reason": finish_reason}],
                       "sglext": {"spec_tokens_details": {"spec_num_proposed_drafts": 18,
                                                          "spec_num_correct_drafts": 6,
                                                          "spec_verify_ct": 2}}}),
                json!({"choices": [], "usage": {"prompt_tokens": 1302,
                                                "completion_tokens": generated}}),
            ],
        };
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .chain(std::iter::once("data: [DONE]\n\n".to_owned()))
            .collect()
    }

    /// Holds on every source, through both `run` and `measure`, and is not retried.
    /// `run` refuses at its warm-up (one send); `measure` does not check its
    /// warm-up, so it refuses at the first counted turn (two sends).
    #[rstest::rstest]
    #[case::length_at_max("length", 64, false)]
    #[case::length_below_max("length", 12, true)]
    #[case::stop_below_max("stop", 12, false)]
    fn a_length_stop_short_of_max_tokens_is_refused(
        #[case] finish_reason: &str,
        #[case] generated: u32,
        #[case] refused: bool,
        #[values(
            TimingsSource::RuntimeFixedDraft(8),
            TimingsSource::Runtime,
            TimingsSource::ClientClock
        )]
        source: TimingsSource,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let server = MockServer::start();
        let sent = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path(super::ENDPOINT);
            then.status(200)
                .body(stream_body(source, finish_reason, generated));
        });
        let http = pipette_http::HttpClient::builder("pipette-test/0")
            .no_proxy()
            .build()?;
        let envelope = Envelope::new(json!({}), source, SamplingTemperature::greedy())?;
        let base_url = server.base_url();
        let params = params(Speculation::On, 1)?;

        let legacy = run(
            &params,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|messages, max_tokens| envelope.send(http.client(), &base_url, messages, max_tokens),
        );
        let legacy_sends = sent.calls();
        let entry = measure(
            &benchmark(params.max_tokens, 8, 1)?,
            &params,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|messages, max_tokens| envelope.stream(http.client(), &base_url, messages, max_tokens),
        );
        let entry_sends = sent.calls() - legacy_sends;

        [
            (legacy.map(drop), legacy_sends, 1),
            (entry.map(drop), entry_sends, 2),
        ]
        .into_iter()
        .try_for_each(|(outcome, sends, refused_after)| {
            match (outcome, refused) {
                (Err(err), true) => {
                    let message = format!("{err:#}");
                    assert!(
                        message.contains(&format!(
                            "at {generated} tokens, short of the 64 max_tokens"
                        )),
                        "{message}"
                    );
                    assert_eq!(sends, refused_after, "{message}");
                }
                // The warm-up and the corpus's four turns.
                (Ok(()), false) => assert_eq!(sends, 5),
                (outcome, refused) => anyhow::bail!("refused {refused}, got {outcome:?}"),
            }
            Ok(())
        })
    }

    /// The whole corpus at `batch_size`, with a gate that counts its calls.
    fn traverse_with(
        corpus: &VlSpecDecodeCorpus,
        engine: &FakeEngine,
        batch_size: usize,
        gate_calls: &AtomicUsize,
    ) -> anyhow::Result<Traversal> {
        traverse(
            &params(Speculation::Off, batch_size)?,
            corpus,
            &|| {
                gate_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            no_observer(),
            &|messages, max_tokens| engine.send(messages, max_tokens),
        )
    }

    fn traversal(turns: Vec<TurnRecord>) -> anyhow::Result<Traversal> {
        Ok(Traversal {
            output_digest: output_digest(&turns)?,
            observations: Vec::new(),
            turns,
            wall: std::time::Duration::from_millis(500),
        })
    }

    fn counted(generated: u64, accepted: u64, counted_forwards: u64) -> TurnRecord {
        TurnRecord {
            predicted_ms: 10.0,
            generated_tokens: generated,
            drafted_tokens: accepted * 2,
            accepted_tokens: accepted,
            counted_forwards,
            ..record("x")
        }
    }

    /// The row is the reduction and nothing an engine adds; the extras carry
    /// three accept lengths: pooled over counted forwards, per request, and
    /// the one the service derives from the counters alone.
    #[test]
    fn a_traversal_reduces_to_the_row_and_its_summary() -> anyhow::Result<()> {
        let traversal = traversal(vec![counted(128, 84, 47), counted(64, 30, 20)])?;
        let digest = traversal.output_digest.clone();
        let response = traversal
            .reduce(
                Speculation::On,
                NonZeroUsize::new(4).ok_or_else(|| anyhow::anyhow!("batch size 0"))?,
            )?
            .build();
        assert_eq!(response.output_digest, Some(digest));
        assert!(response.command.is_empty());
        let BenchmarkResultData::VlSpecDecode {
            vl_spec_decode_generated_tokens,
            vl_spec_decode_accepted_tokens,
            ..
        } = response.result_data
        else {
            anyhow::bail!("expected a vl_spec_decode result");
        };
        assert_eq!(vl_spec_decode_generated_tokens, 192);
        assert_eq!(vl_spec_decode_accepted_tokens, Some(114));
        let extras = &response.client_extras;
        // Rounded to the thousandth, so the float's tail does not matter.
        let reads = |key: &str| {
            extras
                .get(key)
                .and_then(Value::as_f64)
                .map(|v| (v * 1000.0).round() / 1000.0)
        };
        // 190 / 67, the mean of 127 / 47 and 63 / 20, and 192 / (192 - 114).
        [
            ("turns", 2.0),
            ("batch_size", 4.0),
            ("generated_tokens", 192.0),
            ("wall_ms", 500.0),
            ("aggregate_tok_s", 384.0),
            ("decode_forwards", 67.0),
            ("accepted_tokens", 114.0),
            ("accept_length_pooled", 2.836),
            ("accept_length_per_request", 2.926),
            ("accept_length_from_counters", 2.462),
        ]
        .into_iter()
        .for_each(|(key, expected)| {
            assert_eq!(reads(key), Some(expected), "{key}: {extras:?}");
        });
        assert_eq!(extras.len(), 10, "{extras:?}");
        Ok(())
    }

    #[test]
    fn a_reduction_with_speculation_off_or_refused_says_so() -> anyhow::Result<()> {
        let batch = NonZeroUsize::new(1).ok_or_else(|| anyhow::anyhow!("batch size 0"))?;
        let off = traversal(vec![counted(10, 0, 0)])?
            .reduce(Speculation::Off, batch)?
            .build();
        assert_eq!(off.client_extras.len(), 5, "{:?}", off.client_extras);

        let refused = traversal(vec![counted(10, 0, 0)])?
            .reduce(Speculation::On, batch)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(refused.contains("not wired up"), "{refused}");
        Ok(())
    }

    #[test]
    fn one_in_flight_gates_every_sample_and_never_overlaps() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let engine = FakeEngine::new(1);
        let gate_calls = AtomicUsize::new(0);

        let traversal = traverse_with(&corpus, &engine, 1, &gate_calls)?;

        assert_eq!(engine.peak(), 1);
        // Once before the traversal, then before samples 2 and 3.
        assert_eq!(gate_calls.load(Ordering::SeqCst), 3);
        assert_eq!(traversal.turns.len(), 4);
        Ok(())
    }

    #[test]
    fn a_batched_traversal_overlaps_gates_once_and_keeps_file_order() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let engine = FakeEngine::new(3);
        let gate_calls = AtomicUsize::new(0);

        let traversal = traverse_with(&corpus, &engine, 3, &gate_calls)?;

        assert_eq!(engine.peak(), 3, "three slots never ran together");
        assert_eq!(gate_calls.load(Ordering::SeqCst), 1);
        let ids: Vec<(&str, u32)> = traversal
            .observations
            .iter()
            .map(|o| (o.sample_id.as_str(), o.turn))
            .collect();
        assert_eq!(
            ids,
            vec![("img", 0), ("multi", 0), ("multi", 1), ("blank", 0)]
        );
        let seen = engine.seen.lock().unwrap_or_else(PoisonError::into_inner);
        let multi_turns: Vec<&str> = seen
            .iter()
            .filter(|t| *t == "hello" || *t == "and again")
            .map(String::as_str)
            .collect();
        // Three warm-ups plus the sample itself.
        assert_eq!(seen.iter().filter(|t| *t == "what is this?").count(), 4);
        assert_eq!(multi_turns, vec!["hello", "and again"]);
        Ok(())
    }

    #[test]
    fn the_two_procedures_reduce_to_the_same_row() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let gate_calls = AtomicUsize::new(0);

        let serial = traverse_with(&corpus, &FakeEngine::new(1), 1, &gate_calls)?;
        let batched = traverse_with(&corpus, &FakeEngine::new(2), 2, &gate_calls)?;

        assert_eq!(serial.output_digest, batched.output_digest);
        assert_eq!(serial.turns, batched.turns);
        Ok(())
    }

    /// A row over part of the corpus is not a row.
    #[test]
    fn a_failed_sample_fails_a_batched_traversal() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let corpus = corpus(&dir.path().join("c"))?;
        let sends = AtomicUsize::new(0);

        let outcome = traverse(
            &params(Speculation::Off, 2)?,
            &corpus,
            &|| Ok(()),
            no_observer(),
            &|messages, _| {
                sends.fetch_add(1, Ordering::SeqCst);
                let asks_again = messages
                    .iter()
                    .any(|m| m["content"][0]["text"] == "and again");
                if asks_again {
                    return Err(TurnFailure {
                        error: anyhow::anyhow!("the server went away"),
                        retriable: false,
                    });
                }
                Ok(record("fine"))
            },
        );

        let Err(err) = outcome else {
            anyhow::bail!("a failed sample must fail the traversal");
        };
        let message = format!("{err:#}");
        assert!(message.contains("sample multi"), "{message}");
        assert!(message.contains("the server went away"), "{message}");
        Ok(())
    }
}
