//! Eval: every sample of a dataset generated once, each completion appended to
//! the checkpoint as it lands (`docs/methodology/evals.md`).
//! Nothing is timed, so there is no gate, observer or warm-up.

use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Serialize, Serializer};
use serde_json::Value;

use pipette_plan_types::{
    BenchmarkEvalCompletion, BenchmarkEvalCompletionStopReason, BenchmarkId, BenchmarkResultData,
    ChatMessage, ContentPart, EvalBenchmark, EvalSample, MessageContent, VlmEvalBenchmark,
};

use super::AttemptError;
use crate::eval_completions::{
    EvalCompletionSession, MAX_SAMPLE_ATTEMPTS, ensure_complete, ensure_every_answer,
};

/// How often a cell with several samples in flight says it is alive: long
/// enough not to be noise across a multi-hour eval, short enough that the
/// silence between two beats is obviously shorter than a hang.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
/// How often the heartbeat wakes to check whether the run is over, so the end
/// of a run does not wait out a whole interval.
const HEARTBEAT_POLL: Duration = Duration::from_millis(200);

/// What an eval cell runs over: the benchmark's id and its rows, so the text
/// and vision-language cells share one loop.
#[derive(Debug, Clone, Copy)]
pub struct EvalMeasurementDataset<'a> {
    pub benchmark_id: &'a BenchmarkId,
    pub samples: Option<EvalRows<'a>>,
}

impl<'a> From<&'a EvalBenchmark> for EvalMeasurementDataset<'a> {
    fn from(benchmark: &'a EvalBenchmark) -> Self {
        Self {
            benchmark_id: &benchmark.benchmark_id,
            samples: benchmark.samples.as_deref().map(EvalRows::Raw),
        }
    }
}

impl<'a> EvalMeasurementDataset<'a> {
    /// A vlm_eval's rows as its engine resolved them: a catalog entry's come
    /// from its downloaded corpus, not the definition, so they are passed in
    /// rather than read off it.
    pub fn vlm_eval(benchmark: &'a VlmEvalBenchmark, samples: &'a [EvalSample]) -> Self {
        Self {
            benchmark_id: &benchmark.benchmark_id,
            samples: Some(EvalRows::Chat(samples)),
        }
    }
}

/// A dataset's rows: a text eval's as the evals server sent them, a vlm_eval's
/// typed when its definition or corpus was parsed.
#[derive(Debug, Clone, Copy)]
pub enum EvalRows<'a> {
    Raw(&'a [Value]),
    Chat(&'a [EvalSample]),
}

impl EvalRows<'_> {
    fn len(self) -> usize {
        match self {
            Self::Raw(rows) => rows.len(),
            Self::Chat(rows) => rows.len(),
        }
    }
}

/// One row's messages. Serializes as the messages themselves, so an engine
/// puts them straight into its request body.
#[derive(Debug, Clone, Copy)]
pub enum EvalMessages<'a> {
    Raw(&'a Value),
    Chat(&'a [ChatMessage]),
}

impl<'a> EvalMessages<'a> {
    /// A text eval's messages, which its engines read as JSON. Its rows are
    /// always raw, so a typed one means an engine was handed the wrong cell.
    pub fn raw(self) -> Result<&'a Value, EvalMeasurementError> {
        match self {
            Self::Raw(messages) => Ok(messages),
            Self::Chat(_) => Err(EvalMeasurementError::TypedMessages),
        }
    }

    /// The last message's text, or its last text part, for a log line.
    pub fn last_text(self) -> Option<&'a str> {
        match self {
            Self::Raw(messages) => messages.as_array()?.last()?.get("content")?.as_str(),
            Self::Chat(messages) => match &messages.last()?.content {
                MessageContent::Text(text) => Some(text),
                MessageContent::Parts(parts) => parts.iter().rev().find_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => None,
                }),
            },
        }
    }
}

impl Serialize for EvalMessages<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Raw(messages) => messages.serialize(serializer),
            Self::Chat(messages) => messages.serialize(serializer),
        }
    }
}

/// How an eval cell keeps samples in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMeasurementInFlight {
    /// One attempt at a time carrying up to N samples, for an engine that sends
    /// N prompts in one request. A batch lasts as long as its longest sample.
    Batch(NonZeroUsize),
    /// N one-sample attempts at once, the next started as each returns.
    Parallel(NonZeroUsize),
}

/// One dataset row, as an attempt receives it.
#[derive(Debug, Clone)]
pub struct EvalMeasurementSample<'a> {
    /// How the management server joins the completion back to its dataset row.
    pub id: &'a str,
    pub messages: EvalMessages<'a>,
    /// The template arguments the evals server attached to the row (an
    /// asr_eval Voxtral clip's language), for the engine to send unchanged.
    /// `None` on every other row.
    pub chat_template_kwargs: Option<&'a serde_json::Map<String, Value>>,
    /// `eval sample <n>/<total>: id=<id>`, which every log event about it carries.
    pub position: String,
}

/// What an attempt records for one of its samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalMeasurementReading {
    /// The sample generated. `stop_reason` is `Unknown` on an engine that does
    /// not classify its stops, with `stop_detail` saying so.
    Completed {
        completion: String,
        stop_reason: BenchmarkEvalCompletionStopReason,
        stop_detail: Option<String>,
        completion_tokens: Option<u64>,
    },
    /// The sample failed on its own: recorded as failed, and the cell goes on.
    /// Later runs of the cell skip it, except under [`eval_complete()`], which
    /// sends it again while it has attempts left.
    Failed { reason: String },
    /// The sample did not finish, for a reason it shared with the samples in
    /// flight beside it: the server went away, or a batched request failed as
    /// a whole. Sent again alone; lost alone, it is recorded as failed.
    Lost { reason: String },
    /// The sample cannot run in the context this run holds: its bound is past
    /// what one slot holds, or the server ran out of context before its
    /// answer ended. A property of the cell's capacity, not of the sample or
    /// a transient fault, so it is no attempt: recorded as failed in this
    /// run's result, but not written to the checkpoint, so the next run (with
    /// a larger context, as `reason` says) checks it again from scratch, and
    /// re-running unchanged spends nothing.
    Unfit { reason: String },
}

/// Records a reading the moment its sample finishes, by the sample's position
/// in the slice the attempt was sent. An `Err` means the row could not be
/// written; the attempt returns it.
pub type EvalMeasurementRecord<'a> =
    dyn Fn(usize, EvalMeasurementReading) -> anyhow::Result<()> + Sync + 'a;

/// A dataset or a recording an eval cell cannot use.
#[derive(Debug, thiserror::Error)]
pub enum EvalMeasurementError {
    #[error("eval benchmark missing samples")]
    MissingSamples,

    #[error("eval sample {number} missing id")]
    MissingId { number: usize },

    #[error("eval sample {id} missing messages")]
    MissingMessages { id: String },

    #[error("eval sample carries typed chat messages, where a text eval reads raw ones")]
    TypedMessages,

    #[error("engine recorded sample {at} of an attempt carrying {count}")]
    OutOfRange { at: usize, count: usize },

    #[error("engine recorded sample {id} twice")]
    RecordedTwice { id: String },

    #[error("engine returned without recording sample {id}")]
    Unrecorded { id: String },
}

/// Run an eval cell: each sample the checkpoint does not carry, once and in
/// dataset order, its row appended the moment the attempt records it; the
/// result is every row of the checkpoint. An id listed twice runs once.
///
/// A `Lost` reading writes no row and stops new claims; once the attempts in
/// flight return, each lost sample is sent again alone and, lost again,
/// recorded as failed. A sample lost while it already ran alone is failed at
/// once. The engine brings back what it lost when its next attempt starts, so
/// the row of a sample that took the server down alone is on disk before the
/// restart.
///
/// An `Err` from an attempt ends the cell; the rows appended before it stay for
/// the next run to resume from. The caller finalizes the checkpoint once the
/// result is safe ([`crate::eval_completions::EvalCompletionsStore::finalize`]).
///
/// `ctx` is lent to every attempt (shared, as parallel attempts run at once);
/// lend a reference to read it back afterwards, a failed run included.
pub fn eval<'a, C: Sync>(
    benchmark: impl Into<EvalMeasurementDataset<'a>>,
    in_flight: EvalMeasurementInFlight,
    checkpoint: EvalCompletionSession,
    ctx: C,
    attempt: impl Fn(
        &C,
        &[EvalMeasurementSample<'_>],
        &EvalMeasurementRecord<'_>,
    ) -> Result<(), AttemptError>
    + Sync,
) -> anyhow::Result<BenchmarkResultData> {
    let (_, checkpoint) = run_cell(benchmark.into(), in_flight, checkpoint, ctx, attempt, false)?;
    Ok(BenchmarkResultData::Eval {
        completions: checkpoint.into_completions(),
    })
}

/// [`eval()`] for a cell that must not publish failures: a failed row the
/// checkpoint carries is sent again while it has attempts left
/// ([`MAX_SAMPLE_ATTEMPTS`], counted from the file, so across runs), and the
/// result is refused ([`ensure_complete`]) unless every distinct id landed one
/// successful row, or one transport blip would score as a wrong answer.
pub fn eval_complete<'a, C: Sync>(
    benchmark: impl Into<EvalMeasurementDataset<'a>>,
    in_flight: EvalMeasurementInFlight,
    checkpoint: EvalCompletionSession,
    ctx: C,
    attempt: impl Fn(
        &C,
        &[EvalMeasurementSample<'_>],
        &EvalMeasurementRecord<'_>,
    ) -> Result<(), AttemptError>
    + Sync,
) -> anyhow::Result<BenchmarkResultData> {
    let benchmark = benchmark.into();
    let (total, checkpoint) = run_cell(benchmark, in_flight, checkpoint, ctx, attempt, true)?;
    ensure_complete(benchmark.benchmark_id.as_str(), &checkpoint, total)?;
    Ok(BenchmarkResultData::Eval {
        completions: checkpoint.into_completions(),
    })
}

/// [`eval_complete()`] over a dataset too large to hold at once (an asr_eval
/// corpus is gigabytes of audio), read a chunk at a time: `chunk(offset)`
/// returns the rows from `offset`, and each chunk runs to the end, its rows on
/// disk, before the next is asked for. Only one chunk is held at a time.
///
/// `declared` is how many rows the dataset holds, and `id_at(position)` the
/// id of the row at `position` (an asr_eval corpus's index holds every id). So
/// a resume needs no row to know what it has done: reading starts at the first
/// position the checkpoint has not settled ([`chunked_pending`]) and, after
/// each chunk, skips to the next, so a settled stretch is never read again,
/// and a run whose every row is settled reads nothing.
///
/// A chunk may hold any number of rows: the offset advances by the rows it
/// carried, and the dataset ends at `declared`. A chunk that is empty, runs
/// past `declared`, or carries a row whose id is not its position's ends the
/// cell before that chunk runs: the checkpoint is keyed by id, so a stray id
/// would collapse two rows into one or mix datasets, and the shortfall would
/// surface only at the end.
///
/// Unlike [`eval_complete()`], the result is refused by
/// [`ensure_every_answer`]: `PIPETTE_ALLOW_INCOMPLETE_EVAL` is not honored,
/// because an asr_eval is scored only with an answer to every sample.
#[expect(
    clippy::too_many_arguments,
    reason = "eval_complete's arguments, with the dataset as a chunk source"
)]
pub fn eval_complete_chunked<C: Sync>(
    benchmark_id: &BenchmarkId,
    declared: usize,
    id_at: impl Fn(usize) -> String,
    mut chunk: impl FnMut(usize) -> anyhow::Result<Vec<EvalSample>>,
    in_flight: EvalMeasurementInFlight,
    checkpoint: EvalCompletionSession,
    ctx: C,
    attempt: impl Fn(
        &C,
        &[EvalMeasurementSample<'_>],
        &EvalMeasurementRecord<'_>,
    ) -> Result<(), AttemptError>
    + Sync,
) -> anyhow::Result<BenchmarkResultData> {
    let marker = benchmark_id.to_string();
    if !checkpoint.completions().is_empty() {
        tracing::info!(
            cell = marker.as_str(),
            recorded = checkpoint.completions().len(),
            total = declared,
            "resuming eval from its checkpoint"
        );
    }
    let mut checkpoint = checkpoint;
    // What is left is decided once, from the checkpoint as opened: running a
    // chunk settles only that chunk's rows, never a later position's.
    let todo = chunked_pending(&checkpoint, declared, &id_at);
    let todo_ids: HashSet<&str> = todo.iter().map(|(_, id)| id.as_str()).collect();
    let mut cursor = 0;
    while let Some(&(offset, _)) = todo.get(cursor) {
        let rows =
            chunk(offset).with_context(|| format!("{marker}: samples from offset {offset}"))?;
        anyhow::ensure!(
            !rows.is_empty(),
            "{marker}: the samples from offset {offset} are empty, short of the {declared} \
             samples the benchmark declares"
        );
        let end = offset + rows.len();
        anyhow::ensure!(
            end <= declared,
            "{marker}: the samples from offset {offset} run to {end}, past the {declared} \
             samples the benchmark declares"
        );
        if let Some((position, row)) = rows
            .iter()
            .enumerate()
            .map(|(k, row)| (offset + k, row))
            .find(|(position, row)| row.id != id_at(*position))
        {
            anyhow::bail!(
                "{marker}: the sample at position {position} is id `{}`, where the dataset's \
                 is `{}`; the resume checkpoint is keyed by id, so a row out of place would be \
                 skipped, repeated or mixed with another dataset's",
                row.id,
                id_at(position),
            );
        }
        let pending: Vec<EvalMeasurementSample<'_>> = chat_samples(&rows, offset, declared)
            .into_iter()
            .filter(|sample| todo_ids.contains(sample.id))
            .collect();
        if !pending.is_empty() {
            checkpoint = run_samples(
                &marker, &pending, declared, checkpoint, &ctx, &attempt, in_flight,
            )
            .context(marker.clone())?;
        }
        cursor = todo.partition_point(|(position, _)| *position < end);
        if let Some(&(skip_to, _)) = todo.get(cursor).filter(|(skip_to, _)| *skip_to > end) {
            tracing::info!(
                cell = marker.as_str(),
                from = end,
                to = skip_to,
                "skipping samples the checkpoint has settled"
            );
        }
    }
    log_summary(&marker, &checkpoint);
    ensure_every_answer(benchmark_id.as_str(), &checkpoint, declared)?;
    Ok(BenchmarkResultData::Eval {
        completions: checkpoint.into_completions(),
    })
}

/// The positions, ascending, and ids of the rows of a chunked dataset of
/// `declared` rows (the row at each position is `id_at(position)`) that
/// `checkpoint` has not settled: no row, or a failed one with attempts left
/// ([`MAX_SAMPLE_ATTEMPTS`]). Empty when every row is settled, so nothing is
/// left to fetch. One pass over the dataset and one over the checkpoint.
pub fn chunked_pending(
    checkpoint: &EvalCompletionSession,
    declared: usize,
    id_at: impl Fn(usize) -> String,
) -> Vec<(usize, String)> {
    let (retriable, _) = checkpoint.failure_split(MAX_SAMPLE_ATTEMPTS);
    let retriable: HashSet<&str> = retriable.into_iter().collect();
    (0..declared)
        .map(|position| (position, id_at(position)))
        .filter(|(_, id)| !checkpoint.contains(id) || retriable.contains(id.as_str()))
        .collect()
}

/// The cell run behind [`eval()`] and [`eval_complete()`]: the number of
/// distinct ids in the dataset, and the checkpoint once every attempt has
/// returned. `retry_failed` sends failed rows with attempts left again.
fn run_cell<'a, C: Sync>(
    benchmark: EvalMeasurementDataset<'a>,
    in_flight: EvalMeasurementInFlight,
    checkpoint: EvalCompletionSession,
    ctx: C,
    attempt: impl Fn(
        &C,
        &[EvalMeasurementSample<'_>],
        &EvalMeasurementRecord<'_>,
    ) -> Result<(), AttemptError>
    + Sync,
    retry_failed: bool,
) -> anyhow::Result<(usize, EvalCompletionSession)> {
    // Names the cell in every log line and error. TODO: add the job id.
    let marker = benchmark.benchmark_id.to_string();

    let samples = dataset(benchmark).context(marker.clone())?;
    // Ids, not rows: an id listed twice is one row in the checkpoint.
    let total = samples
        .iter()
        .map(|sample| sample.id)
        .collect::<HashSet<_>>()
        .len();
    let pending = pending(&samples, &checkpoint, retry_failed);
    if pending.len() < total {
        tracing::info!(
            cell = marker.as_str(),
            completed = total - pending.len(),
            total,
            "resuming eval from its checkpoint"
        );
    }
    let checkpoint = run_samples(
        &marker, &pending, total, checkpoint, &ctx, &attempt, in_flight,
    )
    .context(marker.clone())?;
    log_summary(&marker, &checkpoint);
    Ok((total, checkpoint))
}

/// Send `pending`, `in_flight` at a time, and hand the checkpoint back once
/// every attempt has returned. `total` is the dataset's, for the progress line.
fn run_samples<C: Sync, A>(
    marker: &str,
    pending: &[EvalMeasurementSample<'_>],
    total: usize,
    checkpoint: EvalCompletionSession,
    ctx: &C,
    attempt: &A,
    in_flight: EvalMeasurementInFlight,
) -> anyhow::Result<EvalCompletionSession>
where
    A: Fn(&C, &[EvalMeasurementSample<'_>], &EvalMeasurementRecord<'_>) -> Result<(), AttemptError>
        + Sync,
{
    let sender = Sender {
        marker,
        total,
        checkpoint: Mutex::new(checkpoint),
        in_flight: AtomicUsize::new(0),
        ctx,
        attempt,
    };
    let (EvalMeasurementInFlight::Batch(count) | EvalMeasurementInFlight::Parallel(count)) =
        in_flight;
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        // With several samples in flight a run can go quiet for as long as its
        // slowest sample takes, which reads the same as a wedged server.
        let heartbeat = (count.get() > 1).then(|| scope.spawn(|| sender.heartbeat(&done)));
        // Raised however the run ends, a panic included: the scope joins the
        // heartbeat before it returns or unwinds.
        let stop = StopOnDrop(&done);
        let run = match in_flight {
            EvalMeasurementInFlight::Batch(size) => {
                if size.get() > 1 {
                    tracing::info!(cell = marker, samples = size.get(), "samples per attempt");
                }
                sender.batches(pending, size)
            }
            EvalMeasurementInFlight::Parallel(workers) => {
                if workers.get() > 1 {
                    tracing::info!(
                        cell = marker,
                        workers = workers.get(),
                        "samples in flight at once; completion order will not match dataset order"
                    );
                }
                sender.pool(pending, workers)
            }
        };
        drop(stop);
        if let Some(heartbeat) = heartbeat {
            let _ = heartbeat.join();
        }
        run
    })?;

    Ok(sender
        .checkpoint
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner))
}

/// What every attempt of one cell shares.
struct Sender<'a, C, A> {
    marker: &'a str,
    /// Distinct ids in the dataset, for the progress line.
    total: usize,
    /// Held only to append a row, never across an attempt.
    checkpoint: Mutex<EvalCompletionSession>,
    /// Samples inside an attempt right now, for the heartbeat.
    in_flight: AtomicUsize,
    ctx: &'a C,
    attempt: &'a A,
}

impl<C, A> Sender<'_, C, A>
where
    C: Sync,
    A: Fn(&C, &[EvalMeasurementSample<'_>], &EvalMeasurementRecord<'_>) -> Result<(), AttemptError>
        + Sync,
{
    /// Say the cell is alive every [`HEARTBEAT_INTERVAL`] until `done`.
    fn heartbeat(&self, done: &AtomicBool) {
        beat_until_stopped(
            done,
            HEARTBEAT_POLL,
            HEARTBEAT_INTERVAL,
            &std::thread::sleep,
            |elapsed| {
                let recorded = lock(&self.checkpoint).completions().len();
                tracing::info!(
                    benchmark_id = %self.marker,
                    recorded,
                    total = self.total,
                    in_flight = self.in_flight.load(Ordering::Relaxed),
                    ?elapsed,
                    "eval running"
                );
            },
        );
    }

    /// `size` samples per attempt, one attempt at a time.
    fn batches(
        &self,
        pending: &[EvalMeasurementSample<'_>],
        size: NonZeroUsize,
    ) -> anyhow::Result<()> {
        pending.chunks(size.get()).try_for_each(|group| {
            let lost = self.send(group, group.len() == 1)?;
            self.settle(&lost)
        })
    }

    /// `workers` one-sample attempts at once. A `Lost` or an error stops the
    /// claiming and the attempts in flight return; after a `Lost`, the lost
    /// samples are settled and the rest start over.
    fn pool(
        &self,
        pending: &[EvalMeasurementSample<'_>],
        workers: NonZeroUsize,
    ) -> anyhow::Result<()> {
        let mut rest = pending;
        while !rest.is_empty() {
            let workers = workers.get().min(rest.len());
            let alone = workers == 1;
            // How many samples are claimed, and whether claiming has stopped:
            // one lock, so no claim slips past a loss or an error.
            let claims = Mutex::new((0, false));
            let lost = Mutex::new(Vec::new());
            let worker = || -> anyhow::Result<()> {
                loop {
                    let sample = {
                        let mut claims = lock(&claims);
                        let (claimed, stopped) = &mut *claims;
                        match rest.get(*claimed) {
                            Some(sample) if !*stopped => {
                                *claimed += 1;
                                sample
                            }
                            _ => return Ok(()),
                        }
                    };
                    let dropped = self
                        .send(std::slice::from_ref(sample), alone)
                        .inspect_err(|_| lock(&claims).1 = true)?;
                    if !dropped.is_empty() {
                        lock(&claims).1 = true;
                        lock(&lost).extend(dropped);
                    }
                }
            };
            if alone {
                // On this thread, so a serial cell spawns nothing.
                worker()?;
            } else {
                let results = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..workers).map(|_| scope.spawn(worker)).collect();
                    handles
                        .into_iter()
                        .map(|handle| {
                            // A panic is a bug, but it must surface as this
                            // cell's error, not lose the siblings' rows.
                            handle.join().unwrap_or_else(|_| {
                                Err(anyhow::anyhow!("eval worker thread panicked"))
                            })
                        })
                        .collect::<Vec<_>>()
                });
                results.into_iter().collect::<anyhow::Result<()>>()?;
            }
            let claimed = lock(&claims).0;
            rest = rest.get(claimed..).unwrap_or_default();
            self.settle(&lock(&lost))?;
        }
        Ok(())
    }

    /// Send each sample lost beside others again, alone.
    fn settle(&self, lost: &[EvalMeasurementSample<'_>]) -> anyhow::Result<()> {
        if !lost.is_empty() {
            tracing::info!(
                cell = self.marker,
                samples = lost.len(),
                "sending lost samples again, one at a time"
            );
        }
        lost.iter()
            .try_for_each(|sample| self.send(std::slice::from_ref(sample), true).map(drop))
    }

    /// One attempt carrying `group`. Each row is appended as the attempt
    /// records it; the samples lost beside others come back. `alone` says
    /// nothing ran beside the attempt, and it carries one sample.
    fn send<'s>(
        &self,
        group: &[EvalMeasurementSample<'s>],
        alone: bool,
    ) -> anyhow::Result<Vec<EvalMeasurementSample<'s>>> {
        let recorded = Mutex::new(vec![false; group.len()]);
        let lost = Mutex::new(Vec::new());
        let record = |at: usize, reading: EvalMeasurementReading| -> anyhow::Result<()> {
            let sample = group.get(at).ok_or(EvalMeasurementError::OutOfRange {
                at,
                count: group.len(),
            })?;
            if lock(&recorded)
                .get_mut(at)
                .is_some_and(|done| std::mem::replace(done, true))
            {
                return Err(EvalMeasurementError::RecordedTwice {
                    id: sample.id.to_owned(),
                }
                .into());
            }
            let row = match reading {
                EvalMeasurementReading::Lost { reason } if !alone => {
                    tracing::warn!(
                        sample = sample.position.as_str(),
                        sample_id = %sample.id,
                        reason = reason.as_str(),
                        "sample lost; to be sent again alone"
                    );
                    lock(&lost).push(sample.clone());
                    return Ok(());
                }
                EvalMeasurementReading::Completed {
                    completion,
                    stop_reason,
                    stop_detail,
                    completion_tokens,
                } => BenchmarkEvalCompletion {
                    id: sample.id.to_owned(),
                    completion,
                    failed: false,
                    failed_reason: None,
                    stop_reason,
                    stop_detail,
                    completion_tokens,
                },
                EvalMeasurementReading::Failed { reason } => failed_row(sample.id, reason),
                EvalMeasurementReading::Unfit { reason } => {
                    tracing::warn!(
                        sample = sample.position.as_str(),
                        sample_id = %sample.id,
                        reason = reason.as_str(),
                        "sample does not fit the context; recorded failed for this run, \
                         not checkpointed"
                    );
                    let mut checkpoint = lock(&self.checkpoint);
                    checkpoint.record_unfit(failed_row(sample.id, reason));
                    log_progress(self.marker, &checkpoint, self.total);
                    return Ok(());
                }
                EvalMeasurementReading::Lost { reason } => {
                    tracing::warn!(
                        sample = sample.position.as_str(),
                        sample_id = %sample.id,
                        reason = reason.as_str(),
                        "sample lost alone; recorded as failed"
                    );
                    failed_row(sample.id, reason)
                }
            };
            let mut checkpoint = lock(&self.checkpoint);
            checkpoint.append(row)?;
            log_progress(self.marker, &checkpoint, self.total);
            Ok(())
        };

        let label = label(group);
        self.in_flight.fetch_add(group.len(), Ordering::Relaxed);
        let attempted = (self.attempt)(self.ctx, group, &record);
        self.in_flight.fetch_sub(group.len(), Ordering::Relaxed);
        attempted
            .map_err(AttemptError::into_anyhow)
            .context(label.clone())?;
        let recorded = recorded
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some((sample, _)) = group.iter().zip(recorded).find(|(_, done)| !done) {
            return Err(EvalMeasurementError::Unrecorded {
                id: sample.id.to_owned(),
            })
            .context(label);
        }
        Ok(lost.into_inner().unwrap_or_else(PoisonError::into_inner))
    }
}

/// The benchmark's rows, each with its id and messages checked before anything
/// runs.
fn dataset<'a>(
    benchmark: EvalMeasurementDataset<'a>,
) -> Result<Vec<EvalMeasurementSample<'a>>, EvalMeasurementError> {
    let rows = benchmark
        .samples
        .ok_or(EvalMeasurementError::MissingSamples)?;
    let total = rows.len();
    let sample = |number: usize, id: &'a str, messages| EvalMeasurementSample {
        id,
        messages,
        chat_template_kwargs: None,
        position: format!("eval sample {number}/{total}: id={id}"),
    };
    match rows {
        EvalRows::Raw(rows) => rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let number = index + 1;
                let id = row
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or(EvalMeasurementError::MissingId { number })?;
                let messages = row
                    .get("messages")
                    .ok_or_else(|| EvalMeasurementError::MissingMessages { id: id.to_owned() })?;
                Ok(sample(number, id, EvalMessages::Raw(messages)))
            })
            .collect(),
        // Typed rows were checked when they were parsed.
        EvalRows::Chat(rows) => Ok(chat_samples(rows, 0, total)),
    }
}

/// Typed rows as samples, numbered from `offset` within a dataset of `total`.
/// Typed rows were checked when they were parsed.
fn chat_samples(
    rows: &[EvalSample],
    offset: usize,
    total: usize,
) -> Vec<EvalMeasurementSample<'_>> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let number = offset + index + 1;
            let id = row.id.as_str();
            EvalMeasurementSample {
                id,
                messages: EvalMessages::Chat(&row.messages),
                chat_template_kwargs: row.chat_template_kwargs.as_ref(),
                position: format!("eval sample {number}/{total}: id={id}"),
            }
        })
        .collect()
}

/// The samples to send: those the checkpoint does not carry, each id once, in
/// dataset order. `retry_failed` also sends a failed row that has attempts
/// left. Those ids are collected once, so a resume stays linear in the rows.
fn pending<'a>(
    samples: &[EvalMeasurementSample<'a>],
    checkpoint: &EvalCompletionSession,
    retry_failed: bool,
) -> Vec<EvalMeasurementSample<'a>> {
    let retry: HashSet<&str> = if retry_failed {
        checkpoint
            .failure_split(MAX_SAMPLE_ATTEMPTS)
            .0
            .into_iter()
            .collect()
    } else {
        HashSet::new()
    };
    let mut seen = HashSet::new();
    samples
        .iter()
        .filter(|sample| {
            let settled = checkpoint.contains(sample.id) && !retry.contains(sample.id);
            !settled && seen.insert(sample.id)
        })
        .cloned()
        .collect()
}

/// Names an attempt in its error: its one sample, or the first of its batch.
fn label(group: &[EvalMeasurementSample<'_>]) -> String {
    match group {
        [only] => only.position.clone(),
        [first, ..] => format!("batch of {} from {}", group.len(), first.position),
        [] => String::new(),
    }
}

/// The row of a sample that produced no completion. `failed` and
/// `failed_reason` are dual-written for legacy consumers until PIP-323
/// retires them.
fn failed_row(id: &str, reason: String) -> BenchmarkEvalCompletion {
    BenchmarkEvalCompletion {
        id: id.to_owned(),
        completion: String::new(),
        failed: true,
        failed_reason: Some(reason.clone()),
        stop_reason: BenchmarkEvalCompletionStopReason::Failure,
        stop_detail: Some(reason),
        completion_tokens: None,
    }
}

/// Per-sample progress, after an append. `processed` counts completed and
/// failed rows so the end-of-cell tick still fires when samples failed; the
/// displayed `completed` stays success-only. Logged every `total.clamp(1,
/// 20)` samples so the log neither spams a tiny set nor falls silent on a
/// huge one.
fn log_progress(benchmark_id: &str, checkpoint: &EvalCompletionSession, total: usize) {
    let processed = checkpoint.completions().len();
    if processed == 1 || processed == total || processed.is_multiple_of(total.clamp(1, 20)) {
        let completed = processed - checkpoint.failed_count();
        tracing::info!(benchmark_id = %benchmark_id, completed, total, "eval progress");
    }
}

/// The final summary an operator looks for: how many samples landed in the
/// result against how many were recorded as failed, with the failed ids
/// inlined, and how many hit the token ceiling.
fn log_summary(benchmark_id: &str, checkpoint: &EvalCompletionSession) {
    let failed_count = checkpoint.failed_count();
    let completed = checkpoint.completions().len() - failed_count;
    if failed_count == 0 {
        tracing::info!(benchmark_id = %benchmark_id, completed, failed = 0, "eval finished");
    } else {
        let ids: Vec<&str> = checkpoint.failed_ids().collect();
        tracing::warn!(
            benchmark_id = %benchmark_id,
            completed,
            failed = failed_count,
            failed_ids = ?ids,
            "eval finished"
        );
    }
    // A truncated completion is scored like any other, so `failed=0` can
    // hide a run where the token budget decided most answers.
    let truncated = checkpoint
        .completions()
        .iter()
        .filter(|c| c.stop_reason == BenchmarkEvalCompletionStopReason::Truncated)
        .count();
    if truncated == 0 {
        return;
    }
    let scored = completed;
    let percent = (truncated * 100).checked_div(scored).unwrap_or(0);
    tracing::warn!(
        benchmark_id = %benchmark_id,
        truncated,
        scored,
        percent,
        "eval completions hit max_tokens rather than stopping on their own; those scores \
         reflect the token budget as much as the model"
    );
}

/// Call `beat` with the time since start once per elapsed `interval`, waking
/// every `poll` to check `stop`. Over injected durations and sleep, so the stop
/// is testable without a run.
fn beat_until_stopped(
    stop: &AtomicBool,
    poll: Duration,
    interval: Duration,
    sleep: &dyn Fn(Duration),
    mut beat: impl FnMut(Duration),
) {
    let started = Instant::now();
    let mut last = started;
    while !stop.load(Ordering::Relaxed) {
        sleep(poll);
        if last.elapsed() < interval {
            continue;
        }
        last = Instant::now();
        beat(started.elapsed());
    }
}

struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// A poisoned lock still holds every row appended before the panic.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Condvar;

    use rstest::rstest;
    use serde_json::json;

    use pipette_plan_types::{
        BenchmarkDefinition, BenchmarkId, DeclaredBound, EvalId, Model, PositiveCount, RunRequest,
        Runtime,
    };

    use super::*;
    use crate::eval_completions::EvalCompletionsStore;

    /// An eval cell over `ids`, and the store its checkpoint opens in.
    fn cell(ids: &[&str]) -> anyhow::Result<(tempfile::TempDir, EvalCompletionsStore, RunRequest)> {
        let dir = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(dir.path());
        let samples = ids
            .iter()
            .map(|id| json!({"id": id, "messages": [{"role": "user", "content": id}]}))
            .collect();
        let req = with_samples(Some(samples))?;
        Ok((dir, store, req))
    }

    fn with_samples(samples: Option<Vec<Value>>) -> anyhow::Result<RunRequest> {
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: BenchmarkId::try_new("eval_entry".to_owned())?,
                parameter_eval_id: EvalId::from("math_500"),
                parameter_dataset_name: "d".to_owned(),
                parameter_max_tokens: PositiveCount::try_new(8)?,
                parameter_max_context: None,
                parameter_mcq_choices: None,
                parameter_temperature: None,
                parameter_top_p: None,
                parameter_top_k: None,
                parameter_min_p: None,
                parameter_seed: None,
                samples,
            }),
        })
    }

    fn batch(size: usize) -> EvalMeasurementInFlight {
        EvalMeasurementInFlight::Batch(NonZeroUsize::new(size).unwrap_or(NonZeroUsize::MIN))
    }

    fn parallel(workers: usize) -> EvalMeasurementInFlight {
        EvalMeasurementInFlight::Parallel(NonZeroUsize::new(workers).unwrap_or(NonZeroUsize::MIN))
    }

    fn completed(id: &str) -> EvalMeasurementReading {
        EvalMeasurementReading::Completed {
            completion: format!("answer to {id}"),
            stop_reason: BenchmarkEvalCompletionStopReason::Eos,
            stop_detail: None,
            completion_tokens: Some(3),
        }
    }

    /// Records a completion for every sample it is sent.
    fn answer_all(
        samples: &[EvalMeasurementSample<'_>],
        record: &EvalMeasurementRecord<'_>,
    ) -> Result<(), AttemptError> {
        samples
            .iter()
            .enumerate()
            .try_for_each(|(at, sample)| record(at, completed(sample.id)))?;
        Ok(())
    }

    fn rows(data: &BenchmarkResultData) -> anyhow::Result<&[BenchmarkEvalCompletion]> {
        let BenchmarkResultData::Eval { completions } = data else {
            anyhow::bail!("expected eval result data, got {data:?}");
        };
        Ok(completions)
    }

    fn sorted_ids(rows: &[BenchmarkEvalCompletion]) -> Vec<&str> {
        let mut ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        ids.sort_unstable();
        ids
    }

    /// Arrivals counted up to `n`. `wait` blocks until all have arrived, and
    /// fails after a bound, so a pool that does not run them together fails a
    /// test instead of hanging it.
    struct Latch {
        arrived: Mutex<usize>,
        all: Condvar,
        n: usize,
    }

    impl Latch {
        fn new(n: usize) -> Self {
            Self {
                arrived: Mutex::new(0),
                all: Condvar::new(),
                n,
            }
        }

        fn arrive(&self) {
            *lock(&self.arrived) += 1;
            self.all.notify_all();
        }

        fn wait(&self) -> anyhow::Result<()> {
            self.arrive();
            let (arrived, waited) = self
                .all
                .wait_timeout_while(lock(&self.arrived), Duration::from_secs(10), |arrived| {
                    *arrived < self.n
                })
                .unwrap_or_else(PoisonError::into_inner);
            anyhow::ensure!(
                !waited.timed_out(),
                "only {} of {} arrived",
                *arrived,
                self.n
            );
            Ok(())
        }
    }

    /// Whatever the mode, a run sends each sample the checkpoint lacks once and
    /// reports every row, the resumed ones included.
    #[rstest]
    #[case::batches_of_one(batch(1))]
    #[case::batches_of_three(batch(3))]
    #[case::one_at_a_time(parallel(1))]
    #[case::three_at_once(parallel(3))]
    fn every_sample_runs_once_and_resume_skips_the_done(
        #[case] in_flight: EvalMeasurementInFlight,
    ) -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2", "s3", "s4", "s5", "s6"])?;
        {
            let mut checkpoint = store.open(&req)?;
            for id in ["s1", "s4"] {
                checkpoint.append(BenchmarkEvalCompletion {
                    id: id.to_owned(),
                    completion: "earlier".to_owned(),
                    ..Default::default()
                })?;
            }
        }

        let sent = Mutex::new(Vec::new());
        let data = eval(
            req.benchmark.as_eval()?,
            in_flight,
            store.open(&req)?,
            (),
            |_, samples, record| {
                lock(&sent).extend(samples.iter().map(|sample| sample.id.to_owned()));
                answer_all(samples, record)
            },
        )?;

        let mut sent = lock(&sent).clone();
        sent.sort_unstable();
        assert_eq!(sent, ["s0", "s2", "s3", "s5", "s6"]);
        let rows = rows(&data)?;
        assert_eq!(sorted_ids(rows), ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]);
        Ok(())
    }

    /// Typed rows `ids`, as a corpus chunk carries them.
    fn typed(ids: &[&str]) -> Vec<EvalSample> {
        ids.iter()
            .map(|id| EvalSample {
                id: (*id).to_owned(),
                messages: Vec::new(),
                chat_template_kwargs: (*id == "s3").then(|| {
                    json!({"language": "nl"})
                        .as_object()
                        .cloned()
                        .unwrap_or_default()
                }),
            })
            .collect()
    }

    /// Chunks over `ids` of the sizes `sizes` (the last repeated), short
    /// anywhere; every offset asked for is recorded.
    fn chunker<'a>(
        ids: &'a [&'a str],
        sizes: &'a [usize],
        asked: &'a Mutex<Vec<usize>>,
    ) -> impl FnMut(usize) -> anyhow::Result<Vec<EvalSample>> + 'a {
        move |offset| {
            let size = {
                let mut asked = lock(asked);
                let size = sizes
                    .get(asked.len())
                    .or_else(|| sizes.last())
                    .copied()
                    .unwrap_or(1);
                asked.push(offset);
                size
            };
            Ok(typed(
                ids.get(offset..(offset + size).min(ids.len()))
                    .unwrap_or_default(),
            ))
        }
    }

    /// The id a test dataset gives a position.
    fn positional(position: usize) -> String {
        format!("s{position}")
    }

    /// Chunked, every row runs once in dataset order and numbered across
    /// chunks, one chunk at a time, whatever size each comes in; a resume
    /// reads only the chunk that holds what the checkpoint lacks, and sends
    /// only that. A row's template arguments reach the attempt.
    #[test]
    fn a_chunked_dataset_runs_chunk_by_chunk_and_resumes() -> anyhow::Result<()> {
        const IDS: &[&str] = &["s0", "s1", "s2", "s3", "s4"];
        let (_dir, store, req) = cell(IDS)?;
        let id = BenchmarkId::try_new("asr_entry".to_owned())?;
        let asked = Mutex::new(Vec::new());
        let sent = Mutex::new(Vec::new());
        let attempt =
            |_: &(), samples: &[EvalMeasurementSample<'_>], record: &EvalMeasurementRecord<'_>| {
                lock(&sent).extend(samples.iter().map(|sample| {
                    (
                        sample.position.clone(),
                        sample
                            .chat_template_kwargs
                            .and_then(|k| k.get("language"))
                            .cloned(),
                    )
                }));
                // s2 fails once: the first run must not publish.
                samples.iter().enumerate().try_for_each(|(at, sample)| {
                    if sample.id == "s2" && lock(&sent).len() <= IDS.len() {
                        return record(
                            at,
                            EvalMeasurementReading::Failed {
                                reason: "x".to_owned(),
                            },
                        );
                    }
                    record(at, completed(sample.id))
                })?;
                Ok(())
            };
        // Short chunks anywhere: 1, then 3, then what is left.
        let first = eval_complete_chunked(
            &id,
            IDS.len(),
            positional,
            chunker(IDS, &[1, 3, 2], &asked),
            parallel(1),
            store.open(&req)?,
            (),
            attempt,
        );
        assert!(first.is_err(), "a failed row must hold the result back");
        assert_eq!(*lock(&asked), [0, 1, 4]);
        assert_eq!(
            lock(&sent)
                .iter()
                .map(|(p, _)| p.as_str())
                .collect::<Vec<_>>(),
            [
                "eval sample 1/5: id=s0",
                "eval sample 2/5: id=s1",
                "eval sample 3/5: id=s2",
                "eval sample 4/5: id=s3",
                "eval sample 5/5: id=s4",
            ]
        );
        assert_eq!(
            lock(&sent).get(3).and_then(|(_, l)| l.clone()),
            Some(json!("nl"))
        );

        lock(&asked).clear();
        let data = eval_complete_chunked(
            &id,
            IDS.len(),
            positional,
            chunker(IDS, &[2], &asked),
            parallel(2),
            store.open(&req)?,
            (),
            attempt,
        )?;
        assert_eq!(*lock(&asked), [2], "only the chunk holding s2 is read");
        assert_eq!(
            lock(&sent).get(IDS.len()..).map(|tail| tail.len()),
            Some(1),
            "only the failed row is sent again"
        );
        assert_eq!(sorted_ids(rows(&data)?), IDS);

        // Settled: nothing is read or sent, and the result stands.
        lock(&asked).clear();
        let again = eval_complete_chunked(
            &id,
            IDS.len(),
            positional,
            chunker(IDS, &[2], &asked),
            parallel(1),
            store.open(&req)?,
            (),
            attempt,
        )?;
        assert!(lock(&asked).is_empty(), "{:?}", lock(&asked));
        assert_eq!(sorted_ids(rows(&again)?), IDS);
        Ok(())
    }

    /// A resume skips every settled stretch: chunks are asked for only where
    /// a row is left, the offset moving past what is done.
    #[test]
    fn a_resume_reads_only_the_chunks_it_needs() -> anyhow::Result<()> {
        const IDS: &[&str] = &["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];
        let (_dir, store, req) = cell(IDS)?;
        let id = BenchmarkId::try_new("asr_entry".to_owned())?;
        let mut session = store.open(&req)?;
        for done in ["s0", "s1", "s2", "s4", "s5", "s6"] {
            session.append(BenchmarkEvalCompletion {
                id: done.to_owned(),
                completion: "a".to_owned(),
                failed: false,
                failed_reason: None,
                stop_reason: BenchmarkEvalCompletionStopReason::Eos,
                stop_detail: None,
                completion_tokens: Some(1),
            })?;
        }
        drop(session);
        let asked = Mutex::new(Vec::new());
        let sent = Mutex::new(Vec::new());
        eval_complete_chunked(
            &id,
            IDS.len(),
            positional,
            chunker(IDS, &[2], &asked),
            parallel(1),
            store.open(&req)?,
            (),
            |_, samples, record| {
                lock(&sent).extend(samples.iter().map(|sample| sample.id.to_owned()));
                answer_all(samples, record)
            },
        )?;
        assert_eq!(*lock(&asked), [3, 7]);
        assert_eq!(*lock(&sent), ["s3", "s7"]);
        Ok(())
    }

    /// A chunk that is empty, runs past the declared count, or carries a row
    /// out of its position ends the cell before it runs.
    #[rstest]
    #[case::short(&["s0", "s1"], 3, 2, "are empty")]
    #[case::long(&["s0", "s1", "s2"], 2, 3, "past the 2")]
    #[case::repeated(&["s0", "s1", "s0"], 3, 2, "position 2 is id `s0`, where the dataset's is `s2`")]
    #[case::another_dataset(&["s0", "x1"], 2, 2, "position 1 is id `x1`")]
    fn a_chunk_that_breaks_the_dataset_ends_the_cell(
        #[case] ids: &[&str],
        #[case] declared: usize,
        #[case] size: usize,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0"])?;
        let asked = Mutex::new(Vec::new());
        let sizes = [size];
        let result = eval_complete_chunked(
            &BenchmarkId::try_new("asr_entry".to_owned())?,
            declared,
            positional,
            chunker(ids, &sizes, &asked),
            parallel(1),
            store.open(&req)?,
            (),
            |_, samples, record| answer_all(samples, record),
        );
        let Err(err) = result else {
            anyhow::bail!("expected a refusal naming {needle}");
        };
        assert!(format!("{err:#}").contains(needle), "{err:#}");
        Ok(())
    }

    #[test]
    fn a_batch_carries_up_to_its_size_in_dataset_order() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2", "s3", "s4", "s5", "s6"])?;
        let groups = Mutex::new(Vec::new());
        eval(
            req.benchmark.as_eval()?,
            batch(3),
            store.open(&req)?,
            (),
            |_, samples, record| {
                lock(&groups).push(
                    samples
                        .iter()
                        .map(|sample| sample.id.to_owned())
                        .collect::<Vec<_>>(),
                );
                answer_all(samples, record)
            },
        )?;
        assert_eq!(
            *lock(&groups),
            [vec!["s0", "s1", "s2"], vec!["s3", "s4", "s5"], vec!["s6"]]
        );
        Ok(())
    }

    /// A row is on disk the moment its sample is recorded, before the attempt
    /// carrying it returns, so a batch that dies halfway keeps the samples it
    /// finished.
    #[test]
    fn a_batch_keeps_what_it_recorded_before_it_failed() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2", "s3"])?;
        let err = eval(
            req.benchmark.as_eval()?,
            batch(4),
            store.open(&req)?,
            (),
            |_, _, record| {
                record(2, completed("s2"))?;
                if !store.open(&req)?.contains("s2") {
                    return Err(anyhow::anyhow!("s2 is not on disk yet").into());
                }
                record(0, completed("s0"))?;
                Err(anyhow::Error::from(std::io::Error::other("server went away")).into())
            },
        )
        .err()
        .context("a failed attempt became a result")?;
        assert!(
            err.chain().any(|cause| cause.is::<std::io::Error>()),
            "{err:#}"
        );

        let resumed = store.open(&req)?;
        assert!(resumed.contains("s0") && resumed.contains("s2"));
        assert!(!resumed.contains("s1") && !resumed.contains("s3"));
        Ok(())
    }

    /// The row says why, in the shape the scorer and later runs read.
    #[test]
    fn a_failed_sample_is_a_row_and_the_cell_goes_on() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2"])?;
        let data = eval(
            req.benchmark.as_eval()?,
            parallel(1),
            store.open(&req)?,
            (),
            |_, samples, record| {
                samples.iter().enumerate().try_for_each(|(at, sample)| {
                    record(
                        at,
                        if sample.id == "s1" {
                            EvalMeasurementReading::Failed {
                                reason: "context overflow".to_owned(),
                            }
                        } else {
                            completed(sample.id)
                        },
                    )
                })?;
                Ok(())
            },
        )?;

        let rows = rows(&data)?;
        assert_eq!(sorted_ids(rows), ["s0", "s1", "s2"]);
        let failed = rows
            .iter()
            .find(|row| row.id == "s1")
            .context("s1 has a row")?;
        assert_eq!(
            failed,
            &BenchmarkEvalCompletion {
                id: "s1".to_owned(),
                completion: String::new(),
                failed: true,
                failed_reason: Some("context overflow".to_owned()),
                stop_reason: BenchmarkEvalCompletionStopReason::Failure,
                stop_detail: Some("context overflow".to_owned()),
                completion_tokens: None,
            }
        );
        let ok = rows
            .iter()
            .find(|row| row.id == "s0")
            .context("s0 has a row")?;
        assert_eq!(ok.completion, "answer to s0");
        assert_eq!(ok.completion_tokens, Some(3));
        Ok(())
    }

    /// A crash takes down every sample in flight, so each is sent again alone:
    /// the one that crashes the server again is recorded as failed, the others
    /// as what they answer. It is sent twice, not three times.
    #[rstest]
    #[case::in_a_batch(batch(3), false)]
    #[case::in_a_pool(parallel(3), true)]
    fn samples_lost_beside_others_run_again_alone(
        #[case] in_flight: EvalMeasurementInFlight,
        #[case] pooled: bool,
    ) -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2", "s3", "s4", "s5"])?;
        let attempts = AtomicUsize::new(0);
        let running = AtomicUsize::new(0);
        // What each sample was sent with: its attempt's size, and how many
        // attempts ran when it started.
        let sends = Mutex::new(Vec::new());
        let down = AtomicBool::new(false);
        let (started, crashed) = (Latch::new(3), Latch::new(3));

        let data = eval(
            req.benchmark.as_eval()?,
            in_flight,
            store.open(&req)?,
            (),
            |_, samples, record| {
                let first_round = pooled && attempts.fetch_add(1, Ordering::SeqCst) < 3;
                let beside = running.fetch_add(1, Ordering::SeqCst) + 1;
                lock(&sends).extend(
                    samples
                        .iter()
                        .map(|sample| (sample.id.to_owned(), samples.len(), beside)),
                );
                // The engine brings back what an earlier attempt lost.
                down.store(false, Ordering::SeqCst);
                if first_round {
                    started.wait()?;
                }
                if samples.iter().any(|sample| sample.id == "s1") {
                    down.store(true, Ordering::SeqCst);
                }
                if first_round {
                    crashed.wait()?;
                }
                let lost = down.load(Ordering::SeqCst);
                running.fetch_sub(1, Ordering::SeqCst);
                samples.iter().enumerate().try_for_each(|(at, sample)| {
                    record(
                        at,
                        if lost {
                            EvalMeasurementReading::Lost {
                                reason: "server crashed".to_owned(),
                            }
                        } else {
                            completed(sample.id)
                        },
                    )
                })?;
                Ok(())
            },
        )?;

        let rows = rows(&data)?;
        assert_eq!(sorted_ids(rows), ["s0", "s1", "s2", "s3", "s4", "s5"]);
        rows.iter().for_each(|row| {
            assert_eq!(row.failed, row.id == "s1", "{row:?}");
        });
        let crasher = rows
            .iter()
            .find(|row| row.id == "s1")
            .context("s1 has a row")?;
        assert_eq!(crasher.failed_reason.as_deref(), Some("server crashed"));

        let sends = lock(&sends);
        let times = |id: &str| sends.iter().filter(|(sent, _, _)| sent == id).count();
        assert_eq!(times("s1"), 2, "{sends:?}");
        assert_eq!((times("s0"), times("s2")), (2, 2), "{sends:?}");
        assert_eq!(
            (times("s3"), times("s4"), times("s5")),
            (1, 1, 1),
            "{sends:?}"
        );
        // The resends are the three sends after the first three.
        let resends = sends.get(3..6).context("six sends at least")?;
        resends.iter().for_each(|(id, size, beside)| {
            assert!(
                ["s0", "s1", "s2"].contains(&id.as_str()) && *size == 1 && *beside == 1,
                "{id} was not sent again alone: {sends:?}"
            );
        });
        Ok(())
    }

    /// A sample lost while running alone is what the server went away on, so
    /// it is recorded as failed rather than sent again, and the cell goes on.
    #[rstest]
    #[case::batch_of_one(batch(1))]
    #[case::one_at_a_time(parallel(1))]
    fn a_sample_lost_alone_is_failed_at_once(
        #[case] in_flight: EvalMeasurementInFlight,
    ) -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2"])?;
        let sent = Mutex::new(Vec::new());
        let data = eval(
            req.benchmark.as_eval()?,
            in_flight,
            store.open(&req)?,
            (),
            |_, samples, record| {
                lock(&sent).extend(samples.iter().map(|sample| sample.id.to_owned()));
                samples.iter().enumerate().try_for_each(|(at, sample)| {
                    record(
                        at,
                        if sample.id == "s1" {
                            EvalMeasurementReading::Lost {
                                reason: "server crashed".to_owned(),
                            }
                        } else {
                            completed(sample.id)
                        },
                    )
                })?;
                Ok(())
            },
        )?;
        assert_eq!(*lock(&sent), ["s0", "s1", "s2"]);
        let rows = rows(&data)?;
        let failed: Vec<&str> = rows
            .iter()
            .filter(|row| row.failed)
            .map(|row| row.id.as_str())
            .collect();
        assert_eq!(failed, ["s1"]);
        Ok(())
    }

    /// A pool refills a slot the moment its sample lands: the others run while
    /// one sample is still generating, instead of waiting for it as a batch
    /// would.
    #[test]
    fn a_pool_refills_while_a_long_sample_runs() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2", "s3", "s4", "s5"])?;
        let everyone = Latch::new(6);
        eval(
            req.benchmark.as_eval()?,
            parallel(2),
            store.open(&req)?,
            (),
            |_, samples, record| {
                if samples.iter().any(|sample| sample.id == "s0") {
                    everyone.wait()?;
                } else {
                    everyone.arrive();
                }
                answer_all(samples, record)
            },
        )?;
        Ok(())
    }

    /// An error ends the cell: the error names the cell and the sample and
    /// keeps its type for the worker's chain walk, the rows recorded before it
    /// stay for the next run, and what the attempts kept in `ctx` is the
    /// caller's to read.
    #[test]
    fn an_attempt_error_ends_the_cell_and_keeps_what_came_before() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0", "s1", "s2"])?;
        let seen = Mutex::new(Vec::new());
        let err = eval(
            req.benchmark.as_eval()?,
            parallel(1),
            store.open(&req)?,
            &seen,
            |seen, samples, record| {
                lock(seen).extend(samples.iter().map(|sample| sample.id.to_owned()));
                if samples.iter().any(|sample| sample.id == "s1") {
                    return Err(anyhow::Error::from(std::io::Error::other("attempt failed")).into());
                }
                answer_all(samples, record)
            },
        )
        .err()
        .context("a failed attempt became a result")?;

        assert_eq!(
            format!("{err:#}"),
            "eval_entry: eval sample 2/3: id=s1: attempt failed"
        );
        assert!(
            err.chain().any(|cause| cause.is::<std::io::Error>()),
            "{err:#}"
        );
        assert_eq!(*lock(&seen), ["s0", "s1"]);
        let resumed = store.open(&req)?;
        assert!(resumed.contains("s0") && !resumed.contains("s1") && !resumed.contains("s2"));
        Ok(())
    }

    #[rstest]
    #[case::twice(&[0, 0], "twice")]
    #[case::none(&[], "unrecorded")]
    #[case::out_of_range(&[5], "out of range")]
    fn a_recording_the_entry_cannot_use_fails_the_cell(
        #[case] positions: &[usize],
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["s0"])?;
        let err = eval(
            req.benchmark.as_eval()?,
            parallel(1),
            store.open(&req)?,
            (),
            |_, _, record| {
                positions
                    .iter()
                    .try_for_each(|&at| record(at, completed("s0")))?;
                Ok(())
            },
        )
        .err()
        .context("a bad recording became a result")?;
        let matched = match err.downcast_ref::<EvalMeasurementError>() {
            Some(EvalMeasurementError::RecordedTwice { id }) => id == "s0" && expected == "twice",
            Some(EvalMeasurementError::Unrecorded { id }) => id == "s0" && expected == "unrecorded",
            Some(EvalMeasurementError::OutOfRange { at: 5, count: 1 }) => {
                expected == "out of range"
            }
            _ => false,
        };
        assert!(matched, "expected {expected}: {err:#}");
        Ok(())
    }

    #[rstest]
    #[case::no_samples(None)]
    #[case::no_id(Some(vec![json!({"messages": []})]))]
    #[case::no_messages(Some(vec![json!({"id": "s0"})]))]
    fn an_unreadable_dataset_fails_before_anything_runs(
        #[case] samples: Option<Vec<Value>>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let store = EvalCompletionsStore::new(dir.path());
        let req = with_samples(samples)?;
        let sent = AtomicBool::new(false);
        let err = eval(
            req.benchmark.as_eval()?,
            parallel(1),
            store.open(&req)?,
            (),
            |_, samples, record| {
                sent.store(true, Ordering::SeqCst);
                answer_all(samples, record)
            },
        )
        .err()
        .context("an unreadable dataset became a result")?;
        assert!(
            err.downcast_ref::<EvalMeasurementError>().is_some(),
            "{err:#}"
        );
        assert!(!sent.load(Ordering::SeqCst));
        Ok(())
    }

    #[test]
    fn an_id_listed_twice_runs_once() -> anyhow::Result<()> {
        let (_dir, store, req) = cell(&["a", "b", "a"])?;
        let sent = Mutex::new(Vec::new());
        let data = eval(
            req.benchmark.as_eval()?,
            parallel(2),
            store.open(&req)?,
            (),
            |_, samples, record| {
                lock(&sent).extend(samples.iter().map(|sample| sample.id.to_owned()));
                answer_all(samples, record)
            },
        )?;
        let mut sent = lock(&sent).clone();
        sent.sort_unstable();
        assert_eq!(sent, ["a", "b"]);
        assert_eq!(sorted_ids(rows(&data)?), ["a", "b"]);
        Ok(())
    }

    /// Beats only when an interval has passed. The injected sleep raises
    /// `stop` after `polls` wake-ups.
    #[rstest]
    #[case::stopped_before_the_first_poll(0, Duration::ZERO, 0)]
    #[case::beats_on_every_poll_past_the_interval(3, Duration::ZERO, 3)]
    #[case::silent_inside_the_interval(3, HEARTBEAT_INTERVAL, 0)]
    fn heartbeat_stops_when_told(
        #[case] polls: usize,
        #[case] interval: Duration,
        #[case] expected_beats: usize,
    ) -> anyhow::Result<()> {
        let stop = AtomicBool::new(polls == 0);
        let slept = std::cell::Cell::new(0);
        let sleep = |_| {
            slept.set(slept.get() + 1);
            if slept.get() == polls {
                stop.store(true, Ordering::Relaxed);
            }
        };
        let mut beats = 0;
        beat_until_stopped(&stop, HEARTBEAT_POLL, interval, &sleep, |_| beats += 1);
        anyhow::ensure!(
            (slept.get(), beats) == (polls, expected_beats),
            "slept {} times and beat {beats} times",
            slept.get(),
        );
        Ok(())
    }

    /// A panic in an attempt on the entry's own thread (a batch, the pool's
    /// last sample, a resend) reaches the caller instead of leaving the scope
    /// waiting on the heartbeat forever.
    #[rstest]
    #[case::a_batch(batch(2), &["s0", "s1"])]
    #[case::the_pools_last_sample(parallel(2), &["s0"])]
    fn a_panicking_attempt_reaches_the_caller(
        #[case] in_flight: EvalMeasurementInFlight,
        #[case] ids: &'static [&'static str],
    ) -> anyhow::Result<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let panicked = std::panic::catch_unwind(|| -> anyhow::Result<()> {
                let (_dir, store, req) = cell(ids)?;
                eval(
                    req.benchmark.as_eval()?,
                    in_flight,
                    store.open(&req)?,
                    (),
                    |_, _, _| std::panic::resume_unwind(Box::new("attempt panicked")),
                )?;
                Ok(())
            })
            .is_err();
            let _ = tx.send(panicked);
        });
        anyhow::ensure!(
            rx.recv_timeout(Duration::from_secs(10))?,
            "the attempt's panic did not reach the caller",
        );
        Ok(())
    }

    #[rstest]
    #[case::text(json!([
        {"role": "system", "content": "Answer with a letter."},
        {"role": "user", "content": "2+2?"},
    ]))]
    #[case::vision(json!([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
        {"type": "text", "text": "Which shape is red?"},
    ]}]))]
    fn messages_serialize_as_sent(#[case] wire: Value) -> anyhow::Result<()> {
        let chat: Vec<ChatMessage> = serde_json::from_value(wire.clone())?;
        for messages in [EvalMessages::Raw(&wire), EvalMessages::Chat(&chat)] {
            assert_eq!(serde_json::to_value(messages)?, wire);
        }
        Ok(())
    }

    #[rstest]
    #[case::text(json!([{"role": "user", "content": "2+2?"}]), Some("2+2?"))]
    #[case::vision(
        json!([{"role": "user", "content": [
            {"type": "text", "text": "Look:"},
            {"type": "text", "text": "Which shape is red?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
        ]}]),
        Some("Which shape is red?")
    )]
    #[case::image_only(
        json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}},
        ]}]),
        None
    )]
    #[case::no_messages(json!([]), None)]
    fn the_preview_is_the_last_text(
        #[case] wire: Value,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let chat: Vec<ChatMessage> = serde_json::from_value(wire)?;
        assert_eq!(EvalMessages::Chat(&chat).last_text(), expected);
        Ok(())
    }
}
