//! The one `llama-server` an eval's in-flight samples share, and what a failed
//! request comes to: a resend after a dropped connection, `Lost` when the
//! server died under it, `Failed` when it outlives the crash window. Shared by
//! `eval`, `vlm_eval` and `asr_eval`, with the chat-completion stream read and
//! the readiness gate the two multimodal eval cells share.

use std::io::{BufReader, Write};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use anyhow::Context;
use reqwest::header::CONNECTION;
use serde_json::Value;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_http::StreamClient;
use pipette_ops::measurement::{self, EvalMeasurementReading, EvalMeasurementSample};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::sse::{self, ChatReply};
use pipette_subprocess::{Process, start_process};

use crate::server::{self, LoggingMarker, ServerCommandBuilder};

/// The multimodal chat endpoint the `vlm_eval` and `asr_eval` cells stream
/// from, as the stream read's errors name it.
const CHAT_ENDPOINT: &str = "/v1/chat/completions";

/// A poisoned slot is still usable: it only changes under its own lock, and one
/// panic should not become N errors.
pub(crate) fn lock_or_poisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A running `llama-server` and the URL it listens on, which a restart after a
/// crash changes.
pub(crate) type LlamaServer = (Process, String);

/// The newest 256 KiB of `llama_server`'s stderr: what it has logged so far
/// while it runs, and all of that once it has exited or been stopped.
pub(crate) fn stderr_of(llama_server: &Process) -> String {
    llama_server.output().runtime_stderr.unwrap_or_default()
}

/// Starts the first server and every restart, so a replacement runs the
/// crashed one's argv.
pub(crate) fn spawn_server(
    launch: &ServerCommandBuilder,
    request_timeout: Duration,
) -> anyhow::Result<LlamaServer> {
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, request_timeout)?;
    let command = launch.build(port)?;
    let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
    llama_server.wait_ready(request_timeout, ready)?;
    Ok((llama_server, url))
}

/// What one slot's share of the context must hold.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SlotNeed {
    /// `eval`'s prompts are characters, not tokens, so the floor leaves the
    /// prompt out: a guessed tokens-per-char would either pass a real problem
    /// or refuse a working config.
    Generation { max_tokens: u32 },
    /// `vlm_eval`'s per-sample context: image, prompt and output.
    Sample { ctx: u32 },
}

impl SlotNeed {
    fn tokens(self) -> u32 {
        match self {
            Self::Generation { max_tokens } => max_tokens,
            Self::Sample { ctx } => ctx,
        }
    }

    fn shortfall(self) -> String {
        match self {
            Self::Generation { max_tokens } => format!(
                "under this benchmark's max_tokens of {max_tokens}, so every generation would \
                 truncate"
            ),
            Self::Sample { ctx } => format!(
                "under this cell's per-sample context of {ctx}, so every sample would overflow"
            ),
        }
    }
}

/// Refuse a slot budget that cannot hold what one sample needs.
///
/// llama-server divides `--ctx-size` across its slots, so a generous-looking
/// total can leave each slot short, and every generation then stops early with
/// nothing in the results to say why.
///
/// Both halves come from what will reach argv. `raw_slots` is a count passed
/// through `raw`, which is not scaled (see [`server::raw_parallel`]) and so
/// divides a context sized for one slot.
pub(crate) fn check_slot_context(
    runner: LoggingMarker,
    parallel: u32,
    raw_slots: Option<u32>,
    operator_ctx: Option<u32>,
    per_cell_ctx: u32,
    need: SlotNeed,
) -> anyhow::Result<()> {
    // A raw flag only counts when the typed knob is absent; supplying both is
    // already refused, and the typed one is what the builder renders.
    let slots = parallel.max(raw_slots.unwrap_or(1));
    if slots <= 1 {
        return Ok(());
    }
    // What `build` will render: an operator total verbatim, the per-cell
    // default multiplied by the TYPED slot count only.
    let total = operator_ctx.unwrap_or_else(|| per_cell_ctx.saturating_mul(parallel));
    let per_slot = total / slots;
    let (knob, lever) = match operator_ctx {
        Some(_) => ("ctx_size", "Raise ctx_size"),
        None => ("the per-cell default", "Set an explicit ctx_size"),
    };
    let needed = need.tokens();
    anyhow::ensure!(
        per_slot >= needed,
        "{runner}: {knob} {total} across {slots} slots leaves {per_slot} tokens per slot, {}. \
         {lever} to at least {} or lower parallel",
        need.shortfall(),
        needed.saturating_mul(slots),
    );
    Ok(())
}

pub(crate) struct ServerSlot {
    pub(crate) server: Server,
    /// Which start the server is, so a failed request can tell whether the
    /// server it went to is still the one here.
    pub(crate) generation: u64,
}

pub(crate) enum Server {
    Up(LlamaServer),
    /// Exited, kept for the argv and log the response reports; the next sample
    /// to need a server starts a replacement.
    Down(Process),
    /// The replacement did not start. The sample that tried ends the cell with
    /// that error; the others are `Lost` rather than each trying again.
    RestartFailed,
}

/// The server and its generation, or `None` once a restart has failed. After a
/// crash the next sample to need one starts it here, once the entry has
/// recorded the sample that crashed it alone, so a failed restart still leaves
/// that row for a resume to skip.
pub(crate) fn slot_server(
    slot: &Mutex<ServerSlot>,
    start: &dyn Fn() -> anyhow::Result<LlamaServer>,
) -> anyhow::Result<Option<(String, u64)>> {
    let mut slot = lock_or_poisoned(slot);
    match &slot.server {
        Server::Up((_, url)) => return Ok(Some((url.clone(), slot.generation))),
        Server::RestartFailed => return Ok(None),
        Server::Down(_) => {}
    }
    tracing::info!("restarting llama-server after crash");
    let (llama_server, url) = start()
        .inspect_err(|_| slot.server = Server::RestartFailed)
        .context("llama-server failed to come back after restart")?;
    slot.generation = slot.generation.saturating_add(1);
    slot.server = Server::Up((llama_server, url.clone()));
    Ok(Some((url, slot.generation)))
}

/// How many times a sample whose connection closed with no response is sent again.
pub(crate) const DROPPED_CONNECTION_RESENDS: u32 = 2;

/// `send` the sample until it reads. After a dropped connection it is sent
/// again, up to [`DROPPED_CONNECTION_RESENDS`] times, if the server outlives
/// the crash window, which is also the wait before each resend.
pub(crate) fn read_sample(
    slot: &Mutex<ServerSlot>,
    generation: u64,
    sample: &EvalMeasurementSample<'_>,
    argv: &[String],
    endpoint: &str,
    send: impl Fn() -> anyhow::Result<EvalMeasurementReading>,
) -> anyhow::Result<EvalMeasurementReading> {
    let mut resends = 0;
    loop {
        let err = match send() {
            Ok(reading) => return Ok(reading),
            Err(err) => err,
        };
        let resend = resends < DROPPED_CONNECTION_RESENDS && is_dropped_connection(&err);
        match failure_reading(slot, generation, &err, resend, sample, argv, endpoint)? {
            Verdict::Record(reading) => return Ok(reading),
            Verdict::Resend => {
                resends += 1;
                tracing::warn!(
                    sample = sample.position.as_str(),
                    sample_id = %sample.id,
                    resend = resends,
                    error = %format_args!("{err:#}"),
                    "llama-server closed the connection with no response and is still up; \
                     sending the sample again"
                );
            }
        }
    }
}

/// Whether `err` is a request whose connection closed before any response,
/// rather than a refused connect or a timeout.
pub(crate) fn is_dropped_connection(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<pipette_http::Error>(),
            Some(pipette_http::Error::Request(e))
                if e.is_request() && !e.is_connect() && !e.is_timeout()
        )
    })
}

pub(crate) enum Verdict {
    Record(EvalMeasurementReading),
    /// The server outlived the window and the caller asked for a resend.
    Resend,
}

/// `Lost` when the server has exited, `Failed` (or a resend, when `resend`
/// asks) when it outlives [`CRASH_OBSERVATION_WINDOW`], and an error when the
/// backend could not allocate, since a restart would meet the same ceiling. The
/// window is polled with the slot released: holding it would stall every other
/// sample.
pub(crate) fn failure_reading(
    slot: &Mutex<ServerSlot>,
    generation: u64,
    err: &anyhow::Error,
    resend: bool,
    sample: &EvalMeasurementSample<'_>,
    argv: &[String],
    endpoint: &str,
) -> anyhow::Result<Verdict> {
    poll_until_arbitrated(
        CRASH_OBSERVATION_WINDOW,
        CRASH_POLL_INTERVAL,
        &std::thread::sleep,
        |deadline| {
            arbitrate(
                slot, generation, err, resend, sample, argv, endpoint, deadline,
            )
        },
    )
}

/// One pass of [`failure_reading`]. `None` means the server has not surfaced
/// as exited and the window has not run out yet.
#[expect(
    clippy::too_many_arguments,
    reason = "the failed request's parts, and the pool it failed against"
)]
pub(crate) fn arbitrate(
    slot: &Mutex<ServerSlot>,
    generation: u64,
    err: &anyhow::Error,
    resend: bool,
    sample: &EvalMeasurementSample<'_>,
    argv: &[String],
    endpoint: &str,
    deadline: Instant,
) -> anyhow::Result<Option<Verdict>> {
    let mut guard = lock_or_poisoned(slot);
    let slot = &mut *guard;
    // Unless this is still the server the request went to, another sample
    // already saw it exit, took it out and logged why; a restart only ever
    // follows that.
    let (Server::Up((llama_server, _)), true) = (&mut slot.server, slot.generation == generation)
    else {
        return Ok(Some(Verdict::Record(EvalMeasurementReading::Lost {
            reason: format!(
                "llama-server went away mid-completion; client saw: {}",
                root_cause(err)
            ),
        })));
    };
    // A zero timeout asks once; the waiting is the caller's, with the slot
    // released.
    let Some(status) = llama_server.wait(Duration::ZERO)? else {
        if Instant::now() < deadline {
            return Ok(None);
        }
        if resend {
            return Ok(Some(Verdict::Resend));
        }
        let stderr = stderr_of(llama_server);
        drop(guard);
        let reason = build_transport_failure_reason(endpoint, err);
        log_sample_failure(sample, false, &reason, err, &stderr);
        return Ok(Some(Verdict::Record(EvalMeasurementReading::Failed {
            reason,
        })));
    };
    let stderr = stderr_of(llama_server);
    slot.server = match std::mem::replace(&mut slot.server, Server::RestartFailed) {
        Server::Up((dead, _)) => Server::Down(dead),
        other @ (Server::Down(_) | Server::RestartFailed) => other,
    };
    drop(guard);
    let reason = build_crash_reason(status, err);
    log_sample_failure(sample, true, &reason, err, &stderr);
    if is_allocation_failure(&stderr) {
        return Err(allocation_failure_error(argv, &stderr));
    }
    Ok(Some(Verdict::Record(EvalMeasurementReading::Lost {
        reason,
    })))
}

/// How long a failed request waits for `llama-server` to surface as exited.
/// Seconds, not one check: on Windows an exit can take a second or more to
/// reach the parent (STATUS_STACK_OVERFLOW typically lands in 100-500 ms;
/// connection resets under SSH can lag further).
pub(crate) const CRASH_OBSERVATION_WINDOW: Duration = Duration::from_secs(3);

pub(crate) const CRASH_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Split from [`failure_reading`] over injected durations and sleep, so a test
/// need not spend three seconds per case.
///
/// The deadline is computed once and handed to every pass: recomputing it
/// would extend the window each pass, and a server that never exits would hold
/// its worker forever.
pub(crate) fn poll_until_arbitrated<T>(
    window: Duration,
    interval: Duration,
    sleep: &dyn Fn(Duration),
    mut arbitrate: impl FnMut(Instant) -> anyhow::Result<Option<T>>,
) -> anyhow::Result<T> {
    let deadline = Instant::now() + window;
    loop {
        if let Some(answer) = arbitrate(deadline)? {
            return Ok(answer);
        }
        sleep(interval);
    }
}

/// The deepest cause, e.g. `connection error: forcibly closed (os error
/// 10054)`, for the checkpoint's `failed_reason` and the mgmt rows.
pub(crate) fn root_cause(err: &anyhow::Error) -> String {
    err.chain()
        .last()
        .map(ToString::to_string)
        .unwrap_or_else(|| err.to_string())
}

/// The `[<rfc3339>]` prefix carries the observed-at time on the wire without a
/// field of its own (see `BenchmarkEvalCompletion::failed_reason`).
pub(crate) fn build_crash_reason(
    exit_status: std::process::ExitStatus,
    err: &anyhow::Error,
) -> String {
    let observed_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "[{observed_at}] llama-server crashed mid-completion: exit {exit_status}; client saw: {}",
        root_cause(err),
    )
}

/// What llama.cpp prints when a graph tensor has no backing buffer. In
/// practice that is a failed device allocation surfacing as a NULL rather
/// than a clean error, and the assert turns it into `SIGABRT`.
pub(crate) const ALLOCATION_ASSERT: &str = "tensor not allocated";

/// A restart cures a prompt that kills the server, not running out of device
/// memory: the replacement runs the same argv and meets the same ceiling, each
/// round resending every sample in flight, and a restart that cannot allocate
/// at all is fatal to the cell. Better to stop and say which knob to turn.
pub(crate) fn is_allocation_failure(stderr: &str) -> bool {
    stderr.contains(ALLOCATION_ASSERT)
}

pub(crate) fn allocation_failure_error(argv: &[String], stderr: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "llama-server could not allocate and aborted ({ALLOCATION_ASSERT}); a restart would use \
         the same argv and hit the same ceiling. Lower `parallel`, or set a smaller `ctx_size` \
         (it is the total across all slots). Flags were: {}\nstderr tail:\n{}",
        argv.join(" "),
        measurement::tail_lines(stderr, 10),
    )
}

pub(crate) fn build_transport_failure_reason(endpoint: &str, err: &anyhow::Error) -> String {
    let observed_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "[{observed_at}] llama-server {endpoint} failed (server still alive): {}",
        root_cause(err),
    )
}

/// `lost` means the server died under the sample: the entry decides what
/// becomes of it, so the line does not say failed.
pub(crate) fn log_sample_failure(
    sample: &EvalMeasurementSample<'_>,
    lost: bool,
    reason: &str,
    err: &anyhow::Error,
    stderr: &str,
) {
    let prompt_preview = sample
        .messages
        .last_text()
        .map(|s| s.chars().take(80).collect::<String>())
        .unwrap_or_default();
    let prompt_preview = prompt_preview.as_str();
    if lost {
        tracing::warn!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            reason,
            prompt_preview,
            error = %format_args!("{err:#}"),
            "sample lost its server"
        );
    } else {
        tracing::warn!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            reason,
            prompt_preview,
            error = %format_args!("{err:#}"),
            "sample failed"
        );
    }
    if !stderr.is_empty() {
        tracing::warn!(
            stderr_tail = measurement::tail_lines(stderr, 20).as_str(),
            "llama-server stderr tail"
        );
    }
}

pub(crate) fn mcq_grammar(choices: &[String]) -> String {
    let alts = choices
        .iter()
        .map(|choice| format!("{choice:?}"))
        .collect::<Vec<_>>()
        .join(" | ");
    format!("root ::= {alts}")
}

/// The readiness gate of a multimodal eval (`vlm_eval`, `asr_eval`). Never
/// fails the run: unlike vl_throughput's numbers, the pinned decode (greedy or
/// seeded) answers the same warm.
pub(crate) fn readiness_best_effort(readiness_gate: ReadinessGate, context: &str) {
    if let Err(e) = readiness_gate() {
        tracing::warn!(
            context,
            error = %format_args!("{e:#}"),
            "readiness gate did not settle, proceeding warm"
        );
    }
}

/// The stderr lock is a LEAF and must stay one: it is held for the whole
/// generation, and it blocks across threads, so anything that acquired a pool
/// lock first would stall every worker for the length of a completion.
pub(crate) fn stream_chat_completion(
    client: &StreamClient,
    url: &str,
    body: &Value,
    tee_stderr: bool,
) -> anyhow::Result<ChatReply> {
    // A non-2xx keeps the head of the server's body: a ctx-overflow or
    // image-decode 400 carries the real message.
    let response = pipette_http::send(
        client
            .http()
            .post(url)
            .header(CONNECTION, "close")
            .json(body),
    )
    .with_context(|| format!("failed to call {CHAT_ENDPOINT}"))?;
    let events = sse::data_lines(BufReader::new(response), client.idle_timeout());
    if !tee_stderr {
        // Taking the lock lazily inside the closure would bring back the very
        // hold this avoids.
        return read_chat_stream(events, |_| {});
    }
    let mut stderr = std::io::stderr().lock();
    let outcome = read_chat_stream(events, |pending| {
        let _ = write!(stderr, "{pending}");
        let _ = stderr.flush();
    })?;
    let _ = writeln!(stderr);
    let _ = stderr.flush();
    Ok(outcome)
}

/// A stream that reaches `[DONE]` without a `finish_reason` was truncated, so
/// it is an `Err`: scoring the partial text would checkpoint it `failed: false`.
pub(crate) fn read_chat_stream(
    events: impl IntoIterator<Item = anyhow::Result<String>>,
    mut on_flush: impl FnMut(&str),
) -> anyhow::Result<ChatReply> {
    let reply = sse::read_chat_stream(events, &mut on_flush)?;
    anyhow::ensure!(
        reply.finish_reason.is_some(),
        "{CHAT_ENDPOINT} stream ended without finish_reason (connection closed early?); \
         discarding {} chars of partial content",
        reply.content.len()
    );
    Ok(reply)
}

/// The per-sample line a completed reading logs, with its text at debug: the
/// text is in the eval checkpoint and the result already, and at info it made
/// up nearly all of a pool job's log.
pub(crate) fn log_completed(sample: &EvalMeasurementSample<'_>, reading: &EvalMeasurementReading) {
    if let EvalMeasurementReading::Completed {
        completion,
        stop_reason,
        stop_detail,
        completion_tokens,
    } = reading
    {
        tracing::info!(
            sample = sample.position.as_str(),
            sample_id = %sample.id,
            completion_chars = completion.chars().count(),
            ?stop_reason,
            ?completion_tokens,
            ?stop_detail,
            "sample completed"
        );
        tracing::debug!(
            sample_id = %sample.id,
            completion = completion.as_str(),
            "sample completion text"
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::serial_is_never_checked(1, None, Some(1024), 8192, true)]
    #[case::no_operator_ctx_uses_the_scaled_default(64, None, None, 8192, true)]
    #[case::exactly_enough(64, None, Some(64 * 8192), 8192, true)]
    #[case::comfortable(64, None, Some(64 * 9216), 8192, true)]
    #[case::one_token_short(64, None, Some(64 * 8192 - 64), 8192, false)]
    #[case::looks_generous_but_is_not(64, None, Some(65_536), 8192, false)]
    #[case::raw_slots_divide_an_operator_total(1, Some(8), Some(32_768), 8192, false)]
    #[case::raw_slots_divide_the_unscaled_default(1, Some(8), None, 8192, false)]
    #[case::raw_slots_with_room_to_spare(1, Some(2), Some(65_536), 8192, true)]
    // A small `max_tokens` leaves each slot plenty, so a config that has always
    // worked is not refused.
    #[case::raw_slots_small_max_tokens(1, Some(2), Some(4096), 16, true)]
    fn slot_context_must_hold_one_generation(
        #[case] parallel: u32,
        #[case] raw_slots: Option<u32>,
        #[case] operator_ctx: Option<u32>,
        #[case] max_tokens: u32,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let per_cell_ctx = 8192u32.saturating_add(max_tokens);
        let got = check_slot_context(
            LoggingMarker::Eval,
            parallel,
            raw_slots,
            operator_ctx,
            per_cell_ctx,
            SlotNeed::Generation { max_tokens },
        );
        match got {
            Ok(()) => anyhow::ensure!(accepted, "{operator_ctx:?} over {parallel} should refuse"),
            Err(e) => {
                anyhow::ensure!(
                    !accepted,
                    "{operator_ctx:?} over {parallel} should pass: {e}"
                );
                let msg = format!("{e}");
                anyhow::ensure!(msg.starts_with("eval:"), "got: {msg}");
                anyhow::ensure!(msg.contains("per slot"), "got: {msg}");
                anyhow::ensure!(msg.contains("max_tokens of"), "got: {msg}");
            }
        }
        Ok(())
    }
}
