//! Progress on stderr (stdout carries results): a cell's fetches, as the line
//! of the artifact fetching now under one summing the cell when it has several,
//! and the long waits ([`WaitProgress`]: the readiness gate and the reindex
//! hold). `pipette-artifacts` knows nothing of cells, so the cell total is
//! assembled here from sizes declared up front. Off a terminal a redraw is a
//! log event instead.

use std::io::{IsTerminal, Write};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use pipette_artifacts::{FetchProgress, ProgressSink};
use pipette_wait::WaitStatus;

use crate::client::worker::{ReindexPhase, ReindexStatus};

/// Redraw interval. Fast enough to look live, slow enough that a 3 GB download
/// costs a few hundred writes rather than four hundred thousand.
const TTY_REDRAW: Duration = Duration::from_millis(100);
/// Off a terminal: dense enough to show a stall, sparse enough not to bury the
/// run's own output.
const LOG_REDRAW: Duration = Duration::from_secs(15);

/// Elapsed time a rate needs before it means anything. One 8 KiB buffer over the
/// microseconds of a first read divides out to gigabytes per second.
const RATE_FLOOR: f64 = 0.25;

/// Renders a cell's fetches. A cache hit reports no bytes, so draws nothing.
pub struct CellProgress {
    state: Arc<Mutex<State>>,
    tty: bool,
    /// More than one artifact, so a sum line is worth drawing.
    composite: bool,
}

struct State {
    done: Vec<(String, u64)>,
    /// What the whole cell will fetch, when every part could be sized.
    cell_total: Option<u64>,
    last_draw: Option<Instant>,
    /// When the first byte landed, not construction: counting the size lookups
    /// made the first rate read a fraction of the real one.
    started: Option<Instant>,
    /// Lines the last draw left on screen, to erase before the next one.
    drawn_lines: usize,
}

impl CellProgress {
    /// One size per artifact: `Some(0)` for one already in the store, `None` for
    /// one that could not be sized, which leaves the cell total unknown rather
    /// than guessed.
    pub fn new(planned: &[Option<u64>]) -> Self {
        let cell_total = planned
            .iter()
            .try_fold(0_u64, |acc, total| Some(acc.saturating_add((*total)?)));
        Self {
            state: Arc::new(Mutex::new(State {
                done: Vec::new(),
                cell_total,
                last_draw: None,
                started: None,
                drawn_lines: 0,
            })),
            tty: std::io::stderr().is_terminal(),
            composite: planned.len() > 1,
        }
    }

    pub fn sink(&self) -> ProgressSink {
        let state = Arc::clone(&self.state);
        let tty = self.tty;
        let composite = self.composite;
        Arc::new(move |p: FetchProgress<'_>| {
            // A previous render panicked: stop drawing rather than fail a fetch.
            let Ok(mut state) = state.lock() else {
                return;
            };
            state.record(p.artifact, p.done_bytes);
            let interval = if tty { TTY_REDRAW } else { LOG_REDRAW };
            if !state.due(interval) {
                return;
            }
            if tty {
                state.draw(&p, composite);
            } else {
                state.log(&p, composite);
            }
        })
    }

    #[cfg(test)]
    fn total(&self) -> anyhow::Result<Option<u64>> {
        Ok(self
            .state
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .cell_total)
    }
}

/// A failed fetch would otherwise leave its last line above the error.
impl Drop for CellProgress {
    fn drop(&mut self) {
        if !self.tty {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            state.erase();
        }
    }
}

impl State {
    fn record(&mut self, artifact: &str, done_bytes: u64) {
        self.started.get_or_insert_with(Instant::now);
        match self.done.iter_mut().find(|(name, _)| name == artifact) {
            Some((_, done)) => *done = done_bytes,
            None => self.done.push((artifact.to_owned(), done_bytes)),
        }
    }

    fn due(&mut self, interval: Duration) -> bool {
        let now = Instant::now();
        let due = self.last_draw.is_none_or(|last| now - last >= interval);
        if due {
            self.last_draw = Some(now);
        }
        due
    }

    fn cell_done(&self) -> u64 {
        self.done.iter().map(|(_, done)| done).sum()
    }

    fn draw(&mut self, current: &FetchProgress<'_>, composite: bool) {
        let mut out = std::io::stderr().lock();
        self.erase_into(&mut out);
        let lines = self.frame(current, composite);
        let _ = writeln!(out, "{}", lines.join("\n"));
        self.drawn_lines = lines.len();
        let _ = out.flush();
    }

    fn log(&self, current: &FetchProgress<'_>, composite: bool) {
        tracing::info!(
            artifact = %current.artifact,
            file = %current.file,
            done_bytes = current.done_bytes,
            total_bytes = current.total_bytes,
            cell_done_bytes = composite.then(|| self.cell_done()),
            cell_total_bytes = self.cell_total.filter(|_| composite),
            bytes_per_sec = self.rate(),
            "fetching"
        );
    }

    /// Across the cell, once [`RATE_FLOOR`] has elapsed.
    fn rate(&self) -> Option<f64> {
        let elapsed = self.started?.elapsed().as_secs_f64();
        (elapsed >= RATE_FLOOR).then(|| self.cell_done() as f64 / elapsed)
    }

    fn frame(&self, current: &FetchProgress<'_>, composite: bool) -> Vec<String> {
        let rate = self.rate().map(rate_of).unwrap_or_default();
        let lines = if composite {
            vec![
                format!(
                    "fetching {} {rate}",
                    bytes_of(self.cell_done(), self.cell_total)
                ),
                format!("  {}", name_of(current)),
            ]
        } else {
            vec![format!("fetching {} {rate}", name_of(current))]
        };
        // The rate is blank at first, and a trailing space is still a character
        // the erase has to cover.
        lines
            .into_iter()
            .map(|line| line.trim_end().to_owned())
            .collect()
    }

    fn erase(&mut self) {
        let mut out = std::io::stderr().lock();
        self.erase_into(&mut out);
        let _ = out.flush();
    }

    fn erase_into(&mut self, out: &mut impl Write) {
        // Up one line and clear it, for each line the last draw wrote. `\r` alone
        // would leave the tail of a longer previous line behind.
        (0..self.drawn_lines).for_each(|_| {
            let _ = write!(out, "\x1b[1A\x1b[2K");
        });
        self.drawn_lines = 0;
    }
}

/// The artifact and the file moving, the file dropped when the artifact's name
/// already ends in it.
fn name_of(current: &FetchProgress<'_>) -> String {
    let bytes = bytes_of(current.done_bytes, current.total_bytes);
    if current.artifact.ends_with(current.file) {
        format!("{} {bytes}", current.artifact)
    } else {
        format!("{} {} {bytes}", current.artifact, current.file)
    }
}

/// `1.4/2.9 GB`, or `48 MB` when the total is unknown. One unit for both halves,
/// chosen from the total, so the pair reads as a single quantity.
fn bytes_of(done: u64, total: Option<u64>) -> String {
    match total {
        Some(total) if total > 0 => {
            let (scale, unit) = unit_for(total as f64);
            format!(
                "{:.1}/{:.1} {unit}",
                done as f64 / scale,
                total as f64 / scale
            )
        }
        _ => {
            let (scale, unit) = unit_for(done as f64);
            format!("{:.1} {unit}", done as f64 / scale)
        }
    }
}

/// Divisor and name for the largest unit `bytes` fills. Decimal, as transfer
/// rates are quoted.
fn unit_for(bytes: f64) -> (f64, &'static str) {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;
    match bytes {
        b if b >= GB => (GB, "GB"),
        b if b >= MB => (MB, "MB"),
        b if b >= KB => (KB, "KB"),
        _ => (1.0, "B"),
    }
}

fn rate_of(bytes_per_sec: f64) -> String {
    let (scale, unit) = unit_for(bytes_per_sec);
    format!("{:.1} {unit}/s", bytes_per_sec / scale)
}

/// What a status adds for [`WaitProgress`]: the terminal row. The log line and
/// its level come from `Display` and [`WaitStatus`].
pub trait LiveFrame: WaitStatus + std::fmt::Display {
    /// Whether the wait is still holding, the only phase worth a live row.
    fn is_live(&self) -> bool;
    /// The row for a live status. Cut to [`MAX_FRAME_WIDTH`] on the way out.
    fn frame(&self) -> String;
}

/// Renders a long wait: one line that rewrites itself on a terminal, the wait's
/// own log line ([`WaitStatus::report_default`]) anywhere else. Installed as the
/// wait's `with_status`, so it takes over the narration rather than competing
/// for the cursor.
pub struct WaitProgress<S> {
    tty: bool,
    id: u64,
    line: Arc<WaitLine>,
    /// `fn(&S)` rather than `S`: the renderer neither owns nor outlives a status.
    status: PhantomData<fn(&S)>,
}

pub type ReadinessProgress = WaitProgress<pipette_readiness::ReadinessStatus>;

pub type ReindexProgress = WaitProgress<ReindexStatus>;

/// The erase clears one terminal row, so a wrapped frame would leave a row
/// behind every poll. 80 is the narrowest terminal worth assuming; a Windows
/// reading listing every sensor is wider.
const MAX_FRAME_WIDTH: usize = 80;

static WAIT_LINE: LazyLock<Arc<WaitLine>> = LazyLock::new(|| Arc::new(WaitLine::default()));
static NEXT_WAIT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct WaitLine {
    state: Mutex<WaitLineState>,
}

#[derive(Default)]
struct WaitLineState {
    owner: Option<u64>,
    /// A status that arrives while a log is being written waits for the next
    /// poll to draw. This also keeps another thread from drawing over a log.
    writers: usize,
}

impl WaitLine {
    fn draw(&self, id: u64, out: &mut impl Write, text: &str) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.writers != 0 {
            return;
        }
        state.clear(out);
        let _ = writeln!(out, "{}", fit(text.trim_end()));
        state.owner = Some(id);
        let _ = out.flush();
    }

    fn erase(&self, id: u64, out: &mut impl Write) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.owner == Some(id) {
            state.clear(out);
            let _ = out.flush();
        }
    }

    fn begin_log(&self, out: &mut impl Write) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        state.clear(out);
        let _ = out.flush();
        state.writers = state.writers.saturating_add(1);
        true
    }

    fn end_log(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.writers -= 1;
        }
    }

    #[cfg(test)]
    fn drawn_by(&self, id: u64) -> bool {
        self.state.lock().is_ok_and(|state| state.owner == Some(id))
    }
}

impl WaitLineState {
    fn clear(&mut self, out: &mut impl Write) {
        if self.owner.take().is_some() {
            let _ = write!(out, "\x1b[1A\x1b[2K");
        }
    }
}

/// The tracing writer clears an active wait row before its event and keeps the
/// row clear until the event finishes. Erasure uses ordinary stderr.
/// `LogStderr` handles closed output for the log event.
pub(crate) struct WaitLogWriter<W> {
    writer: W,
    line: Arc<WaitLine>,
    active: bool,
}

impl<W: Write> WaitLogWriter<W> {
    pub(crate) fn new(writer: W, erase_to: &mut impl Write, tty: bool) -> Self {
        let line = Arc::clone(&WAIT_LINE);
        Self::with_line(writer, erase_to, tty, line)
    }

    fn with_line(writer: W, erase_to: &mut impl Write, tty: bool, line: Arc<WaitLine>) -> Self {
        let active = tty && line.begin_log(erase_to);
        Self {
            writer,
            line,
            active,
        }
    }
}

impl<W: Write> Write for WaitLogWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.write(buf)
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl<W> Drop for WaitLogWriter<W> {
    fn drop(&mut self) {
        if self.active {
            self.line.end_log();
        }
    }
}

impl<S: LiveFrame> Default for WaitProgress<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: LiveFrame> WaitProgress<S> {
    pub fn new() -> Self {
        Self {
            tty: std::io::stderr().is_terminal(),
            id: NEXT_WAIT_ID.fetch_add(1, Ordering::Relaxed),
            line: Arc::clone(&WAIT_LINE),
            status: PhantomData,
        }
    }

    pub fn report(&self, status: &S) {
        if !self.tty {
            // No cursor to draw with, and a journal wants every poll.
            status.report_default();
            return;
        }
        // Erase before logging: a log line written first would be the row
        // it clears.
        self.erase();
        if status.is_live() {
            // The frame already shows the line, so it goes to debug.
            tracing::debug!(summary = status.to_string().as_str(), "wait poll");
            self.draw(&status.frame());
        } else {
            // The end of the wait: the line worth keeping in the scrollback.
            status.report_default();
        }
    }
}

/// Unbounded so [`Drop`] can erase.
impl<S> WaitProgress<S> {
    fn draw(&self, line: &str) {
        // Take the line-state lock before stderr; the tracing writer uses the
        // same order, including when another thread emits a log.
        let mut out = std::io::stderr();
        self.draw_into(&mut out, line);
    }

    fn draw_into(&self, out: &mut impl Write, line: &str) {
        self.line.draw(self.id, out, line);
    }

    fn erase(&self) {
        let mut out = std::io::stderr();
        self.erase_into(&mut out);
    }

    fn erase_into(&self, out: &mut impl Write) {
        self.line.erase(self.id, out);
    }
}

/// A failed or cancelled wait returns without a final status, and its line
/// would otherwise sit above the error.
impl<S> Drop for WaitProgress<S> {
    fn drop(&mut self) {
        if self.tty {
            self.erase();
        }
    }
}

impl LiveFrame for pipette_readiness::ReadinessStatus {
    fn is_live(&self) -> bool {
        self.is_waiting()
    }

    fn frame(&self) -> String {
        cooling_line(self)
    }
}

impl LiveFrame for ReindexStatus {
    fn is_live(&self) -> bool {
        self.phase == ReindexPhase::Waiting
    }

    fn frame(&self) -> String {
        reindex_line(self)
    }
}

/// `waiting for the device to settle 0:20 / 5:00 (<observed>)`. Elapsed against
/// the deadline, not a percentage: the gate races the deadline rather than
/// progressing toward it. The reading says why it is holding.
fn cooling_line(status: &pipette_readiness::ReadinessStatus) -> String {
    format!(
        "waiting for the device to settle {} / {} ({})",
        clock(status.elapsed),
        clock(status.max_wait),
        status.observed,
    )
}

/// `waiting for the server to reindex 0:20 / 5:00 (4 polls)`.
fn reindex_line(status: &ReindexStatus) -> String {
    format!(
        "waiting for the server to reindex {} / {} ({} poll{})",
        clock(status.elapsed),
        clock(status.wait.budget),
        status.polls,
        if status.polls == 1 { "" } else { "s" },
    )
}

/// `line` cut to [`MAX_FRAME_WIDTH`] columns, with `…` marking the cut.
/// Counted in characters, not bytes: the readings carry `°` and `→`.
fn fit(line: &str) -> String {
    if line.chars().count() <= MAX_FRAME_WIDTH {
        return line.to_owned();
    }
    let mut kept: String = line.chars().take(MAX_FRAME_WIDTH - 1).collect();
    kept.push('…');
    kept
}

/// `m:ss`: the values are minutes long.
fn clock(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{}:{:02}", secs / 60, secs % 60)
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::client::worker::ReindexWait;
    use crate::test_support::Written;

    #[test]
    fn one_unsized_artifact_leaves_the_cell_total_unknown() -> anyhow::Result<()> {
        let partly_sized = CellProgress::new(&[Some(100), None]);
        let total = partly_sized.total()?;
        assert_eq!(total, None);

        let sized = CellProgress::new(&[Some(100), Some(250)]);
        assert_eq!(sized.total()?, Some(350));
        Ok(())
    }

    /// An artifact reports its own running total, not a delta.
    #[test]
    fn the_cell_sums_the_latest_from_each_artifact() {
        let mut state = State {
            done: Vec::new(),
            cell_total: Some(300),
            last_draw: None,
            started: None,
            drawn_lines: 0,
        };
        state.record("runtime", 50);
        state.record("model", 100);
        state.record("model", 180);

        assert_eq!(state.cell_done(), 230);
    }

    #[test]
    fn draws_are_rate_limited() {
        let mut state = State {
            done: Vec::new(),
            cell_total: None,
            last_draw: None,
            started: None,
            drawn_lines: 0,
        };
        assert!(
            state.due(Duration::from_millis(100)),
            "the first draw is due"
        );
        assert!(
            !state.due(Duration::from_millis(100)),
            "an immediate redraw is not"
        );
        assert!(state.due(Duration::ZERO), "a lapsed interval is");
    }

    /// A 48 MB archive on a GB scale would read `0.0/0.1 GB` and look stalled.
    #[rstest]
    #[case(1_400_000_000, Some(2_900_000_000), "1.4/2.9 GB")]
    #[case(48_200_000, Some(61_400_000), "48.2/61.4 MB")]
    #[case(512, Some(2_048), "0.5/2.0 KB")]
    // Unknown total: the unit follows what has arrived.
    #[case(1_400_000_000, None, "1.4 GB")]
    #[case(48_200_000, None, "48.2 MB")]
    // A zero total is a size lookup that answered with nothing to say.
    #[case(500_000_000, Some(0), "500.0 MB")]
    fn bytes_render_in_the_unit_the_total_fills(
        #[case] done: u64,
        #[case] total: Option<u64>,
        #[case] expected: &str,
    ) {
        assert_eq!(bytes_of(done, total), expected);
    }

    #[test]
    fn the_file_is_named_only_when_it_adds_something() {
        let single = FetchProgress {
            artifact: "org/repo:weights.gguf",
            file: "weights.gguf",
            done_bytes: 5_000_000,
            total_bytes: Some(10_000_000),
        };
        assert_eq!(name_of(&single), "org/repo:weights.gguf 5.0/10.0 MB");

        let part_of_many = FetchProgress {
            artifact: "org/repo",
            file: "model-00002-of-00003.safetensors",
            done_bytes: 5_000_000,
            total_bytes: Some(10_000_000),
        };
        assert_eq!(
            name_of(&part_of_many),
            "org/repo model-00002-of-00003.safetensors 5.0/10.0 MB"
        );
    }

    #[rstest]
    #[case(12_400_000.0, "12.4 MB/s")]
    #[case(1_200_000_000.0, "1.2 GB/s")]
    #[case(900.0, "900.0 B/s")]
    fn rates_scale_with_the_throughput(#[case] bytes_per_sec: f64, #[case] expected: &str) {
        assert_eq!(rate_of(bytes_per_sec), expected);
    }

    #[rstest]
    #[case(0, "0:00")]
    #[case(9, "0:09")]
    #[case(20, "0:20")]
    #[case(300, "5:00")]
    #[case(420, "7:00")]
    #[case(3_661, "61:01")]
    fn a_cooldown_reads_as_a_clock(#[case] secs: u64, #[case] expected: &str) {
        assert_eq!(clock(Duration::from_secs(secs)), expected);
    }

    #[test]
    fn the_cooling_line_states_the_budget_and_the_reason() {
        let status = pipette_readiness::ReadinessStatus {
            phase: pipette_readiness::ReadinessPhase::Waiting {
                poll_in: Duration::from_millis(1_800),
            },
            platform: pipette_readiness::ReadinessPlatform::MacOs,
            observed: "thermal=pressure:1(moderate) cpu=busy:0.2cores(ok)".to_string(),
            elapsed: Duration::from_secs(20),
            max_wait: Duration::from_secs(420),
        };
        assert_eq!(
            cooling_line(&status),
            "waiting for the device to settle 0:20 / 7:00 \
             (thermal=pressure:1(moderate) cpu=busy:0.2cores(ok))",
        );
    }

    #[rstest]
    #[case::one(1, "waiting for the server to reindex 0:20 / 5:00 (1 poll)")]
    #[case::several(4, "waiting for the server to reindex 0:20 / 5:00 (4 polls)")]
    fn the_reindex_line_counts_its_polls(#[case] polls: usize, #[case] expected: &str) {
        let status = ReindexStatus {
            phase: ReindexPhase::Waiting,
            polls,
            elapsed: Duration::from_secs(20),
            wait: ReindexWait {
                poll_interval: Duration::from_secs(5),
                budget: Duration::from_secs(300),
            },
        };
        assert_eq!(reindex_line(&status), expected);
    }

    fn fetching(done_bytes: u64) -> FetchProgress<'static> {
        FetchProgress {
            artifact: "org/repo:weights.gguf",
            file: "weights.gguf",
            done_bytes,
            total_bytes: Some(10_000_000),
        }
    }

    /// The rate is blank until [`RATE_FLOOR`] has elapsed.
    #[test]
    fn a_frame_is_the_sum_over_the_artifact_and_one_row_alone() -> anyhow::Result<()> {
        let mut state = State {
            done: Vec::new(),
            cell_total: Some(30_000_000),
            last_draw: None,
            started: None,
            drawn_lines: 0,
        };
        state.record("org/repo:weights.gguf", 5_000_000);
        state.record("runtime", 10_000_000);
        let current = fetching(5_000_000);

        assert_eq!(
            state.frame(&current, true),
            [
                "fetching 15.0/30.0 MB",
                "  org/repo:weights.gguf 5.0/10.0 MB"
            ]
        );
        assert_eq!(
            state.frame(&current, false),
            ["fetching org/repo:weights.gguf 5.0/10.0 MB"]
        );

        state.started = Instant::now().checked_sub(Duration::from_secs(1));
        anyhow::ensure!(state.started.is_some(), "a second has passed since boot");
        let with_rate = state.frame(&current, true);
        assert!(
            with_rate[0].starts_with("fetching 15.0/30.0 MB ") && with_rate[0].ends_with(" MB/s"),
            "got {with_rate:?}"
        );
        assert_eq!(with_rate[1], "  org/repo:weights.gguf 5.0/10.0 MB");
        Ok(())
    }

    /// The rate stays out until it means something.
    #[test]
    fn a_redraw_off_a_terminal_is_one_event_with_the_numbers() -> anyhow::Result<()> {
        let mut state = State {
            done: Vec::new(),
            cell_total: Some(30_000_000),
            last_draw: None,
            started: None,
            drawn_lines: 0,
        };
        state.record("org/repo:weights.gguf", 5_000_000);
        state.record("runtime", 10_000_000);
        let written = Written::default();
        let writer = written.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_writer(move || writer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            state.log(&fetching(5_000_000), true);
        });

        let event: serde_json::Value = serde_json::from_str(&written.text()?)?;
        assert_eq!(event["message"], "fetching");
        assert_eq!(event["artifact"], "org/repo:weights.gguf");
        assert_eq!(event["file"], "weights.gguf");
        assert_eq!(event["done_bytes"], 5_000_000);
        assert_eq!(event["total_bytes"], 10_000_000);
        assert_eq!(event["cell_done_bytes"], 15_000_000);
        assert_eq!(event["cell_total_bytes"], 30_000_000);
        assert!(event.get("bytes_per_sec").is_none(), "got {event}");
        Ok(())
    }

    #[test]
    fn the_cell_erase_clears_each_drawn_row_once() {
        let mut state = State {
            done: Vec::new(),
            cell_total: None,
            last_draw: None,
            started: None,
            drawn_lines: 2,
        };
        let mut out = Vec::new();
        state.erase_into(&mut out);
        assert_eq!(out, b"\x1b[1A\x1b[2K\x1b[1A\x1b[2K");
        let mut again = Vec::new();
        state.erase_into(&mut again);
        assert!(again.is_empty());
    }

    #[test]
    fn the_sink_records_what_each_callback_reports() -> anyhow::Result<()> {
        let progress = CellProgress::new(&[Some(10_000_000), Some(20_000_000)]);
        let sink = progress.sink();
        sink(fetching(1_000_000));
        sink(fetching(4_000_000));
        let done = progress
            .state
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .cell_done();
        assert_eq!(done, 4_000_000);
        Ok(())
    }

    fn readiness(waiting: bool) -> pipette_readiness::ReadinessStatus {
        pipette_readiness::ReadinessStatus {
            phase: if waiting {
                pipette_readiness::ReadinessPhase::Waiting {
                    poll_in: Duration::from_secs(1),
                }
            } else {
                pipette_readiness::ReadinessPhase::Proceeding
            },
            platform: pipette_readiness::ReadinessPlatform::MacOs,
            observed: "thermal=nominal".to_string(),
            elapsed: Duration::from_secs(20),
            max_wait: Duration::from_secs(300),
        }
    }

    fn test_progress(tty: bool) -> ReadinessProgress {
        ReadinessProgress {
            tty,
            id: 1,
            line: Arc::new(WaitLine::default()),
            status: PhantomData,
        }
    }

    #[rstest]
    #[case::log(false)]
    #[case::terminal(true)]
    fn a_poll_is_a_row_on_a_terminal_and_a_log_line_elsewhere(#[case] tty: bool) {
        let progress = test_progress(tty);
        progress.report(&readiness(true));
        assert_eq!(progress.line.drawn_by(progress.id), tty);
        progress.report(&readiness(false));
        assert!(
            !progress.line.drawn_by(progress.id),
            "the end of the wait leaves no row"
        );
    }

    /// A second erase would eat whatever the engine printed above the row.
    #[test]
    fn erasing_twice_only_clears_what_was_drawn() {
        let progress = test_progress(true);
        progress.draw_into(&mut Vec::new(), "first");
        let mut first = Vec::new();
        progress.erase_into(&mut first);
        let mut second = Vec::new();
        progress.erase_into(&mut second);

        assert_eq!(first, b"\x1b[1A\x1b[2K");
        assert!(second.is_empty(), "nothing is on screen to erase");
    }

    #[test]
    fn a_wait_is_one_row_rewritten_then_erased() {
        let progress = test_progress(true);
        let mut out = Vec::new();
        progress.draw_into(&mut out, "first");
        progress.draw_into(&mut out, "second");
        progress.erase_into(&mut out);

        assert_eq!(
            String::from_utf8_lossy(&out),
            "first\n\x1b[1A\x1b[2Ksecond\n\x1b[1A\x1b[2K",
        );
        assert!(
            !progress.line.drawn_by(progress.id),
            "nothing is left on screen"
        );
    }

    #[rstest]
    #[case::text(false)]
    #[case::json(true)]
    fn logs_clear_the_frame_once_and_leave_the_failure_visible(
        #[case] json: bool,
    ) -> anyhow::Result<()> {
        let progress = test_progress(true);
        let written = Written::default();
        let log_out = written.clone();
        let erase_out = written.clone();
        let line = Arc::clone(&progress.line);
        let writer = move || {
            WaitLogWriter::with_line(
                log_out.clone(),
                &mut erase_out.clone(),
                true,
                Arc::clone(&line),
            )
        };
        let builder = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_ansi(false);
        let emit = || {
            progress.draw_into(&mut written.clone(), "first");
            tracing::warn!("retry warning");
            tracing::info!("another log during the wait");
            assert!(!progress.line.drawn_by(progress.id));
            progress.draw_into(&mut written.clone(), "next poll");
            tracing::warn!("failing the wait");
        };
        if json {
            tracing::subscriber::with_default(builder.json().flatten_event(true).finish(), emit);
        } else {
            tracing::subscriber::with_default(builder.finish(), emit);
        }
        drop(progress);

        let output = written.text()?;
        assert_eq!(output.matches("\x1b[1A\x1b[2K").count(), 2, "{output}");
        assert!(output.starts_with("first\n\x1b[1A\x1b[2K"), "{output}");
        let retry = output
            .find("retry warning")
            .context("missing retry warning")?;
        let other = output
            .find("another log during the wait")
            .context("missing log")?;
        let next = output.find("next poll\n").context("missing next frame")?;
        let failure = output.find("failing the wait").context("missing failure")?;
        assert!(retry < other && other < next && next < failure, "{output}");
        assert!(output.ends_with('\n'), "drop erased the warning: {output}");
        if json {
            output
                .lines()
                .filter(|line| line.contains("\"message\""))
                .try_for_each(|line| {
                    let event: serde_json::Value =
                        serde_json::from_str(line.trim_start_matches("\x1b[1A\x1b[2K"))?;
                    assert!(event["message"].is_string(), "{event}");
                    Ok::<(), serde_json::Error>(())
                })?;
        }
        Ok(())
    }

    #[test]
    fn a_non_terminal_log_does_not_emit_cursor_controls() -> anyhow::Result<()> {
        let progress = test_progress(false);
        let written = Written::default();
        let log_out = written.clone();
        let erase_out = written.clone();
        let line = Arc::clone(&progress.line);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || {
                WaitLogWriter::with_line(
                    log_out.clone(),
                    &mut erase_out.clone(),
                    false,
                    Arc::clone(&line),
                )
            })
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            progress.report(&readiness(true));
            tracing::warn!("retry warning");
        });
        let output = written.text()?;
        assert!(output.contains("readiness poll"), "{output}");
        assert!(output.contains("retry warning"), "{output}");
        assert!(!output.contains("\x1b["), "{output}");
        Ok(())
    }

    #[test]
    fn a_status_does_not_draw_over_a_log_from_another_thread() -> anyhow::Result<()> {
        let progress = test_progress(true);
        let written = Written::default();
        progress.draw_into(&mut written.clone(), "first");
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (continue_tx, continue_rx) = std::sync::mpsc::channel();
        let line = Arc::clone(&progress.line);
        let log_out = written.clone();
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let log = scope.spawn(move || -> anyhow::Result<()> {
                let mut writer =
                    WaitLogWriter::with_line(log_out.clone(), &mut log_out.clone(), true, line);
                ready_tx.send(())?;
                continue_rx.recv()?;
                writeln!(writer, "warning")?;
                Ok(())
            });
            ready_rx.recv()?;
            progress.draw_into(&mut written.clone(), "hidden status");
            continue_tx.send(())?;
            log.join()
                .map_err(|_| anyhow::anyhow!("log thread panicked"))??;
            Ok(())
        })?;
        progress.draw_into(&mut written.clone(), "next poll");
        let output = written.text()?;
        assert_eq!(output, "first\n\x1b[1A\x1b[2Kwarning\nnext poll\n");
        Ok(())
    }

    /// Counted in characters, so a `°` does not make the cut fall short.
    #[rstest]
    #[case::short("waiting 0:20 / 5:00 (ok)", "waiting 0:20 / 5:00 (ok)")]
    #[case::exact(&"x".repeat(80), &"x".repeat(80))]
    #[case::long(&"x".repeat(81), &format!("{}…", "x".repeat(79)))]
    #[case::wide(&"°".repeat(81), &format!("{}…", "°".repeat(79)))]
    fn a_frame_never_wraps(#[case] line: &str, #[case] expected: &str) {
        assert_eq!(fit(line), expected);
        assert!(fit(line).chars().count() <= MAX_FRAME_WIDTH);
    }
}
