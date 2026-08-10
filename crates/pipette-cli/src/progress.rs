//! Rendering a fetch's progress on stderr.
//!
//! Two levels, because a cell waits on two artifacts and a reader wants both
//! answers: *how long until this file lands* and *how long until the cell runs*.
//! `pipette-artifacts` reports one artifact at a time — it ensures one at a time
//! and knows nothing of cells — so the cell total is assembled here, from the
//! sizes the caller declares before the first byte.
//!
//! stdout stays clean: it carries results, and a progress line that scrolled into
//! a redirected file would corrupt them.

use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pipette_artifacts::{FetchProgress, ProgressSink};

/// Redraw interval. Fast enough to look live, slow enough that a 3 GB download
/// costs a few hundred writes rather than four hundred thousand.
const TTY_REDRAW: Duration = Duration::from_millis(100);
/// A log-scraped run gets one line at this interval instead — dense enough to
/// show a stall, sparse enough not to bury the run's own output.
const LOG_REDRAW: Duration = Duration::from_secs(15);

/// Elapsed time a rate needs before it means anything. One 8 KiB buffer over the
/// microseconds of a first read divides out to gigabytes per second.
const RATE_FLOOR: f64 = 0.25;

/// Renders a cell's fetches: a line per artifact, and one summing them.
///
/// Cheap to hold when nothing is fetching — a cache hit reports no bytes, so
/// nothing is ever drawn.
pub struct CellProgress {
    state: Arc<Mutex<State>>,
    tty: bool,
    /// A cell waiting on one artifact has nothing to sum: the cell line and the
    /// artifact line would carry the same numbers, so only one is drawn.
    composite: bool,
}

struct State {
    /// Bytes fetched per artifact so far, keyed by the artifact's own name.
    done: Vec<(String, u64)>,
    /// What the whole cell will fetch, when every part could be sized.
    cell_total: Option<u64>,
    last_draw: Option<Instant>,
    /// When the first byte landed. Set then rather than at construction: the size
    /// lookups run in between, and counting that wait made the first rate read a
    /// fraction of the real one — at exactly the moment someone is watching.
    started: Option<Instant>,
    /// Lines the last draw left on screen, to erase before the next one.
    drawn_lines: usize,
}

impl CellProgress {
    /// A renderer for a cell that will fetch artifacts of the given sizes — one
    /// entry each, `Some(0)` for one already in the store and `None` for one whose
    /// size could not be established.
    ///
    /// The cell total is the sum, and only a sum: one unsized artifact makes the
    /// cell-level percentage unknowable, and guessing it would be worse than
    /// leaving it off.
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

    /// The sink to install on an [`pipette_artifacts::ArtifactsContext`].
    pub fn sink(&self) -> ProgressSink {
        let state = Arc::clone(&self.state);
        let tty = self.tty;
        let composite = self.composite;
        Arc::new(move |p: FetchProgress<'_>| {
            let mut state = match state.lock() {
                Ok(state) => state,
                // A poisoned lock means a previous render panicked. A fetch is not
                // worth failing over a cosmetic, so stop drawing and let it run.
                Err(_) => return,
            };
            state.record(p.artifact, p.done_bytes);
            let interval = if tty { TTY_REDRAW } else { LOG_REDRAW };
            if state.due(interval) {
                state.draw(&p, tty, composite);
            }
        })
    }

    /// What this cell will fetch in total, once every part could be sized.
    #[cfg(test)]
    fn total(&self) -> anyhow::Result<Option<u64>> {
        Ok(self
            .state
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .cell_total)
    }
}

/// Clears the drawn lines when the fetches go out of scope, however they ended — a
/// failed fetch would otherwise leave its last line above the error. On a log the
/// lines have already scrolled, so there is nothing to erase.
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

    fn draw(&mut self, current: &FetchProgress<'_>, tty: bool, composite: bool) {
        let mut out = std::io::stderr().lock();
        if tty {
            self.erase_into(&mut out);
        }
        let elapsed = self
            .started
            .map(|start| start.elapsed().as_secs_f64())
            .unwrap_or_default();
        // Below a sample worth dividing by, the first frame reads whatever one
        // buffer over a few microseconds comes to — `1.5 GB/s` on a 20 MB/s link.
        // Blank until the window is real.
        let rate = if elapsed >= RATE_FLOOR {
            rate_of(self.cell_done() as f64 / elapsed)
        } else {
            String::new()
        };
        let cell = format!(
            "fetching {} {rate}",
            bytes_of(self.cell_done(), self.cell_total)
        );
        let artifact = format!("  {}", name_of(current));
        // A log line has to stand alone: without the cursor to rewrite, two lines
        // per redraw would double the noise for no extra fact. A single artifact
        // has no sum to show, so its own line carries the rate instead.
        let lines: Vec<String> = match (composite, tty) {
            (false, _) => vec![format!("fetching {} {rate}", name_of(current))],
            (true, true) => vec![cell, artifact],
            (true, false) => vec![format!("{cell}: {}", artifact.trim_start())],
        };
        // Trimmed because the rate is blank for the first quarter-second, and a
        // trailing space is still a character the erase has to cover.
        let rendered: Vec<&str> = lines.iter().map(|line| line.trim_end()).collect();
        let _ = writeln!(out, "{}", rendered.join("\n"));
        self.drawn_lines = if tty { lines.len() } else { 0 };
        let _ = out.flush();
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

/// The artifact and the part of it currently moving, with the file dropped when
/// the artifact's own name already ends in it — a single-file model names itself,
/// and repeating that says nothing twice.
fn name_of(current: &FetchProgress<'_>) -> String {
    let bytes = bytes_of(current.done_bytes, current.total_bytes);
    if current.artifact.ends_with(current.file) {
        format!("{} {bytes}", current.artifact)
    } else {
        format!("{} {} {bytes}", current.artifact, current.file)
    }
}

/// `1.4/2.9 GB`, or `48 MB` when the total is unknown.
///
/// Both halves render in one unit, chosen from the larger, so the pair reads as a
/// single quantity — a 48 MB archive against a GB scale would show `0.0/0.1 GB`
/// and look stalled for its whole download.
fn bytes_of(done: u64, total: Option<u64>) -> String {
    match total {
        Some(total) if total > 0 => {
            let (scale, unit) = unit_for(total);
            format!(
                "{:.1}/{:.1} {unit}",
                done as f64 / scale,
                total as f64 / scale
            )
        }
        _ => {
            let (scale, unit) = unit_for(done);
            format!("{:.1} {unit}", done as f64 / scale)
        }
    }
}

/// Divisor and name for the largest unit `bytes` fills. Decimal, as storage and
/// transfer rates are quoted — a download is not a memory allocation.
fn unit_for(bytes: u64) -> (f64, &'static str) {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;
    match bytes as f64 {
        b if b >= GB => (GB, "GB"),
        b if b >= MB => (MB, "MB"),
        b if b >= KB => (KB, "KB"),
        _ => (1.0, "B"),
    }
}

fn rate_of(bytes_per_sec: f64) -> String {
    let (scale, unit) = unit_for(bytes_per_sec as u64);
    format!("{:.1} {unit}/s", bytes_per_sec / scale)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// The cell total is a sum, and a sum needs every term. One unsized artifact
    /// leaves the cell-level percentage unknown rather than understated.
    #[test]
    fn one_unsized_artifact_leaves_the_cell_total_unknown() -> anyhow::Result<()> {
        let partly_sized = CellProgress::new(&[Some(100), None]);
        let total = partly_sized.total()?;
        assert_eq!(total, None);

        let sized = CellProgress::new(&[Some(100), Some(250)]);
        assert_eq!(sized.total()?, Some(350));
        Ok(())
    }

    /// An artifact reports its own running total, so the cell sums the latest from
    /// each rather than every callback it ever saw.
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

    /// Redraws are rate-limited: a multi-GB fetch calls the sink hundreds of
    /// thousands of times, and drawing each one would cost more than the download.
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

    /// One unit for the pair, chosen from the larger: a 48 MB archive on a GB
    /// scale reads `0.0/0.1 GB` and looks stalled for its whole download.
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

    /// A single-file artifact already ends in its file name, and saying it twice
    /// is the kind of line a reader stops reading.
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
}
