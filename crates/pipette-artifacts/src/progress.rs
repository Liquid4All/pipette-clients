//! Byte-level progress for a fetch, for a caller that wants to render one.
//!
//! Reported per **artifact** — one model or one runtime — because that is the
//! unit the store ensures and the only unit this crate knows. A caller fetching
//! several (a cell needs a runtime *and* a model) sums them itself: it is the one
//! that knows how many are coming, and this crate would have to be told.

use std::io::{Read, Write};
use std::sync::Arc;

/// Where a fetch has got to.
///
/// `done_bytes` counts every part of the artifact, so a two-file vision model or
/// an MLX directory reports one rising number rather than restarting per file.
/// `file` names the part currently moving, which is what a reader wants when the
/// total stalls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchProgress<'a> {
    /// The artifact, as a caller would name it on a command line.
    pub artifact: &'a str,
    /// The part currently being written. A single-file artifact names itself.
    pub file: &'a str,
    /// Bytes written for this artifact so far, across every part.
    pub done_bytes: u64,
    /// Bytes this artifact will write in total, where that can be known before
    /// the first one — a repo listing gives it, a bare URL whose response omits
    /// `content-length` does not. `None` means render a rate, not a percentage.
    pub total_bytes: Option<u64>,
}

impl FetchProgress<'_> {
    /// Completed fraction, or `None` when the total is unknown or zero — a
    /// divide the caller would otherwise have to guard at every render.
    pub fn fraction(&self) -> Option<f64> {
        match self.total_bytes {
            Some(total) if total > 0 => Some((self.done_bytes as f64 / total as f64).min(1.0)),
            _ => None,
        }
    }
}

/// A caller watching a fetch.
///
/// Invoked on the thread doing the writing, once per write — often, and cheaply.
/// Throttling belongs to the sink: the download loop cannot know what a renderer
/// costs, and a sink that only redraws on a timer is one line of state.
pub type ProgressSink = Arc<dyn for<'a> Fn(FetchProgress<'a>) + Send + Sync>;

/// Accumulates one artifact's byte count and forwards it to the sink.
///
/// Held by the fetch loop across the artifact's parts, so `done_bytes` keeps
/// rising as files complete. A fetch with no sink installed carries one of these
/// too — the `Option` is checked once per write, which is cheaper than threading
/// a branch through every call site.
pub(crate) struct Reporter {
    sink: Option<ProgressSink>,
    artifact: String,
    total_bytes: Option<u64>,
    done_bytes: u64,
}

impl Reporter {
    pub(crate) fn new(
        sink: Option<ProgressSink>,
        artifact: String,
        total_bytes: Option<u64>,
    ) -> Self {
        Self {
            sink,
            artifact,
            total_bytes,
            done_bytes: 0,
        }
    }

    /// A reporter nobody is listening to — for a caller that fetches without
    /// having been asked to narrate it.
    #[cfg(test)]
    pub(crate) fn silent() -> Self {
        Self::new(None, String::new(), None)
    }

    /// Add `n` bytes written for `file` and tell the sink.
    pub(crate) fn advance(&mut self, file: &str, n: u64) {
        self.done_bytes = self.done_bytes.saturating_add(n);
        if let Some(sink) = &self.sink {
            sink(FetchProgress {
                artifact: &self.artifact,
                file,
                done_bytes: self.done_bytes,
                total_bytes: self.total_bytes,
            });
        }
    }

    /// Adopt a total the transfer discovered, for an artifact whose size could not
    /// be established beforehand — a response's `content-length`, say. Only ever
    /// widens `None` into a number: a single file's length is not an artifact's
    /// total once there is more than one part, and the first caller to know the
    /// real figure should win.
    pub(crate) fn set_total_if_unknown(&mut self, total: Option<u64>) {
        if self.total_bytes.is_none() {
            self.total_bytes = total;
        }
    }
}

/// `std::io::copy`, counting what it moves.
///
/// Hand-rolled because `io::copy` reports only the total it returns, and a caller
/// watching a multi-GB download needs to hear before the end. The buffer matches
/// `io::copy`'s own, so throughput does not change with a sink installed.
pub(crate) fn copy_reporting(
    reader: &mut impl Read,
    writer: &mut impl Write,
    file: &str,
    reporter: &mut Reporter,
) -> std::io::Result<()> {
    let mut buf = vec![0_u8; 8 * 1024];
    loop {
        let read = match reader.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        writer.write_all(&buf[..read])?;
        reporter.advance(file, read as u64);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn fraction_is_none_without_a_usable_total() {
        let progress = |total| FetchProgress {
            artifact: "m",
            file: "f",
            done_bytes: 10,
            total_bytes: total,
        };
        assert_eq!(progress(None).fraction(), None);
        // A zero total is a listing that answered but had nothing to say; dividing
        // by it would report `inf` rather than "unknown".
        assert_eq!(progress(Some(0)).fraction(), None);
        assert_eq!(progress(Some(40)).fraction(), Some(0.25));
    }

    /// Clamped, so a server that sends more than it promised cannot render past
    /// the end of a bar.
    #[test]
    fn fraction_clamps_when_more_arrives_than_promised() {
        let progress = FetchProgress {
            artifact: "m",
            file: "f",
            done_bytes: 50,
            total_bytes: Some(40),
        };
        assert_eq!(progress.fraction(), Some(1.0));
    }

    /// The invariant the whole feature rests on: a reader trusts the number because
    /// it is the number of bytes that landed, not an estimate beside them.
    #[test]
    fn copy_reports_exactly_what_it_wrote() -> anyhow::Result<()> {
        let reported: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
        let seen = Arc::clone(&reported);
        let sink: ProgressSink = Arc::new(move |p: FetchProgress<'_>| {
            if let Ok(mut seen) = seen.lock() {
                *seen = p.done_bytes;
            }
        });
        let mut reporter = Reporter::new(Some(sink), "a".to_owned(), None);

        // Longer than one 8 KiB chunk, so the loop iterates rather than reading once.
        let source = vec![7_u8; 20_000];
        let mut written = Vec::new();
        copy_reporting(&mut source.as_slice(), &mut written, "f", &mut reporter)?;

        assert_eq!(written.len(), source.len());
        let reported = *reported.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(reported, written.len() as u64);
        Ok(())
    }

    /// The count rises across an artifact's parts — a vision model's second file
    /// continues the first rather than restarting.
    #[test]
    fn advance_accumulates_across_files() -> anyhow::Result<()> {
        let seen: Arc<Mutex<Vec<(String, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&seen);
        let sink: ProgressSink = Arc::new(move |p: FetchProgress<'_>| {
            // A sink cannot report a failure, so a poisoned lock drops the sample
            // rather than taking the download down with it.
            if let Ok(mut seen) = captured.lock() {
                seen.push((p.file.to_owned(), p.done_bytes));
            }
        });

        let mut reporter = Reporter::new(Some(sink), "org/repo".to_owned(), Some(300));
        reporter.advance("weights.gguf", 100);
        reporter.advance("weights.gguf", 100);
        reporter.advance("mmproj.gguf", 50);

        let seen = seen.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(
            *seen,
            vec![
                ("weights.gguf".to_owned(), 100),
                ("weights.gguf".to_owned(), 200),
                ("mmproj.gguf".to_owned(), 250),
            ]
        );
        Ok(())
    }
}
