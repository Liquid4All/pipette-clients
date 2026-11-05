//! What a readiness wait tells its caller per poll, and the callbacks the
//! caller hands it: status, cancel and sleep.

// A target with no poll loop (iOS, the `other.rs` fallback) reports one status
// and uses nothing else; the type keeps one shape so a caller writes one gate.
#![cfg_attr(
    not(any(
        target_os = "android",
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )),
    allow(
        dead_code,
        reason = "a target with no poll loop reports status and nothing else"
    )
)]

use std::time::Duration;

use pipette_wait::{WaitCallbacks, WaitStatus};

/// Which gate produced a [`ReadinessStatus`]: the board rather than the OS
/// where a board has criteria of its own (mirrors `linux::board::Board`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessPlatform {
    Android,
    /// A Linux host with no board-specific throttling rules.
    Linux,
    /// A Raspberry Pi 5, which gates on the SoC firmware's throttle bits.
    RaspberryPi5,
    /// An NVIDIA Jetson Orin, which also gates on GPU load, kernel throttling
    /// and over-current events.
    JetsonOrin,
    MacOs,
    Windows,
    /// A target with no probes, whose gate checks nothing (`other.rs`).
    Other,
}

impl ReadinessPlatform {
    /// The gate's name in its log line and, for a gate that takes settings,
    /// its `[readiness.<gate>]` table (`other` has none).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Android => "android",
            Self::Linux => "linux",
            Self::RaspberryPi5 => "pi5",
            Self::JetsonOrin => "orin",
            Self::MacOs => "macos",
            Self::Windows => "windows",
            Self::Other => "other",
        }
    }
}

impl std::fmt::Display for ReadinessPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a wait stands at the poll that produced a [`ReadinessStatus`]. The
/// iOS port carries the same three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessPhase {
    /// The gate is holding the rep. The next poll follows in `poll_in`: on the
    /// shared loop, the interval's remainder after the probe's own sampling time,
    /// held to the deadline; Windows sleeps its whole `POLL_INTERVAL`.
    Waiting { poll_in: Duration },
    /// The criteria passed; the last status of a successful wait.
    Proceeding,
    /// The deadline passed; the wait returns
    /// [`crate::ReadinessError::TimedOut`].
    TimedOut,
}

impl ReadinessPhase {
    /// The phase's name in the gate's events and its per-rep record.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Waiting { .. } => "waiting",
            Self::Proceeding => "proceeding",
            Self::TimedOut => "timed_out",
        }
    }
}

/// One poll of a readiness wait. [`Display`](std::fmt::Display) renders the
/// gate's log line from these fields, deriving the closing phrase from
/// [`Self::phase`] so the two cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadinessStatus {
    pub phase: ReadinessPhase,
    pub platform: ReadinessPlatform,
    /// The platform's summary of this poll's readings, the same string
    /// [`crate::ReadinessError::TimedOut`] carries. Free text: the vocabulary is
    /// per platform, so a caller branches on the other fields.
    pub observed: String,
    /// Measured from before the first probe.
    pub elapsed: Duration,
    pub max_wait: Duration,
}

impl ReadinessStatus {
    /// The only phase worth a live line; the other two are its last frame.
    pub fn is_waiting(&self) -> bool {
        matches!(self.phase, ReadinessPhase::Waiting { .. })
    }
}

/// `readiness: <platform> <observed> → proceeding` (or `→ waiting 1.8s`,
/// `→ timed out after 300s`). The iOS port logs the same shape, and logs are
/// grepped for it, so tests pin it.
impl std::fmt::Display for ReadinessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "readiness: {} {} → ", self.platform, self.observed)?;
        match self.phase {
            ReadinessPhase::Waiting { poll_in } => write!(f, "waiting {poll_in:?}"),
            ReadinessPhase::Proceeding => f.write_str("proceeding"),
            ReadinessPhase::TimedOut => write!(f, "timed out after {:?}", self.max_wait),
        }
    }
}

/// When the caller installs no status callback: one `info` event per poll.
impl WaitStatus for ReadinessStatus {
    fn report_default(&self) {
        let poll_in = match self.phase {
            ReadinessPhase::Waiting { poll_in } => Some(poll_in),
            ReadinessPhase::Proceeding | ReadinessPhase::TimedOut => None,
        };
        tracing::info!(
            platform = %self.platform,
            // Quoted in text output: the reading is itself `key=value` pairs.
            observed = self.observed.as_str(),
            phase = %self.phase.as_str(),
            poll_in = poll_in.map(tracing::field::debug),
            elapsed = ?self.elapsed,
            max_wait = ?self.max_wait,
            "readiness poll"
        );
    }
}

/// The caller's side of a readiness wait. Callbacks rather than policy: they
/// describe the process doing the measuring, not how a cell was measured, so
/// none lands in `benchmark_flags`. [`Default`] logs every poll at `info`, never
/// cancels and sleeps on the calling thread.
///
/// - **`with_status`** replaces the gate's status narration; a failed probe has
///   no status.
/// - **`with_poll`** observes every completed probe, even if the probe fails.
/// - **`with_cancel`** is read before the first probe, after each probe, and
///   between the [`pipette_wait::CANCEL_CHECK_INTERVAL`] pieces of each sleep. A
///   cancelled wait returns [`crate::ReadinessError::Cancelled`] with no final
///   status, so a renderer must erase its own last line.
/// - **`with_sleep`** lets a test drive the real loop without spending time.
pub type ReadinessCallbacks<'a> = WaitCallbacks<'a, ReadinessStatus>;

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::proceeding(
        ReadinessPhase::Proceeding,
        ReadinessPlatform::Android,
        "thermal=status:0(nominal),hottest:31°C(ok) cpu=busy:4%(ok)",
        "readiness: android thermal=status:0(nominal),hottest:31°C(ok) cpu=busy:4%(ok) → proceeding"
    )]
    #[case::waiting(
        ReadinessPhase::Waiting { poll_in: Duration::from_millis(1_800) },
        ReadinessPlatform::MacOs,
        "thermal=pressure:1(moderate) die:41C(ok) cpu=busy:0.3cores(ok)",
        "readiness: macos thermal=pressure:1(moderate) die:41C(ok) cpu=busy:0.3cores(ok) → waiting 1.8s"
    )]
    #[case::timed_out(
        ReadinessPhase::TimedOut,
        ReadinessPlatform::Linux,
        "thermal=hottest:95°C(hot) cpu=load:12%(ok)",
        "readiness: linux thermal=hottest:95°C(hot) cpu=load:12%(ok) → timed out after 300s"
    )]
    #[case::the_board_names_itself(
        ReadinessPhase::Proceeding,
        ReadinessPlatform::RaspberryPi5,
        "thermal=hottest:44°C(ok;soft80/hard85) cpu=load:3%(ok) throttle=0x0(ok)",
        "readiness: pi5 thermal=hottest:44°C(ok;soft80/hard85) cpu=load:3%(ok) throttle=0x0(ok) → proceeding"
    )]
    fn the_status_renders_the_gates_log_line(
        #[case] phase: ReadinessPhase,
        #[case] platform: ReadinessPlatform,
        #[case] observed: &str,
        #[case] expected: &str,
    ) {
        let status = ReadinessStatus {
            phase,
            platform,
            observed: observed.to_string(),
            elapsed: Duration::from_secs(12),
            max_wait: Duration::from_secs(300),
        };
        assert_eq!(status.to_string(), expected);
    }

    #[rstest]
    #[case(ReadinessPhase::Waiting { poll_in: Duration::from_secs(3) }, true)]
    #[case(ReadinessPhase::Proceeding, false)]
    #[case(ReadinessPhase::TimedOut, false)]
    fn only_waiting_is_worth_drawing(#[case] phase: ReadinessPhase, #[case] want: bool) {
        let status = ReadinessStatus {
            phase,
            platform: ReadinessPlatform::Linux,
            observed: String::new(),
            elapsed: Duration::ZERO,
            max_wait: Duration::from_secs(300),
        };
        assert_eq!(status.is_waiting(), want);
    }

    /// Pinned: fleet logs are grepped by these words.
    #[rstest]
    #[case(ReadinessPlatform::Android, "android")]
    #[case(ReadinessPlatform::Linux, "linux")]
    #[case(ReadinessPlatform::RaspberryPi5, "pi5")]
    #[case(ReadinessPlatform::JetsonOrin, "orin")]
    #[case(ReadinessPlatform::MacOs, "macos")]
    #[case(ReadinessPlatform::Windows, "windows")]
    #[case(ReadinessPlatform::Other, "other")]
    fn each_gate_keeps_the_word_its_log_line_used(
        #[case] platform: ReadinessPlatform,
        #[case] expected: &str,
    ) {
        assert_eq!(platform.as_str(), expected);
        assert_eq!(platform.to_string(), expected);
    }

    #[test]
    fn a_replaced_callback_leaves_the_others_alone() {
        let slept: RefCell<Vec<Duration>> = RefCell::new(Vec::new());
        let sleep = |d: Duration| slept.borrow_mut().push(d);
        let callbacks = ReadinessCallbacks::new().with_sleep(&sleep);

        let flow = callbacks.sleep_cancellably(Duration::from_millis(7));

        assert_eq!(*slept.borrow(), vec![Duration::from_millis(7)]);
        assert_eq!(
            flow,
            std::ops::ControlFlow::Continue(()),
            "the untouched cancel callback must still never cancel",
        );
    }
}
