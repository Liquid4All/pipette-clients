//! `/proc/<pid>` footprint sampling for Linux and Android.
//!
//! `VmHWM` (equivalently `wait4`'s `ru_maxrss`) is a peak *resident* set, which on
//! a device with swap is not the peak the run required: the kernel can compress a
//! page into zram, dropping it out of RSS while the process still owns and needs
//! it. Both terms are kept because neither subsumes the other. `VmHWM` is exact,
//! maintained on the fault path so no interval can step over a transient, but
//! blind to swap; the sampled `VmRSS + VmSwap` sum sees swap but has to land near
//! the peak. There is no kernel watermark for swap, which is why this samples at
//! all. Measurements and rationale: docs/methodology/peak-memory-android.md.

/// One `/proc/<pid>` observation.
#[derive(Clone, Copy, Default)]
struct StatusSample {
    hwm_kib: u64,
    rss_kib: u64,
    swap_kib: u64,
    major_faults: u64,
    /// Process start time, used only to notice that the pid was recycled under
    /// the sampler. Never reported.
    start_time: u64,
}

/// `VmHWM` / `VmRSS` / `VmSwap` in one pass, plus `majflt`. A field the kernel
/// omits reads 0 — `VmSwap` is absent on a swapless kernel, where "nothing
/// swapped" is the correct answer rather than a failure.
fn read_status_sample(pid: u32) -> Option<StatusSample> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let (major_faults, start_time) = read_stat_fields(pid).unwrap_or((0, 0));
    let seed = StatusSample {
        major_faults,
        start_time,
        ..StatusSample::default()
    };
    Some(status.lines().filter_map(|line| line.split_once(':')).fold(
        seed,
        |mut sample, (field, rest)| {
            let Some(kib) = rest.split_whitespace().next().and_then(|v| v.parse().ok()) else {
                return sample;
            };
            match field {
                "VmHWM" => sample.hwm_kib = kib,
                "VmRSS" => sample.rss_kib = kib,
                "VmSwap" => sample.swap_kib = kib,
                _ => {}
            }
            sample
        },
    ))
}

/// `(majflt, starttime)` for `pid`.
///
/// Major faults are pages the run had to pull from storage or decompress out of
/// zram: near zero when it met no pressure, six figures when it thrashed, so they
/// witness a suppressed peak for free. `starttime` only identifies the process.
fn read_stat_fields(pid: u32) -> Option<(u64, u64)> {
    parse_stat_fields(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// `(majflt, starttime)` from `/proc/<pid>/stat`.
///
/// `comm` is parenthesized and may itself contain spaces and parens, so the
/// positional fields start after the LAST ')' — at field 3 (state), putting
/// majflt (field 12) at index 9 and starttime (field 22) nine further on.
fn parse_stat_fields(stat: &str) -> Option<(u64, u64)> {
    let after_comm = stat.rsplit_once(')')?.1;
    let mut fields = after_comm.split_whitespace().skip(9);
    let major_faults = fields.next()?.parse().ok()?;
    let start_time = fields.nth(9)?.parse().ok()?;
    Some((major_faults, start_time))
}

/// What a run required at peak, kept decomposed so a peak suppressed by reclaim
/// is visible rather than silently low.
#[derive(Clone, Copy, Default)]
pub(crate) struct Footprint {
    /// Kernel peak-RSS watermark — exact, but blind to anything in swap.
    pub(crate) peak_rss_kib: u64,
    /// Largest `VmRSS + VmSwap` sampled. The kernel keeps no watermark for
    /// swap, so a sampled maximum is the only way to count what zram holds.
    pub(crate) peak_committed_kib: u64,
    /// How much of the process zram held, and how hard it thrashed to get it
    /// back. Diagnostics: they explain a low `peak_rss_kib`, never replace it.
    pub(crate) max_swap_kib: u64,
    pub(crate) major_faults: u64,
    /// How many reads landed. `1` means only the synchronous seed did, taken
    /// before the child had loaded anything, so the figures describe process
    /// startup rather than the run: a few MB, non-zero, and indistinguishable
    /// from a real measurement once submitted. Consumers that would rather report
    /// nothing than that check this.
    pub(crate) samples: u32,
}

impl Footprint {
    fn observe(&mut self, sample: StatusSample) {
        self.samples = self.samples.saturating_add(1);
        self.peak_rss_kib = self.peak_rss_kib.max(sample.hwm_kib);
        self.peak_committed_kib = self
            .peak_committed_kib
            .max(sample.rss_kib.saturating_add(sample.swap_kib));
        self.max_swap_kib = self.max_swap_kib.max(sample.swap_kib);
        self.major_faults = self.major_faults.max(sample.major_faults);
    }

    /// Peak RAM the run required, wherever those pages sat when the peak was
    /// struck.
    pub(crate) fn peak_ram_kib(&self) -> u64 {
        self.peak_rss_kib.max(self.peak_committed_kib)
    }

    /// The swap term in bytes, for the run's `observation_max_swap_bytes`.
    ///
    /// Contained in [`Self::peak_ram_kib`] rather than additional to it, so a
    /// consumer must never sum the two. Zero is a real reading: the sampler
    /// looked and the run stayed resident.
    pub(crate) fn max_swap_bytes(&self) -> u64 {
        self.max_swap_kib.saturating_mul(1024)
    }
}

/// The peak RAM a run required, refusing a figure the load makes impossible.
///
/// A sampler that never landed a reading gives 0, which must never go on the
/// wire (parity with the macOS `phys_footprint` guard and the Linux arm's
/// `VmHWM == 0` bail).
///
/// Only the Android arm calls this today; it lives in this linux+android module
/// rather than in `max_memory_usage::android` so its tests run on a target CI
/// actually executes tests for.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub(crate) fn peak_ram_bytes(
    footprint: &Footprint,
    model_path: &std::path::Path,
    flags: &pipette_plan_types::RuntimeFlags,
) -> anyhow::Result<u64> {
    use anyhow::Context;

    let peak_kib = footprint.peak_ram_kib();
    anyhow::ensure!(
        peak_kib > 0,
        "footprint sampler captured no readings; max_host_bytes would be meaningless"
    );
    let peak_bytes = peak_kib.saturating_mul(1024);

    // With the weights pinned in anonymous RAM every tensor byte is written, so
    // the file's size is a hard floor on what the run required. A figure below
    // it means the peak was suppressed (zram) or the run did less than asked —
    // either way it is not a measurement, and a submitted row cannot be undone.
    if weights_pinned_in_ram(flags) {
        // The run just loaded this file, so a stat failure here is not a missing
        // model: it is the guard losing its reference value, and skipping the
        // check silently is how an invalid row would slip through.
        let floor = std::fs::metadata(model_path)
            .with_context(|| {
                format!(
                    "cannot size {} to bound the peak against",
                    model_path.display()
                )
            })?
            .len();
        anyhow::ensure!(
            peak_bytes >= floor,
            "peak RAM {peak_bytes} B is below the {floor} B model file, which a \
             completed no-mmap load cannot be (peak_rss {} KiB, max_swap {} KiB, \
             major_faults {}); refusing to submit",
            footprint.peak_rss_kib,
            footprint.max_swap_kib,
            footprint.major_faults,
        );
    }
    Ok(peak_bytes)
}

/// Whether the cell loads weights into anonymous RAM, where the model file size
/// is a floor on the footprint. A mapped load keeps tensors in reclaimable file
/// pages, so no such floor holds there.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn weights_pinned_in_ram(flags: &pipette_plan_types::RuntimeFlags) -> bool {
    pipette_plan_types::RuntimeFlagRef::from(flags.clone()).mmap == Some(false)
}

/// Whether a sample came from the process the poller was pointed at.
///
/// A zero `start_time` means the field could not be read, not that the pid was
/// recycled. Treating it as a mismatch would discard every later sample and
/// leave the footprint stuck at the seed, failing an otherwise healthy run.
fn same_process(identity: Option<u64>, sample_start_time: u64) -> bool {
    match (identity, sample_start_time) {
        (Some(seen), observed) if observed != 0 => seen == observed,
        _ => true,
    }
}

pub(crate) struct FootprintPoller {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// `Option` only so [`Drop`] can coexist with a `stop_and_join` that moves
    /// the handle out to join it.
    handle: Option<std::thread::JoinHandle<Footprint>>,
}

impl Drop for FootprintPoller {
    /// Setting the flag is the only thing that ends the sampling thread, and
    /// `stop_and_join` consumes `self`, so this runs exactly on the paths that
    /// discard the observation: a run that failed before reporting, or a
    /// `llama-server` replaced mid-eval after a crash. Without it each of those
    /// leaves a thread waking forever to read the `/proc` entry of a dead pid.
    ///
    /// Deliberately does not join. This runs on error paths, and blocking one
    /// for up to a full sample interval to collect a footprint nobody will read
    /// is not worth it; the thread observes the flag and exits on its own.
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// How often the sampler reads `/proc/<pid>` when the figure it produces is the
/// benchmark's own result.
///
/// This bounds the only lossy term. `VmHWM` is exact, but when swap suppresses it
/// the reported figure falls back to the sampled `VmRSS + VmSwap` sum, which can
/// only under-read: the miss is roughly the footprint's growth rate times this
/// interval. A no-mmap load faults in ~900 MB/s, so 10 ms costs about 9 MB of
/// resolution where 25 ms cost ~22 MB. Two small reads per tick at 100 Hz is
/// ~1% of one core, which the memory benchmark can afford because it reports
/// bytes, not time. A benchmark that reports time must not pay it: see
/// [`OBSERVATION_INTERVAL`].
const SAMPLE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);

/// How often to read `/proc/<pid>` when the figure is only an observation
/// riding along on a benchmark that measures something else.
///
/// A timing benchmark's number must not move because we watched it, so this
/// costs a tenth of [`SAMPLE_INTERVAL`]'s duty cycle (~0.1% of one core). Not
/// rarer, because `VmHWM` has to be read while the process is alive: the last
/// useful read precedes exit, and that miss window grows in proportion while the
/// saving past ~100 ms does not. What each term gives up is worked through in
/// docs/methodology/peak-memory-usage.md.
const OBSERVATION_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Sample `pid`'s footprint until stopped, at the resolution the memory
/// benchmark's own metric needs.
pub(crate) fn spawn_footprint_poller(pid: u32) -> FootprintPoller {
    spawn_footprint_poller_every(pid, SAMPLE_INTERVAL)
}

/// Sample `pid`'s footprint until stopped, at the cadence for an observation on
/// a benchmark measuring something else. See [`OBSERVATION_INTERVAL`].
pub(crate) fn spawn_observation_poller(pid: u32) -> FootprintPoller {
    spawn_footprint_poller_every(pid, OBSERVATION_INTERVAL)
}

fn spawn_footprint_poller_every(pid: u32, interval: std::time::Duration) -> FootprintPoller {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    // Seed synchronously so a process that exits before the thread is scheduled
    // still yields a non-zero peak (its /proc entry exists once spawned).
    let seed = read_status_sample(pid);
    // Callers that reap before stopping leave a window where this pid can be
    // recycled. Pinning the process start time closes it: a different process
    // wearing the same pid is skipped rather than folded into the peak, which
    // matters more now that a sample can raise the reported figure.
    let identity = seed.and_then(|sample| (sample.start_time != 0).then_some(sample.start_time));
    let stop = Arc::new(AtomicBool::new(false));
    let handle = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            // Accumulated in thread-local state and handed back through `join`:
            // one writer, so the sampling path needs neither atomics nor a lock.
            let mut footprint = Footprint::default();
            if let Some(sample) = seed {
                footprint.observe(sample);
            }
            // Reads after the child is reaped return `None` (its `/proc`
            // entry is gone); the only gap is pid reuse in the tiny window
            // between a caller's reap and `stop` — callers that can should
            // stop the poller BEFORE reaping (the VL runner does); for the
            // wait_with_output-style callers the window is accepted as not
            // worth guarding.
            //
            // Sleeps before reading, not after. The seed already covers t=0, so a
            // read-first loop only duplicated it, and sleeping first makes
            // `samples > 1` mean "something was read after a full interval
            // elapsed" — the distinction `Footprint::samples` exists to carry.
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(interval);
                if let Some(sample) = read_status_sample(pid) {
                    if same_process(identity, sample.start_time) {
                        footprint.observe(sample);
                    }
                }
            }
            footprint
        })
    };
    FootprintPoller {
        stop,
        handle: Some(handle),
    }
}

impl FootprintPoller {
    /// Stop the poller and return what it saw. A sampler that panicked yields
    /// the default, whose zero peak the callers reject rather than submit.
    pub(crate) fn stop_and_join(mut self) -> Footprint {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.handle
            .take()
            .map(|handle| handle.join().unwrap_or_default())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A poller dropped without `stop_and_join` must still end its thread.
    /// Nothing but the flag stops it, and the observer is dropped un-finished on
    /// every failed run and every `llama-server` replaced mid-eval, so a missing
    /// `Drop` leaks one thread per occurrence, each waking forever to read a dead
    /// pid. Asserts the thread actually observes the flag, not merely that it was
    /// set: the thread's own `Arc` clone is released only as it exits.
    #[test]
    fn dropping_a_poller_ends_its_thread() -> anyhow::Result<()> {
        let child = crate::run_memory::SleepChild::spawn("30")?;
        let poller = spawn_footprint_poller_every(child.id(), std::time::Duration::from_millis(5));
        let stop = std::sync::Arc::clone(&poller.stop);
        drop(poller);

        let exited = (0..200).any(|_| {
            if std::sync::Arc::strong_count(&stop) == 1 {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            false
        });

        assert!(
            stop.load(std::sync::atomic::Ordering::Relaxed),
            "drop must signal the sampler to stop"
        );
        assert!(exited, "the sampling thread outlived the dropped poller");
        Ok(())
    }

    /// `comm` is attacker-shaped: it can hold spaces and parens, so both fields
    /// have to be located from the last ')' rather than by absolute token index.
    #[test]
    fn parses_stat_fields_past_a_hostile_comm() {
        let stat = concat!(
            "42 (llama )bench 1) S 1 42 42 0 -1 4194304 1811767 0 13913 0 ",
            "55 3 0 0 20 0 4 0 987654 1",
        );
        assert_eq!(parse_stat_fields(stat), Some((13_913, 987_654)));
    }

    fn flags(mmap: Option<bool>) -> pipette_plan_types::RuntimeFlags {
        pipette_plan_types::RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText {
            threads: Some(4),
            number_gpu_layers: Some(0),
            mmap,
            flash_attention: Some(pipette_plan_types::LlamacppFlashAttention::Off),
            raw: vec![],
        }
    }

    /// The floor only holds when the weights are anonymous.
    #[rstest::rstest]
    #[case::no_mmap(flags(Some(false)), true)]
    #[case::mapped(flags(Some(true)), false)]
    #[case::unset(flags(None), false)]
    fn floor_applies_only_to_anonymous_loads(
        #[case] flags: pipette_plan_types::RuntimeFlags,
        #[case] expected: bool,
    ) {
        assert_eq!(weights_pinned_in_ram(&flags), expected);
    }

    #[test]
    fn a_sampler_with_no_readings_is_refused() {
        let result = peak_ram_bytes(
            &Footprint::default(),
            std::path::Path::new("/nonexistent"),
            &flags(None),
        );
        assert!(result.is_err(), "a zero peak must not reach the wire");
        if let Err(err) = result {
            assert!(err.to_string().contains("no readings"), "{err}");
        }
    }

    /// A no-mmap peak under the model file cannot have come from a completed
    /// load, so it is refused rather than submitted.
    #[test]
    fn a_peak_below_the_model_file_is_refused() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let model = dir.path().join("model.gguf");
        std::fs::write(&model, vec![0u8; 4096])?;

        let mut under = Footprint::default();
        under.observe(sample(2, 2, 0));
        let result = peak_ram_bytes(&under, &model, &flags(Some(false)));
        assert!(
            result.is_err(),
            "2 KiB peak under a 4 KiB model must be refused"
        );

        // Mapped loads keep tensors in reclaimable file pages, so the same
        // figure is legitimate there and must pass.
        assert_eq!(peak_ram_bytes(&under, &model, &flags(Some(true)))?, 2048);
        Ok(())
    }

    /// The floor needs the model's size, so losing it is a failure to check
    /// rather than a pass.
    #[test]
    fn an_unstattable_model_fails_the_floor_instead_of_skipping_it() {
        let mut f = Footprint::default();
        f.observe(sample(8, 8, 0));
        let result = peak_ram_bytes(
            &f,
            std::path::Path::new("/nonexistent/model.gguf"),
            &flags(Some(false)),
        );
        assert!(
            result.is_err(),
            "an unsizable model must not pass the floor"
        );
    }

    /// Covers the thread, the synchronous seed, and the handoff through `join`:
    /// the sampler must report a live process rather than the default.
    #[test]
    fn the_poller_reports_the_process_it_was_pointed_at() -> anyhow::Result<()> {
        let child = crate::run_memory::SleepChild::spawn("2")?;
        let poller = spawn_footprint_poller(child.id());
        std::thread::sleep(std::time::Duration::from_millis(120));
        let footprint = poller.stop_and_join();

        assert!(
            footprint.peak_rss_kib > 0,
            "a live process must yield a non-zero peak"
        );
        assert!(footprint.peak_ram_kib() >= footprint.peak_rss_kib);
        Ok(())
    }

    fn sample(hwm_kib: u64, rss_kib: u64, swap_kib: u64) -> StatusSample {
        StatusSample {
            hwm_kib,
            rss_kib,
            swap_kib,
            major_faults: 0,
            start_time: 0,
        }
    }

    /// A zero `start_time` is an unreadable field, not a recycled pid: filtering
    /// on it would strand the footprint at the seed.
    #[rstest::rstest]
    #[case::identity_unknown(None, 987_654, true)]
    #[case::same_process(Some(987_654), 987_654, true)]
    #[case::pid_recycled(Some(987_654), 111_111, false)]
    #[case::sample_start_time_unreadable(Some(987_654), 0, true)]
    fn identity_guard_admits_only_the_same_process(
        #[case] identity: Option<u64>,
        #[case] sample_start_time: u64,
        #[case] expected: bool,
    ) {
        assert_eq!(same_process(identity, sample_start_time), expected);
    }

    #[rstest::rstest]
    // Zero swap: the watermark is the larger term, so an unpressured run reports
    // exactly what the previous resident-only counter did.
    #[case::unpressured(vec![sample(6_440_484, 6_440_484, 0)], 6_440_484, 0)]
    // Swapped: RSS peaked low because zram held part of the run, so the
    // committed sum carries the honest peak.
    #[case::swapped(vec![sample(6_269_052, 6_100_000, 340_000)], 6_440_000, 340_000)]
    // The two maxima need not coincide: the later sample swaps more while
    // summing to less, so the sum has to be tracked per sample rather than
    // rebuilt from max(rss) + max(swap), which would claim 5_900.
    #[case::sum_tracked_per_sample(
        vec![sample(5_000, 5_000, 0), sample(5_000, 4_000, 900)],
        5_000,
        900
    )]
    fn footprint_reports_whichever_term_is_larger(
        #[case] samples: Vec<StatusSample>,
        #[case] expected_peak_kib: u64,
        #[case] expected_max_swap_kib: u64,
    ) {
        let footprint = samples.iter().fold(Footprint::default(), |mut f, sample| {
            f.observe(*sample);
            f
        });
        assert_eq!(footprint.peak_ram_kib(), expected_peak_kib);
        assert_eq!(footprint.max_swap_kib, expected_max_swap_kib);
    }
}
