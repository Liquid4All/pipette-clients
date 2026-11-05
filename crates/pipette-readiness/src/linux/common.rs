//! Linux readiness probes shared by every board: thermal zones, CPU load, and
//! the summary's CPU fragment.
//!
//! An unconfined process reads the 1-minute load average over
//! `available_parallelism`: it needs no sampling window and its smoothing
//! ignores sub-second housekeeping. A process confined to some of a shared
//! host's CPUs (a SLURM cpuset, `taskset`, `docker --cpuset-cpus`) cannot:
//! `/proc/loadavg` counts the whole host's runnable tasks, so neighbors' work
//! reads as load here. Measured 2026-10-11 in a 2-CPU job on a shared 192-CPU SLURM worker:
//! load average 9.87 while the job's own CPUs were 0% busy, which over a 16-CPU
//! job reads 62%, above [`CPU_LOAD_THRESHOLD`], so every wait ran to its
//! deadline. A confined process therefore reads the busy share of its own CPUs
//! from `/proc/stat` over [`CPU_SAMPLE_WINDOW`].
//!
//! Confinement comes from the affinity mask alone; a `cpu.max` quota is not
//! confinement. The mask must number CPUs as `/proc/stat` does. Under lxcfs
//! `/proc` renumbers a container's CPUs from 0 while the mask keeps host ids,
//! so a mask only partly in the online list is not sampled and the poll falls
//! back to the load average (see [`cpu_load`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::ReadinessError;

/// Default exclusive bound on either [`CpuLoad`] reading when the host's table
/// sets none: the CPUs this process may use are under 30% busy.
pub(super) const CPU_LOAD_THRESHOLD: f64 = 0.30;

/// Matches the macOS and Android gates. `/proc/stat` counts 10 ms ticks, so a
/// second on a single CPU resolves the share to a percent.
const CPU_SAMPLE_WINDOW: Duration = Duration::from_secs(1);

/// Set once [`cpu_load`]'s fallback has been logged above debug level. Its
/// cause is a property of the host and the gate polls every few seconds, so a
/// line per poll would fill a fleet log with one unchanging line.
static FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);

/// `u16` holds any kernel CPU id (`NR_CPUS` is at most 8192 on x86-64 and
/// arm64) and bounds what a mangled range in a CPU list can expand to.
type CpuSet = BTreeSet<u16>;

const NO_ALLOWED_CPUS: &str = "no Cpus_allowed_list in /proc/self/status";

const NO_ALLOWED_CPU_ONLINE: &str = "no allowed CPU is among the online CPUs";

const ALLOWED_CPUS_PARTLY_ONLINE: &str =
    "some allowed CPUs are not online and some online CPUs are not allowed";

pub(super) const NO_THERMAL_ZONES: &str = "no thermal zones";

pub(super) const NO_READABLE_THERMAL_ZONES: &str = "thermal zones present but none readable";

/// Hottest `/sys/class/thermal/thermal_zone*` reading, in °C.
///
/// [`ReadinessError::Io`] if the directory cannot be listed, [`NO_THERMAL_ZONES`]
/// if it holds no zone, [`NO_READABLE_THERMAL_ZONES`] if no zone reads. Zones are
/// counted before any read, so a broken sensor is not reported as a missing one.
pub(super) fn read_hottest_zone_celsius() -> Result<i32, ReadinessError> {
    let dir = fs::read_dir("/sys/class/thermal")?;
    let zones: Vec<PathBuf> = dir
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("thermal_zone"))
        })
        .map(|entry| entry.path())
        .collect();
    if zones.is_empty() {
        return Err(ReadinessError::Unavailable(NO_THERMAL_ZONES));
    }
    zones
        .iter()
        .filter_map(|zone| read_zone_celsius(&zone.join("temp")))
        .max()
        .ok_or(ReadinessError::Unavailable(NO_READABLE_THERMAL_ZONES))
}

fn read_zone_celsius(temp_path: &Path) -> Option<i32> {
    let raw = fs::read_to_string(temp_path).ok()?;
    let millicelsius: i32 = raw.trim().parse().ok()?;
    Some(millicelsius / 1000)
}

/// One poll's CPU load as a share of the CPUs this process may run on. One
/// bound judges both readings, but the summary names which it was: a minute's
/// average and a second's busy share are different evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CpuLoad {
    /// 1-minute load average over the CPUs the process may use.
    LoadAverage(f64),
    /// Busy share of the process's own CPUs over [`CPU_SAMPLE_WINDOW`].
    BusyShare(f64),
}

impl CpuLoad {
    pub(super) fn ratio(self) -> f64 {
        match self {
            Self::LoadAverage(ratio) | Self::BusyShare(ratio) => ratio,
        }
    }
}

/// The CPU load the gate judges: the busy share of this process's own CPUs if
/// it is confined to some of the host's, else the normalized load average.
pub(super) fn read_cpu_load() -> Result<CpuLoad, ReadinessError> {
    cpu_load(read_confined_busy_share(), read_load_average)
}

/// A confinement or busy share that cannot be read (a sandbox that stubs
/// `/proc`) falls back to the load average instead of failing the wait: a
/// missing file says nothing about load.
fn cpu_load(
    confined: Result<Option<f64>, ReadinessError>,
    load_average: impl FnOnce() -> Result<f64, ReadinessError>,
) -> Result<CpuLoad, ReadinessError> {
    match confined {
        Ok(Some(busy)) => return Ok(CpuLoad::BusyShare(busy)),
        Ok(None) => {}
        Err(err) => report_fallback(&err),
    }
    load_average().map(CpuLoad::LoadAverage)
}

/// Above debug level only the first time in a process. A missing
/// `Cpus_allowed_list` is info, not warn: the kernel always writes it, so its
/// absence means a sandbox with its own `/proc` (gVisor). `source` is the file
/// and errno of a failed read, which [`ReadinessError::Io`]'s text leaves out.
fn report_fallback(err: &ReadinessError) {
    let source = std::error::Error::source(err).map(tracing::field::display);
    if FALLBACK_REPORTED.swap(true, Ordering::Relaxed) {
        tracing::debug!(
            error = %err,
            source,
            "the confined CPU reading is unavailable; reading the host-wide load average"
        );
        return;
    }
    if matches!(err, ReadinessError::Unavailable(NO_ALLOWED_CPUS)) {
        tracing::info!(
            error = %err,
            "this host does not report which CPUs this process may run on, so whether it is \
             confined to some of them is unknown; every poll in this process reads the \
             host-wide load average"
        );
        return;
    }
    tracing::warn!(
        error = %err,
        source,
        "cannot read whether this process is confined to some of the host's CPUs, or the busy \
         share of the CPUs it is confined to; every such poll in this process reads the \
         host-wide load average instead, which on a host shared with other jobs counts their \
         work as load here"
    );
}

/// Busy share of this process's CPUs over [`CPU_SAMPLE_WINDOW`] if it is
/// confined to some of the host's, `None` if it may run on all of them.
///
/// Resamples once if the counters did not advance, as the macOS gate does,
/// then reports the reading unavailable rather than guessing a share.
fn read_confined_busy_share() -> Result<Option<f64>, ReadinessError> {
    let (allowed, online) = (read_allowed_cpus()?, read_online_cpus()?);
    confined_busy_share(&allowed, &online, sample_busy_share)
}

/// The sampler is passed in so the resample is testable without waiting two
/// seconds on a real host.
fn confined_busy_share(
    allowed: &CpuSet,
    online: &CpuSet,
    mut sample: impl FnMut(&CpuSet) -> Result<Option<f64>, ReadinessError>,
) -> Result<Option<f64>, ReadinessError> {
    let Some(cpus) = confined_to(allowed, online)? else {
        return Ok(None);
    };
    if let Some(busy) = sample(&cpus)? {
        return Ok(Some(busy));
    }
    tracing::debug!("the /proc/stat counters did not advance; resampling");
    sample(&cpus)?.map(Some).ok_or(ReadinessError::Unavailable(
        "/proc/stat counters of the allowed CPUs did not advance across two windows",
    ))
}

/// A failed read names the file, which `std::io::Error` does not: the line
/// [`report_fallback`] writes is all that says which read failed.
fn read_text(path: &str) -> Result<String, ReadinessError> {
    fs::read_to_string(path)
        .map_err(|err| std::io::Error::new(err.kind(), format!("{path}: {err}")).into())
}

fn read_allowed_cpus() -> Result<CpuSet, ReadinessError> {
    parse_allowed_cpus(&read_text("/proc/self/status")?)
}

fn parse_allowed_cpus(status: &str) -> Result<CpuSet, ReadinessError> {
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or(ReadinessError::Unavailable(NO_ALLOWED_CPUS))?;
    parse_cpu_list(list)
}

fn read_online_cpus() -> Result<CpuSet, ReadinessError> {
    parse_cpu_list(&read_text("/sys/devices/system/cpu/online")?)
}

/// The kernel's CPU-list syntax (`0-7,96-103`), used by both
/// `Cpus_allowed_list` and `/sys/devices/system/cpu/online`.
fn parse_cpu_list(raw: &str) -> Result<CpuSet, ReadinessError> {
    let list = raw.trim();
    let mut cpus = CpuSet::new();
    if list.is_empty() {
        return Ok(cpus);
    }
    let garbage = || ReadinessError::ParseFailure {
        raw: list.to_string(),
    };
    for item in list.split(',') {
        let (first, last) = item.split_once('-').unwrap_or((item, item));
        let first: u16 = first.parse().map_err(|_| garbage())?;
        let last: u16 = last.parse().map_err(|_| garbage())?;
        if first > last {
            return Err(garbage());
        }
        cpus.extend(first..=last);
    }
    Ok(cpus)
}

/// The CPUs to sample, or `None` if the mask holds every online CPU (it may
/// hold more: on a hotplug host it names every possible CPU). Confined means
/// some online CPUs and nothing else, as a cpuset or `taskset` leaves.
///
/// Anything else is an error. The kernel keeps an online CPU in every mask, so
/// a mask with none means the two lists number CPUs differently. A partly
/// online mask may mean the same (or a pin to a CPU since taken offline), and
/// sampling its online part would read some of a renumbered container's CPUs
/// as all of them.
fn confined_to(allowed: &CpuSet, online: &CpuSet) -> Result<Option<CpuSet>, ReadinessError> {
    let usable: CpuSet = allowed.intersection(online).copied().collect();
    if usable.is_empty() {
        return Err(ReadinessError::Unavailable(NO_ALLOWED_CPU_ONLINE));
    }
    if usable.len() == online.len() {
        return Ok(None);
    }
    if usable.len() < allowed.len() {
        return Err(ReadinessError::Unavailable(ALLOWED_CPUS_PARTLY_ONLINE));
    }
    Ok(Some(usable))
}

/// One CPU's `/proc/stat` counters, in ticks since boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CpuTimes {
    /// `idle + iowait`.
    idle: u64,
    /// `user + nice + system + idle + iowait + irq + softirq + steal`. Not
    /// `guest` and `guest_nice`, which the kernel already counts inside `user`
    /// and `nice`.
    total: u64,
}

/// Busy share of `cpus` across one [`CPU_SAMPLE_WINDOW`], or `None` if their
/// counters did not advance inside it.
fn sample_busy_share(cpus: &CpuSet) -> Result<Option<f64>, ReadinessError> {
    let before = read_proc_stat()?;
    std::thread::sleep(CPU_SAMPLE_WINDOW);
    let after = read_proc_stat()?;
    Ok(busy_share(&before, &after, cpus))
}

fn read_proc_stat() -> Result<BTreeMap<u16, CpuTimes>, ReadinessError> {
    parse_proc_stat(&read_text("/proc/stat")?)
}

/// The per-CPU lines of `/proc/stat`, by CPU id. The aggregate `cpu` line is
/// skipped with the rest of the file: it sums over CPUs this process may not
/// run on.
fn parse_proc_stat(raw: &str) -> Result<BTreeMap<u16, CpuTimes>, ReadinessError> {
    let mut cpus = BTreeMap::new();
    for line in raw.lines() {
        let mut fields = line.split_ascii_whitespace();
        let id = fields.next().and_then(|name| name.strip_prefix("cpu"));
        let Some(cpu) = id.and_then(|id| id.parse::<u16>().ok()) else {
            continue;
        };
        let malformed = || ReadinessError::ParseFailure {
            raw: line.to_string(),
        };
        // user, nice, system, idle, iowait, irq, softirq, steal. A line may
        // stop anywhere after `idle`, and what it leaves out counts as zero.
        // `guest` and `guest_nice` follow `steal` and are not read.
        let parsed: Result<Vec<u64>, _> = fields.take(8).map(str::parse).collect();
        let ticks = parsed.map_err(|_| malformed())?;
        let [_user, _nice, _system, idle, rest @ ..] = ticks.as_slice() else {
            return Err(malformed());
        };
        let iowait = rest.first().copied().unwrap_or(0);
        let times = CpuTimes {
            idle: idle.saturating_add(iowait),
            total: ticks.iter().copied().fold(0, u64::saturating_add),
        };
        cpus.insert(cpu, times);
    }
    Ok(cpus)
}

/// Busy share of `cpus` between two snapshots, or `None` if their counters did
/// not advance.
///
/// A CPU missing from either snapshot (it went offline or came online inside
/// the window) is left out. A counter that went backwards counts as not
/// advanced: the kernel documents that `iowait` can decrease.
fn busy_share(
    before: &BTreeMap<u16, CpuTimes>,
    after: &BTreeMap<u16, CpuTimes>,
    cpus: &CpuSet,
) -> Option<f64> {
    let (mut idle, mut total) = (0u64, 0u64);
    for cpu in cpus {
        let (Some(before), Some(after)) = (before.get(cpu), after.get(cpu)) else {
            continue;
        };
        idle = idle.saturating_add(after.idle.saturating_sub(before.idle));
        total = total.saturating_add(after.total.saturating_sub(before.total));
    }
    if total == 0 {
        return None;
    }
    Some((1.0 - idle as f64 / total as f64).clamp(0.0, 1.0))
}

fn read_load_average() -> Result<f64, ReadinessError> {
    let raw = fs::read_to_string("/proc/loadavg")?;
    let loadavg = parse_loadavg_first_field(&raw)?;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as f64)
        .unwrap_or(1.0);
    Ok(loadavg / cores)
}

fn parse_loadavg_first_field(raw: &str) -> Result<f64, ReadinessError> {
    let first = raw
        .split_whitespace()
        .next()
        .ok_or(ReadinessError::Unavailable("empty /proc/loadavg"))?;
    first.parse().map_err(|_| ReadinessError::ParseFailure {
        raw: first.to_string(),
    })
}

pub(super) fn cpu_part(cpu: CpuLoad, load_below: f64) -> String {
    let reading = match cpu {
        CpuLoad::LoadAverage(_) => "load",
        CpuLoad::BusyShare(_) => "busy",
    };
    let cpu_pct = (cpu.ratio() * 100.0).round();
    let verdict = if cpu.ratio() < load_below {
        "ok"
    } else {
        "busy"
    };
    format!("cpu={reading}:{cpu_pct}%({verdict})")
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("0.10 0.20 0.30 1/123 4567", 0.10)]
    #[case("3.50 4.10 4.30 8/200 9999\n", 3.50)]
    fn loadavg_takes_first_field(
        #[case] raw: &str,
        #[case] want: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let got = parse_loadavg_first_field(raw)?;
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        Ok(())
    }

    /// One bound judges either reading, and the fragment names which it was.
    #[rstest]
    #[case(CpuLoad::LoadAverage(0.10), 0.30, "cpu=load:10%(ok)")]
    #[case(CpuLoad::LoadAverage(0.30), 0.30, "cpu=load:30%(busy)")]
    #[case(CpuLoad::LoadAverage(0.40), 0.50, "cpu=load:40%(ok)")]
    #[case(CpuLoad::LoadAverage(0.50), 0.50, "cpu=load:50%(busy)")]
    #[case(CpuLoad::BusyShare(0.0), 0.30, "cpu=busy:0%(ok)")]
    #[case(CpuLoad::BusyShare(0.10), 0.30, "cpu=busy:10%(ok)")]
    #[case(CpuLoad::BusyShare(0.30), 0.30, "cpu=busy:30%(busy)")]
    #[case(CpuLoad::BusyShare(0.40), 0.50, "cpu=busy:40%(ok)")]
    #[case(CpuLoad::BusyShare(1.0), 0.50, "cpu=busy:100%(busy)")]
    fn cpu_part_judges_against_the_bound(
        #[case] cpu: CpuLoad,
        #[case] load_below: f64,
        #[case] want: &str,
    ) {
        assert_eq!(cpu_part(cpu, load_below), want);
    }

    #[test]
    fn loadavg_empty_is_unavailable() -> anyhow::Result<()> {
        let err = parse_loadavg_first_field("")
            .err()
            .context("expected empty /proc/loadavg to be Unavailable")?;
        assert!(
            matches!(err, ReadinessError::Unavailable("empty /proc/loadavg")),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn loadavg_garbage_is_parse_failure() -> anyhow::Result<()> {
        let err = parse_loadavg_first_field("notanumber 0.1 0.2")
            .err()
            .context("expected garbage loadavg to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == "notanumber"),
            "got {err:?}",
        );
        Ok(())
    }

    fn cpus(list: &str) -> Result<CpuSet, ReadinessError> {
        parse_cpu_list(list)
    }

    #[rstest]
    #[case::single("3", &[3])]
    #[case::range("0-3", &[0, 1, 2, 3])]
    #[case::mixed("0-1,4,6-7", &[0, 1, 4, 6, 7])]
    #[case::trailing_newline("64-65\n", &[64, 65])]
    #[case::as_the_status_line_pads_it("\t0-1", &[0, 1])]
    #[case::empty("", &[])]
    #[case::blank_line("\n", &[])]
    fn cpu_list_names_its_cpus(#[case] raw: &str, #[case] want: &[u16]) -> anyhow::Result<()> {
        let want: CpuSet = want.iter().copied().collect();
        assert_eq!(parse_cpu_list(raw)?, want);
        Ok(())
    }

    #[rstest]
    #[case::letters("abc")]
    #[case::a_mask_not_a_list("ff,ffffffff")]
    #[case::open_range("0-")]
    #[case::no_start("-3")]
    #[case::descending("3-1")]
    #[case::empty_item("0,,2")]
    #[case::past_any_cpu_id("0-70000")]
    fn cpu_list_garbage_is_parse_failure(#[case] raw: &str) -> anyhow::Result<()> {
        let err = parse_cpu_list(raw)
            .err()
            .context("expected a garbage CPU list to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { raw: ref got } if got == raw),
            "got {err:?}",
        );
        Ok(())
    }

    /// The list, not the hex mask on the line before it.
    #[test]
    fn allowed_cpus_come_from_the_status_list_line() -> anyhow::Result<()> {
        let status = "\
Name:\tpipette
Threads:\t9
Cpus_allowed:\tffff,ffffffff,ffffffff
Cpus_allowed_list:\t64-65
Mems_allowed:\t00000003
Mems_allowed_list:\t0-1
";
        assert_eq!(parse_allowed_cpus(status)?, cpus("64-65")?);
        Ok(())
    }

    #[test]
    fn status_without_the_list_is_unavailable() -> anyhow::Result<()> {
        let err = parse_allowed_cpus("Name:\tpipette\nThreads:\t9\nMems_allowed_list:\t0\n")
            .err()
            .context("expected a status without Cpus_allowed_list to be Unavailable")?;
        assert!(
            matches!(err, ReadinessError::Unavailable(NO_ALLOWED_CPUS)),
            "got {err:?}",
        );
        Ok(())
    }

    #[test]
    fn a_failed_read_names_its_file() -> anyhow::Result<()> {
        let path = "/nonexistent/pipette-readiness";
        let err = read_text(path)
            .err()
            .context("expected a missing file to fail the read")?;
        let source = std::error::Error::source(&err).context("an Io error has a source")?;
        assert!(
            source.to_string().starts_with(&format!("{path}: ")),
            "got {source}",
        );
        assert!(
            matches!(err, ReadinessError::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound),
            "got {err:?}",
        );
        Ok(())
    }

    #[rstest]
    #[case::whole_host("0-191", "0-191", Ok(None))]
    #[case::slurm_job("64-65", "0-191", Ok(Some("64-65")))]
    #[case::split_cpuset("0-7,96-103", "0-191", Ok(Some("0-7,96-103")))]
    #[case::all_but_one("1-3", "0-3", Ok(Some("1-3")))]
    #[case::mask_names_every_possible_cpu("0-255", "0-191", Ok(None))]
    #[case::one_cpu_host("0", "0", Ok(None))]
    #[case::some_allowed_cpus_offline("0-1,200-201", "0-191", Err(ALLOWED_CPUS_PARTLY_ONLINE))]
    // Four CPUs of a container as lxcfs shows them: the mask keeps the host's
    // ids and the online list counts from 0.
    #[case::renumbered_from_zero("2-5", "0-3", Err(ALLOWED_CPUS_PARTLY_ONLINE))]
    #[case::every_allowed_cpu_offline("200-201", "0-3", Err(NO_ALLOWED_CPU_ONLINE))]
    #[case::empty_mask("", "0-3", Err(NO_ALLOWED_CPU_ONLINE))]
    #[case::nothing_online("0-3", "", Err(NO_ALLOWED_CPU_ONLINE))]
    fn only_some_of_the_online_cpus_is_confined(
        #[case] allowed: &str,
        #[case] online: &str,
        #[case] want: Result<Option<&str>, &str>,
    ) -> anyhow::Result<()> {
        match (confined_to(&cpus(allowed)?, &cpus(online)?), want) {
            (Ok(got), Ok(want)) => assert_eq!(got, want.map(cpus).transpose()?),
            (Err(ReadinessError::Unavailable(got)), Err(want)) => assert_eq!(got, want),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    /// A 2-CPU host as a 6.x kernel prints it: the aggregate line, the per-CPU
    /// lines with `guest` and `guest_nice` set, then everything else.
    const PROC_STAT: &str = "\
cpu  1000 20 500 8000 100 0 50 30 90 10
cpu0 600 20 300 4000 60 0 30 10 90 10
cpu1 400 0 200 4000 40 0 20 20 0 0
intr 123456 0 9 0 0 0 0 0 0 1 0 0 0 0
ctxt 6789
btime 1760000000
processes 4242
procs_running 1
procs_blocked 0
softirq 100 0 20 1 5 0 0 3 40 0 31
";

    fn times(idle: u64, total: u64) -> CpuTimes {
        CpuTimes { idle, total }
    }

    #[rstest]
    // The aggregate line and the non-CPU lines are not CPUs; `guest` and
    // `guest_nice` are not in the total.
    #[case::a_real_file(PROC_STAT, &[(0, times(4060, 5020)), (1, times(4040, 4680))])]
    #[case::through_idle_only("cpu0 1 2 3 4\n", &[(0, times(4, 10))])]
    #[case::through_iowait("cpu7 1 2 3 4 5\n", &[(7, times(9, 15))])]
    #[case::through_softirq("cpu0 1 2 3 4 5 6 7\n", &[(0, times(9, 28))])]
    #[case::aggregate_only("cpu  1 2 3 4 5 6 7 8 9 10\n", &[])]
    #[case::no_cpu_lines("intr 12345\nctxt 6789\n", &[])]
    #[case::empty("", &[])]
    fn proc_stat_parses_per_cpu_lines(
        #[case] raw: &str,
        #[case] want: &[(u16, CpuTimes)],
    ) -> anyhow::Result<()> {
        let want: BTreeMap<u16, CpuTimes> = want.iter().copied().collect();
        assert_eq!(parse_proc_stat(raw)?, want);
        Ok(())
    }

    #[rstest]
    #[case::too_few_fields("cpu0 1 2 3")]
    #[case::garbage_field("cpu0 1 nope 3 4 5")]
    #[case::negative_field("cpu0 1 2 3 -4")]
    fn proc_stat_malformed_cpu_line_is_parse_failure(#[case] line: &str) -> anyhow::Result<()> {
        let err = parse_proc_stat(&format!("cpu  9 9 9 9\n{line}\nintr 1\n"))
            .err()
            .context("expected a malformed cpuN line to be a ParseFailure")?;
        assert!(
            matches!(err, ReadinessError::ParseFailure { ref raw } if raw == line),
            "got {err:?}",
        );
        Ok(())
    }

    /// A 4-CPU host over one window: `(idle, total)` ticks each CPU gained.
    fn window(gained: [(u64, u64); 4]) -> (BTreeMap<u16, CpuTimes>, BTreeMap<u16, CpuTimes>) {
        let before = (0u16..).map(|cpu| (cpu, times(5000, 10_000))).take(4);
        let after = (0u16..)
            .zip(gained)
            .map(|(cpu, (idle, total))| (cpu, times(5000 + idle, 10_000 + total)));
        (before.collect(), after.collect())
    }

    const IDLE: (u64, u64) = (100, 100);
    const BUSY: (u64, u64) = (0, 100);

    #[rstest]
    #[case::idle_host([IDLE, IDLE, IDLE, IDLE], "0-1", Some(0.0))]
    #[case::own_cpus_busy_on_an_idle_host([BUSY, BUSY, IDLE, IDLE], "0-1", Some(1.0))]
    #[case::own_cpus_idle_on_a_busy_host([IDLE, IDLE, BUSY, BUSY], "0-1", Some(0.0))]
    #[case::one_of_two_busy([BUSY, IDLE, BUSY, BUSY], "0-1", Some(0.5))]
    #[case::partly_busy([(70, 100), (70, 100), BUSY, BUSY], "0-1", Some(0.30))]
    #[case::uneven_windows([(0, 50), (150, 150), IDLE, IDLE], "0-1", Some(0.25))]
    #[case::no_counter_advanced([(0, 0), (0, 0), BUSY, BUSY], "0-1", None)]
    #[case::cpus_the_file_does_not_list([BUSY, BUSY, BUSY, BUSY], "8-9", None)]
    #[case::no_cpus([BUSY, BUSY, BUSY, BUSY], "", None)]
    // More idle than total: a counter went backwards inside the window.
    #[case::clamped_at_idle([(120, 100), IDLE, BUSY, BUSY], "0", Some(0.0))]
    fn busy_share_counts_only_the_given_cpus(
        #[case] gained: [(u64, u64); 4],
        #[case] of: &str,
        #[case] want: Option<f64>,
    ) -> anyhow::Result<()> {
        let (before, after) = window(gained);
        match (busy_share(&before, &after, &cpus(of)?), want) {
            (Some(got), Some(want)) => {
                assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
            }
            (None, None) => {}
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    #[rstest]
    #[case::gone_offline(true)]
    #[case::come_online(false)]
    fn busy_share_skips_a_cpu_missing_from_a_snapshot(
        #[case] missing_after: bool,
    ) -> anyhow::Result<()> {
        let (mut before, mut after) = window([(70, 100), BUSY, IDLE, IDLE]);
        if missing_after {
            after.remove(&1);
        } else {
            before.remove(&1);
        }
        let got = busy_share(&before, &after, &cpus("0-1")?)
            .context("CPU 0 advanced in both snapshots")?;
        assert!((got - 0.30).abs() < 1e-9, "got {got}");
        Ok(())
    }

    /// The delta must not wrap into 2^64 ticks. With only the idle counter
    /// going back, everything the total gained was busy.
    #[rstest]
    #[case::idle_and_total(times(4990, 9_990), None)]
    #[case::idle_alone(times(4990, 10_100), Some(1.0))]
    fn busy_share_reads_a_backwards_counter_as_no_advance(
        #[case] after: CpuTimes,
        #[case] want: Option<f64>,
    ) -> anyhow::Result<()> {
        let before = BTreeMap::from([(0, times(5000, 10_000))]);
        let after = BTreeMap::from([(0, after)]);
        assert_eq!(busy_share(&before, &after, &cpus("0")?), want);
        Ok(())
    }

    fn unreadable() -> Result<Option<f64>, ReadinessError> {
        Err(ReadinessError::Unavailable(NO_ALLOWED_CPUS))
    }

    fn no_proc<T>() -> Result<T, ReadinessError> {
        Err(std::io::Error::from(std::io::ErrorKind::NotFound).into())
    }

    /// At most two windows, and none for an unconfined or unplaceable mask. Each
    /// case lists the windows the sampler returns; CPUs 0-3 are online.
    #[rstest]
    #[case::first_window("2-3", vec![Ok(Some(0.25))], Ok(Some(0.25)))]
    #[case::second_window("2-3", vec![Ok(None), Ok(Some(0.25))], Ok(Some(0.25)))]
    #[case::no_window_advanced(
        "2-3",
        vec![Ok(None), Ok(None)],
        Err("did not advance across two windows")
    )]
    #[case::unreadable_counters("2-3", vec![no_proc()], Err("i/o failure"))]
    #[case::unreadable_on_the_resample("2-3", vec![Ok(None), no_proc()], Err("i/o failure"))]
    #[case::not_confined("0-3", vec![], Ok(None))]
    #[case::mask_partly_online("2-5", vec![], Err(ALLOWED_CPUS_PARTLY_ONLINE))]
    fn confined_process_samples_its_own_cpus(
        #[case] allowed: &str,
        #[case] windows: Vec<Result<Option<f64>, ReadinessError>>,
        #[case] expected: Result<Option<f64>, &str>,
    ) -> anyhow::Result<()> {
        let allowed = cpus(allowed)?;
        let want_sampled = vec![allowed.clone(); windows.len()];
        let mut windows = windows.into_iter();
        let mut sampled = Vec::new();
        let got = confined_busy_share(&allowed, &cpus("0-3")?, |cpus| {
            sampled.push(cpus.clone());
            windows.next().unwrap_or(Ok(None))
        });
        match (got, expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(got), Err(want)) => assert!(got.to_string().contains(want), "{got}"),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        assert_eq!(sampled, want_sampled);
        Ok(())
    }

    /// The only failure that reaches the caller is the load average's own.
    #[rstest]
    #[case::confined(Ok(Some(0.02)), Ok(0.62), Ok(CpuLoad::BusyShare(0.02)))]
    #[case::confined_without_a_load_average(
        Ok(Some(0.02)),
        no_proc(),
        Ok(CpuLoad::BusyShare(0.02))
    )]
    #[case::unconfined(Ok(None), Ok(0.10), Ok(CpuLoad::LoadAverage(0.10)))]
    #[case::unreadable_confinement(unreadable(), Ok(0.10), Ok(CpuLoad::LoadAverage(0.10)))]
    #[case::unreadable_counters(no_proc(), Ok(0.10), Ok(CpuLoad::LoadAverage(0.10)))]
    #[case::unconfined_without_a_load_average(Ok(None), no_proc(), Err("i/o failure"))]
    #[case::nothing_readable(no_proc(), no_proc(), Err("i/o failure"))]
    fn a_reading_that_cannot_be_confined_is_the_load_average(
        #[case] confined: Result<Option<f64>, ReadinessError>,
        #[case] load_average: Result<f64, ReadinessError>,
        #[case] expected: Result<CpuLoad, &str>,
    ) -> anyhow::Result<()> {
        match (cpu_load(confined, || load_average), expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(got), Err(want)) => assert!(got.to_string().contains(want), "{got}"),
            (got, want) => anyhow::bail!("got {got:?}, want {want:?}"),
        }
        Ok(())
    }

    /// Holds the reading to this host's real confinement, so a read that fails
    /// and falls back fails here. A sandbox that stubs `/proc` fails this test.
    #[test]
    fn cpu_load_reads_from_this_host() -> anyhow::Result<()> {
        let (allowed, online) = (read_allowed_cpus()?, read_online_cpus()?);
        let confined = confined_to(&allowed, &online)?;
        let stat = read_proc_stat()?;
        let sampled = confined.as_ref().unwrap_or(&online);
        assert!(
            sampled.iter().all(|cpu| stat.contains_key(cpu)),
            "CPUs {sampled:?} of the {:?} in /proc/stat",
            stat.keys(),
        );
        match (read_cpu_load()?, confined) {
            (CpuLoad::BusyShare(busy), Some(_)) => {
                assert!((0.0..=1.0).contains(&busy), "{busy}");
            }
            (CpuLoad::LoadAverage(load), None) => {
                assert!(load.is_finite() && load >= 0.0, "{load}");
            }
            (cpu, confined) => anyhow::bail!("read {cpu:?} while confined to {confined:?}"),
        }
        Ok(())
    }
}
