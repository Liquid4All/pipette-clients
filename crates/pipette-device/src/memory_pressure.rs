//! The host's memory pressure at a cell's start and each rep's edges, for the
//! log alone: free memory, swap traffic, reclaim, PSI and zram, so a row whose
//! memory probe saw swap can be read against when the host swapped.
//!
//! The figures are system-wide on purpose: they show other processes' swap
//! too. Every file is optional (SELinux denies some on Android), and a field
//! no read found is left out of the event rather than logged as 0.

use std::cell::Cell;
use std::time::Instant;

/// One cell's events. A `Cell`, not a lock: the observer is a `&dyn Fn`
/// without `Sync`.
#[derive(Default)]
pub struct MemoryPressure {
    last: Cell<Option<(Instant, Counters)>>,
}

impl MemoryPressure {
    /// Logs one event at `sample_point`; a no-op off Linux and Android, which
    /// have none of these files. A handful of small reads, so keep it out of a
    /// timed window.
    pub fn log(&self, sample_point: &str) {
        if cfg!(any(target_os = "linux", target_os = "android")) {
            self.log_read(sample_point, |path| {
                std::fs::read_to_string(path).unwrap_or_default()
            });
        }
    }

    /// [`Self::log`] on what `read` gives for each file, empty where it is
    /// unreadable.
    fn log_read(&self, sample_point: &str, read: impl Fn(&str) -> String) {
        let meminfo = read("/proc/meminfo");
        let counters = Counters::parse(&read("/proc/vmstat"), &read("/proc/pressure/memory"));
        let [zram_orig_bytes, zram_compr_bytes, zram_used_bytes] =
            leading_numbers(&read("/sys/block/zram0/mm_stat"));
        let now = Instant::now();
        let last = self.last.replace(Some((now, counters)));
        // On the first event nothing has moved yet, so the counters are left out.
        let moved = last.map_or_else(Counters::default, |(_, earlier)| counters.since(earlier));
        let window_ms =
            last.and_then(|(at, _)| u64::try_from(now.duration_since(at).as_millis()).ok());
        let bytes = |key| field(&meminfo, key).map(|kib: u64| kib.saturating_mul(1024));
        tracing::info!(
            sample_point,
            window_ms,
            mem_free_bytes = bytes("MemFree"),
            mem_available_bytes = bytes("MemAvailable"),
            cached_bytes = bytes("Cached"),
            dirty_bytes = bytes("Dirty"),
            anon_pages_bytes = bytes("AnonPages"),
            swap_free_bytes = bytes("SwapFree"),
            swap_cached_bytes = bytes("SwapCached"),
            kreclaimable_bytes = bytes("KReclaimable"),
            dma_heap_pool_bytes = bytes("DmaHeapPool"),
            zram_orig_bytes,
            zram_compr_bytes,
            zram_used_bytes,
            pswpin = moved.pswpin,
            pswpout = moved.pswpout,
            pgmajfault = moved.pgmajfault,
            allocstall = moved.allocstall,
            pgsteal_kswapd = moved.pgsteal_kswapd,
            pgsteal_direct = moved.pgsteal_direct,
            workingset_refault_anon = moved.workingset_refault_anon,
            workingset_refault_file = moved.workingset_refault_file,
            psi_some_ms = moved.psi_some_us.map(|us| us / 1000),
            psi_full_ms = moved.psi_full_us.map(|us| us / 1000),
            "memory pressure"
        );
    }
}

/// The cumulative `/proc/vmstat` and `/proc/pressure/memory` counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    pswpin: Option<u64>,
    pswpout: Option<u64>,
    pgmajfault: Option<u64>,
    /// Every zone's `allocstall_*`, summed.
    allocstall: Option<u64>,
    pgsteal_kswapd: Option<u64>,
    pgsteal_direct: Option<u64>,
    workingset_refault_anon: Option<u64>,
    workingset_refault_file: Option<u64>,
    psi_some_us: Option<u64>,
    psi_full_us: Option<u64>,
}

impl Counters {
    fn parse(vmstat: &str, pressure: &str) -> Self {
        Self {
            pswpin: field(vmstat, "pswpin"),
            pswpout: field(vmstat, "pswpout"),
            pgmajfault: field(vmstat, "pgmajfault"),
            allocstall: sum_prefixed(vmstat, "allocstall"),
            pgsteal_kswapd: field(vmstat, "pgsteal_kswapd"),
            pgsteal_direct: field(vmstat, "pgsteal_direct"),
            workingset_refault_anon: field(vmstat, "workingset_refault_anon"),
            workingset_refault_file: field(vmstat, "workingset_refault_file"),
            psi_some_us: psi_total(pressure, "some"),
            psi_full_us: psi_total(pressure, "full"),
        }
    }

    /// What moved since `earlier`. A counter either read lacks is left out;
    /// one that went down saturates to 0 rather than wrapping.
    fn since(self, earlier: Self) -> Self {
        let moved = |now: Option<u64>, then: Option<u64>| Some(now?.saturating_sub(then?));
        Self {
            pswpin: moved(self.pswpin, earlier.pswpin),
            pswpout: moved(self.pswpout, earlier.pswpout),
            pgmajfault: moved(self.pgmajfault, earlier.pgmajfault),
            allocstall: moved(self.allocstall, earlier.allocstall),
            pgsteal_kswapd: moved(self.pgsteal_kswapd, earlier.pgsteal_kswapd),
            pgsteal_direct: moved(self.pgsteal_direct, earlier.pgsteal_direct),
            workingset_refault_anon: moved(
                self.workingset_refault_anon,
                earlier.workingset_refault_anon,
            ),
            workingset_refault_file: moved(
                self.workingset_refault_file,
                earlier.workingset_refault_file,
            ),
            psi_some_us: moved(self.psi_some_us, earlier.psi_some_us),
            psi_full_us: moved(self.psi_full_us, earlier.psi_full_us),
        }
    }
}

/// `key`'s number in a `key value` (vmstat) or `Key: value kB` (meminfo)
/// listing.
fn field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        if words.next()?.trim_end_matches(':') != key {
            return None;
        }
        words.next()?.parse().ok()
    })
}

/// The sum of every `prefix*` counter in a `key value` listing.
fn sum_prefixed(text: &str, prefix: &str) -> Option<u64> {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            if !key.starts_with(prefix) {
                return None;
            }
            value.trim().parse().ok()
        })
        .reduce(u64::saturating_add)
}

/// The `total=` microseconds of a PSI file's `some` or `full` line.
fn psi_total(pressure: &str, kind: &str) -> Option<u64> {
    pressure.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        if words.next()? != kind {
            return None;
        }
        words
            .find_map(|word| word.strip_prefix("total="))?
            .parse()
            .ok()
    })
}

/// The first `N` numbers of a line: zram's `mm_stat` leads with
/// `orig_data_size compr_data_size mem_used_total`, in bytes.
fn leading_numbers<const N: usize>(text: &str) -> [Option<u64>; N] {
    let mut words = text.split_whitespace();
    std::array::from_fn(|_| words.next()?.parse().ok())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MEMINFO: &str = "\
MemTotal:       11386048 kB
MemFree:          212340 kB
MemAvailable:    3105524 kB
Cached:          2981004 kB
SwapCached:        51200 kB
AnonPages:       5012932 kB
SwapFree:        3145724 kB
Dirty:              1024 kB
KReclaimable:     410000 kB
";

    const VMSTAT: &str = "\
pgmajfault 9000
pswpin 120
pswpout 450
allocstall_dma 0
allocstall_normal 7
allocstall_movable 3
pgsteal_kswapd 81234
pgsteal_direct 512
workingset_refault_anon 66
workingset_refault_file 777
";

    const PRESSURE: &str = "\
some avg10=0.00 avg60=0.12 avg300=0.05 total=123456
full avg10=0.00 avg60=0.00 avg300=0.00 total=7890
";

    #[rstest]
    #[case::meminfo(MEMINFO, "MemFree", Some(212_340))]
    #[case::not_its_swap_twin(MEMINFO, "Cached", Some(2_981_004))]
    #[case::swap_cached(MEMINFO, "SwapCached", Some(51_200))]
    #[case::absent(MEMINFO, "DmaHeapPool", None)]
    #[case::vmstat(VMSTAT, "pswpout", Some(450))]
    #[case::empty("", "MemFree", None)]
    fn field_reads_one_key(#[case] text: &str, #[case] key: &str, #[case] expected: Option<u64>) {
        assert_eq!(field(text, key), expected);
    }

    #[rstest]
    #[case::read(VMSTAT, PRESSURE, Counters {
        pswpin: Some(120),
        pswpout: Some(450),
        pgmajfault: Some(9000),
        allocstall: Some(10),
        pgsteal_kswapd: Some(81_234),
        pgsteal_direct: Some(512),
        workingset_refault_anon: Some(66),
        workingset_refault_file: Some(777),
        psi_some_us: Some(123_456),
        psi_full_us: Some(7890),
    })]
    // SELinux denied both files: every counter is absent, not 0.
    #[case::unreadable("", "", Counters::default())]
    fn counters_parse_vmstat_and_pressure(
        #[case] vmstat: &str,
        #[case] pressure: &str,
        #[case] expected: Counters,
    ) {
        assert_eq!(Counters::parse(vmstat, pressure), expected);
    }

    #[test]
    fn since_subtracts_what_both_reads_have() {
        let earlier = Counters {
            pswpin: Some(100),
            pswpout: Some(500),
            psi_some_us: Some(1),
            ..Counters::default()
        };
        let moved = Counters::parse(VMSTAT, PRESSURE).since(earlier);
        assert_eq!(moved.pswpin, Some(20));
        // A counter that went down saturates instead of wrapping.
        assert_eq!(moved.pswpout, Some(0));
        assert_eq!(moved.psi_some_us, Some(123_455));
        // The earlier read lacked it, so nothing can be said about the window.
        assert_eq!(moved.pgmajfault, None);
    }

    #[rstest]
    #[case::mm_stat(
        "2147483648 536870912 553648128 0 600000000 1024 0 0 0\n",
        [Some(2_147_483_648), Some(536_870_912), Some(553_648_128)],
    )]
    #[case::short("4096\n", [Some(4096), None, None])]
    #[case::unreadable("", [None, None, None])]
    fn leading_numbers_read_zram(#[case] mm_stat: &str, #[case] expected: [Option<u64>; 3]) {
        assert_eq!(leading_numbers::<3>(mm_stat), expected);
    }

    #[rstest]
    #[case::some("some", Some(5))]
    #[case::no_full_line("full", None)]
    fn psi_total_reads_one_line(#[case] kind: &str, #[case] expected: Option<u64>) {
        let some_only = "some avg10=0.00 avg60=0.00 avg300=0.00 total=5\n";
        assert_eq!(psi_total(some_only, kind), expected);
    }

    #[derive(Clone, Default)]
    struct Written(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Written {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The first event logs levels and no window or counters; the next logs
    /// what moved over its window. A file no read found leaves its fields out, not 0.
    #[test]
    fn chained_events_log_what_moved() -> anyhow::Result<()> {
        let written = Written::default();
        let writer = written.clone();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_writer(move || writer.clone())
            .finish();
        let later = VMSTAT.replace("pswpin 120", "pswpin 150");
        let files = |vmstat: &str, path: &str| match path {
            "/proc/meminfo" => MEMINFO.to_owned(),
            "/proc/vmstat" => vmstat.to_owned(),
            // PSI and zram unreadable, as SELinux leaves them on some phones.
            _ => String::new(),
        };

        tracing::subscriber::with_default(subscriber, || {
            let memory = MemoryPressure::default();
            memory.log_read("before_cell", |path| files(VMSTAT, path));
            memory.log_read("before_rep", |path| files(&later, path));
        });

        let bytes = written
            .0
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .clone();
        let events = std::str::from_utf8(&bytes)?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<Vec<serde_json::Value>, _>>()?;
        let [head, next] = events.as_slice() else {
            anyhow::bail!("expected two events, got {events:?}");
        };
        assert_eq!(head["message"], "memory pressure");
        assert_eq!(head["sample_point"], "before_cell");
        assert_eq!(head["mem_free_bytes"], 212_340 * 1024);
        let present: Vec<_> = [
            "window_ms",
            "pswpin",
            "dma_heap_pool_bytes",
            "psi_some_ms",
            "zram_orig_bytes",
        ]
        .into_iter()
        .filter(|key| head.get(key).is_some())
        .collect();
        assert!(present.is_empty(), "{present:?} in {head}");
        assert_eq!(next["sample_point"], "before_rep");
        assert!(next["window_ms"].is_u64(), "got {next}");
        assert_eq!(next["pswpin"], 30);
        assert_eq!(next["pswpout"], 0);
        Ok(())
    }
}
