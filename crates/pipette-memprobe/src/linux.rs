//! The Linux and Android sampler: `/proc/<pid>/status`, whose `VmHWM` is the
//! peak resident set.
//!
//! `VmHWM` is a peak *resident* set, which on a device with swap is not the peak
//! the run required: the kernel can compress a page into zram, dropping it out
//! of RSS while the process still owns and needs it. Both terms are kept because
//! neither subsumes the other. `VmHWM` is the kernel's mark, kept lazily from
//! approximate counters, so reads can fall below it (by up to 2 MiB between 10 ms
//! ticks on Linux 6.8). It is also blind to swap; the sampled `VmRSS + VmSwap`
//! sum sees swap but has to land near the peak. There is no kernel watermark for
//! swap, which is why this samples at all. Reading `/proc` rather than `wait4`
//! also leaves the reap to `Process::wait`.
//!
//! An exited child loses `VmHWM` from `/proc` before it is reaped, so a sampler
//! stopped in between (`Process::wait_success_before_reap`) reads `waitid`'s
//! `ru_maxrss` instead. That figure is the maximum over the process, the image
//! it was before `exec`, and the children it reaped. A forked child's image is a
//! copy of pipette, so the figure counts only when it is above pipette's own
//! peak ([`clears_own_peak`]); otherwise the reads while it ran stand. Either
//! way the stop has to find the process, running or exited and unreaped, or
//! the peak is refused ([`PeakError::UnreadAtStop`]). Measurements and
//! rationale: docs/methodology/peak-memory-android.md.
//!
//! On Android, [`Watch::PeakWithAccelerators`] also reads, each tick, the Adreno
//! memory the KGSL driver mapped into the process and the dmabufs it holds,
//! which its resident set misses but for the KGSL pages the CPU touched. Its
//! `maps` read costs ~0.5 ms on a GPU run, which a benchmark that reports bytes
//! can afford. On Linux, [`Watch::PeakWithGpu`] reads instead what the DRM
//! drivers bill the process's clients ([`crate::drm`]) and, for an NVIDIA GPU,
//! whose driver bills none there, what NVML attributes to it ([`crate::nvml`]).
//! Those are current figures, not watermarks, so for a program that exits they
//! end at its last tick while it ran.

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use pipette_plan_types::MemoryObservation;

#[cfg(target_os = "linux")]
use super::drm::Fds;
#[cfg(target_os = "linux")]
use super::nvml::{NotAvailable, Nvml, Pids};
#[cfg(target_os = "linux")]
use super::{NvmlError, NvmlReadError};
use super::{Peak, PeakError, StartError, SupportedWatch, UnsupportedError, Watch, WatchError};

/// How often to read when the peak is the benchmark's own result.
///
/// This bounds the sampled terms. `VmHWM` is close but not exact (see the module
/// docs), and when swap suppresses it the swap-aware figure
/// ([`Footprint::peak_ram_kib`]) falls back to the sampled `VmRSS + VmSwap` sum,
/// which can only under-read: the miss is roughly the footprint's growth rate
/// times this interval. A no-mmap load faults in
/// ~900 MB/s, so 10 ms costs about 9 MB of resolution where 25 ms cost ~22 MB.
/// Two small reads per tick at 100 Hz is ~1% of one core, which a benchmark that
/// reports bytes can afford.
const PEAK_INTERVAL: Duration = Duration::from_millis(10);

/// How often to read for an observation riding along on a benchmark that reports
/// time: a tenth of [`PEAK_INTERVAL`]'s duty cycle, so watching cannot move the
/// number. `VmHWM` loses little to it when the stop's read takes the watermark,
/// live or as an exit figure above pipette's own peak; an exit figure set aside
/// ([`clears_own_peak`]) leaves the last interval unread. It bounds the sampled
/// swap term, whose resolution past ~100 ms is no longer worth the saving.
/// docs/methodology/peak-memory-usage.md works through what each term gives up.
const OBSERVE_INTERVAL: Duration = Duration::from_millis(100);

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

/// `VmHWM` / `VmRSS` / `VmSwap` from one read, plus `majflt`. `None` without
/// `VmHWM`, which a process prints until it exits. Another field the kernel
/// omits reads 0: `VmSwap` is absent on a swapless kernel, where "nothing
/// swapped" is the correct answer rather than a failure.
fn read_status_sample(pid: u32) -> Option<StatusSample> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let stat = read_stat(pid).unwrap_or_default();
    Some(StatusSample {
        hwm_kib: status_kib(&status, "VmHWM")?,
        rss_kib: status_kib(&status, "VmRSS").unwrap_or(0),
        swap_kib: status_kib(&status, "VmSwap").unwrap_or(0),
        major_faults: stat.major_faults,
        start_time: stat.start_time,
    })
}

/// One `kB` field of a `/proc/<pid>/status` read.
fn status_kib(status: &str, name: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let rest = line.strip_prefix(name)?.strip_prefix(':')?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

/// How far an exit figure has to clear pipette's own `VmHWM` to be the
/// child's. Both come from per-CPU counters: over 5,400 forks and spawns on a
/// 30-CPU host, with threads faulting memory, a small child's figure stayed
/// 16 KiB to 50 MiB below pipette's, about 1.5 MiB for a quiet one. 4 MiB
/// leaves room past that, about 1% of the smallest tool peak measured (305 MiB).
const OWN_PEAK_MARGIN_KIB: u64 = 4 << 10;

/// Whether an exit figure ([`exit_peak_rss_kib`]) can only be the child's own.
/// A forked child's starts at its copy of pipette, no larger than pipette's
/// resident set at the fork, and a spawned one's at pipette's `VmHWM` then;
/// read after the fork, `own_hwm_kib` covers both. Unread, it vouches for
/// nothing.
fn clears_own_peak(exit_kib: u64, own_hwm_kib: Option<u64>) -> bool {
    own_hwm_kib.is_some_and(|own_kib| exit_kib > own_kib.saturating_add(OWN_PEAK_MARGIN_KIB))
}

/// This process's `VmHWM`.
fn own_hwm_kib() -> Option<u64> {
    status_kib(&std::fs::read_to_string("/proc/self/status").ok()?, "VmHWM")
}

// The kernel fills its own `struct rusage`: two `{long, long}` timevals and 14
// longs, 144 bytes on a 64-bit target. libc's `waitid` takes none, so nothing
// else holds libc's copy to that layout (a 32-bit libc with a 64-bit `time_t`
// would not match).
const _: () = assert!(
    size_of::<libc::rusage>() == 18 * size_of::<libc::c_long>(),
    "libc::rusage does not match the kernel's struct rusage"
);

/// `pid`'s peak resident set once it has exited, `waitid`'s `ru_maxrss`: the
/// maximum over the process, its image before `exec`, and the children it
/// reaped (`/proc` drops `VmHWM` at the exit). `None` for a process still
/// running, already reaped, or not this process's child.
/// `WNOWAIT` leaves the child for its owner to reap; the `rusage` argument is
/// Linux's, which libc's `waitid` leaves out.
fn exit_peak_rss_kib(pid: u32) -> Option<u64> {
    let mut info = MaybeUninit::<libc::siginfo_t>::zeroed();
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: both pointers name live, writable structs of the kernel's
    // layout, and `WNOHANG` returns at once.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_waitid,
            libc::P_PID,
            pid,
            info.as_mut_ptr(),
            libc::WEXITED | libc::WNOWAIT | libc::WNOHANG,
            usage.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: zeroed, a valid `siginfo_t`, and filled by the kernel or not.
    let info = unsafe { info.assume_init() };
    if info.si_signo != libc::SIGCHLD {
        return None;
    }
    // SAFETY: zeroed, a valid `rusage`, and filled by the kernel.
    let usage = unsafe { usage.assume_init() };
    u64::try_from(usage.ru_maxrss).ok().filter(|&kib| kib > 0)
}

/// The `/proc/<pid>/stat` fields a sample keeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Stat {
    /// Pages the run had to pull from storage or decompress out of zram: near
    /// zero when it met no pressure, six figures when it thrashed, so they
    /// witness a suppressed peak for free.
    major_faults: u64,
    /// Only identifies the process.
    start_time: u64,
}

/// `pid`'s [`Stat`]. A zombie still has one.
fn read_stat(pid: u32) -> Option<Stat> {
    parse_stat(&std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?)
}

/// `comm` is parenthesized and may itself contain spaces and parens, so the
/// positional fields start after the LAST ')': at field 3 (state), putting
/// majflt (field 12) at index 9 and starttime (field 22) at index 19.
fn parse_stat(stat: &str) -> Option<Stat> {
    let after_comm = stat.rsplit_once(')')?.1;
    let mut fields = after_comm.split_whitespace().skip(9);
    let major_faults = fields.next()?.parse().ok()?;
    let start_time = fields.nth(9)?.parse().ok()?;
    Some(Stat {
        major_faults,
        start_time,
    })
}

/// What a run required at peak, kept decomposed so a peak suppressed by reclaim
/// is visible rather than silently low.
#[derive(Clone, Copy, Debug, Default)]
pub struct Footprint {
    /// The kernel's peak-RSS mark, the largest `VmHWM` read or the exit `ru_maxrss`:
    /// close, not exact, and blind to anything in swap.
    pub peak_rss_kib: u64,
    /// Largest `VmRSS + VmSwap` sampled. The kernel keeps no watermark for
    /// swap, so a sampled maximum is the only way to count what zram holds.
    pub peak_committed_kib: u64,
    /// How much of the process zram held, and how hard it thrashed to get it
    /// back. Diagnostics: they explain a low `peak_rss_kib`, never replace it.
    pub max_swap_kib: u64,
    pub major_faults: u64,
}

impl Footprint {
    fn observe(&mut self, sample: StatusSample) {
        self.peak_rss_kib = self.peak_rss_kib.max(sample.hwm_kib);
        self.peak_committed_kib = self
            .peak_committed_kib
            .max(sample.rss_kib.saturating_add(sample.swap_kib));
        self.max_swap_kib = self.max_swap_kib.max(sample.swap_kib);
        self.major_faults = self.major_faults.max(sample.major_faults);
    }

    /// Peak RAM the run required, wherever those pages sat when the peak was
    /// struck.
    pub fn peak_ram_kib(&self) -> u64 {
        self.peak_rss_kib.max(self.peak_committed_kib)
    }
}

/// What a run is worth reporting as, or nothing: withheld unless the stop
/// found the process ([`Seen::read_at_stop`]).
fn observation(seen: &Seen) -> MemoryObservation {
    if !seen.read_at_stop {
        return MemoryObservation::default();
    }
    let footprint = &seen.footprint;
    // `peak_rss_kib` (VmHWM), deliberately not `peak_ram_kib()`: the host term is
    // the resident watermark, so the field means the same thing on every OS that
    // can report it. The swap-aware sum stays out of it and rides beside it as
    // the swap term, which is what says the resident figure was suppressed. An
    // unreadable VmHWM therefore withholds the swap reading too, rather than
    // reporting a term whose peak is missing.
    MemoryObservation::with_swap(
        footprint.peak_rss_kib.saturating_mul(1024),
        footprint.max_swap_kib.saturating_mul(1024),
    )
}

/// Whether a sample came from the process the sampler was pointed at.
///
/// A zero `start_time` means the field could not be read, not that the pid was
/// recycled. Treating it as a mismatch would discard every read, the stop's
/// included, failing an otherwise healthy run.
fn same_process(identity: Option<u64>, sample_start_time: u64) -> bool {
    match (identity, sample_start_time) {
        (Some(seen), observed) if observed != 0 => seen == observed,
        _ => true,
    }
}

/// The GPU and NPU memory a process held at one read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Accelerators {
    /// The Adreno memory the KGSL driver mapped into the process, or what the
    /// DRM drivers and NVML bill it.
    gpu_bytes: u64,
    /// The dmabufs it holds open or mapped, each counted once.
    npu_bytes: u64,
    #[cfg(target_os = "linux")]
    gpu: GpuRead,
}

impl Accelerators {
    fn new(gpu_bytes: u64, npu_bytes: u64) -> Self {
        Self {
            gpu_bytes,
            npu_bytes,
            #[cfg(target_os = "linux")]
            gpu: GpuRead::default(),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            gpu_bytes: self.gpu_bytes.max(other.gpu_bytes),
            npu_bytes: self.npu_bytes.max(other.npu_bytes),
            #[cfg(target_os = "linux")]
            gpu: GpuRead {
                drm_unread: self.gpu.drm_unread || other.gpu.drm_unread,
                on_nvidia: self.gpu.on_nvidia || other.gpu.on_nvidia,
                nvml: self.gpu.nvml.max(other.gpu.nvml),
            },
        }
    }
}

/// What a GPU watch's reads said beyond the bytes, held across them.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GpuRead {
    /// A DRM fd printed no memory.
    drm_unread: bool,
    /// The process held an NVIDIA device open, whose memory only NVML reads.
    on_nvidia: bool,
    nvml: NvmlRead,
}

/// What NVML said of the process, in the order a later read outranks an
/// earlier one.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum NvmlRead {
    /// It did not open.
    #[default]
    Closed,
    /// A GPU refused one of its process lists.
    Unanswered,
    Answered,
    /// It has no figure for the process: MIG, vGPU, no permission.
    NotAvailable,
}

/// What this sampler reads for a [`Watch`], resolved before the spawn and
/// carried in a [`SupportedWatch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reads {
    Observation,
    Host,
    /// The host peak with the KGSL mappings and the dmabufs, Android's GPU and
    /// NPU memory.
    #[cfg_attr(
        all(target_os = "linux", not(test)),
        expect(
            dead_code,
            reason = "Android's reader, which Linux compiles so its tests run in CI"
        )
    )]
    Mapped,
    /// The host peak with the DRM clients' and NVML's GPU memory.
    #[cfg(target_os = "linux")]
    Gpu,
}

/// What a peak reads beside the host, each tick.
enum DeviceReader {
    None,
    /// The KGSL mappings and the dmabufs ([`Reads::Mapped`]).
    Mapped,
    /// The DRM clients' fdinfo, and NVML's per-process bytes when it opened
    /// ([`Reads::Gpu`]).
    #[cfg(target_os = "linux")]
    Gpu(Result<Nvml, NvmlError>),
}

impl DeviceReader {
    fn start(reads: Reads) -> Self {
        match reads {
            Reads::Observation | Reads::Host => Self::None,
            Reads::Mapped => Self::Mapped,
            #[cfg(target_os = "linux")]
            Reads::Gpu => Self::Gpu(Nvml::open()),
        }
    }

    /// `None` when there is nothing to read, or the read failed.
    #[cfg_attr(
        target_os = "android",
        expect(
            clippy::needless_pass_by_ref_mut,
            reason = "only the NVML reader, which Android lacks, reads mutably"
        )
    )]
    fn read(&mut self, pid: u32) -> Option<Accelerators> {
        match self {
            Self::None => None,
            Self::Mapped => read_accelerators(pid),
            #[cfg(target_os = "linux")]
            Self::Gpu(nvml) => read_gpu(pid, nvml.as_mut().ok()),
        }
    }

    /// Why NVML did not open, once reading is over.
    #[cfg(target_os = "linux")]
    fn nvml_unavailable(self) -> Option<NvmlError> {
        match self {
            Self::None | Self::Mapped => None,
            Self::Gpu(nvml) => nvml.err(),
        }
    }
}

/// What the DRM drivers, and NVML when it opened, bill `pid` now. `None` when
/// its fds cannot be listed.
#[cfg(target_os = "linux")]
fn read_gpu(pid: u32, nvml: Option<&mut Nvml>) -> Option<Accelerators> {
    let fds = super::drm::read(pid)?;
    Some(gpu_read(fds, pid, nvml.map(Nvml::processes)))
}

/// One read's GPU bytes, DRM's plus NVML's for `pid`, and what else it said.
/// `nvml` is `None` when NVML did not open.
#[cfg(target_os = "linux")]
fn gpu_read(fds: Fds, pid: u32, nvml: Option<Result<Pids, NvmlReadError>>) -> Accelerators {
    // A pid NVML does not list holds no GPU memory.
    let nvml = nvml.map(|read| read.ok().map(|mut pids| pids.remove(&pid).unwrap_or(Ok(0))));
    let said = match nvml {
        None => NvmlRead::Closed,
        Some(None) => NvmlRead::Unanswered,
        Some(Some(Ok(_))) => NvmlRead::Answered,
        Some(Some(Err(NotAvailable))) => NvmlRead::NotAvailable,
    };
    let nvml_bytes = nvml.flatten().and_then(Result::ok).unwrap_or(0);
    let drm_unread = fds.drm.is_err();
    Accelerators {
        gpu_bytes: fds.drm.unwrap_or(0).saturating_add(nvml_bytes),
        npu_bytes: 0,
        gpu: GpuRead {
            drm_unread,
            on_nvidia: fds.on_nvidia,
            nvml: said,
        },
    }
}

fn read_accelerators(pid: u32) -> Option<Accelerators> {
    let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).ok()?;
    Some(accelerators(&maps, &open_dmabufs(pid)?))
}

/// `(inode, size)` of each dmabuf the process has open, or `None` when they
/// cannot all be read, rather than leaving one out of the count.
fn open_dmabufs(pid: u32) -> Option<Vec<(u64, u64)>> {
    std::fs::read_dir(format!("/proc/{pid}/fd"))
        .ok()?
        .filter_map(|fd| {
            let fd = fd.ok()?;
            is_dmabuf(std::fs::read_link(fd.path()).ok()?.to_str()?).then(|| fd.file_name())
        })
        .map(|name| {
            let fdinfo = std::fs::read_to_string(format!("/proc/{pid}/fdinfo/{}", name.to_str()?));
            parse_dmabuf_fdinfo(&fdinfo.ok()?)
        })
        .collect()
}

/// A dmabuf's `(inode, size)` from its fdinfo.
fn parse_dmabuf_fdinfo(fdinfo: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        fdinfo
            .lines()
            .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
    };
    Some((field("ino:")?, field("size:")?))
}

/// How a dmabuf is named in `fd/` links and `maps` since Linux 5.3, which gave
/// each its own inode, as counting by inode needs.
fn is_dmabuf(path: &str) -> bool {
    path.starts_with("/dmabuf:")
}

/// A dmabuf mapped but no longer open counts once, by its first mapping, as
/// `dmabuf_dump` counts it.
fn accelerators(maps: &str, open: &[(u64, u64)]) -> Accelerators {
    let open: HashMap<u64, u64> = open.iter().copied().collect();
    let (gpu_bytes, mapped_only) = maps.lines().filter_map(mapping).fold(
        (0u64, HashMap::<u64, u64>::new()),
        |(gpu_bytes, mut mapped_only), (len, inode, path)| {
            if path.starts_with("/dev/kgsl") {
                return (gpu_bytes.saturating_add(len), mapped_only);
            }
            if is_dmabuf(path) && !open.contains_key(&inode) {
                mapped_only.entry(inode).or_insert(len);
            }
            (gpu_bytes, mapped_only)
        },
    );
    let npu_bytes = open
        .values()
        .chain(mapped_only.values())
        .fold(0u64, |sum, &bytes| sum.saturating_add(bytes));
    Accelerators::new(gpu_bytes, npu_bytes)
}

/// A `maps` line's `(length, inode, path)`, from `start-end perms offset dev
/// inode path`; `None` for an anonymous mapping, which has no path.
fn mapping(line: &str) -> Option<(u64, u64, &str)> {
    let mut fields = line.split_whitespace();
    let (start, end) = fields.next()?.split_once('-')?;
    let len = u64::from_str_radix(end, 16)
        .ok()?
        .checked_sub(u64::from_str_radix(start, 16).ok()?)?;
    let inode = fields.nth(3)?.parse().ok()?;
    Some((len, inode, fields.next()?))
}

/// What the sampling thread hands back.
#[derive(Debug, Default)]
struct Seen {
    footprint: Footprint,
    /// Whether the last read, at the stop, found the process: running and the
    /// same one, or exited and not yet reaped, which proves the stop came
    /// before the reap. An exit figure set aside ([`clears_own_peak`]) adds
    /// nothing, so the reads while it ran stand and its last interval is
    /// unread.
    read_at_stop: bool,
    /// `None` until a read of them lands.
    accelerators: Option<Accelerators>,
    /// For the log alone: when `VmSwap` first went above 0 and last raised its
    /// maximum, in ms from [`Sampler::start`].
    first_swap_ms: Option<u64>,
    peak_swap_ms: Option<u64>,
    /// Why NVML did not open, for a GPU watch.
    #[cfg(target_os = "linux")]
    nvml_unavailable: Option<NvmlError>,
}

impl Seen {
    fn observe(&mut self, sample: StatusSample, at: Option<u64>) {
        if sample.swap_kib > self.footprint.max_swap_kib {
            self.first_swap_ms = self.first_swap_ms.or(at);
            self.peak_swap_ms = at;
        }
        self.footprint.observe(sample);
    }
}

/// A watch fails on a pool it never read, and a GPU watch on a GPU that held
/// nothing; any other pool read at zero is `None`, since a peak is never zero.
fn peak(reads: Reads, pid: u32, seen: Seen) -> Result<Peak, PeakError> {
    let footprint = seen.footprint;
    if reads == Reads::Observation {
        return Err(PeakError::NotWatched);
    }
    if footprint.peak_rss_kib == 0 {
        return Err(PeakError::NoVmHwm { pid });
    }
    if !seen.read_at_stop {
        return Err(PeakError::UnreadAtStop { pid });
    }
    let accelerators = match reads {
        Reads::Observation | Reads::Host => Accelerators::default(),
        Reads::Mapped => seen.accelerators.ok_or(PeakError::NoAccelerators { pid })?,
        #[cfg(target_os = "linux")]
        Reads::Gpu => gpu_peak(pid, seen.accelerators, seen.nvml_unavailable)?,
    };
    Ok(Peak {
        host_bytes: footprint.peak_rss_kib.saturating_mul(1024),
        gpu_bytes: (accelerators.gpu_bytes > 0).then_some(accelerators.gpu_bytes),
        npu_bytes: (accelerators.npu_bytes > 0).then_some(accelerators.npu_bytes),
        footprint,
    })
}

/// A GPU watch's reads, or why they are no peak: a pool it could not read, or
/// a GPU that held nothing. NVML's failures count only for a process on an
/// NVIDIA GPU.
#[cfg(target_os = "linux")]
fn gpu_peak(
    pid: u32,
    read: Option<Accelerators>,
    nvml_unavailable: Option<NvmlError>,
) -> Result<Accelerators, PeakError> {
    let read = read.ok_or(PeakError::GpuUnread { pid })?;
    if read.gpu.drm_unread {
        return Err(PeakError::DrmUnread { pid });
    }
    if read.gpu.nvml == NvmlRead::NotAvailable {
        return Err(PeakError::GpuMemoryUnavailable { pid });
    }
    if read.gpu.on_nvidia {
        if let Some(source) = nvml_unavailable {
            return Err(PeakError::NvmlUnavailable { pid, source });
        }
        if read.gpu.nvml == NvmlRead::Unanswered {
            return Err(PeakError::NvmlUnanswered { pid });
        }
    }
    match read.gpu_bytes {
        0 if read.gpu.on_nvidia => Err(PeakError::NoGpuMemory { pid }),
        0 => Err(PeakError::NoDrmMemory { pid }),
        _ => Ok(read),
    }
}

/// A Jetson's iGPU allocates into `VmRSS`, and neither NVML nor DRM fdinfo
/// reports it (docs/methodology/peak-memory-linux.md, "GPU Counter").
#[cfg(target_os = "linux")]
pub(crate) fn gpu_memory_is_resident() -> Result<bool, WatchError> {
    match std::fs::read("/proc/device-tree/compatible") {
        Ok(compatible) => Ok(is_tegra(&compatible)),
        // An x86 host, or an arm64 one booted through ACPI, has no device tree.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(WatchError::DeviceTree { kind: err.kind() }),
    }
}

#[cfg(target_os = "linux")]
fn is_tegra(compatible: &[u8]) -> bool {
    compatible
        .split(|&b| b == 0)
        .any(|c| c.starts_with(b"nvidia,tegra"))
}

/// A `/proc/<pid>` poller on the process, at the cadence of its [`Watch`].
pub struct Sampler {
    pid: u32,
    reads: Reads,
    /// When [`Sampler::start`] ran: the swap times' and the stop's `window_ms`
    /// origin.
    started: Instant,
    /// Dropping it ends the thread within one interval.
    stop: mpsc::Sender<()>,
    thread: JoinHandle<Seen>,
}

impl Sampler {
    pub(crate) const OBSERVATION: Reads = Reads::Observation;

    /// What this sampler reads for `watch`, or why it has no reader for it.
    /// Neither OS reads a figure that is an NPU's alone, which an NPU watch
    /// requires. Linux reads no NPU, so none of whatever devices a run holds
    /// memory on (the KGSL reader is for Android's Adreno driver), and Android
    /// has no DRM or NVML reader.
    pub(crate) fn reads(watch: Watch) -> Result<Reads, WatchError> {
        match watch {
            Watch::Observe => Ok(Reads::Observation),
            Watch::Peak => Ok(Reads::Host),
            #[cfg(target_os = "linux")]
            Watch::PeakWithGpu if gpu_memory_is_resident()? => Ok(Reads::Host),
            #[cfg(target_os = "linux")]
            Watch::PeakWithGpu => Ok(Reads::Gpu),
            #[cfg(target_os = "android")]
            Watch::PeakWithAccelerators => Ok(Reads::Mapped),
            #[cfg(target_os = "linux")]
            Watch::PeakWithNpu | Watch::PeakWithAccelerators => {
                Err(UnsupportedError { watch }.into())
            }
            #[cfg(target_os = "android")]
            Watch::PeakWithGpu | Watch::PeakWithNpu => Err(UnsupportedError { watch }.into()),
        }
    }

    pub fn start(pid: u32, SupportedWatch(reads): SupportedWatch) -> Result<Self, StartError> {
        let interval = if reads == Reads::Observation {
            OBSERVE_INTERVAL
        } else {
            PEAK_INTERVAL
        };
        // A caller that reaps before stopping leaves a window where this pid can be
        // recycled. Pinning the process start time closes it: a different process
        // wearing the same pid is skipped rather than folded into the peak. From
        // `stat`, which a child that has already exited still has.
        let identity = read_stat(pid)
            .map(|stat| stat.start_time)
            .filter(|&start_time| start_time != 0);
        let started = Instant::now();
        // After the identity: opening NVML can take a second.
        let mut devices = DeviceReader::start(reads);
        let (stop, stopped) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            // Accumulated in thread-local state and handed back through `join`:
            // one writer, so the sampling path needs neither atomics nor a lock.
            let mut seen = Seen::default();
            // Reads after each wait, the one a stop ends too, so the last read is
            // taken at the stop.
            loop {
                let stopping = match stopped.recv_timeout(interval) {
                    Err(RecvTimeoutError::Timeout) => false,
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
                };
                // Before the status, so its start time vouches for this read too.
                let read = devices.read(pid);
                let landed = if let Some(sample) = read_status_sample(pid)
                    && same_process(identity, sample.start_time)
                {
                    seen.observe(sample, u64::try_from(started.elapsed().as_millis()).ok());
                    if let Some(read) = read {
                        seen.accelerators = Some(read.max(seen.accelerators.unwrap_or_default()));
                    }
                    true
                } else if stopping && let Some(exit_kib) = exit_peak_rss_kib(pid) {
                    // Exited and unreaped, so the stop came first, whatever its
                    // figure is worth.
                    if clears_own_peak(exit_kib, own_hwm_kib()) {
                        seen.footprint.peak_rss_kib = seen.footprint.peak_rss_kib.max(exit_kib);
                    }
                    true
                } else {
                    false
                };
                if stopping {
                    seen.read_at_stop = landed;
                    #[cfg(target_os = "linux")]
                    {
                        seen.nvml_unavailable = devices.nvml_unavailable();
                    }
                    break seen;
                }
            }
        });
        Ok(Self {
            pid,
            reads,
            started,
            stop,
            thread,
        })
    }

    /// The run's observation, and its resident peak when watched for one. Its
    /// last read needs the process running, or exited and not yet reaped by the
    /// caller, whose child it must be: one already seen to exit
    /// (`Process::is_running`, `wait`, `wait_ready`) has been reaped, and its
    /// peak is refused.
    pub fn stop(self) -> (MemoryObservation, Result<Peak, PeakError>) {
        // Fails only when the thread already exited, which `join` reports.
        let _ = self.stop.send(());
        // A thread that panicked yields the default, whose zero peak fails below.
        let seen = self.thread.join().unwrap_or_default();
        // When this pid's swap rose: the row carries how much, not when.
        tracing::info!(
            pid = self.pid,
            window_ms = u64::try_from(self.started.elapsed().as_millis()).ok(),
            first_swap_ms = seen.first_swap_ms,
            peak_swap_ms = seen.peak_swap_ms,
            "memory probe stopped"
        );
        (observation(&seen), peak(self.reads, self.pid, seen))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IdleChild;
    #[cfg(target_os = "linux")]
    use crate::drm::Unread;

    /// A Snapdragon NPU run's `maps`, abridged: a dmabuf it holds open, one it
    /// only maps, two Adreno mappings, a library whose name says dmabuf, and
    /// anonymous memory.
    const MAPS: &str = "\
7c00a00000-7c11e00000 rw-s 00000000 00:0a 132275                         /dmabuf:system
7c13600000-7c13800000 rw-s 00000000 00:0a 132274                         /dmabuf:system
7c52bb1000-7c52cb1000 rw-s 00000000 00:13 1885                           /dev/kgsl-3d0
7c52cb1000-7c52db1000 rw-s 00000000 00:13 1885                           /dev/kgsl-3d0
7c5c908000-7c5c911000 r--p 00000000 fe:0e 145709821                      /system/lib64/libdmabufheap.so
7bfc960000-7bfc964000 rw-p 00000000 00:00 0
";

    #[rstest::rstest]
    #[case::kgsl_and_dmabufs(
        MAPS,
        vec![(132_275, 289_406_976)],
        Accelerators::new(2 << 20, 289_406_976 + (2 << 20)),
    )]
    #[case::a_dmabuf_open_twice_counts_once(
        "",
        vec![(7, 4096), (7, 4096)],
        Accelerators::new(0, 4096),
    )]
    #[case::a_dmabuf_mapped_twice_counts_once(
        "7c13600000-7c13800000 rw-s 00000000 00:0a 7 /dmabuf:system\n\
         7c13a00000-7c13c00000 rw-s 00000000 00:0a 7 /dmabuf:system\n",
        vec![],
        Accelerators::new(0, 2 << 20),
    )]
    fn accelerators_count_each_buffer_once(
        #[case] maps: &str,
        #[case] open: Vec<(u64, u64)>,
        #[case] expected: Accelerators,
    ) {
        assert_eq!(accelerators(maps, &open), expected);
    }

    /// Android's GPU and NPU read, on Linux run only here: a live process on
    /// no device holds no KGSL mapping and no dmabuf.
    #[test]
    fn a_live_process_holds_no_kgsl_memory_or_dmabufs() -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        assert_eq!(read_accelerators(child.id()), Some(Accelerators::default()));
        Ok(())
    }

    /// The first read with any swap, and the last that raised its maximum; a
    /// fall and a repeat of the peak move neither.
    #[test]
    fn swap_times_mark_the_first_and_the_peak_rise() {
        let mut seen = Seen::default();
        [(10, 0), (20, 512), (30, 256), (40, 1024), (50, 1024)]
            .into_iter()
            .for_each(|(at, swap_kib)| {
                let sample = StatusSample {
                    swap_kib,
                    ..StatusSample::default()
                };
                seen.observe(sample, Some(at));
            });
        assert_eq!(seen.first_swap_ms, Some(20));
        assert_eq!(seen.peak_swap_ms, Some(40));
    }

    /// An fdinfo without `ino:` gives nothing, so the read fails rather than
    /// dropping the buffer from the count.
    #[rstest::rstest]
    #[case::current_kernel(
        "pos:\t0\nflags:\t02000002\nmnt_id:\t9\nino:\t132275\n\
         size:\t289406976\ncount:\t3\nexp_name:\tsystem\nname:\tsystem\n",
        Some((132_275, 289_406_976))
    )]
    #[case::no_inode(
        "pos:\t0\nflags:\t02000002\nmnt_id:\t9\nsize:\t289406976\nexp_name:\tsystem\n",
        None
    )]
    fn a_dmabuf_fdinfo_gives_its_inode_and_size(
        #[case] fdinfo: &str,
        #[case] expected: Option<(u64, u64)>,
    ) {
        assert_eq!(parse_dmabuf_fdinfo(fdinfo), expected);
    }

    #[rstest::rstest]
    #[case::observed(Reads::Observation, 6_000, None, Err("not its peak"))]
    #[case::host(Reads::Host, 6_000, None, Ok((None, None)))]
    #[case::no_vmhwm(Reads::Mapped, 0, Some(Accelerators::default()), Err("VmHWM"))]
    #[case::accelerators_never_read(Reads::Mapped, 6_000, None, Err("/proc/42/maps"))]
    #[case::accelerators_held_none(
        Reads::Mapped,
        6_000,
        Some(Accelerators::default()),
        Ok((None, None))
    )]
    #[case::gpu_without_npu(
        Reads::Mapped,
        6_000,
        Some(Accelerators::new(4096, 0)),
        Ok((Some(4096), None))
    )]
    fn a_peak_has_every_pool_its_watch_reads(
        #[case] reads: Reads,
        #[case] peak_rss_kib: u64,
        #[case] accelerators: Option<Accelerators>,
        #[case] expected: Result<(Option<u64>, Option<u64>), &str>,
    ) -> anyhow::Result<()> {
        let seen = Seen {
            footprint: Footprint {
                peak_rss_kib,
                ..Footprint::default()
            },
            read_at_stop: true,
            accelerators,
            #[cfg(target_os = "linux")]
            nvml_unavailable: None,
            ..Seen::default()
        };
        expect_pools(peak(reads, 42, seen), expected)
    }

    fn expect_pools(
        peak: Result<Peak, PeakError>,
        expected: Result<(Option<u64>, Option<u64>), &str>,
    ) -> anyhow::Result<()> {
        match (peak.map_err(anyhow::Error::from), expected) {
            (Ok(peak), Ok(pools)) => assert_eq!((peak.gpu_bytes, peak.npu_bytes), pools),
            (Err(err), Err(cause)) => assert!(format!("{err:#}").contains(cause), "{err:#}"),
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn on(drm: Result<u64, Unread>, on_nvidia: bool) -> Fds {
        Fds { drm, on_nvidia }
    }

    #[cfg(target_os = "linux")]
    fn read(gpu_bytes: u64, drm_unread: bool, on_nvidia: bool, nvml: NvmlRead) -> Accelerators {
        Accelerators {
            gpu_bytes,
            npu_bytes: 0,
            gpu: GpuRead {
                drm_unread,
                on_nvidia,
                nvml,
            },
        }
    }

    /// NVML's processes on one read, or why there are none.
    #[cfg(target_os = "linux")]
    type NvmlProcesses = Result<Vec<(u32, Result<u64, NotAvailable>)>, NvmlReadError>;

    #[cfg(target_os = "linux")]
    #[rstest::rstest]
    #[case::drm_alone(on(Ok(500), false), None, read(500, false, false, NvmlRead::Closed))]
    #[case::both(
        on(Ok(500), true),
        Some(Ok(vec![(7, Ok(70)), (8, Ok(9))])),
        read(570, false, true, NvmlRead::Answered)
    )]
    #[case::not_listed(
        on(Ok(500), true),
        Some(Ok(vec![(8, Ok(9))])),
        read(500, false, true, NvmlRead::Answered)
    )]
    #[case::nvml_unanswered(
        on(Ok(500), false),
        Some(Err(NvmlReadError::Unanswered)),
        read(500, false, false, NvmlRead::Unanswered)
    )]
    #[case::nvml_not_available(
        on(Ok(0), true),
        Some(Ok(vec![(7, Err(NotAvailable)), (8, Ok(9))])),
        read(0, false, true, NvmlRead::NotAvailable)
    )]
    #[case::drm_unread(on(Err(Unread), false), None, read(0, true, false, NvmlRead::Closed))]
    fn a_gpu_read_adds_drm_and_nvml(
        #[case] fds: Fds,
        #[case] nvml: Option<NvmlProcesses>,
        #[case] expected: Accelerators,
    ) {
        let nvml = nvml.map(|read| read.map(|pids| pids.into_iter().collect()));
        assert_eq!(gpu_read(fds, 7, nvml), expected);
    }

    #[cfg(target_os = "linux")]
    fn no_nvml() -> Option<NvmlError> {
        Some(NvmlError::Load("libnvidia-ml.so.1: cannot open".to_owned()))
    }

    #[cfg(target_os = "linux")]
    #[rstest::rstest]
    #[case::never_read(None, None, Err("could not list /proc/42/fd"))]
    #[case::drm_unread(
        Some(read(4096, true, false, NvmlRead::Answered)),
        None,
        Err("amdgpu before 5.14 prints nothing")
    )]
    #[case::not_available(
        Some(read(4096, false, true, NvmlRead::NotAvailable)),
        None,
        Err("as not available")
    )]
    #[case::nvidia_without_nvml(
        Some(read(4096, false, true, NvmlRead::Closed)),
        no_nvml(),
        Err("NVML is unavailable: dlopen")
    )]
    #[case::nvidia_unanswered(
        Some(read(4096, false, true, NvmlRead::Unanswered)),
        None,
        Err("a GPU refused one")
    )]
    #[case::nvidia_held_none(
        Some(read(0, false, true, NvmlRead::Answered)),
        None,
        Err("NVML attributed it no memory")
    )]
    #[case::drm_held_none(
        Some(read(0, false, false, NvmlRead::Closed)),
        no_nvml(),
        Err("no DRM client of pid 42")
    )]
    #[case::drm_without_nvml(
        Some(read(4096, false, false, NvmlRead::Closed)),
        no_nvml(),
        Ok((Some(4096), None))
    )]
    #[case::drm_while_nvml_is_unanswered(
        Some(read(4096, false, false, NvmlRead::Unanswered)),
        None,
        Ok((Some(4096), None))
    )]
    #[case::nvidia(
        Some(read(4096, false, true, NvmlRead::Answered)),
        None,
        Ok((Some(4096), None))
    )]
    fn a_gpu_peak_needs_every_gpu_it_holds_read(
        #[case] accelerators: Option<Accelerators>,
        #[case] nvml_unavailable: Option<NvmlError>,
        #[case] expected: Result<(Option<u64>, Option<u64>), &str>,
    ) -> anyhow::Result<()> {
        let seen = Seen {
            footprint: Footprint {
                peak_rss_kib: 6_000,
                ..Footprint::default()
            },
            read_at_stop: true,
            accelerators,
            nvml_unavailable,
            ..Seen::default()
        };
        expect_pools(peak(Reads::Gpu, 42, seen), expected)
    }

    /// Reads keep the highest-ranked of what NVML said ([`NvmlRead`]'s order),
    /// and whether any read saw an NVIDIA device.
    #[cfg(target_os = "linux")]
    #[test]
    fn reads_keep_what_any_one_said() {
        let reads = [
            read(10, false, false, NvmlRead::Unanswered),
            read(30, false, true, NvmlRead::Answered),
            read(20, false, false, NvmlRead::Unanswered),
        ];
        let kept = reads.into_iter().reduce(Accelerators::max);
        assert_eq!(kept, Some(read(30, false, true, NvmlRead::Answered)));
    }

    /// Reads before the stop are no peak: one in the last interval can be
    /// missing.
    #[rstest::rstest]
    #[case::nothing_read(0, false, Err("could not read VmHWM"))]
    #[case::unread_at_the_stop(3_000, false, Err("was not read when the sampler stopped"))]
    #[case::read_at_the_stop(6_000, true, Ok((None, None)))]
    fn a_peak_needs_its_read_at_the_stop(
        #[case] peak_rss_kib: u64,
        #[case] read_at_stop: bool,
        #[case] expected: Result<(Option<u64>, Option<u64>), &str>,
    ) -> anyhow::Result<()> {
        let seen = Seen {
            footprint: Footprint {
                peak_rss_kib,
                ..Footprint::default()
            },
            read_at_stop,
            ..Seen::default()
        };
        expect_pools(peak(Reads::Host, 42, seen), expected)
    }

    /// An exit figure counts only above pipette's own peak by the margin: one
    /// at or below it may be pipette's image.
    #[rstest::rstest]
    #[case::pipettes_image(300_000, Some(300_000), false)]
    #[case::within_the_margin(300_000 + OWN_PEAK_MARGIN_KIB, Some(300_000), false)]
    #[case::past_the_margin(300_001 + OWN_PEAK_MARGIN_KIB, Some(300_000), true)]
    #[case::below_pipette(64_000, Some(300_000), false)]
    #[case::pipettes_peak_unread(300_000, None, false)]
    fn an_exit_figure_must_clear_pipettes_own_peak(
        #[case] exit_kib: u64,
        #[case] own_hwm_kib: Option<u64>,
        #[case] expected: bool,
    ) {
        assert_eq!(clears_own_peak(exit_kib, own_hwm_kib), expected);
    }

    /// An exited child keeps its `/proc` entry until it is reaped, without
    /// `VmHWM`; `waitid` still reports the peak, and stops once it is reaped.
    #[test]
    fn an_exited_child_is_read_until_it_is_reaped() -> anyhow::Result<()> {
        let mut child = crate::exiting_child::spawn(1)?;
        let pid = child.id();
        anyhow::ensure!(read_status_sample(pid).is_some(), "a live child reads");
        anyhow::ensure!(exit_peak_rss_kib(pid).is_none(), "a live child has no exit");
        crate::exiting_child::release(&mut child)?;
        crate::exiting_child::wait_unreaped(pid)?;
        anyhow::ensure!(
            read_status_sample(pid).is_none(),
            "an exited child has no VmHWM"
        );
        anyhow::ensure!(read_stat(pid).is_some(), "a zombie keeps its stat");
        let peak_kib = exit_peak_rss_kib(pid).unwrap_or_default();
        anyhow::ensure!(
            peak_kib >= crate::exiting_child::ALLOCATED >> 10,
            "exit peak {peak_kib} KiB"
        );
        child.wait()?;
        anyhow::ensure!(exit_peak_rss_kib(pid).is_none(), "a reaped child is gone");
        Ok(())
    }

    #[test]
    fn dropping_a_sampler_ends_its_thread() -> anyhow::Result<()> {
        let child = IdleChild::spawn()?;
        let Sampler { stop, thread, .. } = Sampler::start(child.id(), Watch::Peak.supported()?)?;
        drop(stop);

        let exited = (0..200).any(|_| {
            if thread.is_finished() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
            false
        });
        assert!(exited, "the sampling thread outlived its sender");
        Ok(())
    }

    #[test]
    fn parses_stat_fields_past_a_hostile_comm() {
        let stat = concat!(
            "42 (llama )bench 1) S 1 42 42 0 -1 4194304 1811767 0 13913 0 ",
            "55 3 0 0 20 0 4 0 987654 1",
        );
        assert_eq!(
            parse_stat(stat),
            Some(Stat {
                major_faults: 13_913,
                start_time: 987_654
            })
        );
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
    // exactly what a resident-only counter does.
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

    /// What reaches a row from a footprint: the resident peak, only once the
    /// read at the stop has landed, and never half an observation.
    ///
    /// Asserts the whole observation rather than the peak alone, since a
    /// withheld peak has to take the swap term with it.
    #[rstest::rstest]
    #[case::unread_at_the_stop_is_withheld(3_000, 0, 0, false, MemoryObservation::default())]
    #[case::nothing_read_is_withheld(0, 0, 0, true, MemoryObservation::default())]
    #[case::read_at_the_stop_is_reported(
        6_440_000, 0, 0, true,
        MemoryObservation::with_swap(6_440_000 * 1024, 0)
    )]
    // A measured S26 Ultra footprint: `VmHWM` 5004 MiB while zram held up to
    // 4105 MiB, and `VmRSS + VmSwap` peaked at the 6138 MiB the Android metric
    // scores. Only the resident term may appear here, or the field means one
    // thing on a host that swapped and another on one that did not.
    #[case::swapping_reports_the_resident_term(
        5_124_688, 6_285_312, 4_203_520, true,
        MemoryObservation::with_swap(5_124_688 * 1024, 4_203_520 * 1024)
    )]
    fn an_observation_reports_the_resident_peak_read_at_the_stop(
        #[case] peak_rss_kib: u64,
        #[case] peak_committed_kib: u64,
        #[case] max_swap_kib: u64,
        #[case] read_at_stop: bool,
        #[case] expected: MemoryObservation,
    ) {
        let seen = Seen {
            footprint: Footprint {
                peak_rss_kib,
                peak_committed_kib,
                max_swap_kib,
                ..Default::default()
            },
            read_at_stop,
            ..Seen::default()
        };
        assert_eq!(observation(&seen), expected);
    }

    #[cfg(target_os = "linux")]
    #[rstest::rstest]
    #[case::agx_orin(
        b"nvidia,p3737-0000+p3701-0005\0nvidia,p3701-0005\0nvidia,tegra234\0",
        true
    )]
    #[case::thor(b"nvidia,tegra264\0", true)]
    #[case::pi5(b"raspberrypi,5-model-b\0brcm,bcm2712\0", false)]
    #[case::no_device_tree(b"", false)]
    fn a_tegra_device_tree_is_a_resident_gpu(#[case] compatible: &[u8], #[case] tegra: bool) {
        assert_eq!(is_tegra(compatible), tegra);
    }
}
