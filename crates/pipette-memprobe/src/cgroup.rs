//! A cgroup's memory, for a server that runs as several processes, such as a
//! container or a uv server, rather than as one pid ([`crate::Sampler`]).
//! Read on the host from its cgroup v2 tree, on Linux only: elsewhere
//! [`Tree::open`] refuses. One thread reads both figures on one tick, from
//! when the caller's `locate` names the cgroup, and once more at the stop:
//!
//! - **Host:** `anon + shmem + kernel` of the cgroup's `memory.stat`, counted
//!   once over every process in it, keeping the largest read, since the kernel
//!   keeps no watermark for that sum. The page cache is left out: the kernel
//!   bills a cached page to the first cgroup to read it, so what a cgroup is
//!   billed for its libraries and the model's files follows what ran on the
//!   host before it, not the server. A cgroup the server shares
//!   ([`Members::Session`]) has its other processes' own memory taken off.
//! - **GPU (NVIDIA, Linux):** NVML's figure for each of the server's processes
//!   whose `/proc/<pid>/cgroup` is the cgroup or one below it, summed, keeping
//!   the maximum, since NVML keeps no watermark. NVML lists the host's pids,
//!   which the host's `/proc` places. A root container denies the fd scan the
//!   pid sampler reads DRM through, so NVML is the one source.
//! - **GPU (AMD, Linux):** the rise of the cards a scheduler such as Slurm
//!   gave this job alone over their floor, keeping the maximum
//!   (`amd::AmdGpu`, on Linux).
//!
//! A peak reads every [`PEAK_TICK`], an observation every [`OBSERVE_TICK`].

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use pipette_plan_types::MemoryObservation;

#[cfg(target_os = "linux")]
use crate::amd::{AmdGpu, AmdGpuError};
#[cfg(target_os = "linux")]
use crate::nvml::{NotAvailable, Nvml, Pids};
#[cfg(target_os = "linux")]
use crate::{NvmlError, NvmlReadError};

/// A peak's tick, as the pid sampler's: it favours accuracy over the read's
/// cost, about 11 µs for a `memory.stat`.
const PEAK_TICK: Duration = Duration::from_millis(10);

/// An observation's tick: it rides on a run that reports time, which watching
/// must not move.
const OBSERVE_TICK: Duration = Duration::from_millis(100);

/// The cgroup's own memory in `memory.stat`: anonymous, shared (tmpfs and shm
/// segments, which `file` also counts) and the kernel's.
const OWN_KEYS: [&str; 3] = ["anon", "shmem", "kernel"];

/// The `memory.stat` keys [`Peak::memory_stat`] keeps: the cgroup's own
/// memory's and the page cache's (`file`), so a row shows the cache it ran
/// with.
const MEMORY_STAT_KEYS: [&str; 4] = ["anon", "file", "shmem", "kernel"];

/// Why the host's cgroup v2 tree cannot count a cgroup's memory.
#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    #[error("a cgroup's memory is read from the host's cgroup v2 tree, which only Linux has")]
    NotLinux,
    #[error(
        "could not read {path}: the host mounts no cgroup v2 tree there (cgroup v1, or a hybrid \
         layout)"
    )]
    NotV2 {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path} lists no `memory` controller, so the cgroups below it count no memory")]
    NoMemoryController { path: PathBuf },
    #[error(
        "could not read {path}: the tree pipette reads does not hold the cgroup (pipette runs in \
         a container of its own)"
    )]
    NotHeld {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not read {path}, which names pipette's own cgroup")]
    OwnUnread {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path} names no cgroup v2 path for pipette")]
    OwnNotV2 { path: PathBuf },
    #[error(
        "{path} is missing, so pipette's own cgroup counts no memory of its own: pipette runs in \
         the host's root cgroup, which holds every process on the host, or in one whose parent \
         hands it no memory controller. Run it from a login session, a service or a job, whose \
         cgroup holds it alone"
    )]
    OwnUncounted { path: PathBuf },
    #[error(
        "{path} lists processes in a cgroup below pipette's own, which the probe cannot tell \
         apart from the server's"
    )]
    OwnHasChildren { path: PathBuf },
}

/// Why a [`Sampler`] has no [`Peak`]. `E` is why its `locate` named no
/// cgroup.
#[derive(Debug, thiserror::Error)]
pub enum PeakError<E> {
    #[error("the cgroup to read was never found, so its memory is unknown")]
    Unlocated(#[source] E),
    #[error(
        "could not read {path}: the tree pipette reads does not hold the cgroup (pipette runs in \
         a container of its own), or the cgroup counts no memory"
    )]
    Unreadable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path} has no `{key}` count; Linux 5.18 or later prints it")]
    NoKey { path: PathBuf, key: &'static str },
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Nvidia(#[from] NvidiaGpuError),
    #[cfg(target_os = "linux")]
    #[error(transparent)]
    Amd(#[from] AmdGpuError),
    #[error("the cgroup sampler panicked")]
    Panicked,
}

/// Why a cgroup on an NVIDIA GPU has no GPU peak.
#[cfg(target_os = "linux")]
#[derive(Debug, thiserror::Error)]
pub enum NvidiaGpuError {
    #[error("NVML, which reads the cgroup's NVIDIA GPU memory, is unavailable")]
    Nvml(#[source] NvmlError),
    #[error(
        "on every read during the run a GPU refused one of NVML's process lists, so the \
         cgroup's GPU memory is unknown"
    )]
    Unanswered,
    #[error(
        "NVML reports the GPU memory of the cgroup's pid {pid} as not available (MIG, vGPU, or \
         no permission)"
    )]
    NotAvailable { pid: u32 },
    #[error(
        "NVML attributed no GPU memory to a process in the cgroup during the run: the server \
         never reached the GPU, pipette runs in another pid namespace than the server, the \
         cgroup is not the server's, or /proc hides other users' processes (hidepid)"
    )]
    NoMemory,
}

/// The host's `/proc` and cgroup v2 tree, which a [`Sampler`] reads through.
#[derive(Debug)]
pub struct Tree {
    proc: PathBuf,
    root: PathBuf,
}

impl Tree {
    /// `/proc` and `/sys/fs/cgroup`. Refused off Linux, and on a host whose
    /// tree counts no memory; a tree that does not hold the cgroup fails the
    /// sampler once `locate` names it.
    pub fn open() -> Result<Self, TreeError> {
        if cfg!(target_os = "linux") {
            Self::at(PathBuf::from("/proc"), PathBuf::from("/sys/fs/cgroup"))
        } else {
            Err(TreeError::NotLinux)
        }
    }

    /// `proc`, and the tree at `root` where its root lists the memory
    /// controller: a test's stand-in.
    pub fn at(proc: PathBuf, root: PathBuf) -> Result<Self, TreeError> {
        let path = root.join("cgroup.controllers");
        match std::fs::read_to_string(&path) {
            Ok(controllers) if lists_memory(&controllers) => Ok(Self { proc, root }),
            Ok(_) => Err(TreeError::NoMemoryController { path }),
            Err(source) => Err(TreeError::NotV2 { path, source }),
        }
    }

    /// `/proc/<pid>/cgroup`, which names the cgroups `pid` is in.
    pub fn cgroups_of(&self, pid: u32) -> PathBuf {
        self.proc.join(pid.to_string()).join("cgroup")
    }

    /// `/proc/self/cgroup`, which names the cgroups of the process reading it.
    pub fn own_cgroups(&self) -> PathBuf {
        self.proc.join("self").join("cgroup")
    }

    /// The cgroup pipette runs in, which a child it starts shares, when it
    /// counts memory of its own: not the host's root cgroup, which holds every
    /// process on the host and has no `memory.current`. Inside a cgroup
    /// namespace its root is pipette's own cgroup, a job's or a container's,
    /// which does, unless the host's whole tree is mounted there: that reads
    /// as the host's root and is refused.
    pub fn own_cgroup(&self) -> Result<Cgroup, TreeError> {
        let path = self.own_cgroups();
        let cgroups = match std::fs::read_to_string(&path) {
            Ok(cgroups) => cgroups,
            Err(source) => return Err(TreeError::OwnUnread { path, source }),
        };
        let own = v2_paths(&cgroups)
            .next()
            .map(|path| Cgroup(path.trim_end_matches('/').to_owned()))
            .ok_or(TreeError::OwnNotV2 { path })?;
        let current = self.dir(&own).join("memory.current");
        if !current.is_file() {
            return Err(TreeError::OwnUncounted { path: current });
        }
        // A container's cgroup namespace can name its root while pipette sits
        // in a cgroup below it, as pyxis's `task_0`: the one listing pipette
        // is its own.
        let own = match lists_below(&self.dir(&own), std::process::id()) {
            Some(below) => Cgroup(format!("{}/{below}", own.0)),
            None => own,
        };
        match populated_below(&self.dir(&own)) {
            Some(path) => Err(TreeError::OwnHasChildren { path }),
            None => Ok(own),
        }
    }

    fn ids_of(&self, pid: u32) -> Option<ProcessIds> {
        let stat = std::fs::read_to_string(self.proc.join(pid.to_string()).join("stat")).ok()?;
        // After the command name, which can hold `)`: state, ppid, pgrp,
        // session.
        let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
        Some(ProcessIds {
            parent: fields.get(1)?.parse().ok()?,
            session: fields.get(3)?.parse().ok()?,
        })
    }

    /// The own memory of `pid` its `/proc/<pid>/status` gives; none once it
    /// is gone.
    fn own_memory_of(&self, pid: u32) -> OwnMemory {
        let status = std::fs::read_to_string(self.proc.join(pid.to_string()).join("status"))
            .unwrap_or_default();
        let bytes = |keys: &[&str]| {
            status
                .lines()
                .filter_map(|line| line.split_once(':'))
                .filter(|(key, _)| keys.contains(key))
                .filter_map(|(_, value)| {
                    value.trim().strip_suffix(" kB")?.trim().parse::<u64>().ok()
                })
                .fold(0u64, |sum, kib| {
                    sum.saturating_add(kib.saturating_mul(1024))
                })
        };
        OwnMemory {
            resident: bytes(&["RssAnon", "RssShmem", "VmPTE"]),
            swapped: bytes(&["VmSwap"]),
        }
    }

    /// Refuses `cgroup` when it cannot count its memory: its parent's
    /// `cgroup.controllers` lists no memory controller to hand down. Read
    /// from that file because it stays once `cgroup` is gone, while the
    /// parent's `cgroup.subtree_control` may drop the controller with it.
    pub fn require_memory_for(&self, cgroup: &Cgroup) -> Result<(), TreeError> {
        let parent = cgroup.0.rsplit_once('/').map_or("", |(parent, _)| parent);
        let path = self
            .dir(&Cgroup(parent.to_owned()))
            .join("cgroup.controllers");
        match std::fs::read_to_string(&path) {
            Ok(controllers) if lists_memory(&controllers) => Ok(()),
            Ok(_) => Err(TreeError::NoMemoryController { path }),
            Err(source) => Err(TreeError::NotHeld { path, source }),
        }
    }

    /// `cgroup`'s directory in the tree.
    fn dir(&self, cgroup: &Cgroup) -> PathBuf {
        cgroup
            .0
            .split('/')
            .filter(|part| !part.is_empty())
            .fold(self.root.clone(), |dir, part| dir.join(part))
    }
}

/// Whether a `cgroup.controllers` lists the memory controller.
fn lists_memory(controllers: &str) -> bool {
    controllers.split_whitespace().any(|name| name == "memory")
}

/// A cgroup, by its path in the tree as `/proc/<pid>/cgroup` prints it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cgroup(String);

impl Cgroup {
    /// The cgroup a `/proc/<pid>/cgroup` places its process in: the cgroup v2
    /// path through its first part `names_it` accepts, so a process in a cgroup
    /// below it gives it too. `None` when no part is accepted, as under cgroup
    /// v1.
    pub fn named(cgroups: &str, names_it: impl Fn(&str) -> bool) -> Option<Self> {
        v2_paths(cgroups).find_map(|path| {
            path.split('/')
                .scan(0, |end: &mut usize, part| {
                    *end = end.saturating_add(part.len()).saturating_add(1);
                    Some((end.saturating_sub(1), part))
                })
                .find(|&(_, part)| !part.is_empty() && names_it(part))
                .and_then(|(end, _)| Some(Self(path.get(..end)?.to_owned())))
        })
    }

    /// Its path in the tree, as `/proc/<pid>/cgroup` prints it.
    pub fn path(&self) -> &str {
        &self.0
    }

    /// Whether a `/proc/<pid>/cgroup` places its process in this cgroup or
    /// below it.
    #[cfg(target_os = "linux")]
    fn holds(&self, cgroups: &str) -> bool {
        v2_paths(cgroups).any(|path| {
            path.strip_prefix(self.0.as_str())
                .is_some_and(|below| below.is_empty() || below.starts_with('/'))
        })
    }
}

/// The cgroup v2 paths of a `/proc/<pid>/cgroup`: one, unless the host runs
/// cgroup v1 alone.
fn v2_paths(cgroups: &str) -> impl Iterator<Item = &str> {
    cgroups.lines().filter_map(|line| line.strip_prefix("0::"))
}

/// Whose memory in the cgroup a [`Sampler`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Members {
    /// Every process in it: a cgroup of the server's own, a container's.
    All,
    /// Session `leader`'s processes and what they start in a session of their
    /// own: a server in pipette's own cgroup ([`Tree::own_cgroup`]). The
    /// cgroup's other processes have their own memory taken off the host
    /// figure and are left out of the GPU's.
    Session { leader: u32 },
}

struct ProcessIds {
    parent: u32,
    session: u32,
}

/// A process's own memory, or several processes' summed.
#[derive(Debug, Default, Clone, Copy)]
struct OwnMemory {
    /// `RssAnon + RssShmem + VmPTE`: what the cgroup counts of it, its page
    /// tables the one part of the kernel's a process reports.
    resident: u64,
    /// `VmSwap`.
    swapped: u64,
}

impl OwnMemory {
    fn plus(self, other: Self) -> Self {
        Self {
            resident: self.resident.saturating_add(other.resident),
            swapped: self.swapped.saturating_add(other.swapped),
        }
    }

    /// The sum over the processes either read lists, each at its larger
    /// figure.
    fn larger_of(before: BTreeMap<u32, Self>, after: BTreeMap<u32, Self>) -> Self {
        let mut larger = before;
        for (pid, other) in after {
            let one = larger.entry(pid).or_default();
            one.resident = one.resident.max(other.resident);
            one.swapped = one.swapped.max(other.swapped);
        }
        larger.into_values().fold(Self::default(), Self::plus)
    }
}

/// Which of a shared cgroup's processes are the server's, each decided once.
/// It reads only the cgroup's own `cgroup.procs`, so it assumes no child
/// cgroups, as a login session or a job has none.
struct Split {
    leader: u32,
    /// This process, pipette, which is in the cgroup whether or not its
    /// `cgroup.procs` lists it: a pid namespace can hide every pid there.
    own: u32,
    server: BTreeSet<u32>,
    other: BTreeSet<u32>,
    /// Whether a read found `cgroup.procs` empty, as a pid namespace shows
    /// it, so only pipette's own memory came off.
    unlisted: bool,
}

impl Split {
    fn new(leader: u32, own: u32) -> Self {
        Self {
            leader,
            own,
            server: BTreeSet::new(),
            other: BTreeSet::new(),
            unlisted: false,
        }
    }

    /// In the leader's session, or below a process that is. pipette, a gone
    /// process and init are not.
    fn counts(&mut self, tree: &Tree, pid: u32) -> bool {
        let mut walked = Vec::new();
        let mut at = pid;
        let counts = loop {
            if at == self.leader || self.server.contains(&at) {
                break true;
            }
            if at == self.own || at <= 1 || self.other.contains(&at) || walked.contains(&at) {
                break false;
            }
            walked.push(at);
            match tree.ids_of(at) {
                Some(ids) if ids.session == self.leader => break true,
                Some(ids) => at = ids.parent,
                None => break false,
            }
        };
        let decided = if counts {
            &mut self.server
        } else {
            &mut self.other
        };
        decided.extend(walked);
        counts
    }

    /// The own memory of each process in the cgroup at `dir` that is not the
    /// server's, pipette's always included.
    fn others(&mut self, tree: &Tree, dir: &Path) -> BTreeMap<u32, OwnMemory> {
        let listed: Vec<u32> = std::fs::read_to_string(dir.join("cgroup.procs"))
            .unwrap_or_default()
            .split_whitespace()
            .filter_map(|pid| pid.parse().ok())
            .collect();
        self.unlisted |= listed.is_empty();
        // A pid that left the cgroup is decided again if it comes back,
        // handed out to another process.
        if !listed.is_empty() {
            self.server.retain(|pid| listed.contains(pid));
            self.other.retain(|pid| listed.contains(pid));
        }
        let own = (!listed.contains(&self.own)).then_some(self.own);
        listed
            .iter()
            .copied()
            .filter(|&pid| !self.counts(tree, pid))
            .chain(own)
            .map(|pid| (pid, tree.own_memory_of(pid)))
            .collect()
    }
}

/// The path below `dir` of the cgroup whose `cgroup.procs` lists `pid`, when
/// `dir`'s own does not.
fn lists_below(dir: &Path, pid: u32) -> Option<String> {
    let lists = |dir: &Path| {
        std::fs::read_to_string(dir.join("cgroup.procs")).is_ok_and(|procs| {
            procs
                .split_whitespace()
                .any(|listed| listed == pid.to_string())
        })
    };
    if lists(dir) {
        return None;
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .find_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            if lists(&entry.path()) {
                Some(name)
            } else {
                lists_below(&entry.path(), pid).map(|below| format!("{name}/{below}"))
            }
        })
}

/// The first cgroup below `dir` whose `cgroup.procs` lists a process.
fn populated_below(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .find_map(|child| {
            let procs = child.join("cgroup.procs");
            let listed = std::fs::read_to_string(&procs).is_ok_and(|text| !text.trim().is_empty());
            if listed {
                Some(procs)
            } else {
                populated_below(&child)
            }
        })
}

/// What a [`Sampler`] reads, and how often.
pub enum Watch {
    /// The host figure every [`OBSERVE_TICK`], for a run that reports time.
    Observe,
    /// The host figure every [`PEAK_TICK`], for a memory benchmark's result.
    Peak,
    /// [`Self::Peak`] with what NVML attributes to the cgroup's processes.
    #[cfg(target_os = "linux")]
    PeakWithNvidiaGpu(NvidiaGpu),
    /// [`Self::Peak`] with the used VRAM of this job's own AMD GPUs.
    #[cfg(target_os = "linux")]
    PeakWithAmdGpu(AmdGpu),
}

/// NVML for a [`Watch::PeakWithNvidiaGpu`], with what the sampler has read of
/// it.
#[cfg(target_os = "linux")]
pub struct NvidiaGpu {
    read: Box<dyn FnMut() -> Result<Pids, NvmlReadError> + Send>,
    seen: GpuSeen,
}

#[cfg(target_os = "linux")]
impl NvidiaGpu {
    /// Open it before the launch: `nvmlInit` takes over a second on a host
    /// that does not keep the driver loaded.
    pub fn open() -> Result<Self, NvidiaGpuError> {
        let mut nvml = Nvml::open().map_err(NvidiaGpuError::Nvml)?;
        Ok(Self {
            read: Box::new(move || nvml.processes()),
            seen: GpuSeen::default(),
        })
    }

    /// `read` in NVML's place.
    #[cfg(any(test, feature = "test-support"))]
    pub fn stand_in(read: impl FnMut() -> Result<Pids, NvmlReadError> + Send + 'static) -> Self {
        Self {
            read: Box::new(read),
            seen: GpuSeen::default(),
        }
    }

    /// One read of the cgroup's processes.
    fn read(&mut self, tree: &Tree, cgroup: &Cgroup, split: Option<&mut Split>) {
        let read = self.held(tree, cgroup, split);
        self.seen.read(read);
    }

    /// NVML's figures for the processes in `cgroup` now, the server's alone
    /// when `split` says which those are. A pid whose cgroups cannot be read
    /// has exited since NVML listed it.
    fn held(
        &mut self,
        tree: &Tree,
        cgroup: &Cgroup,
        mut split: Option<&mut Split>,
    ) -> Result<Pids, NvmlReadError> {
        let mut processes = (self.read)()?;
        processes.retain(|&pid, _| {
            std::fs::read_to_string(tree.cgroups_of(pid))
                .is_ok_and(|cgroups| cgroup.holds(&cgroups))
                && split.as_mut().is_none_or(|split| split.counts(tree, pid))
        });
        Ok(processes)
    }
}

/// What the sampler read of the cgroup's processes on the GPU.
#[cfg(target_os = "linux")]
#[derive(Debug, Default)]
struct GpuSeen {
    /// The reads every GPU answered.
    samples: u64,
    max_bytes: u64,
    /// The first of the cgroup's pids NVML had no figure for.
    not_available: Option<u32>,
}

#[cfg(target_os = "linux")]
impl GpuSeen {
    fn read(&mut self, read: Result<Pids, NvmlReadError>) {
        match read {
            Ok(processes) => {
                self.samples = self.samples.saturating_add(1);
                let bytes = processes
                    .into_iter()
                    .fold(0u64, |sum, (pid, bytes)| match bytes {
                        Ok(bytes) => sum.saturating_add(bytes),
                        Err(NotAvailable) => {
                            self.not_available.get_or_insert(pid);
                            sum
                        }
                    });
                self.max_bytes = self.max_bytes.max(bytes);
            }
            Err(NvmlReadError::Unanswered) => {}
        }
    }

    fn peak(self) -> Result<GpuPeak, NvidiaGpuError> {
        match self {
            Self {
                not_available: Some(pid),
                ..
            } => Err(NvidiaGpuError::NotAvailable { pid }),
            Self { samples: 0, .. } => Err(NvidiaGpuError::Unanswered),
            Self { max_bytes: 0, .. } => Err(NvidiaGpuError::NoMemory),
            Self {
                max_bytes, samples, ..
            } => Ok(GpuPeak {
                bytes: max_bytes,
                samples,
                source: GpuSource::Processes,
            }),
        }
    }
}

/// What a watch read of the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPeak {
    /// The largest figure read.
    pub bytes: u64,
    /// How many of the reads every GPU answered.
    pub samples: u64,
    pub source: GpuSource,
}

/// Whose memory a GPU figure is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSource {
    /// The server's processes', as NVML attributes it.
    Processes,
    /// Whole cards' above their floor, the least they read from the open to
    /// the stop.
    Cards {
        floor_bytes: u64,
        /// Whether the open saw the cards stop handing back a stopped
        /// server's memory.
        settled: bool,
    },
}

/// What came off the host figure in a cgroup the server shares
/// ([`Members::Session`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedCgroup {
    /// The own memory of the cgroup's processes that are not the server's,
    /// pipette's included, at the read at the stop.
    pub others_bytes: u64,
    /// The largest of those over the reads, as a neighbor can shrink before
    /// the stop.
    pub others_max_bytes: u64,
    /// Whether every read listed the cgroup's processes: an empty list, as a
    /// pid namespace shows it, takes off pipette's own memory alone.
    pub listed: bool,
    /// The host figure of the first read, right after the launch: what the
    /// cgroup held before the server grew, kernel memory from earlier work
    /// included.
    pub host_bytes_at_launch: u64,
}

/// What a [`Sampler`] read.
#[derive(Debug, Clone)]
pub struct Peak {
    /// The largest `anon + shmem + kernel` read.
    pub host_bytes: u64,
    /// How many times `memory.stat` was read.
    pub host_samples: u64,
    /// `None` unless the watch reads a GPU.
    pub gpu: Option<GpuPeak>,
    /// `host_bytes` with the swap beside it where the kernel counts it: the
    /// run's observation.
    pub memory: MemoryObservation,
    /// `memory.peak` at the stop, page cache included. `None` before Linux
    /// 5.19.
    pub memory_peak: Option<u64>,
    /// `anon`, `file`, `shmem` and `kernel` of the read at the stop, those the
    /// kernel prints.
    pub memory_stat: BTreeMap<&'static str, u64>,
    /// `None` when every process counts ([`Members::All`]).
    pub shared: Option<SharedCgroup>,
}

/// Why `locate` names no cgroup.
#[derive(Debug)]
pub enum Unlocated<E> {
    /// Not yet: it is asked again on the next tick, and a stop that comes
    /// first fails with the last.
    Yet(E),
    /// Never: the sampler fails with it at once.
    Failed(E),
}

/// A cgroup's memory, read on a thread of its own. Dropped without
/// [`Self::stop`], it still stops and joins its thread, once a `locate` in
/// progress returns.
pub struct Sampler<E> {
    stop: Sender<()>,
    seen: Receiver<Result<Seen, PeakError<E>>>,
    thread: Option<JoinHandle<()>>,
}

impl<E: Send + 'static> Sampler<E> {
    /// From the server's launch, so the load counts: `locate` is asked every
    /// tick until it names the server's cgroup, which is then read every tick
    /// for the `members` that are the server.
    pub fn start(
        tree: Tree,
        watch: Watch,
        members: Members,
        mut locate: impl FnMut(&Tree) -> Result<Cgroup, Unlocated<E>> + Send + 'static,
    ) -> io::Result<Self> {
        let (stop, stopped) = mpsc::channel();
        let (saw, seen) = mpsc::channel();
        let split = match members {
            Members::All => None,
            Members::Session { leader } => Some(Split::new(leader, std::process::id())),
        };
        let thread = thread::Builder::new()
            .name("pipette-cgroup-memory".to_owned())
            .spawn(move || {
                // Fails only once the sampler is gone, with no one to tell.
                let _ = saw.send(sample(&tree, watch, split, &mut locate, &stopped));
            })?;
        Ok(Self {
            stop,
            seen,
            thread: Some(thread),
        })
    }

    /// Call while the server runs: the last read is taken here.
    pub fn stop(self) -> Result<Peak, PeakError<E>> {
        // Fails only once the thread has ended, which `recv` reports.
        let _ = self.stop.send(());
        self.seen.recv().map_err(|_| PeakError::Panicked)??.peak()
    }
}

impl<E> Drop for Sampler<E> {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            // `stop` reports a panic; a sampler dropped without one has no one
            // to tell.
            let _ = thread.join();
        }
    }
}

/// What a watch reads beside the host.
enum Gpu {
    Unread,
    #[cfg(target_os = "linux")]
    Nvidia(NvidiaGpu),
    #[cfg(target_os = "linux")]
    Amd(AmdGpu),
}

/// Reads once `locate` names the cgroup, every tick, and once more at the
/// stop.
fn sample<E>(
    tree: &Tree,
    watch: Watch,
    split: Option<Split>,
    locate: &mut impl FnMut(&Tree) -> Result<Cgroup, Unlocated<E>>,
    stopped: &Receiver<()>,
) -> Result<Seen, PeakError<E>> {
    let (gpu, tick) = match watch {
        Watch::Observe => (Gpu::Unread, OBSERVE_TICK),
        Watch::Peak => (Gpu::Unread, PEAK_TICK),
        #[cfg(target_os = "linux")]
        Watch::PeakWithNvidiaGpu(gpu) => (Gpu::Nvidia(gpu), PEAK_TICK),
        #[cfg(target_os = "linux")]
        Watch::PeakWithAmdGpu(gpu) => (Gpu::Amd(gpu), PEAK_TICK),
    };
    // A launcher starts the cgroup a moment after it starts.
    let cgroup = loop {
        match locate(tree) {
            Ok(cgroup) => break cgroup,
            Err(Unlocated::Failed(err)) => return Err(PeakError::Unlocated(err)),
            Err(Unlocated::Yet(err)) => {
                if stop_came(stopped, tick) {
                    return Err(PeakError::Unlocated(err));
                }
            }
        }
    };
    let dir = tree.dir(&cgroup);
    let mut seen = Seen::new(gpu, split);
    loop {
        seen.read(tree, &dir, &cgroup)?;
        if stop_came(stopped, tick) {
            let last = seen.read(tree, &dir, &cgroup)?;
            let values = stat_values(&last, &MEMORY_STAT_KEYS);
            seen.memory_stat = MEMORY_STAT_KEYS
                .into_iter()
                .filter_map(|key| Some((key, *values.get(key)?)))
                .collect();
            seen.memory_peak = cgroup_value(&dir.join("memory.peak"));
            if seen.shared.is_none() {
                seen.swap = cgroup_value(&dir.join("memory.swap.peak"));
            }
            return Ok(seen);
        }
    }
}

/// Waits up to a tick for the stop, and says whether it came; a sampler that
/// is gone has stopped.
fn stop_came(stopped: &Receiver<()>, tick: Duration) -> bool {
    match stopped.recv_timeout(tick) {
        Err(RecvTimeoutError::Timeout) => false,
        Ok(()) | Err(RecvTimeoutError::Disconnected) => true,
    }
}

/// The values of `keys` in a `memory.stat`, whichever a kernel prints.
fn stat_values<'a>(raw: &'a str, keys: &[&str]) -> BTreeMap<&'a str, u64> {
    raw.lines()
        .filter_map(|line| line.split_once(' '))
        .filter(|(key, _)| keys.contains(key))
        .filter_map(|(key, value)| Some((key, value.trim().parse().ok()?)))
        .collect()
}

/// `anon + shmem + kernel` of a `memory.stat`, or the first of them it lacks.
fn own_bytes(raw: &str) -> Result<u64, &'static str> {
    let values = stat_values(raw, &OWN_KEYS);
    OWN_KEYS.iter().try_fold(0u64, |sum, &key| {
        values
            .get(key)
            .map(|&bytes| sum.saturating_add(bytes))
            .ok_or(key)
    })
}

/// The one number a cgroup file such as `memory.peak` holds; `None` where the
/// kernel keeps no such file.
fn cgroup_value(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// What the sampler's thread read, handed back at the stop.
struct Seen {
    host_samples: u64,
    host_bytes: u64,
    gpu: Gpu,
    shared: Option<Shared>,
    memory_stat: BTreeMap<&'static str, u64>,
    memory_peak: Option<u64>,
    /// `memory.swap.peak` at the stop; in a cgroup the server shares, which
    /// that watermark would count pipette's swap in, the largest
    /// `memory.swap.current` less the others' `VmSwap` over the reads.
    swap: Option<u64>,
}

/// A cgroup the server shares: which of its processes are the server's, and
/// what came off for the others.
struct Shared {
    split: Split,
    /// What the last read took off.
    others: OwnMemory,
    others_max: u64,
    host_bytes_at_launch: u64,
}

impl Seen {
    fn new(gpu: Gpu, split: Option<Split>) -> Self {
        Self {
            host_samples: 0,
            host_bytes: 0,
            gpu,
            shared: split.map(|split| Shared {
                split,
                others: OwnMemory::default(),
                others_max: 0,
                host_bytes_at_launch: 0,
            }),
            memory_stat: BTreeMap::new(),
            memory_peak: None,
            swap: None,
        }
    }

    /// One tick: the `memory.stat` in `dir`, which it returns, less the
    /// other processes' own memory in a shared cgroup, then the GPU.
    #[cfg_attr(
        not(target_os = "linux"),
        expect(unused_variables, reason = "only the Linux GPU reads `cgroup`")
    )]
    fn read<E>(
        &mut self,
        tree: &Tree,
        dir: &Path,
        cgroup: &Cgroup,
    ) -> Result<String, PeakError<E>> {
        let before = self
            .shared
            .as_mut()
            .map(|shared| shared.split.others(tree, dir));
        let path = dir.join("memory.stat");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(source) => return Err(PeakError::Unreadable { path, source }),
        };
        let bytes = own_bytes(&raw).map_err(|key| PeakError::NoKey { path, key })?;
        let others = match (before, self.shared.as_mut()) {
            (Some(before), Some(shared)) => {
                // Read again: one that exits between the reads is still in
                // `memory.stat`, and the peak would keep a figure that counts it.
                let others = OwnMemory::larger_of(before, shared.split.others(tree, dir));
                shared.others = others;
                shared.others_max = shared.others_max.max(others.resident);
                if let Some(swap) = cgroup_value(&dir.join("memory.swap.current")) {
                    let swap = swap.saturating_sub(others.swapped);
                    self.swap = Some(self.swap.map_or(swap, |seen| seen.max(swap)));
                }
                others.resident
            }
            _ => 0,
        };
        let host = bytes.saturating_sub(others);
        if self.host_samples == 0
            && let Some(shared) = self.shared.as_mut()
        {
            shared.host_bytes_at_launch = host;
        }
        self.host_samples = self.host_samples.saturating_add(1);
        self.host_bytes = self.host_bytes.max(host);
        #[cfg(target_os = "linux")]
        match &mut self.gpu {
            Gpu::Unread => {}
            Gpu::Nvidia(gpu) => gpu.read(
                tree,
                cgroup,
                self.shared.as_mut().map(|shared| &mut shared.split),
            ),
            Gpu::Amd(gpu) => gpu.read(),
        }
        Ok(raw)
    }

    fn peak<E>(self) -> Result<Peak, PeakError<E>> {
        let gpu = match self.gpu {
            Gpu::Unread => None,
            #[cfg(target_os = "linux")]
            Gpu::Nvidia(gpu) => Some(gpu.seen.peak()?),
            #[cfg(target_os = "linux")]
            Gpu::Amd(gpu) => Some(gpu.peak()?),
        };
        let host = self.host_bytes;
        let shared = self.shared.map(|shared| SharedCgroup {
            others_bytes: shared.others.resident,
            others_max_bytes: shared.others_max,
            listed: !shared.split.unlisted,
            host_bytes_at_launch: shared.host_bytes_at_launch,
        });
        Ok(Peak {
            host_bytes: host,
            host_samples: self.host_samples,
            gpu,
            memory: self.swap.map_or_else(
                || MemoryObservation::host_only(host),
                |swap| MemoryObservation::with_swap(host, swap),
            ),
            memory_peak: self.memory_peak,
            memory_stat: self.memory_stat,
            shared,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::TryRecvError;

    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    /// The scope [`fake_tree`] holds, as `/proc/<pid>/cgroup` names it.
    const SCOPE: &str = "/system.slice/server.scope";

    /// The scope's `memory.stat` in [`fake_tree`].
    const MEMORY_STAT: &str = "anon 1000\nfile 2000\nkernel 300\nkernel_stack 20\nshmem 40\n";

    /// `anon + shmem + kernel` of [`MEMORY_STAT`].
    const OWN_BYTES: u64 = 1_340;

    /// The scope's `memory.peak` in [`fake_tree`].
    const MEMORY_PEAK: u64 = 123_456_789;

    /// The scope's `memory.swap.peak` in [`fake_tree`].
    const SWAP_BYTES: u64 = 4_096;

    /// The scope's directory in [`fake_tree`]'s tree in `dir`.
    fn scope_dir(dir: &Path) -> PathBuf {
        dir.join("cgroup").join("system.slice").join("server.scope")
    }

    /// A stand-in host in `dir`: a tree counting memory, with [`SCOPE`]'s
    /// `memory.stat`, `memory.peak` and `memory.swap.peak` holding the values
    /// above.
    fn fake_tree(dir: &Path) -> anyhow::Result<Tree> {
        let scope = scope_dir(dir);
        std::fs::create_dir_all(&scope)?;
        std::fs::write(dir.join("cgroup/cgroup.controllers"), "cpu memory pids\n")?;
        std::fs::write(scope.join("memory.stat"), MEMORY_STAT)?;
        std::fs::write(scope.join("memory.peak"), format!("{MEMORY_PEAK}\n"))?;
        std::fs::write(scope.join("memory.swap.peak"), format!("{SWAP_BYTES}\n"))?;
        Ok(Tree::at(dir.join("proc"), dir.join("cgroup"))?)
    }

    fn scope() -> anyhow::Result<Cgroup> {
        Cgroup::named(&format!("0::{SCOPE}\n"), |part| part == "server.scope").context("no scope")
    }

    /// Writes `raw` as the scope's `memory.stat` in [`fake_tree`]'s tree in
    /// `dir`, in one step, so a read never sees half of it.
    fn set_memory_stat(dir: &Path, raw: &str) -> anyhow::Result<()> {
        let scope = scope_dir(dir);
        std::fs::write(scope.join("memory.stat.new"), raw)?;
        std::fs::rename(scope.join("memory.stat.new"), scope.join("memory.stat"))?;
        Ok(())
    }

    /// Why a test's `locate` names nothing.
    #[derive(Debug, thiserror::Error)]
    #[error("the launcher has not started it")]
    struct NotStartedError;

    /// A sampler of [`SCOPE`] in `tree`, which `locate` names at once.
    fn sampler(tree: Tree, watch: Watch) -> anyhow::Result<Sampler<NotStartedError>> {
        let scope = scope()?;
        Ok(Sampler::start(tree, watch, Members::All, move |_| {
            Ok(scope.clone())
        })?)
    }

    /// Only Linux has a cgroup v2 tree. A Linux host running this has one
    /// that counts memory.
    #[test]
    fn the_host_s_tree_is_read_on_linux_only() -> anyhow::Result<()> {
        match (cfg!(target_os = "linux"), Tree::open()) {
            (true, Ok(tree)) => {
                assert_eq!(tree.proc, Path::new("/proc"));
                assert_eq!(tree.root, Path::new("/sys/fs/cgroup"));
            }
            (false, Err(TreeError::NotLinux)) => {}
            (linux, got) => anyhow::bail!("on Linux {linux}: {:?}", got.map(|_| "a tree")),
        }
        Ok(())
    }

    /// A tree whose root lists no memory controller, or that is no cgroup v2
    /// tree at all, is refused, naming the file read.
    #[rstest]
    #[case::cgroup_v2(Some("cpuset cpu io memory hugetlb pids rdma misc\n"), Ok(()))]
    #[case::booted_without_memory(
        Some("cpuset cpu io hugetlb pids rdma misc\n"),
        Err("lists no `memory` controller")
    )]
    #[case::cgroup_v1(None, Err("no cgroup v2 tree there"))]
    fn a_tree_that_counts_no_memory_is_refused(
        #[case] controllers: Option<&str>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let (proc, root) = (dir.path().join("proc"), dir.path().join("cgroup"));
        std::fs::create_dir_all(&root)?;
        if let Some(controllers) = controllers {
            std::fs::write(root.join("cgroup.controllers"), controllers)?;
        }
        match (Tree::at(proc.clone(), root.clone()), expected) {
            (Ok(tree), Ok(())) => {
                assert_eq!(tree.proc, proc);
                assert_eq!(tree.root, root);
            }
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
                let file = root.join("cgroup.controllers");
                assert!(message.contains(&file.display().to_string()), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {:?}", got.map(|_| "a tree")),
        }
        Ok(())
    }

    /// A cgroup counts its memory when its parent can hand the controller
    /// down. One below a parent that lists none, or that the tree does not
    /// hold, is refused, naming the file read; the root's is the tree's.
    #[rstest]
    #[case::handed_down("/user.slice/app.slice/run-1.scope", Some("memory pids\n"), Ok(()))]
    #[case::not_delegated(
        "/user.slice/app.slice/run-1.scope",
        Some("pids\n"),
        Err("lists no `memory` controller")
    )]
    #[case::not_held(
        "/user.slice/app.slice/run-1.scope",
        None,
        Err("does not hold the cgroup")
    )]
    #[case::below_the_root("/run-1.scope", None, Ok(()))]
    fn a_cgroup_counts_memory_when_its_parent_can_hand_it_down(
        #[case] cgroup: &str,
        #[case] parent_controllers: Option<&str>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        let parent = dir
            .path()
            .join("cgroup")
            .join("user.slice")
            .join("app.slice");
        if let Some(controllers) = parent_controllers {
            std::fs::create_dir_all(&parent)?;
            std::fs::write(parent.join("cgroup.controllers"), controllers)?;
        }
        match (
            tree.require_memory_for(&Cgroup(cgroup.to_owned())),
            expected,
        ) {
            (Ok(()), Ok(())) => {}
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
                let file = parent.join("cgroup.controllers");
                assert!(message.contains(&file.display().to_string()), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// The cgroup is the v2 path through the part the caller names, so a
    /// process below it gives it too; cgroup v1 lines, longer names and paths
    /// with no such part give none.
    #[rstest]
    #[case::the_cgroup_itself(format!("0::{SCOPE}\n"), Some(SCOPE))]
    #[case::below_it(format!("0::{SCOPE}/init\n"), Some(SCOPE))]
    #[case::hybrid(
        format!("12:memory:/other.scope\n0::{SCOPE}\n"),
        Some(SCOPE)
    )]
    #[case::cgroup_v1("12:memory:/system.slice/server.scope\n0::/\n".to_owned(), None)]
    #[case::a_longer_name("0::/system.slice/server.scope-1\n".to_owned(), None)]
    #[case::another_cgroup("0::/user.slice/user-1001.slice/session-7.scope\n".to_owned(), None)]
    fn a_process_s_cgroups_name_its_cgroup(
        #[case] cgroups: String,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            Cgroup::named(&cgroups, |part| part == "server.scope"),
            expected.map(|path| Cgroup(path.to_owned()))
        );
    }

    #[cfg(target_os = "linux")]
    #[rstest]
    #[case::in_it(format!("0::{SCOPE}\n"), true)]
    #[case::below_it(format!("0::{SCOPE}/init\n"), true)]
    #[case::a_longer_name(format!("0::{SCOPE}-1\n"), false)]
    #[case::its_parent("0::/system.slice\n".to_owned(), false)]
    #[case::cgroup_v1(format!("12:memory:{SCOPE}\n0::/\n"), false)]
    fn a_cgroup_holds_what_is_in_it(
        #[case] cgroups: String,
        #[case] expected: bool,
    ) -> anyhow::Result<()> {
        assert_eq!(scope()?.holds(&cgroups), expected);
        Ok(())
    }

    /// What [`process`] writes to a `/proc/<pid>/status`, in kB.
    #[derive(Default)]
    struct Status {
        rss_anon: u64,
        rss_shmem: u64,
        vm_pte: u64,
        vm_swap: u64,
    }

    /// Writes `/proc/<pid>/stat` and `/proc/<pid>/status` under `proc`: its
    /// parent and session, in a group of its own, and its own memory.
    fn process(
        proc: &Path,
        pid: u32,
        parent: u32,
        session: u32,
        status: &Status,
    ) -> anyhow::Result<()> {
        let dir = proc.join(pid.to_string());
        std::fs::create_dir_all(&dir)?;
        // A command name with a space and a parenthesis, as one can have.
        std::fs::write(
            dir.join("stat"),
            format!("{pid} (python (x) y) S {parent} {pid} {session} 0 -1 4194560 0 0\n"),
        )?;
        std::fs::write(
            dir.join("status"),
            format!(
                "Name:\tpython\nVmRSS:\t99999 kB\nRssAnon:\t{} kB\nRssFile:\t77777 kB\n\
                 RssShmem:\t{} kB\nVmPTE:\t{} kB\nVmSwap:\t{} kB\n",
                status.rss_anon, status.rss_shmem, status.vm_pte, status.vm_swap
            ),
        )?;
        Ok(())
    }

    /// pipette's own cgroup is the first cgroup v2 path its
    /// `/proc/self/cgroup` names, when it counts memory of its own; the host's
    /// root cgroup, which has no `memory.current`, does not. A cgroup
    /// namespace's root is a job's or a container's cgroup, which does.
    #[rstest]
    #[case::a_login_session(
        Some("0::/user.slice/user-1000.slice/session-7.scope\n"),
        Some("/user.slice/user-1000.slice/session-7.scope"),
        Ok("/user.slice/user-1000.slice/session-7.scope")
    )]
    #[case::a_namespace_root(Some("0::/\n"), Some(""), Ok(""))]
    #[case::the_host_root(Some("0::/\n"), None, Err("runs in the host's root cgroup"))]
    #[case::cgroup_v1(Some("12:memory:/user.slice\n"), None, Err("names no cgroup v2 path"))]
    #[case::unread(None, None, Err("which names pipette's own cgroup"))]
    fn pipette_s_own_cgroup_counts_memory_of_its_own(
        #[case] cgroups: Option<&str>,
        #[case] counted: Option<&str>,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        if let Some(cgroups) = cgroups {
            std::fs::create_dir_all(dir.path().join("proc/self"))?;
            std::fs::write(dir.path().join("proc/self/cgroup"), cgroups)?;
        }
        if let Some(counted) = counted {
            let own = dir
                .path()
                .join("cgroup")
                .join(counted.trim_start_matches('/'));
            std::fs::create_dir_all(&own)?;
            std::fs::write(own.join("memory.current"), "4096\n")?;
        }
        match (tree.own_cgroup(), expected) {
            (Ok(own), Ok(path)) => assert_eq!(own.path(), path),
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// A shared cgroup's processes that are the server's: its leader's
    /// session, a build step whose parent exited included, and what they
    /// start in a session of their own. pipette, what it starts beside the
    /// server, the login above it and an unrelated shell are not, and nor is
    /// a process gone.
    #[test]
    fn the_server_s_processes_are_its_session_and_what_they_start() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        let proc = dir.path().join("proc");
        // pid, parent, session
        let processes = [
            (10, 1, 10),  // sshd
            (11, 10, 10), // pipette
            (20, 11, 20), // the server
            (21, 20, 20), // its engine core
            (30, 1, 20),  // a build step whose parent exited
            (31, 30, 20), // the compiler it runs
            (32, 21, 32), // a worker in a session of its own
            (33, 32, 32), // what it runs
            (40, 11, 40), // something else pipette started
            (50, 1, 50),  // a shell of another login
        ];
        processes
            .into_iter()
            .try_for_each(|(pid, parent, session)| {
                process(&proc, pid, parent, session, &Status::default())
            })?;
        let mut split = Split::new(20, 11);
        let counted: Vec<u32> = [10, 11, 20, 21, 30, 31, 32, 33, 40, 50, 99]
            .into_iter()
            .filter(|&pid| split.counts(&tree, pid))
            .collect();
        assert_eq!(counted, [20, 21, 30, 31, 32, 33]);
        Ok(())
    }

    /// In a cgroup the server shares, the host figure is the cgroup's own
    /// memory less that of its other processes, their page tables included:
    /// pipette's, listed or not, and those of the processes that are not the
    /// server's. The figure taken off is kept beside it, and the swap is
    /// `memory.swap.current` less their `VmSwap`, not the cgroup's watermark.
    /// A cgroup of the server's own takes nothing off.
    #[rstest]
    #[case::shared(
        Members::Session { leader: 20 },
        10_000_000 - 1_642_496,
        Some(1_642_496),
        50_000 - 30_720
    )]
    #[case::its_own(Members::All, 10_000_000, None, SWAP_BYTES)]
    fn a_shared_cgroup_counts_the_server_s_memory_alone(
        #[case] members: Members,
        #[case] host_bytes: u64,
        #[case] others_bytes: Option<u64>,
        #[case] swap_bytes: u64,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        set_memory_stat(
            dir.path(),
            "anon 8000000\nfile 5\nkernel 1000000\nshmem 1000000\n",
        )?;
        let proc = dir.path().join("proc");
        let own = std::process::id();
        let pipette = Status {
            rss_anon: 1000,
            vm_pte: 4,
            vm_swap: 10,
            ..Status::default()
        };
        let server = Status {
            rss_anon: 5000,
            rss_shmem: 100,
            vm_pte: 40,
            vm_swap: 7,
        };
        let shell = Status {
            rss_anon: 500,
            rss_shmem: 100,
            vm_pte: 0,
            vm_swap: 20,
        };
        process(&proc, own, 1, own, &pipette)?;
        process(&proc, 20, own, 20, &server)?;
        process(&proc, 50, 1, 50, &shell)?;
        // pipette is not listed, as a pid namespace can hide it.
        std::fs::write(scope_dir(dir.path()).join("cgroup.procs"), "20\n50\n")?;
        std::fs::write(scope_dir(dir.path()).join("memory.swap.current"), "50000\n")?;
        let scope = scope()?;
        let peak = Sampler::start(tree, Watch::Peak, members, move |_| {
            Ok::<_, Unlocated<NotStartedError>>(scope.clone())
        })?
        .stop()?;
        assert_eq!(peak.host_bytes, host_bytes);
        assert_eq!(
            peak.shared,
            others_bytes.map(|others_bytes| SharedCgroup {
                others_bytes,
                others_max_bytes: others_bytes,
                listed: true,
                host_bytes_at_launch: host_bytes,
            })
        );
        assert_eq!(
            peak.memory,
            MemoryObservation::with_swap(host_bytes, swap_bytes)
        );
        Ok(())
    }

    /// A neighbor large during the run and small at the stop is kept as the
    /// largest figure taken off. Unix only: files are replaced mid-run.
    #[cfg(unix)]
    #[test]
    fn a_neighbor_that_shrinks_before_the_stop_is_kept() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        set_memory_stat(
            dir.path(),
            "anon 8000000
file 5
kernel 0
shmem 0
",
        )?;
        let proc = dir.path().join("proc");
        let own = std::process::id();
        let neighbor = |rss_anon| Status {
            rss_anon,
            ..Status::default()
        };
        process(&proc, own, 1, own, &Status::default())?;
        process(&proc, 20, own, 20, &neighbor(0))?;
        process(&proc, 50, 1, 50, &neighbor(4000))?;
        std::fs::write(scope_dir(dir.path()).join("cgroup.procs"), "20\n50\n")?;
        let scope = scope()?;
        let sampler = Sampler::start(
            tree,
            Watch::Peak,
            Members::Session { leader: 20 },
            move |_| Ok::<_, Unlocated<NotStartedError>>(scope.clone()),
        )?;
        std::thread::sleep(Duration::from_millis(300));
        process(&proc, 50, 1, 50, &neighbor(10))?;
        std::thread::sleep(Duration::from_millis(300));
        let peak = sampler.stop()?;
        let shared = peak.shared.context("not shared")?;
        assert_eq!(shared.others_bytes, 10 * 1024);
        assert_eq!(shared.others_max_bytes, 4000 * 1024);
        Ok(())
    }

    /// A process either read lists counts at its larger figure: one that
    /// exits between the reads was still in `memory.stat`.
    #[test]
    fn each_neighbor_counts_at_its_larger_read() {
        let memory = |resident, swapped| OwnMemory { resident, swapped };
        let sum = OwnMemory::larger_of(
            BTreeMap::from([(1, memory(100, 1)), (2, memory(50, 9))]),
            BTreeMap::from([(2, memory(70, 2)), (3, memory(30, 0))]),
        );
        assert_eq!((sum.resident, sum.swapped), (200, 10));
    }

    /// An empty `cgroup.procs`, as a pid namespace shows it, takes off
    /// pipette's own memory alone, and the row says so.
    #[test]
    fn an_empty_process_list_is_marked() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        let own = std::process::id();
        let pipette = Status {
            rss_anon: 1,
            ..Status::default()
        };
        process(&dir.path().join("proc"), own, 1, own, &pipette)?;
        std::fs::write(scope_dir(dir.path()).join("cgroup.procs"), "")?;
        let scope = scope()?;
        let peak = Sampler::start(
            tree,
            Watch::Peak,
            Members::Session { leader: 20 },
            move |_| Ok::<_, Unlocated<NotStartedError>>(scope.clone()),
        )?
        .stop()?;
        let shared = peak.shared.context("not shared")?;
        assert!(!shared.listed);
        assert_eq!(shared.others_bytes, 1024);
        Ok(())
    }

    /// Where the cgroup `/proc/self/cgroup` names does not list pipette, as a
    /// pyxis container's namespace root, the one below it that does is
    /// pipette's own, and the processes beside pipette there count as others.
    #[test]
    fn pipette_s_own_cgroup_is_the_one_listing_it() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        std::fs::create_dir_all(dir.path().join("proc/self"))?;
        std::fs::write(dir.path().join("proc/self/cgroup"), format!("0::{SCOPE}\n"))?;
        let scope = scope_dir(dir.path());
        std::fs::write(scope.join("memory.current"), "4096\n")?;
        std::fs::write(scope.join("cgroup.procs"), "")?;
        std::fs::create_dir_all(scope.join("task_special"))?;
        std::fs::write(scope.join("task_special/cgroup.procs"), "")?;
        std::fs::create_dir_all(scope.join("task_0"))?;
        std::fs::write(
            scope.join("task_0/cgroup.procs"),
            format!("7\n{}\n9\n", std::process::id()),
        )?;
        assert_eq!(tree.own_cgroup()?.path(), format!("{SCOPE}/task_0"));
        Ok(())
    }

    /// pipette's own cgroup is refused when a cgroup below it holds
    /// processes, whose memory the probe would count as the server's; empty
    /// ones below it are fine.
    #[rstest]
    #[case::an_empty_child("", Ok(()))]
    #[case::a_populated_child("4242\n", Err("in a cgroup below pipette's own"))]
    fn a_cgroup_below_pipette_s_holding_processes_is_refused(
        #[case] procs: &str,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        std::fs::create_dir_all(dir.path().join("proc/self"))?;
        std::fs::write(dir.path().join("proc/self/cgroup"), format!("0::{SCOPE}\n"))?;
        std::fs::write(scope_dir(dir.path()).join("memory.current"), "4096\n")?;
        let child = scope_dir(dir.path()).join("deep/child");
        std::fs::create_dir_all(&child)?;
        std::fs::write(child.join("cgroup.procs"), procs)?;
        match (tree.own_cgroup(), expected) {
            (Ok(_), Ok(())) => {}
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// The keys a row records of `memory.stat`, whichever a kernel prints.
    #[rstest]
    #[case::every_key(
        "anon 2048\nfile 4096\nkernel 512\nkernel_stack 64\nshmem 128\nfile_mapped 99\n",
        &[("anon", 2048), ("file", 4096), ("kernel", 512), ("shmem", 128)]
    )]
    #[case::an_older_kernel("anon 2048\nfile 4096\nshmem 128\n", &[("anon", 2048), ("file", 4096), ("shmem", 128)])]
    fn memory_stat_keeps_its_keys(#[case] raw: &str, #[case] expected: &[(&str, u64)]) {
        assert_eq!(
            stat_values(raw, &MEMORY_STAT_KEYS),
            expected.iter().copied().collect::<BTreeMap<_, _>>()
        );
    }

    /// The cgroup's own memory is `anon + shmem + kernel`; the page cache,
    /// mapped or not, is left out, and a kernel that prints no `kernel` count
    /// gives none.
    #[rstest]
    #[case::every_key(
        "anon 2048\nfile 4096\nkernel 512\nkernel_stack 64\nshmem 128\nfile_mapped 99\n",
        Ok(2_688)
    )]
    #[case::only_its_own("anon 10\nkernel 2\nshmem 1\n", Ok(13))]
    #[case::not_a_prefix("anon_thp 5\nanon 10\nshmem 1\nkernel 2\nshmem_thp 7\n", Ok(13))]
    #[case::an_older_kernel("anon 2048\nfile 4096\nshmem 128\n", Err("kernel"))]
    #[case::no_shmem("anon 2048\nkernel 512\n", Err("shmem"))]
    #[case::unparsed("anon x\nshmem 1\nkernel 2\n", Err("anon"))]
    fn the_cgroup_s_own_memory_is_anon_shmem_and_kernel(
        #[case] raw: &str,
        #[case] expected: Result<u64, &str>,
    ) {
        assert_eq!(own_bytes(raw), expected);
    }

    /// The host figure is the largest of the cgroup's own memory over the
    /// ticks, not the last; the page cache rising meanwhile adds nothing.
    /// Unix only: the file is replaced while the sampler reads it.
    #[cfg(unix)]
    #[test]
    fn the_host_figure_is_the_largest_read() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        set_memory_stat(dir.path(), "anon 900\nfile 9\nshmem 50\nkernel 50\n")?;
        let sampler = sampler(tree, Watch::Peak)?;
        std::thread::sleep(Duration::from_millis(300));
        set_memory_stat(dir.path(), "anon 100\nfile 99999\nshmem 0\nkernel 20\n")?;
        std::thread::sleep(Duration::from_millis(300));
        let peak = sampler.stop()?;
        assert_eq!(peak.host_bytes, 1_000);
        assert!(peak.host_samples > 2, "{}", peak.host_samples);
        Ok(())
    }

    /// A rise just before the stop counts: the stop reads once more, and the
    /// row's `memory.stat` keys are that read's. An observation, on its slow
    /// tick, read the low figure long before. Unix only, as above.
    #[cfg(unix)]
    #[test]
    fn the_read_at_the_stop_counts() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let sampler = sampler(fake_tree(dir.path())?, Watch::Observe)?;
        std::thread::sleep(Duration::from_millis(300));
        set_memory_stat(dir.path(), "anon 5000\nfile 7\nshmem 0\nkernel 1\n")?;
        let peak = sampler.stop()?;
        assert_eq!(peak.host_bytes, 5_001);
        assert_eq!(peak.shared, None);
        assert_eq!(
            peak.memory_stat,
            BTreeMap::from([("anon", 5000), ("file", 7), ("shmem", 0), ("kernel", 1)])
        );
        Ok(())
    }

    /// A tree that does not hold the cgroup's `memory.stat`, or one that
    /// prints no count the figure needs, is refused, naming the file, rather
    /// than recorded without it.
    #[rstest]
    #[case::a_hidden_tree(None, "does not hold the cgroup")]
    #[case::an_older_kernel(Some("anon 1\nfile 2\nshmem 3\n"), "has no `kernel` count")]
    fn a_tree_without_the_cgroup_s_memory_stat_is_refused(
        #[case] memory_stat: Option<&str>,
        #[case] cause: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        let stat = scope_dir(dir.path()).join("memory.stat");
        match memory_stat {
            Some(raw) => set_memory_stat(dir.path(), raw)?,
            None => std::fs::remove_file(&stat)?,
        }
        let err = sampler(tree, Watch::Peak)?
            .stop()
            .err()
            .context("expected a refusal")?;
        let message = format!("{:#}", anyhow::Error::from(err));
        assert!(message.contains(cause), "{message}");
        assert!(message.contains(&stat.display().to_string()), "{message}");
        Ok(())
    }

    /// The figures a peak reads beside its own at the stop: the swap
    /// watermark, `memory.peak`, which counts the page cache too, and the
    /// `memory.stat` keys a row keeps; a kernel that keeps neither watermark
    /// leaves them out.
    #[rstest]
    #[case::kept(
        &[],
        MemoryObservation::with_swap(OWN_BYTES, SWAP_BYTES),
        Some(MEMORY_PEAK)
    )]
    #[case::an_older_kernel(
        &["memory.peak", "memory.swap.peak"],
        MemoryObservation::host_only(OWN_BYTES),
        None
    )]
    fn the_stop_reads_the_kernel_s_watermarks(
        #[case] missing: &[&str],
        #[case] memory: MemoryObservation,
        #[case] memory_peak: Option<u64>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = fake_tree(dir.path())?;
        missing
            .iter()
            .try_for_each(|file| std::fs::remove_file(scope_dir(dir.path()).join(file)))?;
        let peak = sampler(tree, Watch::Peak)?.stop()?;
        assert_eq!(peak.host_bytes, OWN_BYTES);
        assert_eq!(peak.memory, memory);
        assert_eq!(peak.memory_peak, memory_peak);
        assert_eq!(
            peak.memory_stat,
            BTreeMap::from([
                ("anon", 1000),
                ("file", 2000),
                ("kernel", 300),
                ("shmem", 40)
            ])
        );
        Ok(())
    }

    /// The sampler asks `locate` again until it names the cgroup, which a
    /// launcher starts a moment after it starts, then reads.
    #[test]
    fn the_sampler_reads_once_locate_names_the_cgroup() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let scope = scope()?;
        let (named, naming) = mpsc::channel();
        let mut tries = 0u32;
        let sampler = Sampler::start(
            fake_tree(dir.path())?,
            Watch::Peak,
            Members::All,
            move |_| {
                tries = tries.saturating_add(1);
                if tries < 3 {
                    return Err(Unlocated::Yet(NotStartedError));
                }
                let _ = named.send(());
                Ok(scope.clone())
            },
        )?;
        naming
            .recv_timeout(Duration::from_secs(10))
            .context("locate was not asked a third time")?;
        let peak = sampler.stop()?;
        assert_eq!(peak.host_bytes, OWN_BYTES);
        assert!(peak.host_samples > 0, "{}", peak.host_samples);
        Ok(())
    }

    /// A peak reads every 10 ms, for resolution; an observation every 100 ms,
    /// so watching cannot move its time. The bounds leave a busy machine room:
    /// a 10 ms tick read at most every 50 ms still passes, and a 100 ms tick
    /// reads at most once per 100 ms, plus its first read and the stop's.
    #[test]
    fn a_peak_reads_ten_times_as_often_as_an_observation() -> anyhow::Result<()> {
        let (peak_dir, observed_dir) = (tempfile::tempdir()?, tempfile::tempdir()?);
        let started = std::time::Instant::now();
        let peak = sampler(fake_tree(peak_dir.path())?, Watch::Peak)?;
        let observed = sampler(fake_tree(observed_dir.path())?, Watch::Observe)?;
        std::thread::sleep(Duration::from_millis(600));
        let observed = observed.stop()?.host_samples;
        let peak = peak.stop()?.host_samples;
        let elapsed = u64::try_from(started.elapsed().as_millis())?;
        assert!(peak >= elapsed / 50, "{peak} reads in {elapsed} ms");
        assert!(
            observed <= elapsed / 100 + 2,
            "{observed} reads in {elapsed} ms"
        );
        Ok(())
    }

    /// A cgroup `locate` never names fails the sampler, with the last reason;
    /// one it never will fails it at once, without asking again.
    #[rstest]
    #[case::not_yet(false)]
    #[case::never(true)]
    fn a_cgroup_never_named_has_no_peak(#[case] never: bool) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let (asked, asks) = mpsc::channel();
        let sampler = Sampler::start(
            fake_tree(dir.path())?,
            Watch::Peak,
            Members::All,
            move |_| {
                let _ = asked.send(());
                Err(if never {
                    Unlocated::Failed(NotStartedError)
                } else {
                    Unlocated::Yet(NotStartedError)
                })
            },
        )?;
        std::thread::sleep(Duration::from_millis(200));
        let err = sampler.stop().err().context("expected a refusal")?;
        let message = format!("{:#}", anyhow::Error::from(err));
        assert!(
            message.contains("never found") && message.contains("has not started it"),
            "{message}"
        );
        let asked = asks.try_iter().count();
        assert!(
            if never { asked == 1 } else { asked > 1 },
            "asked {asked} times"
        );
        Ok(())
    }

    /// A sampler dropped without a stop, as when the server never gets ready,
    /// ends its thread before the drop returns.
    #[test]
    fn a_dropped_sampler_ends_its_thread() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let (tried, tries) = mpsc::channel();
        // Slow tries, so a thread left running is still in one.
        let sampler = Sampler::start(
            fake_tree(dir.path())?,
            Watch::Peak,
            Members::All,
            move |_| {
                let _ = tried.send(());
                std::thread::sleep(Duration::from_millis(50));
                Err(Unlocated::Yet(NotStartedError))
            },
        )?;
        tries
            .recv_timeout(Duration::from_secs(10))
            .context("the sampler never tried")?;
        drop(sampler);
        tries.try_iter().for_each(drop);
        assert!(
            matches!(tries.try_recv(), Err(TryRecvError::Disconnected)),
            "the sampler still runs"
        );
        Ok(())
    }

    #[cfg(target_os = "linux")]
    mod amd {
        use super::*;
        use crate::amd::test_support::{fake_sys, set_used};
        use crate::amd::{AmdGpu, Settle};

        /// The GPU figure is the largest rise of the cards' used VRAM over
        /// the least they held since they were opened, beside the host figure
        /// of the same ticks: memory a server stopped before was still
        /// handing back when they were opened is not this one's. Cards that
        /// never rose fail the peak.
        #[rstest]
        #[case::read(300, 5_300, Ok(5_000))]
        #[case::freed_after_the_open(45_300, 5_300, Ok(5_000))]
        #[case::never_rose(300, 300, Err("never rose above"))]
        fn a_peak_on_an_amd_gpu_is_the_cards_rise(
            #[case] at_open: u64,
            #[case] used: u64,
            #[case] expected: Result<u64, &str>,
        ) -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let sys = fake_sys(dir.path(), true)?;
            set_used(&sys, at_open)?;
            let gpu = AmdGpu::open_in(&sys, None, Settle::NONE)?;
            set_used(&sys, 300)?;
            let sampler = sampler(fake_tree(dir.path())?, Watch::PeakWithAmdGpu(gpu))?;
            std::thread::sleep(Duration::from_millis(100));
            set_used(&sys, used)?;
            std::thread::sleep(Duration::from_millis(100));
            match (sampler.stop(), expected) {
                (Ok(peak), Ok(bytes)) => {
                    let gpu = peak.gpu.context("no GPU figure")?;
                    assert_eq!(gpu.bytes, bytes);
                    assert_eq!(
                        gpu.source,
                        GpuSource::Cards {
                            floor_bytes: 300,
                            settled: true
                        }
                    );
                    assert_eq!(peak.host_bytes, OWN_BYTES);
                    assert!(gpu.samples > 0);
                }
                (Err(err), Err(cause)) => {
                    let message = format!("{:#}", anyhow::Error::from(err));
                    assert!(message.contains(cause), "{message}");
                }
                (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
            }
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    mod nvidia {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        use super::*;

        /// Each pid's cgroup file under `proc`.
        fn place(proc: &Path, pids: &[(u32, String)]) -> anyhow::Result<()> {
            pids.iter().try_for_each(|(pid, cgroup)| {
                let pid_dir = proc.join(pid.to_string());
                std::fs::create_dir_all(&pid_dir)?;
                std::fs::write(pid_dir.join("cgroup"), cgroup)
            })?;
            Ok(())
        }

        /// [`fake_tree`] with pid 41 in the scope and 42 in another.
        fn scope_tree(dir: &Path) -> anyhow::Result<Tree> {
            let tree = fake_tree(dir)?;
            place(
                &dir.join("proc"),
                &[
                    (41, format!("0::{SCOPE}\n")),
                    (42, "0::/system.slice/other.scope\n".to_owned()),
                ],
            )?;
            Ok(tree)
        }

        /// Of every process NVML lists, the cgroup's count: one whose cgroup
        /// is another, or that exited before its cgroup was read, does not,
        /// whatever NVML says of it.
        #[test]
        fn only_the_cgroup_s_processes_count() -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let tree = fake_tree(dir.path())?;
            place(
                &dir.path().join("proc"),
                &[
                    (41, format!("0::{SCOPE}\n")),
                    (42, "0::/system.slice/other.scope\n".to_owned()),
                    (44, "0::/system.slice/other.scope\n".to_owned()),
                    (46, format!("0::{SCOPE}/init\n")),
                ],
            )?;
            let mut gpu = NvidiaGpu::stand_in(|| {
                Ok(Pids::from([
                    (41, Ok(100)),
                    (42, Ok(7)),
                    (43, Ok(9)),
                    (44, Err(NotAvailable)),
                    (45, Err(NotAvailable)),
                    (46, Err(NotAvailable)),
                ]))
            });
            assert_eq!(
                gpu.held(&tree, &scope()?, None),
                Ok(Pids::from([(41, Ok(100)), (46, Err(NotAvailable))]))
            );
            Ok(())
        }

        /// In a cgroup the server shares, only its processes count on the
        /// GPU: another in the cgroup does not, whatever NVML says of it.
        #[test]
        fn in_a_shared_cgroup_only_the_server_s_processes_count() -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let tree = scope_tree(dir.path())?;
            let proc = dir.path().join("proc");
            place(&proc, &[(46, format!("0::{SCOPE}\n"))])?;
            super::process(&proc, 41, 1, 41, &super::Status::default())?;
            super::process(&proc, 46, 1, 46, &super::Status::default())?;
            let mut gpu = NvidiaGpu::stand_in(|| Ok(Pids::from([(41, Ok(100)), (46, Ok(9))])));
            let mut split = Split::new(41, std::process::id());
            assert_eq!(
                gpu.held(&tree, &scope()?, Some(&mut split)),
                Ok(Pids::from([(41, Ok(100))]))
            );
            Ok(())
        }

        /// The cgroup's processes on one read, or a GPU not answering.
        type CgroupRead = Result<Vec<(u32, Result<u64, NotAvailable>)>, NvmlReadError>;

        #[rstest]
        #[case::the_largest_sum(
            vec![Ok(vec![(41, Ok(5))]), Ok(vec![(41, Ok(9)), (46, Ok(1))]), Ok(vec![(41, Ok(7))])],
            Ok(10)
        )]
        #[case::answered_once(vec![Err(NvmlReadError::Unanswered), Ok(vec![(41, Ok(4))])], Ok(4))]
        #[case::never_answered(
            vec![Err(NvmlReadError::Unanswered), Err(NvmlReadError::Unanswered)],
            Err("a GPU refused one")
        )]
        #[case::not_available(
            vec![Ok(vec![(41, Ok(5))]), Ok(vec![(41, Err(NotAvailable))])],
            Err("pid 41 as not available")
        )]
        #[case::none_listed(vec![Ok(vec![])], Err("NVML attributed no GPU memory"))]
        #[case::listed_with_none(vec![Ok(vec![(41, Ok(0))])], Err("NVML attributed no GPU memory"))]
        fn a_peak_is_the_largest_read_or_why_there_is_none(
            #[case] reads: Vec<CgroupRead>,
            #[case] expected: Result<u64, &str>,
        ) -> anyhow::Result<()> {
            let seen = reads
                .into_iter()
                .fold(GpuSeen::default(), |mut seen, read| {
                    seen.read(read.map(|processes| processes.into_iter().collect()));
                    seen
                });
            match (seen.peak().map(|peak| peak.bytes), expected) {
                (Ok(bytes), Ok(want)) => assert_eq!(bytes, want),
                (Err(err), Err(cause)) => {
                    let message = format!("{:#}", anyhow::Error::from(err));
                    assert!(message.contains(cause), "{message}");
                }
                (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
            }
            Ok(())
        }

        /// A sampler on a stand-in GPU, and a signal for each of its reads.
        struct Running {
            sampler: Sampler<NotStartedError>,
            reads: Receiver<()>,
        }

        /// A sampler of [`scope_tree`]'s scope in `dir`, on a GPU listing pid
        /// 41 at `bytes()` and pid 42 at 7, which signals each read.
        fn running(
            dir: &Path,
            mut bytes: impl FnMut() -> Result<u64, NotAvailable> + Send + 'static,
        ) -> anyhow::Result<Running> {
            let (read, reads) = mpsc::channel();
            let gpu = NvidiaGpu::stand_in(move || {
                let _ = read.send(());
                Ok(Pids::from([(41, bytes()), (42, Ok(7))]))
            });
            let sampler = sampler(scope_tree(dir)?, Watch::PeakWithNvidiaGpu(gpu))?;
            Ok(Running { sampler, reads })
        }

        fn next(read: &Receiver<()>) -> anyhow::Result<()> {
            read.recv_timeout(Duration::from_secs(10))
                .context("the sampler stopped reading")
        }

        /// A rise just before the stop counts: the stop reads the GPU once
        /// more.
        #[test]
        fn the_gpu_read_at_the_stop_counts() -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let risen = Arc::new(AtomicBool::new(false));
            let rising = Arc::clone(&risen);
            let Running { sampler, reads } = running(dir.path(), move || {
                Ok(if rising.load(Ordering::SeqCst) {
                    100
                } else {
                    1
                })
            })?;
            next(&reads)?;
            risen.store(true, Ordering::SeqCst);
            assert_eq!(sampler.stop()?.gpu.map(|gpu| gpu.bytes), Some(100));
            Ok(())
        }

        /// The GPU figure is that of the cgroup's processes, beside the host
        /// figure of the same ticks, and a GPU that showed none fails the peak
        /// rather than leaving the figure out.
        #[rstest]
        #[case::read(Some(5_000), Ok(Some(5_000)))]
        #[case::none(Some(0), Err("NVML attributed no GPU memory"))]
        #[case::not_available(None, Err("pid 41 as not available"))]
        fn a_peak_on_an_nvidia_gpu_has_its_figure_or_fails(
            #[case] reading: Option<u64>,
            #[case] expected: Result<Option<u64>, &str>,
        ) -> anyhow::Result<()> {
            let dir = tempfile::tempdir()?;
            let Running { sampler, reads } =
                running(dir.path(), move || reading.ok_or(NotAvailable))?;
            next(&reads)?;
            match (sampler.stop(), expected) {
                (Ok(peak), Ok(gpu)) => {
                    assert_eq!(peak.gpu.map(|gpu| gpu.bytes), gpu);
                    assert_eq!(peak.host_bytes, OWN_BYTES);
                }
                (Err(err), Err(cause)) => {
                    let message = format!("{:#}", anyhow::Error::from(err));
                    assert!(message.contains(cause), "{message}");
                }
                (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
            }
            Ok(())
        }
    }
}
