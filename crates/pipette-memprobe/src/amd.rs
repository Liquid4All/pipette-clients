//! An AMD GPU's memory as a whole card, for a [`crate::cgroup::Sampler`]
//! whose server runs on cards a scheduler such as Slurm gave the job alone.
//! A card is the job's alone, so all it holds above its floor is the
//! server's: the cards' reading once it stops falling at the open, as a card
//! hands back a stopped server's memory for seconds after its processes exit.
//! The least read since then stays the floor, should the wait end first.
//!
//! Whole cards, as nothing per process fits: amdgpu's fdinfo bills a buffer
//! on every GPU its process maps it to, and KFD names its processes by the
//! host's pids, which a job in a pid namespace of its own cannot match.

use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::cgroup::{GpuPeak, GpuSource};

/// How often [`Settle`] reads the cards, and the drop that counts as still
/// falling.
const SETTLE_POLL: Duration = Duration::from_millis(100);
const SETTLE_SLACK: u64 = 64 << 20;

/// Why the AMD GPUs this job was given cannot be read, or read nothing.
#[derive(Debug, thiserror::Error)]
pub enum AmdGpuError {
    #[error("could not read {path}, the AMD GPU driver's topology")]
    TopologyUnread {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path} lists no AMD GPU this process may open")]
    NoCard { path: PathBuf },
    #[error("could not read {path}, an AMD GPU's id in the driver's topology")]
    NodeUnread {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("SLURM_GPUS_ON_NODE is {value:?}, not a count of GPUs")]
    SlurmCount { value: String },
    #[error(
        "this process may open every AMD GPU on the host, so another process may hold memory on \
         the one the server uses: a card's used VRAM is the server's only on cards a scheduler \
         such as Slurm gave this job alone"
    )]
    Shared,
    #[error(
        "Slurm gave this job {allotted} GPUs on this node (SLURM_GPUS_ON_NODE), but this process \
         may open {open} AMD GPUs, so some of them may hold another job's memory"
    )]
    NotAllotted { allotted: usize, open: usize },
    #[error("could not read {path}, an AMD GPU's used VRAM")]
    Unread {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path}, an AMD GPU's used VRAM, holds no number")]
    Unparsed { path: PathBuf },
    #[error(
        "renderD{minor} is a partition of an AMD GPU, which reports no used VRAM of its own: a \
         card split into compute partitions shares its VRAM between them, so only a card in SPX \
         mode is read"
    )]
    Partition { minor: String },
    #[error("no read of the AMD GPUs' used VRAM answered during the run")]
    Unanswered,
    #[error(
        "the used VRAM of this job's AMD GPUs never rose above what they held before the launch: \
         the server did not reach them"
    )]
    NoMemory,
}

/// The AMD GPUs this job was given alone: Slurm's device cgroup refuses it
/// the others' topology, or their count is Slurm's.
#[derive(Debug)]
pub struct AmdGpu {
    cards: Vec<Card>,
    /// Whether the open's wait saw the cards stop falling.
    settled: bool,
    samples: u64,
    max_bytes: u64,
}

/// How long [`AmdGpu::open_in`] waits for the cards' used VRAM to stop
/// falling, and at most.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settle {
    pub(crate) window: Duration,
    pub(crate) limit: Duration,
}

impl Settle {
    /// A card frees a stopped server's VRAM in steps about a third of a
    /// second apart, so a held reading must span several.
    const AFTER_A_SERVER: Self = Self {
        window: Duration::from_secs(1),
        limit: Duration::from_secs(30),
    };
    /// No wait, for a stand-in `/sys`.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) const NONE: Self = Self {
        window: Duration::ZERO,
        limit: Duration::ZERO,
    };
}

/// A card's used VRAM file, and the least it has read.
#[derive(Debug)]
struct Card {
    used: PathBuf,
    least: u64,
}

impl Card {
    /// Its used VRAM now, which lowers the least read; `None` when unread.
    fn observe(&mut self) -> Option<u64> {
        let used = std::fs::read_to_string(&self.used)
            .ok()?
            .trim()
            .parse()
            .ok()?;
        self.least = self.least.min(used);
        Some(used)
    }
}

impl AmdGpu {
    /// Open it before the launch: it reads the cards' floor.
    pub fn open() -> Result<Self, AmdGpuError> {
        let allotted = slurm_allotted(std::env::var("SLURM_GPUS_ON_NODE").ok())?;
        Self::open_in(Path::new("/sys"), allotted, Settle::AFTER_A_SERVER)
    }

    /// The cards of a stand-in `/sys`, such as
    /// [`test_support::fake_sys`]'s, opened without a wait.
    #[cfg(any(test, feature = "test-support"))]
    pub fn stand_in(sys: &Path) -> Result<Self, AmdGpuError> {
        Self::open_in(sys, None, Settle::NONE)
    }

    /// A topology node whose `gpu_id` this process may read is a card it may
    /// open. `allotted` is Slurm's GPU count on the node, which those must
    /// match; outside Slurm, some card must be refused.
    pub(crate) fn open_in(
        sys: &Path,
        allotted: Option<usize>,
        settle: Settle,
    ) -> Result<Self, AmdGpuError> {
        let nodes = sys.join("class/kfd/kfd/topology/nodes");
        let read = |path: PathBuf| std::fs::read_to_string(path).map(|text| text.trim().to_owned());
        let entries: Vec<PathBuf> = std::fs::read_dir(&nodes)
            .map_err(|source| AmdGpuError::TopologyUnread {
                path: nodes.clone(),
                source,
            })?
            .flatten()
            .map(|entry| entry.path())
            .collect();
        let Topology { open, refused } = entries.iter().try_fold(
            Topology {
                open: Vec::new(),
                refused: false,
            },
            |mut seen, node| {
                let path = node.join("gpu_id");
                match read(path.clone()) {
                    Ok(id) if id != "0" => seen.open.push(node),
                    Ok(_) => {}
                    // Slurm's device cgroup refuses another job's card; any
                    // other failure says nothing about whose it is.
                    Err(source) if source.kind() == io::ErrorKind::PermissionDenied => {
                        seen.refused = true;
                    }
                    Err(source) => return Err(AmdGpuError::NodeUnread { path, source }),
                }
                Ok(seen)
            },
        )?;
        let minors: Vec<String> = open
            .into_iter()
            .filter_map(|node| {
                read(node.join("properties"))
                    .ok()?
                    .lines()
                    .find_map(|line| line.strip_prefix("drm_render_minor "))
                    .map(str::to_owned)
            })
            .collect();
        if minors.is_empty() {
            return Err(AmdGpuError::NoCard { path: nodes });
        }
        match allotted {
            Some(allotted) if allotted != minors.len() => {
                return Err(AmdGpuError::NotAllotted {
                    allotted,
                    open: minors.len(),
                });
            }
            Some(_) => {}
            None if !refused => return Err(AmdGpuError::Shared),
            None => {}
        }
        let cards = minors
            .into_iter()
            .map(|minor| {
                let used = sys.join(format!(
                    "class/drm/renderD{minor}/device/mem_info_vram_used"
                ));
                let text = match std::fs::read_to_string(&used) {
                    Ok(text) => text,
                    Err(_) if is_partition(&sys.join(format!("class/drm/renderD{minor}"))) => {
                        return Err(AmdGpuError::Partition { minor });
                    }
                    Err(source) => return Err(AmdGpuError::Unread { path: used, source }),
                };
                match text.trim().parse() {
                    Ok(least) => Ok(Card { used, least }),
                    Err(_) => Err(AmdGpuError::Unparsed { path: used }),
                }
            })
            .collect::<Result<Vec<_>, AmdGpuError>>()?;
        let mut gpu = Self {
            cards,
            settled: false,
            samples: 0,
            max_bytes: 0,
        };
        gpu.settled = gpu.settle(settle);
        Ok(gpu)
    }

    /// Wait while the cards' used VRAM still falls, up to `limit`, and say
    /// whether it stopped: a server that allocates sooner would hide the
    /// floor.
    fn settle(&mut self, Settle { window, limit }: Settle) -> bool {
        let started = Instant::now();
        let mut floor = self.least();
        let mut steady = Instant::now();
        while steady.elapsed() < window {
            if started.elapsed() >= limit {
                tracing::warn!(
                    limit_s = limit.as_secs(),
                    "the AMD GPUs' used VRAM still falls; reading it from here"
                );
                return false;
            }
            thread::sleep(SETTLE_POLL);
            self.cards.iter_mut().for_each(|card| {
                card.observe();
            });
            let least = self.least();
            if least.saturating_add(SETTLE_SLACK) < floor {
                floor = least;
                steady = Instant::now();
            }
        }
        true
    }

    fn least(&self) -> u64 {
        self.cards
            .iter()
            .fold(0u64, |sum, card| sum.saturating_add(card.least))
    }

    /// One read of every card; a read some card does not answer counts
    /// nothing.
    pub(crate) fn read(&mut self) {
        let bytes = self.cards.iter_mut().try_fold(0u64, |sum, card| {
            let used = card.observe()?;
            Some(sum.saturating_add(used.saturating_sub(card.least)))
        });
        if let Some(bytes) = bytes {
            self.samples = self.samples.saturating_add(1);
            self.max_bytes = self.max_bytes.max(bytes);
        }
    }

    pub(crate) fn peak(self) -> Result<GpuPeak, AmdGpuError> {
        let floor_bytes = self.least();
        match self {
            Self { samples: 0, .. } => Err(AmdGpuError::Unanswered),
            Self { max_bytes: 0, .. } => Err(AmdGpuError::NoMemory),
            Self {
                max_bytes,
                samples,
                settled,
                ..
            } => Ok(GpuPeak {
                bytes: max_bytes,
                samples,
                source: GpuSource::Cards {
                    floor_bytes,
                    settled,
                },
            }),
        }
    }
}

/// The topology nodes this process may read, and whether it was refused one.
struct Topology<'a> {
    open: Vec<&'a PathBuf>,
    refused: bool,
}

/// A render node of one partition of a card: the driver's `amdgpu_xcp_<n>`
/// platform device, not the card's PCI device.
fn is_partition(render: &Path) -> bool {
    std::fs::canonicalize(render.join("device")).is_ok_and(|device| {
        device
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("amdgpu_xcp"))
    })
}

/// The GPUs Slurm gave this job on the node, from `SLURM_GPUS_ON_NODE`;
/// `None` outside Slurm.
fn slurm_allotted(value: Option<String>) -> Result<Option<usize>, AmdGpuError> {
    value
        .map(|value| {
            value
                .trim()
                .parse()
                .map_err(|_| AmdGpuError::SlurmCount { value })
        })
        .transpose()
}

/// A stand-in `/sys`, for this crate's tests and its dependents'.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::io;
    use std::path::{Path, PathBuf};

    /// A stand-in `/sys` under `dir`: a CPU node, a card this job may open
    /// (render minor 128) holding 300 bytes, and, with `other_job`, a card
    /// whose `gpu_id` it may not read, another job's, as Slurm's device
    /// cgroup refuses it. Not as root, which reads it anyway.
    pub fn fake_sys(dir: &Path, other_job: bool) -> io::Result<PathBuf> {
        let sys = dir.join("sys");
        let nodes = sys.join("class/kfd/kfd/topology/nodes");
        std::fs::create_dir_all(nodes.join("0"))?;
        std::fs::write(nodes.join("0/gpu_id"), "0\n")?;
        std::fs::create_dir_all(nodes.join("1"))?;
        std::fs::write(nodes.join("1/gpu_id"), "9091\n")?;
        std::fs::write(
            nodes.join("1/properties"),
            "cpu_cores_count 0\nsimd_count 1216\ndrm_render_minor 128\nunique_id 1\n",
        )?;
        if other_job {
            use std::os::unix::fs::PermissionsExt;

            std::fs::create_dir_all(nodes.join("2"))?;
            std::fs::write(nodes.join("2/gpu_id"), "9092\n")?;
            std::fs::set_permissions(
                nodes.join("2/gpu_id"),
                std::fs::Permissions::from_mode(0o000),
            )?;
        }
        set_used(&sys, 300)?;
        Ok(sys)
    }

    /// The used VRAM of [`fake_sys`]'s card, replaced at once.
    pub fn set_used(sys: &Path, bytes: u64) -> io::Result<()> {
        let device = sys.join("class/drm/renderD128/device");
        std::fs::create_dir_all(&device)?;
        std::fs::write(device.join("mem_info_vram_used.new"), format!("{bytes}\n"))?;
        std::fs::rename(
            device.join("mem_info_vram_used.new"),
            device.join("mem_info_vram_used"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use anyhow::Context;
    use rstest::rstest;

    use super::test_support::{fake_sys, set_used};
    use super::{AmdGpu, Settle, slurm_allotted};

    /// Read where another job's cards are refused, or where they match
    /// Slurm's count; every other case is refused, saying why.
    #[rstest]
    #[case::a_job_s_own_card(true, None, None, Ok(()))]
    #[case::every_card_open(false, None, None, Err("may open every AMD GPU on the host"))]
    #[case::slurm_gave_every_card(false, Some(1), None, Ok(()))]
    #[case::slurm_gave_its_own_card(true, Some(1), None, Ok(()))]
    #[case::slurm_gave_fewer(false, Some(0), None, Err("gave this job 0 GPUs"))]
    #[case::no_driver(true, None, Some("class/kfd"), Err("the AMD GPU driver's topology"))]
    #[case::no_card(
        true,
        None,
        Some("class/kfd/kfd/topology/nodes/1"),
        Err("lists no AMD GPU")
    )]
    #[case::no_vram_count(
        true,
        None,
        Some("class/drm/renderD128/device/mem_info_vram_used"),
        Err("an AMD GPU's used VRAM")
    )]
    fn only_a_job_s_own_cards_are_read(
        #[case] other_job: bool,
        #[case] allotted: Option<usize>,
        #[case] removed: Option<&str>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let sys = fake_sys(dir.path(), other_job)?;
        if let Some(removed) = removed {
            let path = sys.join(removed);
            if path.is_dir() {
                std::fs::remove_dir_all(path)?;
            } else {
                std::fs::remove_file(path)?;
            }
        }
        match (AmdGpu::open_in(&sys, allotted, Settle::NONE), expected) {
            (Ok(gpu), Ok(())) => assert_eq!(gpu.cards.len(), 1),
            (Err(err), Err(cause)) => {
                let message = format!("{:#}", anyhow::Error::from(err));
                assert!(message.contains(cause), "{message}");
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }

    /// A `gpu_id` that fails to read for another reason than a refusal says
    /// nothing about whose the card is.
    fn gpu_id_unreadable(sys: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(sys.join("class/kfd/kfd/topology/nodes/3/gpu_id"))
    }

    /// A partition of a card, the driver's `amdgpu_xcp_<n>` device, reports
    /// no used VRAM, which its card shares with the others.
    fn a_partition(sys: &Path) -> std::io::Result<()> {
        let render = sys.join("class/drm/renderD128");
        std::fs::remove_dir_all(&render)?;
        std::fs::create_dir_all(sys.join("devices/platform/amdgpu_xcp_0"))?;
        std::fs::create_dir_all(&render)?;
        std::os::unix::fs::symlink(
            "../../../devices/platform/amdgpu_xcp_0",
            render.join("device"),
        )
    }

    #[rstest]
    #[case::a_gpu_id_unreadable(gpu_id_unreadable, "an AMD GPU's id")]
    #[case::a_partition(a_partition, "is a partition")]
    fn a_card_s_sysfs_can_refuse_the_open(
        #[case] change: fn(&Path) -> std::io::Result<()>,
        #[case] cause: &str,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let sys = fake_sys(dir.path(), true)?;
        change(&sys)?;
        let err = AmdGpu::open_in(&sys, None, Settle::NONE)
            .err()
            .context("not refused")?;
        let message = format!("{:#}", anyhow::Error::from(err));
        assert!(message.contains(cause), "{message}");
        Ok(())
    }

    #[rstest]
    #[case::until_it_holds(Duration::from_secs(10), 300, true)]
    #[case::up_to_its_limit(Duration::from_millis(250), 150 << 30, false)]
    fn the_open_waits_while_a_card_frees_memory(
        #[case] limit: Duration,
        #[case] least: u64,
        #[case] settled: bool,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let sys = fake_sys(dir.path(), true)?;
        set_used(&sys, 200 << 30)?;
        let freeing = {
            let sys = sys.clone();
            std::thread::spawn(move || {
                [150 << 30, 100 << 30, 50 << 30, 300]
                    .into_iter()
                    .try_for_each(|used| {
                        std::thread::sleep(Duration::from_millis(300));
                        set_used(&sys, used)
                    })
            })
        };
        std::thread::sleep(Duration::from_millis(200));
        let settle = Settle {
            window: Duration::from_millis(400),
            limit,
        };
        let gpu = AmdGpu::open_in(&sys, None, settle)?;
        let joined = freeing.join();
        assert!(matches!(joined, Ok(Ok(()))), "{joined:?}");
        assert_eq!(gpu.least(), least);
        assert_eq!(gpu.settled, settled);
        Ok(())
    }

    /// Slurm's count, or none outside Slurm; one that is not a count is
    /// refused rather than taken for outside Slurm.
    #[rstest]
    #[case::outside_slurm(None, Ok(None))]
    #[case::a_count(Some("8"), Ok(Some(8)))]
    #[case::not_a_count(Some("8(IDX:0-7)"), Err("not a count"))]
    fn slurm_s_gpu_count_is_a_count(
        #[case] value: Option<&str>,
        #[case] expected: Result<Option<usize>, &str>,
    ) -> anyhow::Result<()> {
        match (slurm_allotted(value.map(str::to_owned)), expected) {
            (Ok(got), Ok(want)) => assert_eq!(got, want),
            (Err(err), Err(cause)) => assert!(err.to_string().contains(cause), "{err}"),
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
        Ok(())
    }
}
