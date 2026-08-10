//! Peak memory probe for a running Docker container ([`Probe`]).
//!
//! Linux + Docker only. Assumes pipette-torch-oai runs on the docker host (not
//! itself inside a container); host-namespaced PIDs from `docker top` are
//! cross-referenced against `nvidia-smi`. Two channels:
//!
//! - **Host (CPU/RAM):** read `/sys/fs/cgroup/memory.peak` from inside the
//!   container via `docker exec`. This is the kernel-tracked peak under
//!   cgroup v2, so a single read at the end of a run is enough — no
//!   sampling drift.
//! - **GPU (NVIDIA):** poll `nvidia-smi --query-compute-apps=pid,used_memory
//!   --format=csv,noheader` and sum rows whose PID belongs to the container.
//!   Container PIDs come from `docker top <id>`. NVML reports per-process
//!   committed memory, not peak, so [`Probe`] runs a polling thread and
//!   tracks the running max.
//!
//! `nvidia-smi` is invoked on the host (not inside the container) so that a
//! broken in-container NVIDIA driver or a `--gpus none` launch still produces
//! a host reading; the GPU channel just returns `None`.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::Context;

/// One read of the container's current memory footprint.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Cgroup `memory.current` (bytes).
    pub host_bytes: u64,
    /// Sum of `nvidia-smi --query-compute-apps used_memory` rows for the
    /// container's PIDs, in bytes. `None` when `nvidia-smi` isn't reachable
    /// or no compute-apps row matches.
    pub gpu_bytes: Option<u64>,
}

/// Peak readings collected by a [`Probe`].
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    /// Cgroup `memory.peak` (kernel-tracked, in bytes).
    pub max_host_bytes: u64,
    /// Running max of GPU samples, in bytes. `None` when no GPU sample
    /// succeeded during the run.
    pub max_gpu_bytes: Option<u64>,
    /// Number of GPU samples collected.
    pub samples: u64,
}

/// One-shot sample: cgroup `memory.current` + an attempt at GPU current.
pub fn sample(docker: &Path, container_id: &str) -> anyhow::Result<Sample> {
    let host_bytes = host_current(docker, container_id)?;
    let gpu_bytes = gpu_current(docker, container_id).ok().flatten();
    Ok(Sample {
        host_bytes,
        gpu_bytes,
    })
}

/// Cgroup v2 `memory.current` (bytes), read from inside the container.
fn host_current(docker: &Path, container_id: &str) -> anyhow::Result<u64> {
    read_cgroup_value(docker, container_id, "memory.current")
}

/// Cgroup v2 `memory.peak` (bytes), read from inside the container.
///
/// Cgroup v2 tracks this in-kernel for the lifetime of the cgroup, so a
/// single read at the end of a run yields the true peak with no sampling
/// loss. Requires a kernel with `memory.peak` support (Linux 5.19+);
/// returns an error otherwise.
fn host_peak(docker: &Path, container_id: &str) -> anyhow::Result<u64> {
    read_cgroup_value(docker, container_id, "memory.peak")
}

fn read_cgroup_value(docker: &Path, container_id: &str, name: &str) -> anyhow::Result<u64> {
    let path = format!("/sys/fs/cgroup/{name}");
    let mut cmd = Command::new(docker);
    cmd.args(["exec", container_id, "cat", &path]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to run docker exec {container_id} cat {path}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "docker exec {container_id} cat {path} failed ({}): {}",
            output.status,
            stderr.trim()
        );
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("cgroup file {path} produced empty output");
    }
    if trimmed == "max" {
        anyhow::bail!("cgroup file {path} reports 'max' (unset); cgroup v2 required");
    }
    trimmed
        .parse::<u64>()
        .with_context(|| format!("failed to parse cgroup {path}: {trimmed:?}"))
}

/// PIDs reported by `docker top <container>` (host-namespaced).
fn container_pids(docker: &Path, container_id: &str) -> anyhow::Result<Vec<u32>> {
    // `docker top` accepts ps-style options after the container id; with
    // `-eo pid` we get a header line followed by one PID per line.
    let mut cmd = Command::new(docker);
    cmd.args(["top", container_id, "-eo", "pid"]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to run docker top {container_id}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "docker top {container_id} failed ({}): {}",
            output.status,
            stderr.trim()
        );
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    Ok(raw
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect())
}

/// Current GPU memory committed by the container's processes, in bytes.
///
/// Returns `Ok(None)` when `nvidia-smi` is reachable but reports no
/// compute-apps row matching any of the container's PIDs (e.g. CPU-only
/// container, or the model hasn't allocated yet). Returns `Err` when
/// `nvidia-smi` itself fails (driver mismatch, not installed, etc.).
///
/// Nvidia-only — per-flavor GPU memory sampling for AMD / CPU lives outside
/// this path. AMD / CPU runtimes just won't have a matching PID row and the
/// probe returns `None`.
fn gpu_current(docker: &Path, container_id: &str) -> anyhow::Result<Option<u64>> {
    let pids = container_pids(docker, container_id)?;
    if pids.is_empty() {
        return Ok(None);
    }
    let nvidia_smi = pipette_subprocess::which("nvidia-smi").context(
        "nvidia-smi not found on PATH; install the NVIDIA driver to enable GPU memory probing",
    )?;
    let mut cmd = Command::new(&nvidia_smi);
    cmd.args([
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to run {}", nvidia_smi.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "nvidia-smi --query-compute-apps failed ({}): {}",
            output.status,
            stderr.trim()
        );
    }
    let mut total: u64 = 0;
    let mut matched = false;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let parts: Vec<&str> = line.splitn(2, ',').collect();
        if parts.len() != 2 {
            continue;
        }
        let Ok(pid) = parts[0].trim().parse::<u32>() else {
            continue;
        };
        if !pids.contains(&pid) {
            continue;
        }
        let bytes = parse_nvidia_smi_bytes(parts[1].trim())?;
        total = total.saturating_add(bytes);
        matched = true;
    }
    Ok(matched.then_some(total))
}

/// `nvidia-smi` returns strings like `"1234 MiB"`. Parse number + IEC unit
/// into bytes. The unit is always MiB in current driver versions, but
/// accept KiB/GiB defensively in case that changes.
fn parse_nvidia_smi_bytes(s: &str) -> anyhow::Result<u64> {
    let mut parts = s.split_whitespace();
    let value = parts.next().context("missing value in nvidia-smi cell")?;
    let unit = parts.next().unwrap_or("MiB");
    let value: u64 = value
        .parse()
        .with_context(|| format!("failed to parse nvidia-smi value: {value:?}"))?;
    let bytes = match unit {
        "B" => value,
        "KiB" => value * 1024,
        "MiB" => value * 1024 * 1024,
        "GiB" => value * 1024 * 1024 * 1024,
        other => anyhow::bail!("unrecognized nvidia-smi unit: {other:?}"),
    };
    Ok(bytes)
}

/// Background poller that tracks GPU peak memory; host peak is read from
/// the kernel at [`Probe::stop`] time. Drop without stop signals the
/// thread to exit and discards the reading.
pub struct Probe {
    docker: PathBuf,
    container_id: String,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    state: Arc<ProbeState>,
}

struct ProbeState {
    max_gpu_bytes: AtomicU64,
    gpu_seen: AtomicBool,
    samples: AtomicU64,
}

impl Probe {
    /// Start polling. Returns immediately. Host peak isn't polled here —
    /// cgroup v2 tracks it kernel-side and is read on [`Self::stop`].
    pub fn start(docker: &Path, container_id: &str, interval: Duration) -> anyhow::Result<Self> {
        // Fail fast if the container isn't reachable so callers see the error
        // before they start a benchmark.
        let _ = host_current(docker, container_id).context(
            "failed initial cgroup read; ensure the container is running and uses cgroup v2",
        )?;

        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(ProbeState {
            max_gpu_bytes: AtomicU64::new(0),
            gpu_seen: AtomicBool::new(false),
            samples: AtomicU64::new(0),
        });
        let handle = {
            let docker = docker.to_path_buf();
            let container_id = container_id.to_string();
            let stop = stop.clone();
            let state = state.clone();
            thread::Builder::new()
                .name("pipette-torch-oai-memprobe".into())
                .spawn(move || poll_loop(docker, container_id, interval, stop, state))
                .context("failed to spawn memprobe thread")?
        };
        Ok(Self {
            docker: docker.to_path_buf(),
            container_id: container_id.to_string(),
            stop,
            handle: Some(handle),
            state,
        })
    }

    /// Stop the poller, then read the kernel-tracked host peak.
    pub fn stop(mut self) -> anyhow::Result<Peak> {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
        let max_host_bytes = host_peak(&self.docker, &self.container_id)?;
        let samples = self.state.samples.load(Ordering::SeqCst);
        let max_gpu_bytes = if self.state.gpu_seen.load(Ordering::SeqCst) {
            Some(self.state.max_gpu_bytes.load(Ordering::SeqCst))
        } else {
            None
        };
        Ok(Peak {
            max_host_bytes,
            max_gpu_bytes,
            samples,
        })
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            handle.join().ok();
        }
    }
}

fn poll_loop(
    docker: PathBuf,
    container_id: String,
    interval: Duration,
    stop: Arc<AtomicBool>,
    state: Arc<ProbeState>,
) {
    let mut next_tick = Instant::now();
    while !stop.load(Ordering::SeqCst) {
        let now = Instant::now();
        if now < next_tick {
            // Sleep in short chunks so stop() doesn't wait a full interval.
            let remaining = next_tick - now;
            let chunk = remaining.min(Duration::from_millis(100));
            thread::sleep(chunk);
            continue;
        }
        next_tick = now + interval;

        match gpu_current(&docker, &container_id) {
            Ok(Some(bytes)) => {
                state.gpu_seen.store(true, Ordering::SeqCst);
                state.max_gpu_bytes.fetch_max(bytes, Ordering::SeqCst);
            }
            Ok(None) => {}
            Err(err) => {
                log::debug!("memprobe gpu sample failed: {err:#}");
            }
        }
        state.samples.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_bytes_mib() -> anyhow::Result<()> {
        assert_eq!(parse_nvidia_smi_bytes("1024 MiB")?, 1024 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_bytes_default_unit_is_mib() -> anyhow::Result<()> {
        // `--format=csv,noheader,nounits` strips the unit; assume MiB.
        assert_eq!(parse_nvidia_smi_bytes("256")?, 256 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_bytes_kib() -> anyhow::Result<()> {
        assert_eq!(parse_nvidia_smi_bytes("512 KiB")?, 512 * 1024);
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_bytes_gib() -> anyhow::Result<()> {
        assert_eq!(parse_nvidia_smi_bytes("2 GiB")?, 2u64 * 1024 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn parse_nvidia_smi_bytes_rejects_unknown_unit() {
        assert!(parse_nvidia_smi_bytes("1 foo").is_err());
    }

    #[test]
    fn parse_nvidia_smi_bytes_rejects_garbage() {
        assert!(parse_nvidia_smi_bytes("not a number").is_err());
    }
}
