//! macOS Metal probe: the DYLD-injected `peakmtl/peakmtl.m` shim, a diagnostic
//! beside `phys_footprint`. The consumer owns spawn and wait:
//!
//! ```ignore
//! let watch = Watch::Peak.supported()?;         // before anything spawns
//! let probe = metal::MetalProbeChannel::attach(&mut cmd)?;
//! let mut process = start_process("llama-bench", cmd, Duration::ZERO)?;
//! let sampler = Sampler::start(process.pid(), watch)?;
//! let (memory, peak) = process.wait_success_before_reap(timeout, || sampler.stop())?;
//! let max_host_bytes = peak?.host_bytes;      // reported
//! probe.log_peak(max_host_bytes);             // diagnostic only
//! ```
//!
//! Methodology: `docs/methodology/peak-memory-macos.md`.

use std::{
    error::Error as _,
    ffi::OsString,
    fs::{self, Permissions},
    io::{self, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

use tempfile::{Builder as TempBuilder, NamedTempFile, TempDir};

const ENV_OUT: &str = "PIPETTE_MEMPROBE_OUT";

const DYLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/peakmtl.dylib"));

/// Keys the extract's name, so a rebuild of the same length never reuses a
/// stale extract.
const DYLIB_HASH: &str = env!("PIPETTE_MEMPROBE_PEAKMTL_HASH");

/// Why [`MetalProbeChannel::attach`] failed.
#[derive(Debug, thiserror::Error)]
pub enum AttachError {
    #[error("failed to {step} {}", path.display())]
    Extract {
        step: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    #[error("failed to create the per-run tempdir for the Metal probe's output")]
    Tempdir(#[source] io::Error),
}

/// Why the shim's snapshot could not be read: logged, never fatal.
#[derive(Debug, thiserror::Error)]
enum SnapshotError {
    #[error(
        "peakmtl shim produced no output at {}: DYLD_INSERT_LIBRARIES was likely blocked by \
         Hardened Runtime / Library Validation / SIP on the target binary. See \
         docs/methodology/peak-memory-macos.md for diagnosis.",
        path.display()
    )]
    Missing { path: PathBuf },
    #[error("failed to read peakmtl output at {}", path.display())]
    Read { path: PathBuf, source: io::Error },
    #[error(
        "peakmtl snapshot at {} is missing the required `metal_peak_allocated_bytes` line; the \
         shim ran but did not surface a peak. Snapshot contents:\n{raw}",
        path.display()
    )]
    NoPeak { path: PathBuf, raw: String },
}

/// The shim's snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPeak {
    /// Peak `currentAllocatedSize` summed across devices.
    bytes: u64,
    /// The first device's `hasUnifiedMemory`; `None` when the shim saw no device.
    unified: Option<bool>,
    n_devices: Option<u32>,
}

/// One run's Metal probe. Dropping it deletes the snapshot, so call
/// [`Self::log_peak`] first.
#[must_use = "MetalProbeChannel::log_peak must be called before the channel drops; \
              the per-run tempdir is removed on drop and the snapshot is lost"]
pub struct MetalProbeChannel {
    _tempdir: TempDir,
    output_path: PathBuf,
}

impl MetalProbeChannel {
    /// Extract the bundled dylib and point `cmd`'s `DYLD_INSERT_LIBRARIES` and
    /// `PIPETTE_MEMPROBE_OUT` at it and a per-run output file.
    pub fn attach(cmd: &mut Command) -> Result<Self, AttachError> {
        let dylib = extract_peakmtl_dylib()?;
        let tempdir = TempBuilder::new()
            .prefix("pipette-memprobe-metal-")
            .tempdir()
            .map_err(AttachError::Tempdir)?;
        let output_path = tempdir.path().join("peak");

        wire_dyld_insert(cmd, &dylib);
        cmd.env(ENV_OUT, &output_path);

        Ok(Self {
            _tempdir: tempdir,
            output_path,
        })
    }

    /// Log the shim's peak beside `max_host_bytes`. `phys_footprint` already
    /// bills it on unified memory, so a missing snapshot only warns.
    pub fn log_peak(&self, max_host_bytes: u64) {
        match self.read_peak() {
            Ok(peak) => tracing::debug!(
                max_host_bytes,
                metal_peak_bytes = peak.bytes,
                metal_unified = peak.unified,
                metal_devices = peak.n_devices,
                "peak memory: phys_footprint is max_host_bytes; the Metal peak is a diagnostic \
                 (unified memory, not reported as max_gpu_bytes)"
            ),
            Err(err) => {
                let causes = std::iter::successors(err.source(), |&err| err.source())
                    .map(|cause| format!(": {cause}"))
                    .collect::<String>();
                tracing::warn!(
                    error = %format_args!("{err}{causes}"),
                    "the Metal diagnostic snapshot is unavailable; max_host_bytes \
                     (phys_footprint) is unaffected"
                );
            }
        }
    }

    fn read_peak(&self) -> Result<MetalPeak, SnapshotError> {
        read_peak(&self.output_path)
    }
}

/// Extract the embedded dylib to a hash-keyed path under `$TMPDIR`, reusing
/// one of the right length. The rename makes concurrent extracts safe.
fn extract_peakmtl_dylib() -> Result<PathBuf, AttachError> {
    let dir = std::env::temp_dir().join("pipette-memprobe-peakmtl");
    fs::create_dir_all(&dir).map_err(extract_failed("create", &dir))?;
    let name = format!("peakmtl-{DYLIB_HASH}-{}.dylib", DYLIB_BYTES.len());
    let path = dir.join(name);

    let needs_write = match fs::metadata(&path) {
        Ok(meta) => meta.len() != DYLIB_BYTES.len() as u64,
        Err(_) => true,
    };
    if needs_write {
        let mut tmp =
            NamedTempFile::new_in(&dir).map_err(extract_failed("create a tempfile in", &dir))?;
        tmp.as_file()
            .set_permissions(Permissions::from_mode(0o755))
            .map_err(extract_failed("chmod tempfile", tmp.path()))?;
        tmp.write_all(DYLIB_BYTES)
            .map_err(extract_failed("write", tmp.path()))?;
        tmp.persist(&path)
            .map_err(|err| err.error)
            .map_err(extract_failed("publish", &path))?;
    }
    Ok(path)
}

fn extract_failed(step: &'static str, path: &Path) -> impl FnOnce(io::Error) -> AttachError {
    let path = path.to_owned();
    move |source| AttachError::Extract { step, path, source }
}

/// Append `dylib` to the `DYLD_INSERT_LIBRARIES` the child would get: `cmd`'s
/// own (an `env_remove` counts as empty), else the inherited one.
fn wire_dyld_insert(cmd: &mut Command, dylib: &Path) {
    let key = OsString::from("DYLD_INSERT_LIBRARIES");
    let on_command: Option<Option<OsString>> = cmd
        .get_envs()
        .find(|(k, _)| *k == key.as_os_str())
        .map(|(_, v)| v.map(OsString::from));

    let existing: Option<OsString> = match on_command {
        Some(Some(v)) => Some(v),
        Some(None) => None,
        None => std::env::var_os("DYLD_INSERT_LIBRARIES"),
    };

    let mut value = OsString::new();
    if let Some(prev) = existing
        && !prev.is_empty()
    {
        value.push(&prev);
        value.push(":");
    }
    value.push(dylib);
    cmd.env(key, value);
}

fn read_peak(path: &Path) -> Result<MetalPeak, SnapshotError> {
    let raw = fs::read_to_string(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            SnapshotError::Missing {
                path: path.to_owned(),
            }
        } else {
            SnapshotError::Read {
                path: path.to_owned(),
                source,
            }
        }
    })?;
    parse_snapshot(&raw).ok_or_else(|| SnapshotError::NoPeak {
        path: path.to_owned(),
        raw,
    })
}

fn parse_snapshot(s: &str) -> Option<MetalPeak> {
    let (peak, unified, n_devices) = s.lines().filter_map(|line| line.split_once('=')).fold(
        (None, None, None),
        |(peak, unified, n_devices), (k, v)| match k.trim() {
            "metal_peak_allocated_bytes" => (v.trim().parse().ok(), unified, n_devices),
            "metal_unified" => {
                let unified = match v.trim() {
                    "1" => Some(true),
                    "0" => Some(false),
                    _ => None,
                };
                (peak, unified, n_devices)
            }
            "metal_devices" => (peak, unified, v.trim().parse().ok()),
            _ => (peak, unified, n_devices),
        },
    );
    Some(MetalPeak {
        bytes: peak?,
        unified,
        n_devices,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    #[test]
    fn parses_snapshot_full() -> anyhow::Result<()> {
        let s = "\
metal_peak_allocated_bytes=940261376
metal_peak_recommended_max_bytes=30150672384
metal_unified=1
metal_devices=1
";
        let p = parse_snapshot(s).context("parsed")?;
        assert_eq!(p.bytes, 940_261_376);
        assert_eq!(p.unified, Some(true));
        assert_eq!(p.n_devices, Some(1));
        Ok(())
    }

    #[test]
    fn parse_snapshot_returns_none_on_empty() {
        assert!(parse_snapshot("").is_none());
    }

    #[test]
    fn parse_snapshot_returns_none_when_peak_missing() {
        assert!(parse_snapshot("metal_unified=1\nmetal_devices=1\n").is_none());
    }

    #[test]
    fn extract_peakmtl_dylib_is_idempotent_and_content_keyed() -> anyhow::Result<()> {
        let p1 = extract_peakmtl_dylib().context("extract")?;
        let p2 = extract_peakmtl_dylib().context("extract")?;
        assert_eq!(p1, p2);
        assert!(
            p1.file_name()
                .context("dylib path has no file name")?
                .to_string_lossy()
                .contains(DYLIB_HASH)
        );
        Ok(())
    }

    #[test]
    fn wire_dyld_insert_preserves_command_env_entry() -> anyhow::Result<()> {
        let dylib = PathBuf::from("/tmp/peakmtl.dylib");
        let mut cmd = Command::new("true");
        cmd.env("DYLD_INSERT_LIBRARIES", "/caller/foo.dylib");
        wire_dyld_insert(&mut cmd, &dylib);
        let v = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new("DYLD_INSERT_LIBRARIES"))
            .and_then(|(_, v)| v)
            .map(|v| v.to_string_lossy().into_owned())
            .context("DYLD_INSERT_LIBRARIES not set on command")?;
        assert_eq!(v, "/caller/foo.dylib:/tmp/peakmtl.dylib");
        Ok(())
    }

    #[test]
    fn read_peak_errors_with_dyld_blocked_message_on_missing_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("create tempdir")?;
        let path = dir.path().join("does-not-exist");
        let Err(err) = read_peak(&path) else {
            anyhow::bail!("read_peak unexpectedly succeeded for missing file");
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("DYLD_INSERT_LIBRARIES was likely blocked"),
            "expected DYLD-blocked message, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn read_peak_errors_when_snapshot_present_but_missing_required_field() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("create tempdir")?;
        let path = dir.path().join("peak");
        std::fs::write(&path, "metal_unified=1\nmetal_devices=1\n").context("write snapshot")?;
        let Err(err) = read_peak(&path) else {
            anyhow::bail!("read_peak unexpectedly succeeded for incomplete snapshot");
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("missing the required `metal_peak_allocated_bytes`"),
            "expected shim-anomaly message, got: {msg}"
        );
        Ok(())
    }

    #[test]
    fn read_peak_round_trip_via_simulated_shim() -> anyhow::Result<()> {
        let dir = tempfile::tempdir().context("create tempdir")?;
        let path = dir.path().join("peak");
        std::fs::write(
            &path,
            "metal_peak_allocated_bytes=1234567\n\
             metal_peak_recommended_max_bytes=99999\n\
             metal_unified=1\n\
             metal_devices=2\n",
        )
        .context("write snapshot")?;
        let p = read_peak(&path).context("read_peak")?;
        assert_eq!(p.bytes, 1_234_567);
        assert_eq!(p.unified, Some(true));
        assert_eq!(p.n_devices, Some(2));
        Ok(())
    }

    #[test]
    fn metal_probe_channel_attach_wires_envs_and_owns_tempdir() -> anyhow::Result<()> {
        let mut cmd = Command::new("true");
        let probe = MetalProbeChannel::attach(&mut cmd).context("attach")?;

        let parent = probe
            .output_path
            .parent()
            .context("output_path has no parent")?
            .to_path_buf();
        assert!(parent.exists());

        let env_out = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new(ENV_OUT))
            .and_then(|(_, v)| v)
            .map(|v| v.to_string_lossy().into_owned())
            .context("ENV_OUT not set on command")?;
        assert_eq!(env_out, probe.output_path.to_string_lossy());

        let dyld = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new("DYLD_INSERT_LIBRARIES"))
            .and_then(|(_, v)| v)
            .map(|v| v.to_string_lossy().into_owned())
            .context("DYLD_INSERT_LIBRARIES not set on command")?;
        assert!(dyld.contains("peakmtl-"), "got: {dyld}");

        let Err(err) = probe.read_peak() else {
            anyhow::bail!("read_peak unexpectedly succeeded with no shim output");
        };
        assert!(format!("{err:#}").contains("DYLD_INSERT_LIBRARIES"));

        drop(probe);
        assert!(!parent.exists());
        Ok(())
    }
}
