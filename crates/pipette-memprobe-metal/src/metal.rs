//! macOS Metal probe — DYLD-injected `peakmtl.dylib` shim.
//!
//! Public API is a single channel type plus the parsed-snapshot shape:
//!
//! - [`MetalProbeChannel::attach`] — extract the bundled dylib,
//!   allocate a per-run output tempfile, wire
//!   `DYLD_INSERT_LIBRARIES` and `PIPETTE_MEMPROBE_OUT` into the
//!   `Command`. Returns the channel handle.
//! - [`MetalProbeChannel::read_peak`] — after the child exits, read
//!   the snapshot the shim wrote. Errors are operator-actionable:
//!   missing snapshot file (DYLD blocked), snapshot present but
//!   missing the required `metal_peak_allocated_bytes` line (shim
//!   anomaly), or I/O error.
//! - [`MetalPeak`] — parsed snapshot: bytes (peak
//!   `[MTLDevice currentAllocatedSize]`) plus diagnostic fields.
//!
//! The crate doesn't own the spawn / wait pipeline; the consumer
//! does. Typical Mac consumer pattern:
//!
//! ```ignore
//! let probe = metal::MetalProbeChannel::attach(&mut cmd)?;
//! cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
//! let child = cmd.spawn()?;
//! let phys_poller = host::spawn_phys_footprint_poller(child.id() as i32);
//! let output = child.wait_with_output()?;
//! let max_host_bytes = phys_poller.stop_and_join()?;  // peak phys_footprint
//! let max_gpu_bytes  = probe.read_peak()?.bytes;      // peak Metal allocator
//! // Each peak is reported independently; on Apple UMA they overlap.
//! ```
//!
//! See `docs/methodology/peak-memory-usage.md` for the methodology, and
//! `peakmtl/peakmtl.m` for the in-process shim.

use std::{
    ffi::OsString,
    fs::{self, Permissions},
    io::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;
use tempfile::{Builder as TempBuilder, NamedTempFile, TempDir};

/// Env var the shim reads to find its output tempfile. Set on the
/// child `Command` by [`MetalProbeChannel::attach`]; consumers do not
/// reference it directly.
const ENV_OUT: &str = "PIPETTE_MEMPROBE_OUT";

const DYLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/peakmtl.dylib"));

/// SHA-256 (truncated to 16 hex chars) of `DYLIB_BYTES`, computed at
/// build time. Used to key the extracted file's name so a rebuild
/// that produces different bytes (even with the same length) doesn't
/// reuse a stale extract.
const DYLIB_HASH: &str = env!("PIPETTE_MEMPROBE_PEAKMTL_HASH");

/// Snapshot the shim wrote to the per-run tempfile. `bytes` is the
/// peak `[MTLDevice currentAllocatedSize]` summed across devices;
/// the others are diagnostic and only present when the shim's
/// `atexit` handler fired.
///
/// `#[non_exhaustive]`: future diagnostic fields can be added without
/// breaking external destructure / construction sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MetalPeak {
    /// Peak `[MTLDevice currentAllocatedSize]` summed across devices.
    pub bytes: u64,
    /// `[MTLDevice hasUnifiedMemory]` from the first device. `None`
    /// when the shim's atexit handler didn't fire (CPython + MLX
    /// `_exit()` path). Diagnostic only — consumers don't branch on
    /// it today; each peak is reported independently regardless of
    /// UMA vs discrete.
    pub unified: Option<bool>,
    /// Number of `MTLDevice`s seen by the shim. `None` when the
    /// atexit handler didn't fire. Diagnostic only.
    pub n_devices: Option<u32>,
}

/// One end-to-end Metal-probe attachment: owns the per-run tempdir
/// and the resolved output path, holds them alive for the lifetime
/// of the measurement.
///
/// Drop order matters: the `TempDir` is removed when the channel
/// drops, which deletes the snapshot file. Read it via
/// [`Self::read_peak`] *before* the channel goes out of scope —
/// the `#[must_use]` warning catches the case where a probe is
/// attached but the snapshot is never read.
#[must_use = "MetalProbeChannel::read_peak must be called before the channel drops; \
              the per-run tempdir is removed on drop and the snapshot is lost"]
pub struct MetalProbeChannel {
    /// Kept for its `Drop` — removes the tempdir when the channel
    /// drops. Not read directly.
    _tempdir: TempDir,
    output_path: PathBuf,
}

impl MetalProbeChannel {
    /// Extract the bundled dylib (cached by content hash), allocate
    /// a per-run output tempfile, wire `DYLD_INSERT_LIBRARIES` and
    /// `PIPETTE_MEMPROBE_OUT` into `cmd`. Returns the channel; the
    /// caller spawns / waits / reads.
    ///
    /// `DYLD_INSERT_LIBRARIES` resolution preserves any existing
    /// entry: a value already set on `cmd` wins, otherwise the
    /// parent process's value is appended-to, otherwise the shim is
    /// the only entry.
    pub fn attach(cmd: &mut Command) -> anyhow::Result<Self> {
        let dylib = extract_peakmtl_dylib()?;
        let tempdir = TempBuilder::new()
            .prefix("pipette-memprobe-metal-")
            .tempdir()
            .context("failed to create per-run tempdir for Metal probe output")?;
        let output_path = tempdir.path().join("peak");

        wire_dyld_insert(cmd, &dylib);
        cmd.env(ENV_OUT, &output_path);

        Ok(Self {
            _tempdir: tempdir,
            output_path,
        })
    }

    /// Read and parse the shim's snapshot file. Distinguishes three
    /// failure modes:
    ///
    /// - **Snapshot file missing**: the shim never wrote anything,
    ///   almost always because `DYLD_INSERT_LIBRARIES` was blocked
    ///   by Hardened Runtime / Library Validation / SIP on the
    ///   target binary. Returns an `Err` pointing operators at the
    ///   macOS section of `docs/methodology/peak-memory-usage.md`.
    /// - **Snapshot file present but missing `metal_peak_allocated_bytes`**:
    ///   the shim ran but didn't surface a peak. Indicates a
    ///   shim-side bug or a Metal API anomaly worth investigating.
    /// - **I/O error**: anything else (permissions, disk).
    pub fn read_peak(&self) -> anyhow::Result<MetalPeak> {
        read_peak(&self.output_path)
    }
}

// ────────────────────────────────────────────────────────────────────
// Internals (private)
// ────────────────────────────────────────────────────────────────────

/// Extract the embedded `peakmtl.dylib` to a content-hash-keyed path
/// under `$TMPDIR/pipette-memprobe-peakmtl/`. Reuses an existing
/// extraction at the hash-keyed path when its length matches. The
/// hash is part of the filename, so a rebuild that produces
/// different bytes lands at a different path. Race-safe across
/// concurrent processes via `tempfile::NamedTempFile` + atomic-rename
/// publish.
fn extract_peakmtl_dylib() -> anyhow::Result<PathBuf> {
    let dir = std::env::temp_dir().join("pipette-memprobe-peakmtl");
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let name = format!("peakmtl-{DYLIB_HASH}-{}.dylib", DYLIB_BYTES.len());
    let path = dir.join(name);

    let needs_write = match fs::metadata(&path) {
        Ok(meta) => meta.len() != DYLIB_BYTES.len() as u64,
        Err(_) => true,
    };
    if needs_write {
        let mut tmp = NamedTempFile::new_in(&dir)
            .with_context(|| format!("failed to create tempfile in {}", dir.display()))?;
        tmp.as_file()
            .set_permissions(Permissions::from_mode(0o755))
            .with_context(|| format!("failed to chmod tempfile {}", tmp.path().display()))?;
        tmp.write_all(DYLIB_BYTES)
            .with_context(|| format!("failed to write {}", tmp.path().display()))?;
        tmp.persist(&path)
            .with_context(|| format!("failed to publish {}", path.display()))?;
    }
    Ok(path)
}

/// Append `dylib` to `cmd`'s `DYLD_INSERT_LIBRARIES`, preserving any
/// existing entries.
///
/// Resolution order (first match wins):
///   1. an entry already set on `cmd` via `cmd.env(...)`
///      (caller's configure step);
///   2. the parent process's `DYLD_INSERT_LIBRARIES` (inherited);
///   3. neither set — the shim is the only entry.
///
/// `cmd.env_remove("DYLD_INSERT_LIBRARIES")`-cleared entries are
/// honored as "no existing value."
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
    if let Some(prev) = existing {
        if !prev.is_empty() {
            value.push(&prev);
            value.push(":");
        }
    }
    value.push(dylib);
    cmd.env(key, value);
}

fn read_peak(path: &Path) -> anyhow::Result<MetalPeak> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(anyhow::anyhow!(
                "peakmtl shim produced no output at {}: \
                 DYLD_INSERT_LIBRARIES was likely blocked by Hardened \
                 Runtime / Library Validation / SIP on the target \
                 binary. See docs/methodology/peak-memory-usage.md \
                 (macOS section) for diagnosis.",
                path.display()
            ));
        }
        Err(e) => {
            return Err(e)
                .with_context(|| format!("failed to read peakmtl output at {}", path.display()));
        }
    };
    parse_snapshot(&raw).ok_or_else(|| {
        anyhow::anyhow!(
            "peakmtl snapshot at {} is missing the required \
             `metal_peak_allocated_bytes` line; the shim ran but did \
             not surface a peak. Snapshot contents:\n{}",
            path.display(),
            raw
        )
    })
}

fn parse_snapshot(s: &str) -> Option<MetalPeak> {
    let mut peak = None;
    let mut unified = None;
    let mut n_devices = None;
    for line in s.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "metal_peak_allocated_bytes" => peak = v.trim().parse().ok(),
            "metal_unified" => {
                unified = match v.trim() {
                    "1" => Some(true),
                    "0" => Some(false),
                    _ => None,
                }
            }
            "metal_devices" => n_devices = v.trim().parse().ok(),
            // Other shim-emitted keys (e.g. metal_peak_recommended_max_bytes)
            // are accepted but ignored — forward-compat with shim
            // additions.
            _ => {}
        }
    }
    let bytes = peak?;
    Some(MetalPeak {
        bytes,
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
        // metal_peak_recommended_max_bytes is now accepted-and-ignored
        // (forward-compat with shim additions).
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
        assert!(p1
            .file_name()
            .context("dylib path has no file name")?
            .to_string_lossy()
            .contains(DYLIB_HASH));
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
        let err = match read_peak(&path) {
            Ok(_) => anyhow::bail!("read_peak unexpectedly succeeded for missing file"),
            Err(e) => e,
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
        let err = match read_peak(&path) {
            Ok(_) => anyhow::bail!("read_peak unexpectedly succeeded for incomplete snapshot"),
            Err(e) => e,
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

        // Tempdir under output_path's parent should exist while the
        // channel is alive.
        let parent = probe
            .output_path
            .parent()
            .context("output_path has no parent")?
            .to_path_buf();
        assert!(parent.exists());

        // ENV_OUT is set on the command and points at output_path.
        let env_out = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new(ENV_OUT))
            .and_then(|(_, v)| v)
            .map(|v| v.to_string_lossy().into_owned())
            .context("ENV_OUT not set on command")?;
        assert_eq!(env_out, probe.output_path.to_string_lossy());

        // DYLD_INSERT_LIBRARIES is set on the command and contains a
        // peakmtl-* path.
        let dyld = cmd
            .get_envs()
            .find(|(k, _)| *k == std::ffi::OsStr::new("DYLD_INSERT_LIBRARIES"))
            .and_then(|(_, v)| v)
            .map(|v| v.to_string_lossy().into_owned())
            .context("DYLD_INSERT_LIBRARIES not set on command")?;
        assert!(dyld.contains("peakmtl-"), "got: {dyld}");

        // Read with no shim output: NotFound → DYLD-blocked Err.
        let err = match probe.read_peak() {
            Ok(_) => anyhow::bail!("read_peak unexpectedly succeeded with no shim output"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("DYLD_INSERT_LIBRARIES"));

        // Drop the channel; tempdir is removed.
        drop(probe);
        assert!(!parent.exists());
        Ok(())
    }
}
