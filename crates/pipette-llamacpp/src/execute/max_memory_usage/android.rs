//! Android measurement path: wraps `llama-bench` with vendored toybox
//! `time -v` and parses its `Max RSS (KiB):` line for `max_host_bytes`.
//!
//! Why a wrapper instead of `pipette-memprobe`'s `wait4 ru_maxrss`:
//!
//! - The kernel data is identical (toybox's `time` reads the same
//!   `wait4` rusage) but the wrapper is a single self-contained step
//!   we can run by hand to debug ("just run `toybox time -v llama-bench
//!   ...` on the device") and that doesn't depend on Rust's
//!   `proc_pid_rusage` / `wait4` plumbing being perfect on every
//!   bionic version.
//! - `toybox-aarch64` is statically linked (no shared-library
//!   dependencies), so dropping it onto a stock Android device works
//!   without any system-image changes or vendor-specific tooling.
//! - The binary is shipped via Git LFS at
//!   `crates/pipette-llamacpp/vendor/toybox/toybox-aarch64`
//!   (origin: <https://landley.net/toybox/bin/toybox-aarch64>).
//!
//! `max_gpu_bytes` is `null` on this flavor: Mali GPUs don't expose
//! DRM fdinfo, and there's no in-process Mali probe. Adreno + MSM-DRM
//! could populate it via DRM fdinfo in a future build; the work
//! belongs in `pipette-memprobe::os_counters::linux` if/when wanted.

use std::os::unix::process::CommandExt;
use std::{
    fs::{self, Permissions},
    io::Write,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Context;
use tempfile::NamedTempFile;

use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::RuntimeFlags;
use pipette_subprocess::{argv, echo_info};

use super::super::RunResponse;
use super::Params;
use crate::common::{
    apply_dylib_search_env, deadline_error_message, spawn_timeout_killer, MAX_MEMORY_USAGE_TIMEOUT,
};

/// Embedded toybox-aarch64 binary, served via Git LFS in
/// `vendor/toybox/`. The whole module is gated to Android in
/// `mod.rs`, and Android cross-compiles always target aarch64, so we
/// can include the binary unconditionally here without bloating
/// non-Android builds.
const TOYBOX_BYTES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/toybox/toybox-aarch64"
));

pub(super) fn run(
    llama_bench: &Path,
    params: Params,
    model_path: &Path,
    extra_flags: &[String],
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponse> {
    let toybox_path = extract_toybox().context("failed to extract vendored toybox")?;

    let mut cmd = Command::new(&toybox_path);
    cmd.arg("time").arg("-v").arg(llama_bench);
    // env on the toybox parent inherits to the wrapped llama-bench
    // child after exec, which is what we want.
    apply_dylib_search_env(&mut cmd, llama_bench);
    cmd.arg("--output").arg("json");
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_flags);
    // --n-gen 1: exercise one decode step so we count decode-path
    // backend kernels + sampling buffers in the peak. Prefill alone
    // misses those — they're created on first decode and persist.
    // See pipette-mgmt/docs/methodology/peak-memory.md §2.4
    // "Workload phases captured" for rationale.
    cmd.arg("--n-prompt")
        .arg(params.parameter_prefill_tokens.to_string());
    cmd.arg("--n-gen").arg("1");
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // Run toybox in its own process group so a SIGKILL to -pid takes
    // down both toybox and the llama-bench it execs. Without this, a
    // SIGKILL on the toybox pid alone would orphan llama-bench (it
    // would keep running until it exited on its own — defeating the
    // deadline).
    //
    // Propagate setpgid's errno: if it failed and we then sent
    // `kill(-pid, SIGKILL)`, we'd target whatever pgid happens to
    // equal the child's pid (most likely no group, but possibly the
    // wrong one). Returning Err from `pre_exec` aborts the exec —
    // the safe failure mode.
    //
    // SAFETY: `setpgid(0, 0)` is async-signal-safe and the only call
    // we make between fork and exec. No allocators, no locks.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let preview = argv(&cmd);
    echo_info(&cmd);

    let child = cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", toybox_path.display()))?;
    let pgid = child.id() as libc::pid_t;
    let killer = spawn_timeout_killer(MAX_MEMORY_USAGE_TIMEOUT, move || unsafe {
        // Negative pid → kill the whole process group; takes both
        // toybox and its llama-bench child.
        libc::kill(-pgid, libc::SIGKILL);
    });
    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {}", toybox_path.display()))?;
    let killer_fired = killer.fired();
    drop(killer);

    if killer_fired {
        anyhow::bail!(
            "{}",
            deadline_error_message(
                MAX_MEMORY_USAGE_TIMEOUT,
                &String::from_utf8_lossy(&output.stderr)
            )
        );
    }
    if !output.status.success() {
        anyhow::bail!(
            "toybox time -v llama-bench failed: {}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8")?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let max_rss_kib = parse_max_rss_kib(&stderr).ok_or_else(|| {
        anyhow::anyhow!("toybox `time -v` output did not contain `Max RSS (KiB):`")
    })?;
    let max_host_bytes = max_rss_kib.saturating_mul(1024);
    log::debug!(
        "android max_host_bytes = {max_host_bytes} ({max_rss_kib} KiB) from toybox time -v"
    );

    Ok(RunResponse {
        executable: Some(llama_bench.display().to_string()),
        command: preview,
        runtime_flags: Some(flags.clone()),
        ..RunResponse::new(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
            stdout,
            stderr,
        )
    })
}

/// Parse `Max RSS (KiB): <N>` out of toybox `time -v`'s summary
/// (which lands at the end of stderr after the wrapped command).
fn parse_max_rss_kib(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("Max RSS (KiB):") else {
            continue;
        };
        let v = rest.trim().parse::<u64>().ok()?;
        return Some(v);
    }
    None
}

/// Extract the embedded toybox binary to a content-keyed path under
/// `$TMPDIR/pipette-llamacpp-toybox/`. Same race-safety pattern as
/// `pipette_memprobe_metal::metal` extraction —
/// `NamedTempFile` + atomic rename publish, content-length check
/// skips re-extraction.
fn extract_toybox() -> anyhow::Result<PathBuf> {
    let dir = std::env::temp_dir().join("pipette-llamacpp-toybox");
    fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let path = dir.join(format!("toybox-{}", TOYBOX_BYTES.len()));
    let needs_write = match fs::metadata(&path) {
        Ok(meta) => meta.len() != TOYBOX_BYTES.len() as u64,
        Err(_) => true,
    };
    if needs_write {
        let mut tmp = NamedTempFile::new_in(&dir)
            .with_context(|| format!("failed to create tempfile in {}", dir.display()))?;
        tmp.as_file()
            .set_permissions(Permissions::from_mode(0o755))
            .with_context(|| format!("failed to chmod {}", tmp.path().display()))?;
        tmp.write_all(TOYBOX_BYTES)
            .with_context(|| format!("failed to write {}", tmp.path().display()))?;
        tmp.persist(&path)
            .with_context(|| format!("failed to publish {}", path.display()))?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_max_rss_from_toybox_output() {
        let stderr = "\
load_backend: loaded CPU backend from /data/.../libggml-cpu.so
... lots of llama.cpp output ...
Real time (s): 0.648157
User time (s): 2.594882
System time (s): 0.106206
Max RSS (KiB): 347644
Major faults: 0
";
        assert_eq!(parse_max_rss_kib(stderr), Some(347_644));
    }

    #[test]
    fn returns_none_when_max_rss_absent() {
        assert!(parse_max_rss_kib("Real time (s): 0.001\n").is_none());
    }

    #[test]
    fn extract_toybox_writes_executable() -> anyhow::Result<()> {
        let path = extract_toybox().context("extract")?;
        let meta = std::fs::metadata(&path)?;
        assert_eq!(meta.len(), TOYBOX_BYTES.len() as u64);
        // 0o755: owner rwx, group/other rx
        let mode = std::os::unix::fs::PermissionsExt::mode(&meta.permissions());
        assert_eq!(mode & 0o111, 0o111, "toybox should be executable");
        Ok(())
    }
}
