//! Cross-invocation cleanup for MLX servers whose CLI died without unwinding.
//!
//! [`super::server::ServerHandle`] covers the normal and panic paths, and the
//! shared `cleanup` registry covers ^C / SIGTERM. Neither runs under `SIGKILL`
//! — which on this workload most often means jetsam under memory pressure —
//! and a leaked `mlx_lm` process keeps a multi-GB model resident, making the
//! next run more likely to be killed the same way.
//!
//! macOS has no `/proc`, so unlike torch-oai's `/proc`-walking reaper this
//! reads `ps`. A candidate must satisfy both conditions:
//!
//! 1. its command line runs a materialized server script from *this user's*
//!    temp dir (`std::env::temp_dir()` is per-user on macOS), and
//! 2. its parent is `launchd` (pid 1), meaning the CLI that spawned it is gone
//!
//! A concurrently running server fails (2) — its parent is the live CLI — so a
//! parallel run is never touched.

use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::Context;

use super::server::server_script_marker;

/// Grace between `SIGTERM` and `SIGKILL`. The server holds no state worth
/// flushing, so this only lets it release the GPU cleanly.
const TERM_GRACE: Duration = Duration::from_millis(500);

/// `SIGTERM` every orphaned MLX server, then `SIGKILL` whatever survives.
///
/// Best-effort: a `ps` that fails to run is logged and ignored, since failing
/// to clean up a previous run must not fail this one.
pub fn reap_orphan_servers() {
    let marker = server_script_marker();
    let output = match ps_snapshot() {
        Ok(out) => out,
        Err(err) => {
            log::debug!("skipping MLX orphan reap: {err:#}");
            return;
        }
    };

    let orphans = orphans_from_ps_output(&output, &marker);
    if orphans.is_empty() {
        return;
    }
    log::info!(
        "reaping {} orphaned MLX server(s) from an earlier run: {orphans:?}",
        orphans.len()
    );
    for pid in &orphans {
        signal(*pid, "TERM");
    }
    thread::sleep(TERM_GRACE);
    for pid in &orphans {
        // `kill -0` probes liveness without signalling.
        if signal_succeeds(*pid, "0") {
            log::warn!("MLX server {pid} ignored SIGTERM; sending SIGKILL");
            signal(*pid, "KILL");
        }
    }
}

fn ps_snapshot() -> anyhow::Result<String> {
    let output = Command::new("ps")
        .args(["-axo", "pid=,ppid=,command="])
        .output()
        .context("failed to run `ps`")?;
    if !output.status.success() {
        anyhow::bail!("`ps` exited with {}", output.status);
    }
    String::from_utf8(output.stdout).context("non-UTF-8 `ps` output")
}

/// PIDs whose command line contains `marker` and whose parent is pid 1.
///
/// Split out from the signalling so the selection rule — the part that decides
/// what gets killed — is testable without spawning anything.
fn orphans_from_ps_output(ps_output: &str, marker: &str) -> Vec<u32> {
    ps_output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid: u32 = fields.next()?.parse().ok()?;
            let ppid: u32 = fields.next()?.parse().ok()?;
            // Reparented to launchd ⇒ the CLI that spawned it is gone. A live
            // sibling run has its own CLI as the parent and is skipped.
            (ppid == 1 && line.contains(marker) && pid != std::process::id()).then_some(pid)
        })
        .collect()
}

fn signal(pid: u32, sig: &str) {
    if !signal_succeeds(pid, sig) {
        log::debug!("kill -{sig} {pid} failed (already gone?)");
    }
}

fn signal_succeeds(pid: u32, sig: &str) -> bool {
    Command::new("kill")
        .arg(format!("-{sig}"))
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MARKER: &str = "/var/folders/xx/T/pipette-mlx/pipette_mlx_server-";

    fn ps_line(pid: u32, ppid: u32, command: &str) -> String {
        format!("{pid:>6} {ppid:>6} {command}")
    }

    #[test]
    fn selects_a_reparented_server() {
        let out = ps_line(
            4242,
            1,
            "/venv/bin/python /var/folders/xx/T/pipette-mlx/pipette_mlx_server-abc.py --port 5000",
        );
        assert_eq!(orphans_from_ps_output(&out, MARKER), vec![4242]);
    }

    // The discriminator that keeps a parallel run safe: same command, but its
    // CLI is still alive, so it is not an orphan.
    #[test]
    fn skips_a_server_whose_parent_is_alive() {
        let out = ps_line(
            4242,
            991,
            "/venv/bin/python /var/folders/xx/T/pipette-mlx/pipette_mlx_server-abc.py --port 5000",
        );
        assert!(orphans_from_ps_output(&out, MARKER).is_empty());
    }

    #[test]
    fn skips_unrelated_reparented_processes() {
        let out = [
            ps_line(10, 1, "/usr/sbin/cupsd -l"),
            ps_line(11, 1, "/venv/bin/python /home/me/train.py"),
            // Another tool's python, similar shape, different script.
            ps_line(12, 1, "/venv/bin/python /tmp/other/server-abc.py --port 1"),
        ]
        .join("\n");
        assert!(orphans_from_ps_output(&out, MARKER).is_empty());
    }

    #[test]
    fn collects_every_orphan_in_one_pass() {
        let out = [
            ps_line(1, 0, "/sbin/launchd"),
            ps_line(
                7,
                1,
                "/v/bin/python /var/folders/xx/T/pipette-mlx/pipette_mlx_server-a.py --port 1",
            ),
            ps_line(
                8,
                99,
                "/v/bin/python /var/folders/xx/T/pipette-mlx/pipette_mlx_server-b.py",
            ),
            ps_line(
                9,
                1,
                "/v/bin/python /var/folders/xx/T/pipette-mlx/pipette_mlx_server-c.py --port 2",
            ),
        ]
        .join("\n");
        assert_eq!(orphans_from_ps_output(&out, MARKER), vec![7, 9]);
    }

    #[test]
    fn tolerates_malformed_ps_rows() {
        let out = ["", "not a row", "abc def ghi", "1"].join("\n");
        assert!(orphans_from_ps_output(&out, MARKER).is_empty());
    }

    // Guards the marker against drifting from the path the server actually
    // materializes: a marker that matches nothing would silently reap nothing.
    #[test]
    fn marker_matches_a_materialized_script_path() -> anyhow::Result<()> {
        let marker = server_script_marker();
        let script = crate::execute::server::materialize_server_script_for_test("print('x')\n")?;
        let path = script.to_string_lossy().into_owned();
        assert!(
            path.starts_with(&marker),
            "{path} should start with {marker}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod reap_e2e {
    use super::*;

    /// Spawns a stand-in server that reparents to launchd, then reaps it.
    /// Ignored by default: it signals real processes.
    #[test]
    #[ignore = "spawns and kills a real process; run with --ignored"]
    fn reaps_a_reparented_stand_in_server() -> anyhow::Result<()> {
        let marker = server_script_marker();
        let script = format!("{marker}feedface00000000.py");
        std::fs::create_dir_all(
            std::path::Path::new(&script)
                .parent()
                .context("script path has no parent")?,
        )?;
        std::fs::write(&script, "import time\ntime.sleep(600)\n")?;

        // Via `sh &` so the intermediate shell exits and the python child
        // reparents to pid 1 — the same shape a SIGKILLed CLI leaves behind.
        Command::new("sh")
            .arg("-c")
            .arg(format!("python3 {script} >/dev/null 2>&1 &"))
            .status()?;
        thread::sleep(Duration::from_millis(800));

        let before = orphans_from_ps_output(&ps_snapshot()?, &marker);
        anyhow::ensure!(!before.is_empty(), "stand-in server did not start");

        reap_orphan_servers();
        thread::sleep(Duration::from_millis(500));

        let after = orphans_from_ps_output(&ps_snapshot()?, &marker);
        let _ = std::fs::remove_file(&script);
        anyhow::ensure!(after.is_empty(), "orphans survived the reap: {after:?}");
        Ok(())
    }
}
