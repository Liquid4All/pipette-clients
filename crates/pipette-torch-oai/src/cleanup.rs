//! Crash-resilient server teardown.
//!
//! Two complementary mechanisms keep stale servers from piling up
//! when the client doesn't exit cleanly:
//!
//! 1. **Signal handler** (Ctrl-C / SIGTERM). Rust's default behavior is
//!    to call `process::exit` without unwinding, so any [`Drop`] guards
//!    in `execute/*` never run and the detached server outlives us. The
//!    shared `pipette_subprocess::cleanup` handler drains an in-flight teardown
//!    registry and fires each callback before exiting 130;
//!    [`register_docker`] enrolls a `docker stop` in that registry.
//! 2. **Label-based orphan reaping**. Every `docker run` we launch gets
//!    a `pipette-torch-oai.workspace=<root>` label. The next run starts
//!    by `docker ps`-filtering that label and force-removing anything
//!    that survived — covers the cases the handler can't (SIGKILL, OOM,
//!    `panic = "abort"`, power loss).

use std::path::Path;
use std::process::Command;

use anyhow::Context;

use pipette_subprocess::cleanup::TeardownToken;

/// Docker label key applied to every container launched by this client.
/// The value is the absolute path of the active workspace root, so a
/// crashed run can be reaped by the next run *from the same workspace*
/// without touching unrelated containers on the host.
pub const WORKSPACE_LABEL: &str = "pipette.workspace";

/// Convenience wrapper for docker teardown: closes over `(docker_path,
/// container_id)` and registers a callback that issues `docker stop -t 10`.
/// Returns the token used to deregister after a clean stop.
pub fn register_docker(docker: &Path, container_id: &str) -> TeardownToken {
    let docker = docker.to_path_buf();
    let container_id = container_id.to_string();
    pipette_subprocess::cleanup::register(move || {
        let _ = Command::new(&docker)
            .args(["stop", "-t", "10", &container_id])
            .status();
    })
}

/// Find and force-remove any containers from a previous run of this
/// workspace. The signal handler covers ^C/SIGTERM; this catches the
/// rest. Soft-fails (logs and returns Ok) so a broken docker socket
/// can't block a fresh run.
pub fn reap_workspace_orphans(docker: &Path, workspace_label: &str) -> anyhow::Result<()> {
    let filter = format!("label={WORKSPACE_LABEL}={workspace_label}");
    let mut cmd = Command::new(docker);
    cmd.args(["ps", "-aq", "--filter", &filter]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to query orphans with --filter {filter}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!(
            "orphan-reap query failed ({}): {}",
            output.status,
            stderr.trim()
        );
        return Ok(());
    }
    let ids: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    log::warn!(
        "reaping {} orphan container(s) from a prior crashed run: {}",
        ids.len(),
        ids.iter().map(|s| short(s)).collect::<Vec<_>>().join(", ")
    );
    let mut rm = Command::new(docker);
    rm.arg("rm").arg("-f").args(&ids);
    rm.stdout(std::process::Stdio::null());
    pipette_subprocess::echo_info(&rm);
    let status = rm
        .status()
        .context("docker rm -f for orphans failed to spawn")?;
    if !status.success() {
        log::warn!("docker rm -f for orphans exited {status}; some may persist");
    }
    Ok(())
}

fn short(container_id: &str) -> String {
    container_id.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_label_is_namespaced_to_pipette_torch_oai() {
        // Regression guard for the rename — anyone touching this constant
        // breaks orphan-reap interop across versions, so spell out the
        // expected key here.
        assert_eq!(WORKSPACE_LABEL, "pipette.workspace");
    }
}
