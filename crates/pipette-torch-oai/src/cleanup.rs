//! Reaping containers a crashed run left.
//!
//! `pipette_subprocess` tears down every server it started on ^C / SIGTERM.
//! For what that cannot catch (SIGKILL, OOM, `panic = "abort"`, power loss),
//! every container carries a `pipette.workspace=<root>` label and a
//! `pipette.owner` label naming its run, and the next run removes the ones
//! with its workspace label that outlived their run, so a run in parallel
//! keeps its container. A uv server needs no label: `pipette_subprocess` kills
//! a group a dead run left before the next start.

use std::path::Path;
use std::process::Command;

use anyhow::Context;

/// Docker label key on every container this client launches. The value is the
/// workspace root, so the next run *from the same workspace* reaps a crashed
/// run without touching unrelated containers on the host.
pub const WORKSPACE_LABEL: &str = "pipette.workspace";

/// Docker label key naming the run that started a container, as
/// [`pipette_subprocess::owner`] spells it.
pub const OWNER_LABEL: &str = "pipette.owner";

/// Removes this workspace's containers whose run is gone or is this one, and
/// those without an owner label, which predate it. A query that exits non-zero is skipped so a
/// broken docker socket can't block a fresh run, and logged at debug: a uv run
/// on a node where docker is installed and not usable, as on a Slurm node,
/// meets it every time, and a docker run's own `docker run` reports a broken
/// daemon.
pub fn reap_workspace_orphans(docker: &Path, workspace_label: &str) -> anyhow::Result<()> {
    let filter = format!("label={WORKSPACE_LABEL}={workspace_label}");
    let mut cmd = Command::new(docker);
    let format = format!("{{{{.ID}}}} {{{{.Label \"{OWNER_LABEL}\"}}}}");
    cmd.args(["ps", "-a", "--filter", &filter, "--format", &format]);
    pipette_subprocess::echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to query orphans with --filter {filter}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::debug!(
            exit_status = %output.status,
            stderr = stderr.trim(),
            "orphan-reap query failed"
        );
        return Ok(());
    }
    let ids: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(orphan)
        .map(str::to_owned)
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    tracing::warn!(
        containers = ?ids.iter().map(|s| short(s)).collect::<Vec<_>>(),
        "reaping orphan containers from a prior crashed run"
    );
    let mut rm = Command::new(docker);
    rm.arg("rm").arg("-f").args(&ids);
    rm.stdout(std::process::Stdio::null());
    pipette_subprocess::echo_info(&rm);
    #[expect(
        clippy::disallowed_methods,
        reason = "stdout, the removed ids, is discarded"
    )]
    let status = rm
        .status()
        .context("docker rm -f for orphans failed to spawn")?;
    if !status.success() {
        tracing::warn!(
            exit_status = %status,
            "docker rm -f for orphans failed; some may persist"
        );
    }
    Ok(())
}

/// The workspace root implied by the store layout (`{workspace}/models/<key>/…`
/// or `{workspace}/runtimes/<key>/…`), the value of [`WORKSPACE_LABEL`].
pub fn workspace_label_from_store_path(path_under_store: &Path) -> anyhow::Result<String> {
    let store = path_under_store
        .ancestors()
        .find(|a| {
            matches!(
                a.file_name().and_then(|s| s.to_str()),
                Some("models" | "runtimes")
            )
        })
        .with_context(|| {
            format!(
                "path {} is not under a workspace `models/` or `runtimes/` store directory",
                path_under_store.display()
            )
        })?;
    let ws = store
        .parent()
        .with_context(|| format!("store dir has no parent: {}", store.display()))?;
    Ok(ws.display().to_string())
}

/// The id on a `<id> <owner>` line of `docker ps`, unless another run that
/// still runs owns it: a process runs one cell at a time, so its own is a
/// leftover.
fn orphan(line: &str) -> Option<&str> {
    let (id, owner) = line.split_once(' ').unwrap_or((line, ""));
    let (id, owner) = (id.trim(), owner.trim());
    (!id.is_empty()
        && (owner == pipette_subprocess::owner() || !pipette_subprocess::owner_runs(owner)))
    .then_some(id)
}

fn short(container_id: &str) -> String {
    container_id.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[test]
    fn workspace_label_is_namespaced_to_pipette_torch_oai() {
        // Changing the key stops a newer client reaping an older one's containers.
        assert_eq!(WORKSPACE_LABEL, "pipette.workspace");
    }

    /// Only these platforms can tell a live owner apart.
    #[rstest]
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "macos"),
        case::other_owner_runs(
            pipette_subprocess::owner_of(std::os::unix::process::parent_id()),
            None
        )
    )]
    #[case::own(pipette_subprocess::owner(), Some("abc"))]
    #[case::owner_gone(format!("{}-1", std::process::id()), Some("abc"))]
    #[case::no_owner_label(String::new(), Some("abc"))]
    fn only_a_container_whose_owner_is_gone_is_reaped(
        #[case] owner: String,
        #[case] reaped: Option<&str>,
    ) {
        assert_eq!(orphan(&format!("abc {owner}")), reaped);
    }

    #[rstest]
    #[case::models("/tmp/ws/.pipette/models/abc/blobs", Ok("/tmp/ws/.pipette"))]
    #[case::runtimes("/tmp/ws/.pipette/runtimes/key/blobs/venv", Ok("/tmp/ws/.pipette"))]
    // Innermost store component wins, so a workspace that itself lives under a
    // dir named `models` still resolves to the real workspace root.
    #[case::nested(
        "/data/models/proj/.pipette/models/abc",
        Ok("/data/models/proj/.pipette")
    )]
    #[case::outside_any_store("/tmp/other/abc", Err("not under a workspace"))]
    fn workspace_label_from_store_path_cases(
        #[case] path: &str,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        let result = workspace_label_from_store_path(Path::new(path));
        match expected {
            Ok(label) => assert_eq!(result?, label),
            Err(substr) => {
                let err = result.err().context("expected a rejection error")?;
                assert!(err.to_string().contains(substr), "got {err}");
            }
        }
        Ok(())
    }
}
