//! Locate binaries on `PATH`, render `std::process::Command` instances as
//! shell-quoted one-liners for logs, and echo them at `info` / `debug`
//! before they spawn. Operators read the resulting `$ ...` lines to
//! reproduce a run by hand, so every external invocation should pass
//! through here.
//!
//! Env vars set via `Command::env(...)` are deliberately omitted from
//! the rendered output — they're the secret-safe channel (HF_TOKEN and
//! arbitrary `--docker-env K` inheritance) and must never reach the log.
//!
//! `argv` returns the program + args as a `Vec<String>` for callers
//! that need to persist the invocation in a result record (e.g.
//! `pipette-llamacpp`'s `RunResponse.command`).

use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;

/// The PATH-lookup tool for this host. Windows ships no `which`; `where` is the
/// equivalent, and it also applies `PATHEXT`, so a bare `uv` finds `uv.exe`.
const LOOKUP: &str = if cfg!(windows) { "where" } else { "which" };

/// Locate `name` on `PATH` via the system lookup tool ([`LOOKUP`]).
///
/// `where` prints every match, one per line, most-preferred first; `which`
/// prints one. Taking the first line is therefore correct on both, and is what
/// the shell would actually run.
pub fn which(name: &str) -> anyhow::Result<PathBuf> {
    let mut cmd = Command::new(LOOKUP);
    cmd.arg(name);
    echo_debug(&cmd);
    let output = cmd
        .output()
        .with_context(|| format!("failed to run `{LOOKUP} {name}`"))?;
    if !output.status.success() {
        anyhow::bail!("{name} not found on PATH");
    }
    let stdout = String::from_utf8(output.stdout).context("non-UTF-8 path")?;
    let path = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{LOOKUP} {name} reported no path"))?;
    Ok(PathBuf::from(path))
}

/// Program + args, unquoted. The canonical primitive — used by
/// [`render`] for log formatting and by callers that need to persist
/// the invocation as structured data.
pub fn argv(cmd: &Command) -> Vec<String> {
    std::iter::once(cmd.get_program().to_string_lossy().into_owned())
        .chain(cmd.get_args().map(|a| a.to_string_lossy().into_owned()))
        .collect()
}

/// Shell-quoted one-liner suitable for pasting back into a terminal.
pub fn render(cmd: &Command) -> String {
    argv(cmd)
        .into_iter()
        .map(|s| {
            shlex::try_quote(&s)
                .map(|c| c.into_owned())
                .unwrap_or_else(|_| format!("{s:?}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Echo a command at `info` level — for user-visible operations
/// (docker pull/run/stop, uv venv, llama-server / llama-bench spawn).
/// Default `RUST_LOG=info` surfaces these.
pub fn echo_info(cmd: &Command) {
    log::info!("$ {}", render(cmd));
}

/// Echo a command at `debug` level — for chatty internal probes
/// (docker inspect / top / exec polled at >1 Hz, `which`, version checks).
pub fn echo_debug(cmd: &Command) {
    log::debug!("$ {}", render(cmd));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's lookup tool must actually exist, and must resolve a binary
    /// every target ships. This is the regression guard for shelling out to a
    /// `which` that does not exist on Windows.
    #[test]
    fn which_finds_a_binary_every_host_has() -> anyhow::Result<()> {
        let name = if cfg!(windows) { "cmd" } else { "sh" };
        let found = which(name)?;
        assert!(found.is_absolute(), "{name} resolved to {found:?}");
        assert!(found.exists(), "{name} resolved to a missing {found:?}");
        Ok(())
    }

    #[test]
    fn which_reports_a_missing_binary() {
        assert!(which("pipette-no-such-binary-3f9a1c").is_err());
    }

    #[test]
    fn argv_lists_program_and_args() {
        let mut cmd = Command::new("docker");
        cmd.args(["pull", "vllm/vllm-openai:v0.20.2"]);
        assert_eq!(
            argv(&cmd),
            vec!["docker", "pull", "vllm/vllm-openai:v0.20.2"]
        );
    }

    #[test]
    fn render_simple_command() {
        let mut cmd = Command::new("docker");
        cmd.args(["pull", "vllm/vllm-openai:v0.20.2"]);
        assert_eq!(render(&cmd), "docker pull vllm/vllm-openai:v0.20.2");
    }

    #[test]
    fn render_quotes_args_with_spaces() {
        let mut cmd = Command::new("docker");
        cmd.args(["run", "--shm-size", "16g", "-v", "/host path:/in"]);
        let rendered = render(&cmd);
        assert!(
            rendered.contains("'/host path:/in'"),
            "expected shell-quoted mount arg, got: {rendered}"
        );
    }

    #[test]
    fn render_omits_env_vars() {
        // Env vars carry secrets (HF_TOKEN, --docker-env K inheritance)
        // and must never appear in the rendered output.
        let mut cmd = Command::new("docker");
        cmd.env("HF_TOKEN", "super-secret")
            .args(["run", "-e", "HF_TOKEN", "image"]);
        let rendered = render(&cmd);
        assert!(!rendered.contains("super-secret"), "leaked env value");
        assert!(rendered.contains("-e HF_TOKEN"));
    }
}
