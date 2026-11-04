//! Shell-quoted one-liners of a `Command` for logs, echoed in a `command` field
//! before it spawns, so an operator can rerun it by hand.
//!
//! Variables set with `Command::env` are left out: they are the channel for
//! secrets such as `HF_TOKEN`, which must never reach the log.

use std::process::Command;

/// Program + args, unquoted: what [`render`] quotes, and what a started
/// program keeps as its command.
pub(crate) fn argv(cmd: &Command) -> Vec<String> {
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

/// Echo a command at `info`, for user-visible operations such as a `docker
/// pull` or a server's start.
pub fn echo_info(cmd: &Command) {
    tracing::info!(command = %render(cmd), "running");
}

/// Echo a command at `debug`, for chatty probes such as a polled `docker
/// inspect` or a version check.
pub fn echo_debug(cmd: &Command) {
    tracing::debug!(command = %render(cmd), "running");
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut cmd = Command::new("docker");
        cmd.env("HF_TOKEN", "super-secret")
            .args(["run", "-e", "HF_TOKEN", "image"]);
        let rendered = render(&cmd);
        assert!(!rendered.contains("super-secret"), "leaked env value");
        assert!(rendered.contains("-e HF_TOKEN"));
    }
}
