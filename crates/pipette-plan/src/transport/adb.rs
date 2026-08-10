use crate::transport::{
    process::{run_quiet, run_streaming},
    ExecOutput,
};

pub(crate) fn exec_quiet(
    serial: &str,
    port: Option<u16>,
    shell_cmd: &str,
) -> anyhow::Result<ExecOutput> {
    let args = build_args(serial, port, shell_cmd);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_quiet("adb", &refs)
}

pub(crate) fn exec_streaming(
    serial: &str,
    port: Option<u16>,
    shell_cmd: &str,
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    let args = build_args(serial, port, shell_cmd);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_streaming("adb", &refs, prefix)
}

/// The same `adb … shell <cmd>` invocation as [`build_args`], rendered as one
/// command line for a *remote* posix shell (the `adb_over_ssh` transport).
///
/// `shell_cmd` is quoted so the intermediate host's shell hands it to `adb` as
/// a single argument, exactly as the local `exec_*` path passes it as one argv
/// element. Without that, a cell command's spaces and quotes would be re-split
/// on the way through ssh.
pub(crate) fn remote_command(
    serial: &str,
    adb_port: Option<u16>,
    pre_exec: Option<&str>,
    shell_cmd: &str,
) -> String {
    let mut parts = vec!["adb".to_string()];
    if let Some(p) = adb_port {
        parts.push("-P".to_string());
        parts.push(p.to_string());
    }
    parts.push("-s".to_string());
    parts.push(serial.to_string());
    parts.push("shell".to_string());
    parts.push(crate::shell::posix_quote(shell_cmd));
    crate::shell::prefix_pre_exec(pre_exec, parts.join(" "))
}

fn build_args(serial: &str, port: Option<u16>, shell_cmd: &str) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(p) = port {
        args.push("-P".into());
        args.push(p.to_string());
    }
    args.push("-s".into());
    args.push(serial.to_string());
    args.push("shell".into());
    args.push(shell_cmd.to_string());
    args
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::{build_args, remote_command};

    #[test]
    fn build_args_no_port() {
        let args = build_args("ABC123", None, "echo ok");
        assert_eq!(args, vec!["-s", "ABC123", "shell", "echo ok"]);
    }

    #[test]
    fn build_args_with_port() {
        let args = build_args("ABC123", Some(5038), "echo ok");
        assert_eq!(args, vec!["-P", "5038", "-s", "ABC123", "shell", "echo ok"]);
    }

    /// One row per rendering rule: the device command is quoted so it crosses
    /// the intermediate shell intact, `adb_port` becomes `-P`, and `pre_exec`
    /// lands outside the quotes (blank behaving as absent).
    #[rstest]
    #[case::quotes_device_command(
        None,
        None,
        "cd /data/local/tmp && ./pipette run",
        "adb -s ABC123 shell 'cd /data/local/tmp && ./pipette run'"
    )]
    #[case::carries_adb_server_port(
        Some(5038),
        None,
        "echo ok",
        "adb -P 5038 -s ABC123 shell 'echo ok'"
    )]
    #[case::prefixes_pre_exec(
        None,
        Some("export PATH=$PATH:/opt/sdk"),
        "echo ok",
        "export PATH=$PATH:/opt/sdk && adb -s ABC123 shell 'echo ok'"
    )]
    #[case::blank_pre_exec_is_absent(None, Some("  "), "echo ok", "adb -s ABC123 shell 'echo ok'")]
    fn remote_command_renders(
        #[case] adb_port: Option<u16>,
        #[case] pre_exec: Option<&str>,
        #[case] shell_cmd: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            remote_command("ABC123", adb_port, pre_exec, shell_cmd),
            expected
        );
    }

    /// A cell command carries JSON model/runtime descriptors full of quotes.
    /// What matters is that the intermediate shell splits the line back into
    /// `adb … shell <cmd>` with `cmd` intact — not which quoting style the
    /// quoter picked, so assert the round trip rather than the bytes.
    #[test]
    fn remote_command_survives_embedded_quotes() -> anyhow::Result<()> {
        let cmd = r#"./pipette run --model '{"type":"gguf_text"}' --runtime "a b""#;
        let rendered = remote_command("ABC123", Some(5037), None, cmd);
        let tokens = shlex::split(&rendered)
            .with_context(|| format!("rendered line does not re-split: {rendered}"))?;
        assert_eq!(tokens[..5], ["adb", "-P", "5037", "-s", "ABC123"]);
        assert_eq!(tokens[5], "shell");
        assert_eq!(tokens[6], cmd);
        assert_eq!(tokens.len(), 7, "extra tokens in {rendered}");
        Ok(())
    }
}
