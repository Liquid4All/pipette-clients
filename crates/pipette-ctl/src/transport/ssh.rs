use crate::transport::{
    ExecOutput,
    process::{run_quiet, run_streaming, run_streaming_scanning},
};

pub(crate) fn exec_quiet(
    host: &str,
    user: Option<&str>,
    port: Option<u16>,
    shell_cmd: &str,
) -> anyhow::Result<ExecOutput> {
    let args = build_args(host, user, port, shell_cmd)?;
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_quiet("ssh", &refs)
}

/// Like [`exec_streaming`], but also returns the last `sentinel` value on the
/// remote stdout, for `ios_over_ssh`'s `BENCH_DONE <n>`.
pub(crate) fn exec_streaming_scanning(
    host: &str,
    user: Option<&str>,
    port: Option<u16>,
    shell_cmd: &str,
    prefix: Option<&str>,
    sentinel: &str,
) -> anyhow::Result<(ExecOutput, Option<i32>)> {
    let args = build_args(host, user, port, shell_cmd)?;
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_streaming_scanning("ssh", &refs, prefix, Some(sentinel))
}

pub(crate) fn exec_streaming(
    host: &str,
    user: Option<&str>,
    port: Option<u16>,
    shell_cmd: &str,
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    let args = build_args(host, user, port, shell_cmd)?;
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_streaming("ssh", &refs, prefix)
}

/// Long enough for a healthy box on a slow link, short enough that an
/// unreachable one cannot hold the fleet's start behind it.
const CONNECT_TIMEOUT_SECS: u64 = 15;

/// Hostnames, IP literals (`:` and `%` for IPv6 and its zone ids) and
/// `ssh_config` aliases.
const HOST_ALLOWED: &str = "letters, digits, '.', '-', '_', ':' and '%'";

/// Account names, including the spaced display names Windows OpenSSH logs in as
/// and `DOMAIN\\user` forms.
const USER_ALLOWED: &str = "letters, digits, '.', '-', '_', '\\' and spaces";

fn build_args(
    host: &str,
    user: Option<&str>,
    port: Option<u16>,
    shell_cmd: &str,
) -> anyhow::Result<Vec<String>> {
    check_field("host", host, HOST_ALLOWED, |c| {
        c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '%')
    })?;
    if let Some(u) = user {
        check_field("user", u, USER_ALLOWED, |c| {
            c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '\\' | ' ')
        })?;
    }

    let mut args = Vec::new();
    if let Some(p) = port {
        args.push("-p".into());
        args.push(p.to_string());
    }
    args.push("-o".into());
    args.push("BatchMode=yes".into());
    // A box that drops packets (a sleeping Mac, a firewalled Windows host)
    // otherwise leaves ssh in connect for the TCP default, and nothing above
    // bounds an `exec`, so the initial sync every cell waits behind stalls.
    // Connect only: keepalives would also tear down a live benchmark on a
    // loaded box that stopped answering.
    args.push("-o".into());
    args.push(format!("ConnectTimeout={CONNECT_TIMEOUT_SECS}"));
    let target = match user {
        Some(u) => format!("{u}@{host}"),
        None => host.to_string(),
    };
    args.push(target);
    args.push(shell_cmd.to_string());
    Ok(args)
}

/// Reject a plan-supplied target field that `ssh` would read as something
/// other than a name.
///
/// A value starting with `-` is parsed as an option wherever it sits, so a
/// `host` of `-oProxyCommand=…` runs that command on the driver during the
/// first probe. `--` is no defence: OpenSSH does not document it as an
/// end-of-options guard for the target. The value is reported, not rewritten,
/// so nobody reaches a host they did not name.
fn check_field(
    field: &str,
    value: &str,
    allowed_desc: &str,
    allowed: impl Fn(char) -> bool,
) -> anyhow::Result<()> {
    if value.is_empty() {
        anyhow::bail!("invalid ssh {field}: empty");
    }
    if value.starts_with('-') {
        anyhow::bail!(
            "invalid ssh {field} {value:?}: a leading '-' makes ssh read it as an option \
             instead of a target"
        );
    }
    if let Some(bad) = value.chars().find(|c| !allowed(*c)) {
        anyhow::bail!(
            "invalid ssh {field} {value:?}: character {bad:?} is not one of {allowed_desc}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::build_args;

    #[rstest]
    #[case::bare_host(
        "edge-ci-linux1",
        None,
        None,
        vec!["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "edge-ci-linux1", "echo ok"]
    )]
    #[case::user_and_port(
        "edge-ci-linux1",
        Some("liquid"),
        Some(2222),
        vec![
            "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "liquid@edge-ci-linux1", "echo ok",
        ]
    )]
    #[case::ipv6_literal(
        "fe80::1%en0",
        None,
        None,
        vec!["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "fe80::1%en0", "echo ok"]
    )]
    #[case::windows_account(
        "win-box",
        Some("Liquid AI"),
        None,
        vec!["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "Liquid AI@win-box", "echo ok"]
    )]
    fn build_args_renders(
        #[case] host: &str,
        #[case] user: Option<&str>,
        #[case] port: Option<u16>,
        #[case] expected: Vec<&str>,
    ) -> anyhow::Result<()> {
        assert_eq!(build_args(host, user, port, "echo ok")?, expected);
        Ok(())
    }

    #[rstest]
    #[case::proxy_command_host("-oProxyCommand=touch /tmp/pwned", None, "host")]
    #[case::proxy_command_user("edge-ci-linux1", Some("-oProxyCommand=touch /tmp/pwned"), "user")]
    #[case::flag_host("-E/tmp/log", None, "host")]
    #[case::empty_host("", None, "host")]
    #[case::empty_user("edge-ci-linux1", Some(""), "user")]
    #[case::substitution_host("$(touch /tmp/pwned)", None, "host")]
    #[case::space_host("edge-ci-linux1 extra", None, "host")]
    fn build_args_rejects(
        #[case] host: &str,
        #[case] user: Option<&str>,
        #[case] field: &str,
    ) -> anyhow::Result<()> {
        let err = build_args(host, user, None, "echo ok")
            .err()
            .context("expected the target to be rejected")?;
        let message = err.to_string();
        assert!(
            message.starts_with(&format!("invalid ssh {field}")),
            "error does not name the offending field: {message}"
        );
        Ok(())
    }
}
