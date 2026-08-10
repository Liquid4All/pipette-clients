use pipette_plan_types::ShellType;

pub struct RemoteExecRequest {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: Option<String>,
    /// Human label for this execution, used only by the slurm transport
    /// as the `srun --job-name`. Other transports ignore it.
    pub job_name: Option<String>,
}

/// Build a single shell command string from a remote exec request.
///
/// The returned string is meant to be passed as a single argument to
/// `adb shell <cmd>` or `ssh <host> <cmd>` — i.e. interpreted by the remote
/// shell exactly once.
///
/// `request.env` values are inlined as `KEY=value` prefixes, so a secret in
/// there lands in the remote process's argv and is readable by any other user
/// on that host (`ps`, and `squeue`/`sacct` under SLURM). That is accepted
/// rather than fixed: feeding values over stdin differs per transport. The
/// consequences for a plan's HF `auth_token` are documented for operators under
/// "Secrets in the remote command line" in `docs/pipette-plan/plan-runner.md`.
/// Use [`quote_argv`] where the env wrapper is not wanted.
pub fn build_shell_command(shell: ShellType, request: &RemoteExecRequest) -> String {
    match shell {
        ShellType::Posix => build_posix_command(request),
        ShellType::PowerShell => build_powershell_command(request),
    }
}

/// Quote an argv vector for `shell` and join it into a single command
/// line — the program and its arguments only, with no `cd`/env prefix.
/// Used by `pipette-plan commands` to print copy-pasteable invocations
/// without leaking the cwd/env wrapper (or its secret values) that
/// [`build_shell_command`] inlines.
pub fn quote_argv(shell: ShellType, argv: &[String]) -> String {
    match shell {
        ShellType::Posix => argv
            .iter()
            .map(|arg| posix_quote(arg))
            .collect::<Vec<_>>()
            .join(" "),
        ShellType::PowerShell => argv
            .iter()
            .map(|arg| cmd_quote(arg))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

// ---------------------------------------------------------------------------
// POSIX
// ---------------------------------------------------------------------------

fn build_posix_command(request: &RemoteExecRequest) -> String {
    let cwd_part = request
        .cwd
        .iter()
        .map(|cwd| format!("cd {} &&", posix_quote(cwd)));
    let env_parts = request
        .env
        .iter()
        .map(|(key, value)| format!("{}={}", key, posix_quote(value)));
    // Reuse `quote_argv` for the argv segment so this path and
    // `pipette-plan commands` render the command identically.
    let argv_part = std::iter::once(quote_argv(ShellType::Posix, &request.argv));

    cwd_part
        .chain(env_parts)
        .chain(argv_part)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prefix a command with a `pre_exec` snippet, joined by `&&`. Non-interactive
/// ssh skips the login profile, so a transport reached that way may need to set
/// PATH before its tool is callable. A blank snippet is the same as none — no
/// dangling `&&`.
pub(crate) fn prefix_pre_exec(pre_exec: Option<&str>, cmd: String) -> String {
    match pre_exec.map(str::trim) {
        Some(pre) if !pre.is_empty() => format!("{pre} && {cmd}"),
        _ => cmd,
    }
}

/// Quote one argument for a posix shell. Also used when a whole command line
/// has to survive a second shell — `adb_over_ssh` nests the device command
/// inside the intermediate host's shell.
pub(crate) fn posix_quote(s: &str) -> String {
    match shlex::try_quote(s) {
        Ok(quoted) => quoted.into_owned(),
        Err(_) => {
            let escaped = s.replace('\'', "'\"'\"'");
            format!("'{escaped}'")
        }
    }
}

// ---------------------------------------------------------------------------
// PowerShell
// ---------------------------------------------------------------------------

fn build_powershell_command(request: &RemoteExecRequest) -> String {
    let cwd_part = request
        .cwd
        .iter()
        .map(|cwd| format!("cd /d {}", cmd_quote(cwd)));
    let env_parts = request
        .env
        .iter()
        .map(|(key, value)| format!("set \"{}={}\"", key, cmd_escape(value)));
    let argv_part = std::iter::once(quote_argv(ShellType::PowerShell, &request.argv));

    let parts: Vec<_> = cwd_part.chain(env_parts).chain(argv_part).collect();
    format!("cmd /c \"{}\"", parts.join(" && "))
}

/// Characters that force an argument to be quoted for `cmd /c "…"`.
///
/// `"` is the one that was missing: a JSON `--model` argument has none of the
/// others, so it went through bare, cmd ate every quote, and the client
/// reported `key must be a string at line 1 column 2`.
///
/// Not unconditional quoting — cmd does not recognise a quoted builtin, so
/// `"echo" "ok"` breaks the transport probe.
///
/// Keep in step with [`cmd_escape`]: it neutralizes `%` as `%%`, which only
/// works inside quotes, so `%` has to stay on this list.
const CMD_NEEDS_QUOTING: &[char] = &[' ', '\t', '"', '&', '^', '|', '<', '>', '(', ')', '%', '!'];

fn cmd_quote(s: &str) -> String {
    if s.contains(CMD_NEEDS_QUOTING) {
        format!("\"{}\"", cmd_escape(s))
    } else {
        s.to_string()
    }
}

/// Escape the contents of a `cmd` double-quoted string.
///
/// `""` for an embedded quote, verified against `CommandLineToArgvW` on the
/// target rather than derived: it round-trips a JSON blob whose final bytes are
/// `\\n"}`, i.e. a backslash immediately before a quote, which is the case that
/// breaks the backslash-doubling encodings.
fn cmd_escape(s: &str) -> String {
    s.chars()
        .fold(String::with_capacity(s.len()), |mut out, ch| {
            match ch {
                '"' => out.push_str("\"\""),
                '%' => out.push_str("%%"),
                _ => out.push(ch),
            }
            out
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn req(argv: &[&str], env: &[(&str, &str)], cwd: Option<&str>) -> RemoteExecRequest {
        RemoteExecRequest {
            argv: argv.iter().map(|s| s.to_string()).collect(),
            env: env
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            cwd: cwd.map(|s| s.to_string()),
            job_name: None,
        }
    }

    #[test]
    fn posix_simple_command() {
        let cmd = build_shell_command(ShellType::Posix, &req(&["/bin/app", "run"], &[], None));
        assert_eq!(cmd, "/bin/app run");
    }

    #[test]
    fn posix_with_cwd_and_env() {
        let cmd = build_shell_command(
            ShellType::Posix,
            &req(&["/bin/app"], &[("HF_TOKEN", "hf_abc")], Some("/data")),
        );
        assert_eq!(cmd, "cd /data && HF_TOKEN=hf_abc /bin/app");
    }

    #[test]
    fn posix_quotes_spaces() {
        let cmd = build_shell_command(ShellType::Posix, &req(&["echo", "hello world"], &[], None));
        assert!(cmd.contains("'hello world'") || cmd.contains("hello\\ world"));
    }

    #[test]
    fn cmd_simple_command() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["C:\\app.exe", "run"], &[], None),
        );
        assert!(cmd.starts_with("cmd /c \""));
        assert!(cmd.contains("C:\\app.exe run"));
    }

    #[test]
    fn cmd_with_cwd_and_env() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["app.exe"], &[("HF_TOKEN", "hf_abc")], Some("C:\\data")),
        );
        assert!(cmd.contains("cd /d C:\\data"));
        assert!(cmd.contains("set \"HF_TOKEN=hf_abc\""));
        assert!(cmd.contains("app.exe"));
    }

    #[test]
    fn cmd_quotes_spaces_in_args() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["C:\\my path\\app.exe", "--flag", "value"], &[], None),
        );
        assert!(cmd.contains("\"C:\\my path\\app.exe\""));
    }

    #[test]
    fn cmd_env_values_are_quoted() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["app.exe"], &[("TOKEN", "a&b|c^d")], None),
        );
        assert!(cmd.contains("set \"TOKEN=a&b|c^d\""));
    }

    #[test]
    fn cmd_env_escapes_double_quotes() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["app.exe"], &[("VAL", "say \"hello\"")], None),
        );
        assert!(cmd.contains("set \"VAL=say \"\"hello\"\"\""));
    }

    #[test]
    fn cmd_env_escapes_percent() {
        let cmd = build_shell_command(
            ShellType::PowerShell,
            &req(&["app.exe"], &[("VAL", "100%")], None),
        );
        assert!(cmd.contains("set \"VAL=100%%\""));
    }

    #[test]
    fn posix_quotes_special_chars() {
        let cmd = build_shell_command(ShellType::Posix, &req(&["echo", "it's"], &[], None));
        assert!(!cmd.contains("it's ") || cmd.contains("it\\'s") || cmd.contains("'\"'\"'"));
    }

    #[test]
    fn posix_env_with_special_chars() {
        let cmd = build_shell_command(ShellType::Posix, &req(&["echo"], &[("VAR", "a b&c")], None));
        assert!(cmd.contains("VAR=") && !cmd.contains("VAR=a b&c"));
    }

    /// The regression: a JSON `--model` argument contains no space, `&`, `^` or
    /// `|`, so the previous sniffing quote left it bare and cmd stripped every
    /// `"`. The client then reported `key must be a string at line 1 column 2`.
    #[test]
    fn powershell_quotes_a_json_argument() {
        let json = r#"{"type":"openvino","org":"LiquidAI"}"#;
        let rendered = quote_argv(ShellType::PowerShell, &[json.to_owned()]);
        assert!(rendered.starts_with('"'), "must be quoted: {rendered}");
        assert!(
            rendered.contains(r#"""type"""#),
            "inner quotes must be doubled: {rendered}"
        );
    }

    /// The trigger list is the whole fix, so pin each entry: dropping one would
    /// otherwise reintroduce the bug for that character silently. `"` has its
    /// own test above because it is the one that was missing.
    #[rstest]
    #[case::space("a b")]
    #[case::tab("a\tb")]
    #[case::ampersand("a&b")]
    #[case::caret("a^b")]
    #[case::pipe("a|b")]
    #[case::lt("a<b")]
    #[case::gt("a>b")]
    #[case::open_paren("a(b")]
    #[case::close_paren("a)b")]
    #[case::percent("50%done")]
    #[case::bang("a!b")]
    fn cmd_quotes_every_metacharacter(#[case] arg: &str) {
        let rendered = quote_argv(ShellType::PowerShell, &[arg.to_owned()]);
        assert!(
            rendered.starts_with('"'),
            "{arg} must be quoted: {rendered}"
        );
    }

    /// `cmd` does not recognise a quoted builtin, so the probe's `echo ok` must
    /// go through bare — quoting everything unconditionally broke it.
    #[test]
    fn a_cmd_builtin_is_left_unquoted() {
        assert_eq!(
            quote_argv(ShellType::PowerShell, &["echo".into(), "ok".into()]),
            "echo ok"
        );
    }

    /// The encoding for a backslash immediately before a quote — the tail of
    /// the runtime JSON, `...2026.2.1.0\n"}`. That it round-trips through
    /// `CommandLineToArgvW` was checked on a Windows host; this pins only the
    /// bytes we emit.
    #[test]
    fn a_backslash_before_a_quote_is_left_alone() {
        let arg = "a\\n\"b";
        let rendered = quote_argv(ShellType::PowerShell, &[arg.to_owned()]);
        assert!(rendered.contains("a\\n\"\"b"), "got {rendered}");
    }
}
