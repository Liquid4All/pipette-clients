//! iOS transport primitive: launch the Pipette app in `headlessrun` mode via
//! `xcrun devicectl`. `devicectl` launches a process with an argv, so this
//! takes app args rather than a shell string.

use pipette_readiness::{MAX_WAIT_ENV, SKIP_THERMAL_ENV};

use crate::transport::{
    ExecOutput,
    process::{run_quiet, run_streaming_scanning},
};

/// The app's stdout result line: `BENCH_DONE <status>`.
pub(crate) const BENCH_DONE_SENTINEL: &str = "BENCH_DONE";

/// The only request variables the app reads. `PIPETTE_HF_TOKEN` is not one, and
/// `--environment-variables` would put it on the Mac's command line.
const APP_ENV: [&str; 2] = [MAX_WAIT_ENV, SKIP_THERMAL_ENV];

pub(crate) fn app_env(env: &[(String, String)]) -> impl Iterator<Item = &(String, String)> {
    env.iter()
        .filter(|(key, _)| APP_ENV.contains(&key.as_str()))
}

/// The args to `xcrun`.
///
/// `--terminate-existing` because a live instance otherwise blocks the launch
/// and `devicectl` waits rather than failing: one interrupted cell (a jetsam
/// kill, a cancelled run) stranded every later cell of the plan with no output.
/// A cell owns the phone, so a live instance is leftover state, not a queue.
fn launch_argv(
    device_udid: &str,
    bundle_id: &str,
    env: &[(String, String)],
    app_args: &[String],
) -> Vec<String> {
    let mut argv = vec![
        "devicectl".to_string(),
        "device".to_string(),
        "process".to_string(),
        "launch".to_string(),
        "--device".to_string(),
        device_udid.to_string(),
        "--terminate-existing".to_string(),
        "--console".to_string(),
    ];
    let vars: serde_json::Map<String, serde_json::Value> = app_env(env)
        .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
        .collect();
    // Omitted when empty: a launch without overrides keeps the plain argv.
    if !vars.is_empty() {
        argv.push("--environment-variables".to_string());
        argv.push(serde_json::Value::Object(vars).to_string());
    }
    argv.push(bundle_id.to_string());
    argv.extend_from_slice(app_args);
    argv
}

/// The launch as a shell command for the Mac the device is paired with
/// (`ios_over_ssh`). Each element is quoted so the remote shell does not
/// re-split the app args' JSON.
pub(crate) fn remote_command(
    device_udid: &str,
    bundle_id: &str,
    env: &[(String, String)],
    app_args: &[String],
) -> String {
    render(launch_argv(device_udid, bundle_id, env, app_args))
}

pub(crate) fn remote_probe_command(device_udid: &str) -> String {
    render(probe_argv(device_udid))
}

/// A `devicectl` call that hangs would hang the whole `kill` sweep, which waits
/// on every transport, and an unreachable device is the case it must survive.
const DEVICECTL_TIMEOUT_SECS: u32 = 30;

/// Terminate the app on `device_udid`, as one posix shell command.
///
/// Two lookups, because `devicectl` has no bundle-id target: `process
/// terminate` takes a pid, and the process listing carries only PID and path.
/// So the app's name is resolved from the bundle id and matched against
/// `…/<Name>.app/`.
///
/// Exits 0 killed, 1 the app is absent or idle, 2 `devicectl` failed (so an
/// unreachable device does not read as "nothing running").
///
/// SIGTERM, not SIGKILL: a cell uploads its result in-process. Anything that
/// survives is cleared by the next launch's `--terminate-existing`.
pub(crate) fn kill_command(device_udid: &str, bundle_id: &str) -> String {
    let device = crate::shell::posix_quote(device_udid);
    let bundle = crate::shell::posix_quote(bundle_id);
    let t = DEVICECTL_TIMEOUT_SECS;
    format!(
        "apps=$(xcrun devicectl device info apps -d {device} --timeout {t} 2>/dev/null) || exit 2; \
         n=$(printf '%s\\n' \"$apps\" | awk -v b={bundle} '$2 == b {{print $1; exit}}'); \
         [ -n \"$n\" ] || exit 1; \
         procs=$(xcrun devicectl device info processes -d {device} --timeout {t} 2>/dev/null) \
           || exit 2; \
         p=$(printf '%s\\n' \"$procs\" | awk -v n=\"/$n.app/\" 'index($0, n) {{print $1; exit}}'); \
         [ -n \"$p\" ] || exit 1; \
         xcrun devicectl device process terminate -d {device} --pid \"$p\" \
           --timeout {t} >/dev/null 2>&1"
    )
}

fn render(argv: Vec<String>) -> String {
    std::iter::once("xcrun".to_string())
        .chain(argv)
        .map(|part| crate::shell::posix_quote(&part))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The status is the app's `BENCH_DONE <status>` when present, else the
/// `xcrun` exit code.
pub(crate) fn exec_streaming(
    device_udid: &str,
    bundle_id: &str,
    env: &[(String, String)],
    app_args: &[String],
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    let argv = launch_argv(device_udid, bundle_id, env, app_args);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let (out, scanned) = run_streaming_scanning("xcrun", &refs, prefix, Some(BENCH_DONE_SENTINEL))?;
    Ok(ExecOutput {
        status: scanned.unwrap_or(out.status),
    })
}

/// Discards output, so no `BENCH_DONE`: the status is the `xcrun` exit code.
pub(crate) fn exec_quiet(
    device_udid: &str,
    bundle_id: &str,
    env: &[(String, String)],
    app_args: &[String],
) -> anyhow::Result<ExecOutput> {
    let argv = launch_argv(device_udid, bundle_id, env, app_args);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    run_quiet("xcrun", &refs)
}

/// Queries device info rather than launching the app. Exit 0 means the UDID
/// is connected and `devicectl` works.
pub(crate) fn probe(device_udid: &str) -> anyhow::Result<ExecOutput> {
    let argv = probe_argv(device_udid);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    run_quiet("xcrun", &refs)
}

/// Shared so the local and ssh probes cannot ask different questions.
fn probe_argv(device_udid: &str) -> Vec<String> {
    [
        "devicectl",
        "device",
        "info",
        "details",
        "--device",
        device_udid,
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_command_quotes_each_arg_so_json_specs_survive_the_hop() {
        let app_args = vec![
            "headlessrun".to_string(),
            "bench".to_string(),
            r#"spec={"type":"hf_mlx","repo_name":"Qwen3.5-0.8B-4bit"}"#.to_string(),
            "benchmarks=decode_throughput_256".to_string(),
            "submit=1".to_string(),
        ];
        let cmd = remote_command("UDID-1", "ai.liquid.liquid-pipette", &[], &app_args);
        assert!(cmd.starts_with("xcrun devicectl device process launch"));
        assert!(
            cmd.contains(r#"'spec={"type":"hf_mlx","repo_name":"Qwen3.5-0.8B-4bit"}'"#),
            "spec was not quoted as a single argument: {cmd}"
        );
        // `posix_quote` quotes only what needs it, so a bare UDID stays bare.
        assert!(cmd.contains("--device UDID-1"), "{cmd}");
    }

    #[test]
    fn remote_probe_command_queries_info_without_launching() {
        let cmd = remote_probe_command("UDID-1");
        assert!(cmd.contains("device info details"), "{cmd}");
        assert!(!cmd.contains("process launch"), "{cmd}");
    }

    #[test]
    fn launch_argv_wraps_app_args() {
        let app_args = vec![
            "headlessrun".to_string(),
            r#"runtime={"type":"apple_foundation"}"#.to_string(),
            "benchmarks=decode_throughput_256".to_string(),
            "submit=1".to_string(),
        ];
        assert_eq!(
            launch_argv("UDID-1", "ai.liquid.liquid-pipette", &[], &app_args),
            vec![
                "devicectl",
                "device",
                "process",
                "launch",
                "--device",
                "UDID-1",
                "--terminate-existing",
                "--console",
                "ai.liquid.liquid-pipette",
                "headlessrun",
                r#"runtime={"type":"apple_foundation"}"#,
                "benchmarks=decode_throughput_256",
                "submit=1",
            ]
        );
    }

    fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    /// `devicectl` reads `--environment-variables` only ahead of the bundle id.
    #[test]
    fn launch_argv_forwards_only_the_readiness_overrides() -> anyhow::Result<()> {
        let request_env = env(&[
            ("PIPETTE_HF_TOKEN", "hf_secret"),
            ("PIPETTE_READINESS_MAX_WAIT_SECS", "1800"),
            ("PIPETTE_READINESS_SKIP_THERMAL", "1"),
        ]);
        let argv = launch_argv(
            "UDID-1",
            "ai.liquid.liquid-pipette",
            &request_env,
            &["headlessrun".to_string()],
        );
        let position = |needle: &str| {
            argv.iter()
                .position(|a| a == needle)
                .ok_or_else(|| anyhow::anyhow!("no {needle} in {argv:?}"))
        };
        let flag = position("--environment-variables")?;
        assert!(flag < position("ai.liquid.liquid-pipette")?, "{argv:?}");
        let json = argv
            .get(flag + 1)
            .ok_or_else(|| anyhow::anyhow!("flag without a value: {argv:?}"))?;
        // Parsed, not compared as text: key order follows serde_json's features.
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(json)?,
            serde_json::json!({
                "PIPETTE_READINESS_MAX_WAIT_SECS": "1800",
                "PIPETTE_READINESS_SKIP_THERMAL": "1",
            })
        );
        assert!(!argv.iter().any(|a| a.contains("hf_secret")), "{argv:?}");
        Ok(())
    }

    #[test]
    fn launch_argv_omits_the_flag_without_readiness_overrides() {
        let argv = launch_argv(
            "UDID-1",
            "ai.liquid.liquid-pipette",
            &env(&[("PIPETTE_HF_TOKEN", "hf_secret")]),
            &[],
        );
        assert!(
            !argv
                .iter()
                .any(|a| a == "--environment-variables" || a.contains("hf_secret")),
            "{argv:?}"
        );
    }

    #[test]
    fn remote_command_quotes_the_environment_json_as_one_argument() {
        let cmd = remote_command(
            "UDID-1",
            "ai.liquid.liquid-pipette",
            &env(&[("PIPETTE_READINESS_MAX_WAIT_SECS", "1800")]),
            &[],
        );
        assert!(
            cmd.contains(r#"--environment-variables '{"PIPETTE_READINESS_MAX_WAIT_SECS":"1800"}'"#),
            "{cmd}"
        );
    }

    #[test]
    fn kill_command_resolves_the_pid_and_terminates_it() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(cmd.contains("devicectl device info apps"), "{cmd}");
        assert!(cmd.contains("devicectl device process terminate"), "{cmd}");
        assert!(!cmd.contains("process launch"), "{cmd}");
    }

    /// Taking the last line instead yields `Apps` (from `Apps installed:`) on
    /// a device without the app, so the absent-app guard never fires.
    #[test]
    fn kill_command_matches_the_bundle_id_against_its_own_column() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        // `posix_quote` leaves a dotted bundle id bare, as it does the UDID.
        assert!(
            cmd.contains("awk -v b=ai.liquid.liquid-pipette '$2 == b"),
            "{cmd}"
        );
        assert!(!cmd.contains("tail -1"), "{cmd}");
    }

    #[test]
    fn kill_command_exits_nonzero_when_nothing_is_running() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(cmd.contains(r#"[ -n "$n" ] || exit 1"#), "{cmd}");
        assert!(cmd.contains(r#"[ -n "$p" ] || exit 1"#), "{cmd}");
    }

    #[test]
    fn kill_command_separates_device_failure_from_nothing_running() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert_eq!(cmd.matches("|| exit 2").count(), 2, "{cmd}");
    }

    #[test]
    fn kill_command_bounds_every_devicectl_call() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert_eq!(
            cmd.matches("--timeout").count(),
            cmd.matches("xcrun devicectl").count(),
            "{cmd}"
        );
    }

    #[test]
    fn kill_command_does_not_force_sigkill() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(!cmd.contains("--kill"), "{cmd}");
    }
}
