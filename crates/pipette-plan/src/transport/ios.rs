//! iOS transport primitive: drive a device from the host Mac via
//! `xcrun devicectl`, launching the Pipette app in `headlessrun` mode.
//!
//! Unlike adb/ssh (which run a shell command on the device), `devicectl`
//! launches a *process* with an argv, so this takes the app args
//! directly rather than a shell string. `--console` streams the app's
//! stdout back; the app prints `BENCH_DONE <status>` as its result
//! contract, which the caller trusts over the `devicectl` exit code.

use crate::transport::{
    process::{run_quiet, run_streaming_scanning},
    ExecOutput,
};

/// The app's stdout result line: `BENCH_DONE <status>`.
pub(crate) const BENCH_DONE_SENTINEL: &str = "BENCH_DONE";

/// `["devicectl", "device", "process", "launch", "--device", <device_udid>,
/// "--terminate-existing", "--console", <bundle_id>, <app_args…>]` — the args to `xcrun`.
///
/// `--terminate-existing` because a live instance otherwise blocks the launch and
/// `devicectl` waits on it rather than failing: one interrupted cell — a jetsam kill, a
/// cancelled run, a `settings run` worker left resident — stranded every later cell of the
/// plan with no output at all. A cell owns the phone for its duration, so an instance
/// still up is leftover state, not something to queue behind.
fn launch_argv(device_udid: &str, bundle_id: &str, app_args: &[String]) -> Vec<String> {
    let mut argv = vec![
        "devicectl".to_string(),
        "device".to_string(),
        "process".to_string(),
        "launch".to_string(),
        "--device".to_string(),
        device_udid.to_string(),
        "--terminate-existing".to_string(),
        "--console".to_string(),
        bundle_id.to_string(),
    ];
    argv.extend_from_slice(app_args);
    argv
}

/// The same launch rendered as a shell command for an intermediate Mac, so
/// `ios_over_ssh` runs `devicectl` where the device is paired rather than where
/// the driver happens to be.
///
/// Every argv element is quoted individually: the app args carry JSON specs and
/// `key=value` pairs whose spaces and quotes would otherwise be re-split by the
/// remote shell, exactly as `adb_over_ssh` quotes its device command.
pub(crate) fn remote_command(device_udid: &str, bundle_id: &str, app_args: &[String]) -> String {
    render(launch_argv(device_udid, bundle_id, app_args))
}

/// Reachability check for `ios_over_ssh`, rendered for the intermediate host.
pub(crate) fn remote_probe_command(device_udid: &str) -> String {
    render(probe_argv(device_udid))
}

/// A `devicectl` call that hangs would stall the whole sweep: `kill` walks the
/// plan's transports serially, and an unreachable device is exactly the case
/// this command exists to survive.
const DEVICECTL_TIMEOUT_SECS: u32 = 30;

/// Terminate the app on `device_udid`, as one posix shell command.
///
/// Two lookups, because `devicectl` offers no bundle-id target: `process
/// terminate` takes a pid, and the process listing carries only PID and
/// executable path. So the app's *name* is resolved from the bundle id and
/// matched against `…/<Name>.app/`.
///
/// Three outcomes the caller distinguishes: 0 killed, 1 the app is absent or
/// idle, 2 `devicectl` itself failed. Without the last one an unreachable
/// device reports as "nothing was running", which is the opposite of the truth.
///
/// SIGTERM, not SIGKILL: a cell uploads its result in-process, so an immediate
/// kill can drop one that was in flight. Anything that survives is cleared by
/// the next launch's `--terminate-existing`.
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

/// Quote an `xcrun` argv into one command line for a posix shell.
fn render(argv: Vec<String>) -> String {
    std::iter::once("xcrun".to_string())
        .chain(argv)
        .map(|part| crate::shell::posix_quote(&part))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Launch the app and stream its console. Success is the app's
/// `BENCH_DONE <status>` when present, else the `xcrun` exit code.
pub(crate) fn exec_streaming(
    device_udid: &str,
    bundle_id: &str,
    app_args: &[String],
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    let argv = launch_argv(device_udid, bundle_id, app_args);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let (out, scanned) = run_streaming_scanning("xcrun", &refs, prefix, Some(BENCH_DONE_SENTINEL))?;
    Ok(ExecOutput {
        status: scanned.unwrap_or(out.status),
    })
}

/// Launch the app discarding output. Used for reachability probes; the
/// benchmark path uses [`exec_streaming`] so it can read `BENCH_DONE`.
pub(crate) fn exec_quiet(
    device_udid: &str,
    bundle_id: &str,
    app_args: &[String],
) -> anyhow::Result<ExecOutput> {
    let argv = launch_argv(device_udid, bundle_id, app_args);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    run_quiet("xcrun", &refs)
}

/// Reachability check that does not launch the app: query device info.
/// Exit 0 means the UDID is connected and `devicectl` works.
pub(crate) fn probe(device_udid: &str) -> anyhow::Result<ExecOutput> {
    let argv = probe_argv(device_udid);
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    run_quiet("xcrun", &refs)
}

/// Device-info query, shared by [`probe`] and [`remote_probe_command`] so the
/// local and ssh paths cannot ask different questions.
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

    /// The spec form carries JSON with spaces, braces and quotes. Rendered for
    /// a remote shell it has to arrive as one argv element, or `devicectl` sees
    /// a garbled command — the failure `adb_over_ssh` quoting exists to prevent.
    #[test]
    fn remote_command_quotes_each_arg_so_json_specs_survive_the_hop() {
        let app_args = vec![
            "headlessrun".to_string(),
            "bench".to_string(),
            r#"spec={"type":"hf_mlx","repo_name":"Qwen3.5-0.8B-4bit"}"#.to_string(),
            "benchmarks=decode_throughput_256".to_string(),
            "submit=1".to_string(),
        ];
        let cmd = remote_command("UDID-1", "ai.liquid.liquid-pipette", &app_args);
        assert!(cmd.starts_with("xcrun devicectl device process launch"));
        // The whole spec is one quoted token: no bare space splits it.
        assert!(
            cmd.contains(r#"'spec={"type":"hf_mlx","repo_name":"Qwen3.5-0.8B-4bit"}'"#),
            "spec was not quoted as a single argument: {cmd}"
        );
        // `posix_quote` quotes only what needs it, so a bare UDID stays bare.
        assert!(cmd.contains("--device UDID-1"), "{cmd}");
    }

    /// The remote probe must not launch the app — same contract as `probe`.
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
            launch_argv("UDID-1", "ai.liquid.liquid-pipette", &app_args),
            vec![
                "devicectl",
                "device",
                "process",
                "launch",
                "--device",
                "UDID-1",
                // A live instance would otherwise block the launch indefinitely.
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

    /// The kill has to reach `devicectl`, not the app: an iOS transport turns
    /// argv into app arguments, so a command shaped like the desktop `pkill`
    /// sweep would launch the app instead of ending it.
    #[test]
    fn kill_command_resolves_the_pid_and_terminates_it() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(cmd.contains("devicectl device info apps"), "{cmd}");
        assert!(cmd.contains("devicectl device process terminate"), "{cmd}");
        assert!(!cmd.contains("process launch"), "{cmd}");
    }

    /// The bundle id has to match a row exactly. Taking the last line instead
    /// yields `Apps` (from `Apps installed:`) on a device without the app —
    /// non-empty, so the absent-app guard never fires and the process search
    /// runs against `/Apps.app/`.
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

    /// A device with no app, and one with the app idle, both have to report as
    /// "no matching process" rather than as killed — otherwise a sweep claims
    /// to have stopped work on every phone in the plan.
    #[test]
    fn kill_command_exits_nonzero_when_nothing_is_running() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(cmd.contains(r#"[ -n "$n" ] || exit 1"#), "{cmd}");
        assert!(cmd.contains(r#"[ -n "$p" ] || exit 1"#), "{cmd}");
    }

    /// A device that cannot be queried is not a device with nothing running —
    /// conflating them reports an unreachable phone as idle.
    #[test]
    fn kill_command_separates_device_failure_from_nothing_running() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert_eq!(cmd.matches("|| exit 2").count(), 2, "{cmd}");
    }

    /// Unbounded, one unreachable device stalls the serial sweep over the plan.
    #[test]
    fn kill_command_bounds_every_devicectl_call() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert_eq!(
            cmd.matches("--timeout").count(),
            cmd.matches("xcrun devicectl").count(),
            "{cmd}"
        );
    }

    /// SIGKILL would drop a result mid-upload; the app submits in-process.
    #[test]
    fn kill_command_does_not_force_sigkill() {
        let cmd = kill_command("UDID-1", "ai.liquid.liquid-pipette");
        assert!(!cmd.contains("--kill"), "{cmd}");
    }
}
