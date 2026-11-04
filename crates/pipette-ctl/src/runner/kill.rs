use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use pipette_plan_types::{Plan, ShellType, TransportConfig};

use super::{host_semaphore::HostSemaphore, run::physical_id_capacities};
use crate::{shell::RemoteExecRequest, transport::Transport};

/// Exit status a device-addressed kill uses for "could not query the device",
/// as opposed to querying it fine and finding nothing to kill.
const DEVICE_UNREACHABLE: i32 = 2;

/// llama.cpp tools the client spawns. Killing the client alone leaves these
/// holding the GPU and the model's memory. iOS is exempt: it runs llama.cpp
/// in-process.
const SPAWNED_TOOLS: [&str; 3] = ["llama-server", "llama-bench", "llama-batched-bench"];

/// The bytes of a process name Linux keeps, which is all `pkill -x` matches
/// there; macOS matches the whole name.
const LINUX_NAME_LEN: usize = 15;

/// Kill the pipette client, and the llama.cpp tools it spawns, on every
/// transport in the plan. A non-zero exit (nothing to kill) is reported but
/// does not abort the sweep.
///
/// Transports are swept concurrently under the runner's per-`physical_id`
/// budget: a box that tolerates one cell at a time tolerates one kill.
pub fn kill_transports(plan: &Plan, adb_port: Option<u16>) -> anyhow::Result<()> {
    let host_locks: HashMap<String, HostSemaphore> = physical_id_capacities(plan)
        .into_iter()
        .map(|(key, cap)| (key, HostSemaphore::new(cap)))
        .collect();
    // Resolved before any thread starts, so a gap in the capacity map fails
    // the sweep instead of leaving a box unthrottled.
    let jobs = plan
        .transports
        .iter()
        .map(|cfg| {
            let lock = host_locks
                .get(&cfg.physical_id())
                .context("host lock present for every transport's physical_id")?;
            Ok((cfg, lock))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    std::thread::scope(|scope| {
        jobs.into_iter().for_each(|(cfg, lock)| {
            scope.spawn(move || {
                let _guard = lock.acquire();
                kill_transport(cfg, adb_port);
            });
        });
    });
    Ok(())
}

/// Failures are reported, not propagated: one unreachable box must not
/// strand the rest of the fleet with a live llama.cpp on it.
fn kill_transport(cfg: &TransportConfig, adb_port: Option<u16>) {
    let transport = Transport::from_config_with_adb_port(cfg, adb_port);
    let label = transport.target_label();
    // Device-addressed transports kill on the device (see `kill_client`).
    if let Some(result) = transport.kill_client() {
        eprintln!("[{label}] terminating the app on the device");
        match result {
            Ok(o) if o.status == 0 => eprintln!("[{label}] killed"),
            Ok(o) if o.status == DEVICE_UNREACHABLE => {
                eprintln!("[{label}] device error: could not query it")
            }
            Ok(o) => eprintln!("[{label}] no matching process (exit {})", o.status),
            Err(e) => eprintln!("[{label}] error: {e:#}"),
        }
        return;
    }
    let shell = cfg.shell();
    kill_names(cfg.binary_path(), shell)
        .into_iter()
        .for_each(|name| {
            let request = RemoteExecRequest {
                argv: kill_argv(&name, shell),
                env: Vec::new(),
                cwd: None,
                job_name: Some("pipette-kill".to_string()),
            };
            eprintln!("[{label}] killing {name}");
            match transport.exec(request, Some(label.as_str())) {
                Ok(o) if o.status == 0 => eprintln!("[{label}] killed {name}"),
                Ok(o) => eprintln!("[{label}] no {name} running (exit {})", o.status),
                Err(e) => eprintln!("[{label}] error killing {name}: {e:#}"),
            }
        });
}

/// The process names one transport sweeps: the plan's client binary, then
/// the tools it spawns. Client first, so a client that reaps its own
/// children on the way down makes the rest a no-op. A Posix host may be Linux
/// or macOS, so a name longer than Linux keeps is swept in both spellings.
fn kill_names(binary_path: &str, shell: ShellType) -> Vec<String> {
    std::iter::once(binary_path)
        .chain(SPAWNED_TOOLS.iter().copied())
        .map(|name| image_name(&binary_basename(name), shell))
        .flat_map(|name| {
            let linux = match shell {
                ShellType::Posix if name.len() > LINUX_NAME_LEN => {
                    name.get(..LINUX_NAME_LEN).map(str::to_owned)
                }
                ShellType::Posix | ShellType::PowerShell => None,
            };
            std::iter::once(name).chain(linux)
        })
        .collect()
}

fn kill_argv(name: &str, shell: ShellType) -> Vec<String> {
    match shell {
        ShellType::PowerShell => vec![
            "taskkill".to_string(),
            "/F".to_string(),
            "/IM".to_string(),
            name.to_string(),
        ],
        ShellType::Posix => vec!["pkill".to_string(), "-x".to_string(), name.to_string()],
    }
}

/// `taskkill /IM` matches on the process image name, which carries an `.exe`
/// a bare tool name lacks. Idempotent, so a `binary_path` that already spells
/// the suffix out passes through unchanged.
fn image_name(name: &str, shell: ShellType) -> String {
    match shell {
        ShellType::PowerShell
            if !Path::new(name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("exe")) =>
        {
            format!("{name}.exe")
        }
        ShellType::Posix | ShellType::PowerShell => name.to_string(),
    }
}

fn binary_basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// `.exe` is appended once, whatever case the name already spells it in.
    #[rstest]
    #[case::bare("pipette", ShellType::PowerShell, "pipette.exe")]
    #[case::lower("pipette.exe", ShellType::PowerShell, "pipette.exe")]
    #[case::upper("PIPETTE.EXE", ShellType::PowerShell, "PIPETTE.EXE")]
    #[case::posix("pipette", ShellType::Posix, "pipette")]
    fn image_name_cases(#[case] name: &str, #[case] shell: ShellType, #[case] want: &str) {
        assert_eq!(image_name(name, shell), want);
    }

    #[test]
    fn basename_windows_path() {
        assert_eq!(
            binary_basename(r"C:\edge-evals\pipette-llamacpp.exe"),
            "pipette-llamacpp.exe"
        );
    }

    #[test]
    fn basename_posix_path() {
        assert_eq!(
            binary_basename("/data/local/tmp/pipette-llamacpp"),
            "pipette-llamacpp"
        );
    }

    #[test]
    fn basename_bare_name() {
        assert_eq!(binary_basename("pipette-llamacpp"), "pipette-llamacpp");
    }

    #[test]
    fn basename_mixed_separators() {
        assert_eq!(
            binary_basename(r"C:/edge-evals\pipette-llamacpp.exe"),
            "pipette-llamacpp.exe"
        );
    }

    #[rstest]
    #[case::tool_on_windows("llama-server", ShellType::PowerShell, "llama-server.exe")]
    #[case::tool_on_posix("llama-server", ShellType::Posix, "llama-server")]
    // Already suffixed: appending again would make `taskkill` match nothing.
    #[case::already_suffixed("pipette-llamacpp.exe", ShellType::PowerShell, "pipette-llamacpp.exe")]
    fn image_name_suffixes_per_shell(
        #[case] name: &str,
        #[case] shell: ShellType,
        #[case] expected: &str,
    ) {
        assert_eq!(image_name(name, shell), expected);
    }

    #[rstest]
    #[case::powershell(
        "llama-bench.exe",
        ShellType::PowerShell,
        vec!["taskkill", "/F", "/IM", "llama-bench.exe"]
    )]
    #[case::posix("llama-bench", ShellType::Posix, vec!["pkill", "-x", "llama-bench"])]
    fn kill_argv_per_shell(
        #[case] name: &str,
        #[case] shell: ShellType,
        #[case] expected: Vec<&str>,
    ) {
        assert_eq!(kill_argv(name, shell), expected);
    }

    /// Linux keeps `llama-batched-b` of the tool's name, macOS all of it, and
    /// Windows matches the image name whole.
    #[rstest]
    #[case::posix(
        "/data/local/tmp/pipette",
        ShellType::Posix,
        vec!["pipette", "llama-server", "llama-bench", "llama-batched-bench", "llama-batched-b"]
    )]
    #[case::powershell(
        r"C:\pipette\pipette.exe",
        ShellType::PowerShell,
        vec!["pipette.exe", "llama-server.exe", "llama-bench.exe", "llama-batched-bench.exe"]
    )]
    fn sweep_covers_the_client_then_the_tools_it_spawns(
        #[case] binary_path: &str,
        #[case] shell: ShellType,
        #[case] expected: Vec<&str>,
    ) {
        assert_eq!(kill_names(binary_path, shell), expected);
    }
}
