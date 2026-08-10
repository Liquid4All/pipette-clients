use pipette_plan_types::{Plan, ShellType};

use crate::{shell::RemoteExecRequest, transport::Transport};

/// Exit status a device-addressed kill uses for "could not query the device",
/// as opposed to querying it fine and finding nothing to kill.
const DEVICE_UNREACHABLE: i32 = 2;

/// Kill the pipette client process on every transport in the plan.
///
/// Targets the binary referenced by each `transports[*].binary_path` by
/// its basename: `taskkill /F /IM <name>` under PowerShell, `pkill -x
/// <name>` under Posix. A non-zero exit (no matching process) is
/// reported but does not abort the sweep.
pub fn kill_transports(plan: &Plan, adb_port: Option<u16>) -> anyhow::Result<()> {
    for cfg in plan.transports.iter() {
        let transport = Transport::from_config_with_adb_port(cfg, adb_port);
        // Device-addressed transports kill on the device — see `kill_client`.
        if let Some(result) = transport.kill_client() {
            let label = transport.target_label();
            eprintln!("[{label}] terminating the app on the device");
            match result {
                Ok(o) if o.status == 0 => eprintln!("[{label}] killed"),
                Ok(o) if o.status == DEVICE_UNREACHABLE => {
                    eprintln!("[{label}] device error: could not query it")
                }
                Ok(o) => eprintln!("[{label}] no matching process (exit {})", o.status),
                Err(e) => eprintln!("[{label}] error: {e:#}"),
            }
            continue;
        }
        let label = transport.target_label();
        let basename = binary_basename(cfg.binary_path());
        let argv = match cfg.shell() {
            ShellType::PowerShell => vec![
                "taskkill".to_string(),
                "/F".to_string(),
                "/IM".to_string(),
                basename.clone(),
            ],
            ShellType::Posix => vec!["pkill".to_string(), "-x".to_string(), basename.clone()],
        };
        let request = RemoteExecRequest {
            argv,
            env: Vec::new(),
            cwd: None,
            job_name: Some("pipette-kill".to_string()),
        };
        eprintln!("[{label}] killing {basename}");
        match transport.exec(request, Some(label.as_str())) {
            Ok(o) if o.status == 0 => eprintln!("[{label}] killed"),
            Ok(o) => eprintln!("[{label}] no matching process (exit {})", o.status),
            Err(e) => eprintln!("[{label}] error: {e:#}"),
        }
    }
    Ok(())
}

fn binary_basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
