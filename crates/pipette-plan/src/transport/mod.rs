mod adb;
mod ios;
mod local;
mod process;
mod slurm;
mod ssh;

use pipette_plan_types::{ShellType, TransportConfig};

use crate::shell::{build_shell_command, RemoteExecRequest};

pub struct ExecOutput {
    pub status: i32,
}

/// `srun` resource flags carried by a slurm transport. Each `None`
/// field is omitted from the command line so the cluster default
/// applies.
#[derive(Default)]
pub struct SlurmResources {
    /// Shell command run before `srun` (prefixed with `&&`).
    pub pre_exec: Option<String>,
    pub partition: Option<String>,
    pub account: Option<String>,
    pub gpus: Option<u32>,
    pub cpus: Option<u32>,
    pub time_limit: Option<String>,
    pub mem: Option<String>,
    /// When set, `srun` writes per-job `--output`/`--error` files here
    /// (`%x-%j` patterns). When unset, output streams to the driver.
    pub log_dir: Option<String>,
    pub extra_args: Vec<String>,
}

pub enum Transport {
    Adb {
        serial: String,
        port: Option<u16>,
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    /// adb-over-ssh: the `adb` command runs on `host`, via the `ssh`
    /// primitive — the driver needs no adb and no tunnel.
    AdbOverSsh {
        host: String,
        user: Option<String>,
        port: Option<u16>,
        serial: String,
        adb_port: Option<u16>,
        pre_exec: Option<String>,
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    Ssh {
        host: String,
        user: Option<String>,
        port: Option<u16>,
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    Local {
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    /// slurm-local: `srun` runs on this machine (the login node), via
    /// the `local` primitive.
    SlurmLocal {
        resources: SlurmResources,
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    /// slurm-over-ssh: `srun` runs on `host`, via the `ssh` primitive.
    SlurmOverSsh {
        host: String,
        user: Option<String>,
        port: Option<u16>,
        resources: SlurmResources,
        binary_path: String,
        work_dir: String,
        shell: ShellType,
    },
    /// iOS device driven from the host Mac via `xcrun devicectl`. No
    /// remote binary or work dir; the app command is the cell's
    /// `headlessrun` args, launched via the `ios` primitive.
    Ios {
        device_udid: String,
        bundle_id: String,
    },
    IosOverSsh {
        host: String,
        user: Option<String>,
        port: Option<u16>,
        device_udid: String,
        bundle_id: String,
    },
}

impl Transport {
    /// Build a `Transport` from a config, optionally overriding the
    /// adb server port. `adb_port_override = Some(p)` wins over any
    /// per-transport `port` value in the config; `None` falls back to
    /// the config's own value (which may itself be `None`).
    pub fn from_config_with_adb_port(
        config: &TransportConfig,
        adb_port_override: Option<u16>,
    ) -> Self {
        match config {
            TransportConfig::Adb {
                serial,
                port,
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::Adb {
                serial: serial.clone(),
                port: adb_port_override.or(*port),
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            // The `--adb-port` override is for a tunnel from the driver;
            // this variant reaches the adb server in place, so its own
            // `adb_port` is the only source.
            TransportConfig::AdbOverSsh {
                host,
                user,
                port,
                serial,
                adb_port,
                pre_exec,
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::AdbOverSsh {
                host: host.clone(),
                user: user.clone(),
                port: *port,
                serial: serial.clone(),
                adb_port: *adb_port,
                pre_exec: pre_exec.clone(),
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            TransportConfig::Ssh {
                host,
                user,
                port,
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::Ssh {
                host: host.clone(),
                user: user.clone(),
                port: *port,
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            TransportConfig::Local {
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::Local {
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            TransportConfig::SlurmLocal {
                pre_exec,
                partition,
                account,
                gpus,
                cpus,
                time_limit,
                mem,
                log_dir,
                extra_srun_args,
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::SlurmLocal {
                resources: SlurmResources {
                    pre_exec: pre_exec.clone(),
                    partition: partition.clone(),
                    account: account.clone(),
                    gpus: *gpus,
                    cpus: *cpus,
                    time_limit: time_limit.clone(),
                    mem: mem.clone(),
                    log_dir: log_dir.clone(),
                    extra_args: extra_srun_args.clone(),
                },
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            TransportConfig::SlurmOverSsh {
                host,
                user,
                port,
                pre_exec,
                partition,
                account,
                gpus,
                cpus,
                time_limit,
                mem,
                log_dir,
                extra_srun_args,
                binary_path,
                work_dir,
                shell,
                ..
            } => Transport::SlurmOverSsh {
                host: host.clone(),
                user: user.clone(),
                port: *port,
                resources: SlurmResources {
                    pre_exec: pre_exec.clone(),
                    partition: partition.clone(),
                    account: account.clone(),
                    gpus: *gpus,
                    cpus: *cpus,
                    time_limit: time_limit.clone(),
                    mem: mem.clone(),
                    log_dir: log_dir.clone(),
                    extra_args: extra_srun_args.clone(),
                },
                binary_path: binary_path.clone(),
                work_dir: work_dir.clone(),
                shell: *shell,
            },
            TransportConfig::Ios {
                device_udid,
                bundle_id,
                ..
            } => Transport::Ios {
                device_udid: device_udid.clone(),
                bundle_id: bundle_id.clone(),
            },
            TransportConfig::IosOverSsh {
                host,
                user,
                port,
                device_udid,
                bundle_id,
                ..
            } => Transport::IosOverSsh {
                host: host.clone(),
                user: user.clone(),
                port: *port,
                device_udid: device_udid.clone(),
                bundle_id: bundle_id.clone(),
            },
        }
    }

    pub fn binary_path(&self) -> &str {
        match self {
            Transport::Adb { binary_path, .. }
            | Transport::AdbOverSsh { binary_path, .. }
            | Transport::Ssh { binary_path, .. }
            | Transport::Local { binary_path, .. }
            | Transport::SlurmLocal { binary_path, .. }
            | Transport::SlurmOverSsh { binary_path, .. } => binary_path,
            // iOS launches the app via `xcrun`; no pre-provisioned binary.
            Transport::Ios { .. } | Transport::IosOverSsh { .. } => "xcrun",
        }
    }

    pub fn work_dir(&self) -> &str {
        match self {
            Transport::Adb { work_dir, .. }
            | Transport::AdbOverSsh { work_dir, .. }
            | Transport::Ssh { work_dir, .. }
            | Transport::Local { work_dir, .. }
            | Transport::SlurmLocal { work_dir, .. }
            | Transport::SlurmOverSsh { work_dir, .. } => work_dir,
            Transport::Ios { .. } | Transport::IosOverSsh { .. } => "",
        }
    }

    /// Kill the client where the transport addresses a device rather than a
    /// binary; `None` when it has a `binary_path` for `runner::kill` to sweep
    /// by name instead. Only the iOS transports take the former route today.
    ///
    /// Not routed through [`Self::exec`]: an iOS transport turns argv into *app*
    /// arguments, so a kill sent that way would launch the app rather than end
    /// it. `devicectl` runs where the device is paired — locally for `Ios`, on
    /// the intermediate host for `IosOverSsh`.
    pub fn kill_client(&self) -> Option<anyhow::Result<ExecOutput>> {
        match self {
            Transport::Ios {
                device_udid,
                bundle_id,
            } => {
                let cmd = ios::kill_command(device_udid, bundle_id);
                Some(process::run_quiet("sh", &["-c", &cmd]))
            }
            Transport::IosOverSsh {
                host,
                user,
                port,
                device_udid,
                bundle_id,
            } => {
                let cmd = ios::kill_command(device_udid, bundle_id);
                Some(ssh::exec_quiet(host, user.as_deref(), *port, &cmd))
            }
            _ => None,
        }
    }

    pub fn target_label(&self) -> String {
        match self {
            Transport::Adb { serial, port, .. } => match port {
                Some(p) => format!("adb:{serial}@{p}"),
                None => format!("adb:{serial}"),
            },
            // `user@host` already carries an `@`, so the label separates
            // with `:`.
            Transport::AdbOverSsh {
                host, user, serial, ..
            } => format!("adb-ssh:{}:{serial}", userhost(host, user)),
            Transport::Ssh {
                host, user, port, ..
            } => {
                let reach = userhost(host, user);
                match port {
                    Some(p) => format!("ssh:{reach}:{p}"),
                    None => format!("ssh:{reach}"),
                }
            }
            Transport::Local { .. } => "local".to_string(),
            Transport::SlurmLocal { resources, .. } => slurm_label("local", &resources.partition),
            Transport::SlurmOverSsh {
                host,
                user,
                resources,
                ..
            } => slurm_label(&userhost(host, user), &resources.partition),
            Transport::Ios { device_udid, .. } => format!("ios:{device_udid}"),
            Transport::IosOverSsh {
                host,
                user,
                device_udid,
                ..
            } => format!("ios-ssh:{}:{device_udid}", userhost(host, user)),
        }
    }

    fn shell_type(&self) -> ShellType {
        match self {
            Transport::Adb { shell, .. }
            | Transport::AdbOverSsh { shell, .. }
            | Transport::Ssh { shell, .. }
            | Transport::Local { shell, .. }
            | Transport::SlurmLocal { shell, .. }
            | Transport::SlurmOverSsh { shell, .. } => *shell,
            // Unused: iOS bypasses `build_shell_command` (see `exec`).
            Transport::Ios { .. } | Transport::IosOverSsh { .. } => ShellType::Posix,
        }
    }

    /// Preview the command that would be executed remotely.
    pub fn preview_exec(&self, request: &RemoteExecRequest) -> String {
        let shell = self.shell_type();
        let mut lines = Vec::new();
        lines.push(format!("target: {}", self.target_label()));
        lines.push(format!(
            "remote shell: {}",
            match shell {
                ShellType::Posix => "posix",
                ShellType::PowerShell => "powershell",
            }
        ));
        if let Some(cwd) = &request.cwd {
            lines.push(format!("remote cwd: {cwd}"));
        }
        let env_str = if request.env.is_empty() {
            "(none)".to_string()
        } else {
            request
                .env
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(format!("remote env: {env_str}"));
        let argv_display: Vec<String> = request
            .argv
            .iter()
            .map(|arg| {
                if arg.contains(' ') || arg.contains('\\') {
                    format!("\"{}\"", arg)
                } else {
                    arg.clone()
                }
            })
            .collect();
        lines.push(format!("remote argv: {}", argv_display.join(" ")));
        lines.join("\n")
    }

    /// Cheap reachability check for the executor host. Must NOT allocate
    /// cluster resources: for slurm this deliberately bypasses `srun`
    /// (which would queue for and hold a GPU just to echo), checking the
    /// login node directly instead — local for slurm-local, a bare
    /// `ssh host echo ok` for slurm-over-ssh. Other transports echo
    /// through their normal path, which is already cheap.
    pub fn probe(&self) -> anyhow::Result<ExecOutput> {
        match self {
            Transport::SlurmLocal { shell, .. } => local::exec_quiet(*shell, "echo ok"),
            Transport::SlurmOverSsh {
                host, user, port, ..
            } => ssh::exec_quiet(host, user.as_deref(), *port, "echo ok"),
            // Don't launch the app to probe — query device info instead.
            Transport::Ios { device_udid, .. } => ios::probe(device_udid),
            Transport::IosOverSsh {
                host,
                user,
                port,
                device_udid,
                ..
            } => ssh::exec_quiet(
                host,
                user.as_deref(),
                *port,
                &ios::remote_probe_command(device_udid),
            ),
            _ => self.exec_quiet(RemoteExecRequest {
                argv: vec!["echo".to_string(), "ok".to_string()],
                env: Vec::new(),
                cwd: None,
                job_name: None,
            }),
        }
    }

    /// Execute a command on the remote device, discarding stdout/stderr.
    pub fn exec_quiet(&self, request: RemoteExecRequest) -> anyhow::Result<ExecOutput> {
        let shell_cmd = build_shell_command(self.shell_type(), &request);
        let job_name = request.job_name.as_deref();
        match self {
            Transport::Adb { serial, port, .. } => adb::exec_quiet(serial, *port, &shell_cmd),
            // adb-over-ssh reuses both primitives: render the `adb` command
            // for the intermediate host's shell, then run it there over ssh.
            Transport::AdbOverSsh {
                host,
                user,
                port,
                serial,
                adb_port,
                pre_exec,
                ..
            } => ssh::exec_quiet(
                host,
                user.as_deref(),
                *port,
                &adb::remote_command(serial, *adb_port, pre_exec.as_deref(), &shell_cmd),
            ),
            Transport::Ssh {
                host, user, port, ..
            } => ssh::exec_quiet(host, user.as_deref(), *port, &shell_cmd),
            Transport::Local { shell, .. } => local::exec_quiet(*shell, &shell_cmd),
            // slurm reuses the local/ssh primitives: build the `srun`
            // command once, then run it here (local) or there (ssh).
            Transport::SlurmLocal {
                resources, shell, ..
            } => local::exec_quiet(
                *shell,
                &slurm::srun_command(resources, job_name, &shell_cmd)?,
            ),
            Transport::SlurmOverSsh {
                host,
                user,
                port,
                resources,
                ..
            } => ssh::exec_quiet(
                host,
                user.as_deref(),
                *port,
                &slurm::srun_command(resources, job_name, &shell_cmd)?,
            ),
            // iOS launches a process with an argv, not a shell command,
            // so it takes `request.argv` directly and ignores `shell_cmd`.
            Transport::Ios {
                device_udid,
                bundle_id,
            } => ios::exec_quiet(device_udid, bundle_id, &request.argv),
            Transport::IosOverSsh {
                host,
                user,
                port,
                device_udid,
                bundle_id,
            } => ssh::exec_quiet(
                host,
                user.as_deref(),
                *port,
                &ios::remote_command(device_udid, bundle_id, &request.argv),
            ),
        }
    }

    /// Execute a command on the remote device, streaming stdout and stderr to
    /// the local terminal in real time.
    pub fn exec(
        &self,
        request: RemoteExecRequest,
        prefix: Option<&str>,
    ) -> anyhow::Result<ExecOutput> {
        let shell_cmd = build_shell_command(self.shell_type(), &request);
        let job_name = request.job_name.as_deref();
        match self {
            Transport::Adb { serial, port, .. } => {
                adb::exec_streaming(serial, *port, &shell_cmd, prefix)
            }
            Transport::AdbOverSsh {
                host,
                user,
                port,
                serial,
                adb_port,
                pre_exec,
                ..
            } => ssh::exec_streaming(
                host,
                user.as_deref(),
                *port,
                &adb::remote_command(serial, *adb_port, pre_exec.as_deref(), &shell_cmd),
                prefix,
            ),
            Transport::Ssh {
                host, user, port, ..
            } => ssh::exec_streaming(host, user.as_deref(), *port, &shell_cmd, prefix),
            Transport::Local { shell, .. } => local::exec_streaming(*shell, &shell_cmd, prefix),
            Transport::SlurmLocal {
                resources, shell, ..
            } => local::exec_streaming(
                *shell,
                &slurm::srun_command(resources, job_name, &shell_cmd)?,
                prefix,
            ),
            Transport::SlurmOverSsh {
                host,
                user,
                port,
                resources,
                ..
            } => ssh::exec_streaming(
                host,
                user.as_deref(),
                *port,
                &slurm::srun_command(resources, job_name, &shell_cmd)?,
                prefix,
            ),
            Transport::Ios {
                device_udid,
                bundle_id,
            } => ios::exec_streaming(device_udid, bundle_id, &request.argv, prefix),
            // The app's `BENCH_DONE` line is the result contract, so the ssh
            // hop has to scan stdout for it exactly as the local path does —
            // `devicectl`'s own exit code does not carry the app's status.
            //
            // `request.env` is dropped here as it is for `Ios`, and the ssh hop
            // is not a way in: assignments would set variables on the
            // intermediate Mac, not inside the app on the device, so a
            // forwarded PIPETTE_HF_TOKEN still never arrives.
            Transport::IosOverSsh {
                host,
                user,
                port,
                device_udid,
                bundle_id,
            } => ssh::exec_streaming_scanning(
                host,
                user.as_deref(),
                *port,
                &ios::remote_command(device_udid, bundle_id, &request.argv),
                prefix,
                ios::BENCH_DONE_SENTINEL,
            )
            .map(|(out, scanned)| ExecOutput {
                status: scanned.unwrap_or(out.status),
            }),
        }
    }
}

fn userhost(host: &str, user: &Option<String>) -> String {
    match user {
        Some(u) => format!("{u}@{host}"),
        None => host.to_string(),
    }
}

/// Format a slurm target label as `slurm:<reach>[:<partition>]`.
fn slurm_label(reach: &str, partition: &Option<String>) -> String {
    match partition {
        Some(p) => format!("slurm:{reach}:{p}"),
        None => format!("slurm:{reach}"),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// `kill_client` decides which transports kill on the device and which fall
    /// through to `runner::kill`'s binary-name sweep. A transport that wrongly
    /// answered `Some` would have its `pkill` skipped and never be killed at
    /// all; one that wrongly answered `None` would get `pkill -x xcrun` on the
    /// intermediate Mac.
    #[rstest]
    #[case::ios(
        r#"type = "ios"
client_id = "phone-1"
device_udid = "UDID-1"
"#,
        true
    )]
    #[case::ios_over_ssh(
        r#"type = "ios_over_ssh"
client_id = "phone-1"
host = "mac-1"
device_udid = "UDID-1"
"#,
        true
    )]
    #[case::ssh(
        r#"type = "ssh"
client_id = "box-1"
host = "box-1"
binary_path = "/opt/pipette"
work_dir = "/opt"
"#,
        false
    )]
    #[case::adb(
        r#"type = "adb"
client_id = "phone-1"
serial = "R3GL30CRBGM"
binary_path = "/data/local/tmp/pipette"
work_dir = "/data/local/tmp"
"#,
        false
    )]
    #[case::local(
        r#"type = "local"
client_id = "here"
binary_path = "/opt/pipette"
work_dir = "/opt"
"#,
        false
    )]
    fn kill_client_targets_only_device_addressed_transports(
        #[case] toml_src: &str,
        #[case] kills_on_device: bool,
    ) -> anyhow::Result<()> {
        let cfg: TransportConfig = toml::from_str(toml_src)?;
        let transport = Transport::from_config_with_adb_port(&cfg, None);
        assert_eq!(transport.kill_client().is_some(), kills_on_device);
        Ok(())
    }

    fn adb_over_ssh(extra: &str) -> anyhow::Result<TransportConfig> {
        Ok(toml::from_str(&format!(
            "type = \"adb_over_ssh\"\n\
             client_id = \"phone-1\"\n\
             host = \"controller\"\n\
             serial = \"R3GL30CRBGM\"\n\
             binary_path = \"/data/local/tmp/pipette-evals/pipette\"\n\
             work_dir = \"/data/local/tmp/pipette-evals\"\n\
             {extra}"
        ))?)
    }

    /// The override retargets a tunnel from the driver; this transport reaches
    /// the adb server in place, so its own `adb_port` must survive the flag.
    #[rstest]
    #[case::override_ignored("adb_port = 5037\n", Some(5038), Some(5037))]
    #[case::no_port_stays_none("", Some(5038), None)]
    fn adb_over_ssh_ignores_the_adb_port_override(
        #[case] extra: &str,
        #[case] flag: Option<u16>,
        #[case] expected: Option<u16>,
    ) -> anyhow::Result<()> {
        let transport = Transport::from_config_with_adb_port(&adb_over_ssh(extra)?, flag);
        match transport {
            Transport::AdbOverSsh { adb_port, .. } => assert_eq!(adb_port, expected),
            other => anyhow::bail!("expected AdbOverSsh, got {}", other.target_label()),
        }
        Ok(())
    }

    /// The label names both hops, so a log line identifies the controller and
    /// the handset — and marks the transport kind, since a plain `adb` label
    /// carries the adb server port in the same position.
    #[rstest]
    #[case::with_user("user = \"liquid\"\n", "adb-ssh:liquid@controller:R3GL30CRBGM")]
    #[case::without_user("", "adb-ssh:controller:R3GL30CRBGM")]
    fn adb_over_ssh_label_names_the_hop_and_the_device(
        #[case] extra: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let transport = Transport::from_config_with_adb_port(&adb_over_ssh(extra)?, None);
        assert_eq!(transport.target_label(), expected);
        Ok(())
    }
}
