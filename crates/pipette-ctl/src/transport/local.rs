use pipette_plan_types::ShellType;

use crate::transport::{
    ExecOutput,
    process::{run_quiet, run_streaming},
};

pub(crate) fn exec_quiet(shell: ShellType, shell_cmd: &str) -> anyhow::Result<ExecOutput> {
    let (program, args) = wrap(shell, shell_cmd);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_quiet(program, &refs)
}

pub(crate) fn exec_streaming(
    shell: ShellType,
    shell_cmd: &str,
    prefix: Option<&str>,
) -> anyhow::Result<ExecOutput> {
    let (program, args) = wrap(shell, shell_cmd);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_streaming(program, &refs, prefix)
}

fn wrap(shell: ShellType, shell_cmd: &str) -> (&'static str, Vec<String>) {
    match shell {
        ShellType::Posix => ("sh", vec!["-c".into(), shell_cmd.into()]),
        ShellType::PowerShell => (
            "powershell.exe",
            vec!["-NoProfile".into(), "-Command".into(), shell_cmd.into()],
        ),
    }
}
