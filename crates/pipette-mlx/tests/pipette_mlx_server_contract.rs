#![cfg(target_os = "macos")]

use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use anyhow::Context;

fn python_command() -> Command {
    let mut command = Command::new("/usr/bin/env");
    command.arg("python3");
    command
}

fn has_python3() -> bool {
    let status = python_command()
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(status, Ok(status) if status.success())
}

#[test]
fn python_server_contract_tests_pass() -> anyhow::Result<()> {
    if !has_python3() {
        eprintln!("skipping: python3 not on this host");
        return Ok(());
    }

    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_path = crate_dir.join("tests").join("pipette_mlx_server_test.py");
    let status = python_command()
        .arg(&test_path)
        .current_dir(&crate_dir)
        .status()
        .with_context(|| format!("failed to run {}", test_path.display()))?;
    if !status.success() {
        anyhow::bail!("{} failed with {status}", test_path.display());
    }
    Ok(())
}
