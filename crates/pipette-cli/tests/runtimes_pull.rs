//! `pipette runtimes pull` through the binary: stdout carries only the line the
//! command produces, and what the programs it runs report goes to stderr.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

use rstest::rstest;

const FAKE_DOCKER: &str = r#"echo "Status: Downloaded newer image for $2""#;

/// Prints on every call, so each uv step is pinned. `venv` also makes the
/// interpreter the install checks for.
const FAKE_UV: &str = r#"echo "uv $1 $2"
if [ "$1" = venv ]; then
  for venv; do :; done
  mkdir -p "$venv/bin" && touch "$venv/bin/python"
fi"#;

const DOCKER_VLLM: &str = r#"{"type":"docker_vllm","image_name":"vllm/vllm-openai","image_tag":"v0.20.2","flavor":"cpu"}"#;

const UV_VLLM: &str = r#"{"type":"uv_vllm","server_version":"0.10.0","build":"cu121","python_version":"3.12","source":{"type":"pip_requirements_text","contents":"vllm==0.10.0"}}"#;

#[rstest]
#[case::docker(
    "docker",
    "--docker-path",
    FAKE_DOCKER,
    DOCKER_VLLM,
    &["Status: Downloaded newer image for vllm/vllm-openai:v0.20.2"]
)]
#[cfg_attr(not(target_os = "linux"), ignore = "uv vLLM installs only on Linux")]
#[case::uv(
    "uv",
    "--uv-path",
    FAKE_UV,
    UV_VLLM,
    &["uv venv --quiet", "uv pip install"]
)]
fn a_runtime_pull_prints_only_its_result(
    #[case] program: &str,
    #[case] flag: &str,
    #[case] script: &str,
    #[case] runtime: &str,
    #[case] reported: &[&str],
) -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let fake = dir.path().join(program);
    std::fs::write(&fake, format!("#!/bin/sh\n{script}\n"))?;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))?;
    let pipette = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_pipette"));
        command.arg("--work-dir").arg(dir.path());
        command
    };
    let init = pipette().arg("init").output()?;
    anyhow::ensure!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let output = pipette()
        .arg(flag)
        .arg(&fake)
        .args(["runtimes", "pull", "--runtime", runtime])
        .output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        matches!(
            stdout.lines().collect::<Vec<_>>().as_slice(),
            [line] if line.starts_with("Installed `")
        ),
        "stdout: {stdout}\nstderr: {stderr}"
    );
    let missing: Vec<_> = reported
        .iter()
        .copied()
        .filter(|line| !stderr.contains(line))
        .collect();
    assert!(missing.is_empty(), "{missing:?} not in stderr: {stderr}");
    Ok(())
}
