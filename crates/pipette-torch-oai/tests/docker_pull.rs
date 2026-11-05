#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

/// Docker reports a pull on its stdout; the operator still sees it, on stderr.
#[test]
fn a_pull_prints_nothing_on_the_clis_stdout() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let docker = dir.path().join("docker");
    // The image is missing, the pull reports as docker does, and `docker run` exits.
    std::fs::write(
        &docker,
        r#"#!/bin/sh
case "$1" in
  image) exit 1 ;;
  pull) echo "Status: Downloaded newer image for $2" ;;
esac
"#,
    )?;
    std::fs::set_permissions(&docker, std::fs::Permissions::from_mode(0o755))?;
    // Under a workspace's `models/`, where the orphan reap reads its label.
    let model = dir.path().join("models").join("m");
    std::fs::create_dir_all(&model)?;
    std::fs::write(model.join("config.json"), "{}")?;

    let output = Command::new(env!("CARGO_BIN_EXE_start_docker_server"))
        .arg(&docker)
        .arg("vllm/vllm-openai:v0.20.2")
        .arg(&model)
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "",
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("Status: Downloaded newer image for vllm/vllm-openai:v0.20.2"),
        "{stderr}"
    );
    Ok(())
}
