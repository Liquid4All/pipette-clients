//! `pipette benchmarks run` whose stderr reader closes, as when the ssh session
//! of `pipette-ctl` drops: the run stops instead of measuring on.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn a_benchmark_run_stops_when_its_stderr_closes() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let marker = dir.path().join("docker-called");
    let fake = dir.path().join("docker");
    // Quiet and stuck in its first call, so only the closed pipe can stop the run.
    std::fs::write(
        &fake,
        format!("#!/bin/sh\ntouch '{}'\nexec sleep 30\n", marker.display()),
    )?;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755))?;
    let pipette = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_pipette"));
        command.arg("--work-dir").arg(dir.path());
        command
    };
    [&["init"][..], &["benchmarks", "init-local"]]
        .iter()
        .try_for_each(|setup| -> anyhow::Result<()> {
            let output = pipette().args(*setup).output()?;
            anyhow::ensure!(
                output.status.success(),
                "{setup:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            Ok(())
        })?;

    #[expect(clippy::disallowed_methods, reason = "stdout is discarded")]
    let mut run = pipette()
        .arg("--docker-path")
        .arg(&fake)
        .args([
            "benchmarks",
            "run",
            "--benchmark",
            "local/eval_smoke",
            "--model",
            "torch://repo=Qwen/Qwen2.5-0.5B-Instruct",
            "--runtime",
            "docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=cpu",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !marker.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(run.stderr.take());
    let deadline = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(status) = run.try_wait()? {
            break Some(status);
        }
        if Instant::now() >= deadline {
            run.kill()?;
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    anyhow::ensure!(marker.exists(), "the run never called docker");
    anyhow::ensure!(
        status.and_then(|status| status.code()) == Some(141),
        "expected exit 141, got {status:?}"
    );
    Ok(())
}
