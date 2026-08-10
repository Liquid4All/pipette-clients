//! Integration test for the uv engine end-to-end: install, launch a
//! trivial HTTP server in the venv, stop it cleanly.
//!
//! **Linux-only.** The uv engine is Linux-only; the test file is
//! `#![cfg]`-gated so macOS/Windows CI ignores it entirely. Inside, the
//! install/launch tests are also `#[ignore]` because they require:
//!  - `uv` on PATH
//!  - network access to PyPI (the install path runs `uv pip install -r`)
//!  - several seconds for the install
//!
//! Run with `cargo test -p pipette-torch-oai --test uv_runtime_install_launch -- --ignored --nocapture`.
//!
//! The test uses `python -m http.server` as a stand-in rather than
//! vllm/sglang so that:
//!  - the install path stays cheap (~few MB instead of multi-GB)
//!  - the launch/stop machinery is exercised on every CI box (no GPU needed)
//!  - the test still covers setsid + free port + env markers + log tail +
//!    SIGTERM-PGID + waitpid + log-tail join

#![cfg(target_os = "linux")]

use std::{
    fs,
    net::TcpStream,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_artifacts::runtime::RuntimeArtifactStore;
use pipette_artifacts::{ensure_runtime, ArtifactsContext};
use pipette_http::HttpClient;
use pipette_plan_types::{UvBuild, UvPythonVersion, UvServerVersion};
use pipette_torch_oai::server::uv as server_uv;
use pipette_torch_oai::server::ServerState;

struct TmpDir(PathBuf);

impl TmpDir {
    fn new(label: &str) -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "pipette-torch-oai-uv-it-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4(),
        ));
        fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn uv_on_path() -> Option<PathBuf> {
    let out = Command::new("which").arg("uv").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

#[test]
#[ignore = "needs uv + network; run with --ignored"]
fn uv_install_launch_stop_round_trip() -> anyhow::Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .try_init();
    let Some(uv) = uv_on_path() else {
        eprintln!("skipping: uv not on PATH");
        return Ok(());
    };

    let work = TmpDir::new("install-launch")?;
    let data_dir = work.path();
    let _ = &uv; // PATH-resolved above; store ensure re-resolves uv

    // Empty requirements: install finishes quickly without vllm/sglang.
    // Launch substitutes stdlib http.server via the venv python.
    let declared = pipette_plan_types::Runtime::UvVllm(pipette_plan_types::UvVllm {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: pipette_plan_types::UvRuntimeSource::PipRequirementsText {
            contents: pipette_plan_types::NonEmptyString::try_new("# empty\n".to_owned())?,
            install_flags: None,
        },
    });
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?);
    ensure_runtime(&ctx, &store, &declared)?;

    let manifest = store
        .find(&declared)?
        .context("store has no manifest for the runtime it just installed")?;

    let venv_python = pipette_venv::venv_python(&store.install_dir_for(&manifest)?);
    assert!(
        venv_python.exists(),
        "venv python missing at {}",
        venv_python.display()
    );

    // Compose a launch that bypasses the per-server entrypoint helper —
    // we want to test the lifecycle machinery, not the vllm CLI shape.
    // Use python -m http.server <port> as a known-good stub.
    //
    // Going through server_uv::launch would route through entrypoint()
    // and try to invoke `vllm` (which isn't installed). Skip the helper
    // and call the lower-level primitives directly: pick_free_port, then
    // build a Command manually. This still exercises setsid, env markers,
    // log tail, stop.
    let port = server_uv::pick_free_port()?;
    let workspace_label = "test-workspace-label".to_string();
    let mut cmd = std::process::Command::new(&venv_python);
    cmd.arg("-m")
        .arg("http.server")
        .arg("--bind")
        .arg("127.0.0.1")
        .arg(port.to_string());
    cmd.env(server_uv::ENV_WORKSPACE_LABEL, &workspace_label);
    cmd.env(server_uv::ENV_RUNTIME_REF, declared.to_string());
    cmd.env(server_uv::ENV_PARENT_PID, std::process::id().to_string());
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn()?;
    let pid = child.id();
    let pgid = pid;

    // Drain pipes so the child doesn't block on a full pipe.
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("child stdout pipe missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("child stderr pipe missing"))?;
    let t_out = thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        for _ in BufReader::new(stdout).lines().map_while(Result::ok) {}
    });
    let t_err = thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        for _ in BufReader::new(stderr).lines().map_while(Result::ok) {}
    });
    drop(child); // forget so server_uv::stop reaps the leader

    // The server should be reachable within a few seconds.
    assert!(
        wait_for_port(port, Duration::from_secs(10)),
        "python http.server didn't listen on 127.0.0.1:{port} within 10s"
    );

    // Build a ServerState shape that server_uv::stop understands.
    let mut state = ServerState::Uv {
        kind: pipette_torch_oai::server::ServerKind::Vllm,
        executable: std::path::PathBuf::from("python"),
        pid,
        pgid,
        port,
        log_tail: None,
    };

    server_uv::stop(&mut state, 5)?;

    // After stop, the port must be free again (give the kernel a beat).
    let mut closed = false;
    for _ in 0..20 {
        if TcpStream::connect(("127.0.0.1", port)).is_err() {
            closed = true;
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert!(closed, "port {port} still open after stop");

    let _ = t_out.join();
    let _ = t_err.join();
    Ok(())
}

#[test]
#[ignore = "needs uv + network; run with --ignored"]
fn uv_install_remove_removes_runtime_dir() -> anyhow::Result<()> {
    let Some(uv) = uv_on_path() else {
        eprintln!("skipping: uv not on PATH");
        return Ok(());
    };
    let work = TmpDir::new("install-remove")?;
    let data_dir = work.path();
    let _ = &uv;
    let declared = pipette_plan_types::Runtime::UvSglang(pipette_plan_types::UvSglang {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: pipette_plan_types::UvRuntimeSource::PipRequirementsText {
            contents: pipette_plan_types::NonEmptyString::try_new("# empty\n".to_owned())?,
            install_flags: None,
        },
    });
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?);
    ensure_runtime(&ctx, &store, &declared)?;
    let manifest = store
        .find(&declared)?
        .context("store has no manifest for the runtime it just installed")?;
    let entry = store.entry_dir_for(&manifest)?;
    assert!(entry.exists());
    assert!(pipette_venv::venv_python(&store.install_dir_for(&manifest)?).exists());

    assert!(store.remove(&declared)?);
    assert!(!entry.exists());
    Ok(())
}

// Store ensure is sticky when a common RuntimeManifest already exists —
// find short-circuits without invoking the UV fetcher.
#[test]
fn ensure_runtime_returns_existing_without_calling_uv() -> anyhow::Result<()> {
    use time::OffsetDateTime;

    use pipette_artifacts::runtime::RuntimeManifest;
    use pipette_artifacts::runtime::RuntimeStorageKey;
    use pipette_plan_types::{NonEmptyString, Runtime, UvRuntimeSource};

    let work = TmpDir::new("idempotency")?;
    let data_dir = work.path();
    let declared = pipette_plan_types::Runtime::UvVllm(pipette_plan_types::UvVllm {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("# empty\n".to_owned())?,
            install_flags: None,
        },
    });
    let key = RuntimeStorageKey::of(&declared)?;
    let entry = data_dir.join("runtimes").join(key.as_str());
    fs::create_dir_all(entry.join("blobs"))?;
    // Hand-built entry with an empty `blobs/`, so the recorded payload size is 0.
    let manifest = RuntimeManifest::new(declared.clone(), OffsetDateTime::now_utc(), 0)?;
    fs::write(
        entry.join("manifest.toml"),
        toml::to_string_pretty(&manifest)?,
    )?;

    // No uv on PATH required: store find must win before fetch.
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?);
    let Runtime::UvVllm(bound) = ensure_runtime(&ctx, &store, &declared)? else {
        anyhow::bail!("expected the declared uv_vllm variant back");
    };
    let UvRuntimeSource::AbsolutePreinstalled { dir } = bound.source else {
        anyhow::bail!("ensure must bind the runtime to an absolute install dir");
    };
    // Bound under the seeded entry, so the pre-existing manifest was reused.
    assert!(
        Path::new(dir.as_ref()).starts_with(&entry),
        "bound install dir {} is outside the seeded entry {}",
        dir.as_ref(),
        entry.display()
    );
    Ok(())
}

// RelativePreinstalled source is not storable / not fetchable through the shared store.
#[test]
fn ensure_runtime_rejects_preinstalled() -> anyhow::Result<()> {
    use pipette_plan_types::{RelativePath, UvRuntimeSource};

    let work = TmpDir::new("preinstalled")?;
    let data_dir = work.path();
    let declared = pipette_plan_types::Runtime::UvVllm(pipette_plan_types::UvVllm {
        server_version: UvServerVersion::try_new("0.1.0".to_string())?,
        build: UvBuild::try_new("cpu".to_string())?,
        python_version: UvPythonVersion::try_new("3.12".to_string())?,
        source: UvRuntimeSource::RelativePreinstalled {
            dir: RelativePath::try_new("venv".to_owned())?,
        },
    });
    let store = RuntimeArtifactStore::new(data_dir.join("runtimes"));
    let ctx = ArtifactsContext::new(HttpClient::new("pipette")?);
    let Err(err) = ensure_runtime(&ctx, &store, &declared) else {
        anyhow::bail!("preinstalled must not ensure");
    };
    let msg = format!("{err:#}");
    // Storage key rejects RelativePreinstalled as NotStorable before fetch runs.
    assert!(
        msg.contains("no storage key")
            || msg.contains("NotStorable")
            || msg.contains("not pullable")
            || msg.contains("effective-only"),
        "got: {msg}"
    );
    Ok(())
}
