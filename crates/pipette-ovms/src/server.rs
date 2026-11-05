//! One OVMS server per cell: launched on a free loopback port, ready once its
//! model is.

use std::{ffi::OsString, path::Path, process::Command, time::Duration};

use anyhow::Context;

use pipette_artifacts_catalog::runtime::ovms::layout::OvmsInstall;
use pipette_http::HttpClient;
use pipette_plan_types::{BatchSize, OvmsExtraRuntimeConfiguration, RuntimeFlags, RuntimeOutput};
use pipette_subprocess::{NotReadyError, start_process};

use crate::completions;

pub(crate) const PROGRAM: &str = "ovms";
pub(crate) const MODEL_NAME: &str = "pipette";
/// A cold NPU compile of a large model takes minutes; a warm GPU load ~1 s.
pub(crate) const READY_TIMEOUT: Duration = Duration::from_secs(3600);
/// The server holds nothing worth flushing, so a stop kills it at once.
pub(crate) const STOP_GRACE: Duration = Duration::ZERO;

/// Creates the compile cache, and puts the release's libraries first on the
/// loader path.
pub(crate) fn command(
    install: &OvmsInstall,
    model_dir: &Path,
    flags: &RuntimeFlags,
    batch_size: BatchSize,
    compile_cache: &Path,
    port: u16,
    client_config: Option<&OvmsExtraRuntimeConfiguration>,
) -> anyhow::Result<Command> {
    std::fs::create_dir_all(compile_cache).with_context(|| {
        format!(
            "creating the OpenVINO compile cache at {}",
            compile_cache.display()
        )
    })?;
    let flag_args = [
        crate::flags::server_args(flags)?,
        crate::flags::batch_args(batch_size.as_u32()),
    ]
    .concat();
    let mut command = Command::new(install.executable());
    pipette_subprocess::apply_client_config(&mut command, client_config);
    let (var, dir) = install.library_path();
    command.env(var, prepend_path(&command, var, &dir)?);
    command.args(server_args(port, model_dir, compile_cache, &flag_args)?);
    Ok(command)
}

/// The argv after the executable. `--model_path` rather than `--source_model`,
/// which writes a `graph.pbtxt` into the (shared, stored) model directory.
fn server_args(
    port: u16,
    model_dir: &Path,
    cache_dir: &Path,
    flag_args: &[String],
) -> anyhow::Result<Vec<String>> {
    let utf8 = |path: &Path, what: &str| {
        path.to_str()
            .map(str::to_owned)
            .with_context(|| format!("{what} is not valid UTF-8: {}", path.display()))
    };
    let mut args: Vec<String> = [
        "--rest_port",
        &port.to_string(),
        "--rest_bind_address",
        "127.0.0.1",
        "--model_path",
        &utf8(model_dir, "model path")?,
        "--model_name",
        MODEL_NAME,
        "--task",
        "text_generation",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    args.extend(flag_args.iter().cloned());
    args.extend(["--cache_dir".to_owned(), utf8(cache_dir, "cache path")?]);
    Ok(args)
}

pub(crate) fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

/// `/v2/models/<name>/ready` is 404 until the model has loaded. OVMS exits on
/// a bad device or model path, which ends the wait with its log.
pub(crate) fn ready(
    url: &str,
) -> anyhow::Result<impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>> {
    let http = HttpClient::loopback("pipette", Duration::from_secs(10))?;
    let model = format!("{url}/v2/models/{MODEL_NAME}/ready");
    Ok(move |_: &RuntimeOutput| {
        pipette_http::send(http.client().get(&model))
            .with_context(|| format!("GET {model}"))
            .map_err(NotReadyError::yet)?;
        Ok(())
    })
}

/// A prompt of `tokens` tokens, sized on an ovms started from `command` on
/// `url` and stopped after: each `/v3/tokenize` call breaks the pooled graph it
/// ran on for the next completion (`docs/ovms.md`).
pub(crate) fn size_prompt(
    command: Command,
    url: &str,
    http: &HttpClient,
    tokens: u32,
) -> anyhow::Result<String> {
    let mut ovms = start_process(PROGRAM, command, STOP_GRACE)?;
    ovms.wait_ready(READY_TIMEOUT, ready(url)?)?;
    completions::prompt_text(http, url, tokens).map_err(|err| err.context(ovms.stop_with_tail()))
}

/// `dir` ahead of what the child would otherwise see for `var`: the client
/// configuration's value when it set one, else this process's.
fn prepend_path(command: &Command, var: &str, dir: &Path) -> anyhow::Result<OsString> {
    let current = command
        .get_envs()
        .find(|(key, _)| key.eq_ignore_ascii_case(var))
        .map_or_else(
            || std::env::var_os(var),
            |(_, value)| value.map(OsString::from),
        );
    let rest = current
        .as_deref()
        .map(|value| std::env::split_paths(value).collect::<Vec<_>>())
        .unwrap_or_default();
    std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(rest))
        .with_context(|| format!("cannot put {} on {var}", dir.display()))
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use httpmock::MockServer;

    #[cfg(unix)]
    use pipette_plan_types::{
        BenchmarkType, ModelType, OpenvinoDevice, RuntimeFlagRef, RuntimeType,
    };
    #[cfg(unix)]
    use pipette_subprocess::{Process, Stream};

    use super::*;

    /// `ovms` from a shell script standing in for the release under `root`,
    /// ready once `mock` answers `/v2/models/<name>/ready`.
    #[cfg(unix)]
    fn start(
        root: &Path,
        body: &str,
        mock: &MockServer,
        ready_timeout: Duration,
    ) -> anyhow::Result<Process> {
        use std::os::unix::fs::PermissionsExt;
        let install = OvmsInstall::under(root);
        let executable = install.executable();
        std::fs::create_dir_all(executable.parent().context("the server has no parent")?)?;
        std::fs::write(&executable, format!("#!/bin/sh\n{body}\n"))?;
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755))?;
        let mut flags = RuntimeFlagRef::new(
            BenchmarkType::EndToEndLatency,
            RuntimeType::Ovms,
            ModelType::Openvino,
        );
        flags.device = Some(OpenvinoDevice::Cpu);
        flags.prefix_caching = Some(false);
        let mut ovms = start_process(
            PROGRAM,
            command(
                &install,
                root,
                &RuntimeFlags::try_from(flags)?,
                BatchSize::one(),
                &root.join("cache"),
                mock.port(),
                None,
            )?,
            STOP_GRACE,
        )?;
        ovms.wait_ready(ready_timeout, ready(&mock.base_url())?)?;
        Ok(ovms)
    }

    #[cfg(unix)]
    #[expect(clippy::disallowed_methods, reason = "`kill -0` prints nothing")]
    fn pid_alive(pid: u32) -> bool {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn the_argv_serves_the_model_directory_under_a_fixed_name() -> anyhow::Result<()> {
        let args = server_args(
            8000,
            Path::new("/m/ir"),
            Path::new("/cache/ovms"),
            &["--target_device".to_owned(), "GPU".to_owned()],
        )?;
        assert_eq!(
            args,
            [
                "--rest_port",
                "8000",
                "--rest_bind_address",
                "127.0.0.1",
                "--model_path",
                "/m/ir",
                "--model_name",
                MODEL_NAME,
                "--task",
                "text_generation",
                "--target_device",
                "GPU",
                "--cache_dir",
                "/cache/ovms"
            ]
        );
        Ok(())
    }

    /// The wait ends once `/v2/models/<name>/ready` answers 200, the server
    /// runs with the release's libraries on its loader path, and stopping it
    /// kills it and keeps what it printed.
    #[cfg(unix)]
    #[test]
    fn the_wait_ends_when_the_model_reports_ready() -> anyhow::Result<()> {
        let mock = MockServer::start();
        let ready = mock.mock(|when, then| {
            when.method("GET")
                .path(format!("/v2/models/{MODEL_NAME}/ready"));
            then.status(200);
        });
        let tmp = tempfile::tempdir()?;
        let mut ovms = start(
            tmp.path(),
            "echo \"lib=$LD_LIBRARY_PATH\"\nexec sleep 60",
            &mock,
            Duration::from_secs(10),
        )?;
        ready.assert();
        // The mock is ready before the script has run; wait for its line.
        let (_, lib) = OvmsInstall::under(tmp.path()).library_path();
        let expected = format!("lib={}", lib.display());
        ovms.wait_for_output(Stream::Stdout, Duration::from_secs(5), |text| {
            text.contains(&expected)
        });
        let pid = ovms.pid();
        ovms.stop();
        assert!(!pid_alive(pid), "the server outlived its stop");
        let stdout = ovms.output().runtime_stdout.unwrap_or_default();
        assert!(stdout.contains(&expected), "{stdout}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_loader_path_leads_with_the_release_libraries() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let mut command = Command::new("true");
        command.env("LD_LIBRARY_PATH", "/opt/a:/opt/b");
        let value = prepend_path(&command, "LD_LIBRARY_PATH", &tmp.path().join("lib"))?;
        assert_eq!(
            value,
            OsString::from(format!(
                "{}:/opt/a:/opt/b",
                tmp.path().join("lib").display()
            ))
        );
        Ok(())
    }

    /// A client configuration's `Path` is the Windows `PATH`: the release goes
    /// ahead of its value rather than into a second variable.
    #[test]
    fn the_loader_path_variable_matches_case_insensitively() -> anyhow::Result<()> {
        let mut command = Command::new("true");
        command.env("Path", "/opt/a");
        let release = Path::new("/release");
        assert_eq!(
            prepend_path(&command, "PATH", release)?,
            std::env::join_paths([release, Path::new("/opt/a")])?
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_server_that_exits_fails_the_wait_with_its_log() -> anyhow::Result<()> {
        let mock = MockServer::start();
        let tmp = tempfile::tempdir()?;
        let Err(err) = start(
            tmp.path(),
            "echo 'target_device: BOGUS is not allowed'\nexit 3",
            &mock,
            Duration::from_secs(30),
        ) else {
            anyhow::bail!("expected the wait to fail");
        };
        let message = format!("{err:#}");
        assert!(message.contains("ovms exited with"), "{message}");
        assert!(message.contains("BOGUS is not allowed"), "{message}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_model_that_never_loads_times_out_and_is_killed() -> anyhow::Result<()> {
        let mock = MockServer::start();
        mock.mock(|when, then| {
            when.method("GET")
                .path(format!("/v2/models/{MODEL_NAME}/ready"));
            then.status(404)
                .body(r#"{"error":"Model with requested name is not found"}"#);
        });
        let tmp = tempfile::tempdir()?;
        let Err(err) = start(
            tmp.path(),
            "exec sleep 60",
            &mock,
            Duration::from_millis(300),
        ) else {
            anyhow::bail!("expected a readiness timeout");
        };
        assert!(
            format!("{err:#}").contains("was not ready within"),
            "{err:#}"
        );
        Ok(())
    }
}
