//! A container serving requests, run attached so the docker CLI's output and
//! exit are the container's.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::time::Duration;

use pipette_plan_types::{EnvForward, RuntimeOutput};

use crate::command::{echo_info, render};
use crate::env::apply_envs;
use crate::error::{Chain, NotReadyError, ReadyError, StartError, StatusError, Stopped};
use crate::output::output_within;
use crate::spawned::{Spawned, Stop, Stream};

/// How long the docker CLI gets to exit once its container has stopped: it
/// waits for `--rm` to remove it first.
pub(crate) const CLI_EXIT: Duration = Duration::from_secs(10);
/// How long a `docker stop` or `docker kill` may take on top of its own grace.
const COMMAND_LIMIT: Duration = Duration::from_secs(30);

/// What [`start_docker_process`] runs, after the `docker run --rm --name` it
/// owns.
#[derive(Clone)]
pub struct DockerRun {
    pub name: String,
    pub image: String,
    /// Before the image: `-p`, `--gpus`, `-v`, `--label`, `--shm-size`...
    pub options: Vec<String>,
    /// After the image: the server's own arguments.
    pub args: Vec<String>,
    /// Set on the docker CLI's own environment by [`apply_envs`] and named with
    /// `-e NAME`, so no value reaches the logged argv. The CLI runs with them
    /// too, so a `DOCKER_*` or `PATH` forward changes the CLI itself.
    pub envs: Vec<EnvForward>,
}

/// A container serving requests, started by [`start_docker_process`].
/// Dropping it stops it.
pub struct DockerProcess {
    spawned: Spawned,
    container: String,
}

/// `docker run --rm --name <name> <options> -e <NAME>... <image> <args>`,
/// attached. Returns while it runs. Errors call it `container <name>`, and its
/// events carry a `container{container, pid}` span.
///
/// `stop_grace` is the `docker stop -t` that [`DockerProcess::stop`] passes.
/// Each line it prints goes to [`log_line`](crate::log_line) until
/// [`DockerProcess::on_line`] names another callback.
pub fn start_docker_process(
    docker: &Path,
    run: DockerRun,
    stop_grace: Duration,
) -> Result<DockerProcess, StartError> {
    let stop = Stop::Docker {
        docker: docker.to_path_buf(),
        container: run.name.clone(),
        grace: stop_grace,
    };
    Ok(DockerProcess {
        spawned: Spawned::start(
            run_command(docker, &run),
            format!("container {}", run.name),
            stop,
        )?,
        container: run.name,
    })
}

/// Names only the variables the forwards set: `-e NAME` with no value on the
/// CLI unsets the image's own.
fn run_command(docker: &Path, run: &DockerRun) -> Command {
    let mut command = Command::new(docker);
    command
        .args(["run", "--rm", "--name", run.name.as_str()])
        .args(&run.options);
    apply_envs(&mut command, &run.envs);
    let names: Vec<OsString> = command
        .get_envs()
        .map(|(name, _)| name.to_owned())
        .collect();
    command
        .args(
            names
                .iter()
                .flat_map(|name| [OsStr::new("-e"), name.as_os_str()]),
        )
        .arg(&run.image)
        .args(&run.args);
    command
}

impl DockerProcess {
    /// The container's name, for `docker container inspect`.
    pub fn container(&self) -> &str {
        &self.container
    }

    /// The `docker run` command line, without the environment.
    pub fn command(&self) -> &[String] {
        self.spawned.command()
    }

    pub fn is_running(&mut self) -> bool {
        self.spawned.is_running()
    }

    /// Wait up to `timeout` for the container to exit: `Some(status)` if it
    /// did, `None` if it is still running. Never stops it.
    pub fn wait(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, StatusError> {
        self.spawned.wait(timeout)
    }

    /// Ask `ready` every 200 ms about what the container has printed so far
    /// until it returns `Ok`. Fails once [`NotReadyError::Failed`] is the answer, the
    /// container exits, or `timeout` has passed, stopping it; the error quotes
    /// the end of its output and keeps the check's last error as its cause.
    pub fn wait_ready(
        &mut self,
        timeout: Duration,
        mut ready: impl FnMut(&RuntimeOutput) -> Result<(), NotReadyError>,
    ) -> Result<(), ReadyError> {
        self.spawned.wait_ready(timeout, &mut ready)
    }

    /// The newest 256 KiB of each stream: what it has printed so far while it
    /// runs, and all of that once it has exited or been stopped.
    pub fn output(&self) -> RuntimeOutput {
        self.spawned.output()
    }

    /// Wait up to `timeout` for `found` to hold on what `stream` has printed
    /// (its newest 256 KiB, as `output` keeps): that text, or `None` once the
    /// timeout passes or the stream closes without it.
    pub fn wait_for_output(
        &self,
        stream: Stream,
        timeout: Duration,
        found: impl Fn(&str) -> bool,
    ) -> Option<String> {
        self.spawned.wait_for_output(stream, timeout, &found)
    }

    /// Hand each line the container prints to `on_line` instead of
    /// [`log_line`](crate::log_line): the lines kept so far now, stdout first,
    /// then each new one on its stream's reader thread, all inside its span.
    pub fn on_line(&self, on_line: impl Fn(Stream, &str) + Send + Sync + 'static) {
        self.spawned.on_line(Arc::new(on_line));
    }

    /// `docker stop -t <stop grace>`, then wait for the container to be gone.
    /// Nothing to do once it has exited.
    pub fn stop(&mut self) {
        self.spawned.stop();
    }

    /// [`Self::stop`] it over a failure, and say whether it had exited and the
    /// end of its output, which otherwise logs only at debug. Put it on the
    /// failure as context, `err.context(process.stop_with_tail())`, which keeps
    /// `err`'s chain.
    pub fn stop_with_tail(&mut self) -> Stopped {
        self.spawned.stop_with_tail()
    }
}

/// `docker stop -t <grace> <container>`, bounded so a wedged daemon cannot
/// hang a stop. Whether the container is stopped or gone.
pub(crate) fn stop_container(docker: &Path, container: &str, grace: Duration) -> bool {
    run_docker(
        docker,
        &["stop", "-t", &stop_seconds(grace).to_string(), container],
        grace.saturating_add(COMMAND_LIMIT),
    )
}

pub(crate) fn kill_container(docker: &Path, container: &str) -> bool {
    run_docker(docker, &["kill", container], COMMAND_LIMIT)
}

/// `docker stop -t` takes whole seconds; a fraction rounds up.
fn stop_seconds(grace: Duration) -> u64 {
    grace.as_secs() + u64::from(grace.subsec_nanos() > 0)
}

/// Whether the command succeeded or found no container, which `--rm` removed
/// when it exited on its own. Any other failure is logged.
fn run_docker(docker: &Path, args: &[&str], limit: Duration) -> bool {
    let mut command = Command::new(docker);
    command.args(args);
    echo_info(&command);
    let rendered = render(&command);
    let ran = output_within(command, limit).map(|output| {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        (output.status, stderr)
    });
    match ran {
        Ok((status, _)) if status.success() => true,
        Ok((_, stderr)) if stderr.to_ascii_lowercase().contains("no such container") => {
            tracing::debug!(
                command = %rendered,
                stderr = stderr.trim(),
                "no such container: `--rm` already removed it"
            );
            true
        }
        Ok((status, stderr)) => {
            tracing::warn!(
                command = %rendered,
                exit_status = %status,
                stderr = stderr.trim(),
                "docker command failed"
            );
            false
        }
        Err(err) => {
            tracing::warn!(
                command = %rendered,
                error = %Chain(&err),
                "could not run docker command"
            );
            false
        }
    }
}

/// Unix-only: `docker` is a shell script standing in for the CLI.
#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::time::Instant;

    use rstest::rstest;

    use super::*;
    use crate::env::forwards;
    use crate::spawned;

    const LONG: Duration = Duration::from_secs(10);
    const NAME: &str = "pipette-test";

    /// `run` that serves until SIGTERM, as an attached CLI does until its
    /// container stops.
    const SERVE: &str = r#"echo "$@" > "$DIR/run"; echo $$ > "$DIR/cli"
    trap 'echo stopped; exit 143' TERM
    echo "up with $SECRET"; touch "$DIR/ready"
    while :; do sleep 0.05; done"#;
    /// `stop` or `kill` that records itself and stops the CLI.
    const SIGNAL: &str = r#"echo "$@" >> "$DIR/calls"; kill -TERM "$(cat "$DIR/cli")""#;

    /// A `docker` whose `run`, `stop` and `kill` run the given shell, with
    /// `$DIR` the directory it is written to.
    fn fake_docker(dir: &Path, run: &str, stop: &str, kill: &str) -> anyhow::Result<PathBuf> {
        let path = dir.join("docker");
        let script = format!(
            "#!/bin/sh\nDIR='{}'\ncase \"$1\" in\n  run) {run} ;;\n  stop) {stop} ;;\n  kill) {kill} ;;\nesac\n",
            dir.display()
        );
        std::fs::write(&path, script)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        Ok(path)
    }

    fn vllm() -> anyhow::Result<DockerRun> {
        Ok(DockerRun {
            name: NAME.to_owned(),
            image: "vllm/vllm-openai:v1".to_owned(),
            options: vec!["-p".to_owned(), "127.0.0.1:20001:8000".to_owned()],
            args: vec!["--model".to_owned(), "m".to_owned()],
            envs: forwards(&["SECRET=from-the-cli-env"])?,
        })
    }

    fn calls(dir: &Path) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string(dir.join("calls"))?)
    }

    fn up(ready: &Path) -> Result<(), NotReadyError> {
        ready
            .exists()
            .then_some(())
            .ok_or_else(|| NotReadyError::yet("not up yet"))
    }

    #[test]
    fn a_container_runs_attached_and_is_stopped_through_docker() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), SERVE, SIGNAL, SIGNAL)?;
        let ready = dir.path().join("ready");
        let mut service = start_docker_process(&docker, vllm()?, Duration::from_secs(2))?;
        service.wait_ready(LONG, |_| up(&ready))?;
        assert_eq!(service.container(), NAME);
        let run = "run --rm --name pipette-test -p 127.0.0.1:20001:8000 -e SECRET \
                   vllm/vllm-openai:v1 --model m";
        assert_eq!(
            std::fs::read_to_string(dir.path().join("run"))?,
            format!("{run}\n")
        );
        assert_eq!(
            service.command().join(" "),
            format!("{} {run}", docker.display())
        );

        service.stop();
        assert!(!service.is_running());
        assert_eq!(calls(dir.path())?, "stop -t 2 pipette-test\n");
        assert_eq!(
            service.output().runtime_stdout.as_deref(),
            Some("up with from-the-cli-env\nstopped\n")
        );
        Ok(())
    }

    #[rstest]
    #[case::a_value(&["SECRET=s3cret"], &["SECRET"])]
    #[case::an_empty_value(&["GGML_X="], &["GGML_X"])]
    #[case::named_once(&["X=2", "X=4"], &["X"])]
    #[case::an_inherited_value(&["PATH"], &["PATH"])]
    #[case::nothing_to_inherit(&["PIPETTE_DOCKER_RUN_TEST_ABSENT"], &[])]
    fn each_variable_the_forwards_set_goes_in_by_name(
        #[case] entries: &[&str],
        #[case] named: &[&str],
    ) -> anyhow::Result<()> {
        let run = DockerRun {
            envs: forwards(entries)?,
            ..vllm()?
        };
        let args = crate::command::argv(&run_command(Path::new("docker"), &run));
        let by_name: Vec<&str> = args
            .windows(2)
            .filter(|pair| pair[0] == "-e")
            .map(|pair| pair[1].as_str())
            .collect();
        assert_eq!(by_name, named, "{args:?}");
        Ok(())
    }

    #[test]
    fn a_container_that_fails_to_start_is_named_with_the_daemons_error() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let refused =
            "echo 'docker: Error response from daemon: port is already allocated.' >&2; exit 125";
        let docker = fake_docker(dir.path(), refused, SIGNAL, SIGNAL)?;
        let mut service = start_docker_process(&docker, vllm()?, LONG)?;
        let Err(err) = service.wait_ready(LONG, |_| Err(NotReadyError::yet("not up yet"))) else {
            anyhow::bail!("expected the wait to fail");
        };
        let message = format!("{err:#}");
        assert!(
            message.contains("container pipette-test exited with"),
            "{message}"
        );
        assert!(message.contains("port is already allocated"), "{message}");
        Ok(())
    }

    /// A daemon that refuses the stop gets a `docker kill` at once, not after
    /// the wait for the CLI.
    #[test]
    fn a_refused_stop_falls_back_to_docker_kill() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let refuse =
            r#"echo "$@" >> "$DIR/calls"; echo 'Error response from daemon: no' >&2; exit 1"#;
        let docker = fake_docker(dir.path(), SERVE, refuse, SIGNAL)?;
        let ready = dir.path().join("ready");
        let mut service = start_docker_process(&docker, vllm()?, Duration::from_secs(1))?;
        service.wait_ready(LONG, |_| up(&ready))?;
        let started = Instant::now();
        service.stop();
        assert!(started.elapsed() < CLI_EXIT, "waited for the CLI first");
        assert!(!service.is_running());
        assert_eq!(
            calls(dir.path())?,
            "stop -t 1 pipette-test\nkill pipette-test\n"
        );
        Ok(())
    }

    /// A container `--rm` already removed is stopped, so no `docker kill`
    /// follows; a daemon that refuses is not.
    #[rstest]
    #[case::removed(
        "echo 'Error response from daemon: No such container: pipette-test' >&2; exit 1",
        true
    )]
    #[case::refused("echo 'Error response from daemon: no' >&2; exit 1", false)]
    fn a_removed_container_counts_as_stopped(
        #[case] stop: &str,
        #[case] stopped: bool,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), SERVE, stop, SIGNAL)?;
        assert_eq!(
            stop_container(&docker, NAME, Duration::from_secs(1)),
            stopped
        );
        Ok(())
    }

    #[test]
    fn an_interrupt_kills_the_container() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), SERVE, SIGNAL, SIGNAL)?;
        let ready = dir.path().join("ready");
        let mut service = start_docker_process(&docker, vllm()?, LONG)?;
        service.wait_ready(LONG, |_| up(&ready))?;
        let stop = Stop::Docker {
            docker,
            container: NAME.to_owned(),
            grace: LONG,
        };
        let session = crate::session::Session::led_by(service.spawned.pid());
        spawned::on_interrupt(&stop, session)();
        assert!(
            service.wait(LONG)?.is_some(),
            "the CLI outlived the interrupt"
        );
        assert_eq!(calls(dir.path())?, "kill pipette-test\n");
        Ok(())
    }

    #[rstest]
    #[case::none(Duration::ZERO, 0)]
    #[case::whole(Duration::from_secs(30), 30)]
    #[case::a_fraction_rounds_up(Duration::from_millis(1500), 2)]
    #[case::under_a_second(Duration::from_millis(250), 1)]
    fn the_stop_grace_is_whole_seconds(#[case] grace: Duration, #[case] seconds: u64) {
        assert_eq!(stop_seconds(grace), seconds);
    }
}
