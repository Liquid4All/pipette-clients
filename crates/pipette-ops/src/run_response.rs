//! The [`RunResponse`] a runtime crate answers a run with, built the same way
//! on every engine ([`RunResponseBuilder`]). It fills the engine's half; the
//! caller's half, `thermal` and `benchmark_flags`, has no setter: `pipette-cli`
//! fills it.

use serde_json::Value;

use pipette_plan_types::{
    BenchmarkResultData, MemoryObservation, RunResponse, RuntimeFlags, RuntimeOutput, Sha256,
    TurnObservation,
};
use pipette_subprocess::Process;

/// A program a run started, as [`RunResponseBuilder::process`] records it: a
/// [`Process`], or an engine's own handle on a server that is not one.
pub trait RecordedProcess {
    /// Stops it, so [`Self::output`] is whole.
    fn stop(&mut self);

    /// The program and its arguments.
    fn command(&self) -> &[String];

    fn output(&self) -> RuntimeOutput;
}

impl RecordedProcess for Process {
    fn stop(&mut self) {
        Process::stop(self);
    }

    fn command(&self) -> &[String] {
        Process::command(self)
    }

    fn output(&self) -> RuntimeOutput {
        Process::output(self)
    }
}

/// A run's [`RunResponse`], filled in as the run ends.
#[must_use]
pub struct RunResponseBuilder {
    response: RunResponse,
    /// Named by [`Self::command`] and [`Self::executable`]: no process
    /// replaces them.
    command: Option<Vec<String>>,
    executable: Option<String>,
}

impl RunResponseBuilder {
    pub fn new(result_data: BenchmarkResultData) -> Self {
        Self {
            response: RunResponse::new(result_data, RuntimeOutput::default()),
            command: None,
            executable: None,
        }
    }

    /// Stops `process` so its output is whole, then records its output, argv
    /// and program; [`Self::command`] and [`Self::executable`] win, set before
    /// or after. A later process replaces it, so a tool run once per rep is
    /// recorded by its last run.
    pub fn process(mut self, process: &mut impl RecordedProcess) -> Self {
        process.stop();
        let command = process.command().to_vec();
        self.response.executable = command.first().cloned();
        self.response.command = command;
        self.response.runtime_output = process.output();
        self
    }

    /// The command to record instead of a process's argv: the request a cell
    /// sent, or a script named without its per-run path.
    pub fn command(mut self, command: Vec<String>) -> Self {
        self.command = Some(command);
        self
    }

    /// The executable to record instead of a process's program.
    pub fn executable(mut self, executable: String) -> Self {
        self.executable = Some(executable);
        self
    }

    /// Merged with what was recorded before: the response holds the most any
    /// observation saw.
    pub fn memory(mut self, memory: MemoryObservation) -> Self {
        self.response.memory = self.response.memory.merge_max(memory);
        self
    }

    /// The flags the run resolved, or `None` for a cell that carries none.
    pub fn runtime_flags(mut self, runtime_flags: impl Into<Option<RuntimeFlags>>) -> Self {
        self.response.runtime_flags = runtime_flags.into();
        self
    }

    /// Added to the extras recorded before.
    pub fn client_extras(mut self, extras: impl IntoIterator<Item = (String, Value)>) -> Self {
        self.response.client_extras.extend(extras);
        self
    }

    pub fn dataset_digest(mut self, dataset_digest: Sha256) -> Self {
        self.response.dataset_digest = Some(dataset_digest);
        self
    }

    /// A traversal's replies: their digest, and one line per turn.
    pub fn traversal(mut self, output_digest: Sha256, turns: Vec<TurnObservation>) -> Self {
        self.response.output_digest = Some(output_digest);
        self.response.turns = turns;
        self
    }

    pub fn build(self) -> RunResponse {
        RunResponse {
            command: self.command.unwrap_or(self.response.command),
            executable: self.executable.or(self.response.executable),
            ..self.response
        }
    }
}

/// Unix-only: the processes are shell one-liners.
#[cfg(all(test, unix))]
mod tests {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use anyhow::Context;

    use pipette_subprocess::{Stream, start_process};

    use super::*;

    fn prefill() -> BenchmarkResultData {
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms: 1.0,
            prefill_time_ms_stddev: None,
        }
    }

    fn ran(script: &str) -> anyhow::Result<Process> {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        let mut process = start_process("sh", command, Duration::ZERO)?;
        process.wait(Duration::from_secs(10))?;
        Ok(process)
    }

    #[test]
    fn a_running_process_is_stopped_and_recorded() -> anyhow::Result<()> {
        let script = "echo printed >&2; exec sleep 30";
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        let mut process = start_process("sh", command, Duration::ZERO)?;
        process
            .wait_for_output(Stream::Stderr, Duration::from_secs(10), |text| {
                text.contains("printed")
            })
            .context("the process never printed")?;

        let started = Instant::now();
        let response = RunResponseBuilder::new(prefill())
            .process(&mut process)
            .build();

        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(!process.is_running());
        assert_eq!(response.executable.as_deref(), Some("sh"));
        assert_eq!(response.command, ["sh", "-c", script]);
        assert_eq!(
            response.runtime_output.runtime_stderr.as_deref(),
            Some("printed\n")
        );
        Ok(())
    }

    /// A tool run once per rep: the last run's output, and the most memory any
    /// run held, which need not be the last's.
    #[test]
    fn runs_keep_the_last_output_and_the_most_memory() -> anyhow::Result<()> {
        let runs = [("echo 'run 1' >&2", 300), ("echo 'run 2' >&2", 200)];
        let response = runs
            .into_iter()
            .try_fold(
                RunResponseBuilder::new(prefill()),
                |response, (script, host_bytes)| {
                    anyhow::Ok(
                        response
                            .process(&mut ran(script)?)
                            .memory(MemoryObservation::host_only(host_bytes)),
                    )
                },
            )?
            .build();

        assert_eq!(
            response.runtime_output.runtime_stderr.as_deref(),
            Some("run 2\n")
        );
        assert_eq!(response.memory, MemoryObservation::host_only(300));
        Ok(())
    }

    /// A server that is not a [`Process`]: what it printed is whole only once
    /// it has been stopped.
    struct Container {
        stopped: bool,
        command: Vec<String>,
    }

    impl RecordedProcess for Container {
        fn stop(&mut self) {
            self.stopped = true;
        }

        fn command(&self) -> &[String] {
            &self.command
        }

        fn output(&self) -> RuntimeOutput {
            let printed = if self.stopped { "whole" } else { "partial" };
            RuntimeOutput {
                runtime_stdout: Some(printed.to_owned()),
                runtime_stderr: None,
            }
        }
    }

    #[test]
    fn a_handle_that_is_not_a_process_is_stopped_before_it_is_read() {
        let mut container = Container {
            stopped: false,
            command: vec!["docker".to_owned(), "run".to_owned()],
        };

        let response = RunResponseBuilder::new(prefill())
            .process(&mut container)
            .build();

        assert_eq!(
            response.runtime_output.runtime_stdout.as_deref(),
            Some("whole")
        );
        assert_eq!(response.command, ["docker", "run"]);
        assert_eq!(response.executable.as_deref(), Some("docker"));
    }

    #[rstest::rstest]
    #[case::after_the_process(false)]
    #[case::before_the_process(true)]
    fn a_command_replaces_the_argv_alone(#[case] named_first: bool) -> anyhow::Result<()> {
        let script = vec!["sh".to_owned(), "server.py".to_owned()];
        let mut process = ran("echo served")?;
        let response = if named_first {
            RunResponseBuilder::new(prefill())
                .command(script.clone())
                .process(&mut process)
        } else {
            RunResponseBuilder::new(prefill())
                .process(&mut process)
                .command(script.clone())
        }
        .build();

        assert_eq!(response.command, script);
        assert_eq!(response.executable.as_deref(), Some("sh"));
        assert_eq!(
            response.runtime_output.runtime_stdout.as_deref(),
            Some("served\n")
        );
        Ok(())
    }

    #[test]
    fn an_executable_replaces_the_program_alone() -> anyhow::Result<()> {
        let response = RunResponseBuilder::new(prefill())
            .executable("/venv/bin/python".to_owned())
            .process(&mut ran("echo served")?)
            .build();

        assert_eq!(response.executable.as_deref(), Some("/venv/bin/python"));
        assert_eq!(response.command, ["sh", "-c", "echo served"]);
        Ok(())
    }
}
