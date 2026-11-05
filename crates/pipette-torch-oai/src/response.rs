//! What the timing cells return, and the server as the dispatch records it
//! beside the flags.

use pipette_ops::run_response::{RecordedProcess, RunResponseBuilder};
use pipette_plan_types::{BatchSize, BenchmarkResultData, RuntimeOutput};

use crate::server::{Server, ServerState};

/// The container or the uv server, stopped and read the way a process is. Its
/// command line is never the row's: every cell names the request it sent.
impl RecordedProcess for Server {
    fn stop(&mut self) {
        Server::stop(self);
    }

    fn command(&self) -> &[String] {
        Server::command(self)
    }

    fn output(&self) -> RuntimeOutput {
        Server::output(self)
    }
}

/// What a timing cell records: the server's launcher and the request it timed.
pub(crate) fn timing_response(
    result_data: BenchmarkResultData,
    state: &ServerState,
    request: String,
    batch_size: BatchSize,
) -> RunResponseBuilder {
    let mut command = vec![format!("POST {}/v1/completions", state.base_url()), request];
    // A batch names its size; a single request keeps the two-line command.
    if batch_size.as_u32() > 1 {
        command.push(format!("batch_size={batch_size}"));
    }
    RunResponseBuilder::new(result_data)
        .executable(state.executable.display().to_string())
        .command(command)
}
