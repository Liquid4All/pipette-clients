//! How a peak-memory cell runs a server: one process per attempt, so the
//! entry's warm-up ([`MemoryMeasurementWarmup::Once`]) fills the Metal shader
//! cache in a process of its own. The macOS sampler reads a process's lifetime
//! maximum, so a warm-up in the measured server would raise its peak.
//!
//! [`MemoryMeasurementWarmup::Once`]: pipette_ops::measurement::MemoryMeasurementWarmup::Once

use std::process::Command;
use std::time::Instant;

use pipette_memprobe::{Sampler, SupportedWatch, metal};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{BenchmarkResultData, MemoryObservation, RunResponse};
use pipette_subprocess::{Process, start_process};

use crate::server;

/// One server from its launch through `measure`, left in `served` for the
/// entry to check its reading and the cell to record or quote it. One an
/// earlier attempt left there stops first; the observation covers both.
/// Returns the server's peak host bytes.
pub(crate) fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    module: &str,
    command: impl FnOnce(u16) -> anyhow::Result<Command>,
    watch: SupportedWatch,
    measure: impl FnOnce(&str) -> anyhow::Result<()>,
) -> anyhow::Result<u64> {
    let earlier = match served.take() {
        Some((mut mlx_server, memory)) => {
            mlx_server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut command = command(port)?;
    let probe = metal::MetalProbeChannel::attach(&mut command)?;
    let mlx_server = start_process(module, command, server::STOP_GRACE)?;
    // Held from here, so the cell quotes it on any failure.
    let (mlx_server, memory) = served.insert((mlx_server, earlier));
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    // Sampled from readiness on: the lifetime maximum still holds the load.
    let sampler = Sampler::start(mlx_server.pid(), watch)?;
    let sent = Instant::now();
    let measured = measure(&url);
    let request = sent.elapsed();
    let (observed, peak) = sampler.stop();
    *memory = memory.merge_max(observed);
    // The request's own failure first, as the likelier root cause.
    measured?;
    let max_host_bytes = peak?.host_bytes;
    // A request that still compiled Metal kernels takes seconds, not a tenth
    // of one, and its peak is off the warm figure.
    tracing::info!(
        pid = mlx_server.pid(),
        ?request,
        max_host_bytes,
        "peak-memory server measured"
    );
    probe.log_peak(max_host_bytes);
    Ok(max_host_bytes)
}

/// The cell's response from the server [`serve`] left in `served`, or its
/// failure quoting that server's output.
pub(crate) fn respond(
    result: anyhow::Result<BenchmarkResultData>,
    served: &mut Option<(Process, MemoryObservation)>,
) -> anyhow::Result<RunResponse> {
    let result_data = result.map_err(|err| match served.as_mut() {
        Some((mlx_server, _)) => err.context(mlx_server.stop_with_tail()),
        None => err,
    })?;
    Ok(served
        .iter_mut()
        .fold(
            RunResponseBuilder::new(result_data),
            |response, (mlx_server, memory)| response.process(mlx_server).memory(*memory),
        )
        .build())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use pipette_memprobe::Watch;
    use pipette_programs::ExternalProgram;

    use super::*;

    /// Answers every GET with `{}` on the given port, after filling `mib` MiB.
    const STUB: &str = r#"
import http.server, sys
held = b"\x01" * (int(sys.argv[2]) << 20)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"#;

    const WARM_UP_MIB: u64 = 128;

    fn stub(mib: u64) -> impl FnOnce(u16) -> anyhow::Result<Command> {
        move |port| {
            let mut command = Command::new(ExternalProgram::PYTHON.locate()?);
            command.args(["-c", STUB, &port.to_string(), &mib.to_string()]);
            Ok(command)
        }
    }

    /// A second attempt stops the first server before its own starts. Its
    /// reading is its own, below the first's, and the observation covers both.
    #[test]
    fn a_second_server_replaces_the_first_and_the_observation_spans_both() -> anyhow::Result<()> {
        let peak = Watch::Peak.supported()?;
        let mut served = None;
        let first = serve(&mut served, "stub", stub(WARM_UP_MIB), peak, |_| Ok(()))?;
        let first_pid = served.as_ref().context("no first server")?.0.pid();
        let second = serve(&mut served, "stub", stub(0), peak, |_| Ok(()))?;

        assert!(first >= WARM_UP_MIB << 20, "{first}");
        assert!(second < WARM_UP_MIB << 20, "{second}");
        let (mlx_server, memory) = served.as_ref().context("no second server")?;
        assert_ne!(mlx_server.pid(), first_pid);
        assert!(
            memory
                .max_host_bytes
                .is_some_and(|bytes| bytes >= WARM_UP_MIB << 20),
            "{memory:?}"
        );
        Ok(())
    }

    /// A failed request fails the attempt and leaves its server in `served`,
    /// so the cell can quote it.
    #[test]
    fn a_failed_request_keeps_its_server_for_the_cell() -> anyhow::Result<()> {
        let peak = Watch::Peak.supported()?;
        let mut served = None;
        let Err(err) = serve(&mut served, "stub", stub(0), peak, |url| {
            anyhow::bail!("no reply from {url}")
        }) else {
            anyhow::bail!("a failed request was measured");
        };
        let message = format!("{err:#}");
        assert!(
            message.contains("no reply from http://127.0.0.1:"),
            "{message}"
        );
        served.as_mut().context("the server was dropped")?.0.stop();
        Ok(())
    }
}
