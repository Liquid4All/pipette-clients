//! Peak memory for one OpenVINO cell.
//!
//! Single-shot, not a measured series: peak memory is a high-water mark, so
//! repeating it would report the max of five identical maxima. This mirrors the
//! other backends, whose memory paths also sit outside
//! [`pipette_ops::measurement`].
//!
//! The reading comes from the driver process itself rather than a poller on
//! this side. That process is the one holding the compiled pipeline and the
//! weights, so its own peak RSS is the measurement — no sampling interval to
//! miss the peak through, and no cross-process race. On Windows that is
//! `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize` via PSAPI; on Linux
//! `ru_maxrss`.
//!
//! `max_gpu_bytes` and `max_npu_bytes` are left `None`. OpenVINO exposes no
//! per-process accelerator counter through GenAI, and on the Lunar Lake target
//! the iGPU and NPU share the same on-package memory as the CPU — so the host
//! figure already covers the weights wherever they were placed. Reporting a
//! fabricated device split would be worse than reporting none.

use pipette_ops::measurement;
use pipette_ops::prompt_seed::PROMPT_SEED_TEXT;
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::{RunRequest, RunResponse};

use super::driver::{DriverRequest, Mode};
use super::Cell;

pub(super) fn run(
    req: &RunRequest,
    compile_cache: &std::path::Path,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_max_memory_usage()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let cell = Cell::bind(req, compile_cache)?;
    let model_dir = cell.model_dir_str()?.to_owned();

    let (result, output) = cell.script.invoke(
        &cell.python,
        &DriverRequest {
            model_dir: &model_dir,
            device: crate::runtimes::device_property(&cell.device),
            // Prefill mode generates exactly one token, which is all the
            // memory pass needs: the peak is set by the weights plus the
            // prefill KV cache, both resident by the time that token lands.
            mode: Mode::Prefill,
            prefill_tokens,
            decode_tokens: 1,
            // No warmup: a second pass cannot raise a high-water mark that the
            // first already set, and on NPU it is pure added device pressure.
            warmup: None,
            prompt: None,
            properties: cell.properties(),
            prompt_seed: PROMPT_SEED_TEXT,
        },
    )?;
    measurement::expect_tokens("input_tokens", result.input_tokens, prefill_tokens)?;

    let max_host_bytes = result.peak_host_bytes.ok_or_else(|| {
        anyhow::anyhow!(
            "the driver reported no peak memory counter on this host; \
             max_host_bytes would be a guess"
        )
    })?;
    if max_host_bytes == 0 {
        anyhow::bail!("the driver reported a zero peak working set, which cannot be right");
    }

    Ok(cell.respond(
        BenchmarkResultData::MaxMemoryUsage {
            max_host_bytes,
            max_gpu_bytes: None,
            max_npu_bytes: None,
        },
        output.stdout,
        output.stderr,
    ))
}
