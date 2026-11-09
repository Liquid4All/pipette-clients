use anyhow::Context;

use pipette_completions::TimingRequest;
use pipette_http::HttpClient;
use pipette_memprobe::cgroup::{GpuPeak, GpuSource};
use pipette_ops::measurement::{
    self, MaxMemoryMeasurementReading, MeasurementObserver, MemoryMeasurementWarmup,
};
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{MaxMemoryUsageBenchmark, MemoryObservation, PositiveSeconds};

use crate::{completions, memprobe, server::ServerState};

/// One is enough: vLLM and SGLang reserve the KV cache and decode graphs at
/// startup, and a decode step needs less scratch than the prefill.
pub(crate) const DECODE_TOKENS: u32 = 1;

/// Others' memory in pipette's cgroup past which the figure is less certain.
/// A worker that has run for days holds about 160 MB.
const OTHERS_WARN_BYTES: u64 = 512 * 1024 * 1024;

/// Max-memory cell: the server's peak host and GPU bytes, its container's or,
/// for a uv server, its processes' in pipette's own cgroup, from its launch
/// through one short completion. `probe` started with the server.
pub(super) fn run(
    benchmark: &MaxMemoryUsageBenchmark,
    http_timeout: Option<PositiveSeconds>,
    model: &str,
    state: &ServerState,
    probe: memprobe::Probe,
    observer: MeasurementObserver<'_>,
) -> anyhow::Result<RunResponseBuilder> {
    let prefill_tokens = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs();
    let decode_tokens = DECODE_TOKENS;
    // One request, always: this cell takes one ungated measurement, and
    // `max_memory_usage` carries no batch size.
    let timeout = measurement::http_timeout_for(http_timeout, 1);
    let http = HttpClient::loopback("pipette", timeout)?;
    let base_url = state.base_url();
    let prompt_text = completions::build_prompt_text(&http, &base_url, model, prefill_tokens)?;
    let request = TimingRequest::text(Some(model), &prompt_text, decode_tokens);
    let mut client_extras = serde_json::Map::new();
    let mut memory = MemoryObservation::default();
    let mut shared = false;
    // Cold: the probe started with the server, outside the attempt, so it
    // stops once.
    let mut probe = Some(probe);
    let result_data = measurement::max_memory(
        &benchmark.benchmark_id,
        observer,
        MemoryMeasurementWarmup::Cold,
        (),
        |()| {
            let probe = probe.take().context("the server's probe already stopped")?;
            let generation =
                pipette_completions::reply(&http, &completions::url(&base_url), &request);
            // Stopped while the server runs; a request error wins over a probe error.
            let peak = probe.stop().map_err(anyhow::Error::from);
            let usage = generation?.counts_at_limit()?;
            measurement::expect_tokens("prompt_tokens", usage.prompt_tokens, prefill_tokens)?;
            measurement::expect_tokens(
                "completion_tokens",
                usage.completion_tokens,
                decode_tokens,
            )?;
            let peak = peak?;

            tracing::info!(
                prefill_target_tokens = prefill_tokens,
                prompt_tokens = usage.prompt_tokens,
                completion_tokens = usage.completion_tokens,
                samples = peak.gpu.map_or(0, |gpu| gpu.samples),
                host_samples = peak.host_samples,
                "peak memory probe"
            );
            let memory_stat: serde_json::Map<String, serde_json::Value> = peak
                .memory_stat
                .into_iter()
                .map(|(key, bytes)| (key.to_owned(), bytes.into()))
                .collect();
            client_extras.extend([
                ("prompt_tokens".to_owned(), usage.prompt_tokens.into()),
                (
                    "completion_tokens".to_owned(),
                    usage.completion_tokens.into(),
                ),
                (
                    "samples".to_owned(),
                    peak.gpu.map_or(0, |gpu| gpu.samples).into(),
                ),
                ("host_samples".to_owned(), peak.host_samples.into()),
                // The cgroup at the stop (pipette's, for a uv server), page cache
                // included (`file` here; `memory.peak` below, the host figure of
                // earlier builds, counts it too), so a row compares with theirs.
                ("memory_stat".to_owned(), memory_stat.into()),
            ]);
            // A uv server's cgroup is pipette's: what came off `memory_stat`'s
            // figure for pipette and the others in it.
            if let Some(cgroup) = peak.shared {
                shared = true;
                if cgroup.others_max_bytes >= OTHERS_WARN_BYTES {
                    tracing::warn!(
                        others_max_bytes = cgroup.others_max_bytes,
                        "other processes in pipette's cgroup held much memory of their own, \
                         taken off the server's figure; run memory cells where pipette's cgroup \
                         holds little else"
                    );
                }
                client_extras.extend([
                    ("others_bytes".to_owned(), cgroup.others_bytes.into()),
                    (
                        "others_max_bytes".to_owned(),
                        cgroup.others_max_bytes.into(),
                    ),
                    ("others_listed".to_owned(), cgroup.listed.into()),
                    (
                        "host_bytes_at_launch".to_owned(),
                        cgroup.host_bytes_at_launch.into(),
                    ),
                ]);
            } else if let Some(bytes) = peak.memory_peak {
                // Its own cgroup's watermark; pipette's covers the whole
                // session or job.
                client_extras.insert("cgroup_memory_peak_bytes".to_owned(), bytes.into());
            }
            // What came off an AMD card's figure, and whether the open saw
            // the card stop handing back a stopped server's memory.
            if let Some(GpuPeak {
                source:
                    GpuSource::Cards {
                        floor_bytes,
                        settled,
                    },
                ..
            }) = peak.gpu
            {
                client_extras.insert("gpu_floor_bytes".to_owned(), floor_bytes.into());
                client_extras.insert("gpu_floor_settled".to_owned(), settled.into());
            }
            memory = peak.memory;
            Ok(MaxMemoryMeasurementReading {
                max_host_bytes: peak.host_bytes,
                max_gpu_bytes: peak.gpu.map(|gpu| gpu.bytes),
                max_npu_bytes: None,
            })
        },
    )?;

    Ok(RunResponseBuilder::new(result_data)
        .executable(state.executable.display().to_string())
        .command(vec![
            format!("POST {}/v1/completions", state.base_url()),
            format!("model={model} prefill_target_tokens={prefill_tokens}"),
            format!(
                "memprobe: {} every 10 ms + NVML on an NVIDIA GPU, or the job's cards' \
                 mem_info_vram_used less their least on an AMD GPU",
                if shared {
                    "pipette's cgroup memory.stat anon+shmem+kernel less its other processes' \
                     RssAnon+RssShmem+VmPTE"
                } else {
                    "cgroup memory.stat anon+shmem+kernel"
                }
            ),
        ])
        .memory(memory)
        .client_extras(client_extras))
}
