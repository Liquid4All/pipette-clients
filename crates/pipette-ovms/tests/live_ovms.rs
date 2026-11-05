//! Live run of the four cells, then the three timing cells batched, against a
//! real `ovms` release, through the engine's own entry point, with nothing of
//! the CLI around it but what the CLI would bind.
//!
//! **`#[ignore]`.** It needs an unpacked release and an IR directory, the
//! store entries `pipette runtimes pull` / `models pull` materialize:
//!
//! - `PIPETTE_LIVE_OVMS_DIR`: the install directory holding `ovms/`
//!   (`…/runtimes/<key>/blobs`)
//! - `PIPETTE_LIVE_OVMS_MODEL_DIR`: the `Model::Openvino` directory
//! - `PIPETTE_LIVE_OVMS_DEVICE`: `cpu`, `gpu` (default) or `npu`
//! - `PIPETTE_LIVE_OVMS_CACHE`: the compile cache (default: a fresh temp dir)
//! - `PIPETTE_LIVE_OVMS_BATCH`: the batched cells' batch size (default 4; `1`
//!   skips them, as does the NPU, which does not batch)
//!
//! On Windows it also checks that no `ovms.exe` outlives the cells, so run it
//! with no other OVMS on the host.
//!
//! ```text
//! cargo test -p pipette-ovms --test live_ovms -- --ignored --nocapture
//! ```

use anyhow::Context;

use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{
    AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, BenchmarkResultData, DeclaredBound,
    DecodeThroughputBenchmark, EndToEndLatencyBenchmark, MaxMemoryUsageBenchmark, Model,
    ModelSource, ModelType, NonNegativeCount, Openvino, OpenvinoDevice, PositiveCount,
    PrefillThroughputBenchmark, RunRequest, RuntimeFlagRef, RuntimeFlags, RuntimeType,
};

const PREFILL: i32 = 1024;
const DECODE: i32 = 128;

fn env(name: &str) -> anyhow::Result<String> {
    std::env::var(name).with_context(|| format!("set {name}; see the file's doc comment"))
}

fn request(
    benchmark: BenchmarkDefinition,
    install: &str,
    model: &str,
    device: &OpenvinoDevice,
) -> anyhow::Result<RunRequest> {
    let mut flags = RuntimeFlagRef::new(
        benchmark.benchmark_type(),
        RuntimeType::Ovms,
        ModelType::Openvino,
    );
    flags.device = Some(device.clone());
    Ok(RunRequest {
        runtime: DeclaredBound::already_bound(serde_json::from_value(serde_json::json!({
            "type": "ovms", "source": "absolute_dir", "dir": install,
            "flavor": if cfg!(windows) { "windows-x64" } else { "ubuntu24-x64" }
        }))?),
        model: DeclaredBound::already_bound(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model.to_owned())?,
            },
        })),
        runtime_flags: Some(RuntimeFlags::try_from(flags)?),
        model_flags: None,
        benchmark_flags: None,
        benchmark,
    })
}

fn timing_cells(batch: u32) -> anyhow::Result<[BenchmarkDefinition; 3]> {
    let suffix = if batch == 1 {
        String::new()
    } else {
        format!("_bs{batch}")
    };
    let batch_size = BatchSize::try_new(batch)?;
    Ok([
        BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("prefill_throughput_{PREFILL}{suffix}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL)?,
            parameter_batch_size: batch_size,
        }),
        BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new(format!(
                "decode_throughput_{PREFILL}_{DECODE}{suffix}"
            ))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL)?,
            parameter_decode_tokens: PositiveCount::try_new(DECODE)?,
            parameter_batch_size: batch_size,
        }),
        BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
            benchmark_id: BenchmarkId::try_new(format!(
                "end_to_end_latency_{PREFILL}_{DECODE}{suffix}"
            ))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL)?,
            parameter_decode_tokens: PositiveCount::try_new(DECODE)?,
            parameter_batch_size: batch_size,
        }),
    ])
}

fn cells(batch: u32) -> anyhow::Result<Vec<BenchmarkDefinition>> {
    let mut cells = timing_cells(1)?.to_vec();
    cells.push(BenchmarkDefinition::MaxMemoryUsage(
        MaxMemoryUsageBenchmark {
            benchmark_id: BenchmarkId::try_new(format!("max_memory_usage_{PREFILL}"))?,
            parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL)?,
        },
    ));
    if batch > 1 {
        cells.extend(timing_cells(batch)?);
    }
    Ok(cells)
}

fn headline(data: &BenchmarkResultData) -> anyhow::Result<String> {
    Ok(match data {
        BenchmarkResultData::PrefillThroughput {
            prefill_time_ms, ..
        } => format!(
            "ttft {prefill_time_ms:.1} ms ({:.0} tok/s)",
            f64::from(PREFILL) / prefill_time_ms * 1000.0
        ),
        BenchmarkResultData::DecodeThroughput { decode_time_ms, .. } => format!(
            "decode {decode_time_ms:.1} ms ({:.1} tok/s)",
            f64::from(DECODE) / decode_time_ms * 1000.0
        ),
        BenchmarkResultData::EndToEndLatency { total_time_ms, .. } => {
            format!("total {total_time_ms:.1} ms")
        }
        BenchmarkResultData::MaxMemoryUsage { max_host_bytes, .. } => {
            format!("peak {:.1} MiB", *max_host_bytes as f64 / 1_048_576.0)
        }
        other @ (BenchmarkResultData::Eval { .. }
        | BenchmarkResultData::VlThroughput { .. }
        | BenchmarkResultData::AsrThroughput { .. }
        | BenchmarkResultData::VlSpecDecode { .. }) => {
            anyhow::bail!("unexpected result {other:?}")
        }
    })
}

#[test]
#[ignore = "needs an unpacked ovms release and an OpenVINO IR directory"]
fn the_four_cells_run_against_a_live_server() -> anyhow::Result<()> {
    let install = env("PIPETTE_LIVE_OVMS_DIR")?;
    let model = env("PIPETTE_LIVE_OVMS_MODEL_DIR")?;
    let device = OpenvinoDevice::parse(
        &std::env::var("PIPETTE_LIVE_OVMS_DEVICE").unwrap_or_else(|_| "gpu".to_owned()),
    );
    let scratch = tempfile::tempdir()?;
    let cache = std::env::var("PIPETTE_LIVE_OVMS_CACHE")
        .map_or_else(|_| scratch.path().to_path_buf(), std::path::PathBuf::from);
    let observer = MeasurementObserver {
        on_before_rep_start: &|| Ok(()),
        on_after_rep_end: &|| Ok(()),
    };
    let batch = if device == OpenvinoDevice::Npu {
        1
    } else {
        std::env::var("PIPETTE_LIVE_OVMS_BATCH").map_or(Ok(4), |b| b.parse())?
    };

    cells(batch)?.into_iter().try_for_each(|benchmark| {
        let id = benchmark.benchmark_id().clone();
        let batched = benchmark
            .batch_size()
            .is_some_and(|b| b != BatchSize::one());
        let req = request(benchmark, &install, &model, &device)?;
        let response = pipette_ovms::run(&req, &|| Ok(()), observer, &cache, None)
            .with_context(|| format!("{id}"))?;
        println!(
            "{id} on {device}: {}, observed {:?}",
            headline(&response.result_data)?,
            response.memory
        );
        anyhow::ensure!(
            response.command.iter().any(|arg| arg == "--max_num_seqs") == batched,
            "{:?}",
            response.command
        );
        let flags = RuntimeFlagRef::from(response.runtime_flags.context("no resolved flags")?);
        anyhow::ensure!(flags.device == Some(device.clone()), "{flags:?}");
        anyhow::ensure!(flags.prefix_caching == Some(false), "{flags:?}");
        anyhow::ensure!(
            response
                .command
                .iter()
                .any(|arg| arg == "--enable_prefix_caching"),
            "{:?}",
            response.command
        );
        // Every cell watches its server, the timing cells too.
        anyhow::ensure!(response.memory.max_host_bytes.is_some(), "{id}");
        Ok(())
    })?;
    no_orphan_server()
}

#[cfg(windows)]
fn no_orphan_server() -> anyhow::Result<()> {
    let out = std::process::Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq ovms.exe", "/NH"])
        .output()?;
    let listing = String::from_utf8_lossy(&out.stdout);
    anyhow::ensure!(
        !listing.to_ascii_lowercase().contains("ovms.exe"),
        "an ovms.exe outlived its cell:\n{listing}"
    );
    Ok(())
}

#[cfg(not(windows))]
fn no_orphan_server() -> anyhow::Result<()> {
    Ok(())
}
