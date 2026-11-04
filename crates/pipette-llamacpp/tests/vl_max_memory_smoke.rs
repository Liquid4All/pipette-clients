//! `vl_max_memory` end to end against the fake `llama-server`: spawn,
//! readiness, the one-image workload, the stop, and a real peak off the fake's
//! pid. Runs on macOS (`phys_footprint`) and Linux (`/proc`), the probe the
//! host-built fake runtime picks.
#![cfg(any(target_os = "macos", target_os = "linux"))]

mod common;

use std::path::Path;

use anyhow::Context;
use common::{
    assert_distinct_knobs, assert_server_log, fake_run_request, fake_runtime,
    fake_runtime_install_dir, fake_vision_model, ignore_reps, no_corpus, no_readiness_gate,
    with_distinct_knobs,
};
use rstest::rstest;

use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, BenchmarkId, BenchmarkResultData, EnvForward,
    NonNegativeCount, PositiveCount, PositiveSeconds, RunRequest, RunResponse, Runtime,
    RuntimeFlags, VlMaxMemoryBenchmark,
};

/// A one-image cell on the fake model, with its runtime.
fn request(root: &Path, runtime: Runtime) -> anyhow::Result<RunRequest> {
    let model_path = root.join("fake-model.gguf");
    std::fs::write(&model_path, b"fake-model-bytes")?;
    let mmproj_path = root.join("fake-mmproj.gguf");
    std::fs::write(&mmproj_path, b"fake-mmproj-bytes")?;

    let benchmark = BenchmarkDefinition::VlMaxMemory(VlMaxMemoryBenchmark {
        benchmark_id: BenchmarkId::try_new("vl-max-memory-smoke".to_owned())?,
        parameter_image_width: PositiveCount::try_new(224)?,
        parameter_image_height: PositiveCount::try_new(224)?,
        parameter_text_tokens: NonNegativeCount::try_new(0)?,
        parameter_decode_tokens: None,
        parameter_num_images: NonNegativeCount::try_new(1)?,
    });
    let model = fake_vision_model(&model_path, &mmproj_path)?;
    with_distinct_knobs(fake_run_request(runtime, model, benchmark, None))
}

#[test]
fn vl_max_memory_measures_a_real_peak_and_reports_derived_flags() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let runtime = fake_runtime(&fake_runtime_install_dir(root.path())?)?;
    let req = request(root.path(), runtime)?;

    let outcome = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
    .context("vl_max_memory should complete against the fake server")?;
    assert_distinct_knobs(&req.benchmark, &outcome.command)?;

    let BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes,
        max_gpu_bytes,
        max_npu_bytes,
    } = outcome.result_data
    else {
        anyhow::bail!("expected MaxMemoryUsage result data");
    };
    // Zero would mean the poller never sampled.
    assert!(
        max_host_bytes > 0,
        "the peak should be non-zero for a live process"
    );
    // No GPU pool on either OS: unified memory on macOS, CPU flavors on Linux.
    assert_eq!(max_gpu_bytes, None);
    assert_eq!(max_npu_bytes, None);
    // The observation covers the warm-up server too; on Linux the metric adds
    // the swap the observation carries beside it.
    let observed = outcome
        .memory
        .max_host_bytes
        .unwrap_or(0)
        .saturating_add(outcome.memory.max_swap_bytes.unwrap_or(0));
    assert!(observed >= max_host_bytes, "{:?}", outcome.memory);
    // Only `/proc` samples swap, and it reports `Some(0)` for a resident run.
    assert_eq!(
        outcome.memory.max_swap_bytes.is_some(),
        cfg!(target_os = "linux")
    );
    // The Linux row carries the `/proc` terms behind its swap-aware peak.
    let mut extras = outcome
        .client_extras
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    extras.sort_unstable();
    let expected: &[&str] = if cfg!(target_os = "linux") {
        &[
            "major_faults",
            "max_swap_kib",
            "peak_committed_kib",
            "peak_rss_kib",
        ]
    } else {
        &[]
    };
    assert_eq!(extras, expected);
    assert_server_log(&outcome.runtime_output)?;
    // The derived ctx_size is how the warehouse learns the server's `-c`.
    let flags = outcome
        .runtime_flags
        .context("vl_max_memory should report its resolved runtime_flags")?;
    let wire = serde_json::to_string(&flags)?;
    assert!(
        wire.contains("ctx_size"),
        "runtime_flags should carry the derived ctx_size, got: {wire}"
    );
    Ok(())
}

/// A failure once the server is up heads with its cause, then quotes the
/// server's output around the cell's whole error, the entry's marker inside,
/// so a reading the entry refuses is quoted as well. A Linux GPU build's
/// sampler finds no GPU memory on the fake, so the warm-up fails.
#[cfg(target_os = "linux")]
#[test]
fn a_vl_max_memory_failure_quotes_the_servers_tail() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let Runtime::LlamacppCliStockTools(mut tools) =
        fake_runtime(&fake_runtime_install_dir(root.path())?)?
    else {
        anyhow::bail!("the fake runtime is not llama.cpp's");
    };
    tools.flavor = pipette_plan_types::LlamaCppFlavor::LinuxX64Vulkan;
    let req = request(root.path(), Runtime::LlamacppCliStockTools(tools))?;

    let Err(err) = pipette_llamacpp::run(
        &req,
        &no_corpus(root.path()),
        &no_readiness_gate,
        ignore_reps(),
        None,
    ) else {
        anyhow::bail!("a GPU build read GPU memory off the fake server");
    };
    let message = format!("{err:#}");
    assert!(
        message.starts_with(&err.root_cause().to_string()),
        "{message}"
    );
    assert!(
        message
            .contains("llama-server was still running\nstderr tail:\nfake-llama-server: listening"),
        "{message}"
    );
    assert!(
        message.contains("\nerror: vl-max-memory-smoke warm-up: "),
        "{message}"
    );
    let sampled = err.chain().any(|cause| {
        cause
            .downcast_ref::<pipette_memprobe::PeakError>()
            .is_some()
    });
    assert!(sampled, "{message}");
    Ok(())
}

/// [`request`], with the fake told through the cell's own `envs` to log each
/// process's pid to `pids`, its arguments to `args` and each chat request's
/// image count to `chats` under `root`, plus `envs`.
fn logged_request(root: &Path, envs: &[&str]) -> anyhow::Result<RunRequest> {
    let mut req = request(root, fake_runtime(&fake_runtime_install_dir(root)?)?)?;
    let mut flags = req.runtime_flags_ref()?;
    flags.envs = [
        format!("FAKE_LLAMA_PID_FILE={}", root.join("pids").display()),
        format!("FAKE_LLAMA_ARGS_FILE={}", root.join("args").display()),
        format!(
            "FAKE_LLAMA_CHAT_IMAGES_FILE={}",
            root.join("chats").display()
        ),
    ]
    .into_iter()
    .chain(envs.iter().map(|env| (*env).to_owned()))
    .map(EnvForward::try_new)
    .collect::<Result<_, _>>()?;
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    Ok(req)
}

fn run(root: &Path, req: &RunRequest) -> anyhow::Result<RunResponse> {
    pipette_llamacpp::run(
        req,
        &no_corpus(root),
        &no_readiness_gate,
        ignore_reps(),
        None,
    )
}

/// The lines `root`'s `name` log holds, none if it was never written.
fn logged(root: &Path, name: &str) -> anyhow::Result<Vec<String>> {
    match std::fs::read_to_string(root.join(name)) {
        Ok(text) => Ok(text.lines().map(str::to_owned).collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => Err(err.into()),
    }
}

/// What the first server fills, far over the fake's own footprint.
const FIRST_SERVER_MIB: u64 = 128;

/// A cell first serves its request from a server of its own, launched on the
/// measured one's command and stopped before the measured one starts. The
/// result and the log are the measured server's; the observation covers both.
#[test]
fn a_cell_warms_up_on_a_server_of_its_own() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let fill = format!("FAKE_LLAMA_ALLOCATE_FIRST_MIB={FIRST_SERVER_MIB}");
    let response = run(root.path(), &logged_request(root.path(), &[fill.as_str()])?)?;

    let BenchmarkResultData::MaxMemoryUsage { max_host_bytes, .. } = response.result_data else {
        anyhow::bail!("not a peak-memory result: {:?}", response.result_data);
    };
    let first_server_bytes = FIRST_SERVER_MIB << 20;
    assert!(max_host_bytes < first_server_bytes, "{max_host_bytes}");
    let observed = response
        .memory
        .max_host_bytes
        .context("no host bytes observed")?;
    assert!(observed >= first_server_bytes, "{observed}");

    let pids = logged(root.path(), "pids")?;
    assert_eq!(pids.len(), 2, "{pids:?}");
    let args = logged(root.path(), "args")?;
    assert_eq!(args.len(), 2, "{args:?}");
    assert!(
        args.iter().all(|line| Some(line) == args.first()),
        "{args:?}"
    );
    assert!(
        args.iter().all(|line| line.contains("--mmproj ")),
        "{args:?}"
    );
    // One request per server, each with the cell's one image.
    assert_eq!(logged(root.path(), "chats")?, vec!["1"; 2]);
    let log = response.runtime_output.runtime_stderr.unwrap_or_default();
    let named = pids
        .iter()
        .filter(|pid| log.contains(&format!("pid {pid}\n")))
        .collect::<Vec<_>>();
    assert_eq!(named, pids.last().into_iter().collect::<Vec<_>>(), "{log}");
    Ok(())
}

/// A flavor whose peak cannot be watched runs nothing: neither the
/// calibration's helper, which loads the model, nor a server. An operator's
/// build is refused on every OS.
#[test]
fn a_refused_flavor_runs_no_calibration_and_no_server() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir()?;
    let mut req = logged_request(root.path(), &[])?;
    let calibrated = root.path().join("calibrated");
    let helper = root.path().join("runtime-install").join("llama-mtmd-cli");
    std::fs::write(
        &helper,
        format!("#!/bin/sh\ntouch '{}'\n", calibrated.display()),
    )?;
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))?;
    let Runtime::LlamacppCliStockTools(mut tools) =
        fake_runtime(&root.path().join("runtime-install"))?
    else {
        anyhow::bail!("the fake runtime is not llama.cpp's");
    };
    tools.flavor = pipette_plan_types::LlamaCppFlavor::Custom("my-build".to_owned());
    req.runtime =
        pipette_plan_types::DeclaredBound::already_bound(Runtime::LlamacppCliStockTools(tools));

    let Err(err) = run(root.path(), &req) else {
        anyhow::bail!("an operator's build was measured");
    };
    let message = format!("{err:#}");
    assert!(message.contains("is an operator's build"), "{message}");
    assert!(
        !calibrated.exists(),
        "the calibration ran before the refusal"
    );
    assert_eq!(logged(root.path(), "pids")?, Vec::<String>::new());
    Ok(())
}

/// A warm-up whose request fails fails the cell, saying so and quoting its
/// server's output, before the measured server starts.
#[test]
fn a_failed_warm_up_fails_the_vl_cell() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let req = logged_request(root.path(), &["FAKE_LLAMA_CHAT_400=no image today"])?;
    let Err(err) = run(root.path(), &req) else {
        anyhow::bail!("a refused warm-up request must fail the cell");
    };
    let message = format!("{err:#}");
    assert!(
        message.contains("vl-max-memory-smoke warm-up: "),
        "{message}"
    );
    assert!(message.contains("no image today"), "{message}");
    assert!(
        message.contains("fake-llama-server: listening"),
        "{message}"
    );
    assert_eq!(logged(root.path(), "pids")?.len(), 1);
    Ok(())
}

/// The warm-up server and the measured one each get the cell's deadline, so
/// two workloads that each fit it pass where their sum would not.
#[test]
fn each_server_gets_the_deadline() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut req = logged_request(root.path(), &["FAKE_LLAMA_CHAT_DELAY_MS=4000"])?;
    req.benchmark_flags = Some(BenchmarkFlags::VlMaxMemoryLlamacppCliStockToolsGgufVision {
        http_timeout_seconds: None,
        bench_timeout_seconds: Some(PositiveSeconds::try_new(6)?),
    });
    run(root.path(), &req)?;
    assert_eq!(logged(root.path(), "pids")?.len(), 2);
    Ok(())
}

/// The output length also grows both servers' context; a cell without one
/// decodes one token.
#[test]
fn both_servers_decode_the_cells_output_length() -> anyhow::Result<()> {
    let served = |decode: Option<i32>| -> anyhow::Result<(Vec<Option<u64>>, u32)> {
        let root = tempfile::tempdir()?;
        let bodies = format!(
            "FAKE_LLAMA_BODY_FILE={}",
            root.path().join("bodies").display()
        );
        let mut req = logged_request(root.path(), &[bodies.as_str()])?;
        let BenchmarkDefinition::VlMaxMemory(benchmark) = &mut req.benchmark else {
            anyhow::bail!("not a vl_max_memory cell");
        };
        benchmark.parameter_decode_tokens = decode.map(PositiveCount::try_new).transpose()?;
        let response = run(root.path(), &req)?;
        let max_tokens = logged(root.path(), "bodies")?
            .iter()
            .map(|body| Ok(serde_json::from_str::<serde_json::Value>(body)?["max_tokens"].as_u64()))
            .collect::<anyhow::Result<_>>()?;
        let Some(RuntimeFlags::VlMaxMemoryLlamacppCliStockToolsGgufVision(flags)) =
            response.runtime_flags
        else {
            anyhow::bail!("no vl_max_memory flags: {:?}", response.runtime_flags);
        };
        Ok((max_tokens, flags.ctx_size.context("no derived ctx_size")?))
    };
    let (one, one_ctx) = served(None)?;
    let (four, four_ctx) = served(Some(4))?;
    assert_eq!(one, [Some(1); 2]);
    assert_eq!(four, [Some(4); 2]);
    assert_eq!(four_ctx - one_ctx, 3);
    Ok(())
}

fn has_pair(tokens: &[&str], flag: &str, value: &str) -> bool {
    tokens.windows(2).any(|pair| pair == [flag, value])
}

/// The calibration helper loads where the cell's server does: it gets the
/// cell's GPU layers ([`with_distinct_knobs`] sets 7).
#[test]
fn the_calibration_helper_gets_the_cells_gpu_layers() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir()?;
    let req = logged_request(root.path(), &[])?;
    let helper = root.path().join("runtime-install").join("llama-mtmd-cli");
    std::fs::write(
        &helper,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n\
             echo 'decoding image batch 1/1, n_tokens_batch = 64' >&2\n",
            root.path().join("helper-args").display()
        ),
    )?;
    std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))?;
    run(root.path(), &req)?;
    let args = logged(root.path(), "helper-args")?;
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    assert!(has_pair(&args, "--n-gpu-layers", "7"), "{args:?}");
    Ok(())
}

/// Both servers and the recorded command carry the plan's flags and nothing
/// more: `no_cache` with a raw `--fit off` is the like-for-like plan with
/// `llama-bench`; with the cache on none of it renders.
#[rstest]
#[case::like_for_like_with_llama_bench(
    Some(true),
    &["--fit", "off"],
    &[("--parallel", "1"), ("--ctx-checkpoints", "0"), ("--fit", "off")],
    &[]
)]
#[case::cache_on(
    Some(false),
    &[],
    &[],
    &["--parallel", "--ctx-checkpoints", "--fit", "--no-cache-prompt"]
)]
fn both_servers_run_the_plan_flags(
    #[case] no_cache: Option<bool>,
    #[case] raw: &[&str],
    #[case] pairs: &[(&str, &str)],
    #[case] absent: &[&str],
) -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let mut req = logged_request(root.path(), &[])?;
    let mut flags = req.runtime_flags_ref()?;
    flags.no_cache = no_cache;
    flags.raw = raw.iter().map(|token| (*token).to_owned()).collect();
    req.runtime_flags = Some(RuntimeFlags::try_from(flags)?);
    let response = run(root.path(), &req)?;

    let args = logged(root.path(), "args")?;
    anyhow::ensure!(args.len() == 2, "{args:?}");
    let command = response.command.join(" ");
    args.iter().chain([&command]).try_for_each(|line| {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        anyhow::ensure!(
            pairs
                .iter()
                .all(|(flag, value)| has_pair(&tokens, flag, value))
                && absent.iter().all(|flag| !tokens.contains(flag)),
            "{line}"
        );
        Ok(())
    })
}
