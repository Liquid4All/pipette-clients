//! Benchmark execution against a docker- or uv-hosted OpenAI-compatible server.
//!
//! Public entry [`run`] builds launch config from a prepared [`RunRequest`],
//! brings the server up for the duration of the cell, tears it down after,
//! and returns [`RunResponse`]. Kind modules talk to the server over
//! HTTP — they don't know which engine is hosting it.
//!
//! CLI owns prepare/record. Docker/uv orphan labels are a torch-host detail:
//! derived from the bound model store path (`…/models/<key>/…` → workspace
//! root), not a field on [`RunRequest`].

mod cell_flags;
mod end_to_end_latency;
mod eval;
mod launch;
mod max_memory_usage;

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::Context;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::run::RunResponse;
use pipette_plan_types::RuntimeFlags;
use pipette_subprocess::cleanup::TeardownToken;

use crate::{
    cleanup,
    models::require_torch_model_dir,
    server::{self, uv, DockerLaunchEnv, LaunchSpec, ServerState},
};

const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 120;

/// Top-level torch-oai dispatch: launch server, run cell, tear down.
///
/// Kind bodies and cell knobs come from [`RunRequest`]; `eval_completions`
/// only for eval resume. `readiness_gate` is the caller's readiness wait.
/// Matches mlx/llamacpp: `run(req, evals, readiness_gate)`.
pub fn run(
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let model_dir = require_torch_model_dir(req)?;
    let workspace_label = workspace_label_from_store_path(&model_dir)?;

    // Resolved once here and threaded into the launch build, so both the reap
    // and `docker run` see the same binary. Only a docker-bound run needs it;
    // elsewhere it stays best-effort because the reap is workspace-scoped, not
    // runtime-scoped — a uv run still clears containers an earlier crashed
    // docker run left behind.
    let docker_bin = match server::resolve_docker() {
        Ok(docker) => Some(docker),
        Err(err) if is_docker_runtime(&req.runtime.bound) => return Err(err),
        Err(err) => {
            log::debug!("skipping docker orphan reap: {err:#}");
            None
        }
    };
    // The uv server spawns from the venv's own shims, so this only asserts the
    // host can still manage that venv — but assert it before the server comes
    // up rather than discovering it mid-benchmark.
    if is_uv_runtime(&req.runtime.bound) {
        pipette_venv::resolve_uv(None)?;
    }
    if let Some(docker) = &docker_bin {
        cleanup::reap_workspace_orphans(docker, &workspace_label)?;
    }
    uv::reap_workspace_orphans(&workspace_label)?;

    let launch = launch::build_launch(req, &model_dir, workspace_label, docker_bin)?;
    let mut outcome = run_with_server(&launch, req, eval_completions, readiness_gate, observer)?;
    outcome.runtime_flags = Some(launch.runtime_flags.clone());
    Ok(outcome)
}

fn is_docker_runtime(runtime: &pipette_plan_types::Runtime) -> bool {
    matches!(
        runtime,
        pipette_plan_types::Runtime::DockerVllm(_) | pipette_plan_types::Runtime::DockerSglang(_)
    )
}

fn is_uv_runtime(runtime: &pipette_plan_types::Runtime) -> bool {
    matches!(
        runtime,
        pipette_plan_types::Runtime::UvVllm(_) | pipette_plan_types::Runtime::UvSglang(_)
    )
}

/// Orphan-reap cookie: absolute workspace root as implied by the store layout
/// (`{workspace}/models/<key>/…` or `{workspace}/runtimes/<key>/…`).
/// Torch-local only — not plan identity and not a [`RunRequest`] field.
fn workspace_label_from_store_path(path_under_store: &Path) -> anyhow::Result<String> {
    let store = path_under_store
        .ancestors()
        .find(|a| {
            matches!(
                a.file_name().and_then(|s| s.to_str()),
                Some("models" | "runtimes")
            )
        })
        .with_context(|| {
            format!(
                "path {} is not under a workspace `models/` or `runtimes/` store directory",
                path_under_store.display()
            )
        })?;
    let ws = store
        .parent()
        .with_context(|| format!("store dir has no parent: {}", store.display()))?;
    Ok(ws.display().to_string())
}

/// Engine-tagged launch inputs. Paired with [`Launch::launch_spec`] by
/// [`launch::build_launch`].
#[derive(Debug, Clone)]
pub(super) enum LaunchEnv {
    Docker {
        /// Path to the docker binary used for `docker run` / `stop` /
        /// `logs -f`. Resolved once at the top of `benchmarks run` so
        /// the launch and teardown sites see the same binary.
        docker_bin: PathBuf,
        env: DockerLaunchEnv,
    },
    Uv {
        /// Bound UV venv root (`AbsolutePreinstalled` after prepare). The
        /// server is spawned from this venv's shims, so no uv binary is needed.
        venv_dir: PathBuf,
        env: uv::UvLaunchEnv,
    },
}

/// Private launch handle for server start/stop + API model id.
///
/// Per-cell settings (timeouts, doomloop, enable_thinking, readiness) stay on
/// [`RunRequest`]; kind runners read them from `req`.
#[derive(Debug)]
pub(super) struct Launch {
    pub(super) launch_env: LaunchEnv,
    pub(super) launch_spec: LaunchSpec,
    /// OpenAI `model` field / engine `--model` arg (container mount or host path).
    pub(super) model: String,
    pub(super) ready_timeout_secs: u64,
    /// The cell's flags as launched — plan entry plus the client's derived
    /// values — recorded on the outcome for provenance.
    pub(super) runtime_flags: RuntimeFlags,
}

/// Start the server (docker or uv), run the benchmark over HTTP, then tear down.
fn run_with_server(
    launch: &Launch,
    req: &RunRequest,
    eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let (state, teardown_token) = start_server(&launch.launch_env, &launch.launch_spec)
        .context("failed to start server for benchmark")?;

    // Guard first, before anything that could fail or panic: it tears the
    // server down on unwind, not just on Err — the previous closure-then-stop
    // pattern leaked containers when dispatch panicked. The teardown token
    // moves into the guard so a clean stop can deregister it from the
    // signal-handler registry, per the discipline on `cleanup::register`.
    let mut guard = ServerGuard {
        launch_env: &launch.launch_env,
        state,
        log_tail: None,
        teardown_token: Some(teardown_token),
    };

    // Stream the container's stdout/stderr to the parent on the docker path.
    // The uv engine already line-tails the child's pipes from inside its
    // launch site (see `uv::spawn_log_tail`), so the parent has
    // no extra stream to attach.
    if let LaunchEnv::Docker { docker_bin, .. } = &launch.launch_env {
        let container_id = guard
            .state
            .container_id()
            .context("docker launch did not produce docker server state")?;
        match server::tail_logs(docker_bin, container_id.as_str()) {
            Ok(child) => guard.log_tail = Some(child),
            Err(err) => log::warn!("failed to start `docker logs -f` tail: {err:#}"),
        }
    }

    let ready_url = guard.state.ready_url();

    let ready_start = Instant::now();
    server::wait_ready(&guard.state, Duration::from_secs(launch.ready_timeout_secs))
        .context("server failed to become ready before benchmark")?;
    log::info!(
        "[server] ready at {ready_url} after {:.1}s",
        ready_start.elapsed().as_secs_f64()
    );
    log::info!("[bench] running {}", req.benchmark.benchmark_id());
    let docker_bin = match &launch.launch_env {
        LaunchEnv::Docker { docker_bin, .. } => Some(docker_bin.as_path()),
        LaunchEnv::Uv { .. } => None,
    };
    dispatch(
        req,
        &launch.model,
        docker_bin,
        &guard.state,
        eval_completions,
        readiness_gate,
        observer,
    )
}

/// Routes to docker or uv from `LaunchEnv`; `LaunchSpec` kind must agree
/// (`DockerVllm`/`UvVllm`/...) — `build_launch` guarantees this.
fn start_server(
    launch_env: &LaunchEnv,
    spec: &LaunchSpec,
) -> anyhow::Result<(ServerState, TeardownToken)> {
    match launch_env {
        LaunchEnv::Docker { docker_bin, env } => {
            server::start(docker_bin, env.clone(), spec.clone())
        }
        LaunchEnv::Uv { venv_dir, env } => uv::launch(venv_dir, spec, env),
    }
}

struct ServerGuard<'a> {
    launch_env: &'a LaunchEnv,
    state: ServerState,
    log_tail: Option<std::process::Child>,
    /// Token from `cleanup::register_*` taken at launch time.
    /// `Option` to allow `take()`-and-deregister in Drop after a clean
    /// stop. If the stop fails the token stays in the registry so the
    /// signal handler can retry on ^C.
    teardown_token: Option<TeardownToken>,
}

impl Drop for ServerGuard<'_> {
    fn drop(&mut self) {
        // Stop the docker log tail first so its output doesn't interleave
        // with the teardown messages below. `docker logs -f` would exit on
        // its own once the container stops, but explicit kill avoids the
        // race and keeps the child from being orphaned if our process aborts.
        // The uv engine has no separate log-tail child (the JoinHandle is
        // drained by `uv::stop`).
        if let Some(mut child) = self.log_tail.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let stop_result = match self.launch_env {
            LaunchEnv::Docker { docker_bin, .. } => {
                let label = match self.state.container_id() {
                    Some(id) => format!("container {id}"),
                    None => "docker container".to_string(),
                };
                let res = server::stop(docker_bin, &self.state, 30);
                res.map(|()| label.clone()).map_err(|err| (label, err))
            }
            LaunchEnv::Uv { .. } => {
                let pid = self.state.pid().unwrap_or(0);
                let res = uv::stop(&mut self.state, 30);
                res.map(|()| format!("uv server pid={pid}"))
                    .map_err(|err| (format!("uv server pid={pid}"), err))
            }
        };
        match stop_result {
            Ok(label) => {
                if let Some(token) = self.teardown_token.take() {
                    pipette_subprocess::cleanup::deregister(token);
                }
                log::info!("torn down benchmark {label}");
            }
            Err((label, err)) => log::warn!("failed to stop benchmark {label} cleanly: {err:#}"),
        }
    }
}

/// Kind dispatch after the server is up. Takes only what benches need from
/// the host (API model id + optional docker binary) — not the full Launch bag.
fn dispatch(
    req: &RunRequest,
    model: &str,
    docker_bin: Option<&Path>,
    state: &server::ServerState,
    eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    match req.benchmark.benchmark_type() {
        pipette_plan_types::BenchmarkType::EndToEndLatency => {
            end_to_end_latency::run(req, model, state, readiness_gate, observer)
        }
        pipette_plan_types::BenchmarkType::MaxMemoryUsage => {
            let docker_bin = docker_bin.ok_or_else(|| {
                anyhow::anyhow!(
                    "max_memory_usage benchmark is not yet supported on uv runtimes; \
                     use a docker runtime, or run a different benchmark type"
                )
            })?;
            max_memory_usage::run(req, model, docker_bin, state)
        }
        pipette_plan_types::BenchmarkType::Eval => eval::run(req, model, state, eval_completions),
        kind @ (pipette_plan_types::BenchmarkType::PrefillThroughput
        | pipette_plan_types::BenchmarkType::DecodeThroughput
        | pipette_plan_types::BenchmarkType::VlThroughput) => anyhow::bail!(
            "benchmark type {kind:?} is not yet implemented in pipette-torch-oai; \
             currently supported: end_to_end_latency, max_memory_usage, eval"
        ),
    }
}

/// HTTP client timeout from cell `benchmark_flags`, with a 120s default.
pub(crate) fn http_timeout(req: &RunRequest) -> Duration {
    Duration::from_secs(
        req.benchmark_flags
            .as_ref()
            .and_then(|f| f.http_timeout())
            .unwrap_or(DEFAULT_HTTP_TIMEOUT_SECS)
            .max(1),
    )
}

/// Build a prompt string that the server will tokenize to exactly
/// `target_tokens`. Sending text keeps the server's tokenize step
/// inside the request measured by latency and memory benchmarks.
pub(crate) fn build_prompt_text(
    base_url: &str,
    model: &str,
    target_tokens: u32,
    timeout: Duration,
) -> anyhow::Result<String> {
    pipette_ops::prompt_seed::build_prompt_text(target_tokens, |text| {
        let response = crate::openai::tokenize(
            base_url,
            &crate::openai::TokenizeRequest {
                model,
                prompt: text,
                // Match /v1/completions string-prompt accounting. Some
                // tokenizers add BOS here, and the e2e validator checks the
                // server-reported prompt_tokens from the timed request.
                add_special_tokens: Some(true),
            },
            timeout,
        )
        .context("failed to /tokenize while building prompt")?;
        Ok(response.tokens.len())
    })
}

pub(crate) fn validate_completion_usage(
    usage: Option<&crate::openai::Usage>,
    expected_prompt_tokens: u32,
    expected_completion_tokens: u32,
) -> anyhow::Result<&crate::openai::Usage> {
    let Some(usage) = usage else {
        anyhow::bail!("/v1/completions response missing usage block");
    };
    if usage.prompt_tokens != expected_prompt_tokens {
        anyhow::bail!(
            "/v1/completions returned prompt_tokens {}, expected {}",
            usage.prompt_tokens,
            expected_prompt_tokens
        );
    }
    if usage.completion_tokens != expected_completion_tokens {
        anyhow::bail!(
            "/v1/completions returned completion_tokens {}, expected {}",
            usage.completion_tokens,
            expected_completion_tokens
        );
    }
    Ok(usage)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn usage(prompt_tokens: u32, completion_tokens: u32) -> crate::openai::Usage {
        crate::openai::Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    // `expected`: `Ok(total_tokens)` on a match, `Err(substring)` on rejection.
    #[rstest]
    #[case::matching(Some(usage(7, 3)), Ok(10))]
    #[case::missing_usage(None, Err("missing usage"))]
    #[case::prompt_mismatch(Some(usage(6, 3)), Err("prompt_tokens"))]
    #[case::completion_mismatch(Some(usage(7, 2)), Err("completion_tokens"))]
    fn validate_completion_usage_cases(
        #[case] usage: Option<crate::openai::Usage>,
        #[case] expected: Result<u32, &'static str>,
    ) -> anyhow::Result<()> {
        let result = validate_completion_usage(usage.as_ref(), 7, 3);
        match expected {
            Ok(total) => assert_eq!(result?.total_tokens, total),
            Err(substr) => {
                let err = result.err().context("expected a rejection error")?;
                assert!(err.to_string().contains(substr), "got {err}");
            }
        }
        Ok(())
    }

    // `expected`: `Ok(workspace root)`, or `Err(substring)` when the path sits
    // outside a store.
    #[rstest]
    #[case::models("/tmp/ws/.pipette/models/abc/blobs", Ok("/tmp/ws/.pipette"))]
    #[case::runtimes("/tmp/ws/.pipette/runtimes/key/blobs/venv", Ok("/tmp/ws/.pipette"))]
    // Innermost store component wins, so a workspace that itself lives under a
    // dir named `models` still resolves to the real workspace root.
    #[case::nested(
        "/data/models/proj/.pipette/models/abc",
        Ok("/data/models/proj/.pipette")
    )]
    #[case::outside_any_store("/tmp/other/abc", Err("not under a workspace"))]
    fn workspace_label_from_store_path_cases(
        #[case] path: &str,
        #[case] expected: Result<&str, &str>,
    ) -> anyhow::Result<()> {
        let result = workspace_label_from_store_path(Path::new(path));
        match expected {
            Ok(label) => assert_eq!(result?, label),
            Err(substr) => {
                let err = result.err().context("expected a rejection error")?;
                assert!(err.to_string().contains(substr), "got {err}");
            }
        }
        Ok(())
    }
}
