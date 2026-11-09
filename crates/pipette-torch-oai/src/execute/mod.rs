//! Benchmark execution against a docker- or uv-hosted OpenAI-compatible server.
//!
//! Kind modules talk to the server over HTTP and don't know which engine hosts
//! it. The CLI owns prepare and record.

mod asr_eval;
mod decode_throughput;
mod end_to_end_latency;
mod eval;
pub(crate) mod max_memory_usage;
mod prefill_throughput;
mod vl_spec_decode;
mod vlm_eval;

use anyhow::Context;

use pipette_memprobe::cgroup;
use pipette_ops::EvalCompletionsStore;
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_context::RunContext;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::vl_spec_decode::Speculation;
use pipette_plan_types::{
    BenchmarkDefinition, BenchmarkFlags, MemoryObservation, RunRequest, RunResponse, Runtime,
    RuntimeFlags, TorchOaiExtraRuntimeConfiguration,
};
use pipette_programs::ExternalPrograms;
use pipette_vl_spec_decode_corpus::VlSpecDecodeCorpusLease;
use pipette_vl_spec_decode_corpus::vlm_eval::VlmEvalCorpus;

use crate::{
    cell_flags, launch, memprobe,
    server::{self, Server},
};

/// Launch the server, run the cell, tear the server down. `run_context` lends
/// the eval checkpoints and the corpus benchmark's download. Beyond what
/// mlx/llamacpp take, `programs` is the `docker` and `uv` located at startup:
/// this engine spawns a host binary and looks none up itself.
pub fn run(
    req: &RunRequest,
    run_context: &dyn RunContext,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
    programs: &ExternalPrograms,
) -> anyhow::Result<RunResponse> {
    run_with(
        req,
        run_context,
        readiness_gate,
        observer,
        client_config,
        programs,
        cgroup::Tree::open(),
        memprobe::memory_watch,
    )
}

/// [`run`], reading the server's cgroup through `tree` and a memory cell's
/// server through what `open_watch` opens: a test's stand-ins.
#[expect(clippy::too_many_arguments, reason = "`run`'s, and the two stand-ins")]
fn run_with(
    req: &RunRequest,
    run_context: &dyn RunContext,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&TorchOaiExtraRuntimeConfiguration>,
    programs: &ExternalPrograms,
    tree: Result<cgroup::Tree, cgroup::TreeError>,
    open_watch: impl FnOnce(&Runtime) -> anyhow::Result<cgroup::Watch>,
) -> anyhow::Result<RunResponse> {
    // This engine runs no ASR speed cell, so one is refused before a model
    // load (and perhaps an image pull).
    if let BenchmarkDefinition::AsrThroughput(_) | BenchmarkDefinition::AsrMaxMemory(_) =
        &req.benchmark
    {
        return Err(not_implemented(req.benchmark.benchmark_type()));
    }
    // First, as on the other engines: a corpus refusal should not cost a launch.
    let lease = if let BenchmarkDefinition::VlSpecDecode(body) = &req.benchmark {
        Some(observer.outside(|| {
            pipette_vl_spec_decode_corpus::download(run_context, &body.benchmark_id, &body.dataset)
        })?)
    } else {
        None
    };
    // Likewise, as on llama.cpp: a catalog entry runs only its recipe's
    // models, in its recipe's thinking mode, refused before its corpus
    // downloads, and the corpus is checked against both its pins before the
    // server loads.
    let vlm_corpus = if let BenchmarkDefinition::VlmEval(body) = &req.benchmark {
        pipette_vl_spec_decode_corpus::vlm_eval::on_recipe(
            body,
            &req.model.declared,
            req.model_flags.as_ref(),
        )?;
        pipette_vl_spec_decode_corpus::vlm_eval::download(run_context, body)?
    } else {
        None
    };
    let (model_dir, drafter) = launch::model_dirs(req)?;
    // One derivation, two consumers: the server's argv and the flags the
    // response reports, so what is reported cannot drift from what launched.
    let flags = cell_flags::resolve_runtime_flags(req)?;
    // And what a vlm_eval cell refuses that needs no server.
    if let BenchmarkDefinition::VlmEval(body) = &req.benchmark {
        vlm_eval::preflight(req, body, vlm_corpus.as_ref(), &flags)?;
    }
    // An asr_eval as on llama.cpp: its recipe and what vLLM can run of it
    // before anything is fetched, and a checkpoint that settles every clip
    // downloads nothing and starts no server. Otherwise the corpus, checked
    // against both its pins, before the server loads.
    let asr = if let BenchmarkDefinition::AsrEval(body) = &req.benchmark {
        pipette_vl_spec_decode_corpus::asr_eval::on_recipe(
            body,
            &req.model.declared,
            req.model_flags.as_ref(),
        )?;
        let cell = asr_eval::preflight(body, &flags)?;
        let checkpoint = run_context.eval_completions().open(req)?;
        if pipette_ops::measurement::chunked_pending(
            &checkpoint,
            cell.declared(),
            asr_eval::id_at(body),
        )
        .is_empty()
        {
            tracing::info!(
                benchmark_id = %body.benchmark_id,
                "every sample is settled in the checkpoint; the corpus is not downloaded"
            );
            // The run that settled them was killed, and may have left its corpus.
            pipette_vl_spec_decode_corpus::asr_eval::remove_leftover(run_context, body)?;
            return Ok(asr_eval::settled(body, &cell, checkpoint)?
                .runtime_flags(flags)
                .build());
        }
        let corpus =
            pipette_vl_spec_decode_corpus::asr_eval::download(run_context, body, |_| Ok(()))?;
        pipette_vl_spec_decode_corpus::asr_eval::check_positions(
            body.benchmark_id.as_str(),
            &corpus,
            cell.declared(),
            asr_eval::id_at(body),
        )?;
        Some(asr_eval::AsrRun {
            cell,
            corpus,
            checkpoint,
        })
    } else {
        None
    };
    let server_args = launch::server_args(req, &flags, drafter.as_ref())?;
    let model = launch::model_value(&req.runtime.bound, &model_dir);
    // Before the launch: off Linux, or on a tree that counts no memory, a
    // memory cell is refused here and another docker cell goes unobserved,
    // with a warning. A tree that does not hold the server's cgroup fails the
    // probe at its stop. A uv memory cell is refused when pipette's own cgroup,
    // which its server shares, counts no memory; another uv cell is not
    // observed. An eval is not observed on any engine: a run resumed from its
    // checkpoint started this server for the samples left.
    let watch = match &req.benchmark {
        BenchmarkDefinition::MaxMemoryUsage(_) => {
            let watch = open_watch(&req.runtime.bound)?;
            let tree = tree?;
            if cell_flags::is_uv(&req.runtime.bound) {
                tree.own_cgroup()?;
            }
            Some((tree, watch))
        }
        BenchmarkDefinition::Eval(_)
        | BenchmarkDefinition::VlmEval(_)
        | BenchmarkDefinition::AsrEval(_) => None,
        BenchmarkDefinition::PrefillThroughput(_)
        | BenchmarkDefinition::DecodeThroughput(_)
        | BenchmarkDefinition::EndToEndLatency(_)
        | BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::VlSpecDecode(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => cell_flags::is_docker(&req.runtime.bound)
            .then_some(tree)
            .and_then(|tree| {
                tree.inspect_err(|err| {
                    tracing::warn!(error = %err, "no memory observation from the container");
                })
                .ok()
            })
            .map(|tree| (tree, cgroup::Watch::Observe)),
    };
    // Off the clock: a docker launch may pull its image first.
    let mut server = observer.outside(|| {
        server::launch(
            &req.runtime.bound,
            &server_args,
            &flags,
            &model_dir,
            drafter.as_ref(),
            client_config,
            programs,
        )
    })?;
    // From the launch, so the load counts, as on every engine.
    let mut probe = watch
        .map(|(tree, watch)| {
            let server_cgroup = server.cgroup();
            cgroup::Sampler::start(tree, watch, server_cgroup.members(), move |tree| {
                server_cgroup.locate(tree)
            })
        })
        .transpose()
        .context("could not start the memory probe's thread")?;
    server.wait_ready(server::READY_TIMEOUT)?;
    tracing::info!(
        benchmark_id = %req.benchmark.benchmark_id(),
        "running benchmark"
    );
    let response = dispatch(
        req,
        &model,
        &server,
        &mut probe,
        &run_context.eval_completions(),
        lease.as_ref().map(VlSpecDecodeCorpusLease::corpus),
        vlm_corpus.as_ref(),
        asr,
        readiness_gate,
        observer,
        &flags,
    )
    .map_err(|err| err.context(server.stop_with_tail()))?;
    // Before the response stops the server: the probe reads the live
    // server. A memory cell took the probe and records what it read.
    let memory = probe.map_or_else(MemoryObservation::default, memprobe::observe);
    Ok(response
        .process(&mut server)
        .runtime_flags(flags)
        .memory(memory)
        .build())
}

#[expect(clippy::too_many_arguments, reason = "one per thing a cell reads")]
fn dispatch(
    req: &RunRequest,
    model: &str,
    server: &Server,
    probe: &mut Option<memprobe::Probe>,
    eval_completions: &EvalCompletionsStore,
    corpus: Option<&pipette_vl_spec_decode_corpus::VlSpecDecodeCorpus>,
    vlm_corpus: Option<&VlmEvalCorpus>,
    asr: Option<asr_eval::AsrRun>,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    flags: &RuntimeFlags,
) -> anyhow::Result<RunResponseBuilder> {
    let state = server.state();
    let kind = req.benchmark.benchmark_type();
    let http_timeout = req
        .benchmark_flags
        .as_ref()
        .and_then(BenchmarkFlags::http_timeout);
    match &req.benchmark {
        BenchmarkDefinition::PrefillThroughput(benchmark) => prefill_throughput::run(
            benchmark,
            http_timeout,
            model,
            state,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::DecodeThroughput(benchmark) => decode_throughput::run(
            benchmark,
            http_timeout,
            model,
            state,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::EndToEndLatency(benchmark) => end_to_end_latency::run(
            benchmark,
            http_timeout,
            model,
            state,
            readiness_gate,
            observer,
        ),
        BenchmarkDefinition::MaxMemoryUsage(benchmark) => max_memory_usage::run(
            benchmark,
            http_timeout,
            model,
            state,
            probe
                .take()
                .context("`run` starts the memory cell's probe with its server")?,
            observer,
        ),
        BenchmarkDefinition::Eval(_) => eval::run(req, model, state, eval_completions),
        BenchmarkDefinition::VlSpecDecode(benchmark) => {
            // The launch refuses a drafter on vLLM; the flags refuse the cell
            // itself there, since the plan types define no vLLM variant. This
            // is the message for the run that gets past neither.
            anyhow::ensure!(
                matches!(state.kind, server::ServerKind::Sglang),
                "{kind} runs on SGLang only: vLLM's OpenAI endpoint reports no \
                 per-request draft counters"
            );
            vl_spec_decode::run(
                benchmark,
                Speculation::from_enabled(flags.speculative_enabled() == Some(true)),
                req.benchmark_flags.as_ref(),
                model,
                state,
                corpus.with_context(|| {
                    format!("{kind} needs a corpus; `run` downloads it before the launch")
                })?,
                readiness_gate,
                observer,
            )
        }
        BenchmarkDefinition::VlmEval(benchmark) => vlm_eval::run(
            req,
            benchmark,
            vlm_corpus,
            model,
            state,
            eval_completions,
            flags,
        ),
        BenchmarkDefinition::AsrEval(benchmark) => {
            let asr = asr.with_context(|| {
                format!("{kind} needs its corpus; `run` downloads it before the launch")
            })?;
            asr_eval::run(
                req,
                benchmark,
                &asr.cell,
                &asr.corpus,
                asr.checkpoint,
                model,
                state,
            )
        }
        // The ASR speed cells are refused before the launch, in `run_with`.
        BenchmarkDefinition::VlThroughput(_)
        | BenchmarkDefinition::VlMaxMemory(_)
        | BenchmarkDefinition::AsrThroughput(_)
        | BenchmarkDefinition::AsrMaxMemory(_) => Err(not_implemented(kind)),
    }
}

fn not_implemented(kind: pipette_plan_types::BenchmarkType) -> anyhow::Error {
    anyhow::anyhow!(
        "benchmark type {kind:?} is not yet implemented in pipette-torch-oai; currently \
         supported: prefill_throughput, decode_throughput, end_to_end_latency, \
         max_memory_usage, eval, vl_spec_decode, vlm_eval and asr_eval (vLLM)"
    )
}

/// The three timing cells under load, driven through a fake `/v1/completions`
/// server.
///
/// Concurrency is the property here, and it is invisible to a unit test of
/// any one function: the batch is assembled by the cell, dispatched by
/// `measurement::in_flight`, and only the server can say how many requests
/// were actually on the wire at once. So the fixture is a real socket that
/// holds each request open long enough for its siblings to arrive and records
/// the high-water mark.
#[cfg(test)]
mod batch_tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use rstest::rstest;

    use pipette_plan_types::{
        BatchSize, BenchmarkId, BenchmarkResultData, DecodeThroughputBenchmark,
        EndToEndLatencyBenchmark, NonNegativeCount, PositiveCount, PrefillThroughputBenchmark,
    };

    use super::*;
    use crate::server::test_support::crash_fake_docker;
    #[cfg(unix)]
    use crate::server::test_support::{
        FAKE_OWN_BYTES, FAKE_SWAP_BYTES, fake_host, fake_scope, fake_tree,
    };
    use crate::server::{ServerKind, ServerState};

    /// The fake `/tokenize` counts one token per byte, so a converged prompt is
    /// exactly this many bytes and the cells' `usage` validation is real.
    const PREFILL_TOKENS: i32 = 64;
    const DECODE_TOKENS: i32 = 4;

    /// How long each `/v1/completions` is held open. Without it a fast reply
    /// can retire before its siblings are dispatched, and the peak would read
    /// 1 on a batch that is genuinely concurrent.
    const HOLD: Duration = Duration::from_millis(80);

    struct FakeServer {
        port: u16,
        peak_in_flight: Arc<AtomicUsize>,
        completions: Arc<AtomicUsize>,
    }

    impl FakeServer {
        /// `wrong_usage_on` is the 1-based completion whose `usage` comes back
        /// with a prompt count the cell did not ask for.
        fn spawn(wrong_usage_on: Option<usize>) -> anyhow::Result<Self> {
            Self::on(TcpListener::bind("127.0.0.1:0")?, wrong_usage_on, None)
        }

        /// Serve on `listener`: the port a stand-in `docker` was told to
        /// publish. With `crash`, the stand-in in that directory crashes
        /// before the wrong usage goes out.
        fn on(
            listener: TcpListener,
            wrong_usage_on: Option<usize>,
            crash: Option<PathBuf>,
        ) -> anyhow::Result<Self> {
            let port = listener.local_addr()?.port();
            let peak_in_flight = Arc::new(AtomicUsize::new(0));
            let completions = Arc::new(AtomicUsize::new(0));
            let in_flight = Arc::new(AtomicUsize::new(0));
            let (peak, live, served) = (
                Arc::clone(&peak_in_flight),
                Arc::clone(&in_flight),
                Arc::clone(&completions),
            );
            std::thread::spawn(move || {
                listener
                    .incoming()
                    .filter_map(Result::ok)
                    .for_each(|stream| {
                        let (peak, live, served) =
                            (Arc::clone(&peak), Arc::clone(&live), Arc::clone(&served));
                        let crash = crash.clone();
                        std::thread::spawn(move || {
                            let _ = answer(
                                stream,
                                &peak,
                                &live,
                                &served,
                                wrong_usage_on,
                                crash.as_deref(),
                            );
                        });
                    });
            });
            Ok(Self {
                port,
                peak_in_flight,
                completions,
            })
        }

        fn state(&self) -> ServerState {
            ServerState {
                kind: ServerKind::Vllm,
                port: self.port,
                executable: PathBuf::from("python"),
            }
        }

        fn peak(&self) -> usize {
            self.peak_in_flight.load(Ordering::SeqCst)
        }

        fn completions(&self) -> usize {
            self.completions.load(Ordering::SeqCst)
        }
    }

    /// One request: `/tokenize` and an eval's chat answer immediately (neither
    /// is part of the measured batch, and must not count toward the peak); a
    /// completion is held open, counted, and answered streamed or whole as
    /// asked.
    fn answer(
        mut stream: TcpStream,
        peak: &AtomicUsize,
        live: &AtomicUsize,
        served: &AtomicUsize,
        wrong_usage_on: Option<usize>,
        crash: Option<&Path>,
    ) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line == "\r\n" {
                break;
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = String::new();
        reader.take(length as u64).read_to_string(&mut body)?;
        let body: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();

        if request_line.starts_with("GET /v1/models ") {
            return write_json(&mut stream, &serde_json::json!({ "data": [] }));
        }
        if request_line.contains("/tokenize") {
            let count = body["prompt"].as_str().unwrap_or_default().len();
            let tokens = vec![0u32; count];
            return write_json(
                &mut stream,
                &serde_json::json!({ "tokens": tokens, "count": count }),
            );
        }
        if request_line.contains("/v1/chat/completions") {
            // An eval's MCQ sample, answered with its first choice.
            return write_json(
                &mut stream,
                &serde_json::json!({ "choices": [{ "message": { "content": "A" } }] }),
            );
        }

        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(HOLD);
        let n = served.fetch_add(1, Ordering::SeqCst) + 1;
        live.fetch_sub(1, Ordering::SeqCst);

        let prompt_tokens = body["prompt"].as_str().unwrap_or_default().len();
        let prompt_tokens = if wrong_usage_on == Some(n) {
            if let Some(dir) = crash {
                crash_fake_docker(dir)?;
            }
            prompt_tokens + 1
        } else {
            prompt_tokens
        };
        let completion_tokens =
            usize::try_from(body["max_tokens"].as_u64().unwrap_or(1)).unwrap_or(1);
        let usage = serde_json::json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        });
        if body["stream"] == serde_json::json!(true) {
            write_stream(&mut stream, completion_tokens, &usage)
        } else {
            write_json(
                &mut stream,
                &serde_json::json!({
                    "choices": [{"index": 0, "text": "x", "finish_reason": "length"}],
                    "usage": usage,
                }),
            )
        }
    }

    fn write_json(stream: &mut TcpStream, body: &serde_json::Value) -> std::io::Result<()> {
        let body = body.to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )?;
        stream.flush()
    }

    /// A vLLM-shaped SSE stream: one chunk per token, a terminal chunk with
    /// the finish reason, the usage chunk, then `[DONE]`. The tokens are
    /// spaced so the first-to-last interval the decode cell reads is a
    /// positive span rather than zero.
    fn write_stream(
        stream: &mut TcpStream,
        tokens: usize,
        usage: &serde_json::Value,
    ) -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
             Connection: close\r\n\r\n"
        )?;
        stream.flush()?;
        for _ in 0..tokens {
            write!(
                stream,
                "data: {}\n\n",
                serde_json::json!({"choices": [{"index": 0, "text": "x", "finish_reason": null}]})
            )?;
            stream.flush()?;
            std::thread::sleep(Duration::from_millis(2));
        }
        write!(
            stream,
            "data: {}\n\n",
            serde_json::json!({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]})
        )?;
        write!(
            stream,
            "data: {}\n\n",
            serde_json::json!({"choices": [], "usage": usage})
        )?;
        write!(stream, "data: [DONE]\n\n")?;
        stream.flush()
    }

    /// The three cells this module batches, named so a case reads as the cell
    /// rather than as an arm of the open `BenchmarkType`.
    #[derive(Debug, Clone, Copy)]
    enum Cell {
        Prefill,
        Decode,
        EndToEnd,
    }

    fn benchmark(cell: Cell, batch_size: BatchSize) -> anyhow::Result<BenchmarkDefinition> {
        Ok(match cell {
            Cell::Prefill => BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "prefill_throughput_{PREFILL_TOKENS}_bs{batch_size}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL_TOKENS)?,
                parameter_batch_size: batch_size,
            }),
            Cell::Decode => BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!("decode_throughput_bs{batch_size}"))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL_TOKENS)?,
                parameter_decode_tokens: PositiveCount::try_new(DECODE_TOKENS)?,
                parameter_batch_size: batch_size,
            }),
            Cell::EndToEnd => BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new(format!("end_to_end_latency_bs{batch_size}"))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL_TOKENS)?,
                parameter_decode_tokens: PositiveCount::try_new(DECODE_TOKENS)?,
                parameter_batch_size: batch_size,
            }),
        })
    }

    fn no_observer() -> MeasurementObserver<'static> {
        MeasurementObserver::QUIET
    }

    fn run_cell(
        cell: Cell,
        batch_size: u32,
        state: &ServerState,
        readiness_gate: ReadinessGate,
        observer: MeasurementObserver<'_>,
    ) -> anyhow::Result<RunResponse> {
        let response = match benchmark(cell, BatchSize::try_new(batch_size)?)? {
            BenchmarkDefinition::PrefillThroughput(benchmark) => {
                prefill_throughput::run(&benchmark, None, "m", state, readiness_gate, observer)
            }
            BenchmarkDefinition::DecodeThroughput(benchmark) => {
                decode_throughput::run(&benchmark, None, "m", state, readiness_gate, observer)
            }
            BenchmarkDefinition::EndToEndLatency(benchmark) => {
                end_to_end_latency::run(&benchmark, None, "m", state, readiness_gate, observer)
            }
            other @ (BenchmarkDefinition::MaxMemoryUsage(_)
            | BenchmarkDefinition::Eval(_)
            | BenchmarkDefinition::VlThroughput(_)
            | BenchmarkDefinition::VlMaxMemory(_)
            | BenchmarkDefinition::VlmEval(_)
            | BenchmarkDefinition::AsrEval(_)
            | BenchmarkDefinition::VlSpecDecode(_)
            | BenchmarkDefinition::AsrThroughput(_)
            | BenchmarkDefinition::AsrMaxMemory(_)) => {
                anyhow::bail!("not a timing benchmark: {}", other.benchmark_type())
            }
        }?;
        Ok(response.build())
    }

    /// The peak is exact: every slot is held open past its siblings' dispatch,
    /// so a cell that fanned out by less reads low and one that overlapped its
    /// reps reads high. The gate falls between batches (a gate inside one would
    /// stall the slots it describes), so its count is the same at every size.
    #[rstest]
    #[case::prefill_reference(Cell::Prefill, 1)]
    #[case::prefill_pair(Cell::Prefill, 2)]
    #[case::prefill_four(Cell::Prefill, 4)]
    #[case::decode_reference(Cell::Decode, 1)]
    #[case::decode_pair(Cell::Decode, 2)]
    #[case::decode_four(Cell::Decode, 4)]
    #[case::e2e_reference(Cell::EndToEnd, 1)]
    #[case::e2e_pair(Cell::EndToEnd, 2)]
    #[case::e2e_four(Cell::EndToEnd, 4)]
    fn the_batch_size_is_what_is_in_flight(
        #[case] cell: Cell,
        #[case] batch_size: u32,
    ) -> anyhow::Result<()> {
        let server = FakeServer::spawn(None)?;
        let gate_calls = AtomicUsize::new(0);
        let gate = || {
            gate_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
        let out = run_cell(cell, batch_size, &server.state(), &gate, no_observer())?;

        let slots = usize::try_from(batch_size)?;
        assert_eq!(server.peak(), slots, "peak in flight");
        assert_eq!(
            server.completions(),
            slots * (pipette_ops::measurement::REPS + 1),
            "warm-up fills every slot, then one batch per rep"
        );
        assert_eq!(
            gate_calls.load(Ordering::SeqCst),
            pipette_ops::measurement::REPS + 1,
            "once before the prompt, once per rep, whatever the batch size"
        );
        let reported = match out.result_data {
            BenchmarkResultData::PrefillThroughput {
                prefill_time_ms, ..
            } => prefill_time_ms,
            BenchmarkResultData::DecodeThroughput { decode_time_ms, .. } => decode_time_ms,
            BenchmarkResultData::EndToEndLatency { total_time_ms, .. } => total_time_ms,
            other @ (BenchmarkResultData::MaxMemoryUsage { .. }
            | BenchmarkResultData::Eval { .. }
            | BenchmarkResultData::VlThroughput { .. }
            | BenchmarkResultData::AsrThroughput { .. }
            | BenchmarkResultData::VlSpecDecode { .. }) => {
                anyhow::bail!("unexpected result data {other:?}")
            }
        };
        assert!(reported > 0.0, "reported {reported} ms");
        Ok(())
    }

    /// A batch names itself in the recorded command, so rows at different
    /// concurrency can be told apart; the reference command has no batch line.
    #[rstest]
    #[case::prefill(Cell::Prefill)]
    #[case::decode(Cell::Decode)]
    #[case::e2e(Cell::EndToEnd)]
    fn the_reference_command_is_unchanged(#[case] cell: Cell) -> anyhow::Result<()> {
        let server = FakeServer::spawn(None)?;
        let reference = run_cell(cell, 1, &server.state(), &|| Ok(()), no_observer())?;
        assert_eq!(reference.command.len(), 2, "{:?}", reference.command);
        assert!(
            !reference.command.iter().any(|l| l.contains("batch_size")),
            "{:?}",
            reference.command
        );

        let batched = run_cell(cell, 2, &server.state(), &|| Ok(()), no_observer())?;
        assert_eq!(
            batched.command.last().map(String::as_str),
            Some("batch_size=2"),
            "{:?}",
            batched.command
        );
        // The two previews agree on everything the reference said.
        assert_eq!(batched.command.get(..2), reference.command.get(..2));
        Ok(())
    }

    /// Each slot is validated against its own `usage` rather than averaged
    /// into a mean that describes a workload nobody asked for.
    #[rstest]
    #[case::prefill(Cell::Prefill)]
    #[case::decode(Cell::Decode)]
    #[case::e2e(Cell::EndToEnd)]
    fn a_slot_that_came_back_wrong_fails_the_cell(#[case] cell: Cell) -> anyhow::Result<()> {
        // The 7th completion: past the four warm-up slots, inside rep 1.
        let server = FakeServer::spawn(Some(7))?;
        let err = run_cell(cell, 4, &server.state(), &|| Ok(()), no_observer())
            .err()
            .context("expected the mismatched slot to fail the cell")?;
        let message = format!("{err:#}");
        assert!(
            message.contains("rep 1/5") && message.contains("prefill_tokens 65, expected 64"),
            "{message}"
        );
        Ok(())
    }

    #[cfg(unix)]
    struct NoCorpus(PathBuf);

    #[cfg(unix)]
    impl RunContext for NoCorpus {
        fn fetch_dataset(&self, _: &pipette_plan_types::Dataset) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("no corpus in this test")
        }

        fn datasets_dir(&self) -> PathBuf {
            self.0.join("datasets")
        }

        fn eval_completions(&self) -> EvalCompletionsStore {
            EvalCompletionsStore::new(self.0.join("evals"))
        }
    }

    /// `run` through a stand-in `docker`: what the server printed lands on the
    /// row, or its last lines on the error, ahead of the cause. It gets one
    /// `docker stop` (none once it has crashed and gone) and no `docker kill`;
    /// the memory probe's `container inspect` aside.
    #[cfg(unix)]
    #[rstest]
    #[case::measured(None, false, None)]
    #[case::wrong_usage(
        Some(2),
        false,
        Some("was still running\nstderr tail:\nloading weights\nstdout tail:\nstopped")
    )]
    #[case::crashed(
        Some(2),
        true,
        Some("\nstderr tail:\nloading weights\nRuntimeError: CUDA error: out of memory")
    )]
    fn a_run_reports_what_the_server_printed(
        #[case] wrong_usage_on: Option<usize>,
        #[case] crash: bool,
        #[case] tail: Option<&str>,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::{DeclaredBound, Runtime, RuntimeOutput};
        use pipette_programs::ResolvedProgram;

        use crate::models::test_support::bound_model;
        use crate::server::test_support::{fake_docker, published_port, serve_fake};

        let dir = tempfile::tempdir()?;
        let docker = fake_docker(dir.path(), r#"echo "loading weights" >&2"#)?;
        let crash_dir = crash.then(|| dir.path().to_path_buf());
        let http = serve_fake(&dir.path().join("argv"), published_port, move |listener| {
            FakeServer::on(listener, wrong_usage_on, crash_dir).map(drop)
        })?;
        let programs = ExternalPrograms {
            docker: ResolvedProgram::at(&docker),
            ..ExternalPrograms::at_missing_paths()
        };
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "docker_vllm", "image_name": "vllm/vllm-openai", "image_tag": "v0.20.2",
            "flavor": "cpu"
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            // Under a `models/` store, which the orphan label is read off.
            model: DeclaredBound::already_bound(bound_model(&dir.path().join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: benchmark(Cell::EndToEnd, BatchSize::one())?,
        };

        let run = crate::run(
            &req,
            &NoCorpus(dir.path().to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &programs,
        );
        // The stand-in writes `cli` before the fifo the reader waits on, so a
        // run that failed before launching it would block the join for good.
        anyhow::ensure!(
            dir.path().join("cli").exists(),
            "the stand-in docker never ran: {:?}",
            run.as_ref().err()
        );
        http.join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        // A crash races the stop: the stand-in may still be exiting.
        let calls = std::fs::read_to_string(dir.path().join("calls")).unwrap_or_default();
        let calls: Vec<&str> = calls
            .lines()
            .filter(|line| !line.starts_with("container inspect "))
            .collect();
        let stops = calls
            .iter()
            .filter(|line| line.starts_with("stop -t 30 pipette-bench-"))
            .count();
        assert!(
            stops == calls.len() && (stops == 1 || crash && stops == 0),
            "{calls:?}"
        );
        match tail {
            None => assert_eq!(
                run?.runtime_output,
                RuntimeOutput {
                    runtime_stdout: Some("stopped\n".to_owned()),
                    runtime_stderr: Some("loading weights\n".to_owned()),
                }
            ),
            Some(tail) => {
                let err = run.err().context("the wrong usage should fail the cell")?;
                let message = format!("{err:#}");
                assert!(
                    message.starts_with("container pipette-bench-")
                        && message.contains(&format!("{tail}\nerror: ")),
                    "{message}"
                );
                assert!(message.contains("prefill_tokens"), "{message}");
            }
        }
        Ok(())
    }

    /// `run` on `benchmark` through a stand-in `docker` in `dir`, whose server
    /// answers every request, on [`fake_host`]'s tree there.
    #[cfg(unix)]
    fn run_on_stand_in(dir: &Path, benchmark: BenchmarkDefinition) -> anyhow::Result<RunResponse> {
        run_on_stand_in_with(
            dir,
            benchmark,
            "cpu",
            "",
            Ok(fake_host(dir)?),
            memprobe::memory_watch,
        )
    }

    /// [`run_on_stand_in`] on `flavor`, the stand-in running `greeting` before
    /// it serves, reading the container through `tree`, and a memory cell
    /// reading what `open_watch` opens.
    #[cfg(unix)]
    fn run_on_stand_in_with(
        dir: &Path,
        benchmark: BenchmarkDefinition,
        flavor: &str,
        greeting: &str,
        tree: Result<cgroup::Tree, cgroup::TreeError>,
        open_watch: impl FnOnce(&Runtime) -> anyhow::Result<cgroup::Watch>,
    ) -> anyhow::Result<RunResponse> {
        use pipette_plan_types::DeclaredBound;
        use pipette_programs::ResolvedProgram;

        use crate::models::test_support::bound_model;
        use crate::server::test_support::{fake_docker, published_port, serve_fake};

        let docker = fake_docker(dir, greeting)?;
        let http = serve_fake(&dir.join("argv"), published_port, |listener| {
            FakeServer::on(listener, None, None).map(drop)
        })?;
        let programs = ExternalPrograms {
            docker: ResolvedProgram::at(&docker),
            ..ExternalPrograms::at_missing_paths()
        };
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "docker_vllm", "image_name": "vllm/vllm-openai", "image_tag": "v0.20.2",
            "flavor": flavor
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(bound_model(&dir.join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        };
        let run = run_with(
            &req,
            &NoCorpus(dir.to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &programs,
            tree,
            open_watch,
        );
        // The stand-in writes `cli` before the fifo the reader waits on, so a
        // run that failed before launching it would block the join for good.
        anyhow::ensure!(
            dir.join("cli").exists(),
            "the stand-in docker never ran: {:?}",
            run.as_ref().err()
        );
        http.join()
            .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        run
    }

    /// The row carries the container's own memory, sampled from its cgroup
    /// on the host from the launch, with its swap watermark beside it where
    /// the kernel keeps one.
    #[cfg(unix)]
    #[rstest]
    #[case::with_a_swap_watermark(
        false,
        MemoryObservation::with_swap(FAKE_OWN_BYTES, FAKE_SWAP_BYTES)
    )]
    #[case::on_an_older_kernel(true, MemoryObservation::host_only(FAKE_OWN_BYTES))]
    fn a_run_observes_its_container(
        #[case] older_kernel: bool,
        #[case] expected: MemoryObservation,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let host = fake_host(dir.path())?;
        if older_kernel {
            std::fs::remove_file(fake_scope(dir.path()).join("memory.swap.peak"))?;
        }
        let response = run_on_stand_in_with(
            dir.path(),
            benchmark(Cell::EndToEnd, BatchSize::one())?,
            "cpu",
            "",
            Ok(host),
            memprobe::memory_watch,
        )?;
        assert_eq!(response.memory, expected);
        Ok(())
    }

    /// What `f` logs, as text, beside what it returns.
    #[cfg(unix)]
    fn logged<R>(f: impl FnOnce() -> R) -> anyhow::Result<(R, String)> {
        #[derive(Clone, Default)]
        struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

        impl Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .map_err(|_| std::io::Error::other("poisoned"))?
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let result = tracing::subscriber::with_default(subscriber, f);
        let bytes = captured
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned"))?
            .clone();
        Ok((result, String::from_utf8(bytes)?))
    }

    /// A docker run whose container's memory cannot be read goes unobserved,
    /// with a warning, rather than failing: watching cannot fail a cell that
    /// reports time. A refused host is known before the launch; a tree that
    /// does not hold the container fails the probe.
    #[cfg(unix)]
    #[rstest]
    #[case::refused(true)]
    #[case::unread(false)]
    fn a_run_on_an_unread_host_observes_nothing(#[case] refused: bool) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let tree = if refused {
            Err(cgroup::TreeError::NotLinux)
        } else {
            Ok(fake_tree(dir.path())?)
        };
        let cell = benchmark(Cell::EndToEnd, BatchSize::one())?;
        let (response, log) = logged(|| {
            run_on_stand_in_with(dir.path(), cell, "cpu", "", tree, memprobe::memory_watch)
        })?;
        assert_eq!(response?.memory, MemoryObservation::default());
        assert!(
            log.contains("no memory observation from the container"),
            "{log}"
        );
        Ok(())
    }

    /// A uv server has no container, so a host that cannot be read warns of
    /// none: the run never had an observation to lose.
    #[cfg(unix)]
    #[test]
    fn a_uv_run_warns_of_no_container() -> anyhow::Result<()> {
        use pipette_plan_types::{
            AbsolutePath, DeclaredBound, UvBuild, UvPythonVersion, UvRuntimeSource,
            UvServerVersion, UvVllm,
        };
        use pipette_programs::ResolvedProgram;

        use crate::models::test_support::bound_model;
        use crate::server::test_support::fake_docker;

        let dir = tempfile::tempdir()?;
        let programs = ExternalPrograms {
            docker: ResolvedProgram::at(fake_docker(dir.path(), "")?),
            ..ExternalPrograms::at_missing_paths()
        };
        // A python and no server, so the run ends at the launch, past the
        // watch.
        let venv = dir.path().join("venv");
        let python = pipette_venv::venv_python(&venv);
        std::fs::create_dir_all(python.parent().context("a venv python has a directory")?)?;
        std::fs::write(&python, "")?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.11.0".to_owned())?,
                build: UvBuild::try_new("cu129".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: UvRuntimeSource::AbsolutePreinstalled {
                    dir: AbsolutePath::try_new(venv.display().to_string())?,
                },
            })),
            model: DeclaredBound::already_bound(bound_model(&dir.path().join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: benchmark(Cell::EndToEnd, BatchSize::one())?,
        };
        let (run, log) = logged(|| {
            run_with(
                &req,
                &NoCorpus(dir.path().to_path_buf()),
                &|| Ok(()),
                no_observer(),
                None,
                &programs,
                Err(cgroup::TreeError::NotLinux),
                memprobe::memory_watch,
            )
        })?;
        let err = run.err().context("a venv with no server launched")?;
        assert!(
            format!("{err:#}").contains("failed to start server for benchmark"),
            "{err:#}"
        );
        assert!(!log.contains("no memory observation"), "{log}");
        Ok(())
    }

    /// An ASR cell is refused before the launch: with no docker to start, the
    /// error is the refusal, not the launch's.
    #[cfg(unix)]
    #[test]
    fn an_asr_cell_is_refused_before_the_server_starts() -> anyhow::Result<()> {
        use pipette_plan_types::{
            AsrDecodeTokens, AsrMaxMemoryBenchmark, AudioChannels, AudioDurationMillis,
            AudioSampleRate, DeclaredBound, NonNegativeCount,
        };

        use crate::models::test_support::bound_model;

        let dir = tempfile::tempdir()?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "docker_vllm", "image_name": "vllm/vllm-openai", "image_tag": "v0.20.2",
            "flavor": "cpu"
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(bound_model(&dir.path().join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::AsrMaxMemory(AsrMaxMemoryBenchmark {
                benchmark_id: BenchmarkId::try_new(
                    "asr_max_memory_10000ms_text0_decode256".to_owned(),
                )?,
                parameter_audio_duration_ms: AudioDurationMillis::try_new(10_000)?,
                parameter_audio_sample_rate_hz: AudioSampleRate::try_new(16_000)?,
                parameter_audio_channels: AudioChannels::try_new(1)?,
                parameter_text_tokens: NonNegativeCount::try_new(0)?,
                parameter_decode_tokens: AsrDecodeTokens::try_new(256)?,
            }),
        };
        let err = run_with(
            &req,
            &NoCorpus(dir.path().to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &ExternalPrograms::at_missing_paths(),
            Err(cgroup::TreeError::NotLinux),
            memprobe::memory_watch,
        )
        .err()
        .context("an ASR cell ran on torch-oai")?;
        assert!(
            format!("{err:#}").starts_with("benchmark type AsrMaxMemory is not yet implemented"),
            "{err:#}"
        );
        Ok(())
    }

    /// A vlm_eval catalog entry runs only its recipe's models: one it does not
    /// cover is refused before its corpus downloads (the stand-in context
    /// lends none) and before any server starts.
    #[cfg(unix)]
    #[test]
    fn a_vlm_eval_cell_off_its_recipe_is_refused_before_the_launch() -> anyhow::Result<()> {
        use pipette_plan_types::DeclaredBound;

        use crate::models::test_support::bound_model;

        let dir = tempfile::tempdir()?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "uv_vllm", "server_version": "0.31.0", "build": "rocm723",
            "python_version": "3.12", "source": {"type": "absolute_preinstalled", "dir": "/venv"},
        }))?;
        let benchmark = serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval",
            "benchmark_id": "vlm_eval_mmstar_qwen3_5_4b",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 1500,
            "parameter_max_tokens": 32768,
            "dataset": {
                "format": "vlm-eval-corpus/1", "adapter": "evals-server@1",
                "digest": "1".repeat(64), "bytes": 1,
                "upstream": {"source": "evals_server", "eval_id": "mmstar",
                             "dataset_version": "2026.10.1", "identity": "2".repeat(64)},
            },
            "recipe": {"family": "Qwen3.5 4B", "models": [{"repo": "Qwen/Qwen3.5-4B"}],
                       "enable_thinking": true, "sources": ["https://huggingface.co/Qwen/Qwen3.5-4B"]},
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(bound_model(&dir.path().join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        };
        let err = run_with(
            &req,
            &NoCorpus(dir.path().to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &ExternalPrograms::at_missing_paths(),
            Err(cgroup::TreeError::NotLinux),
            memprobe::memory_watch,
        )
        .err()
        .context("a vlm_eval cell off its recipe ran")?;
        assert!(format!("{err:#}").contains("is off its recipe"), "{err:#}");
        Ok(())
    }

    /// An asr_eval catalog entry runs only its recipe's models, and only what
    /// vLLM can run of it: an off-recipe model, or an on-recipe one whose
    /// profile vLLM has no model for, is refused before its corpus downloads
    /// or a server starts (the stand-in context has no corpus to give).
    #[cfg(unix)]
    #[rstest::rstest]
    #[case::off_recipe("voxtral", "Other", "Model", "is off its recipe")]
    #[case::no_vllm_model(
        "lfm25_audio",
        "LiquidAI",
        "LFM2.5-Audio-1.5B",
        "vLLM has no implementation of its model"
    )]
    fn an_asr_eval_cell_vllm_cannot_run_is_refused_before_the_launch(
        #[case] profile: &str,
        #[case] org: &str,
        #[case] repo: &str,
        #[case] refused: &str,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::DeclaredBound;

        use crate::models::test_support::bound_model;

        let dir = tempfile::tempdir()?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "uv_vllm", "server_version": "0.32.0", "build": "rocm723",
            "python_version": "3.12", "source": {"type": "absolute_preinstalled", "dir": "/venv"},
        }))?;
        let benchmark = serde_json::from_value(serde_json::json!({
            "benchmark_type": "asr_eval",
            "benchmark_id": "asr_eval_fleurs_nl_nl",
            "parameter_asr_task": "asr_fleurs_nl_nl",
            "parameter_dataset_version": "2026.10.1",
            "parameter_sample_count": 364,
            "parameter_prompt_profile": profile,
            "parameter_max_tokens": 256,
            "dataset": {
                "format": "asr-eval-corpus/1", "adapter": "evals-server@1",
                "digest": "1".repeat(64), "bytes": 1,
                "upstream": {"source": "evals_server", "eval_id": "asr_fleurs_nl_nl",
                             "dataset_version": "2026.10.1", "prompt_profile": profile,
                             "identity": "2".repeat(64)},
            },
            "recipe": {"family": "A family", "models": [{"repo": "LiquidAI/LFM2.5-Audio-1.5B"}, {"repo": "mistralai/Voxtral-Mini-3B-2507"}],
                       "enable_thinking": false, "sources": ["https://huggingface.co/x"]},
        }))?;
        let declared = serde_json::from_value(serde_json::json!({
            "type": "torch", "source": "huggingface", "org": org, "repo_name": repo,
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound {
                declared,
                bound: bound_model(&dir.path().join("models"), false)?,
            },
            runtime_flags: Some(RuntimeFlags::AsrEvalUvVllmTorch {
                tensor_parallel_size: None,
                dtype: None,
                max_model_len: None,
                prefix_caching: None,
                parallel: None,
                envs: Vec::new(),
                raw: Vec::new(),
            }),
            model_flags: Some(pipette_plan_types::ModelFlags::AsrEvalTorch {
                enable_thinking: Some(false),
            }),
            benchmark_flags: None,
            benchmark,
        };
        let err = run_with(
            &req,
            &NoCorpus(dir.path().to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &ExternalPrograms::at_missing_paths(),
            Err(cgroup::TreeError::NotLinux),
            memprobe::memory_watch,
        )
        .err()
        .context("an asr_eval cell vLLM cannot run ran")?;
        assert!(format!("{err:#}").contains(refused), "{err:#}");
        Ok(())
    }

    /// What a vlm_eval cell refuses that needs no server is refused before
    /// the launch: here a pinned context with no room for a prompt, on a host
    /// with no `uv` to launch with.
    #[cfg(unix)]
    #[test]
    fn a_vlm_eval_cell_with_no_room_is_refused_before_the_launch() -> anyhow::Result<()> {
        use pipette_plan_types::DeclaredBound;

        use crate::models::test_support::bound_model;

        let dir = tempfile::tempdir()?;
        let runtime: Runtime = serde_json::from_value(serde_json::json!({
            "type": "uv_vllm", "server_version": "0.31.0", "build": "rocm723",
            "python_version": "3.12", "source": {"type": "absolute_preinstalled", "dir": "/venv"},
        }))?;
        let benchmark = serde_json::from_value(serde_json::json!({
            "benchmark_type": "vlm_eval",
            "benchmark_id": "vlm_eval_local",
            "parameter_lmms_task": "mmstar",
            "parameter_dataset_version": "local",
            "parameter_sample_count": 1,
            "parameter_max_tokens": 4096,
            "samples": [{"id": "a", "messages": [{"role": "user", "content": "q"}]}],
        }))?;
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(bound_model(&dir.path().join("models"), false)?),
            runtime_flags: Some(RuntimeFlags::VlmEvalUvVllmTorch {
                tensor_parallel_size: None,
                dtype: None,
                max_model_len: Some(pipette_plan_types::PositiveCount::try_new(4096)?),
                prefix_caching: None,
                parallel: None,
                envs: Vec::new(),
                raw: Vec::new(),
            }),
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        };
        let err = run_with(
            &req,
            &NoCorpus(dir.path().to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &ExternalPrograms::at_missing_paths(),
            Err(cgroup::TreeError::NotLinux),
            memprobe::memory_watch,
        )
        .err()
        .context("a vlm_eval cell with no room for a prompt ran")?;
        assert!(
            format!("{err:#}").contains("pins max_model_len=4096"),
            "{err:#}"
        );
        Ok(())
    }

    #[cfg(unix)]
    fn memory_benchmark() -> anyhow::Result<BenchmarkDefinition> {
        use pipette_plan_types::MaxMemoryUsageBenchmark;

        Ok(BenchmarkDefinition::MaxMemoryUsage(
            MaxMemoryUsageBenchmark {
                benchmark_id: BenchmarkId::try_new(format!("max_memory_usage_{PREFILL_TOKENS}"))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(PREFILL_TOKENS)?,
            },
        ))
    }

    /// A memory cell off the GPU records the container's own memory, sampled
    /// from its cgroup on the host, and no GPU figure; its observation is
    /// that figure, and its extras what the stop's `memory.stat` and
    /// `memory.peak` held, page cache included. No `docker exec` reads it.
    #[cfg(unix)]
    #[test]
    fn a_memory_cell_on_the_cpu_records_the_container_s_own_memory() -> anyhow::Result<()> {
        use crate::server::test_support::FAKE_MEMORY_PEAK;

        let dir = tempfile::tempdir()?;
        let response = run_on_stand_in(dir.path(), memory_benchmark()?)?;
        assert_eq!(
            response.result_data,
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: FAKE_OWN_BYTES,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            }
        );
        assert_eq!(
            response.memory,
            MemoryObservation::with_swap(FAKE_OWN_BYTES, FAKE_SWAP_BYTES)
        );
        let extra = |key: &str| response.client_extras.get(key);
        assert_eq!(
            extra("memory_stat"),
            Some(&serde_json::json!({"anon": 1000, "file": 2000, "kernel": 300, "shmem": 40}))
        );
        assert_eq!(
            extra("cgroup_memory_peak_bytes"),
            Some(&serde_json::json!(FAKE_MEMORY_PEAK))
        );
        assert!(
            extra("host_samples")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|samples| samples > 0),
            "{:?}",
            response.client_extras
        );
        assert_eq!(extra("samples"), Some(&serde_json::json!(0)));
        let calls = std::fs::read_to_string(dir.path().join("calls"))?;
        assert!(
            calls.lines().all(|call| !call.starts_with("exec")),
            "{calls}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_memory_cell_on_an_unread_host_is_refused_before_the_launch() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let err = run_on_stand_in_with(
            dir.path(),
            memory_benchmark()?,
            "cpu",
            "",
            Err(cgroup::TreeError::NotLinux),
            memprobe::memory_watch,
        )
        .err()
        .context("expected a refusal")?;
        assert!(
            format!("{err:#}").contains("the stand-in docker never ran"),
            "{err:#}"
        );
        assert!(
            format!("{err:#}").contains("which only Linux has"),
            "{err:#}"
        );
        Ok(())
    }

    /// On whole AMD cards the row records the floor that came off their
    /// figure, and whether the open saw them stop handing memory back.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_amd_memory_cell_records_its_cards_floor() -> anyhow::Result<()> {
        use pipette_memprobe::amd::AmdGpu;
        use pipette_memprobe::amd::test_support::{fake_sys, set_used};

        let dir = tempfile::tempdir()?;
        let host = fake_host(dir.path())?;
        let sys = fake_sys(dir.path(), true)?;
        let open_watch = move |_: &Runtime| {
            let gpu = AmdGpu::stand_in(&sys)?;
            set_used(&sys, 5_300)?;
            Ok(cgroup::Watch::PeakWithAmdGpu(gpu))
        };
        let response = run_on_stand_in_with(
            dir.path(),
            memory_benchmark()?,
            "amd_gpu",
            "",
            Ok(host),
            open_watch,
        )?;
        assert_eq!(
            response.result_data,
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: FAKE_OWN_BYTES,
                max_gpu_bytes: Some(5_000),
                max_npu_bytes: None,
            }
        );
        assert_eq!(
            response.client_extras.get("gpu_floor_bytes"),
            Some(&serde_json::json!(300))
        );
        assert_eq!(
            response.client_extras.get("gpu_floor_settled"),
            Some(&serde_json::json!(true))
        );
        Ok(())
    }

    /// A memory cell's probe reads from the container's launch, before the
    /// server is ready, so the load counts: the stand-in `docker` serves only
    /// once the GPU has been read, and gives up waiting after 10 s. The GPU
    /// figure is that of the processes in the container's cgroup, the host
    /// figure the container's own memory, read on the same ticks.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_memory_cell_reads_its_gpu_from_the_launch() -> anyhow::Result<()> {
        use std::collections::BTreeMap;

        use pipette_memprobe::NotAvailable;
        use pipette_memprobe::cgroup::NvidiaGpu;

        use crate::server::test_support::FAKE_CONTAINER_ID;

        let dir = tempfile::tempdir()?;
        let host = fake_host(dir.path())?;
        let proc = dir.path().join("proc");
        [
            (
                41,
                format!("0::/system.slice/docker-{FAKE_CONTAINER_ID}.scope\n"),
            ),
            (
                42,
                "0::/user.slice/user-1001.slice/session-7.scope\n".to_owned(),
            ),
        ]
        .into_iter()
        .try_for_each(|(pid, cgroup)| {
            std::fs::create_dir_all(proc.join(format!("{pid}")))?;
            std::fs::write(proc.join(format!("{pid}/cgroup")), cgroup)
        })?;
        let probed = dir.path().join("probed");
        let open_watch = move |_: &Runtime| {
            Ok(cgroup::Watch::PeakWithNvidiaGpu(NvidiaGpu::stand_in(
                move || {
                    let _ = std::fs::write(&probed, "");
                    Ok(BTreeMap::from([(41, Ok(5_000)), (42, Err(NotAvailable))]))
                },
            )))
        };
        let response = run_on_stand_in_with(
            dir.path(),
            memory_benchmark()?,
            "nvidia_gpu",
            r#"tries=0
    until [ -e "$DIR/probed" ]; do
      tries=$((tries + 1))
      if [ "$tries" -gt 200 ]; then touch "$DIR/late"; break; fi
      sleep 0.05
    done"#,
            Ok(host),
            open_watch,
        )?;
        anyhow::ensure!(
            !dir.path().join("late").exists(),
            "the server was ready before the probe read the GPU"
        );
        assert_eq!(
            response.result_data,
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: FAKE_OWN_BYTES,
                max_gpu_bytes: Some(5_000),
                max_npu_bytes: None,
            }
        );
        let calls = std::fs::read_to_string(dir.path().join("calls"))?;
        assert!(
            calls.starts_with("container inspect --format {{.Id}} {{.State.Pid}} pipette-bench-"),
            "{calls}"
        );
        Ok(())
    }

    /// `benchmark` on a uv `build`, through a venv in `dir` whose `vllm`
    /// stands in for the server, on [`fake_session`]'s host there (`counted`
    /// as it says), a memory cell reading what `open_watch` opens; `serve`
    /// once the cell gets past its refusals, or exit at once. The server
    /// places itself in the session and writes its pid to `server-pid`.
    #[cfg(target_os = "linux")]
    fn run_uv_cell(
        dir: &Path,
        build: &str,
        benchmark: BenchmarkDefinition,
        open_watch: impl FnOnce(&Runtime) -> anyhow::Result<cgroup::Watch>,
        serve: bool,
        counted: bool,
    ) -> anyhow::Result<RunResponse> {
        use pipette_plan_types::{
            AbsolutePath, DeclaredBound, UvBuild, UvPythonVersion, UvRuntimeSource,
            UvServerVersion, UvVllm,
        };
        use pipette_programs::ResolvedProgram;

        use crate::models::test_support::bound_model;
        use crate::server::test_support::{
            FAKE_SESSION, fake_docker, fake_session, serve_fake, write_executable,
        };

        let venv = dir.join("venv");
        let bin = pipette_venv::venv_bin(&venv);
        std::fs::create_dir_all(&bin)?;
        std::fs::write(pipette_venv::venv_python(&venv), "")?;
        // A server no one serves exits at once, so a cell expected to be
        // refused fails fast if it launches anyway.
        let body = if serve {
            "while :; do sleep 0.05; done"
        } else {
            "exit 1"
        };
        write_executable(
            &bin.join("vllm"),
            &format!(
                r#"#!/bin/sh
trap 'exit 0' TERM
DIR='{}'
mkdir -p "$DIR/proc/$$"
echo "0::{FAKE_SESSION}" > "$DIR/proc/$$/cgroup"
echo $$ > "$DIR/server-pid"
touch "$(dirname "$0")/started"
echo "$@" > "$(dirname "$0")/argv"
{body}
"#,
                dir.display()
            ),
        )?;
        let tree = fake_session(dir, counted)?;
        let port_flag = |words: &[&str]| -> Option<u16> {
            let flag = words.iter().position(|word| *word == "--port")?;
            words.get(flag + 1)?.parse().ok()
        };
        let http = serve
            .then(|| {
                serve_fake(&bin.join("argv"), port_flag, |listener| {
                    FakeServer::on(listener, None, None).map(drop)
                })
            })
            .transpose()?;
        // A docker the orphan reap finds nothing with.
        let programs = ExternalPrograms {
            docker: ResolvedProgram::at(fake_docker(dir, "")?),
            uv: ResolvedProgram::at(bin.join("uv")),
            ..ExternalPrograms::at_missing_paths()
        };
        let req = RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.21.0".to_owned())?,
                build: UvBuild::try_new(build.to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: UvRuntimeSource::AbsolutePreinstalled {
                    dir: AbsolutePath::try_new(venv.display().to_string())?,
                },
            })),
            model: DeclaredBound::already_bound(bound_model(&dir.join("models"), false)?),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        };
        let run = run_with(
            &req,
            &NoCorpus(dir.to_path_buf()),
            &|| Ok(()),
            no_observer(),
            None,
            &programs,
            Ok(tree),
            open_watch,
        );
        if let Some(http) = http {
            // The server writes `started` before the fifo the reader waits
            // on, so a run that failed before launching it would block the
            // join for good.
            anyhow::ensure!(
                bin.join("started").exists(),
                "the server never ran: {:?}",
                run.as_ref().err()
            );
            http.join()
                .map_err(|_| anyhow::anyhow!("the fake server panicked"))??;
        }
        run
    }

    /// Host: pipette's cgroup less pipette's own memory. GPU: the server's
    /// processes alone, not another in the cgroup.
    #[cfg(target_os = "linux")]
    #[rstest]
    #[case::cpu("cpu", None)]
    #[case::cuda("cu129", Some(5_000))]
    fn a_uv_memory_cell_reads_the_server_in_pipette_s_cgroup(
        #[case] build: &str,
        #[case] gpu_bytes: Option<u64>,
    ) -> anyhow::Result<()> {
        use std::collections::BTreeMap;

        use pipette_memprobe::cgroup::NvidiaGpu;

        use crate::server::test_support::{FAKE_OTHER_PID, FAKE_PIPETTE_BYTES, FAKE_SERVER_BYTES};

        let dir = tempfile::tempdir()?;
        let server_pid = dir.path().join("server-pid");
        let open_watch = move |runtime: &Runtime| match gpu_bytes {
            Some(bytes) => Ok(cgroup::Watch::PeakWithNvidiaGpu(NvidiaGpu::stand_in(
                move || {
                    let server = std::fs::read_to_string(&server_pid)
                        .ok()
                        .and_then(|pid| pid.trim().parse().ok());
                    Ok(server
                        .map(|pid| (pid, Ok(bytes)))
                        .into_iter()
                        .chain([(FAKE_OTHER_PID, Ok(7))])
                        .collect::<BTreeMap<_, _>>())
                },
            ))),
            None => memprobe::memory_watch(runtime),
        };
        let response = run_uv_cell(
            dir.path(),
            build,
            memory_benchmark()?,
            open_watch,
            true,
            true,
        )?;
        assert_eq!(
            response.result_data,
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: FAKE_SERVER_BYTES,
                max_gpu_bytes: gpu_bytes,
                max_npu_bytes: None,
            }
        );
        assert_eq!(
            response.memory,
            MemoryObservation::with_swap(FAKE_SERVER_BYTES, FAKE_SWAP_BYTES)
        );
        assert_eq!(
            response.client_extras.get("others_bytes"),
            Some(&serde_json::json!(FAKE_PIPETTE_BYTES))
        );
        assert_eq!(
            response.client_extras.get("others_max_bytes"),
            Some(&serde_json::json!(FAKE_PIPETTE_BYTES))
        );
        assert_eq!(
            response.client_extras.get("others_listed"),
            Some(&serde_json::json!(true))
        );
        assert!(response.client_extras.contains_key("host_bytes_at_launch"));
        // The whole session's watermark, not the server's.
        assert!(
            !response
                .client_extras
                .contains_key("cgroup_memory_peak_bytes")
        );
        let venv_vllm = pipette_venv::venv_bin(&dir.path().join("venv")).join("vllm");
        assert_eq!(
            response.executable.as_deref(),
            Some(venv_vllm.display().to_string().as_str())
        );
        Ok(())
    }

    /// Refused before the launch where pipette's cgroup counts no memory of
    /// its own, as the host's root.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_uv_memory_cell_in_an_uncounted_cgroup_is_refused_before_the_launch() -> anyhow::Result<()>
    {
        let dir = tempfile::tempdir()?;
        let err = run_uv_cell(
            dir.path(),
            "cpu",
            memory_benchmark()?,
            memprobe::memory_watch,
            false,
            false,
        )
        .err()
        .context("expected a refusal")?;
        let message = format!("{err:#}");
        assert!(
            message.contains("runs in the host's root cgroup"),
            "{message}"
        );
        let argv = pipette_venv::venv_bin(&dir.path().join("venv")).join("argv");
        assert!(!argv.exists(), "the server launched");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_uv_timing_cell_observes_nothing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let response = run_uv_cell(
            dir.path(),
            "cpu",
            benchmark(Cell::EndToEnd, BatchSize::one())?,
            memprobe::memory_watch,
            true,
            false,
        )?;
        assert_eq!(response.memory, MemoryObservation::default());
        Ok(())
    }

    /// An eval records no observation, though its container's peak can be
    /// read: a run resumed from the checkpoint would have started it for the
    /// samples left.
    #[cfg(unix)]
    #[test]
    fn an_eval_records_no_observation() -> anyhow::Result<()> {
        use pipette_plan_types::{EvalBenchmark, EvalId, PositiveCount};

        let dir = tempfile::tempdir()?;
        let eval = BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_mcq".to_owned())?,
            parameter_eval_id: EvalId::from("math_500"),
            parameter_dataset_name: "d".to_owned(),
            parameter_max_tokens: PositiveCount::try_new(8)?,
            parameter_max_context: None,
            parameter_mcq_choices: Some(vec!["A".to_owned(), "B".to_owned()]),
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: Some(vec![serde_json::json!({
                "id": "s1", "messages": [{"role": "user", "content": "q"}]
            })]),
        });
        let response = run_on_stand_in(dir.path(), eval)?;
        assert_eq!(response.memory, MemoryObservation::default());
        let calls = std::fs::read_to_string(dir.path().join("calls"))?;
        assert!(!calls.contains("container inspect"), "{calls}");
        Ok(())
    }
}
