//! End-to-end latency on `llama-server`: each rep sends the batch's identical
//! prompts as one streamed `/completion` and times every prompt from the send
//! to its own final chunk, the one with `stop: true`.

use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Context;
use reqwest::blocking::Client;
use reqwest::header::CONNECTION;
use serde::Deserialize;
use serde_json::{Value, json};

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::{self, Arrived, Counts, Finish, Observation};
use pipette_ops::measurement::{
    self, EndToEndMeasurementPerPromptReading, EndToEndMeasurementReading, MeasurementObserver,
    MeasurementWarmup,
};
use pipette_ops::prompt_seed;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_ops::sse;
use pipette_plan_types::{
    BenchmarkFlagRef, EndToEndLatencyBenchmark, EndToEndLatencyLlamacppCliStockToolsGgufText,
    LlamacppCliStockToolsExtraRuntimeConfiguration, RunResponse, RuntimeFlags,
};
use pipette_subprocess::start_process;

use crate::models::ResolvedGgufTextModel;
use crate::runtimes::{self, ResolvedLlamaServer};
use crate::server::{self, LoggingMarker, ServerCommandBuilder};

#[expect(
    clippy::too_many_arguments,
    reason = "the request's parts, and the seams its entry takes"
)]
pub fn run(
    benchmark: &EndToEndLatencyBenchmark,
    runtime_flags: EndToEndLatencyLlamacppCliStockToolsGgufText,
    benchmark_flags: &BenchmarkFlagRef,
    program: &ResolvedLlamaServer,
    model: &ResolvedGgufTextModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let slots = benchmark.parameter_batch_size.as_u32();
    let per_slot_ctx = benchmark
        .parameter_prefill_tokens
        .into_inner()
        .unsigned_abs()
        .saturating_add(
            benchmark
                .parameter_decode_tokens
                .into_inner()
                .unsigned_abs(),
        );
    let runtime_flags = EndToEndLatencyLlamacppCliStockToolsGgufText {
        load_mode: runtimes::get_load_mode(runtime_flags.mmap, runtime_flags.load_mode),
        ctx_size: Some(match runtime_flags.ctx_size {
            Some(total) => total,
            None => server::ctx_across_slots(per_slot_ctx, slots)?,
        }),
        ..runtime_flags
    };
    let launch = launch(&runtime_flags, program, model, slots, client_config);
    let timeout = measurement::http_timeout_for(benchmark_flags.http_timeout_seconds, slots);
    let http = HttpClient::loopback("pipette", timeout)?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let ready = server::health(&url, timeout)?;
    let command = launch.build(port)?;

    let mut llama_server = start_process("llama-server", command, Duration::ZERO)?;
    let sampler = Sampler::start(llama_server.pid(), SupportedWatch::OBSERVE)?;
    llama_server.wait_ready(timeout, ready)?;
    let worked = (|| {
        let prompts = benchmark.parameter_batch_size.in_flight()?.get();
        let prompt = prompt_seed::build_prompt_text(
            benchmark
                .parameter_prefill_tokens
                .into_inner()
                .unsigned_abs(),
            |text| server::tokenize(&http, &url, text),
        )?;
        let body = completion_body(
            &prompt,
            prompts,
            benchmark
                .parameter_decode_tokens
                .into_inner()
                .unsigned_abs(),
        );
        let url = format!("{url}/completion");
        // The warm-up sends the measured request, safe with `cache_prompt:
        // false`, and so puts a prompt on every slot before rep 1.
        measurement::end_to_end(
            benchmark,
            NonZeroUsize::try_from(measurement::REPS)?,
            readiness_gate,
            observer,
            MeasurementWarmup::Once,
            (),
            |()| Ok(send(http.client(), &url, &body, prompts)?),
        )
    })();
    let (memory, _) = sampler.stop();
    let data = worked.map_err(|err| err.context(llama_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(data)
        .process(&mut llama_server)
        .memory(memory)
        .runtime_flags(RuntimeFlags::EndToEndLatencyLlamacppCliStockToolsGgufText(
            runtime_flags,
        ))
        .build())
}

/// One rep. The request is built before the clock starts, so serializing it is
/// not timed.
fn send(
    client: &Client,
    url: &str,
    body: &Value,
    prompts: usize,
) -> anyhow::Result<EndToEndMeasurementReading> {
    let request = client
        .post(url)
        // llama-server drops a stream's connection although its headers keep
        // it alive, so a pooled one can fail the next rep.
        .header(CONNECTION, "close")
        .json(body);
    let sent = Instant::now();
    let response = pipette_http::send(request).context("failed to call /completion")?;
    let observed = observe(BufReader::new(response).lines(), prompts, &Instant::now)?;
    let readings = generation::generations(prompts, sent, &observed)?
        .iter()
        .map(EndToEndMeasurementPerPromptReading::from_generation)
        .collect::<anyhow::Result<_>>()?;
    Ok(EndToEndMeasurementReading::PerPrompt(readings))
}

/// Streamed so each prompt's end can be timed on its own.
fn completion_body(prompt: &str, prompts: usize, decode_tokens: u32) -> Value {
    // Text rather than token ids, so the server's tokenizing is inside the
    // timed window.
    json!({
        "prompt": vec![prompt; prompts],
        "stream": true,
        "temperature": 0.0,
        "n_predict": decode_tokens,
        // The server masks every end-of-generation token under it (llama.cpp
        // b8718+, #21203), so each prompt runs to `n_predict`.
        "ignore_eos": true,
        "cache_prompt": false,
    })
}

/// What llama-server's native stream said for each of `prompts` prompts, each
/// event stamped as it arrives. The stream carries no `[DONE]`, so reading
/// stops at the last prompt's end.
fn observe(
    lines: impl Iterator<Item = std::io::Result<String>>,
    prompts: usize,
    now: &dyn Fn() -> Instant,
) -> anyhow::Result<Vec<Observation>> {
    let mut ended = vec![false; prompts];
    let mut observed = Vec::new();
    let mut events = sse::payloads(lines);
    while ended.contains(&false) {
        let done = ended.iter().filter(|ended| **ended).count();
        let Some(data) = events.next() else {
            anyhow::bail!("the stream ended after {done} of {prompts} prompts");
        };
        let data = data.with_context(|| format!("after {done} of {prompts} prompts ended"))?;
        let at = now();
        // From b9425 every prompt after the first opens with a `null` event.
        let Some(event) =
            serde_json::from_str::<Option<Event>>(&data).context("a stream event is not JSON")?
        else {
            continue;
        };
        if let Some(error) = event.error {
            anyhow::bail!("llama-server failed mid-stream: {error}");
        }
        let prompt = event.index;
        let slot = ended.get_mut(prompt).with_context(|| {
            format!("llama-server answered for prompt {prompt} of a {prompts}-prompt request")
        })?;
        if event.stop {
            anyhow::ensure!(!*slot, "llama-server ended prompt {prompt} twice");
            *slot = true;
        }
        observed.extend(
            event
                .arrivals()
                .map(|what| Observation { prompt, at, what }),
        );
    }
    Ok(observed)
}

/// One chunk of the native stream: a token, or a prompt's final chunk with
/// `stop: true`, why it stopped and its counts.
#[derive(Debug, Deserialize)]
struct Event {
    /// The prompt's place in the request; `0` on a build that leaves it out.
    #[serde(default)]
    index: usize,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    stopped_limit: bool,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default)]
    timings: Option<CompletionTimings>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CompletionTimings {
    prompt_n: u32,
    predicted_n: u32,
}

impl Event {
    fn arrivals(self) -> impl Iterator<Item = Arrived> {
        // Streamed, the final chunk carries no text: the last token came before.
        let token = (!self.stop).then_some(Arrived::Token);
        let finish = match self.stop_type.as_deref() {
            _ if !self.stop => None,
            _ if self.stopped_limit => Some(Finish::Limit),
            Some("limit") => Some(Finish::Limit),
            Some("eos" | "word") => Some(Finish::Stop),
            Some(other) => Some(Finish::Other(other.to_owned())),
            None => None,
        };
        let counts = self.timings.filter(|_| self.stop).map(|timings| {
            Arrived::Counts(Counts {
                prompt_tokens: timings.prompt_n,
                completion_tokens: timings.predicted_n,
            })
        });
        token
            .into_iter()
            .chain(finish.map(Arrived::Finish))
            .chain(counts)
            .chain(self.stop.then_some(Arrived::End))
    }
}

/// The cell's server, every field named: a flag plan-types adds fails to compile
/// here until this cell renders it or sets it aside.
///
/// `slots` is the benchmark's. At the reference batch size it sets none, so a
/// `--parallel` in `raw` passes through with a warning; above it the builder
/// refuses the pair, as llama-server would honor one while the context is sized
/// for the other.
fn launch(
    flags: &EndToEndLatencyLlamacppCliStockToolsGgufText,
    program: &ResolvedLlamaServer,
    model: &Path,
    slots: u32,
    client_config: Option<&LlamacppCliStockToolsExtraRuntimeConfiguration>,
) -> ServerCommandBuilder {
    let EndToEndLatencyLlamacppCliStockToolsGgufText {
        threads,
        number_gpu_layers,
        mmap,
        load_mode,
        flash_attention,
        ctx_size,
        no_cache,
        envs,
        raw,
    } = flags.clone();
    ServerCommandBuilder::new(program.clone(), model, LoggingMarker::EndToEndLatency)
        .threads(threads)
        .number_gpu_layers(number_gpu_layers)
        .mmap(mmap)
        .load_mode(load_mode)
        .flash_attention(flash_attention)
        .raw(raw)
        .ctx_size(ctx_size)
        .no_cache(no_cache)
        .slots((slots > 1).then_some(slots))
        .envs(envs)
        .client_config(client_config.cloned())
}

#[cfg(test)]
mod tests {
    use pipette_ops::generation::Generation;
    use pipette_plan_types::{LlamacppFlashAttention, LlamacppLoadMode};

    use super::*;
    use crate::runtimes::test_llama_server;

    fn server_args(
        flags: &EndToEndLatencyLlamacppCliStockToolsGgufText,
        slots: u32,
    ) -> Result<Vec<String>, server::ServerArgsError> {
        launch(
            flags,
            &test_llama_server(),
            Path::new("model.gguf"),
            slots,
            None,
        )
        .args()
    }

    /// Every knob set to its own value, so a knob that lands on another's flag shows.
    fn every_knob_set() -> EndToEndLatencyLlamacppCliStockToolsGgufText {
        EndToEndLatencyLlamacppCliStockToolsGgufText {
            threads: Some(3),
            number_gpu_layers: Some(7),
            mmap: None,
            load_mode: Some(LlamacppLoadMode::DirectIo),
            flash_attention: Some(LlamacppFlashAttention::On),
            ctx_size: Some(4096),
            no_cache: Some(true),
            envs: Vec::new(),
            raw: vec!["--poll".to_owned(), "0".to_owned()],
        }
    }

    #[test]
    fn the_server_takes_every_knob_and_the_slots() -> anyhow::Result<()> {
        assert_eq!(
            server_args(&every_knob_set(), 2)?.join(" "),
            "--ctx-checkpoints 0 --ctx-size 4096 --flash-attn on --load-mode dio --n-gpu-layers 7 --no-cache-prompt --no-warmup --parallel 2 --poll 0 --threads 3"
        );
        Ok(())
    }

    /// An e2e cell with the context `run` would have overlaid for `slots`
    /// slots and nothing else pinned.
    fn runtime_flags(slots: u32) -> EndToEndLatencyLlamacppCliStockToolsGgufText {
        EndToEndLatencyLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            ctx_size: Some(768 * slots),
            no_cache: None,
            envs: Vec::new(),
            raw: Vec::new(),
        }
    }

    /// At the reference batch size the slots render no `--parallel`: the flags
    /// are stored with the row, and a stray `--parallel 1` would split identical
    /// configurations. Above it each request gets the window a serial run gave one.
    #[rstest::rstest]
    #[case::reference(1, None, "768")]
    #[case::batched(4, Some("4"), "3072")]
    fn the_slot_count_and_context_follow_the_batch_size(
        #[case] slots: u32,
        #[case] parallel: Option<&str>,
        #[case] ctx: &str,
    ) -> anyhow::Result<()> {
        let args = server_args(&runtime_flags(slots), slots)?;
        let rendered = args
            .windows(2)
            .find(|w| w[0] == "--parallel")
            .map(|w| w[1].as_str());
        assert_eq!(rendered, parallel, "{args:?}");
        assert!(
            args.windows(2).any(|w| w == ["--ctx-size", ctx]),
            "{args:?}"
        );
        Ok(())
    }

    fn ticking(start: Instant) -> impl Fn() -> Instant {
        let calls = std::cell::Cell::new(0);
        move || {
            calls.set(calls.get() + 1);
            start + Duration::from_millis(calls.get())
        }
    }

    fn sse(events: &[&str]) -> impl Iterator<Item = std::io::Result<String>> {
        events
            .iter()
            .flat_map(|event| [format!("data: {event}"), String::new()])
            .map(Ok)
    }

    fn read(events: &[&str], prompts: usize) -> anyhow::Result<Vec<Generation>> {
        let start = Instant::now();
        let observed = observe(sse(events), prompts, &ticking(start))?;
        generation::generations(prompts, start, &observed)
    }

    const PARTIAL_0: &str = r#"{"index":0,"content":"a","stop":false}"#;
    const PARTIAL_1: &str = r#"{"index":1,"content":"b","stop":false}"#;
    const PARTIAL_UNTAGGED: &str = r#"{"content":"a","stop":false}"#;
    const END_0: &str =
        r#"{"index":0,"stop":true,"stop_type":"limit","timings":{"prompt_n":16,"predicted_n":4}}"#;
    const END_1: &str =
        r#"{"index":1,"stop":true,"stop_type":"limit","timings":{"prompt_n":16,"predicted_n":4}}"#;
    const END_UNTAGGED: &str =
        r#"{"stop":true,"stop_type":"limit","timings":{"prompt_n":16,"predicted_n":4}}"#;

    /// Either spelling of "stopped at the limit" is accepted; a stop on EOS
    /// or a stop string is not.
    #[rstest::rstest]
    #[case::stopped_limit(
        r#"{"stop":true,"stopped_limit":true,"timings":{"prompt_n":8,"predicted_n":8}}"#,
        true
    )]
    #[case::stop_type_limit(
        r#"{"stop":true,"stop_type":"limit","timings":{"prompt_n":8,"predicted_n":8}}"#,
        true
    )]
    #[case::eos(
        r#"{"stop":true,"stop_type":"eos","timings":{"prompt_n":8,"predicted_n":8}}"#,
        false
    )]
    #[case::unsaid(r#"{"stop":true,"timings":{"prompt_n":8,"predicted_n":8}}"#, false)]
    fn the_generation_must_stop_at_the_limit(
        #[case] end: &str,
        #[case] accepted: bool,
    ) -> anyhow::Result<()> {
        let [generation] = <[Generation; 1]>::try_from(read(&[PARTIAL_UNTAGGED, end], 1)?)
            .map_err(|got| anyhow::anyhow!("expected one generation, got {got:?}"))?;
        let result = generation.counts_at_limit();
        assert_eq!(result.is_ok(), accepted, "{result:?}");
        Ok(())
    }

    /// The prompts' chunks interleave and their ends arrive in any order; each
    /// prompt's last token is its last partial chunk, its end its final chunk,
    /// returned in request order. An end with no index is the first prompt's,
    /// and a `null` event carries nothing.
    #[rstest::rstest]
    #[case::interleaved(&[PARTIAL_0, PARTIAL_1, END_1, PARTIAL_0, END_0], 2, &[(4.0, 5.0), (2.0, 3.0)])]
    #[case::null_opens_a_later_prompt(&[PARTIAL_0, "null", PARTIAL_1, END_1, PARTIAL_0, END_0], 2, &[(5.0, 6.0), (3.0, 4.0)])]
    #[case::untagged(&[PARTIAL_UNTAGGED, END_UNTAGGED], 1, &[(1.0, 2.0)])]
    fn each_prompt_is_timed_to_its_own_end(
        #[case] events: &[&str],
        #[case] prompts: usize,
        #[case] expected: &[(f64, f64)],
    ) -> anyhow::Result<()> {
        let stamps: Vec<(f64, f64)> = read(events, prompts)?
            .iter()
            .map(|generation| (generation.last_token_ms, generation.end_ms))
            .collect();
        anyhow::ensure!(stamps == expected, "{stamps:?}");
        Ok(())
    }

    #[rstest::rstest]
    #[case::cut_short(&[PARTIAL_0, END_1], "after 1 of 2 prompts")]
    #[case::error_event(&[PARTIAL_0, r#"{"error":{"message":"slot unavailable"}}"#], "slot unavailable")]
    #[case::unknown_prompt(&[r#"{"index":2,"stop":true,"timings":{"prompt_n":16,"predicted_n":4}}"#], "prompt 2 of a 2-prompt request")]
    #[case::ended_twice(&[END_1, END_1], "prompt 1 twice")]
    fn a_broken_stream_fails_the_rep(
        #[case] events: &[&str],
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let err = observe(sse(events), 2, &Instant::now)
            .err()
            .context("expected the rep to fail")?;
        assert!(format!("{err:#}").contains(needle), "{err:#}");
        Ok(())
    }

    #[test]
    fn a_batch_asks_for_the_prompt_once_per_slot() {
        assert_eq!(
            completion_body("seed text", 3, 4),
            json!({
                "prompt": ["seed text", "seed text", "seed text"],
                "stream": true,
                "temperature": 0.0,
                "n_predict": 4,
                "ignore_eos": true,
                "cache_prompt": false,
            })
        );
    }
}
