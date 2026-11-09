//! The corpus benchmark (`vl_spec_decode`) on `mlx_vlm.server`: one VL model,
//! one server process, one traversal, one row. Speculation and the batch size
//! are launch settings on this server (`--draft-model`, `--draft-block-size`,
//! `--max-num-seqs`), not request fields. Its draft counters are diffs of one
//! global tally, so they are per-turn only at a batch size of 1. The traversal
//! and the SSE reader are shared with llama.cpp
//! (`pipette_vl_spec_decode_corpus::turn`, `pipette_ops::vl_spec_decode`).
//!
//! See `docs/methodology/vl-spec-decode.md`.

use std::path::{Path, PathBuf};

use serde_json::json;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::measurement::MeasurementObserver;
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::vl_spec_decode::{Speculation, TimingsSource};
use pipette_plan_types::{
    BenchmarkFlags, MlxMacosPipetteExtraRuntimeConfiguration, Model, RunResponse, RuntimeFlags,
    SamplingTemperature, VlSpecDecodeBenchmark,
};
use pipette_subprocess::start_process;
use pipette_vl_spec_decode_corpus::{VlSpecDecodeCorpus, turn};

use crate::models::{ResolvedMlxModel, ResolvedMlxSpeculativeModel};
use crate::runtimes::ResolvedMlxPython;
use crate::server;

/// `runtime_flags` carries `speculative_enabled` resolved against `model`,
/// which decides whether the drafter loads ([`model_dirs`]).
#[expect(
    clippy::too_many_arguments,
    reason = "the benchmark's parts, the corpus, and the seams its entry takes"
)]
pub(super) fn run(
    benchmark: &VlSpecDecodeBenchmark,
    benchmark_flags: Option<&BenchmarkFlags>,
    runtime_flags: RuntimeFlags,
    venv_python: &ResolvedMlxPython,
    model: &Model,
    corpus: &VlSpecDecodeCorpus,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    let speculation = Speculation::from_enabled(runtime_flags.speculative_enabled() == Some(true));
    let params = turn::Params::new(
        benchmark,
        benchmark_flags.and_then(BenchmarkFlags::http_timeout),
        speculation,
    )?;
    let (target_dir, draft) = model_dirs(model, speculation)?;

    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let command = server::vision_command(
        venv_python,
        &target_dir,
        port,
        draft.as_deref().map(|dir| (dir, params.draft_length)),
        params.batch_size,
        client_config,
    )?;
    let mut mlx_server = start_process(server::VISION_SERVER_MODULE, command, server::STOP_GRACE)?;
    let sampler = Sampler::start(mlx_server.pid(), SupportedWatch::OBSERVE)?;
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let client = HttpClient::loopback("pipette", params.http_timeout)?
        .client()
        .clone();
    let envelope = envelope(&target_dir, params.temperature)?;
    let outcome = if params.batch_size.get() == 1 {
        turn::measure(
            benchmark,
            &params,
            corpus,
            readiness_gate,
            observer,
            &|messages, max_tokens| envelope.stream(&client, &url, messages, max_tokens),
        )
    } else {
        // The typed entry sends a batch as one call; `mlx_vlm.server` takes
        // one conversation per request.
        turn::run(
            &params,
            corpus,
            readiness_gate,
            observer,
            &|messages, max_tokens| envelope.send(&client, &url, messages, max_tokens),
        )
    };
    // The sampler stops first, while the server can still be read.
    let (memory, _) = sampler.stop();
    let response = outcome.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(response
        .process(&mut mlx_server)
        .memory(memory)
        .runtime_flags(runtime_flags)
        .build())
}

/// An MLX drafter is a separate bundle, so speculation off means the target
/// alone, on a regular model and a spec-decoding pair alike.
fn model_dirs(
    model: &Model,
    speculation: Speculation,
) -> anyhow::Result<(ResolvedMlxModel, Option<PathBuf>)> {
    match (speculation, model) {
        (Speculation::On, model) => {
            let model = ResolvedMlxSpeculativeModel::try_from(model)?;
            Ok((model.target, Some(model.draft)))
        }
        (Speculation::Off, model @ Model::MlxSpeculative(_)) => {
            Ok((ResolvedMlxSpeculativeModel::try_from(model)?.target, None))
        }
        (Speculation::Off, model) => Ok((ResolvedMlxModel::try_from(model)?, None)),
    }
}

/// What `mlx_vlm.server` wants around the shared body. Without
/// `include_usage` the stream never carries the complete `timings` the counters
/// are read from. No prompt cache field: the server ignores one. Prefix
/// caching is off at launch (`server::vision_command`), and a turn with a
/// nonzero `cache_n` is refused. `model` is the served path: any other value
/// is fetched from the Hub.
fn envelope(model_id: &Path, temperature: SamplingTemperature) -> anyhow::Result<turn::Envelope> {
    turn::Envelope::new(
        json!({
            "model": model_id.display().to_string(),
            "stream_options": { "include_usage": true },
        }),
        TimingsSource::Runtime,
        temperature,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Context;
    use rstest::rstest;

    use pipette_ops::vl_spec_decode::TurnRecord;
    use pipette_plan_types::BatchSize;

    use super::*;
    use crate::models::test_support::bound_model;

    /// One real turn with speculation on and off: the launcher, the envelope
    /// and the shared reader against a live `mlx_vlm.server`. Needs Apple
    /// silicon, a venv holding `mlx-vlm`, and a VL target and its drafter:
    ///
    /// ```text
    /// PIPETTE_MLX_LIVE_PYTHON=<venv>/bin/python \
    /// PIPETTE_MLX_LIVE_TARGET=<dir> PIPETTE_MLX_LIVE_DRAFT=<dir> \
    /// PIPETTE_MLX_LIVE_IMAGE=<jpeg> \
    ///   cargo test -p pipette-mlx --lib live_both_cells -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn live_both_cells_through_mlx_vlm_server() -> anyhow::Result<()> {
        use base64::Engine;

        let var = |name: &str| {
            std::env::var(name)
                .map(PathBuf::from)
                .with_context(|| format!("{name} must be set to run the live test"))
        };
        let python = var("PIPETTE_MLX_LIVE_PYTHON")?;
        let target = var("PIPETTE_MLX_LIVE_TARGET")?;
        let draft = var("PIPETTE_MLX_LIVE_DRAFT")?;
        let image = var("PIPETTE_MLX_LIVE_IMAGE")?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(std::fs::read(&image)?);
        let messages = vec![json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "Describe this image in detail." },
                { "type": "image_url",
                  "image_url": { "url": format!("data:image/jpeg;base64,{encoded}") } },
            ],
        })];
        let envelope = envelope(&target, SamplingTemperature::greedy())?;
        let client = HttpClient::loopback(
            "pipette",
            Duration::from_secs(turn::DEFAULT_TURN_TIMEOUT_SECS),
        )?
        .client()
        .clone();

        // Both conditions, each on its own process, as two jobs would.
        let run_cell =
            |speculation: Speculation, draft: Option<(&Path, u32)>| -> anyhow::Result<TurnRecord> {
                let one = BatchSize::one().in_flight()?;
                let port = pipette_subprocess::pick_server_port()?;
                let url = server::url(port);
                let mut mlx_server = start_process(
                    server::VISION_SERVER_MODULE,
                    server::vision_command(&python, &target, port, draft, one, None)?,
                    server::STOP_GRACE,
                )?;
                mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
                let record = envelope
                    .send(&client, &url, messages.clone(), 200)
                    .map_err(|failure| failure.error)
                    .with_context(|| speculation.label());
                mlx_server.stop();
                record
            };
        let spec = run_cell(Speculation::On, Some((&draft, 8)))?;
        let base = run_cell(Speculation::Off, None)?;

        let forwards = spec.counted_forwards;
        eprintln!(
            "speculation on:  generated={} decode_tokens={} drafted={} accepted={} \
             decode_forwards={forwards} accept_length={:.3} predicted_ms={:.1}",
            spec.generated_tokens,
            spec.decode_tokens(),
            spec.drafted_tokens,
            spec.accepted_tokens,
            spec.decode_tokens() as f64 / forwards as f64,
            spec.predicted_ms,
        );
        eprintln!(
            "speculation off: generated={} predicted_ms={:.1}",
            base.generated_tokens, base.predicted_ms
        );

        // What the reduction would refuse, asserted per turn so the failure
        // names the cell rather than the sum.
        anyhow::ensure!(
            spec.drafted_tokens > 0,
            "the speculative turn drafted nothing: the drafter is not wired up"
        );
        anyhow::ensure!(spec.accepted_tokens <= spec.drafted_tokens);
        anyhow::ensure!(
            spec.counted_forwards > 0,
            "the speculative turn reported no draft_rounds"
        );
        anyhow::ensure!(
            base.drafted_tokens == 0 && base.accepted_tokens == 0,
            "the turn with speculation off reported speculation counters"
        );
        anyhow::ensure!(spec.predicted_ms > 0.0 && base.predicted_ms > 0.0);
        // Greedy identity, which the join reads off the two rows' output
        // digests.
        anyhow::ensure!(
            spec.text == base.text && spec.generated_tokens == base.generated_tokens,
            "the cells diverged: {} vs {} tokens",
            spec.generated_tokens,
            base.generated_tokens
        );
        Ok(())
    }

    /// Off, a pair's target runs alone rather than being refused as not a
    /// plain `mlx` model.
    #[rstest]
    #[case::on_a_pair(Speculation::On, true)]
    #[case::off_on_a_pair(Speculation::Off, true)]
    #[case::off_on_a_plain_model(Speculation::Off, false)]
    fn only_speculation_on_loads_the_drafter(
        #[case] speculation: Speculation,
        #[case] speculative: bool,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let (target, draft) = model_dirs(&bound_model(tmp.path(), speculative)?, speculation)?;
        assert_eq!(
            (target.to_path_buf(), draft),
            (
                tmp.path().join("target"),
                (speculation == Speculation::On).then(|| tmp.path().join("draft"))
            )
        );
        Ok(())
    }

    #[test]
    fn the_body_asks_for_the_counters_the_reduction_needs() -> anyhow::Result<()> {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let envelope = envelope(
            Path::new("/models/entry/blobs/target"),
            SamplingTemperature::greedy(),
        )?;
        assert_eq!(
            envelope.body(messages.clone(), 2048),
            json!({
                "model": "/models/entry/blobs/target",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 2048,
                "stream": true,
                "stream_options": { "include_usage": true },
            })
        );
        assert_eq!(envelope.source(), TimingsSource::Runtime);
        Ok(())
    }
}
