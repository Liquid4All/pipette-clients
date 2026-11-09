//! VL throughput over `mlx_vlm.server`, through
//! [`measurement::vl_throughput()`].
//!
//! Each rep is one streamed `/v1/chat/completions` on this client's clock:
//! `prompt_ms` is the send to the first token, `predicted_ms` the first token
//! to the last, scaled to every decode step, and `total_time_ms` the send to
//! the chunk carrying `finish_reason`. The server's own prefill time is not
//! used: on the model families that cache image features (Qwen3.5, Gemma 4)
//! the image encode runs before its clock starts. EOS is biased away so the
//! full `max_tokens` comes out.

use std::num::NonZeroUsize;

use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::{Sampler, SupportedWatch};
use pipette_ops::generation::Clock;
use pipette_ops::measurement::{
    self, MeasurementObserver, MeasurementWarmup, VlThroughputMeasurementReading,
};
use pipette_ops::readiness::ReadinessGate;
use pipette_ops::run_response::RunResponseBuilder;
use pipette_plan_types::{
    BenchmarkFlags, MlxMacosPipetteExtraRuntimeConfiguration, PositiveCount, RunResponse,
    VlThroughputBenchmark,
};
use pipette_subprocess::start_process;

use crate::image_tokens;
use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{chat, completions, server, tokenizer};

pub(super) fn run(
    benchmark: &VlThroughputBenchmark,
    benchmark_flags: Option<&BenchmarkFlags>,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    readiness_gate: ReadinessGate,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the model loads.
    measurement::require_decode_steps(benchmark.parameter_decode_tokens)?;
    // A 0 would drop the image the cell measures; read it as the default, 1,
    // as llama.cpp does.
    let num_images = benchmark
        .parameter_num_images
        .into_inner()
        .unsigned_abs()
        .max(1);
    let (text, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        benchmark.parameter_text_tokens.into_inner().unsigned_abs(),
        server::VISION_TRUSTS_REMOTE_CODE,
    )?;
    let eos_bias = completions::eos_bias(&eos_ids);
    // A cell without the count still measures its times and reports none. A
    // count the measured prompt cannot hold fails the cell instead: the helper
    // runs the server's own processor, so the images did not reach that prompt.
    let image_tokens = match image_tokens::count(
        venv_python,
        model_dir,
        benchmark.parameter_image_width.into_inner().unsigned_abs(),
        benchmark.parameter_image_height.into_inner().unsigned_abs(),
        num_images,
        server::VISION_TRUSTS_REMOTE_CODE,
        client_config,
    )
    .and_then(|tokens| Ok(PositiveCount::try_from(tokens)?))
    {
        Ok(tokens) => Some(tokens),
        Err(err) => {
            tracing::warn!(
                error = %format_args!("{err:#}"),
                "could not count the images' tokens; the row reports none"
            );
            None
        }
    };
    let png = base64::engine::general_purpose::STANDARD.encode(pipette_ops::png::gray(
        benchmark.parameter_image_width.into_inner().unsigned_abs(),
        benchmark.parameter_image_height.into_inner().unsigned_abs(),
    )?);
    let http_timeout = measurement::http_timeout_with_default(
        benchmark_flags.and_then(BenchmarkFlags::http_timeout),
        1,
        completions::TIMING_TIMEOUT.as_secs(),
    );

    readiness_gate()?;
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut mlx_server = start_process(
        server::VISION_SERVER_MODULE,
        server::vision_command(
            venv_python,
            model_dir,
            port,
            None,
            NonZeroUsize::MIN,
            client_config,
        )?,
        server::STOP_GRACE,
    )?;
    let sampler = Sampler::start(mlx_server.pid(), SupportedWatch::OBSERVE)?;
    mlx_server.wait_ready(server::READY_TIMEOUT, server::ready(&url)?)?;
    let client = HttpClient::loopback("pipette", http_timeout)?;
    let chat = |text: &str, image_url: &str, images: u32, max_tokens: u32| {
        let body = chat::body(model_dir, text, image_url, images, max_tokens, &eos_bias)?;
        chat::stream(&client, &url, &body)
    };
    let mut attempt = 0;
    let result = measurement::vl_throughput(
        benchmark,
        NonZeroUsize::try_from(measurement::REPS)?,
        readiness_gate,
        observer,
        MeasurementWarmup::Once,
        (),
        |()| {
            attempt += 1;
            let generation = chat(
                &text,
                &image_url(&png, &attempt.to_string()),
                num_images,
                benchmark
                    .parameter_decode_tokens
                    .into_inner()
                    .unsigned_abs(),
            )?;
            Ok(VlThroughputMeasurementReading::from_generation(
                &generation,
                image_tokens,
                Clock::Client,
            )?)
        },
    );
    let (memory, _) = sampler.stop();
    let result_data = result.map_err(|err| err.context(mlx_server.stop_with_tail()))?;
    Ok(RunResponseBuilder::new(result_data)
        .process(&mut mlx_server)
        .memory(memory)
        .build())
}

/// The image as a data URL `tag` makes unique: the server caches image
/// features by URL, so a repeated one would take the encode out of every rep
/// after the warm-up.
fn image_url(png: &str, tag: &str) -> String {
    format!("data:image/png;attempt={tag};base64,{png}")
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use pipette_plan_types::{BenchmarkId, BenchmarkResultData, NonNegativeCount};

    use super::*;
    use crate::models::test_support::bound_model;
    use crate::runtimes::test_support::{bound_runtime, live_vl};

    /// `mlx_vlm` reads a data URL's payload after its first comma.
    #[test]
    fn a_tag_changes_the_url_not_the_image() {
        let (one, two) = (image_url("AAAA", "1"), image_url("AAAA", "2"));
        assert_ne!(one, two);
        assert_eq!(one.split_once(',').map(|(_, png)| png), Some("AAAA"));
        assert_eq!(two.split_once(',').map(|(_, png)| png), Some("AAAA"));
    }

    /// Refused before the gate and the tokenizer, too.
    #[test]
    fn a_single_decode_token_is_refused_before_the_model_loads() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let benchmark = VlThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_throughput_224x224_t0_d1_f1")?,
            parameter_image_width: PositiveCount::try_new(224)?,
            parameter_image_height: PositiveCount::try_new(224)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: PositiveCount::try_new(1)?,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        };
        let gated = Cell::new(false);
        let Err(err) = run(
            &benchmark,
            None,
            &ResolvedMlxPython::try_from(&bound_runtime(tmp.path())?)?,
            &ResolvedMlxModel::try_from(&bound_model(tmp.path(), false)?)?,
            &|| {
                gated.set(true);
                Ok(())
            },
            MeasurementObserver::QUIET,
            None,
        ) else {
            anyhow::bail!("a one-token decode cell was measured");
        };
        assert!(
            matches!(
                err.downcast_ref::<measurement::VlThroughputMeasurementError>(),
                Some(measurement::VlThroughputMeasurementError::TooFewTokens { .. })
            ),
            "{err:#}"
        );
        assert!(!gated.get(), "refused only after the gate");
        Ok(())
    }

    /// Needs Apple silicon, a venv holding `mlx-vlm` and an MLX VL model:
    ///
    /// ```text
    /// PIPETTE_MLX_LIVE_PYTHON=<venv>/bin/python PIPETTE_MLX_LIVE_TARGET=<dir> \
    ///   cargo test -p pipette-mlx --lib live_vl_throughput -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn live_vl_throughput_through_mlx_vlm_server() -> anyhow::Result<()> {
        let (venv_python, model_dir) = live_vl()?;
        let benchmark = VlThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_throughput_512x512_t16_d32_f2")?,
            parameter_image_width: PositiveCount::try_new(512)?,
            parameter_image_height: PositiveCount::try_new(512)?,
            parameter_text_tokens: NonNegativeCount::try_new(16)?,
            parameter_decode_tokens: PositiveCount::try_new(32)?,
            parameter_num_images: NonNegativeCount::try_new(2)?,
        };
        let response = run(
            &benchmark,
            None,
            &venv_python,
            &model_dir,
            &|| Ok(()),
            MeasurementObserver::QUIET,
            None,
        )?;
        eprintln!("{:?}", response.result_data);
        let BenchmarkResultData::VlThroughput {
            prompt_tokens,
            image_tokens,
            total_time_ms,
            ..
        } = response.result_data
        else {
            anyhow::bail!("not a VL throughput result: {:?}", response.result_data);
        };
        anyhow::ensure!(
            image_tokens.is_some_and(|image_tokens| image_tokens + 16 <= prompt_tokens),
            "{image_tokens:?} image tokens of {prompt_tokens}"
        );
        anyhow::ensure!(
            total_time_ms.is_some(),
            "no total_time_ms: {:?}",
            response.result_data
        );
        Ok(())
    }
}
