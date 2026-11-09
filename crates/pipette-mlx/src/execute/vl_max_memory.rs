//! VL peak memory over `mlx_vlm.server`, through
//! [`measurement::vl_max_memory()`], a warm-up server first
//! ([`crate::memory`]).
//!
//! `phys_footprint` is sampled from `/health` on; its lifetime maximum still
//! holds the model load's peak. Nothing is sized for the request: `mlx_vlm`'s
//! KV cache grows with the prompt.

use std::num::NonZeroUsize;
use std::path::Path;

use base64::Engine;

use pipette_http::HttpClient;
use pipette_memprobe::Watch;
use pipette_ops::measurement::{
    self, MeasurementObserver, MemoryMeasurementWarmup, VlMaxMemoryMeasurementReading,
};
use pipette_plan_types::{
    MlxMacosPipetteExtraRuntimeConfiguration, RunResponse, VlMaxMemoryBenchmark,
};

use crate::models::ResolvedMlxModel;
use crate::runtimes::ResolvedMlxPython;
use crate::{chat, completions, memory, server, tokenizer};

pub(super) fn run(
    benchmark: &VlMaxMemoryBenchmark,
    venv_python: &ResolvedMlxPython,
    model_dir: &ResolvedMlxModel,
    observer: MeasurementObserver<'_>,
    client_config: Option<&MlxMacosPipetteExtraRuntimeConfiguration>,
) -> anyhow::Result<RunResponse> {
    // Before the tokenizer, the first process this cell starts.
    let watch = Watch::Peak.supported()?;
    // Then a size out of range fails before the tokenizer loads too.
    let png = base64::engine::general_purpose::STANDARD.encode(pipette_ops::png::gray(
        benchmark.parameter_image_width.into_inner().unsigned_abs(),
        benchmark.parameter_image_height.into_inner().unsigned_abs(),
    )?);
    let (text, eos_ids) = tokenizer::prompt(
        venv_python,
        model_dir,
        benchmark.parameter_text_tokens.into_inner().unsigned_abs(),
        server::VISION_TRUSTS_REMOTE_CODE,
    )?;
    let decode_tokens = benchmark
        .parameter_decode_tokens
        .map_or(1, |tokens| tokens.into_inner().unsigned_abs());
    let body = request_body(
        model_dir,
        &text,
        &png,
        benchmark.parameter_num_images.into_inner().unsigned_abs(),
        decode_tokens,
        &eos_ids,
    )?;
    let client = HttpClient::loopback("pipette", completions::TIMING_TIMEOUT)?;

    // A reply short of the output length fails the cell: the bias covers only
    // the tokenizer's EOS ids, so the server can still stop early.
    let stream = |url: &str| {
        chat::stream(&client, url, &body)
            .and_then(|generation| measurement::decoded_in_full(&generation, decode_tokens))
    };
    let vision_command = |port| {
        server::vision_command(
            venv_python,
            model_dir,
            port,
            None,
            NonZeroUsize::MIN,
            client_config,
        )
    };

    let mut served = None;
    let result = measurement::vl_max_memory(
        benchmark,
        observer,
        MemoryMeasurementWarmup::Once,
        &mut served,
        |served| {
            let max_host_bytes = memory::serve(
                served,
                server::VISION_SERVER_MODULE,
                vision_command,
                watch,
                stream,
            )?;
            Ok(VlMaxMemoryMeasurementReading {
                max_host_bytes,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            })
        },
    );
    memory::respond(result, &mut served)
}

fn request_body(
    model_dir: &Path,
    text: &str,
    png: &str,
    num_images: u32,
    decode_tokens: u32,
    eos_ids: &[u32],
) -> anyhow::Result<serde_json::Value> {
    chat::body(
        model_dir,
        text,
        &format!("data:image/png;base64,{png}"),
        // A 0 would drop the image the cell measures.
        num_images.max(1),
        decode_tokens,
        &completions::eos_bias(eos_ids),
    )
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, BenchmarkResultData, NonNegativeCount, PositiveCount};

    use super::*;
    use crate::models::test_support::bound_model;
    use crate::runtimes::test_support::{bound_runtime, live_vl};

    #[rstest]
    #[case::none_reads_as_one(0, 1, 1)]
    #[case::one(1, 1, 1)]
    #[case::two(2, 2, 1)]
    #[case::decode256(1, 1, 256)]
    fn the_request_asks_the_cells_tokens_of_at_least_one_image(
        #[case] num_images: u32,
        #[case] image_parts: usize,
        #[case] decode_tokens: u32,
    ) -> anyhow::Result<()> {
        let body = request_body(
            Path::new("/models/vl"),
            "hi",
            "AAAA",
            num_images,
            decode_tokens,
            &[],
        )?;
        assert_eq!(body["max_tokens"], decode_tokens);
        let images = body["messages"][0]["content"]
            .as_array()
            .context("the message has no content parts")?
            .iter()
            .filter(|part| part["type"] == "image_url")
            .count();
        assert_eq!(images, image_parts);
        Ok(())
    }

    /// The interpreter is an empty file, so reaching the tokenizer would fail
    /// with a spawn error instead.
    #[test]
    fn an_image_out_of_range_is_refused_before_the_tokenizer_loads() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let benchmark = VlMaxMemoryBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_max_memory_16384x16384_text0_img1")?,
            parameter_image_width: PositiveCount::try_new(16_384)?,
            parameter_image_height: PositiveCount::try_new(16_384)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: None,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        };
        let err = run(
            &benchmark,
            &ResolvedMlxPython::try_from(&bound_runtime(tmp.path())?)?,
            &ResolvedMlxModel::try_from(&bound_model(tmp.path(), false)?)?,
            MeasurementObserver::QUIET,
            None,
        )
        .err()
        .context("an oversized image was measured")?;
        assert!(format!("{err:#}").contains("out of range"), "{err:#}");
        Ok(())
    }

    /// Needs Apple silicon, a venv holding `mlx-vlm` and an MLX VL model:
    ///
    /// ```text
    /// PIPETTE_MLX_LIVE_PYTHON=<venv>/bin/python PIPETTE_MLX_LIVE_TARGET=<dir> \
    ///   cargo test -p pipette-mlx --lib live_vl_max_memory -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn live_vl_max_memory_through_mlx_vlm_server() -> anyhow::Result<()> {
        let (venv_python, model_dir) = live_vl()?;
        let benchmark = VlMaxMemoryBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_max_memory_512x512_text16_img2")?,
            parameter_image_width: PositiveCount::try_new(512)?,
            parameter_image_height: PositiveCount::try_new(512)?,
            parameter_text_tokens: NonNegativeCount::try_new(16)?,
            parameter_decode_tokens: None,
            parameter_num_images: NonNegativeCount::try_new(2)?,
        };
        let response = run(
            &benchmark,
            &venv_python,
            &model_dir,
            MeasurementObserver::QUIET,
            None,
        )?;
        eprintln!("{:?}", response.result_data);
        let weights = std::fs::read_dir(&*model_dir)?
            .map(|entry| -> anyhow::Result<u64> {
                let path = entry?.path();
                let safetensors = path.extension().is_some_and(|ext| ext == "safetensors");
                Ok(if safetensors {
                    path.metadata()?.len()
                } else {
                    0
                })
            })
            .sum::<anyhow::Result<u64>>()?;
        let BenchmarkResultData::MaxMemoryUsage {
            max_host_bytes,
            max_gpu_bytes: None,
            max_npu_bytes: None,
        } = response.result_data
        else {
            anyhow::bail!("not a host-only peak: {:?}", response.result_data);
        };
        anyhow::ensure!(
            max_host_bytes > weights,
            "a peak of {max_host_bytes} bytes is below the {weights} bytes of weights"
        );
        Ok(())
    }
}
