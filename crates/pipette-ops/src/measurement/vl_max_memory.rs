//! VL peak memory: the most memory one model and projector load and one image
//! workload held, read once per cell (`docs/methodology/vl-max-memory.md`).
//! Reports under [`super::max_memory`]'s wire result.

use anyhow::Context;

use pipette_plan_types::{BenchmarkResultData, ByteCount, VlMaxMemoryBenchmark};

use super::max_memory::{PEAK_MEMORY_MEASURED, PEAK_MEMORY_WARM_UP};
use super::{AttemptError, MeasurementObserver, MemoryMeasurementWarmup};
use crate::generation::Generation;

/// A reading a VL peak-memory cell cannot use.
#[derive(Debug, thiserror::Error)]
pub enum VlMaxMemoryMeasurementError {
    #[error("{field} is 0: a peak reads zero only when no read landed")]
    ZeroPeak { field: &'static str },
    #[error("{field} is {bytes}, past the largest byte count a result holds")]
    TooLarge { field: &'static str, bytes: u64 },
}

/// What the attempt read, as the engine reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlMaxMemoryMeasurementReading {
    pub max_host_bytes: u64,
    /// Where the engine probes a GPU pool: PDH on Windows, the KGSL mappings on
    /// Android's Snapdragon build. `None` elsewhere, Apple's unified memory
    /// included, where the host figure already holds it.
    pub max_gpu_bytes: Option<u64>,
    /// Where the engine probes an NPU pool: PDH on Windows, the dmabufs on
    /// Android's Snapdragon build.
    pub max_npu_bytes: Option<u64>,
}

/// Measure a VL peak-memory cell: one attempt, which loads the model and its
/// projector, runs `parameter_num_images` images and `parameter_text_tokens`
/// text, and decodes `parameter_decode_tokens` (one when unset), and whose
/// peaks are the result.
///
/// One measured attempt, a warm-up before it only with
/// [`MemoryMeasurementWarmup::Once`], and no readiness gate, as in
/// [`super::max_memory`]. The attempt owns its checks: an engine that can
/// count the reply refuses one that did not run to the output length.
///
/// `ctx` is lent to the attempt, for what the engine records while it runs: the
/// server it sends to, or the process it ran. Lend a reference to read it back
/// afterwards, a failed or refused run included.
pub fn vl_max_memory<C>(
    benchmark: &VlMaxMemoryBenchmark,
    observer: MeasurementObserver<'_>,
    warmup: MemoryMeasurementWarmup,
    mut ctx: C,
    mut attempt: impl FnMut(&mut C) -> Result<VlMaxMemoryMeasurementReading, AttemptError>,
) -> anyhow::Result<BenchmarkResultData> {
    // Names the cell in every log line and error. TODO: add the job id.
    let marker = benchmark.benchmark_id.to_string();

    if warmup == MemoryMeasurementWarmup::Once {
        let label = format!("{marker} warm-up");
        (observer.on_attempt_start)(PEAK_MEMORY_WARM_UP);
        measure(&label, &mut ctx, &mut attempt).context(label)?;
    }
    (observer.on_attempt_start)(PEAK_MEMORY_MEASURED);
    measure(&marker, &mut ctx, &mut attempt).context(marker)
}

/// One attempt, its reading logged and then checked.
fn measure<C>(
    label: &str,
    ctx: &mut C,
    attempt: &mut impl FnMut(&mut C) -> Result<VlMaxMemoryMeasurementReading, AttemptError>,
) -> anyhow::Result<BenchmarkResultData> {
    tracing::info!(cell = label, "running");
    let reading = attempt(ctx).map_err(AttemptError::into_anyhow)?;
    // Before the refusals, so a rejected reading is in the log too.
    tracing::info!(
        cell = label,
        max_host_bytes = reading.max_host_bytes,
        max_gpu_bytes = reading.max_gpu_bytes,
        max_npu_bytes = reading.max_npu_bytes,
        "peak memory reading"
    );
    Ok(result_data(&reading)?)
}

/// The reply ran to the cell's output length, as the text cell's does, or the
/// row would be labelled with an output length that never ran.
pub fn decoded_in_full(generation: &Generation, decode_tokens: u32) -> anyhow::Result<()> {
    let counts = generation.counts_at_limit()?;
    super::expect_tokens("completion_tokens", counts.completion_tokens, decode_tokens)
}

fn result_data(
    reading: &VlMaxMemoryMeasurementReading,
) -> Result<BenchmarkResultData, VlMaxMemoryMeasurementError> {
    Ok(BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes: bytes("max_host_bytes", reading.max_host_bytes)?,
        max_gpu_bytes: pool("max_gpu_bytes", reading.max_gpu_bytes)?,
        max_npu_bytes: pool("max_npu_bytes", reading.max_npu_bytes)?,
    })
}

/// A peak on the wire, refused at zero and past what a [`ByteCount`] holds,
/// the server's bound on a byte count.
fn bytes(field: &'static str, peak: u64) -> Result<u64, VlMaxMemoryMeasurementError> {
    match peak {
        0 => Err(VlMaxMemoryMeasurementError::ZeroPeak { field }),
        bytes if ByteCount::try_from(bytes).is_err() => {
            Err(VlMaxMemoryMeasurementError::TooLarge { field, bytes })
        }
        bytes => Ok(bytes),
    }
}

/// [`bytes`] for a pool the engine may not probe.
fn pool(
    field: &'static str,
    peak: Option<u64>,
) -> Result<Option<u64>, VlMaxMemoryMeasurementError> {
    peak.map(|peak| bytes(field, peak)).transpose()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{BenchmarkId, NonNegativeCount, PositiveCount};

    use super::*;
    use crate::generation::{Counts, Finish};

    fn body() -> anyhow::Result<VlMaxMemoryBenchmark> {
        Ok(VlMaxMemoryBenchmark {
            benchmark_id: BenchmarkId::try_new("vl_max_memory_512x512_text0_img1")?,
            parameter_image_width: PositiveCount::try_new(512)?,
            parameter_image_height: PositiveCount::try_new(512)?,
            parameter_text_tokens: NonNegativeCount::try_new(0)?,
            parameter_decode_tokens: None,
            parameter_num_images: NonNegativeCount::try_new(1)?,
        })
    }

    fn reading(host: u64, gpu: Option<u64>, npu: Option<u64>) -> VlMaxMemoryMeasurementReading {
        VlMaxMemoryMeasurementReading {
            max_host_bytes: host,
            max_gpu_bytes: gpu,
            max_npu_bytes: npu,
        }
    }

    /// One past the largest byte count the server accepts.
    const TOO_LARGE: u64 = 1 << 63;

    /// The peaks pass through to the `max_memory_usage` result; a zero peak is
    /// no reading.
    #[rstest]
    #[case::host(reading(8, None, None), Ok((8, None, None)))]
    #[case::gpu_pool(reading(8, Some(4), None), Ok((8, Some(4), None)))]
    #[case::npu_pool(reading(8, None, Some(2)), Ok((8, None, Some(2))))]
    #[case::zero_host(reading(0, None, None), Err("max_host_bytes"))]
    #[case::zero_gpu(reading(8, Some(0), None), Err("max_gpu_bytes"))]
    #[case::zero_npu(reading(8, None, Some(0)), Err("max_npu_bytes"))]
    fn a_reading_becomes_the_peak(
        #[case] reading: VlMaxMemoryMeasurementReading,
        #[case] expected: Result<(u64, Option<u64>, Option<u64>), &str>,
    ) -> anyhow::Result<()> {
        match (
            vl_max_memory(
                &body()?,
                MeasurementObserver::QUIET,
                MemoryMeasurementWarmup::Cold,
                (),
                |()| Ok(reading),
            ),
            expected,
        ) {
            (Ok(result_data), Ok((host, gpu, npu))) => assert_eq!(
                result_data,
                BenchmarkResultData::MaxMemoryUsage {
                    max_host_bytes: host,
                    max_gpu_bytes: gpu,
                    max_npu_bytes: npu,
                }
            ),
            (Err(err), Err(field)) => {
                assert!(
                    format!("{err:#}").starts_with("vl_max_memory_512x512_text0_img1: "),
                    "{err:#}"
                );
                assert!(
                    matches!(
                        err.downcast_ref::<VlMaxMemoryMeasurementError>(),
                        Some(VlMaxMemoryMeasurementError::ZeroPeak { field: f }) if *f == field
                    ),
                    "{err:#}"
                );
            }
            (got, expected) => anyhow::bail!("expected {expected:?}, got {got:?}"),
        }
        Ok(())
    }

    /// A reply must run to the cell's length, one token included.
    #[rstest]
    #[case::one_token(1, Some(Finish::Limit), Some(1), true)]
    #[case::one_token_uncounted(1, Some(Finish::Limit), None, false)]
    #[case::in_full(4, Some(Finish::Limit), Some(4), true)]
    #[case::short(4, Some(Finish::Limit), Some(3), false)]
    #[case::stopped(4, Some(Finish::Stop), Some(4), false)]
    #[case::uncounted(4, Some(Finish::Limit), None, false)]
    fn a_reply_decodes_in_full(
        #[case] decode_tokens: u32,
        #[case] finish: Option<Finish>,
        #[case] completion_tokens: Option<u32>,
        #[case] accepted: bool,
    ) {
        let generation = Generation {
            first_token_ms: 1.0,
            last_token_ms: 2.0,
            end_ms: 2.0,
            finish,
            counts: completion_tokens.map(|completion_tokens| Counts {
                prompt_tokens: 8,
                completion_tokens,
            }),
            server_clock: None,
        };
        assert_eq!(
            decoded_in_full(&generation, decode_tokens).is_ok(),
            accepted
        );
    }

    /// The largest byte count passes; one more is refused, naming its field.
    #[rstest]
    #[case::host(reading(TOO_LARGE, None, None), "max_host_bytes")]
    #[case::gpu_pool(reading(8, Some(TOO_LARGE), None), "max_gpu_bytes")]
    #[case::npu_pool(reading(8, None, Some(TOO_LARGE)), "max_npu_bytes")]
    fn a_peak_past_the_wire_range_is_refused(
        #[case] too_large: VlMaxMemoryMeasurementReading,
        #[case] field: &str,
    ) -> anyhow::Result<()> {
        let fits = reading(TOO_LARGE - 1, Some(TOO_LARGE - 1), Some(TOO_LARGE - 1));
        assert!(
            vl_max_memory(
                &body()?,
                MeasurementObserver::QUIET,
                MemoryMeasurementWarmup::Cold,
                (),
                |()| Ok(fits)
            )
            .is_ok()
        );

        let Err(err) = vl_max_memory(
            &body()?,
            MeasurementObserver::QUIET,
            MemoryMeasurementWarmup::Cold,
            (),
            |()| Ok(too_large),
        ) else {
            anyhow::bail!("a peak of {TOO_LARGE} bytes was accepted");
        };
        assert!(
            matches!(
                err.downcast_ref::<VlMaxMemoryMeasurementError>(),
                Some(VlMaxMemoryMeasurementError::TooLarge { field: f, bytes: TOO_LARGE }) if *f == field
            ),
            "{err:#}"
        );
        Ok(())
    }

    /// A failed attempt is the cell's error, named by the marker, and what it
    /// recorded before failing is still the caller's; the error keeps its type.
    #[test]
    fn a_failed_attempt_is_named_by_the_marker() -> anyhow::Result<()> {
        let mut recorded = Vec::new();
        let err = vl_max_memory(
            &body()?,
            MeasurementObserver::QUIET,
            MemoryMeasurementWarmup::Cold,
            &mut recorded,
            |recorded| {
                recorded.push("loaded");
                Err(anyhow::Error::from(std::io::Error::other("attempt failed")).into())
            },
        )
        .err()
        .context("a failed attempt became a peak")?;
        assert_eq!(
            format!("{err:#}"),
            "vl_max_memory_512x512_text0_img1: attempt failed"
        );
        assert!(
            err.chain().any(|cause| cause.is::<std::io::Error>()),
            "{err:#}"
        );
        assert_eq!(recorded, ["loaded"]);
        Ok(())
    }

    /// A warm-up runs the attempt first; its reading is checked, then dropped
    /// for the measured one's. Each starts an attempt of its own.
    #[rstest]
    #[case::cold(MemoryMeasurementWarmup::Cold, vec![8], &[PEAK_MEMORY_MEASURED])]
    #[case::once(
        MemoryMeasurementWarmup::Once,
        vec![16, 8],
        &[PEAK_MEMORY_WARM_UP, PEAK_MEMORY_MEASURED]
    )]
    fn a_warm_up_runs_the_attempt_first(
        #[case] warmup: MemoryMeasurementWarmup,
        #[case] expected: Vec<u64>,
        #[case] attempts: &[&str],
    ) -> anyhow::Result<()> {
        let started = std::sync::Mutex::new(Vec::new());
        let observer = MeasurementObserver {
            on_attempt_start: &|what| {
                started
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(what.to_owned());
            },
            ..MeasurementObserver::QUIET
        };
        let mut peaks = expected.clone().into_iter();
        let mut read = Vec::new();
        let result_data = vl_max_memory(&body()?, observer, warmup, &mut read, |read| {
            let peak = peaks.next().context("the attempt ran once too often")?;
            read.push(peak);
            Ok(reading(peak, None, None))
        })?;
        let started = started
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        anyhow::ensure!(started == attempts, "{started:?}");
        assert_eq!(read, expected);
        assert_eq!(
            result_data,
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: 8,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            }
        );
        Ok(())
    }

    /// A failed or refused warm-up is the cell's error, named as the warm-up,
    /// and the measured attempt never runs. What the warm-up recorded is kept,
    /// so the engine can quote the server whose reading was refused.
    #[rstest]
    #[case::failed(
        Err("attempt failed"),
        "vl_max_memory_512x512_text0_img1 warm-up: attempt failed"
    )]
    #[case::refused(
        Ok(0),
        "vl_max_memory_512x512_text0_img1 warm-up: max_host_bytes is 0: a peak reads zero only when no \
         read landed"
    )]
    fn a_failed_warm_up_skips_the_measured_attempt(
        #[case] warm_up: Result<u64, &'static str>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let mut recorded = Vec::new();
        let err = vl_max_memory(
            &body()?,
            MeasurementObserver::QUIET,
            MemoryMeasurementWarmup::Once,
            &mut recorded,
            |recorded| {
                recorded.push("warm-up");
                match warm_up {
                    Ok(peak) => Ok(reading(peak, None, None)),
                    Err(message) => Err(anyhow::anyhow!(message).into()),
                }
            },
        )
        .err()
        .context("a failed warm-up was measured")?;
        assert_eq!(format!("{err:#}"), expected);
        assert_eq!(recorded, ["warm-up"]);
        Ok(())
    }
}
