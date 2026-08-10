//! End-to-end latency: prefill plus the full decode, as one number.
//!
//! Reconstructed from the driver's metrics (`ttft + tpot × (n-1)`) rather than
//! taken from the process wall clock, which would fold in the pipeline compile
//! — ~1s on CPU and ~18s on NPU against a workload of a similar order. The
//! compile is a property of the one-shot process model, not of the model's
//! latency, so it must not land in the number.
//!
//! The one cell that measures a *text* prompt. Latency is what a caller waits
//! for, and a caller sends text, so tokenizing it belongs inside the number —
//! the same reason llama.cpp, MLX and torch-oai all send a string here and
//! token ids everywhere else. Sizing that text to an exact token count needs
//! the model's tokenizer, which those three get from their running server;
//! this backend has no server, so it opens one tokenizer-only driver per cell
//! (see [`driver::TokenizeSession`]) and runs the same shared convergence loop
//! against it.

use anyhow::Context;

use pipette_ops::measurement;
use pipette_ops::prompt_seed::{self, PROMPT_SEED_TEXT};
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::result::BenchmarkResultData;
use pipette_plan_types::run::{RunRequest, RunResponse};

use super::driver::{self, DriverRequest, Mode};
use super::{invoke, take_output, Cell, LastOutput};

pub(super) fn run(
    req: &RunRequest,
    compile_cache: &std::path::Path,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
) -> anyhow::Result<RunResponse> {
    let benchmark = req
        .benchmark
        .as_end_to_end_latency()
        .map_err(anyhow::Error::from)?;
    let prefill_tokens = benchmark.parameter_prefill_tokens;
    let decode_tokens = benchmark.parameter_decode_tokens;
    let cell = Cell::bind(req, compile_cache)?;
    let model_dir = cell.model_dir_str()?.to_owned();
    let properties = cell.properties();

    // Built once per cell, before the reps: the text is fixed for the whole
    // series, and the tokenizer that sizes it must not be alive while a
    // measured rep runs.
    let prompt = {
        let mut tokenizer = driver::TokenizeSession::start(&cell.script, &cell.python, &model_dir)?;
        prompt_seed::build_prompt_text(prefill_tokens, |text| tokenizer.count(text))
            .context("sizing the end-to-end prompt")?
    };

    let last = LastOutput::default();

    let measured = measurement::run(
        "openvino/end_to_end_latency",
        readiness_gate,
        observer,
        |_| Ok(()),
        |_| {
            invoke(
                &cell,
                &DriverRequest {
                    model_dir: &model_dir,
                    device: crate::runtimes::device_property(&cell.device),
                    mode: Mode::Decode,
                    prefill_tokens,
                    decode_tokens,
                    prompt: Some(&prompt),
                    warmup: Some(cell.warmup(prefill_tokens, decode_tokens)),
                    properties: properties.clone(),
                    prompt_seed: PROMPT_SEED_TEXT,
                },
                &last,
            )
        },
        |_, rep| {
            measurement::expect_tokens("input_tokens", rep.value.input_tokens, prefill_tokens)?;
            measurement::expect_tokens(
                "generated_tokens",
                rep.value.generated_tokens,
                decode_tokens,
            )?;
            total_ms(rep.value.ttft_ms, rep.value.tpot_ms, decode_tokens)
        },
    )?;
    let stats = measured.stats();
    let (stdout, stderr) = take_output(last);
    Ok(cell.respond(
        BenchmarkResultData::EndToEndLatency {
            total_time_ms: stats.mean_ms,
            total_time_ms_stddev: Some(stats.stddev_ms),
        },
        stdout,
        stderr,
    ))
}

/// `ttft + tpot × (decode_tokens - 1)`.
///
/// TTFT already covers prefill *and* the first generated token, so the
/// remaining `n - 1` tokens are what TPOT accounts for. A single-token cell
/// therefore needs no TPOT at all, which is also the only case where the driver
/// omits it.
fn total_ms(ttft_ms: f64, tpot_ms: Option<f64>, decode_tokens: u32) -> anyhow::Result<f64> {
    let ttft = measurement::positive_finite("ttft_ms", ttft_ms)?;
    if decode_tokens <= 1 {
        return Ok(ttft);
    }
    let tpot = tpot_ms.ok_or_else(|| {
        anyhow::anyhow!("driver reported no tpot_ms for a {decode_tokens}-token cell")
    })?;
    Ok(ttft + measurement::positive_finite("tpot_ms", tpot)? * f64::from(decode_tokens - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_is_ttft_plus_the_remaining_tokens() -> anyhow::Result<()> {
        // 100ms to the first token, then 99 more at 10ms each.
        assert_eq!(total_ms(100.0, Some(10.0), 100)?, 100.0 + 990.0);
        Ok(())
    }

    /// The first token is inside TTFT, so a one-token cell is TTFT exactly —
    /// and needs no TPOT, which is the case the driver omits.
    #[test]
    fn a_single_token_cell_is_just_ttft() -> anyhow::Result<()> {
        assert_eq!(total_ms(100.0, None, 1)?, 100.0);
        Ok(())
    }

    #[test]
    fn a_multi_token_cell_without_tpot_is_an_error() {
        assert!(total_ms(100.0, None, 100).is_err());
        assert!(total_ms(100.0, Some(f64::NAN), 100).is_err());
        assert!(total_ms(0.0, Some(10.0), 100).is_err());
    }
}
