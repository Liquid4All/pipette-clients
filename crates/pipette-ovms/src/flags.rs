//! The device and the NPU static-shape settings resolve as they do for
//! `uv_openvino`: OVMS runs the same `openvino_genai` pipeline on the same
//! devices, so the NPU limits in `docs/openvino-ir.md` apply unchanged. The
//! rules are `pipette-openvino`'s `flags.rs`, copied, since no runtime crate
//! uses another: a change to them belongs in both.
//! `prefix_caching` is OVMS's own, and resolves to off when the cell leaves it
//! unset, because OVMS defaults it on and a cache hit reads as a fast prefill.

use anyhow::Context;
use serde_json::{Map, Value};

use pipette_plan_types::{
    BenchmarkType, OpenvinoDevice, OpenvinoGenerateHint, PositiveCount, RunRequest, RuntimeFlagRef,
    RuntimeFlags,
};

/// GenAI's default response reservation. A cell generating more than this on
/// the NPU must raise it or the static shape truncates the run.
const DEFAULT_MIN_RESPONSE_LEN: u32 = 128;

/// GenAI's default prompt bound on NPU. Not raised by default: the standard
/// suite fits inside it, and raising it has costs `docs/openvino-ir.md`
/// records. A cell may still override it.
const NPU_MAX_PROMPT_LEN: u32 = 1024;

/// The flags this cell runs with, derived values included. An unset
/// `prefix_caching` resolves to `false`: a record without it would describe a
/// cache-on run as unconfigured.
///
/// Only the named NPU takes the static-shape settings; elsewhere an authored
/// value is dropped, so the record never claims a setting the server did not get.
pub fn resolve_runtime_flags(req: &RunRequest) -> anyhow::Result<RuntimeFlagRef> {
    let mut r = req.runtime_flags_ref()?;
    r.prefix_caching = Some(r.prefix_caching.unwrap_or(false));
    let device = r.device.clone().context(
        "this OVMS cell names no `device`; set one on the cell's runtime_flags \
         (`cpu`, `gpu`, `npu`, or a device OpenVINO resolves such as `GPU.1`)",
    )?;
    if device != OpenvinoDevice::Npu {
        r.max_prompt_len = None;
        r.min_response_len = None;
        r.generate_hint = None;
        return Ok(r);
    }

    let (prompt_tokens, response_tokens) = cell_shape(req)?;
    // An authored bound is the author's call, including the costs the constant
    // documents; only the default is defended here.
    let bound = r
        .max_prompt_len
        .map_or(NPU_MAX_PROMPT_LEN, |n| n.into_inner().unsigned_abs());
    if prompt_tokens > bound {
        anyhow::bail!(
            "this cell prompts with {prompt_tokens} tokens, over the NPU's \
             {bound}-token static bound. Raising it costs superlinear compile \
             time and hits unresolved behaviour above 1024 \
             (docs/openvino-ir.md); run this cell on cpu/gpu instead, or set \
             `max_prompt_len` on the cell if you accept those costs."
        );
    }

    // Only raise the response reservation when the cell needs more than GenAI's
    // default; leaving it alone keeps the compiled shape as small as possible.
    if r.min_response_len.is_none() && response_tokens > DEFAULT_MIN_RESPONSE_LEN {
        r.min_response_len = Some(PositiveCount::try_from(response_tokens)?);
    }
    r.generate_hint = r.generate_hint.or(Some(OpenvinoGenerateHint::BestPerf));
    Ok(r)
}

/// `(prompt_tokens, response_tokens)` for the cell: what the static shape has
/// to accommodate.
fn cell_shape(req: &RunRequest) -> anyhow::Result<(u32, u32)> {
    Ok(match req.benchmark.benchmark_type() {
        BenchmarkType::PrefillThroughput => (
            req.benchmark
                .as_prefill_throughput()
                .map_err(anyhow::Error::from)?
                .parameter_prefill_tokens
                .into_inner()
                .unsigned_abs(),
            1,
        ),
        BenchmarkType::DecodeThroughput => {
            let b = req
                .benchmark
                .as_decode_throughput()
                .map_err(anyhow::Error::from)?;
            (
                b.parameter_prefill_tokens.into_inner().unsigned_abs(),
                b.parameter_decode_tokens.into_inner().unsigned_abs(),
            )
        }
        BenchmarkType::EndToEndLatency => {
            let b = req
                .benchmark
                .as_end_to_end_latency()
                .map_err(anyhow::Error::from)?;
            (
                b.parameter_prefill_tokens.into_inner().unsigned_abs(),
                b.parameter_decode_tokens.into_inner().unsigned_abs(),
            )
        }
        BenchmarkType::MaxMemoryUsage => (
            req.benchmark
                .as_max_memory_usage()
                .map_err(anyhow::Error::from)?
                .parameter_prefill_tokens
                .into_inner()
                .unsigned_abs(),
            1,
        ),
        other @ (BenchmarkType::Eval
        | BenchmarkType::VlThroughput
        | BenchmarkType::VlMaxMemory
        | BenchmarkType::VlmEval
        | BenchmarkType::AsrEval
        | BenchmarkType::VlSpecDecode
        | BenchmarkType::AsrThroughput
        | BenchmarkType::AsrMaxMemory) => {
            anyhow::bail!("{other:?} cells are not supported for OVMS")
        }
    })
}

/// A custom device is passed verbatim: `GPU.1`, `AUTO` and `HETERO:GPU,CPU`
/// are OpenVINO's own spellings.
pub(crate) fn device_property(device: &OpenvinoDevice) -> &str {
    match device {
        OpenvinoDevice::Cpu => "CPU",
        OpenvinoDevice::Gpu => "GPU",
        OpenvinoDevice::Npu => "NPU",
        OpenvinoDevice::Custom(device) => device,
    }
}

/// `--max_prompt_len` is OVMS's own flag; the other NPU properties travel in
/// `--plugin_config`, which OVMS merges into the pipeline's properties beside
/// it.
pub(crate) fn server_args(flags: &RuntimeFlags) -> anyhow::Result<Vec<String>> {
    let r = RuntimeFlagRef::from(flags.clone());
    let device = r.device.as_ref().context("resolved flags name no device")?;
    let mut args = vec![
        "--target_device".to_owned(),
        device_property(device).to_owned(),
        "--enable_prefix_caching".to_owned(),
        r.prefix_caching
            .context("resolved flags carry no prefix_caching")?
            .to_string(),
    ];
    if let Some(len) = r.max_prompt_len {
        args.extend(["--max_prompt_len".to_owned(), len.to_string()]);
    }
    let mut plugin_config = Map::new();
    if let Some(len) = r.min_response_len {
        plugin_config.insert("MIN_RESPONSE_LEN".to_owned(), Value::from(len.into_inner()));
    }
    if let Some(hint) = r.generate_hint {
        plugin_config.insert("GENERATE_HINT".to_owned(), Value::from(hint.as_property()));
    }
    if !plugin_config.is_empty() {
        args.extend([
            "--plugin_config".to_owned(),
            Value::Object(plugin_config).to_string(),
        ]);
    }
    Ok(args)
}

/// Empty at the reference batch size, so its recorded command is unchanged.
/// `LM_CB` makes OVMS refuse a device it cannot batch rather than fall back to
/// its serial servable.
pub(crate) fn batch_args(slots: u32) -> Vec<String> {
    if slots <= 1 {
        return Vec::new();
    }
    [
        "--pipeline_type",
        "LM_CB",
        "--max_num_seqs",
        &slots.to_string(),
    ]
    .map(str::to_owned)
    .to_vec()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkId, DecodeThroughputBenchmark, ModelType,
        NonNegativeCount, OpenvinoDevice, OpenvinoGenerateHint, PositiveCount,
        PrefillThroughputBenchmark, RuntimeType,
    };

    use super::*;
    use crate::test_support::{request, unbound};

    fn decode(decode_tokens: u32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::DecodeThroughput(
            DecodeThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "decode_throughput_512_{decode_tokens}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
                parameter_decode_tokens: PositiveCount::try_from(decode_tokens)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn prefill(prefill_tokens: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new(format!("prefill_throughput_{prefill_tokens}"))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill_tokens)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    fn req_for(
        benchmark: BenchmarkDefinition,
        edit: impl FnOnce(&mut RuntimeFlagRef) -> anyhow::Result<()>,
    ) -> anyhow::Result<RunRequest> {
        let mut flags = RuntimeFlagRef::new(
            benchmark.benchmark_type(),
            RuntimeType::Ovms,
            ModelType::Openvino,
        );
        edit(&mut flags)?;
        request(unbound(), Some(RuntimeFlags::try_from(flags)?), benchmark)
    }

    /// A 512-prompt, 256-token decode cell.
    fn req(
        edit: impl FnOnce(&mut RuntimeFlagRef) -> anyhow::Result<()>,
    ) -> anyhow::Result<RunRequest> {
        req_for(decode(256)?, edit)
    }

    fn args_of(req: &RunRequest) -> anyhow::Result<Vec<String>> {
        server_args(&RuntimeFlags::try_from(resolve_runtime_flags(req)?)?)
    }

    /// Parsed: key order follows serde_json's features, not this code.
    fn plugin_config(args: &[String]) -> anyhow::Result<Value> {
        let at = args
            .iter()
            .position(|arg| arg == "--plugin_config")
            .context("no --plugin_config")?;
        let value = args.get(at + 1).context("--plugin_config has no value")?;
        Ok(serde_json::from_str(value)?)
    }

    /// Each value differs from what the derivation would pick, so one surviving
    /// proves the author's won.
    fn authored_on(
        device: OpenvinoDevice,
    ) -> impl FnOnce(&mut RuntimeFlagRef) -> anyhow::Result<()> {
        move |f| {
            f.device = Some(device);
            f.max_prompt_len = Some(PositiveCount::try_new(2048)?);
            f.min_response_len = Some(PositiveCount::try_new(512)?);
            f.generate_hint = Some(OpenvinoGenerateHint::FastCompile);
            Ok(())
        }
    }

    #[rstest]
    #[case::unset(None, "false")]
    #[case::off(Some(false), "false")]
    #[case::on(Some(true), "true")]
    fn prefix_caching_resolves_to_off_unless_authored(
        #[case] authored: Option<bool>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let req = req(|f| {
            f.device = Some(OpenvinoDevice::Gpu);
            f.prefix_caching = authored;
            Ok(())
        })?;
        let flags = RuntimeFlags::try_from(resolve_runtime_flags(&req)?)?;
        assert_eq!(
            RuntimeFlagRef::from(flags.clone()).prefix_caching,
            Some(expected == "true")
        );
        assert_eq!(
            server_args(&flags)?,
            [
                "--target_device",
                "GPU",
                "--enable_prefix_caching",
                expected
            ]
        );
        Ok(())
    }

    /// The NPU takes the static-shape settings: `--max_prompt_len` only when
    /// set, and in `--plugin_config` the hint plus the reservation once the cell
    /// outgrows GenAI's own 128 tokens.
    #[rstest]
    #[case::default_bound(256, None, &[], json!({ "GENERATE_HINT": "BEST_PERF", "MIN_RESPONSE_LEN": 256 }))]
    #[case::authored_bound(256, Some(512), &["--max_prompt_len", "512"], json!({ "GENERATE_HINT": "BEST_PERF", "MIN_RESPONSE_LEN": 256 }))]
    #[case::default_reservation(100, None, &[], json!({ "GENERATE_HINT": "BEST_PERF" }))]
    fn npu_carries_the_static_shape(
        #[case] decode_tokens: u32,
        #[case] max_prompt_len: Option<i32>,
        #[case] bound_args: &[&str],
        #[case] expected_plugin_config: Value,
    ) -> anyhow::Result<()> {
        let args = args_of(&req_for(decode(decode_tokens)?, |f| {
            f.device = Some(OpenvinoDevice::Npu);
            f.max_prompt_len = max_prompt_len.map(PositiveCount::try_new).transpose()?;
            Ok(())
        })?)?;
        let Some((_, head)) = args.split_last() else {
            anyhow::bail!("no args");
        };
        let expected: Vec<&str> = ["--target_device", "NPU", "--enable_prefix_caching", "false"]
            .into_iter()
            .chain(bound_args.iter().copied())
            .chain(["--plugin_config"])
            .collect();
        assert_eq!(head, expected);
        assert_eq!(plugin_config(&args)?, expected_plugin_config);
        Ok(())
    }

    /// Re-scope the cell rather than raising the bound: the message has to say
    /// so, because raising it is the tempting wrong fix.
    #[test]
    fn npu_rejects_a_prompt_over_the_static_bound() -> anyhow::Result<()> {
        let req = req_for(prefill(2048)?, |f| {
            f.device = Some(OpenvinoDevice::Npu);
            Ok(())
        })?;
        let Err(err) = resolve_runtime_flags(&req) else {
            anyhow::bail!("expected an over-bound rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains("2048"), "{msg}");
        assert!(msg.contains("cpu/gpu"), "{msg}");
        Ok(())
    }

    #[test]
    fn an_authored_bound_admits_a_prompt_the_default_would_reject() -> anyhow::Result<()> {
        let args = args_of(&req_for(prefill(2048)?, |f| {
            f.device = Some(OpenvinoDevice::Npu);
            f.max_prompt_len = Some(PositiveCount::try_new(4096)?);
            Ok(())
        })?)?;
        assert!(
            args.windows(2).any(|w| w == ["--max_prompt_len", "4096"]),
            "{args:?}"
        );
        Ok(())
    }

    #[test]
    fn an_authored_override_reaches_the_record_and_the_server() -> anyhow::Result<()> {
        let req = req(authored_on(OpenvinoDevice::Npu))?;
        let flags = RuntimeFlags::try_from(resolve_runtime_flags(&req)?)?;
        let r = RuntimeFlagRef::from(flags.clone());
        assert_eq!(r.max_prompt_len.map(i32::from), Some(2048));
        assert_eq!(r.min_response_len.map(i32::from), Some(512));
        assert_eq!(r.generate_hint, Some(OpenvinoGenerateHint::FastCompile));
        let args = server_args(&flags)?;
        assert!(
            args.windows(2).any(|w| w == ["--max_prompt_len", "2048"]),
            "{args:?}"
        );
        assert_eq!(
            plugin_config(&args)?,
            serde_json::json!({ "GENERATE_HINT": "FAST_COMPILE", "MIN_RESPONSE_LEN": 512 })
        );
        Ok(())
    }

    /// CPU and GPU are dynamic-shape, and a custom device may not reach an NPU.
    #[rstest]
    #[case::cpu(OpenvinoDevice::Cpu, "CPU")]
    #[case::gpu(OpenvinoDevice::Gpu, "GPU")]
    #[case::custom(OpenvinoDevice::Custom("GPU.1".to_owned()), "GPU.1")]
    fn only_the_npu_takes_the_npu_settings(
        #[case] device: OpenvinoDevice,
        #[case] spelled: &str,
    ) -> anyhow::Result<()> {
        let req = req(authored_on(device))?;
        let r = RuntimeFlagRef::from(RuntimeFlags::try_from(resolve_runtime_flags(&req)?)?);
        assert_eq!(
            (r.max_prompt_len, r.min_response_len, r.generate_hint),
            (None, None, None)
        );
        assert_eq!(
            args_of(&req)?,
            [
                "--target_device",
                spelled,
                "--enable_prefix_caching",
                "false"
            ]
        );
        Ok(())
    }

    #[rstest]
    #[case::reference(1, &[])]
    #[case::batched(4, &["--pipeline_type", "LM_CB", "--max_num_seqs", "4"])]
    fn only_a_batch_reaches_the_argv(#[case] slots: u32, #[case] expected: &[&str]) {
        assert_eq!(batch_args(slots), expected);
    }

    #[test]
    fn a_cell_naming_no_device_is_refused() -> anyhow::Result<()> {
        let Err(err) = resolve_runtime_flags(&req(|_| Ok(()))?) else {
            anyhow::bail!("expected a missing-device rejection");
        };
        assert!(err.to_string().contains("names no `device`"), "{err}");
        Ok(())
    }
}
