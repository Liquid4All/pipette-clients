//! `LLMPipeline` properties for one cell: defaults derived from the benchmark
//! shape and the device, so a cell is correct without its author knowing the
//! device's quirks, with the cell's authored overrides on top. A wrong default
//! fails the run rather than skewing it.
//!
//! [`resolve_runtime_flags`] settles what the cell runs with and
//! [`pipeline_properties`] renders that for `LLMPipeline`. The result records
//! the resolved flags, so the record and the pipeline cannot disagree.

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

/// The flags this cell runs with: the plan's entry plus every value this client
/// derives for it.
///
/// Only the named NPU takes the static-shape settings; CPU and GPU are
/// dynamic-shape. An authored value is dropped on those devices, since it
/// never reaches the pipeline and a record claiming it would describe a run
/// that did not happen. A custom device may or may not reach an NPU, and
/// guessing from its string would be the support-matrix belief `models`
/// rejects, so it gets the dynamic-shape treatment and refuses at compile if
/// that is wrong.
pub fn resolve_runtime_flags(req: &RunRequest) -> anyhow::Result<RuntimeFlagRef> {
    let mut r = req.runtime_flags_ref()?;
    if crate::runtimes::require_openvino_device(req)? != OpenvinoDevice::Npu {
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

/// `LLMPipeline` properties for resolved flags.
///
/// Reads the flat [`RuntimeFlagRef`] form rather than matching each cell's
/// variant, so all four OpenVINO cells share one renderer.
pub fn pipeline_properties(flags: &RuntimeFlags) -> Map<String, Value> {
    let r = RuntimeFlagRef::from(flags.clone());
    let mut props = Map::new();
    if let Some(len) = r.max_prompt_len {
        props.insert("MAX_PROMPT_LEN".to_owned(), Value::from(len.into_inner()));
    }
    if let Some(len) = r.min_response_len {
        props.insert("MIN_RESPONSE_LEN".to_owned(), Value::from(len.into_inner()));
    }
    if let Some(hint) = r.generate_hint {
        props.insert("GENERATE_HINT".to_owned(), Value::from(hint.as_property()));
    }
    props
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
        | BenchmarkType::VlmEval
        | BenchmarkType::AsrEval
        | BenchmarkType::VlThroughput
        | BenchmarkType::VlMaxMemory
        | BenchmarkType::VlSpecDecode
        | BenchmarkType::AsrThroughput
        | BenchmarkType::AsrMaxMemory) => {
            anyhow::bail!("{other:?} cells are not supported for OpenVINO")
        }
    })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        AbsolutePath, BatchSize, BenchmarkDefinition, BenchmarkId, DeclaredBound,
        DecodeThroughputBenchmark, EndToEndLatencyBenchmark, EndToEndLatencyUvOpenvinoOpenvino,
        Model, ModelSource, ModelType, NonEmptyString, NonNegativeCount, Openvino,
        PrefillThroughputBenchmark, PrefillThroughputUvOpenvinoOpenvino, Runtime, RuntimeType,
        UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    };

    use super::*;

    /// A cell on `device`, carried in its flags: the runtime names none.
    fn req(device: OpenvinoDevice, benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        req_with_flags(device, benchmark, |_| Ok(()))
    }

    /// The same, with `edit` applied to the flags before they are typed.
    fn req_with_flags(
        device: OpenvinoDevice,
        benchmark: BenchmarkDefinition,
        edit: impl FnOnce(&mut RuntimeFlagRef) -> anyhow::Result<()>,
    ) -> anyhow::Result<RunRequest> {
        let runtime = Runtime::UvOpenvino(UvOpenvino {
            server_version: UvServerVersion::try_new("2026.4.1".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("openvino-genai==2026.4.1.0\n".to_owned())?,
                install_flags: None,
            },
        });
        // A real OpenVINO model: the flag axes are read off the cell, so an
        // unrelated model would route the reported flags to another variant.
        let model = Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(if cfg!(windows) {
                    r"C:\tmp\ir".to_owned()
                } else {
                    "/tmp/ir".to_owned()
                })?,
            },
        });
        let mut flags = RuntimeFlagRef::new(
            benchmark.benchmark_type(),
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        flags.device = Some(device);
        edit(&mut flags)?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: Some(RuntimeFlags::try_from(flags)?),
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    /// All three static-shape settings, each different from what the
    /// derivation would pick, so a surviving value proves the author's won.
    fn authored_overrides(flags: &mut RuntimeFlagRef) -> anyhow::Result<()> {
        flags.max_prompt_len = Some(PositiveCount::try_new(2048)?);
        flags.min_response_len = Some(PositiveCount::try_new(512)?);
        flags.generate_hint = Some(OpenvinoGenerateHint::FastCompile);
        Ok(())
    }

    fn properties_of(req: &RunRequest) -> anyhow::Result<Map<String, Value>> {
        Ok(pipeline_properties(&RuntimeFlags::try_from(
            resolve_runtime_flags(req)?,
        )?))
    }

    fn reported(req: &RunRequest) -> anyhow::Result<RuntimeFlagRef> {
        Ok(RuntimeFlagRef::from(RuntimeFlags::try_from(
            resolve_runtime_flags(req)?,
        )?))
    }

    fn e2e(prefill: i32, decode: i32) -> anyhow::Result<BenchmarkDefinition> {
        Ok(BenchmarkDefinition::EndToEndLatency(
            EndToEndLatencyBenchmark {
                benchmark_id: BenchmarkId::try_new(format!(
                    "end_to_end_latency_{prefill}_{decode}"
                ))?,
                parameter_prefill_tokens: NonNegativeCount::try_new(prefill)?,
                parameter_decode_tokens: PositiveCount::try_new(decode)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))
    }

    #[test]
    fn cpu_and_gpu_take_no_properties() -> anyhow::Result<()> {
        [OpenvinoDevice::Cpu, OpenvinoDevice::Gpu]
            .into_iter()
            .try_for_each(|device| {
                let props = properties_of(&req(device.clone(), e2e(512, 256)?)?)?;
                assert!(props.is_empty(), "{device:?} got {props:?}");
                anyhow::Ok(())
            })?;
        Ok(())
    }

    /// The case the default gets wrong: 256 output tokens against GenAI's
    /// 128-token reservation would truncate on NPU.
    #[test]
    fn npu_raises_the_response_reservation_when_the_cell_needs_it() -> anyhow::Result<()> {
        let props = properties_of(&req(OpenvinoDevice::Npu, e2e(512, 256)?)?)?;
        assert_eq!(props.get("GENERATE_HINT"), Some(&Value::from("BEST_PERF")));
        assert_eq!(props.get("MIN_RESPONSE_LEN"), Some(&Value::from(256)));
        // Left at the default, deliberately.
        assert!(props.get("MAX_PROMPT_LEN").is_none());
        Ok(())
    }

    #[test]
    fn npu_leaves_the_reservation_alone_when_the_default_suffices() -> anyhow::Result<()> {
        let decode = BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("decode_throughput_512_100".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(512)?,
            parameter_decode_tokens: PositiveCount::try_new(100)?,
            parameter_batch_size: BatchSize::one(),
        });
        let props = properties_of(&req(OpenvinoDevice::Npu, decode)?)?;
        assert!(props.get("MIN_RESPONSE_LEN").is_none(), "{props:?}");
        Ok(())
    }

    /// The message has to steer to cpu/gpu: raising the bound is the tempting
    /// fix, and it has costs.
    #[test]
    fn npu_rejects_a_prompt_over_the_static_bound() -> anyhow::Result<()> {
        let prefill = BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_2048".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(2048)?,
            parameter_batch_size: BatchSize::one(),
        });
        let Err(err) = properties_of(&req(OpenvinoDevice::Npu, prefill)?) else {
            anyhow::bail!("expected an over-bound rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains("2048"), "got {msg}");
        assert!(msg.contains("cpu/gpu"), "got {msg}");
        Ok(())
    }
    #[test]
    fn authored_flags_override_the_derived_defaults() -> anyhow::Result<()> {
        let mut req = req(OpenvinoDevice::Npu, e2e(512, 256)?)?;
        req.runtime_flags = Some(RuntimeFlags::EndToEndLatencyUvOpenvinoOpenvino(
            EndToEndLatencyUvOpenvinoOpenvino {
                device: Some(OpenvinoDevice::Npu),
                max_prompt_len: Some(PositiveCount::try_new(2048)?),
                min_response_len: Some(PositiveCount::try_new(512)?),
                generate_hint: Some(OpenvinoGenerateHint::BestPerf),
            },
        ));
        let props = properties_of(&req)?;
        assert_eq!(props.get("MAX_PROMPT_LEN"), Some(&Value::from(2048)));
        // 512 authored, not the 256 the cell shape would have derived.
        assert_eq!(props.get("MIN_RESPONSE_LEN"), Some(&Value::from(512)));
        assert_eq!(props.get("GENERATE_HINT"), Some(&Value::from("BEST_PERF")));
        Ok(())
    }

    /// The guard defends the default bound, not an authored one.
    #[test]
    fn an_authored_bound_admits_a_prompt_the_default_would_reject() -> anyhow::Result<()> {
        let over = BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
            benchmark_id: BenchmarkId::try_new("prefill_throughput_2048".to_owned())?,
            parameter_prefill_tokens: NonNegativeCount::try_new(2048)?,
            parameter_batch_size: BatchSize::one(),
        });
        let mut req = req(OpenvinoDevice::Npu, over)?;
        assert!(properties_of(&req).is_err(), "default should refuse");

        req.runtime_flags = Some(RuntimeFlags::PrefillThroughputUvOpenvinoOpenvino(
            PrefillThroughputUvOpenvinoOpenvino {
                device: Some(OpenvinoDevice::Npu),
                max_prompt_len: Some(PositiveCount::try_new(4096)?),
                min_response_len: None,
                generate_hint: None,
            },
        ));
        let props = properties_of(&req)?;
        assert_eq!(props.get("MAX_PROMPT_LEN"), Some(&Value::from(4096)));
        Ok(())
    }

    #[test]
    fn authored_flags_still_do_not_reach_cpu() -> anyhow::Result<()> {
        let mut req = req(OpenvinoDevice::Cpu, e2e(512, 256)?)?;
        req.runtime_flags = Some(RuntimeFlags::EndToEndLatencyUvOpenvinoOpenvino(
            EndToEndLatencyUvOpenvinoOpenvino {
                device: Some(OpenvinoDevice::Cpu),
                max_prompt_len: Some(PositiveCount::try_new(2048)?),
                min_response_len: Some(PositiveCount::try_new(512)?),
                generate_hint: None,
            },
        ));
        assert!(properties_of(&req)?.is_empty());
        Ok(())
    }

    /// An NPU result has to be read against the reservation the engine chose,
    /// and the cell authored none of it, so only the record carries it.
    #[test]
    fn the_record_reports_the_derived_reservation() -> anyhow::Result<()> {
        let r = reported(&req(OpenvinoDevice::Npu, e2e(512, 256)?)?)?;
        assert_eq!(r.min_response_len.map(i32::from), Some(256));
        assert_eq!(r.generate_hint, Some(OpenvinoGenerateHint::BestPerf));
        // Left at GenAI's default, so nothing to report.
        assert_eq!(r.max_prompt_len, None);
        Ok(())
    }

    /// A cpu cell's authored settings never reach the pipeline, so the record
    /// must not carry them: the diff against the request is what says they were
    /// ignored.
    #[test]
    fn a_cpu_cell_reports_nothing_it_did_not_apply() -> anyhow::Result<()> {
        let req = req_with_flags(OpenvinoDevice::Cpu, e2e(512, 256)?, authored_overrides)?;
        let r = reported(&req)?;
        assert_eq!(r.max_prompt_len, None);
        assert_eq!(r.min_response_len, None);
        assert_eq!(r.generate_hint, None);
        Ok(())
    }

    /// The two renderings are one decision, so an authored override has to show
    /// up in both or the record describes a run that did not happen.
    #[test]
    fn the_record_and_the_pipeline_agree_on_an_authored_override() -> anyhow::Result<()> {
        let req = req_with_flags(OpenvinoDevice::Npu, e2e(512, 256)?, authored_overrides)?;
        let flags = RuntimeFlags::try_from(resolve_runtime_flags(&req)?)?;
        let props = pipeline_properties(&flags);
        let r = RuntimeFlagRef::from(flags);

        assert_eq!(r.max_prompt_len.map(i32::from), Some(2048));
        assert_eq!(r.min_response_len.map(i32::from), Some(512));
        assert_eq!(r.generate_hint, Some(OpenvinoGenerateHint::FastCompile));
        assert_eq!(props.get("MAX_PROMPT_LEN"), Some(&Value::from(2048)));
        assert_eq!(props.get("MIN_RESPONSE_LEN"), Some(&Value::from(512)));
        assert_eq!(
            props.get("GENERATE_HINT"),
            Some(&Value::from("FAST_COMPILE"))
        );
        Ok(())
    }
}
