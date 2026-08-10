//! `LLMPipeline` properties for one cell.
//!
//! Derived from the benchmark shape and the device rather than authored, for
//! now: the values that matter for correctness follow from the cell, and a
//! default that is wrong fails the run rather than skewing it.
//!
//! Defaults first, then the cell's authored overrides on top. The derivation
//! is what makes a cell correct without anyone having to know the device's
//! quirks; the overrides are for when someone does.
//!
//! Two steps, as every backend does it: [`resolve_runtime_flags`] settles what
//! the cell runs with in the plan's own typed form, and [`pipeline_properties`]
//! renders that into what `LLMPipeline` accepts. The resolved flags are also
//! what the result records, so the record and the pipeline cannot disagree.

use serde_json::{Map, Value};

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{
    BenchmarkType, OpenvinoDevice, OpenvinoGenerateHint, RuntimeFlagRef, RuntimeFlags,
};

/// GenAI's default response reservation. A cell generating more than this on
/// the NPU must raise it or the static shape truncates the run.
const DEFAULT_MIN_RESPONSE_LEN: u32 = 128;

/// GenAI's default prompt bound on NPU. Not raised by default: the standard
/// suite fits inside it, and raising it has costs `docs/openvino-ir.md`
/// records. A cell may still override it.
const NPU_MAX_PROMPT_LEN: u32 = 1024;

/// The flags this cell runs with: the plan's entry plus every value this client
/// derives for it. The result is the variant a plan authors, so it round-trips
/// to the record as what the run actually applied.
///
/// Only the NPU takes any: CPU and GPU are dynamic-shape, so the static-shape
/// settings are meaningless there. An authored value is dropped rather than
/// carried on those devices — it never reaches the pipeline, and a record
/// claiming it would describe a run that did not happen.
///
/// The named NPU only. A custom device may or may not reach an NPU, and
/// guessing from its string would be exactly the support-matrix belief the
/// `models` module argues against — so it gets the dynamic-shape treatment and
/// refuses at compile if that is wrong.
pub fn resolve_runtime_flags(req: &RunRequest) -> anyhow::Result<RuntimeFlags> {
    let mut r = req.runtime_flags_ref()?;
    if crate::runtimes::require_openvino_device(req)? != OpenvinoDevice::Npu {
        r.max_prompt_len = None;
        r.min_response_len = None;
        r.generate_hint = None;
        return RuntimeFlags::try_from(r).map_err(anyhow::Error::from);
    }

    let (prompt_tokens, response_tokens) = cell_shape(req)?;
    // An authored bound is the author's call, including the costs the constant
    // documents; only the default is defended here.
    let bound = r.max_prompt_len.unwrap_or(NPU_MAX_PROMPT_LEN);
    // The prompt has to fit the static bound. Re-scope the cell rather than
    // raising the bound — see NPU_MAX_PROMPT_LEN.
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
    r.min_response_len = r
        .min_response_len
        .or_else(|| (response_tokens > DEFAULT_MIN_RESPONSE_LEN).then_some(response_tokens));
    r.generate_hint = r.generate_hint.or(Some(OpenvinoGenerateHint::BestPerf));
    RuntimeFlags::try_from(r).map_err(anyhow::Error::from)
}

/// `LLMPipeline` properties for resolved flags.
///
/// Reads the flat [`RuntimeFlagRef`] form rather than matching each cell's
/// variant, so all four OpenVINO cells share one renderer.
pub fn pipeline_properties(flags: &RuntimeFlags) -> Map<String, Value> {
    let r = RuntimeFlagRef::from(flags.clone());
    let mut props = Map::new();
    if let Some(len) = r.max_prompt_len {
        props.insert("MAX_PROMPT_LEN".to_owned(), Value::from(len));
    }
    if let Some(len) = r.min_response_len {
        props.insert("MIN_RESPONSE_LEN".to_owned(), Value::from(len));
    }
    if let Some(hint) = r.generate_hint {
        props.insert("GENERATE_HINT".to_owned(), Value::from(hint.as_property()));
    }
    props
}

/// `(prompt_tokens, response_tokens)` for the cell — what the static shape has
/// to accommodate.
fn cell_shape(req: &RunRequest) -> anyhow::Result<(u32, u32)> {
    Ok(match req.benchmark.benchmark_type() {
        BenchmarkType::PrefillThroughput => (
            req.benchmark
                .as_prefill_throughput()
                .map_err(anyhow::Error::from)?
                .parameter_prefill_tokens,
            1,
        ),
        BenchmarkType::DecodeThroughput => {
            let b = req
                .benchmark
                .as_decode_throughput()
                .map_err(anyhow::Error::from)?;
            (b.parameter_prefill_tokens, b.parameter_decode_tokens)
        }
        BenchmarkType::EndToEndLatency => {
            let b = req
                .benchmark
                .as_end_to_end_latency()
                .map_err(anyhow::Error::from)?;
            (b.parameter_prefill_tokens, b.parameter_decode_tokens)
        }
        BenchmarkType::MaxMemoryUsage => (
            req.benchmark
                .as_max_memory_usage()
                .map_err(anyhow::Error::from)?
                .parameter_prefill_tokens,
            1,
        ),
        other => anyhow::bail!("{other:?} cells are not supported for OpenVINO"),
    })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{
        BenchmarkDefinition, DecodeThroughput, EndToEndLatency, PrefillThroughput,
    };
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{
        AbsolutePath, Model, ModelSource, ModelType, NonEmptyString, Openvino, Runtime,
        RuntimeFlagRef, RuntimeType, UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    };

    use super::*;

    /// A cell on `device`, with the device carried where it now lives: the
    /// runtime is the artifact and names none.
    fn req(device: OpenvinoDevice, benchmark: BenchmarkDefinition) -> anyhow::Result<RunRequest> {
        req_with_flags(device, benchmark, |_| {})
    }

    /// The same, with `edit` applied to the flags before they are typed — for
    /// the cases that author a setting alongside the device.
    fn req_with_flags(
        device: OpenvinoDevice,
        benchmark: BenchmarkDefinition,
        edit: impl FnOnce(&mut RuntimeFlagRef),
    ) -> anyhow::Result<RunRequest> {
        let runtime = Runtime::UvOpenvino(UvOpenvino {
            server_version: UvServerVersion::try_new("2026.2.1".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("openvino-genai==2026.2.1.0\n".to_owned())?,
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
        edit(&mut flags);
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: Some(RuntimeFlags::try_from(flags)?),
            model_flags: None,
            benchmark_flags: None,
            benchmark,
        })
    }

    /// A cell that authors all three static-shape settings, each different from
    /// what the derivation would pick — so a value surviving proves the author's
    /// won, not that the two happened to agree.
    fn authored_overrides(flags: &mut RuntimeFlagRef) {
        flags.max_prompt_len = Some(2048);
        flags.min_response_len = Some(512);
        flags.generate_hint = Some(OpenvinoGenerateHint::FastCompile);
    }

    /// The properties half, for the cases that only care about what reached the
    /// pipeline.
    fn properties_of(req: &RunRequest) -> anyhow::Result<Map<String, Value>> {
        Ok(pipeline_properties(&resolve_runtime_flags(req)?))
    }

    /// The resolved flags as a flat ref, which is how a record consumer reads
    /// them back.
    fn reported(req: &RunRequest) -> anyhow::Result<RuntimeFlagRef> {
        Ok(RuntimeFlagRef::from(resolve_runtime_flags(req)?))
    }

    fn e2e(prefill: u32, decode: u32) -> BenchmarkDefinition {
        BenchmarkDefinition::EndToEndLatency(EndToEndLatency {
            benchmark_id: format!("end_to_end_latency_{prefill}_{decode}"),
            parameter_prefill_tokens: prefill,
            parameter_decode_tokens: decode,
        })
    }

    #[test]
    fn cpu_and_gpu_take_no_properties() -> anyhow::Result<()> {
        [OpenvinoDevice::Cpu, OpenvinoDevice::Gpu]
            .into_iter()
            .try_for_each(|device| {
                let props = properties_of(&req(device.clone(), e2e(512, 256))?)?;
                assert!(props.is_empty(), "{device:?} got {props:?}");
                anyhow::Ok(())
            })?;
        Ok(())
    }

    /// The case the default gets wrong: 256 output tokens against GenAI's
    /// 128-token reservation would truncate on NPU.
    #[test]
    fn npu_raises_the_response_reservation_when_the_cell_needs_it() -> anyhow::Result<()> {
        let props = properties_of(&req(OpenvinoDevice::Npu, e2e(512, 256))?)?;
        assert_eq!(props.get("GENERATE_HINT"), Some(&Value::from("BEST_PERF")));
        assert_eq!(props.get("MIN_RESPONSE_LEN"), Some(&Value::from(256)));
        // Left at the default, deliberately.
        assert!(props.get("MAX_PROMPT_LEN").is_none());
        Ok(())
    }

    #[test]
    fn npu_leaves_the_reservation_alone_when_the_default_suffices() -> anyhow::Result<()> {
        let decode = BenchmarkDefinition::DecodeThroughput(DecodeThroughput {
            benchmark_id: "decode_throughput_512_100".into(),
            parameter_prefill_tokens: 512,
            parameter_decode_tokens: 100,
        });
        let props = properties_of(&req(OpenvinoDevice::Npu, decode)?)?;
        assert!(props.get("MIN_RESPONSE_LEN").is_none(), "{props:?}");
        Ok(())
    }

    /// Re-scope the cell rather than raising the bound: the message has to say
    /// so, because raising it is the tempting wrong fix.
    #[test]
    fn npu_rejects_a_prompt_over_the_static_bound() -> anyhow::Result<()> {
        let prefill = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_2048".into(),
            parameter_prefill_tokens: 2048,
        });
        let Err(err) = properties_of(&req(OpenvinoDevice::Npu, prefill)?) else {
            anyhow::bail!("expected an over-bound rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains("2048"), "got {msg}");
        assert!(msg.contains("cpu/gpu"), "got {msg}");
        Ok(())
    }
    /// An authored value wins over the derivation, including where the
    /// derivation would have chosen differently.
    #[test]
    fn authored_flags_override_the_derived_defaults() -> anyhow::Result<()> {
        let mut req = req(OpenvinoDevice::Npu, e2e(512, 256))?;
        req.runtime_flags = Some(RuntimeFlags::EndToEndUvOpenvinoOpenvino {
            device: Some(OpenvinoDevice::Npu),
            max_prompt_len: Some(2048),
            min_response_len: Some(512),
            generate_hint: Some(OpenvinoGenerateHint::BestPerf),
        });
        let props = properties_of(&req)?;
        assert_eq!(props.get("MAX_PROMPT_LEN"), Some(&Value::from(2048)));
        // 512 authored, not the 256 the cell shape would have derived.
        assert_eq!(props.get("MIN_RESPONSE_LEN"), Some(&Value::from(512)));
        assert_eq!(props.get("GENERATE_HINT"), Some(&Value::from("BEST_PERF")));
        Ok(())
    }

    /// Raising the bound is the author's call — the guard defends the default,
    /// not the author's judgement.
    #[test]
    fn an_authored_bound_admits_a_prompt_the_default_would_reject() -> anyhow::Result<()> {
        let over = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_2048".into(),
            parameter_prefill_tokens: 2048,
        });
        let mut req = req(OpenvinoDevice::Npu, over)?;
        assert!(properties_of(&req).is_err(), "default should refuse");

        req.runtime_flags = Some(RuntimeFlags::PrefillUvOpenvinoOpenvino {
            device: Some(OpenvinoDevice::Npu),
            max_prompt_len: Some(4096),
            min_response_len: None,
            generate_hint: None,
        });
        let props = properties_of(&req)?;
        assert_eq!(props.get("MAX_PROMPT_LEN"), Some(&Value::from(4096)));
        Ok(())
    }

    /// Flags are NPU-shaped; CPU and GPU are dynamic-shape and take none, even
    /// when a cell authors them.
    #[test]
    fn authored_flags_still_do_not_reach_cpu() -> anyhow::Result<()> {
        let mut req = req(OpenvinoDevice::Cpu, e2e(512, 256))?;
        req.runtime_flags = Some(RuntimeFlags::EndToEndUvOpenvinoOpenvino {
            device: Some(OpenvinoDevice::Cpu),
            max_prompt_len: Some(2048),
            min_response_len: Some(512),
            generate_hint: None,
        });
        assert!(properties_of(&req)?.is_empty());
        Ok(())
    }

    /// The reservation the engine chose is the number an NPU result has to be
    /// read against, and the cell authored none of it — so the record is the
    /// only place it can come from.
    #[test]
    fn the_record_reports_the_derived_reservation() -> anyhow::Result<()> {
        let r = reported(&req(OpenvinoDevice::Npu, e2e(512, 256))?)?;
        assert_eq!(r.min_response_len, Some(256));
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
        let req = req_with_flags(OpenvinoDevice::Cpu, e2e(512, 256), authored_overrides)?;
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
        let req = req_with_flags(OpenvinoDevice::Npu, e2e(512, 256), authored_overrides)?;
        let flags = resolve_runtime_flags(&req)?;
        let props = pipeline_properties(&flags);
        let r = RuntimeFlagRef::from(flags);

        assert_eq!(r.max_prompt_len, Some(2048));
        assert_eq!(r.min_response_len, Some(512));
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
