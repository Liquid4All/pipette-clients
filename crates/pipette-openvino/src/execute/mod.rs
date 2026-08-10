//! OpenVINO execute: kind dispatch from a prepared [`RunRequest`].
//!
//! Every kind follows the same shape — bind the venv python, the model dir and
//! the declared device; then hand
//! `prepare`/`work`/`sample` to [`pipette_ops::measurement::run`], which owns
//! the rep count, the readiness wait, the observer reports and the reduction.
//!
//! Each rep is a fresh driver process holding one compiled pipeline. That is
//! the hardware's constraint rather than a style choice: compiling more than
//! once per process took an NPU down mid-run (`docs/openvino-ir.md`). Every rep
//! therefore constructs its own pipeline, but a blob cache warmed before the
//! first one keeps that to a load rather than a compile — see
//! `docs/openvino-measurement.md`.

mod driver;
mod end_to_end_latency;
mod max_memory_usage;
mod throughput;

use anyhow::Context;

use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_ops::EvalCompletionsStore;
use pipette_plan_types::run::{RunRequest, RunResponse};
use pipette_plan_types::BenchmarkType;

/// Top-level OpenVINO dispatch: route a prepared [`RunRequest`] by kind.
///
/// CLI owns prepare/record; this crate only runs the cell. `eval_completions`
/// is unused — eval is not implemented for this backend — but the parameter
/// keeps the engine seam identical to the other backends'.
pub fn run(
    req: &RunRequest,
    _eval_completions: &EvalCompletionsStore,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
    compile_cache: &std::path::Path,
) -> anyhow::Result<RunResponse> {
    match req.benchmark.benchmark_type() {
        BenchmarkType::PrefillThroughput => {
            throughput::run_prefill(req, compile_cache, readiness_gate, observer)
        }
        BenchmarkType::DecodeThroughput => {
            throughput::run_decode(req, compile_cache, readiness_gate, observer)
        }
        BenchmarkType::EndToEndLatency => {
            end_to_end_latency::run(req, compile_cache, readiness_gate, observer)
        }
        BenchmarkType::MaxMemoryUsage => max_memory_usage::run(req, compile_cache),
        BenchmarkType::Eval => {
            anyhow::bail!("eval benchmarks are not yet supported for OpenVINO")
        }
        BenchmarkType::VlThroughput => {
            anyhow::bail!("VL throughput benchmarks are not yet supported for OpenVINO")
        }
    }
}

/// The per-cell context every kind binds before measuring: where python is,
/// where the model is, and which device the cell declared.
struct Cell {
    python: std::path::PathBuf,
    model_dir: std::path::PathBuf,
    device: pipette_plan_types::OpenvinoDevice,
    /// What this cell runs with: the plan's entry plus every value this client
    /// derived for it. Resolved once here rather than per kind, so the
    /// properties the pipeline gets and the flags the record reports cannot
    /// come from two different resolutions.
    flags: pipette_plan_types::RuntimeFlags,
    /// Where OpenVINO keeps compiled blobs. Shared across cells and runs, so a
    /// cell whose (model, device, properties) has been compiled before starts
    /// warm — see `docs/openvino-measurement.md`.
    compile_cache: std::path::PathBuf,
    script: driver::DriverScript,
}

impl Cell {
    /// Bind: resolve the venv python, the model directory and the declared
    /// device. No device/model compatibility is decided here — a runtime that
    /// cannot run a pairing reports that itself, and which pairings are worth
    /// attempting is the plan's call.
    fn bind(req: &RunRequest, compile_cache: &std::path::Path) -> anyhow::Result<Self> {
        let python = crate::runtimes::require_openvino_python(req)?;
        let model_dir = crate::models::require_openvino_model_dir(req)?;
        let device = crate::runtimes::require_openvino_device(req)?;
        let flags = crate::flags::resolve_runtime_flags(req)?;
        std::fs::create_dir_all(compile_cache).with_context(|| {
            format!(
                "creating the OpenVINO compile cache at {}",
                compile_cache.display()
            )
        })?;
        let cell = Self {
            python,
            model_dir,
            device,
            flags,
            compile_cache: compile_cache.to_path_buf(),
            script: driver::DriverScript::materialize()?,
        };
        cell.precompile()?;
        Ok(cell)
    }

    /// Compile once, untimed, so every measured rep loads a blob instead of the
    /// first one compiling and the rest loading. A rep that met the device in a
    /// different state from its peers is the asymmetry this exists to avoid.
    fn precompile(&self) -> anyhow::Result<()> {
        self.script.precompile(
            &self.python,
            &driver::DriverRequest {
                model_dir: self.model_dir_str()?,
                device: crate::runtimes::device_property(&self.device),
                mode: driver::Mode::Compile,
                prefill_tokens: 0,
                decode_tokens: 0,
                warmup: None,
                prompt: None,
                properties: self.properties(),
                prompt_seed: pipette_ops::prompt_seed::PROMPT_SEED_TEXT,
            },
        )
    }

    /// The `LLMPipeline` properties for this cell: the resolved flags, plus the
    /// blob cache.
    ///
    /// `CACHE_DIR` is added here rather than in `flags.rs` because it is this
    /// host's scratch location, not something the cell declared — it must not
    /// reach the recorded flags.
    fn properties(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut props = crate::flags::pipeline_properties(&self.flags);
        props.insert(
            "CACHE_DIR".to_owned(),
            serde_json::Value::from(self.compile_cache.to_string_lossy().into_owned()),
        );
        props
    }

    /// The response for a finished cell: the result, what ran, and the flags it
    /// ran with.
    ///
    /// The only constructor, so a benchmark kind cannot report a number without
    /// saying which flags produced it — the omission this exists to prevent, and
    /// one every kind would otherwise have to remember separately.
    fn respond(
        &self,
        data: pipette_plan_types::result::BenchmarkResultData,
        stdout: String,
        stderr: String,
    ) -> RunResponse {
        let (executable, command) = self.invocation();
        RunResponse {
            executable,
            command,
            runtime_flags: Some(self.flags.clone()),
            ..RunResponse::new(data, stdout, stderr)
        }
    }

    fn model_dir_str(&self) -> anyhow::Result<&str> {
        self.model_dir
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))
    }

    /// The warm-up pass for this cell: the measured shape, on every device.
    ///
    /// No device is exempt. The NPU was, until it was measured — see
    /// `docs/openvino-measurement.md`.
    fn warmup(&self, prefill_tokens: u32, decode_tokens: u32) -> driver::WarmupShape {
        driver::WarmupShape {
            prefill_tokens,
            decode_tokens,
        }
    }

    /// What the record says ran: `(executable, command)`.
    ///
    /// The executable is the venv interpreter, which is what identifies the
    /// install this cell used. The preview names the driver script rather than
    /// its path — the script is a per-run temp copy, so the path would record a
    /// different string every run — and carries no arguments, because the
    /// driver's request travels on stdin.
    fn invocation(&self) -> (Option<String>, Vec<String>) {
        let python = self.python.display().to_string();
        (
            Some(python.clone()),
            vec![python, driver::DRIVER_FILENAME.to_owned()],
        )
    }
}

/// The last invocation's captured output, for the `RunResponse`.
///
/// A `RefCell` because `measurement::run` holds `work` and `sample` at once, so
/// the output cannot be moved out of a `&mut` capture, and `Measurement` does
/// not expose its reps.
type LastOutput = std::cell::RefCell<driver::DriverOutput>;

/// One measured invocation: run the driver, remember what it printed, hand back
/// the parsed result.
fn invoke(
    cell: &Cell,
    request: &driver::DriverRequest<'_>,
    last: &LastOutput,
) -> anyhow::Result<driver::DriverResult> {
    let (result, output) = cell.script.invoke(&cell.python, request)?;
    *last.borrow_mut() = output;
    Ok(result)
}

fn take_output(last: LastOutput) -> (String, String) {
    let out = last.into_inner();
    (out.stdout, out.stderr)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::result::BenchmarkResultData;
    use pipette_plan_types::{
        BenchmarkType, ModelType, OpenvinoDevice, OpenvinoGenerateHint, RuntimeFlagRef,
        RuntimeFlags, RuntimeType,
    };

    use super::*;

    /// A cell built directly rather than through `bind`, which would need a
    /// real venv and model directory on disk to reach the same two fields.
    fn cell(flags: RuntimeFlags) -> anyhow::Result<Cell> {
        Ok(Cell {
            python: std::path::PathBuf::from("/venv/bin/python"),
            model_dir: std::path::PathBuf::from("/models/ir"),
            device: OpenvinoDevice::Npu,
            flags,
            compile_cache: std::path::PathBuf::from("/cache/uv-openvino__key"),
            script: driver::DriverScript::materialize()?,
        })
    }

    fn resolved_flags() -> anyhow::Result<RuntimeFlags> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::PrefillThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.device = Some(OpenvinoDevice::Npu);
        r.generate_hint = Some(OpenvinoGenerateHint::BestPerf);
        Ok(RuntimeFlags::try_from(r)?)
    }

    /// `CACHE_DIR` reaches the pipeline but never the record: it is this host's
    /// scratch path, and a record carrying it would describe local disk layout
    /// as if the cell had asked for it.
    #[test]
    fn the_cache_reaches_the_pipeline_but_not_the_record() -> anyhow::Result<()> {
        let cell = cell(resolved_flags()?)?;
        let props = cell.properties();
        assert!(props.contains_key("CACHE_DIR"), "got {props:?}");

        let reported = cell.respond(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: 1,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
            String::new(),
            String::new(),
        );
        let json = serde_json::to_string(&reported.runtime_flags)?;
        assert!(!json.contains("CACHE_DIR"), "got {json}");
        assert!(!json.contains("cache"), "got {json}");
        Ok(())
    }

    /// The derived flags reach the record. Without this the run is still
    /// correct and the number is still right — it is the record that silently
    /// stops explaining which settings produced it.
    #[test]
    fn a_response_reports_the_flags_the_cell_resolved() -> anyhow::Result<()> {
        let flags = resolved_flags()?;
        let response = cell(flags.clone())?.respond(
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes: 1,
                max_gpu_bytes: None,
                max_npu_bytes: None,
            },
            String::new(),
            String::new(),
        );
        assert_eq!(response.runtime_flags, Some(flags));
        // The venv interpreter is what identifies the install this cell used.
        assert_eq!(
            response.executable.as_deref(),
            Some("/venv/bin/python"),
            "got {:?}",
            response.executable
        );
        Ok(())
    }
}
