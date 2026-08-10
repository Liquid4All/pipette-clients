//! Bound OpenVINO runtime projection for a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_runtime`. This module finds the venv
//! `python` under a bound `AbsolutePreinstalled` dir, and reads the compute
//! device off the **declared** runtime.

use std::path::PathBuf;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{OpenvinoDevice, RuntimeType};
use pipette_venv::{require_bound_venv, venv_python};

/// Bound OpenVINO venv python: `UvOpenvino` + `AbsolutePreinstalled` after bind.
pub fn require_openvino_python(req: &RunRequest) -> anyhow::Result<PathBuf> {
    let venv = require_bound_venv(&req.runtime.bound, &[RuntimeType::UvOpenvino])?;
    Ok(venv_python(&venv))
}

/// The device this cell runs on, from its flags.
///
/// A cell that names none is refused rather than defaulted: OpenVINO would
/// pick CPU on its own, and a CPU number filed under whatever the author meant
/// is worse than a run that does not start.
pub fn require_openvino_device(req: &RunRequest) -> anyhow::Result<OpenvinoDevice> {
    req.runtime_flags_ref()?.device.ok_or_else(|| {
        anyhow::anyhow!(
            "this OpenVINO cell names no `device`; set one on the cell's \
             runtime_flags (`cpu`, `gpu`, `npu`, or a device OpenVINO resolves \
             such as `GPU.1`)"
        )
    })
}

/// The `LLMPipeline` device string OpenVINO expects (`"CPU"`, `"GPU"`, `"NPU"`).
///
/// A custom device is passed verbatim: `GPU.1`, `AUTO` and `HETERO:GPU,CPU` are
/// OpenVINO's own spellings, so the author's string is the one the plugin has
/// to resolve.
pub fn device_property(device: &OpenvinoDevice) -> &str {
    match device {
        OpenvinoDevice::Cpu => "CPU",
        OpenvinoDevice::Gpu => "GPU",
        OpenvinoDevice::Npu => "NPU",
        OpenvinoDevice::Custom(device) => device,
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{
        AbsolutePath, Model, ModelSource, ModelType, Openvino, Runtime, RuntimeFlagRef,
        RuntimeFlags, RuntimeType, UvOpenvino, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    };

    use super::*;

    fn runtime(source: UvRuntimeSource) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvOpenvino(UvOpenvino {
            server_version: UvServerVersion::try_new("2026.2.1".to_string())?,
            python_version: UvPythonVersion::try_new("3.11".to_string())?,
            source,
        }))
    }

    fn openvino_model() -> anyhow::Result<Model> {
        Ok(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(if cfg!(windows) {
                    r"C:\tmp\ir".to_string()
                } else {
                    "/tmp/ir".to_string()
                })?,
            },
        }))
    }

    fn flags(device: Option<OpenvinoDevice>) -> anyhow::Result<Option<RuntimeFlags>> {
        let mut r = RuntimeFlagRef::new(
            pipette_plan_types::BenchmarkType::PrefillThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.device = device;
        Ok(Some(RuntimeFlags::try_from(r)?))
    }

    fn req(declared: Runtime, bound: Runtime) -> anyhow::Result<RunRequest> {
        req_with_device(declared, bound, flags(Some(OpenvinoDevice::Npu))?)
    }

    /// Fallible on purpose: a helper that swallowed a construction failure
    /// would report it as the cell naming no device, which is a different bug
    /// from the one that happened.
    fn req_with_device(
        declared: Runtime,
        bound: Runtime,
        runtime_flags: Option<RuntimeFlags>,
    ) -> anyhow::Result<RunRequest> {
        Ok(RunRequest {
            runtime: DeclaredBound { declared, bound },
            model: DeclaredBound::already_bound(openvino_model()?),
            runtime_flags,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "prefill".into(),
                parameter_prefill_tokens: 512,
            }),
        })
    }

    /// The device is the cell's, so binding cannot change it — the hazard that
    /// forced this to read the *declared* runtime is gone with the field.
    #[test]
    fn device_comes_from_the_cells_flags() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let declared = runtime(UvRuntimeSource::RelativePreinstalled {
            dir: pipette_plan_types::RelativePath::try_new("venv".to_string())?,
        })?;
        let bound = runtime(UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
        })?;
        assert_eq!(
            require_openvino_device(&req(declared, bound)?)?,
            OpenvinoDevice::Npu
        );
        Ok(())
    }

    /// OpenVINO would pick CPU on its own, so a cell that names no device is
    /// refused rather than filed under a silicon nobody chose.
    #[test]
    fn a_cell_naming_no_device_is_refused() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let rt = runtime(UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
        })?;
        let Err(err) = require_openvino_device(&req_with_device(rt.clone(), rt, None)?) else {
            anyhow::bail!("expected a missing-device rejection");
        };
        assert!(err.to_string().contains("names no `device`"), "got {err}");
        Ok(())
    }

    #[test]
    fn require_openvino_python_rejects_a_venv_without_python() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let bound = runtime(UvRuntimeSource::AbsolutePreinstalled {
            dir: AbsolutePath::try_new(tmp.path().to_string_lossy().into_owned())?,
        })?;
        let Err(err) = require_openvino_python(&req(bound.clone(), bound)?) else {
            anyhow::bail!("expected a missing-interpreter error");
        };
        assert!(format!("{err:#}").contains("python missing"), "got {err:#}");
        Ok(())
    }

    #[test]
    fn require_openvino_python_rejects_another_runtime() -> anyhow::Result<()> {
        let other = Runtime::AppleFoundation(Default::default());
        let Err(err) = require_openvino_python(&req(other.clone(), other)?) else {
            anyhow::bail!("expected a wrong-runtime rejection");
        };
        assert!(
            err.to_string().contains("expected uv_openvino"),
            "got {err}"
        );
        Ok(())
    }

    #[test]
    fn device_property_spells_devices_as_openvino_expects() {
        assert_eq!(device_property(&OpenvinoDevice::Cpu), "CPU");
        assert_eq!(device_property(&OpenvinoDevice::Gpu), "GPU");
        assert_eq!(device_property(&OpenvinoDevice::Npu), "NPU");
        // A custom device is the author's spelling, passed through untouched.
        assert_eq!(
            device_property(&OpenvinoDevice::Custom("GPU.1".to_owned())),
            "GPU.1"
        );
    }
}
