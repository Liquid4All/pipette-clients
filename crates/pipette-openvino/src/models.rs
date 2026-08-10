//! Bound OpenVINO model projection: find the directory, check it is complete.
//!
//! Install is `pipette_artifacts::ensure_model`. This module no longer decides
//! whether a model may run on a device.
//!
//! It used to hold two such rules and both are gone. The precision rule refused
//! asymmetric weights on the NPU on the theory that they load and then generate
//! badly; measured, asymmetric int8 runs correctly there (slowly — 10.3 tok/s
//! against 95 for int4-sym, a result worth having rather than suppressing),
//! asymmetric int4 refuses at compile on its own, and the precision that
//! misbehaves quietly turned out to be a *symmetric* one. The mixture-of-experts
//! rule refused MoE on the NPU; measured since, that pairing refuses at compile
//! on its own ("illegal group-wise pattern" from the vpux compiler), so the
//! guard was only replacing one clear error with another.
//!
//! Both were support-matrix beliefs compiled into a measurement harness, and a
//! wrong one is expensive in a way that is hard to see: it does not produce a
//! bad number, it produces *no* number, and the gap looks like a limitation of
//! the hardware rather than of the guard. Which pairs are worth attempting
//! belongs to the plan, which is authored per campaign and can be changed
//! without a release.
//!
//! See `docs/openvino-ir.md` for what each precision actually does on the NPU.

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::ModelType;

/// Files `openvino_genai.LLMPipeline` needs. The tokenizer pair is the one
/// people miss: GenAI runs the tokenizer as a compiled model rather than
/// through the `tokenizers` library, so a directory without it loads and then
/// cannot generate.
const REQUIRED_FILES: &[&str] = &[
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
];

/// Bound OpenVINO model directory: `Model::Openvino` + `AbsoluteDir` after
/// ensure/bind, with the pipeline's required files present.
pub fn require_openvino_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    // Flattened rather than `.context`, which would put the advice in front of
    // the missing filename — and the filename is the part that identifies the
    // problem.
    require_bound_model_dir(&req.model.bound, ModelType::Openvino, REQUIRED_FILES).map_err(|e| {
        anyhow::anyhow!(
            "{e}; an IR export needs the model and the tokenizer/detokenizer pair \
             (re-export with openvino-tokenizers installed)"
        )
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use rstest::rstest;

    use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
    use pipette_plan_types::run::DeclaredBound;
    use pipette_plan_types::{AbsolutePath, Model, ModelSource, Openvino, OpenvinoDevice, Runtime};

    use super::*;

    fn stub_req(model_dir: &Path) -> anyhow::Result<RunRequest> {
        let model = Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model_dir.to_string_lossy().into_owned())?,
            },
        });
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "prefill".into(),
                parameter_prefill_tokens: 512,
            }),
        })
    }

    /// The tokenizer pair is the omission people actually hit, so the error has
    /// to name the file rather than say the directory is incomplete.
    #[test]
    fn require_model_dir_names_the_missing_tokenizer() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(tmp.path().join("openvino_model.xml"), "<net/>")?;
        std::fs::write(tmp.path().join("openvino_model.bin"), [0u8])?;
        let Err(err) = require_openvino_model_dir(&stub_req(tmp.path())?) else {
            anyhow::bail!("expected a missing-tokenizer rejection");
        };
        assert!(
            err.to_string().contains("openvino_tokenizer.xml"),
            "got {err}"
        );
        Ok(())
    }

    /// A complete directory is accepted whatever device the cell named — this
    /// module stopped deciding that.
    #[rstest]
    #[case::npu(OpenvinoDevice::Npu)]
    #[case::cpu(OpenvinoDevice::Cpu)]
    #[case::gpu(OpenvinoDevice::Gpu)]
    fn a_complete_directory_is_accepted_for_any_device(
        #[case] _device: OpenvinoDevice,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        REQUIRED_FILES
            .iter()
            .try_for_each(|f| std::fs::write(tmp.path().join(f), [0u8]))?;
        require_openvino_model_dir(&stub_req(tmp.path())?)?;
        Ok(())
    }
}
