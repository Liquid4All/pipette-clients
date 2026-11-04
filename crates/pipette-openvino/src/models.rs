//! The IR directory of a bound OpenVINO model, resolved from the plan's
//! [`Model`] and checked to be complete.
//!
//! Install is `pipette_artifacts::ensure_model`.
//!
//! Nothing here decides whether a model may run on a device: a wrong
//! support-matrix guard produces no number rather than a bad one, and the
//! plan, not a release, decides which pairs to attempt. `docs/openvino-ir.md`
//! ("Why there are no preflight guards") has the measurements.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::{Model, ModelType};

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

/// A bound `openvino` model's IR directory, holding the pipeline's required
/// files.
#[derive(Debug)]
pub struct ResolvedOpenvinoModel(PathBuf);

impl TryFrom<&Model> for ResolvedOpenvinoModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        // Flattened rather than `.context`, which would put the advice in front
        // of the missing filename, the part that identifies the problem.
        require_bound_model_dir(model, ModelType::Openvino, REQUIRED_FILES)
            .map(Self)
            .map_err(|e| {
                anyhow::anyhow!(
                    "{e}; an IR export needs the model and the tokenizer/detokenizer pair \
                     (re-export with openvino-tokenizers installed)"
                )
            })
    }
}

impl Deref for ResolvedOpenvinoModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AbsolutePath, ModelSource, Openvino};

    use super::*;

    fn ir_model(model_dir: &Path) -> anyhow::Result<Model> {
        Ok(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model_dir.to_string_lossy().into_owned())?,
            },
        }))
    }

    /// The tokenizer pair is the omission people actually hit, so the error has
    /// to name the file rather than say the directory is incomplete.
    #[test]
    fn resolved_openvino_model_names_the_missing_tokenizer() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(tmp.path().join("openvino_model.xml"), "<net/>")?;
        std::fs::write(tmp.path().join("openvino_model.bin"), [0u8])?;
        let Err(err) = ResolvedOpenvinoModel::try_from(&ir_model(tmp.path())?) else {
            anyhow::bail!("expected a missing-tokenizer rejection");
        };
        assert!(
            err.to_string().contains("openvino_tokenizer.xml"),
            "got {err}"
        );
        Ok(())
    }

    #[test]
    fn a_complete_directory_is_accepted() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        REQUIRED_FILES
            .iter()
            .try_for_each(|f| std::fs::write(tmp.path().join(f), [0u8]))?;
        let resolved = ResolvedOpenvinoModel::try_from(&ir_model(tmp.path())?)?;
        assert_eq!(resolved.to_path_buf(), tmp.path());
        Ok(())
    }
}
