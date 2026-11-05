//! The GGUF files of a bound llama.cpp model, resolved from the plan's
//! [`Model`].
//!
//! Install is `pipette_artifacts::ensure_model`.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use pipette_plan_types::{
    GgufText, GgufTextSource, GgufVision, GgufVisionAbsoluteFiles, GgufVisionSource,
    GgufVisionSpeculative, GgufVisionSpeculativeSource, Model,
};

/// A bound GGUF text model's weights.
#[derive(Debug)]
pub struct ResolvedGgufTextModel(PathBuf);

impl TryFrom<&Model> for ResolvedGgufTextModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        match model {
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path },
            }) => require_file(PathBuf::from(path.as_ref()), "GGUF text model").map(Self),
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleCoreAi(_)
            | Model::AppleFoundationText) => anyhow::bail!(
                "expected bound GgufText AbsoluteFile (after ensure_model/bind_under), got {other}"
            ),
        }
    }
}

impl Deref for ResolvedGgufTextModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A bound `gguf_vision` model's files.
#[derive(Debug)]
pub struct ResolvedGgufVisionModel {
    pub backbone: PathBuf,
    pub mmproj: PathBuf,
}

impl TryFrom<&Model> for ResolvedGgufVisionModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        match model {
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::AbsoluteFiles(files),
            }) => vision_files(files),
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleCoreAi(_)
            | Model::AppleFoundationText) => anyhow::bail!(
                "expected bound GgufVision AbsoluteFiles (after ensure_model/bind_under), got {other}"
            ),
        }
    }
}

/// A bound `gguf_vision_speculative` model's files.
#[derive(Debug)]
pub struct ResolvedGgufVisionSpeculativeModel {
    pub vision: ResolvedGgufVisionModel,
    /// `None` when the drafter is merged into the backbone: DSpark's release
    /// ships one gguf holding both.
    pub draft: Option<PathBuf>,
}

impl TryFrom<&Model> for ResolvedGgufVisionSpeculativeModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        match model {
            Model::GgufVisionSpeculative(GgufVisionSpeculative {
                source: GgufVisionSpeculativeSource::AbsoluteFiles { files, draft },
            }) => Ok(Self {
                vision: vision_files(files)?,
                draft: draft
                    .as_ref()
                    .map(|draft| require_file(PathBuf::from(draft.as_ref()), "GGUF drafter"))
                    .transpose()?,
            }),
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleCoreAi(_)
            | Model::AppleFoundationText) => anyhow::bail!(
                "expected bound GgufVisionSpeculative AbsoluteFiles \
                 (after ensure_model/bind_under), got {other}"
            ),
        }
    }
}

fn vision_files(files: &GgufVisionAbsoluteFiles) -> anyhow::Result<ResolvedGgufVisionModel> {
    Ok(ResolvedGgufVisionModel {
        backbone: require_file(PathBuf::from(files.model.as_ref()), "GGUF vision model")?,
        mmproj: require_file(PathBuf::from(files.mmproj.as_ref()), "GGUF mmproj")?,
    })
}

fn require_file(path: PathBuf, what: &str) -> anyhow::Result<PathBuf> {
    if !path.is_file() {
        anyhow::bail!(
            "bound {what} is not a file: {} (ensure/bind may be incomplete)",
            path.display()
        );
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use pipette_plan_types::AbsolutePath;

    use super::*;

    /// A bound vision model of `kind` whose files exist under `dir`.
    fn vision_model(dir: &Path, kind: &str, separate_drafter: bool) -> anyhow::Result<Model> {
        let file = |name: &str| -> anyhow::Result<PathBuf> {
            let path = dir.join(name);
            std::fs::write(&path, b"gguf")?;
            Ok(path)
        };
        let mut spec = json!({
            "type": kind,
            "source": "absolute_files",
            "model": file("model.gguf")?,
            "mmproj": file("mmproj.gguf")?,
        });
        if separate_drafter {
            spec["draft"] = json!(file("draft.gguf")?);
        }
        Ok(serde_json::from_value(spec)?)
    }

    #[test]
    fn resolved_gguf_text_model_rejects_wrong_variant() -> anyhow::Result<()> {
        let model = Model::Mlx(pipette_plan_types::Mlx {
            source: pipette_plan_types::ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/m".to_owned())?,
            },
        });
        assert!(ResolvedGgufTextModel::try_from(&model).is_err());
        Ok(())
    }

    #[rstest]
    #[case::present(true)]
    #[case::missing(false)]
    fn resolved_gguf_text_model_checks_the_file(#[case] present: bool) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("model.gguf");
        if present {
            std::fs::write(&path, b"gguf")?;
        }
        let model: Model = serde_json::from_value(
            json!({"type": "gguf_text", "source": "absolute_file", "path": &path}),
        )?;
        match (present, ResolvedGgufTextModel::try_from(&model)) {
            (true, Ok(resolved)) => assert_eq!(resolved.to_path_buf(), path),
            (false, Err(e)) => assert!(format!("{e:#}").contains("is not a file"), "{e:#}"),
            (_, outcome) => anyhow::bail!("present={present}, got {outcome:?}"),
        }
        Ok(())
    }

    #[test]
    fn resolved_gguf_vision_model_takes_only_a_plain_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let resolved =
            ResolvedGgufVisionModel::try_from(&vision_model(tmp.path(), "gguf_vision", false)?)?;
        assert_eq!(
            (resolved.backbone, resolved.mmproj),
            (
                tmp.path().join("model.gguf"),
                tmp.path().join("mmproj.gguf")
            )
        );
        let speculative = vision_model(tmp.path(), "gguf_vision_speculative", false)?;
        assert!(ResolvedGgufVisionModel::try_from(&speculative).is_err());
        Ok(())
    }

    /// Only a drafter shipped as its own gguf fills `draft`.
    #[rstest]
    #[case::separate_drafter(true)]
    #[case::merged_drafter(false)]
    fn resolved_gguf_vision_speculative_model_takes_only_a_speculative_model(
        #[case] separate_drafter: bool,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = vision_model(tmp.path(), "gguf_vision_speculative", separate_drafter)?;
        let resolved = ResolvedGgufVisionSpeculativeModel::try_from(&model)?;
        assert_eq!(
            (resolved.vision.backbone, resolved.draft),
            (
                tmp.path().join("model.gguf"),
                separate_drafter.then(|| tmp.path().join("draft.gguf"))
            )
        );
        let plain = vision_model(tmp.path(), "gguf_vision", false)?;
        assert!(ResolvedGgufVisionSpeculativeModel::try_from(&plain).is_err());
        Ok(())
    }
}
