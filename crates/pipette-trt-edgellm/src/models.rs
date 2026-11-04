//! The GGUF pair of a bound `gguf_vision` model, the weights the server
//! converts and builds its engines from. `pipette-llamacpp`'s check, copied: no
//! runtime crate uses another.

use std::path::PathBuf;

use pipette_plan_types::{GgufVision, GgufVisionSource, Model};

#[derive(Debug)]
pub struct ResolvedGgufVisionModel {
    pub backbone: PathBuf,
    pub mmproj: PathBuf,
}

impl TryFrom<&Model> for ResolvedGgufVisionModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        let Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(files),
        }) = model
        else {
            anyhow::bail!(
                "expected bound GgufVision AbsoluteFiles (after ensure_model/bind_under), got \
                 {model}"
            );
        };
        Ok(Self {
            backbone: require_file(PathBuf::from(files.model.as_ref()), "GGUF vision model")?,
            mmproj: require_file(PathBuf::from(files.mmproj.as_ref()), "GGUF mmproj")?,
        })
    }
}

fn require_file(path: PathBuf, what: &str) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        path.is_file(),
        "bound {what} is not a file: {} (ensure/bind may be incomplete)",
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_projector_is_named() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = tmp.path().join("model.gguf");
        std::fs::write(&model, b"gguf")?;
        let bound: Model = serde_json::from_value(serde_json::json!({
            "type": "gguf_vision", "source": "absolute_files",
            "model": model, "mmproj": tmp.path().join("mmproj.gguf"),
        }))?;
        let Err(err) = ResolvedGgufVisionModel::try_from(&bound) else {
            anyhow::bail!("a missing mmproj was accepted");
        };
        assert!(err.to_string().contains("GGUF mmproj"), "{err}");
        Ok(())
    }
}
