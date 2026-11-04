//! The directories of a bound torch model, resolved from the plan's [`Model`].
//!
//! Install is `pipette_artifacts::ensure_model`.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use pipette_ops::models::{require_bound_model_dir, require_bound_model_dir_pair};
use pipette_plan_types::{Model, ModelType};

/// Only `config.json` is universal (tokenizer layout varies by model), and it
/// is enough to catch a snapshot that stopped partway, which otherwise fails
/// inside the server at load time.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// A bound `torch` model's directory.
#[derive(Debug)]
pub struct ResolvedTorchModel(PathBuf);

impl TryFrom<&Model> for ResolvedTorchModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        require_bound_model_dir(model, ModelType::Torch, REQUIRED_FILES).map(Self)
    }
}

impl Deref for ResolvedTorchModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A bound `torch_speculative` model's directories.
#[derive(Debug)]
pub struct ResolvedTorchSpeculativeModel {
    pub target: ResolvedTorchModel,
    /// Checked for the same `config.json`: SGLang reads the drafter's
    /// `block_size` off it.
    pub draft: PathBuf,
}

impl TryFrom<&Model> for ResolvedTorchSpeculativeModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        let (target, draft) =
            require_bound_model_dir_pair(model, ModelType::TorchSpeculative, REQUIRED_FILES)?;
        Ok(Self {
            target: ResolvedTorchModel(target),
            draft,
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;

    use serde_json::json;

    use pipette_plan_types::Model;

    /// A bound `torch` model or `torch_speculative` pair whose directories
    /// hold `config.json`.
    pub(crate) fn bound_model(dir: &Path, speculative: bool) -> anyhow::Result<Model> {
        let checkpoint = |name: &str| -> anyhow::Result<serde_json::Value> {
            let path = dir.join(name);
            std::fs::create_dir_all(&path)?;
            std::fs::write(path.join("config.json"), b"{}")?;
            Ok(json!({"source": "absolute_dir", "dir": path}))
        };
        let spec = if speculative {
            json!({"type": "torch_speculative", "target": checkpoint("target")?, "draft": checkpoint("draft")?})
        } else {
            let mut spec = checkpoint("target")?;
            spec["type"] = json!("torch");
            spec
        };
        Ok(serde_json::from_value(spec)?)
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AbsolutePath, Mlx, ModelSource, Torch};

    use super::test_support::bound_model;
    use super::*;

    #[test]
    fn resolved_torch_model_rejects_wrong_variant() -> anyhow::Result<()> {
        let model = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/m".to_owned())?,
            },
        });
        assert!(ResolvedTorchModel::try_from(&model).is_err());
        Ok(())
    }

    #[test]
    fn resolved_torch_model_rejects_missing_directory() -> anyhow::Result<()> {
        let missing = std::env::temp_dir().join(format!(
            "pipette-torch-missing-model-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let model = Model::Torch(Torch {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(missing.to_string_lossy().into_owned())?,
            },
        });
        match ResolvedTorchModel::try_from(&model) {
            Ok(path) => anyhow::bail!("expected missing-dir error, got {}", path.display()),
            Err(err) => assert!(
                err.to_string().contains("not a directory"),
                "unexpected error: {err:#}"
            ),
        }
        Ok(())
    }

    #[test]
    fn resolved_torch_model_takes_only_a_plain_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let resolved = ResolvedTorchModel::try_from(&bound_model(tmp.path(), false)?)?;
        assert_eq!(resolved.to_path_buf(), tmp.path().join("target"));
        let speculative = bound_model(tmp.path(), true)?;
        assert!(ResolvedTorchModel::try_from(&speculative).is_err());
        Ok(())
    }

    #[test]
    fn resolved_torch_speculative_model_takes_only_a_speculative_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let resolved = ResolvedTorchSpeculativeModel::try_from(&bound_model(tmp.path(), true)?)?;
        assert_eq!(
            (resolved.target.to_path_buf(), resolved.draft),
            (tmp.path().join("target"), tmp.path().join("draft"))
        );
        let plain = bound_model(tmp.path(), false)?;
        assert!(ResolvedTorchSpeculativeModel::try_from(&plain).is_err());
        Ok(())
    }
}
