//! The directories of a bound MLX model, resolved from the plan's [`Model`].
//!
//! Install is `pipette_artifacts::ensure_model`.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use pipette_ops::models::{require_bound_model_dir, require_bound_model_dir_pair};
use pipette_plan_types::{Model, ModelType};

/// `config.json` only: the weights are sharded or not depending on the repo,
/// and the tokenizer is whatever `AutoTokenizer` accepts (`tokenizer.json` or a
/// sentencepiece `tokenizer.model`), so requiring either by name would refuse
/// snapshots that load fine. This catches a download that stopped before
/// anything landed.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// A bound `mlx` model's directory.
#[derive(Debug)]
pub struct ResolvedMlxModel(PathBuf);

impl TryFrom<&Model> for ResolvedMlxModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        require_bound_model_dir(model, ModelType::Mlx, REQUIRED_FILES).map(Self)
    }
}

impl Deref for ResolvedMlxModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A bound `mlx_speculative` model's directories.
#[derive(Debug)]
pub struct ResolvedMlxSpeculativeModel {
    pub target: ResolvedMlxModel,
    /// Checked for the same `config.json`: `load_drafter` reads it to tell
    /// which drafter the directory holds.
    pub draft: PathBuf,
}

impl TryFrom<&Model> for ResolvedMlxSpeculativeModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        let (target, draft) =
            require_bound_model_dir_pair(model, ModelType::MlxSpeculative, REQUIRED_FILES)?;
        Ok(Self {
            target: ResolvedMlxModel(target),
            draft,
        })
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;

    use serde_json::json;

    use pipette_plan_types::Model;

    /// A bound `mlx` model or `mlx_speculative` pair whose directories hold
    /// `config.json`.
    pub(crate) fn bound_model(dir: &Path, speculative: bool) -> anyhow::Result<Model> {
        let bundle = |name: &str| -> anyhow::Result<serde_json::Value> {
            let path = dir.join(name);
            std::fs::create_dir_all(&path)?;
            std::fs::write(path.join("config.json"), b"{}")?;
            Ok(json!({"source": "absolute_dir", "dir": path}))
        };
        let spec = if speculative {
            json!({"type": "mlx_speculative", "target": bundle("target")?, "draft": bundle("draft")?})
        } else {
            let mut spec = bundle("target")?;
            spec["type"] = json!("mlx");
            spec
        };
        Ok(serde_json::from_value(spec)?)
    }
}

#[cfg(test)]
mod tests {
    use pipette_artifacts::model::ModelArtifactStore;
    use pipette_plan_types::{AbsolutePath, HfRepo, Mlx, ModelSource};

    use super::test_support::bound_model;
    use super::*;

    fn fake_fetch(_declared: &Model, into: &Model) -> anyhow::Result<()> {
        let Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir { dir },
        }) = into
        else {
            anyhow::bail!("fake fetch only handles a local mlx dir dest");
        };
        let dir = Path::new(dir.as_ref());
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.json"), b"{}")?;
        Ok(())
    }

    #[test]
    fn ensure_bind_and_resolve_mlx_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let models_root = tmp.path().join("models");
        let store = ModelArtifactStore::new(models_root.clone());
        let declared = Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: HfRepo::parse_org_repo("mlx-community/LFM2-350M-4bit")
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                prefix: None,
            },
        });
        let manifest = store.ensure(&declared, fake_fetch)?;
        assert_eq!(
            manifest.declared.to_string(),
            "mlx-community/LFM2-350M-4bit"
        );

        let dir = ResolvedMlxModel::try_from(&manifest.bind_under(&models_root)?)?;
        assert!(dir.starts_with(&models_root));
        assert!(dir.join("config.json").exists());

        let missing = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(tmp.path().join("missing").display().to_string())?,
            },
        });
        assert!(ResolvedMlxModel::try_from(&missing).is_err());
        Ok(())
    }

    #[test]
    fn resolved_mlx_model_takes_only_a_plain_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let resolved = ResolvedMlxModel::try_from(&bound_model(tmp.path(), false)?)?;
        assert_eq!(resolved.to_path_buf(), tmp.path().join("target"));
        let speculative = bound_model(tmp.path(), true)?;
        assert!(ResolvedMlxModel::try_from(&speculative).is_err());
        Ok(())
    }

    #[test]
    fn resolved_mlx_speculative_model_takes_only_a_speculative_model() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let resolved = ResolvedMlxSpeculativeModel::try_from(&bound_model(tmp.path(), true)?)?;
        assert_eq!(
            (resolved.target.to_path_buf(), resolved.draft),
            (tmp.path().join("target"), tmp.path().join("draft"))
        );
        let plain = bound_model(tmp.path(), false)?;
        assert!(ResolvedMlxSpeculativeModel::try_from(&plain).is_err());
        Ok(())
    }
}
