//! Bound MLX model path projection from a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_model`.

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::ModelType;

/// Files `mlx_lm.load` needs.
///
/// `config.json` only: the weights are sharded or not depending on the repo,
/// and the tokenizer is whatever `AutoTokenizer` accepts — a `tokenizer.json`
/// or a sentencepiece `tokenizer.model` — so requiring either by name would
/// refuse snapshots that load fine. This catches the download that stopped
/// before anything landed, and leaves the rest to mlx-lm.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// Bound MLX model directory: `Model::Mlx` + `AbsoluteDir` after ensure/bind.
pub fn require_mlx_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_model_dir(&req.model.bound, ModelType::Mlx, REQUIRED_FILES)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pipette_artifacts::model::ModelArtifactStore;
    use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
    use pipette_plan_types::run::{DeclaredBound, RunRequest};
    use pipette_plan_types::{AbsolutePath, HfRepo, Mlx, Model, ModelSource};

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

    fn stub_req(declared: Model, bound: Model) -> anyhow::Result<RunRequest> {
        let runtime: pipette_plan_types::Runtime = serde_json::from_value(serde_json::json!({
            "type": "mlx_macos_pipette",
            "version": "0.31.3",
            "flavor": "macos-arm64",
            "source": {
                "type": "pip_requirements_text",
                "contents": "mlx-lm==0.31.3"
            }
        }))?;
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound { declared, bound },
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "p".into(),
                parameter_prefill_tokens: 1,
            }),
        })
    }

    #[test]
    fn ensure_bind_and_require_mlx_model_dir() -> anyhow::Result<()> {
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

        let bound = manifest.bind_under(&models_root)?;
        let dir = require_mlx_model_dir(&stub_req(declared.clone(), bound)?)?;
        assert!(dir.starts_with(&models_root));
        assert!(dir.join("config.json").exists());

        let missing = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(tmp.path().join("missing").display().to_string())?,
            },
        });
        assert!(require_mlx_model_dir(&stub_req(declared, missing)?).is_err());
        Ok(())
    }
}
