//! Bound torch model path projection for a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_model`. This module only
//! checks the bound plan model and returns its host directory for execute.

use std::path::PathBuf;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::run::RunRequest;
use pipette_plan_types::ModelType;

/// Files a transformers-style load needs. Only `config.json` is universal —
/// tokenizer layout varies by model — but that one is enough to catch a
/// snapshot that stopped partway, which otherwise surfaces from inside the
/// server at load time.
const REQUIRED_FILES: &[&str] = &["config.json"];

/// Bound torch model directory: `Model::Torch` + `AbsoluteDir` after ensure/bind.
pub fn require_torch_model_dir(req: &RunRequest) -> anyhow::Result<PathBuf> {
    require_bound_model_dir(&req.model.bound, ModelType::Torch, REQUIRED_FILES)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{Model, Torch};

    use super::*;

    fn stub_req(bound: Model) -> anyhow::Result<RunRequest> {
        use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
        use pipette_plan_types::run::DeclaredBound;

        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(pipette_plan_types::Runtime::AppleFoundation(
                Default::default(),
            )),
            model: DeclaredBound::already_bound(bound),
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
    fn require_torch_model_dir_rejects_wrong_variant() -> anyhow::Result<()> {
        use pipette_plan_types::{AbsolutePath, Mlx, ModelSource};

        let model = Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/m".to_owned())?,
            },
        });
        assert!(require_torch_model_dir(&stub_req(model)?).is_err());
        Ok(())
    }

    #[test]
    fn require_torch_model_dir_rejects_missing_directory() -> anyhow::Result<()> {
        use pipette_plan_types::{AbsolutePath, ModelSource};

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
        match require_torch_model_dir(&stub_req(model)?) {
            Ok(path) => anyhow::bail!("expected missing-dir error, got {}", path.display()),
            Err(err) => assert!(
                err.to_string().contains("not a directory"),
                "unexpected error: {err:#}"
            ),
        }
        Ok(())
    }
}
