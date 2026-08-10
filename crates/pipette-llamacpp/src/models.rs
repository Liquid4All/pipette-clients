//! Bound GGUF path projection for a prepared [`RunRequest`].
//!
//! Install is `pipette_artifacts::ensure_model`. This module only
//! checks the bound plan model and returns host paths for execute.

use std::path::PathBuf;

use pipette_plan_types::run::RunRequest;
use pipette_plan_types::{GgufText, GgufTextSource, GgufVision, GgufVisionSource, Model};

fn require_file(path: PathBuf, what: &str) -> anyhow::Result<PathBuf> {
    if !path.is_file() {
        anyhow::bail!(
            "bound {what} is not a file: {} (ensure/bind may be incomplete)",
            path.display()
        );
    }
    Ok(path)
}

/// Bound main GGUF for text cells: `Model::GgufText` + `AbsoluteFile` after bind.
pub fn require_gguf_text(req: &RunRequest) -> anyhow::Result<PathBuf> {
    match &req.model.bound {
        Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) => require_file(PathBuf::from(path.as_ref()), "GGUF text model"),
        other => anyhow::bail!(
            "expected bound GgufText AbsoluteFile (after ensure_model/bind_under), got {other}"
        ),
    }
}

/// Bound vision weights + mmproj: `Model::GgufVision` + `AbsoluteFiles` after bind.
///
/// Returns `(model_gguf, mmproj_gguf)`.
pub fn require_gguf_vision(req: &RunRequest) -> anyhow::Result<(PathBuf, PathBuf)> {
    match &req.model.bound {
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles { model, mmproj },
        }) => Ok((
            require_file(PathBuf::from(model.as_ref()), "GGUF vision model")?,
            require_file(PathBuf::from(mmproj.as_ref()), "GGUF mmproj")?,
        )),
        other => anyhow::bail!(
            "expected bound GgufVision AbsoluteFiles (after ensure_model/bind_under), got {other}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_gguf_text_rejects_wrong_variant() -> anyhow::Result<()> {
        use pipette_plan_types::benchmark::{BenchmarkDefinition, PrefillThroughput};
        use pipette_plan_types::run::DeclaredBound;
        use pipette_plan_types::{AbsolutePath, Model};

        let model = Model::Mlx(pipette_plan_types::Mlx {
            source: pipette_plan_types::ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new("/tmp/m".to_owned())?,
            },
        });
        let req = pipette_plan_types::run::RunRequest {
            runtime: DeclaredBound::already_bound(pipette_plan_types::Runtime::AppleFoundation(
                Default::default(),
            )),
            model: DeclaredBound::already_bound(model),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
                benchmark_id: "p".into(),
                parameter_prefill_tokens: 1,
            }),
        };
        assert!(require_gguf_text(&req).is_err());
        Ok(())
    }
}
