//! The bound-model projection every directory-shaped backend shares.
//!
//! Install is `pipette_artifacts::ensure_model`; this is the other end — what
//! the run path checks before handing a path to an engine: the model is the
//! kind this engine runs, it was bound to a host directory, and that directory
//! holds what the loader needs.

use std::path::PathBuf;

use pipette_plan_types::{Model, ModelSource, ModelType};

/// Bound model directory for a model of kind `expected`, verified to exist and
/// to contain every one of `required_files`.
///
/// `required_files` is the loader's manifest, so it belongs to the caller: a
/// GenAI pipeline needs its compiled tokenizer pair, while `mlx_lm` accepts
/// several tokenizer layouts. Name only files the loader always needs — the
/// point is to catch a truncated snapshot before the engine does, not to
/// second-guess it.
pub fn require_bound_model_dir(
    bound: &Model,
    expected: ModelType,
    required_files: &[&str],
) -> anyhow::Result<PathBuf> {
    let actual = ModelType::of(bound);
    anyhow::ensure!(
        actual == expected,
        "expected a bound {expected} model, got {actual}"
    );
    // Exhaustive, so a new directory-shaped `Model` has to answer here rather
    // than falling into the not-a-directory arm.
    let source = match bound {
        Model::Mlx(m) => &m.source,
        Model::Torch(m) => &m.source,
        Model::Openvino(m) => &m.source,
        Model::GgufText(_) | Model::GgufVision(_) | Model::AppleFoundationText => {
            anyhow::bail!("{expected} is not a directory-shaped model")
        }
    };
    let ModelSource::AbsoluteDir { dir } = source else {
        anyhow::bail!(
            "expected bound {expected} AbsoluteDir (after ensure_model/bind_under), got {bound}"
        );
    };

    let dir = PathBuf::from(dir.as_ref());
    if !dir.is_dir() {
        anyhow::bail!(
            "bound {expected} model is not a directory: {} (ensure/bind may be incomplete)",
            dir.display()
        );
    }
    if let Some(missing) = required_files.iter().find(|f| !dir.join(f).is_file()) {
        anyhow::bail!("{expected} model at {} is missing {missing}", dir.display());
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AbsolutePath, Mlx, Openvino, Torch};

    use super::*;

    /// A bound model of `kind`. Exhaustive on purpose: a catch-all would hand
    /// back an Mlx model for a Torch case and the test would pass for the
    /// wrong reason.
    fn dir_model(kind: ModelType, dir: &std::path::Path) -> anyhow::Result<Model> {
        let source = ModelSource::AbsoluteDir {
            dir: AbsolutePath::try_new(dir.to_string_lossy().into_owned())?,
        };
        match kind {
            ModelType::Mlx => Ok(Model::Mlx(Mlx { source })),
            ModelType::Torch => Ok(Model::Torch(Torch { source })),
            ModelType::Openvino => Ok(Model::Openvino(Openvino { source })),
            other => anyhow::bail!("{other} has no directory form"),
        }
    }

    #[test]
    fn a_bound_directory_of_the_expected_kind_is_returned() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = dir_model(ModelType::Mlx, tmp.path())?;
        assert_eq!(
            require_bound_model_dir(&model, ModelType::Mlx, &[])?,
            tmp.path()
        );
        Ok(())
    }

    /// Another engine's model reaching this one is a dispatch bug; the error
    /// has to name both kinds or it reads as a missing file.
    #[test]
    fn another_kind_is_refused_by_name() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = dir_model(ModelType::Mlx, tmp.path())?;
        let Err(err) = require_bound_model_dir(&model, ModelType::Openvino, &[]) else {
            anyhow::bail!("expected a wrong-kind rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains("openvino") && msg.contains("mlx"), "got {msg}");
        Ok(())
    }

    /// The omission people actually hit is a partial snapshot, so the error
    /// names the file rather than calling the directory incomplete.
    #[test]
    fn a_missing_required_file_is_named() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(tmp.path().join("openvino_model.xml"), "<net/>")?;
        let model = dir_model(ModelType::Openvino, tmp.path())?;
        let Err(err) = require_bound_model_dir(
            &model,
            ModelType::Openvino,
            &["openvino_model.xml", "openvino_tokenizer.xml"],
        ) else {
            anyhow::bail!("expected a missing-file rejection");
        };
        assert!(
            err.to_string().contains("openvino_tokenizer.xml"),
            "got {err}"
        );
        Ok(())
    }

    /// An unbound model means ensure/bind never ran, which must not look like
    /// an empty directory.
    #[test]
    fn an_unbound_model_is_refused() -> anyhow::Result<()> {
        let model = Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: pipette_plan_types::HfRepo::parse_org_repo("o/r")
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
                prefix: None,
            },
        });
        let Err(err) = require_bound_model_dir(&model, ModelType::Mlx, &[]) else {
            anyhow::bail!("expected an unbound-model rejection");
        };
        assert!(err.to_string().contains("bind_under"), "got {err}");
        Ok(())
    }
}
