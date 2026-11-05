//! The bound-model projection every directory-shaped backend shares.
//!
//! What the run path checks before handing a path to an engine (install is
//! `pipette_artifacts::ensure_model`): the model is the kind this engine runs,
//! it was bound to a host directory, and that directory holds what the loader
//! needs.

use std::path::PathBuf;

use pipette_plan_types::{Model, ModelSource, ModelType};

/// What a bound directory is being checked *as*, so a refusal names which half
/// of a speculative pair is missing rather than just the model type.
#[derive(Debug, Clone, Copy)]
enum Role {
    /// The model itself, for a type that is one directory.
    Whole,
    Target,
    Draft,
}

impl Role {
    /// Appended after the model type in a refusal, leading space included.
    fn suffix(self) -> &'static str {
        match self {
            Role::Whole => "",
            Role::Target => " target",
            Role::Draft => " drafter",
        }
    }
}

/// Bound model directory for a model of kind `expected`, verified to exist and
/// to contain every one of `required_files`.
///
/// `required_files` is the loader's manifest, so it belongs to the caller: a
/// GenAI pipeline needs its compiled tokenizer pair, while `mlx_lm` accepts
/// several tokenizer layouts. Name only files the loader always needs: the
/// point is to catch a truncated snapshot, not to second-guess the loader.
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
        Model::AppleCoreAi(m) => &m.source,
        Model::Torch(m) => &m.source,
        Model::Openvino(m) => &m.source,
        Model::OnnxGenai(m) => &m.source,
        // Two directories, so it has no single answer to give here.
        Model::MlxSpeculative(_) | Model::TorchSpeculative(_) => {
            anyhow::bail!("{expected} is a directory pair; use require_bound_model_dir_pair")
        }
        Model::GgufText(_)
        | Model::GgufVision(_)
        | Model::GgufVisionSpeculative(_)
        | Model::AppleFoundationText => {
            anyhow::bail!("{expected} is not a directory-shaped model")
        }
    };
    bound_dir(source, expected, Role::Whole, required_files, bound)
}

/// [`require_bound_model_dir`] for a model that is **two** bound directories:
/// the target and the drafter it is measured against, in that order.
///
/// A tuple rather than a `Vec`: an engine that mixed the two up would report
/// the drafter's numbers as the target's.
pub fn require_bound_model_dir_pair(
    bound: &Model,
    expected: ModelType,
    required_files: &[&str],
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let actual = ModelType::of(bound);
    anyhow::ensure!(
        actual == expected,
        "expected a bound {expected} model, got {actual}"
    );
    let (Model::MlxSpeculative(m) | Model::TorchSpeculative(m)) = bound else {
        anyhow::bail!("{expected} is not a directory-pair model");
    };
    Ok((
        bound_dir(&m.target, expected, Role::Target, required_files, bound)?,
        bound_dir(&m.draft, expected, Role::Draft, required_files, bound)?,
    ))
}

fn bound_dir(
    source: &ModelSource,
    expected: ModelType,
    role: Role,
    required_files: &[&str],
    bound: &Model,
) -> anyhow::Result<PathBuf> {
    let suffix = role.suffix();
    let ModelSource::AbsoluteDir { dir } = source else {
        anyhow::bail!(
            "expected bound {expected}{suffix} AbsoluteDir (after ensure_model/bind_under), got {bound}"
        );
    };

    let dir = PathBuf::from(dir.as_ref());
    if !dir.is_dir() {
        anyhow::bail!(
            "bound {expected}{suffix} model is not a directory: {} (ensure/bind may be incomplete)",
            dir.display()
        );
    }
    if let Some(missing) = required_files.iter().find(|f| !dir.join(f).is_file()) {
        anyhow::bail!(
            "{expected}{suffix} model at {} is missing {missing}",
            dir.display()
        );
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
            ModelType::AppleCoreAi => Ok(Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
                source,
            })),
            ModelType::Torch => Ok(Model::Torch(Torch { source })),
            ModelType::Openvino => Ok(Model::Openvino(Openvino { source })),
            other @ (ModelType::GgufText
            | ModelType::GgufVision
            | ModelType::GgufVisionSpeculative
            | ModelType::MlxSpeculative
            | ModelType::TorchSpeculative
            | ModelType::OnnxGenai
            | ModelType::AppleFoundationText) => anyhow::bail!("{other} has no directory form"),
        }
    }

    fn mlx_pair(target: &std::path::Path, draft: &std::path::Path) -> anyhow::Result<Model> {
        let dir = |p: &std::path::Path| -> anyhow::Result<ModelSource> {
            Ok(ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(p.to_string_lossy().into_owned())?,
            })
        };
        Ok(Model::MlxSpeculative(pipette_plan_types::MlxSpeculative {
            target: dir(target)?,
            draft: dir(draft)?,
        }))
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

    /// Naming both kinds keeps a dispatch bug from reading as a missing file.
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

    #[test]
    fn a_bound_pair_is_returned_target_first() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let (target, draft) = (tmp.path().join("target"), tmp.path().join("draft"));
        [&target, &draft].iter().try_for_each(|dir| {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("config.json"), b"{}")
        })?;

        let got = require_bound_model_dir_pair(
            &mlx_pair(&target, &draft)?,
            ModelType::MlxSpeculative,
            &["config.json"],
        )?;
        assert_eq!(got, (target, draft));
        Ok(())
    }

    /// "mlx_speculative model is missing config.json" alone would send the
    /// reader to the target when the drafter is the half that is missing.
    #[rstest::rstest]
    #[case::target_incomplete(false, true, "target")]
    #[case::draft_incomplete(true, false, "drafter")]
    fn an_incomplete_half_of_a_pair_is_named(
        #[case] target_ok: bool,
        #[case] draft_ok: bool,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let (target, draft) = (tmp.path().join("target"), tmp.path().join("draft"));
        [(&target, target_ok), (&draft, draft_ok)]
            .iter()
            .try_for_each(|(dir, ok)| {
                std::fs::create_dir_all(dir)?;
                if *ok {
                    std::fs::write(dir.join("config.json"), b"{}")?;
                }
                Ok::<_, std::io::Error>(())
            })?;

        let Err(err) = require_bound_model_dir_pair(
            &mlx_pair(&target, &draft)?,
            ModelType::MlxSpeculative,
            &["config.json"],
        ) else {
            anyhow::bail!("expected an incomplete-half rejection");
        };
        let msg = err.to_string();
        assert!(msg.contains(expected), "got {msg}");
        assert!(msg.contains("config.json"), "got {msg}");
        Ok(())
    }

    #[test]
    fn a_pair_is_refused_by_the_single_directory_entry_point() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let model = mlx_pair(&tmp.path().join("target"), &tmp.path().join("draft"))?;
        let Err(err) = require_bound_model_dir(&model, ModelType::MlxSpeculative, &[]) else {
            anyhow::bail!("expected a directory-pair rejection");
        };
        assert!(
            err.to_string().contains("require_bound_model_dir_pair"),
            "got {err}"
        );
        Ok(())
    }

    /// An unbound model must not look like an empty directory.
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
