//! [`to_stored`] maps a declared model to its installed form under a base dir
//! (the manifest's `stored` under `<key>/blobs`, or a fetch's staged `blobs/`).
//! [`under_root`] joins a `stored` model's relative paths under a models dir.

use std::path::Path;

use pipette_plan_types::{
    AbsolutePath, GgufText, GgufTextSource, GgufVision, GgufVisionAbsoluteFiles, GgufVisionHfFiles,
    GgufVisionRelativeFiles, GgufVisionSource, GgufVisionSpeculative, GgufVisionSpeculativeSource,
    GgufVisionUrlFiles, GgufVisionVaultUrlFiles, Mlx, MlxSpeculative, Model, ModelSource,
    OnnxGenai, Openvino, RelativePath, RepoSubpath, ResourceUrl, Torch, TorchSpeculative,
    VaultPath,
};

/// The subdirectories a speculative pair's target and drafter are stored under
/// (MLX and torch alike). Fixed names rather than [`dir_to_stored`]'s
/// prefix-derived one: two sources with no `prefix` would both resolve to the
/// entry root.
pub const MLX_TARGET_DIR: &str = "target";
/// See [`MLX_TARGET_DIR`].
pub const MLX_DRAFT_DIR: &str = "draft";

/// Why [`to_stored`] couldn't map a model to on-disk paths.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelStoredError {
    #[error("model URL `{0}` names no file")]
    InvalidUrlPath(String),
    #[error("URL filename `{0}` is not a valid model path")]
    InvalidUrlFilename(String),
    #[error("`{0}` is not a valid local path")]
    InvalidPath(String),
    /// Two files of one model re-home to the same leaf under `base`, so one
    /// would overwrite the other.
    #[error("model and mmproj re-home to the same path `{0}`")]
    CollidingPaths(String),
    #[error("expected Relative source paths, got {0}")]
    NotRelative(String),
}

/// `model` with its sources rewritten to paths under `base`: `Relative*`
/// arms for a relative `base`, `Absolute*` for an absolute one. Apple Foundation
/// is unchanged.
pub fn to_stored(model: &Model, base: &Path) -> Result<Model, ModelStoredError> {
    let abs = base.is_absolute();
    Ok(match model {
        Model::GgufText(m) => Model::GgufText(GgufText {
            source: match &m.source {
                GgufTextSource::HuggingFace { path, .. } => {
                    gguf_text_under(base, path.as_ref(), abs)?
                }
                GgufTextSource::Url { url, .. } => {
                    gguf_text_under(base, url_filename(url)?.as_ref(), abs)?
                }
                GgufTextSource::VaultUrl { url, .. } => {
                    gguf_text_under(base, &vault_filename(url), abs)?
                }
                GgufTextSource::RelativeFile { path } => {
                    gguf_text_under(base, path_leaf(path.as_ref())?, abs)?
                }
                GgufTextSource::AbsoluteFile { path } => {
                    gguf_text_under(base, path_leaf(path.as_ref())?, abs)?
                }
            },
        }),
        Model::GgufVision(m) => Model::GgufVision(GgufVision {
            source: match &m.source {
                GgufVisionSource::HuggingFace(GgufVisionHfFiles { model, mmproj, .. }) => {
                    gguf_vision_under(base, model.as_ref(), mmproj.as_ref(), abs)?
                }
                GgufVisionSource::Url(GgufVisionUrlFiles { model, mmproj, .. }) => {
                    gguf_vision_under(
                        base,
                        url_filename(model)?.as_ref(),
                        url_filename(mmproj)?.as_ref(),
                        abs,
                    )?
                }
                GgufVisionSource::VaultUrl(GgufVisionVaultUrlFiles { model, mmproj, .. }) => {
                    gguf_vision_under(base, &vault_filename(model), &vault_filename(mmproj), abs)?
                }
                GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model, mmproj }) => {
                    gguf_vision_under(
                        base,
                        path_leaf(model.as_ref())?,
                        path_leaf(mmproj.as_ref())?,
                        abs,
                    )?
                }
                GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj }) => {
                    gguf_vision_under(
                        base,
                        path_leaf(model.as_ref())?,
                        path_leaf(mmproj.as_ref())?,
                        abs,
                    )?
                }
            },
        }),
        Model::GgufVisionSpeculative(m) => {
            let vision = to_stored(
                &Model::GgufVision(GgufVision {
                    source: m.source.vision(),
                }),
                base,
            )?;
            let Model::GgufVision(GgufVision { source: vision }) = vision else {
                return Err(ModelStoredError::InvalidPath(
                    "gguf_vision did not map to a gguf_vision source".to_owned(),
                ));
            };
            Model::GgufVisionSpeculative(GgufVisionSpeculative {
                source: speculative_under(&m.source, vision, base, abs)?,
            })
        }
        Model::AppleCoreAi(m) => Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::Mlx(m) => Model::Mlx(Mlx {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::MlxSpeculative(_) => Model::MlxSpeculative(MlxSpeculative {
            target: dir_source(under_joined(base, MLX_TARGET_DIR, abs)?),
            draft: dir_source(under_joined(base, MLX_DRAFT_DIR, abs)?),
        }),
        Model::TorchSpeculative(_) => Model::TorchSpeculative(TorchSpeculative {
            target: dir_source(under_joined(base, MLX_TARGET_DIR, abs)?),
            draft: dir_source(under_joined(base, MLX_DRAFT_DIR, abs)?),
        }),
        Model::Torch(m) => Model::Torch(Torch {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::Openvino(m) => Model::Openvino(Openvino {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::OnnxGenai(m) => Model::OnnxGenai(OnnxGenai {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::AppleFoundationText => Model::AppleFoundationText,
    })
}

/// Join every path in `stored`, which must be in `Relative*` form, under `root`.
/// An absolute `root` yields `Absolute*` arms.
pub fn under_root(stored: &Model, root: &Path) -> Result<Model, ModelStoredError> {
    let abs_root = root.is_absolute();
    let join = |rel: &str| -> Result<JoinedPath, ModelStoredError> {
        let rel_path = Path::new(rel);
        if rel_path.is_absolute() {
            return Err(ModelStoredError::InvalidPath(rel.to_owned()));
        }
        path_from_joined(&root.join(rel_path), abs_root)
    };
    Ok(match stored {
        Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile { path },
        }) => Model::GgufText(GgufText {
            source: match join(path.as_ref())? {
                JoinedPath::Rel(path) => GgufTextSource::RelativeFile { path },
                JoinedPath::Abs(path) => GgufTextSource::AbsoluteFile { path },
            },
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model, mmproj }),
        }) => {
            let model_j = join(model.as_ref())?;
            let mmproj_j = join(mmproj.as_ref())?;
            Model::GgufVision(GgufVision {
                source: match (model_j, mmproj_j) {
                    (JoinedPath::Rel(model), JoinedPath::Rel(mmproj)) => {
                        GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model, mmproj })
                    }
                    (JoinedPath::Abs(model), JoinedPath::Abs(mmproj)) => {
                        GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj })
                    }
                    _ => {
                        return Err(ModelStoredError::InvalidPath(
                            "mixed relative/absolute vision paths".to_owned(),
                        ));
                    }
                },
            })
        }
        Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::RelativeFiles { files, draft },
        }) => {
            let model_j = join(files.model.as_ref())?;
            let mmproj_j = join(files.mmproj.as_ref())?;
            let draft_j = draft.as_ref().map(|d| join(d.as_ref())).transpose()?;
            Model::GgufVisionSpeculative(GgufVisionSpeculative {
                source: match (model_j, mmproj_j, draft_j) {
                    (JoinedPath::Rel(model), JoinedPath::Rel(mmproj), draft) => {
                        GgufVisionSpeculativeSource::RelativeFiles {
                            files: GgufVisionRelativeFiles { model, mmproj },
                            draft: relative_draft(draft)?,
                        }
                    }
                    (JoinedPath::Abs(model), JoinedPath::Abs(mmproj), draft) => {
                        GgufVisionSpeculativeSource::AbsoluteFiles {
                            files: GgufVisionAbsoluteFiles { model, mmproj },
                            draft: absolute_draft(draft)?,
                        }
                    }
                    _ => {
                        return Err(ModelStoredError::InvalidPath(
                            "mixed relative/absolute vision paths".to_owned(),
                        ));
                    }
                },
            })
        }
        Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
            source: ModelSource::RelativeDir { dir },
        }) => Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
        }),
        Model::Mlx(Mlx {
            source: ModelSource::RelativeDir { dir },
        }) => Model::Mlx(Mlx {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
        }),
        Model::MlxSpeculative(MlxSpeculative {
            target: ModelSource::RelativeDir { dir: target },
            draft: ModelSource::RelativeDir { dir: draft },
        }) => Model::MlxSpeculative(MlxSpeculative {
            target: dir_source(join(target.as_ref())?),
            draft: dir_source(join(draft.as_ref())?),
        }),
        Model::TorchSpeculative(TorchSpeculative {
            target: ModelSource::RelativeDir { dir: target },
            draft: ModelSource::RelativeDir { dir: draft },
        }) => Model::TorchSpeculative(TorchSpeculative {
            target: dir_source(join(target.as_ref())?),
            draft: dir_source(join(draft.as_ref())?),
        }),
        Model::Torch(Torch {
            source: ModelSource::RelativeDir { dir },
        }) => Model::Torch(Torch {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
        }),
        Model::Openvino(Openvino {
            source: ModelSource::RelativeDir { dir },
        }) => Model::Openvino(Openvino {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
        }),
        Model::OnnxGenai(OnnxGenai {
            source: ModelSource::RelativeDir { dir },
        }) => Model::OnnxGenai(OnnxGenai {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
        }),
        Model::AppleFoundationText => Model::AppleFoundationText,
        other @ (Model::GgufText(_)
        | Model::GgufVision(_)
        | Model::GgufVisionSpeculative(_)
        | Model::Mlx(_)
        | Model::MlxSpeculative(_)
        | Model::TorchSpeculative(_)
        | Model::Torch(_)
        | Model::Openvino(_)
        | Model::OnnxGenai(_)
        | Model::AppleCoreAi(_)) => return Err(ModelStoredError::NotRelative(other.to_string())),
    })
}

fn dir_to_stored(
    source: &ModelSource,
    base: &Path,
    abs: bool,
) -> Result<ModelSource, ModelStoredError> {
    Ok(match source {
        ModelSource::HuggingFace { prefix, .. } => {
            let joined = match prefix {
                Some(prefix) => under_joined(base, prefix.as_ref(), abs)?,
                None => path_from_base(base, abs)?,
            };
            dir_source(joined)
        }
        ModelSource::VaultArchive { .. }
        | ModelSource::RelativeDir { .. }
        | ModelSource::AbsoluteDir { .. } => dir_source(path_from_base(base, abs)?),
    })
}

enum JoinedPath {
    Rel(RelativePath),
    Abs(AbsolutePath),
}

fn dir_source(joined: JoinedPath) -> ModelSource {
    match joined {
        JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
        JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
    }
}

/// The drafter's stored path beside `vision`, the pair [`to_stored`] already
/// mapped. Refuses a drafter that would land on one of those two files.
fn speculative_under(
    declared: &GgufVisionSpeculativeSource,
    vision: GgufVisionSource,
    base: &Path,
    abs: bool,
) -> Result<GgufVisionSpeculativeSource, ModelStoredError> {
    let leaf = match declared {
        GgufVisionSpeculativeSource::HuggingFace { draft, .. } => {
            draft.as_ref().map(|d| d.as_ref().to_owned())
        }
        GgufVisionSpeculativeSource::Url { draft, .. } => draft
            .as_ref()
            .map(|d| url_filename(d).map(|n| n.as_ref().to_owned()))
            .transpose()?,
        GgufVisionSpeculativeSource::VaultUrl { draft, .. } => draft.as_ref().map(vault_filename),
        GgufVisionSpeculativeSource::RelativeFiles { draft, .. } => draft
            .as_ref()
            .map(|d| path_leaf(d.as_ref()).map(str::to_owned))
            .transpose()?,
        GgufVisionSpeculativeSource::AbsoluteFiles { draft, .. } => draft
            .as_ref()
            .map(|d| path_leaf(d.as_ref()).map(str::to_owned))
            .transpose()?,
    };
    let draft = leaf
        .as_deref()
        .map(|leaf| under_joined(base, leaf, abs))
        .transpose()?;
    Ok(match (vision, draft) {
        (GgufVisionSource::RelativeFiles(files), draft) => {
            let draft = relative_draft(draft)?;
            if let Some(hit) = draft.as_ref().filter(|d| {
                d.as_ref() == files.model.as_ref() || d.as_ref() == files.mmproj.as_ref()
            }) {
                return Err(ModelStoredError::CollidingPaths(hit.to_string()));
            }
            GgufVisionSpeculativeSource::RelativeFiles { files, draft }
        }
        (GgufVisionSource::AbsoluteFiles(files), draft) => {
            let draft = absolute_draft(draft)?;
            if let Some(hit) = draft.as_ref().filter(|d| {
                d.as_ref() == files.model.as_ref() || d.as_ref() == files.mmproj.as_ref()
            }) {
                return Err(ModelStoredError::CollidingPaths(hit.to_string()));
            }
            GgufVisionSpeculativeSource::AbsoluteFiles { files, draft }
        }
        (other, _) => {
            return Err(ModelStoredError::InvalidPath(format!(
                "a stored gguf-vision source must be a path pair, got {other:?}"
            )));
        }
    })
}

fn relative_draft(joined: Option<JoinedPath>) -> Result<Option<RelativePath>, ModelStoredError> {
    match joined {
        None => Ok(None),
        Some(JoinedPath::Rel(path)) => Ok(Some(path)),
        Some(JoinedPath::Abs(path)) => Err(ModelStoredError::InvalidPath(path.to_string())),
    }
}

fn absolute_draft(joined: Option<JoinedPath>) -> Result<Option<AbsolutePath>, ModelStoredError> {
    match joined {
        None => Ok(None),
        Some(JoinedPath::Abs(path)) => Ok(Some(path)),
        Some(JoinedPath::Rel(path)) => Err(ModelStoredError::InvalidPath(path.to_string())),
    }
}

fn gguf_text_under(base: &Path, name: &str, abs: bool) -> Result<GgufTextSource, ModelStoredError> {
    Ok(match under_joined(base, name, abs)? {
        JoinedPath::Rel(path) => GgufTextSource::RelativeFile { path },
        JoinedPath::Abs(path) => GgufTextSource::AbsoluteFile { path },
    })
}

fn gguf_vision_under(
    base: &Path,
    model: &str,
    mmproj: &str,
    abs: bool,
) -> Result<GgufVisionSource, ModelStoredError> {
    let m = under_joined(base, model, abs)?;
    let p = under_joined(base, mmproj, abs)?;
    match (m, p) {
        (JoinedPath::Rel(model), JoinedPath::Rel(mmproj)) => {
            if model.as_ref() == mmproj.as_ref() {
                return Err(ModelStoredError::CollidingPaths(model.to_string()));
            }
            Ok(GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles {
                model,
                mmproj,
            }))
        }
        (JoinedPath::Abs(model), JoinedPath::Abs(mmproj)) => {
            if model.as_ref() == mmproj.as_ref() {
                return Err(ModelStoredError::CollidingPaths(model.to_string()));
            }
            Ok(GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
                model,
                mmproj,
            }))
        }
        _ => Err(ModelStoredError::InvalidPath(
            "mixed relative/absolute vision paths".to_owned(),
        )),
    }
}

fn under_joined(base: &Path, name: &str, abs: bool) -> Result<JoinedPath, ModelStoredError> {
    let joined = base.join(name);
    path_from_joined(&joined, abs)
}

fn path_from_base(base: &Path, abs: bool) -> Result<JoinedPath, ModelStoredError> {
    path_from_joined(base, abs)
}

fn path_from_joined(path: &Path, abs: bool) -> Result<JoinedPath, ModelStoredError> {
    let s = path.to_string_lossy().replace('\\', "/");
    if abs {
        Ok(JoinedPath::Abs(absolute_path(&s)?))
    } else {
        Ok(JoinedPath::Rel(relative_path(&s)?))
    }
}

fn path_leaf(path: &str) -> Result<&str, ModelStoredError> {
    path.rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ModelStoredError::InvalidPath(path.to_owned()))
}

fn relative_path(path: &str) -> Result<RelativePath, ModelStoredError> {
    let path = path.replace('\\', "/");
    RelativePath::try_new(path.clone()).map_err(|_| ModelStoredError::InvalidPath(path))
}

fn absolute_path(path: &str) -> Result<AbsolutePath, ModelStoredError> {
    let path = path.replace('\\', "/");
    AbsolutePath::try_new(path.clone()).map_err(|_| ModelStoredError::InvalidPath(path))
}

/// The URL's last path segment without query or fragment, so a signed URL
/// (`.../w.gguf?token=...`) stores as `w.gguf`. The full URL stays the identity.
fn url_filename(url: &ResourceUrl) -> Result<RepoSubpath, ModelStoredError> {
    let leaf = url
        .as_ref()
        .rsplit('/')
        .next()
        .and_then(|segment| segment.split(['?', '#']).next())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ModelStoredError::InvalidUrlPath(url.to_string()))?;
    RepoSubpath::try_new(leaf).map_err(|_| ModelStoredError::InvalidUrlFilename(leaf.to_owned()))
}

/// `<resource_id>.gguf`: a vault path's last segment is `content`, which
/// names nothing and would collide across a vision pair.
fn vault_filename(path: &VaultPath) -> String {
    format!("{}.gguf", path.resource_id())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{HfOrg, HfRepo, HfRepoName, NonEmptyTrimmedString, RepoSubpath};

    use super::*;

    fn hf(org: &str, repo: &str) -> anyhow::Result<HfRepo> {
        Ok(HfRepo {
            org: HfOrg::try_new(org.to_owned())?,
            repo_name: HfRepoName::try_new(repo.to_owned())?,
            revision: None,
            auth_token: None,
        })
    }

    /// Platform-shaped because [`to_stored`] picks its `Absolute*` arms off
    /// `Path::is_absolute`, and a driveless `/store/entry` is relative on Windows.
    /// The `/elsewhere/...` inputs below need no such treatment: `AbsolutePath`
    /// validates as a platform-independent string.
    #[cfg(windows)]
    const BASE: &str = "C:/store/entry";
    #[cfg(not(windows))]
    const BASE: &str = "/store/entry";

    fn base() -> &'static Path {
        Path::new(BASE)
    }

    fn under(relative: &str) -> String {
        format!("{BASE}/{relative}")
    }

    #[cfg(windows)]
    const MODELS_ROOT: &str = "C:/ws/models";
    #[cfg(not(windows))]
    const MODELS_ROOT: &str = "/ws/models";

    fn relative_base() -> &'static Path {
        Path::new("entry/blobs")
    }

    /// A merged drafter stays absent rather than becoming a file that does not
    /// exist.
    #[rstest]
    #[case::separate_drafter(Some("repo/draft-4L.gguf"), Some("draft-4L.gguf"))]
    #[case::merged_drafter(None, None)]
    fn a_speculative_model_stores_and_rehomes_with_its_drafter(
        #[case] declared_draft: Option<&str>,
        #[case] expected_leaf: Option<&str>,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::{GgufVisionSpeculative, GgufVisionSpeculativeSource};

        let declared = Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::HuggingFace {
                files: GgufVisionHfFiles {
                    repo: hf("liquidai", "vl")?,
                    model: RepoSubpath::try_new("weights/q4.gguf")?,
                    model_sha256: None,
                    mmproj: RepoSubpath::try_new("mm.gguf")?,
                    mmproj_sha256: None,
                },
                draft: declared_draft
                    .map(|d| RepoSubpath::try_new(d.to_owned()))
                    .transpose()?,
                draft_sha256: None,
            },
        });

        let stored = to_stored(&declared, relative_base())?;
        let Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::RelativeFiles { files, draft },
        }) = &stored
        else {
            anyhow::bail!("a relative base must yield the RelativeFiles arm, got {stored:?}");
        };
        // The HF arm keeps the repo-relative path, as `gguf_vision` does.
        assert_eq!(files.model.as_ref(), "entry/blobs/weights/q4.gguf");
        assert_eq!(files.mmproj.as_ref(), "entry/blobs/mm.gguf");
        assert_eq!(
            draft.as_ref().map(AsRef::as_ref),
            declared_draft
                .map(|d| format!("entry/blobs/{d}"))
                .as_deref()
        );

        let launched = under_root(&stored, Path::new(MODELS_ROOT))?;
        let Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::AbsoluteFiles { files, draft },
        }) = &launched
        else {
            anyhow::bail!("an absolute root must yield the AbsoluteFiles arm, got {launched:?}");
        };
        assert_eq!(
            files.model.as_ref(),
            format!("{MODELS_ROOT}/entry/blobs/weights/q4.gguf")
        );
        assert_eq!(
            draft.as_ref().map(AsRef::as_ref),
            expected_leaf
                .map(|_| format!(
                    "{MODELS_ROOT}/entry/blobs/{}",
                    declared_draft.unwrap_or_default()
                ))
                .as_deref()
        );
        Ok(())
    }

    /// A `prefix` is not part of the stored path: `plan_dir_downloads` strips it
    /// on the way in.
    #[rstest]
    #[case::no_prefixes(None, None)]
    #[case::prefixed_target(Some("4bit"), None)]
    #[case::both_prefixed(Some("4bit"), Some("8bit"))]
    fn an_mlx_pair_stores_and_rehomes_under_its_role_dirs(
        #[case] target_prefix: Option<&str>,
        #[case] draft_prefix: Option<&str>,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::MlxSpeculative;

        let source = |repo: &str, prefix: Option<&str>| -> anyhow::Result<ModelSource> {
            Ok(ModelSource::HuggingFace {
                repo: hf("liquidai", repo)?,
                prefix: prefix
                    .map(|p| RepoSubpath::try_new(p.to_owned()))
                    .transpose()?,
            })
        };
        let declared = Model::MlxSpeculative(MlxSpeculative {
            target: source("vl-mlx", target_prefix)?,
            draft: source("vl-dspark", draft_prefix)?,
        });

        let stored = to_stored(&declared, relative_base())?;
        let Model::MlxSpeculative(MlxSpeculative {
            target: ModelSource::RelativeDir { dir: target },
            draft: ModelSource::RelativeDir { dir: draft },
        }) = &stored
        else {
            anyhow::bail!("a relative base must yield two RelativeDirs, got {stored:?}");
        };
        assert_eq!(target.as_ref(), "entry/blobs/target");
        assert_eq!(draft.as_ref(), "entry/blobs/draft");

        let launched = under_root(&stored, Path::new(MODELS_ROOT))?;
        let Model::MlxSpeculative(MlxSpeculative {
            target: ModelSource::AbsoluteDir { dir: target },
            draft: ModelSource::AbsoluteDir { dir: draft },
        }) = &launched
        else {
            anyhow::bail!("an absolute root must yield two AbsoluteDirs, got {launched:?}");
        };
        assert_eq!(
            target.as_ref(),
            format!("{MODELS_ROOT}/entry/blobs/{MLX_TARGET_DIR}")
        );
        assert_eq!(
            draft.as_ref(),
            format!("{MODELS_ROOT}/entry/blobs/{MLX_DRAFT_DIR}")
        );
        assert_ne!(target, draft, "the two halves cannot share a directory");
        Ok(())
    }

    #[rstest]
    #[case::absolute_target(true, false)]
    #[case::absolute_draft(false, true)]
    fn an_mlx_pair_with_a_non_relative_half_is_not_rehomed(
        #[case] absolute_target: bool,
        #[case] absolute_draft: bool,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::MlxSpeculative;

        let half = |absolute: bool, name: &str| -> anyhow::Result<ModelSource> {
            Ok(if absolute {
                ModelSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(under(name))?,
                }
            } else {
                ModelSource::RelativeDir {
                    dir: RelativePath::try_new(format!("entry/blobs/{name}"))?,
                }
            })
        };
        let stored = Model::MlxSpeculative(MlxSpeculative {
            target: half(absolute_target, MLX_TARGET_DIR)?,
            draft: half(absolute_draft, MLX_DRAFT_DIR)?,
        });
        assert!(matches!(
            under_root(&stored, Path::new(MODELS_ROOT)),
            Err(ModelStoredError::NotRelative(_))
        ));
        Ok(())
    }

    #[test]
    fn a_drafter_colliding_with_the_weights_is_refused() -> anyhow::Result<()> {
        use pipette_plan_types::{GgufVisionSpeculative, GgufVisionSpeculativeSource};

        let declared = Model::GgufVisionSpeculative(GgufVisionSpeculative {
            source: GgufVisionSpeculativeSource::Url {
                files: GgufVisionUrlFiles {
                    model: ResourceUrl::try_new("https://ex.com/a/w.gguf")?,
                    model_sha256: None,
                    mmproj: ResourceUrl::try_new("https://ex.com/mm.gguf")?,
                    mmproj_sha256: None,
                },
                draft: Some(ResourceUrl::try_new("https://ex.com/b/w.gguf")?),
                draft_sha256: None,
            },
        });
        assert!(matches!(
            to_stored(&declared, relative_base()),
            Err(ModelStoredError::CollidingPaths(_))
        ));
        Ok(())
    }

    fn assert_all_local_paths_relative(model: &Model) -> anyhow::Result<()> {
        let paths: Vec<&str> = match model {
            Model::GgufText(GgufText {
                source: GgufTextSource::RelativeFile { path },
            }) => vec![path.as_ref()],
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model, mmproj }),
            }) => vec![model.as_ref(), mmproj.as_ref()],
            Model::Mlx(Mlx {
                source: ModelSource::RelativeDir { dir },
            })
            | Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
                source: ModelSource::RelativeDir { dir },
            })
            | Model::Torch(Torch {
                source: ModelSource::RelativeDir { dir },
            }) => vec![dir.as_ref()],
            Model::AppleFoundationText => vec![],
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleCoreAi(_)) => {
                anyhow::bail!("expected a Relative source form, got {other:?}")
            }
        };
        paths.iter().for_each(|p| {
            assert!(
                !p.starts_with('/'),
                "stored path must be relative to the model storage root: {p}"
            );
        });
        Ok(())
    }

    #[rstest]
    #[case::gguf_text_hf(
        Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: hf("org", "repo")?,
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        }),
        Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile { path: RelativePath::try_new("entry/blobs/Q4.gguf".to_owned())?,
            },
        }),
    )]
    #[case::gguf_text_url(
        Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new("https://ex.com/dir/Q4.gguf")?,
                sha256: None,
            },
        }),
        Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile { path: RelativePath::try_new("entry/blobs/Q4.gguf".to_owned())?,
            },
        }),
    )]
    #[case::gguf_text_local(
        Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path: AbsolutePath::try_new("/elsewhere/Q4.gguf".to_owned())?,
            },
        }),
        Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile { path: RelativePath::try_new("entry/blobs/Q4.gguf".to_owned())?,
            },
        }),
    )]
    #[case::gguf_vision_hf(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                repo: hf("org", "vl")?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            }),
        }),
    )]
    #[case::gguf_vision_url(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url(GgufVisionUrlFiles {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            }),
        }),
    )]
    #[case::gguf_vision_local(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model: AbsolutePath::try_new("/a/model.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/b/mmproj.gguf".to_owned())?,
            }),
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            }),
        }),
    )]
    #[case::mlx_hf_bare(
        Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: None,
            },
        }),
        Model::Mlx(Mlx {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs".to_owned())?,
            },
        }),
    )]
    #[case::mlx_hf_prefix(
        Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: Some(RepoSubpath::try_new("4bit")?),
            },
        }),
        Model::Mlx(Mlx {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs/4bit".to_owned())?,
            },
        }),
    )]
    #[case::mlx_local(
        Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/x".to_owned())?,
            },
        }),
        Model::Mlx(Mlx {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs".to_owned())?,
            },
        }),
    )]
    #[case::torch_hf_bare(
        Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: None,
            },
        }),
        Model::Torch(Torch {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs".to_owned())?,
            },
        }),
    )]
    #[case::torch_hf_prefix(
        Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: Some(RepoSubpath::try_new("8bit")?),
            },
        }),
        Model::Torch(Torch {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs/8bit".to_owned())?,
            },
        }),
    )]
    #[case::torch_local(
        Model::Torch(Torch {
            source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/torch".to_owned())?,
            },
        }),
        Model::Torch(Torch {
            source: ModelSource::RelativeDir { dir: RelativePath::try_new("entry/blobs".to_owned())?,
            },
        }),
    )]
    fn stored_form_is_relative_to_model_storage_root(
        #[case] declared: Model,
        #[case] expected: Model,
    ) -> anyhow::Result<()> {
        let stored = to_stored(&declared, relative_base())?;
        assert_eq!(stored, expected);
        assert_all_local_paths_relative(&stored)?;
        Ok(())
    }

    #[test]
    fn gguf_text_hf_points_under_the_entry() -> anyhow::Result<()> {
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: hf("org", "repo")?,
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        });
        let local = to_stored(&model, base())?;
        assert_eq!(
            local,
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile {
                    path: AbsolutePath::try_new(under("Q4.gguf"))?,
                },
            })
        );
        Ok(())
    }

    fn gguf_url(url: &str) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::Url {
                url: ResourceUrl::try_new(url)?,
                sha256: None,
            },
        }))
    }

    #[rstest]
    #[case("https://ex.com/dir/w.gguf")]
    #[case("https://ex.com/dir/w.gguf?token=abc&expires=1")]
    #[case("https://ex.com/dir/w.gguf#section")]
    fn url_source_strips_query_and_fragment(#[case] url: &str) -> anyhow::Result<()> {
        let Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) = to_stored(&gguf_url(url)?, base())?
        else {
            return Err(anyhow::anyhow!("expected Absolute gguf-text source"));
        };
        assert_eq!(path, AbsolutePath::try_new(under("w.gguf"))?);
        Ok(())
    }

    #[rstest]
    #[case("https://ex.com/dir/", ModelStoredError::InvalidUrlPath("https://ex.com/dir/".to_owned()))]
    #[case("https://ex.com/my%20model.gguf", ModelStoredError::InvalidUrlFilename("my%20model.gguf".to_owned()))]
    fn url_source_is_rejected(
        #[case] url: &str,
        #[case] expected: ModelStoredError,
    ) -> anyhow::Result<()> {
        assert_eq!(to_stored(&gguf_url(url)?, base()), Err(expected));
        Ok(())
    }

    fn vault_url(id: &str) -> anyhow::Result<VaultPath> {
        Ok(format!("/vault/resources/{id}/content").parse()?)
    }

    fn custom_label(label: Option<&str>) -> anyhow::Result<Option<NonEmptyTrimmedString>> {
        Ok(label.map(NonEmptyTrimmedString::try_new).transpose()?)
    }

    /// The label cases prove a record's name does not change the path.
    #[rstest]
    #[case::unlabelled(None)]
    #[case::labelled(Some("acme LFM3 q4"))]
    fn vault_gguf_lands_under_its_resource_id(#[case] label: Option<&str>) -> anyhow::Result<()> {
        let model = Model::GgufText(GgufText {
            source: GgufTextSource::VaultUrl {
                url: vault_url("vault_1")?,
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: custom_label(label)?,
            },
        });
        let Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile { path },
        }) = to_stored(&model, base())?
        else {
            return Err(anyhow::anyhow!("expected Absolute gguf-text source"));
        };
        assert_eq!(path, AbsolutePath::try_new(under("vault_1.gguf"))?);
        Ok(())
    }

    /// A vision pair naming one resource twice would land on one file.
    #[rstest]
    #[case::unlabelled(None)]
    #[case::labelled(Some("acme LFM3 q4"))]
    fn vault_vision_pair_refuses_a_shared_resource(
        #[case] label: Option<&str>,
    ) -> anyhow::Result<()> {
        let pair = |model: &str, mmproj: &str| -> anyhow::Result<Model> {
            Ok(Model::GgufVision(GgufVision {
                source: GgufVisionSource::VaultUrl(GgufVisionVaultUrlFiles {
                    model: vault_url(model)?,
                    model_sha256: None,
                    mmproj: vault_url(mmproj)?,
                    mmproj_sha256: None,
                    principal_id: "org_1".parse()?,
                    custom_label: custom_label(label)?,
                }),
            }))
        };
        assert!(to_stored(&pair("vault_1", "vault_2")?, base()).is_ok());
        assert_eq!(
            to_stored(&pair("vault_1", "vault_1")?, base()),
            Err(ModelStoredError::CollidingPaths(under("vault_1.gguf")))
        );
        Ok(())
    }

    /// A vault archive is the entry root, like a prefix-less HF snapshot.
    #[rstest]
    #[case::unlabelled(None)]
    #[case::labelled(Some("acme LFM3 q4"))]
    fn vault_archive_is_the_entry(#[case] label: Option<&str>) -> anyhow::Result<()> {
        let model = Model::Mlx(Mlx {
            source: ModelSource::VaultArchive {
                url: vault_url("vault_1")?,
                sha256: None,
                principal_id: "org_1".parse()?,
                custom_label: custom_label(label)?,
            },
        });
        let Model::Mlx(Mlx {
            source: ModelSource::AbsoluteDir { dir },
        }) = to_stored(&model, base())?
        else {
            return Err(anyhow::anyhow!("expected Absolute mlx source"));
        };
        assert_eq!(dir.as_ref(), BASE);
        Ok(())
    }

    #[test]
    fn dir_model_prefix_nests_under_entry_bare_is_the_entry() -> anyhow::Result<()> {
        let bare = Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: None,
            },
        });
        let sub = Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: hf("org", "repo")?,
                prefix: Some(RepoSubpath::try_new("4bit")?),
            },
        });
        let dir_of = |m: Model| -> anyhow::Result<String> {
            let (Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir },
            })
            | Model::Torch(Torch {
                source: ModelSource::AbsoluteDir { dir },
            })) = m
            else {
                return Err(anyhow::anyhow!("expected a local dir source"));
            };
            Ok(dir.as_ref().to_owned())
        };
        assert_eq!(dir_of(to_stored(&bare, base())?)?, BASE);
        assert_eq!(dir_of(to_stored(&sub, base())?)?, under("4bit"));
        Ok(())
    }

    #[test]
    fn local_file_is_rehomed_under_base() -> anyhow::Result<()> {
        let local = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new("/elsewhere/w.gguf".to_owned())?,
            },
        });
        assert_eq!(
            to_stored(&local, base())?,
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile {
                    path: AbsolutePath::try_new(under("w.gguf"))?,
                },
            })
        );
        Ok(())
    }

    #[rstest]
    #[case::mlx(Model::Mlx(Mlx {
        source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/x".to_owned())?,
        },
    }))]
    #[case::torch(Model::Torch(Torch {
        source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/torch".to_owned())?,
        },
    }))]
    fn local_dir_is_rehomed_to_base(#[case] local: Model) -> anyhow::Result<()> {
        let dir = match to_stored(&local, base())? {
            Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir },
            })
            | Model::AppleCoreAi(pipette_plan_types::AppleCoreAi {
                source: ModelSource::AbsoluteDir { dir },
            })
            | Model::Torch(Torch {
                source: ModelSource::AbsoluteDir { dir },
            }) => dir,
            other @ (Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleFoundationText
            | Model::AppleCoreAi(_)) => {
                anyhow::bail!("expected absolute dir model, got {other:?}")
            }
        };
        assert_eq!(dir.as_ref(), BASE);
        Ok(())
    }

    #[rstest]
    #[case::hf(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                repo: hf("org", "vl")?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        })
    )]
    #[case::url(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url(GgufVisionUrlFiles {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            }),
        })
    )]
    #[case::local(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model: AbsolutePath::try_new("/elsewhere/model.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/elsewhere/mmproj.gguf".to_owned())?,
            }),
        })
    )]
    fn vision_both_files_land_under_base(#[case] declared: Model) -> anyhow::Result<()> {
        assert_eq!(
            to_stored(&declared, base())?,
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
                    model: AbsolutePath::try_new(under("model.gguf"))?,
                    mmproj: AbsolutePath::try_new(under("mmproj.gguf"))?,
                }),
            })
        );
        Ok(())
    }

    #[test]
    fn local_vision_rejects_colliding_leaf_names() -> anyhow::Result<()> {
        let local = Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles {
                model: AbsolutePath::try_new("/data/a/weights.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/data/b/weights.gguf".to_owned())?,
            }),
        });
        assert_eq!(
            to_stored(&local, base()),
            Err(ModelStoredError::CollidingPaths(under("weights.gguf")))
        );
        Ok(())
    }

    #[test]
    fn afm_is_unchanged() -> anyhow::Result<()> {
        assert_eq!(
            to_stored(&Model::AppleFoundationText, base())?,
            Model::AppleFoundationText
        );
        Ok(())
    }

    #[test]
    fn under_root_joins_relative_local_paths() -> anyhow::Result<()> {
        let local = Model::GgufText(GgufText {
            source: GgufTextSource::RelativeFile {
                path: RelativePath::try_new("key/blobs/w.gguf".to_owned())?,
            },
        });
        assert_eq!(
            under_root(&local, Path::new(MODELS_ROOT))?,
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile {
                    path: AbsolutePath::try_new(format!("{MODELS_ROOT}/key/blobs/w.gguf"))?,
                },
            })
        );
        Ok(())
    }

    #[test]
    fn under_root_rejects_already_absolute_local() -> anyhow::Result<()> {
        let local = Model::GgufText(GgufText {
            source: GgufTextSource::AbsoluteFile {
                path: AbsolutePath::try_new("/elsewhere/w.gguf".to_owned())?,
            },
        });
        assert!(matches!(
            under_root(&local, Path::new(MODELS_ROOT)),
            Err(ModelStoredError::NotRelative(_))
        ));
        Ok(())
    }

    #[test]
    fn under_root_rejects_non_local_sources() -> anyhow::Result<()> {
        let remote = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: hf("org", "repo")?,
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        });
        assert!(matches!(
            under_root(&remote, Path::new(MODELS_ROOT)),
            Err(ModelStoredError::NotRelative(_))
        ));
        Ok(())
    }
}
