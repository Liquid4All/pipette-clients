//! [`to_stored`] — map a declared model to its effective (installed) form under
//! a base dir — the model-side counterpart to
//! [`crate::runtime::to_stored`].
//! [`under_root`] — prefix an already-Relative model with a root via [`Path::join`].
//!
//! Store usage:
//! - **manifest `stored`** — `to_stored(declared, &key.blobs_dir())`
//!   (models-root-relative Relative paths).
//! - **loader paths** — `under_root(&stored, models_dir)` (join the relative
//!   `stored` paths under the concrete models dir).
//! - **fetch staging** — `to_stored(declared, &staged_blobs)` (absolute base).

use std::path::Path;

use pipette_plan_types::{
    AbsolutePath, GgufText, GgufTextSource, GgufVision, GgufVisionSource, Mlx, Model, ModelSource,
    Openvino, RelativePath, RepoSubpath, ResourceUrl, Torch,
};

/// Why [`to_stored`] couldn't map a model to on-disk paths.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelStoredError {
    /// A `Url` source's path ends in `/`, so it names no file to store under.
    #[error("model URL `{0}` names no file")]
    InvalidUrlPath(String),
    /// A `Url` source's filename isn't a safe model path (an escape sequence or
    /// similar) even after its query string and fragment are stripped.
    #[error("URL filename `{0}` is not a valid model path")]
    InvalidUrlFilename(String),
    /// `base` joined with a file/dir didn't form a valid [`AbsolutePath`] (e.g. a
    /// non-normalized `base`).
    #[error("`{0}` is not a valid local path")]
    InvalidPath(String),
    /// Vision model + mmproj re-home to the same on-disk path (same leaf under
    /// `base`), so import would overwrite one file with the other.
    #[error("model and mmproj re-home to the same path `{0}`")]
    CollidingPaths(String),
    /// [`under_root`] needs a model whose sources are already `Relative`
    /// (e.g. manifest `stored`).
    #[error("expected Relative source paths, got {0}")]
    NotRelative(String),
}

/// Effective form of `model` under `base` (manifest `stored` / fetch target).
///
/// Rewrites sources to store paths under `base`. Relativity of the result
/// matches `base` (relative → `Relative` arms; absolute → `Absolute` arms).
/// OS-bundled models are unchanged. Multi-file arms reject path collisions.
pub fn to_stored(model: &Model, base: &Path) -> Result<Model, ModelStoredError> {
    // Relative `base` (manifest `stored`) → Relative* arms; absolute base → Absolute*.
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
                GgufVisionSource::HuggingFace { model, mmproj, .. } => {
                    gguf_vision_under(base, model.as_ref(), mmproj.as_ref(), abs)?
                }
                GgufVisionSource::Url { model, mmproj, .. } => gguf_vision_under(
                    base,
                    url_filename(model)?.as_ref(),
                    url_filename(mmproj)?.as_ref(),
                    abs,
                )?,
                GgufVisionSource::RelativeFiles { model, mmproj } => gguf_vision_under(
                    base,
                    path_leaf(model.as_ref())?,
                    path_leaf(mmproj.as_ref())?,
                    abs,
                )?,
                GgufVisionSource::AbsoluteFiles { model, mmproj } => gguf_vision_under(
                    base,
                    path_leaf(model.as_ref())?,
                    path_leaf(mmproj.as_ref())?,
                    abs,
                )?,
            },
        }),
        Model::Mlx(m) => Model::Mlx(Mlx {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::Torch(m) => Model::Torch(Torch {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::Openvino(m) => Model::Openvino(Openvino {
            source: dir_to_stored(&m.source, base, abs)?,
        }),
        Model::AppleFoundationText => Model::AppleFoundationText,
    })
}

/// Prefix every **relative** path in `stored` with `root` ([`Path::join`]).
///
/// Absolute `root` → Absolute* arms; relative `root` stays Relative*.
/// `stored` must already be Relative* form.
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
            source: GgufVisionSource::RelativeFiles { model, mmproj },
        }) => {
            let model_j = join(model.as_ref())?;
            let mmproj_j = join(mmproj.as_ref())?;
            Model::GgufVision(GgufVision {
                source: match (model_j, mmproj_j) {
                    (JoinedPath::Rel(model), JoinedPath::Rel(mmproj)) => {
                        GgufVisionSource::RelativeFiles { model, mmproj }
                    }
                    (JoinedPath::Abs(model), JoinedPath::Abs(mmproj)) => {
                        GgufVisionSource::AbsoluteFiles { model, mmproj }
                    }
                    _ => {
                        return Err(ModelStoredError::InvalidPath(
                            "mixed relative/absolute vision paths".to_owned(),
                        ))
                    }
                },
            })
        }
        Model::Mlx(Mlx {
            source: ModelSource::RelativeDir { dir },
        }) => Model::Mlx(Mlx {
            source: match join(dir.as_ref())? {
                JoinedPath::Rel(dir) => ModelSource::RelativeDir { dir },
                JoinedPath::Abs(dir) => ModelSource::AbsoluteDir { dir },
            },
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
        Model::AppleFoundationText => Model::AppleFoundationText,
        other => return Err(ModelStoredError::NotRelative(other.to_string())),
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
        ModelSource::RelativeDir { .. } | ModelSource::AbsoluteDir { .. } => {
            dir_source(path_from_base(base, abs)?)
        }
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
            Ok(GgufVisionSource::RelativeFiles { model, mmproj })
        }
        (JoinedPath::Abs(model), JoinedPath::Abs(mmproj)) => {
            if model.as_ref() == mmproj.as_ref() {
                return Err(ModelStoredError::CollidingPaths(model.to_string()));
            }
            Ok(GgufVisionSource::AbsoluteFiles { model, mmproj })
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

/// Last path segment of a local path (for re-homing into the store).
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

/// The download URL's trailing path segment as a safe model path — the filename
/// the fetcher lands on disk. The query string and fragment are stripped first,
/// so a signed URL (`…/w.gguf?token=…`) stores as `w.gguf`; the full URL remains
/// the model's identity. Rejects a URL that names no file, or a filename that
/// isn't a valid [`RepoSubpath`] (an escape sequence, …) once trimmed.
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{HfOrg, HfRepo, HfRepoName, RepoSubpath};

    use super::*;

    fn hf(org: &str, repo: &str) -> anyhow::Result<HfRepo> {
        Ok(HfRepo {
            org: HfOrg::try_new(org.to_owned())?,
            repo_name: HfRepoName::try_new(repo.to_owned())?,
            revision: None,
            auth_token: None,
        })
    }

    /// Absolute base — staging / loader shape. Platform-shaped because
    /// [`to_stored`] picks its `Absolute*` arms off `Path::is_absolute`, whose
    /// answer is per-platform: a bare `/store/entry` has a root but no drive
    /// prefix, so Windows reads it as relative, every re-home takes the
    /// `Relative*` branch, and `RelativePath` then rejects the leading `/` —
    /// the test fails on path spelling without reaching what it meant to assert.
    ///
    /// Note the asymmetry that makes this easy to miss: `AbsolutePath`'s own
    /// validator is a platform-independent string check that accepts the Unix
    /// spelling everywhere, so the `/elsewhere/w.gguf`-style *inputs* below need
    /// no such treatment — only bases, and the expectations derived from them.
    ///
    /// Forward slashes on both platforms because the joins normalize separators
    /// to `/`; only the prefix differs, which is what lets [`under`] compose
    /// every expectation from the one string.
    #[cfg(windows)]
    const BASE: &str = "C:/store/entry";
    #[cfg(not(windows))]
    const BASE: &str = "/store/entry";

    fn base() -> &'static Path {
        Path::new(BASE)
    }

    /// `relative` re-homed under [`BASE`], spelled the way the joins spell it.
    fn under(relative: &str) -> String {
        format!("{BASE}/{relative}")
    }

    /// Models root for the [`under_root`] tests, platform-shaped for the same
    /// reason as [`BASE`].
    #[cfg(windows)]
    const MODELS_ROOT: &str = "C:/ws/models";
    #[cfg(not(windows))]
    const MODELS_ROOT: &str = "/ws/models";

    /// Models-root-relative base — the store's `stored` shape (`<key>/blobs`).
    fn relative_base() -> &'static Path {
        Path::new("entry/blobs")
    }

    /// Every Relative path in `model` has no leading `/`.
    fn assert_all_local_paths_relative(model: &Model) -> anyhow::Result<()> {
        let paths: Vec<&str> = match model {
            Model::GgufText(GgufText {
                source: GgufTextSource::RelativeFile { path },
            }) => vec![path.as_ref()],
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::RelativeFiles { model, mmproj },
            }) => vec![model.as_ref(), mmproj.as_ref()],
            Model::Mlx(Mlx {
                source: ModelSource::RelativeDir { dir },
            })
            | Model::Torch(Torch {
                source: ModelSource::RelativeDir { dir },
            }) => vec![dir.as_ref()],
            Model::AppleFoundationText => vec![],
            other => anyhow::bail!("expected a Relative source form, got {other:?}"),
        };
        paths.iter().for_each(|p| {
            assert!(
                !p.starts_with('/'),
                "stored path must be relative to the model storage root: {p}"
            );
        });
        Ok(())
    }

    /// Manifest `stored` form: every storable variant + source arm lands under a
    /// models-root-relative base (no absolute paths).
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
            source: GgufVisionSource::HuggingFace {
                repo: hf("org", "vl")?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            },
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            },
        }),
    )]
    #[case::gguf_vision_url(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            },
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            },
        }),
    )]
    #[case::gguf_vision_local(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles { model: AbsolutePath::try_new("/a/model.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/b/mmproj.gguf".to_owned())?,
            },
        }),
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::RelativeFiles { model: RelativePath::try_new("entry/blobs/model.gguf".to_owned())?,
                mmproj: RelativePath::try_new("entry/blobs/mmproj.gguf".to_owned())?,
            },
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

    /// The on-disk filename is the URL's trailing segment with any query string
    /// or fragment stripped, so a signed URL lands as a plain `w.gguf`.
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
            match m {
                Model::Mlx(Mlx {
                    source: ModelSource::AbsoluteDir { dir },
                })
                | Model::Torch(Torch {
                    source: ModelSource::AbsoluteDir { dir },
                }) => Ok(dir.as_ref().to_owned()),
                _ => Err(anyhow::anyhow!("expected a local dir source")),
            }
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
            | Model::Torch(Torch {
                source: ModelSource::AbsoluteDir { dir },
            }) => dir,
            other => anyhow::bail!("expected absolute dir model, got {other:?}"),
        };
        assert_eq!(dir.as_ref(), BASE);
        Ok(())
    }

    #[rstest]
    #[case::hf(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace {
                repo: hf("org", "vl")?,
                model: RepoSubpath::try_new("model.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf")?,
                mmproj_sha256: None,
            },
        })
    )]
    #[case::url(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::Url {
                model: ResourceUrl::try_new("https://ex.com/model.gguf")?,
                model_sha256: None,
                mmproj: ResourceUrl::try_new("https://ex.com/mmproj.gguf")?,
                mmproj_sha256: None,
            },
        })
    )]
    #[case::local(
        Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles { model: AbsolutePath::try_new("/elsewhere/model.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/elsewhere/mmproj.gguf".to_owned())?,
            },
        })
    )]
    fn vision_both_files_land_under_base(#[case] declared: Model) -> anyhow::Result<()> {
        assert_eq!(
            to_stored(&declared, base())?,
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::AbsoluteFiles {
                    model: AbsolutePath::try_new(under("model.gguf"))?,
                    mmproj: AbsolutePath::try_new(under("mmproj.gguf"))?,
                },
            })
        );
        Ok(())
    }

    #[test]
    fn local_vision_rejects_colliding_leaf_names() -> anyhow::Result<()> {
        // Same leaf under different dirs would both land at base/weights.gguf.
        let local = Model::GgufVision(GgufVision {
            source: GgufVisionSource::AbsoluteFiles {
                model: AbsolutePath::try_new("/data/a/weights.gguf".to_owned())?,
                mmproj: AbsolutePath::try_new("/data/b/weights.gguf".to_owned())?,
            },
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
        // `Absolute` is already host-absolute; under_root only accepts `Relative`.
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
