//! [`ModelStorageKey`]: a model's identity segments (org, repo, revision,
//! file/prefix, URL, vault resource or absolute path), made filesystem-safe and
//! joined with `__` into one flat directory name. The same declaration always
//! yields the same key, so a re-run finds what it stored.
//!
//! The key omits the model type: the segments imply it in practice (a `.gguf`
//! name, formats in distinct repos) and the manifest records the typed model.
//! So `Mlx`, `Torch`, `Openvino` and `OnnxGenai` declared against the same HF
//! `org/repo` with no `prefix` share one entry.
//! The first type to land owns it and
//! [`crate::model::ModelArtifactStore::find`] refuses the others with
//! `ModelStoreError::TypeCollision`; a `prefix` or `models remove` separates them.
//!
//! Keys are capped at [`MAX_LEN`]: a longer slug keeps its head plus 8 hex chars
//! of the full slug's SHA-256, so distinct long models keep distinct keys.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use pipette_plan_types::{
    GgufTextSource, GgufVision, GgufVisionAbsoluteFiles, GgufVisionHfFiles, GgufVisionSource,
    GgufVisionSpeculativeSource, GgufVisionUrlFiles, GgufVisionVaultUrlFiles, HfRepo, Mlx, Model,
    ModelSource, OnnxGenai, Openvino, Torch, VaultPath,
};

use crate::entry::BLOBS_DIR_NAME;

const MAX_LEN: usize = 32;
const HASH_LEN: usize = 8;

/// Why a [`Model`] has no [`ModelStorageKey`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelStorageKeyError {
    /// Apple Foundation, or a store-relative (`Relative*`) source.
    #[error("model `{0}` has no storage key (not pullable / store-relative)")]
    NotStorable(String),
}

/// Flat storage key for a stored [`Model`] (see the module docs). Distinct
/// from [`Model`]'s `Display`, the CLI/reporting reference.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelStorageKey(String);

impl ModelStorageKey {
    /// Key for a declared `model`. `Relative*` sources and Apple Foundation are
    /// [`ModelStorageKeyError::NotStorable`].
    pub fn of(model: &Model) -> Result<Self, ModelStorageKeyError> {
        let not_storable = || ModelStorageKeyError::NotStorable(model.to_string());
        let segments = match model {
            Model::GgufText(m) => match &m.source {
                GgufTextSource::HuggingFace { repo, path, .. } => repo_segments(repo)
                    .into_iter()
                    .chain([path.to_string()])
                    .collect(),
                GgufTextSource::Url { url, .. } => vec![url.to_string()],
                GgufTextSource::VaultUrl { url, .. } => vault_segments(url),
                GgufTextSource::AbsoluteFile { path } => vec![path.to_string()],
                GgufTextSource::RelativeFile { .. } => return Err(not_storable()),
            },
            // Both files distinguish a VL instance: two can share a repo and
            // differ only in the weights or projector filename.
            Model::GgufVision(m) => match &m.source {
                GgufVisionSource::HuggingFace(GgufVisionHfFiles {
                    repo,
                    model,
                    mmproj,
                    ..
                }) => repo_segments(repo)
                    .into_iter()
                    .chain([model.to_string(), mmproj.to_string()])
                    .collect(),
                GgufVisionSource::Url(GgufVisionUrlFiles { model, mmproj, .. }) => {
                    vec![model.to_string(), mmproj.to_string()]
                }
                GgufVisionSource::VaultUrl(GgufVisionVaultUrlFiles { model, mmproj, .. }) => {
                    vault_segments(model)
                        .into_iter()
                        .chain(vault_segments(mmproj).into_iter().skip(1))
                        .collect()
                }
                GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj }) => {
                    vec![model.to_string(), mmproj.to_string()]
                }
                GgufVisionSource::RelativeFiles(_) => return Err(not_storable()),
            },
            // `spec` keeps a merged speculative model apart from the plain
            // `gguf_vision` over the same files: one entry holds one model type.
            Model::GgufVisionSpeculative(m) => {
                let vision = ModelStorageKey::of(&Model::GgufVision(GgufVision {
                    source: m.source.vision(),
                }))?;
                let draft = match &m.source {
                    GgufVisionSpeculativeSource::HuggingFace { draft, .. } => {
                        draft.as_ref().map(ToString::to_string)
                    }
                    GgufVisionSpeculativeSource::RelativeFiles { .. } => return Err(not_storable()),
                    GgufVisionSpeculativeSource::AbsoluteFiles { draft, .. } => {
                        draft.as_ref().map(ToString::to_string)
                    }
                    GgufVisionSpeculativeSource::Url { draft, .. } => {
                        draft.as_ref().map(ToString::to_string)
                    }
                    GgufVisionSpeculativeSource::VaultUrl { draft, .. } => {
                        draft.as_ref().map(|url| vault_segments(url).join("_"))
                    }
                };
                return Ok(ModelStorageKey(bound(slug_from(
                    &["spec".to_owned(), vision.0]
                        .into_iter()
                        .chain(draft)
                        .collect::<Vec<_>>(),
                ))));
            }
            Model::Mlx(Mlx { source })
            | Model::AppleCoreAi(pipette_plan_types::AppleCoreAi { source })
            | Model::Torch(Torch { source })
            | Model::Openvino(Openvino { source })
            | Model::OnnxGenai(OnnxGenai { source }) => {
                dir_segments(source).ok_or_else(not_storable)?
            }
            // `spec` and the drafter's own segments keep this apart from the
            // plain `mlx` entry over the same target: one entry holds one
            // model type, and swapping the drafter is a different model.
            Model::MlxSpeculative(m) | Model::TorchSpeculative(m) => ["spec".to_owned()]
                .into_iter()
                .chain(
                    m.sources()
                        .into_iter()
                        .map(|source| dir_segments(source).ok_or_else(not_storable))
                        .collect::<Result<Vec<_>, _>>()?
                        .concat(),
                )
                .collect(),
            Model::AppleFoundationText => return Err(not_storable()),
        };
        Ok(ModelStorageKey(bound(slug_from(&segments))))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// This model's store subdirectory, relative to the `models/` root.
    pub fn relative_dir(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }

    /// `<key>/blobs`, relative to the `models/` root: the dir
    /// [`crate::model::to_stored`] places the manifest's `stored` paths under.
    pub fn blobs_dir(&self) -> PathBuf {
        self.relative_dir().join(BLOBS_DIR_NAME)
    }
}

impl std::fmt::Display for ModelStorageKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `None` for a store-relative path, which names nothing outside the store.
fn dir_segments(source: &ModelSource) -> Option<Vec<String>> {
    match source {
        ModelSource::HuggingFace { repo, prefix } => Some(
            repo_segments(repo)
                .into_iter()
                .chain(prefix.iter().map(|p| p.to_string()))
                .collect(),
        ),
        ModelSource::VaultArchive { url, .. } => Some(vault_segments(url)),
        ModelSource::AbsoluteDir { dir } => Some(vec![dir.to_string()]),
        ModelSource::RelativeDir { .. } => None,
    }
}

/// `vault`, then the resource id: the route around it is the same for every
/// resource.
fn vault_segments(path: &VaultPath) -> Vec<String> {
    vec!["vault".to_owned(), path.resource_id().to_string()]
}

fn repo_segments(repo: &HfRepo) -> Vec<String> {
    [repo.org.to_string(), repo.repo_name.to_string()]
        .into_iter()
        .chain(repo.revision.iter().map(|r| r.to_string()))
        .collect()
}

/// Each run of characters outside `[A-Za-z0-9._-]` becomes one `_`, except a
/// leading or trailing run, which drops (`/on/disk/w.gguf` is `on_disk_w.gguf`).
/// Empty segments drop out, and segments join with `__` so the boundary reads
/// apart from an in-segment `_`. Shared with [`crate::runtime`] so both stores
/// flatten identically.
pub(crate) fn slug_from(segments: &[String]) -> String {
    segments
        .iter()
        .map(|s| sanitize(s))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("__")
}

fn sanitize(segment: &str) -> String {
    segment
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')))
        .filter(|run| !run.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

pub(crate) fn bound(slug: String) -> String {
    bound_to(slug, MAX_LEN)
}

/// [`bound`] against a caller-chosen cap. [`crate::runtime`] uses a wider one
/// to keep its full requirements digest legible; widening [`MAX_LEN`] would
/// re-key every stored model.
pub(crate) fn bound_to(slug: String, max_len: usize) -> String {
    if slug.len() <= max_len {
        return slug;
    }
    let mut hash = hex::encode(Sha256::digest(slug.as_bytes()));
    hash.truncate(HASH_LEN);
    let mut head = slug;
    head.truncate(head.floor_char_boundary(max_len - HASH_LEN - 1));
    format!("{}_{hash}", head.trim_end_matches('_'))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, GgufText, GgufVision, GgufVisionRelativeFiles, GgufVisionSpeculative,
        GgufVisionSpeculativeSource, HfOrg, HfRepoName, HfRevision, RepoSubpath, ResourceUrl,
    };

    use super::*;
    use crate::test_support::labelled;

    fn hf(org: &str, repo: &str, revision: Option<&str>) -> anyhow::Result<HfRepo> {
        Ok(HfRepo {
            org: HfOrg::try_new(org.to_owned())?,
            repo_name: HfRepoName::try_new(repo.to_owned())?,
            revision: revision
                .map(|r| HfRevision::try_new(r.to_owned()))
                .transpose()?,
            auth_token: None,
        })
    }

    fn gguf_text(source: GgufTextSource) -> Model {
        Model::GgufText(GgufText { source })
    }

    /// Also proves a vault source's `custom_label` does not change the key.
    #[rstest]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("meta", "llama", None)?, path: RepoSubpath::try_new("Q4.gguf")?, sha256: None }), "meta__llama__Q4.gguf")]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("org", "repo", None)?, path: RepoSubpath::try_new("sub/dir/w.gguf")?, sha256: None }), "org__repo__sub_dir_w.gguf")]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("org", "repo", Some("v1"))?, path: RepoSubpath::try_new("w.gguf")?, sha256: None }), "org__repo__v1__w.gguf")]
    #[case(gguf_text(GgufTextSource::Url { url: ResourceUrl::try_new("https://ex.com/m/w.gguf")?, sha256: None }), "https_ex.com_m_w.gguf")]
    #[case(Model::GgufVision(GgufVision { source: GgufVisionSource::HuggingFace(GgufVisionHfFiles { repo: hf("liquidai", "vl", None)?, model: RepoSubpath::try_new("q4.gguf")?, model_sha256: None, mmproj: RepoSubpath::try_new("mm.gguf")?, mmproj_sha256: None }) }), "liquidai__vl__q4.gguf__mm.gguf")]
    #[case(Model::Torch(Torch { source: ModelSource::HuggingFace { repo: hf("meta", "Llama", None)?, prefix: None } }), "meta__Llama")]
    #[case(Model::Torch(Torch { source: ModelSource::HuggingFace { repo: hf("meta", "Llama", None)?, prefix: Some(RepoSubpath::try_new("4bit")?) } }), "meta__Llama__4bit")]
    #[case(Model::Openvino(Openvino { source: ModelSource::HuggingFace { repo: hf("liquidai", "lfm2-ov", None)?, prefix: Some(RepoSubpath::try_new("int4-sym-cw")?) } }), "liquidai__lfm2-ov__int4-sym-cw")]
    #[case(Model::Mlx(Mlx { source: ModelSource::HuggingFace { repo: hf("liquidai", "lfm2-mlx", None)?, prefix: None } }), "liquidai__lfm2-mlx")]
    // A nested `prefix` collapses its separator to one `_`, so one ONNX repo's
    // per-provider variants each key to their own entry.
    #[case(Model::OnnxGenai(OnnxGenai { source: ModelSource::HuggingFace { repo: hf("org", "onnx", None)?, prefix: Some(RepoSubpath::try_new("cpu/int4")?) } }), "org__onnx__cpu_int4")]
    // Folds at `MAX_LEN`; the hash tail covers the `prefix`.
    #[case(Model::OnnxGenai(OnnxGenai { source: ModelSource::HuggingFace { repo: hf("microsoft", "Phi-3-mini-4k-instruct-onnx", None)?, prefix: Some(RepoSubpath::try_new("cpu_and_mobile/cpu-int4-rtn-block-32")?) } }), "microsoft__Phi-3-mini-4_671e5a0d")]
    #[case(gguf_text(GgufTextSource::VaultUrl { url: "/vault/resources/vault_0195c3e2/content".parse()?, sha256: None, principal_id: "org_1".parse()?, custom_label: None }), "vault__vault_0195c3e2")]
    #[case(Model::GgufVision(GgufVision { source: GgufVisionSource::VaultUrl(GgufVisionVaultUrlFiles { model: "/vault/resources/vault_a/content".parse()?, model_sha256: None, mmproj: "/vault/resources/vault_b/content".parse()?, mmproj_sha256: None, principal_id: "org_1".parse()?, custom_label: None }) }), "vault__vault_a__vault_b")]
    #[case(Model::Mlx(Mlx { source: ModelSource::VaultArchive { url: "/vault/resources/vault_0195c3e2/content".parse()?, sha256: None, principal_id: "user_1".parse()?, custom_label: None } }), "vault__vault_0195c3e2")]
    // `spec`, the vision key as one segment, then the drafter as one `vault_<id>` segment.
    #[case(Model::GgufVisionSpeculative(GgufVisionSpeculative { source: GgufVisionSpeculativeSource::VaultUrl { files: GgufVisionVaultUrlFiles { model: "/vault/resources/vault_a/content".parse()?, model_sha256: None, mmproj: "/vault/resources/vault_b/content".parse()?, mmproj_sha256: None, principal_id: "org_1".parse()?, custom_label: None }, draft: Some("/vault/resources/vault_c/content".parse()?), draft_sha256: None } }), "spec__vault__vault_a__v_b465a129")]
    fn declared_model_keys(#[case] model: Model, #[case] expected: &str) -> anyhow::Result<()> {
        let key = ModelStorageKey::of(&model)?;
        assert_eq!(key.as_str(), expected);
        assert!(!key.as_str().contains('/'));
        assert_eq!(key.relative_dir(), PathBuf::from(expected));
        assert_eq!(ModelStorageKey::of(&model)?, key);
        let labelled = labelled(&model)?;
        assert_eq!(labelled != model, expected.contains("vault__"));
        assert_eq!(ModelStorageKey::of(&labelled)?, key);
        Ok(())
    }

    /// The store refuses a second type in one entry, so these must not collide.
    #[test]
    fn a_speculative_model_keys_apart_from_its_plain_vl_twin() -> anyhow::Result<()> {
        let files = || -> anyhow::Result<GgufVisionHfFiles> {
            Ok(GgufVisionHfFiles {
                repo: hf("liquidai", "vl", None)?,
                model: RepoSubpath::try_new("q4.gguf")?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mm.gguf")?,
                mmproj_sha256: None,
            })
        };
        let speculative = |draft: Option<&str>| -> anyhow::Result<Model> {
            Ok(Model::GgufVisionSpeculative(GgufVisionSpeculative {
                source: GgufVisionSpeculativeSource::HuggingFace {
                    files: files()?,
                    draft: draft
                        .map(|d| RepoSubpath::try_new(d.to_owned()))
                        .transpose()?,
                    draft_sha256: None,
                },
            }))
        };
        let vision = Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace(files()?),
        });

        let plain = ModelStorageKey::of(&vision)?;
        let merged = ModelStorageKey::of(&speculative(None)?)?;
        let separate = ModelStorageKey::of(&speculative(Some("draft.gguf"))?)?;

        assert_eq!(merged.as_str(), "spec__liquidai__vl__q4._83c51347");
        assert_ne!(plain, merged, "one entry cannot hold both types");
        assert_ne!(merged, separate, "the drafter is part of the identity");
        [&plain, &merged, &separate]
            .iter()
            .for_each(|key| assert!(!key.as_str().contains('/')));
        Ok(())
    }

    #[test]
    fn an_mlx_pair_keys_apart_from_its_target_and_from_another_drafter() -> anyhow::Result<()> {
        use pipette_plan_types::MlxSpeculative;

        let source = |org: &str, repo: &str| -> anyhow::Result<ModelSource> {
            Ok(ModelSource::HuggingFace {
                repo: hf(org, repo, None)?,
                prefix: None,
            })
        };
        let pair = |draft_repo: &str| -> anyhow::Result<Model> {
            Ok(Model::MlxSpeculative(MlxSpeculative {
                target: source("liquidai", "vl-mlx")?,
                draft: source("liquidai", draft_repo)?,
            }))
        };

        let plain = ModelStorageKey::of(&Model::Mlx(Mlx {
            source: source("liquidai", "vl-mlx")?,
        }))?;
        let dspark = ModelStorageKey::of(&pair("vl-dspark")?)?;
        let dflash = ModelStorageKey::of(&pair("vl-dflash")?)?;

        // Folds at `MAX_LEN`; the hash tail is what keeps the drafters apart.
        assert_eq!(dspark.as_str(), "spec__liquidai__vl-mlx_62ee9a91");
        assert_ne!(plain, dspark, "one entry cannot hold both types");
        assert_ne!(dspark, dflash, "the drafter is part of the identity");
        [&plain, &dspark, &dflash]
            .iter()
            .for_each(|key| assert!(!key.as_str().contains('/')));
        Ok(())
    }

    #[rstest]
    #[case::relative_draft(false, true)]
    #[case::relative_target(true, false)]
    #[case::both_relative(true, true)]
    fn an_mlx_pair_with_a_store_relative_half_is_not_storable(
        #[case] relative_target: bool,
        #[case] relative_draft: bool,
    ) -> anyhow::Result<()> {
        use pipette_plan_types::{MlxSpeculative, RelativePath};

        let half = |relative: bool, name: &str| -> anyhow::Result<ModelSource> {
            Ok(if relative {
                ModelSource::RelativeDir {
                    dir: RelativePath::try_new(name.to_owned())?,
                }
            } else {
                ModelSource::HuggingFace {
                    repo: hf("liquidai", name, None)?,
                    prefix: None,
                }
            })
        };
        let model = Model::MlxSpeculative(MlxSpeculative {
            target: half(relative_target, "target")?,
            draft: half(relative_draft, "draft")?,
        });
        assert!(matches!(
            ModelStorageKey::of(&model),
            Err(ModelStorageKeyError::NotStorable(_))
        ));
        Ok(())
    }

    #[test]
    fn directory_models_of_the_same_repo_alias() -> anyhow::Result<()> {
        let same_repo = || -> anyhow::Result<ModelSource> {
            Ok(ModelSource::HuggingFace {
                repo: hf("org", "repo", None)?,
                prefix: None,
            })
        };
        let torch = ModelStorageKey::of(&Model::Torch(Torch {
            source: same_repo()?,
        }))?;
        let mlx = ModelStorageKey::of(&Model::Mlx(Mlx {
            source: same_repo()?,
        }))?;
        let openvino = ModelStorageKey::of(&Model::Openvino(Openvino {
            source: same_repo()?,
        }))?;
        let onnx_genai = ModelStorageKey::of(&Model::OnnxGenai(OnnxGenai {
            source: same_repo()?,
        }))?;
        assert_eq!(torch, mlx);
        assert_eq!(torch, openvino);
        assert_eq!(torch, onnx_genai);
        Ok(())
    }

    #[test]
    fn folded_onnx_variants_of_one_repo_keep_distinct_keys() -> anyhow::Result<()> {
        let variant = |prefix: &str| -> anyhow::Result<ModelStorageKey> {
            Ok(ModelStorageKey::of(&Model::OnnxGenai(OnnxGenai {
                source: ModelSource::HuggingFace {
                    repo: hf("microsoft", "Phi-3-mini-4k-instruct-onnx", None)?,
                    prefix: Some(RepoSubpath::try_new(prefix)?),
                },
            }))?)
        };
        let int4 = variant("cpu_and_mobile/cpu-int4-rtn-block-32")?;
        let fp32 = variant("cpu_and_mobile/cpu-fp32")?;
        assert_ne!(
            int4.as_str(),
            fp32.as_str(),
            "two precisions of one repo must not share an entry"
        );
        assert!(int4.as_str().len() <= 32, "key stays bounded");
        Ok(())
    }

    #[rstest]
    #[case::afm(Model::AppleFoundationText)]
    #[case::gguf_relative(gguf_text(GgufTextSource::RelativeFile {
        path: pipette_plan_types::RelativePath::try_new("entry/blobs/w.gguf".to_owned())?,
    }))]
    #[case::vision_relative(Model::GgufVision(GgufVision {
        source: GgufVisionSource::RelativeFiles(GgufVisionRelativeFiles {
            model: pipette_plan_types::RelativePath::try_new("entry/blobs/m.gguf".to_owned())?,
            mmproj: pipette_plan_types::RelativePath::try_new("entry/blobs/mm.gguf".to_owned())?,
        }),
    }))]
    #[case::mlx_relative(Model::Mlx(Mlx {
        source: ModelSource::RelativeDir {
            dir: pipette_plan_types::RelativePath::try_new("entry/blobs".to_owned())?,
        },
    }))]
    fn no_storage_key_for_non_declared(#[case] model: Model) -> anyhow::Result<()> {
        assert!(matches!(
            ModelStorageKey::of(&model),
            Err(ModelStorageKeyError::NotStorable(_))
        ));
        Ok(())
    }

    #[rstest]
    #[case(gguf_text(GgufTextSource::AbsoluteFile { path: AbsolutePath::try_new("/on/disk/w.gguf")? }), "on_disk_w.gguf")]
    #[case(Model::Mlx(Mlx { source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/x")? } }), "models_x")]
    #[case(Model::GgufVision(GgufVision { source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model: AbsolutePath::try_new("/a/m.gguf")?, mmproj: AbsolutePath::try_new("/a/mm.gguf")? }) }), "a_m.gguf__a_mm.gguf")]
    fn absolute_local_models_have_path_keys(
        #[case] model: Model,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let key = ModelStorageKey::of(&model)?;
        assert_eq!(key.as_str(), expected);
        assert_eq!(ModelStorageKey::of(&model)?, key);
        Ok(())
    }

    #[test]
    fn a_non_ascii_slug_folds_on_a_char_boundary() -> anyhow::Result<()> {
        let folded = bound_to("é".repeat(40), 21);
        anyhow::ensure!(folded.len() <= 21, "folded to {} bytes", folded.len());
        anyhow::ensure!(folded.starts_with("éééééé_"), "head: {folded}");
        Ok(())
    }

    #[test]
    fn long_slugs_are_folded_and_stay_distinct() -> anyhow::Result<()> {
        let long = |repo: &str| -> anyhow::Result<ModelStorageKey> {
            Ok(ModelStorageKey::of(&gguf_text(
                GgufTextSource::HuggingFace {
                    repo: hf("really-long-shared-organization", repo, None)?,
                    path: RepoSubpath::try_new("weights-q4-k-m.gguf")?,
                    sha256: None,
                },
            ))?)
        };
        let a = long("repo-a")?;
        let b = long("repo-b")?;
        assert!(a.as_str().len() <= MAX_LEN && b.as_str().len() <= MAX_LEN);
        assert!(a.as_str().starts_with("really-long-shared-orga"));
        assert_ne!(a, b);
        Ok(())
    }
}
