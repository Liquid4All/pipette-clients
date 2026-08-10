//! [`ModelStorageKey`] — a filesystem-safe, flat storage identity for a [`Model`].
//!
//! The store addresses a model by a single flat directory name rather than a
//! nested `<org>/<repo>/<file>` tree. The key is the model's identity segments
//! (org, repo, revision, file/prefix, URL, or **absolute** local path) each
//! normalized to a filesystem-safe token and joined with `__`. The same
//! declaration always yields the same key, so a re-run of the plan finds the
//! artifact it stored; the segments are chosen so distinct models generally get
//! distinct keys.
//!
//! Keys are built only for **declared / fetchable or abs-local** forms:
//! HuggingFace, Url, and Absolute* path arms. Store-relative `Relative*` arms
//! (manifest `stored` / entry layout) and Apple Foundation are
//! [`ModelStorageKeyError::NotStorable`] — no key is constructed for them.
//!
//! The key omits the model type: type is implied by the segments in practice
//! (gguf files carry a `.gguf` name; MLX/Torch/OpenVINO weights live in distinct
//! repos), and the manifest embeds the full typed [`Model`] anyway. The one
//! aliasing case is two directory-shaped models — `Mlx`, `Torch`, `Openvino` —
//! declared against the *same* HF `org/repo`: they share a key. Sound, since the
//! entry is a snapshot of that repo, but a repo publishing two formats has to
//! separate them by `prefix` rather than by model type.
//!
//! Keys are capped at [`MAX_LEN`] characters. A longer slug keeps its head and
//! gets 8 hex chars of the full slug's SHA-256 as its tail, so the on-disk name
//! stays bounded while distinct long models keep distinct keys.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

use pipette_plan_types::{
    GgufTextSource, GgufVisionSource, HfRepo, Mlx, Model, ModelSource, Openvino, Torch,
};

use crate::entry::BLOBS_DIR_NAME;

/// Maximum key length. A slug over this is folded to `<head>_<hash8>`.
const MAX_LEN: usize = 32;
/// Hex chars of SHA-256 kept as the disambiguating tail when folding.
const HASH_LEN: usize = 8;

/// Why a [`Model`] has no [`ModelStorageKey`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelStorageKeyError {
    /// Not a warehouse identity: Apple Foundation, or a store-relative
    /// (`Relative*`) source that is never used as a storage key.
    #[error("model `{0}` has no storage key (not pullable / store-relative)")]
    NotStorable(String),
}

/// Flat storage key for a stored [`Model`] — its identity segments, normalized
/// and `__`-joined (see the module docs). HF, URL, and absolute-local sources
/// have keys; store-relative `Relative*` and Apple Foundation do not. Distinct
/// from [`Model`]'s `Display` (CLI/reporting reference).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelStorageKey(String);

impl ModelStorageKey {
    /// Key for a **declared** fetchable or abs-local `model`, or
    /// [`ModelStorageKeyError::NotStorable`] when no store entry should be keyed
    /// from this form. Never longer than [`MAX_LEN`].
    ///
    /// Exhaustive match on kind + source; `Relative*` and Apple Foundation →
    /// `NotStorable` (no segments built).
    pub fn of(model: &Model) -> Result<Self, ModelStorageKeyError> {
        let not_storable = || ModelStorageKeyError::NotStorable(model.to_string());
        let segments = match model {
            Model::GgufText(m) => match &m.source {
                GgufTextSource::HuggingFace { repo, path, .. } => repo_segments(repo)
                    .into_iter()
                    .chain([path.to_string()])
                    .collect(),
                GgufTextSource::Url { url, .. } => vec![url.to_string()],
                GgufTextSource::AbsoluteFile { path } => vec![path.to_string()],
                GgufTextSource::RelativeFile { .. } => return Err(not_storable()),
            },
            // Both files distinguish a VL instance: two can share a repo and
            // differ only in the weights or projector filename.
            Model::GgufVision(m) => match &m.source {
                GgufVisionSource::HuggingFace {
                    repo,
                    model,
                    mmproj,
                    ..
                } => repo_segments(repo)
                    .into_iter()
                    .chain([model.to_string(), mmproj.to_string()])
                    .collect(),
                GgufVisionSource::Url { model, mmproj, .. } => {
                    vec![model.to_string(), mmproj.to_string()]
                }
                GgufVisionSource::AbsoluteFiles { model, mmproj } => {
                    vec![model.to_string(), mmproj.to_string()]
                }
                GgufVisionSource::RelativeFiles { .. } => return Err(not_storable()),
            },
            Model::Mlx(Mlx { source })
            | Model::Torch(Torch { source })
            | Model::Openvino(Openvino { source }) => match source {
                ModelSource::HuggingFace { repo, prefix } => repo_segments(repo)
                    .into_iter()
                    .chain(prefix.iter().map(|p| p.to_string()))
                    .collect(),
                ModelSource::AbsoluteDir { dir } => vec![dir.to_string()],
                ModelSource::RelativeDir { .. } => return Err(not_storable()),
            },
            Model::AppleFoundationText => return Err(not_storable()),
        };
        Ok(ModelStorageKey(bound(slug_from(&segments))))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// This model's store subdirectory, relative to the `models/` root. A single
    /// flat component (the key), never a nested path.
    pub fn relative_dir(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }

    /// Payload dir under the entry, relative to the `models/` root: `<key>/blobs`.
    /// Pass to [`crate::model::to_stored`] for the manifest `stored` field.
    pub fn blobs_dir(&self) -> PathBuf {
        self.relative_dir().join(BLOBS_DIR_NAME)
    }
}

impl std::fmt::Display for ModelStorageKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `org`, `repo`, and (when pinned) `revision` — the shared prefix of every HF
/// source's identity.
fn repo_segments(repo: &HfRepo) -> Vec<String> {
    [repo.org.to_string(), repo.repo_name.to_string()]
        .into_iter()
        .chain(repo.revision.iter().map(|r| r.to_string()))
        .collect()
}

/// Normalize each segment to the filesystem-safe charset (`[A-Za-z0-9._-]`) —
/// every other character (path `/`, the HF `@`/`:` joiners, URL `://`) becomes a
/// single `_` — then join the segments with `__`. The wider `__` join reads as
/// the segment boundary against the single `_` used within a segment. Empty
/// segments (e.g. a leading `/` on a local path) drop out.
///
/// `pub(crate)` so [`crate::runtime`] shares the exact same
/// normalization — the model and runtime stores flatten identity segments the
/// same way.
pub(crate) fn slug_from(segments: &[String]) -> String {
    segments
        .iter()
        .map(|s| sanitize(s))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("__")
}

/// Split on unsafe chars and rejoin the non-empty runs with `_` — this collapses
/// every run of unsafe chars (and any leading/trailing run) to a single `_`.
fn sanitize(segment: &str) -> String {
    segment
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')))
        .filter(|run| !run.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Cap the slug at [`MAX_LEN`]: a longer slug keeps its head and gets an 8-hex
/// SHA-256 tail of the full slug, so distinct long models stay distinct.
pub(crate) fn bound(slug: String) -> String {
    bound_to(slug, MAX_LEN)
}

/// [`bound`] against a caller-chosen cap. The slug is ASCII, so byte-slicing
/// the head is char-safe.
///
/// `pub(crate)` so [`crate::runtime`] folds identically while choosing its own
/// ceiling: a runtime key ends in a full requirements digest that is worth
/// keeping legible on disk, whereas a model key is already readable coordinates
/// and stays at the tighter [`MAX_LEN`] rather than re-keying every stored
/// model.
pub(crate) fn bound_to(slug: String, max_len: usize) -> String {
    if slug.len() <= max_len {
        return slug;
    }
    let hash = hex::encode(Sha256::digest(slug.as_bytes()));
    let head = slug[..max_len - HASH_LEN - 1].trim_end_matches('_');
    format!("{head}_{}", &hash[..HASH_LEN])
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, GgufText, GgufVision, HfOrg, HfRepoName, HfRevision, RepoSubpath, ResourceUrl,
    };

    use super::*;

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

    /// Exact key per declared model, covering every source shape and the
    /// normalization rules. Each case asserts the flat-component invariant and
    /// determinism too, so the table is the single home for "declared → key".
    #[rstest]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("meta", "llama", None)?, path: RepoSubpath::try_new("Q4.gguf")?, sha256: None }), "meta__llama__Q4.gguf")]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("org", "repo", None)?, path: RepoSubpath::try_new("sub/dir/w.gguf")?, sha256: None }), "org__repo__sub_dir_w.gguf")]
    #[case(gguf_text(GgufTextSource::HuggingFace { repo: hf("org", "repo", Some("v1"))?, path: RepoSubpath::try_new("w.gguf")?, sha256: None }), "org__repo__v1__w.gguf")]
    #[case(gguf_text(GgufTextSource::Url { url: ResourceUrl::try_new("https://ex.com/m/w.gguf")?, sha256: None }), "https_ex.com_m_w.gguf")]
    #[case(Model::GgufVision(GgufVision { source: GgufVisionSource::HuggingFace { repo: hf("liquidai", "vl", None)?, model: RepoSubpath::try_new("q4.gguf")?, model_sha256: None, mmproj: RepoSubpath::try_new("mm.gguf")?, mmproj_sha256: None } }), "liquidai__vl__q4.gguf__mm.gguf")]
    #[case(Model::Torch(Torch { source: ModelSource::HuggingFace { repo: hf("meta", "Llama", None)?, prefix: None } }), "meta__Llama")]
    #[case(Model::Torch(Torch { source: ModelSource::HuggingFace { repo: hf("meta", "Llama", None)?, prefix: Some(RepoSubpath::try_new("4bit")?) } }), "meta__Llama__4bit")]
    #[case(Model::Openvino(Openvino { source: ModelSource::HuggingFace { repo: hf("liquidai", "lfm2-ov", None)?, prefix: Some(RepoSubpath::try_new("int4-sym-cw")?) } }), "liquidai__lfm2-ov__int4-sym-cw")]
    fn declared_model_keys(#[case] model: Model, #[case] expected: &str) -> anyhow::Result<()> {
        let key = ModelStorageKey::of(&model)?;
        assert_eq!(key.as_str(), expected);
        assert!(!key.as_str().contains('/')); // single flat component
        assert_eq!(key.relative_dir(), PathBuf::from(expected));
        assert_eq!(ModelStorageKey::of(&model)?, key); // deterministic
        Ok(())
    }

    #[test]
    fn directory_models_of_the_same_repo_alias() -> anyhow::Result<()> {
        // Documented aliasing — see the module docs. The same HF repo keys
        // identically across every directory-shaped model type.
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
        assert_eq!(torch, mlx);
        assert_eq!(torch, openvino);
        Ok(())
    }

    #[rstest]
    #[case::afm(Model::AppleFoundationText)]
    #[case::gguf_relative(gguf_text(GgufTextSource::RelativeFile {
        path: pipette_plan_types::RelativePath::try_new("entry/blobs/w.gguf".to_owned())?,
    }))]
    #[case::vision_relative(Model::GgufVision(GgufVision {
        source: GgufVisionSource::RelativeFiles {
            model: pipette_plan_types::RelativePath::try_new("entry/blobs/m.gguf".to_owned())?,
            mmproj: pipette_plan_types::RelativePath::try_new("entry/blobs/mm.gguf".to_owned())?,
        },
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

    /// Abs-local authoring paths are warehouse identity (import into the store).
    #[rstest]
    #[case(gguf_text(GgufTextSource::AbsoluteFile { path: AbsolutePath::try_new("/on/disk/w.gguf")? }), "on_disk_w.gguf")]
    #[case(Model::Mlx(Mlx { source: ModelSource::AbsoluteDir { dir: AbsolutePath::try_new("/models/x")? } }), "models_x")]
    #[case(Model::GgufVision(GgufVision { source: GgufVisionSource::AbsoluteFiles { model: AbsolutePath::try_new("/a/m.gguf")?, mmproj: AbsolutePath::try_new("/a/mm.gguf")? } }), "a_m.gguf__a_mm.gguf")]
    fn absolute_local_models_have_path_keys(
        #[case] model: Model,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let key = ModelStorageKey::of(&model)?;
        assert_eq!(key.as_str(), expected);
        assert_eq!(ModelStorageKey::of(&model)?, key);
        Ok(())
    }

    /// A slug over the cap is folded to a bounded `<head>_<hash>`, and two long
    /// models sharing a head stay distinct via the hash tail.
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
        assert_ne!(a, b); // shared head, distinct hash tail
        Ok(())
    }
}
