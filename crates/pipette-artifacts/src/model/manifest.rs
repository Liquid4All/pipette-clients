//! [`ModelManifest`] — the on-disk record for one stored model.
//!
//! A stored model is one directory under the workspace `models/` tree, named by
//! its [`crate::model::ModelStorageKey`] segment. The entry separates
//! metadata from payload:
//!
//! ```text
//! models/<key>/
//! ├── manifest.toml         # this record, at the entry root
//! └── blobs/                # the model's files, isolated from metadata
//! ```
//!
//! The manifest records the model twice: `declared` (as authored in the plan —
//! the reporting identity) and `stored` (the same model with `Local` source
//! arms, paths relative to the model-storage root, i.e. under `<key>/blobs`).
//! Build `stored` with [`crate::model::to_stored`] under
//! [`crate::model::ModelStorageKey::blobs_dir`]; bind those relative
//! paths under a concrete models dir with [`ModelManifest::bind_under`].
//!
//! This type is a **serde record** only. Loading/writing policy (TOML decode,
//! auth strip, version, storage key, `stored` drift) lives in
//! [`crate::model`].

use std::path::Path;

use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_plan_types::Model;

use super::stored::{under_root, ModelStoredError};

pub const MANIFEST_VERSION: u32 = 2;
/// Aliases [`crate::entry::BLOBS_DIR_NAME`] so model and runtime stores agree.
pub const BLOBS_DIR_NAME: &str = crate::entry::BLOBS_DIR_NAME;

/// On-disk model store entry metadata (`manifest.toml`).
///
/// Field types are enforced by serde. Domain rules are applied by the model
/// store when reading or writing entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub manifest_version: u32,
    /// When the fetch started — stamped before the download runs, not after.
    #[serde(with = "time::serde::rfc3339")]
    pub fetched_at: OffsetDateTime,
    /// When an `ensure` last resolved this entry — the eviction order. Seeded to
    /// `fetched_at` on publish, rewritten best-effort on every hit.
    #[serde(with = "time::serde::rfc3339")]
    pub last_used_at: OffsetDateTime,
    /// Plan identity (store strips auth before accepting/persisting).
    pub declared: Model,
    /// Local form of `declared` under
    /// [`crate::model::ModelStorageKey::blobs_dir`] (store-enforced).
    pub stored: Model,
    /// Bytes `blobs/` occupied at publish, so the quota sweep can total a store
    /// by reading manifests instead of walking every payload file.
    ///
    /// Measures `blobs/` alone, not the entry directory: `last_used_at` is
    /// rewritten on every cache hit, and a total that included `manifest.toml`
    /// would drift with it. `None` on entries published before this field, or
    /// where the payload lives outside the store (a docker image is held by the
    /// daemon) — [`crate::quota`] falls back to measuring the payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blobs_bytes: Option<u64>,
}

impl ModelManifest {
    /// Loader form: prefix `stored`'s models-root-relative Local paths with
    /// `models_dir` ([`Path::join`]). Does not re-derive from `declared`.
    pub fn bind_under(&self, models_dir: &Path) -> Result<Model, ModelStoredError> {
        under_root(&self.stored, models_dir)
    }

    /// Absolute payload paths this entry should have on disk, under
    /// `models_dir`. Empty for a model that stores nothing (Apple Foundation is
    /// supplied by the OS), and empty when `stored` will not bind — which the
    /// store already reports as corrupt.
    ///
    /// Telling a live entry from a husk: a manifest is not evidence its bytes
    /// are still there.
    pub(crate) fn payload_paths(&self, models_dir: &Path) -> Vec<std::path::PathBuf> {
        use pipette_plan_types::{
            GgufText, GgufTextSource, GgufVision, GgufVisionSource, Mlx, ModelSource,
        };
        let Ok(bound) = self.bind_under(models_dir) else {
            return Vec::new();
        };
        let abs = |p: &str| std::path::PathBuf::from(p);
        match bound {
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile { path },
            }) => vec![abs(path.as_ref())],
            Model::GgufVision(GgufVision {
                source: GgufVisionSource::AbsoluteFiles { model, mmproj },
            }) => vec![abs(model.as_ref()), abs(mmproj.as_ref())],
            Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir },
            }) => vec![abs(dir.as_ref())],
            _ => Vec::new(),
        }
    }

    /// RFC 3339 form of [`Self::fetched_at`] for display / logs.
    pub fn fetched_at_rfc3339(&self) -> Result<String, time::error::Format> {
        self.fetched_at.format(&Rfc3339)
    }

    /// RFC 3339 form of [`Self::last_used_at`] for display / logs.
    pub fn last_used_at_rfc3339(&self) -> Result<String, time::error::Format> {
        self.last_used_at.format(&Rfc3339)
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        AbsolutePath, GgufText, GgufTextSource, HfOrg, HfRepo, HfRepoName, RelativePath,
        RepoSubpath,
    };

    use super::super::key::ModelStorageKey;
    use super::super::stored::to_stored;
    use super::*;

    fn sample_manifest() -> anyhow::Result<ModelManifest> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("meta".to_owned())?,
                    repo_name: HfRepoName::try_new("llama".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        });
        let key = ModelStorageKey::of(&declared)?;
        let fetched_at = OffsetDateTime::parse("2026-01-01T00:00:00Z", &Rfc3339)?;
        Ok(ModelManifest {
            manifest_version: MANIFEST_VERSION,
            fetched_at,
            last_used_at: OffsetDateTime::parse("2026-02-02T00:00:00Z", &Rfc3339)?,
            stored: to_stored(&declared, &key.blobs_dir())?,
            declared,
            blobs_bytes: Some(4096),
        })
    }

    #[test]
    fn stored_field_is_to_stored_under_key_blobs_dir() -> anyhow::Result<()> {
        let declared = Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo {
                    org: HfOrg::try_new("meta".to_owned())?,
                    repo_name: HfRepoName::try_new("llama".to_owned())?,
                    revision: None,
                    auth_token: None,
                },
                path: RepoSubpath::try_new("Q4.gguf")?,
                sha256: None,
            },
        });
        let key = ModelStorageKey::of(&declared)?;
        assert_eq!(
            key.blobs_dir(),
            std::path::PathBuf::from("meta__llama__Q4.gguf/blobs")
        );
        assert_eq!(
            to_stored(&declared, &key.blobs_dir())?,
            Model::GgufText(GgufText {
                source: GgufTextSource::RelativeFile {
                    path: RelativePath::try_new("meta__llama__Q4.gguf/blobs/Q4.gguf".to_owned())?,
                },
            })
        );
        Ok(())
    }

    #[test]
    fn bind_under_joins_models_dir_onto_stored_paths() -> anyhow::Result<()> {
        let manifest = sample_manifest()?;
        assert_eq!(
            manifest.stored,
            Model::GgufText(GgufText {
                source: GgufTextSource::RelativeFile {
                    path: RelativePath::try_new("meta__llama__Q4.gguf/blobs/Q4.gguf".to_owned())?,
                },
            })
        );
        assert_eq!(
            manifest.bind_under(Path::new("/ws/models"))?,
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile {
                    path: AbsolutePath::try_new(
                        "/ws/models/meta__llama__Q4.gguf/blobs/Q4.gguf".to_owned()
                    )?,
                },
            })
        );
        // Relative models_dir stays relative (Relative arm, not Local).
        assert_eq!(
            manifest.bind_under(Path::new("models"))?,
            Model::GgufText(GgufText {
                source: GgufTextSource::RelativeFile {
                    path: RelativePath::try_new(
                        "models/meta__llama__Q4.gguf/blobs/Q4.gguf".to_owned()
                    )?,
                },
            })
        );
        Ok(())
    }

    #[test]
    fn serde_round_trips_fields() -> anyhow::Result<()> {
        let manifest = sample_manifest()?;
        let parsed: ModelManifest = toml::from_str(&toml::to_string(&manifest)?)?;
        assert_eq!(parsed, manifest);
        assert_ne!(
            parsed.last_used_at, parsed.fetched_at,
            "the two timestamps are carried independently"
        );
        Ok(())
    }

    /// No migration: a v1 record has no `last_used_at`, so it does not decode.
    #[test]
    fn a_manifest_without_last_used_at_does_not_deserialize() -> anyhow::Result<()> {
        let rendered = toml::to_string(&sample_manifest()?)?;
        let mut table: toml::Table = toml::from_str(&rendered)?;
        table.remove("last_used_at");
        assert!(toml::from_str::<ModelManifest>(&toml::to_string(&table)?).is_err());
        Ok(())
    }
}
