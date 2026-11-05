//! [`ModelManifest`]: the `manifest.toml` at the root of a stored model's
//! `models/<key>/` entry, next to its `blobs/` payload.
//!
//! A serde record only. The store applies the policy (auth strip, version,
//! storage key, `stored` drift) when it reads or writes one.

use std::path::Path;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_plan_types::Model;

use super::stored::{ModelStoredError, under_root};

pub const MANIFEST_VERSION: u32 = 2;
/// Aliases [`crate::entry::BLOBS_DIR_NAME`] so model and runtime stores agree.
pub const BLOBS_DIR_NAME: &str = crate::entry::BLOBS_DIR_NAME;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub manifest_version: u32,
    /// When the fetch started: stamped before the download runs, not after.
    #[serde(with = "time::serde::rfc3339")]
    pub fetched_at: OffsetDateTime,
    /// When an `ensure` last resolved this entry: the eviction order. Seeded to
    /// `fetched_at` on publish, rewritten best-effort on every hit.
    #[serde(with = "time::serde::rfc3339")]
    pub last_used_at: OffsetDateTime,
    /// The model as the plan declared it, auth stripped: the reporting identity.
    pub declared: Model,
    /// `declared` with `Relative*` sources under
    /// [`crate::model::ModelStorageKey::blobs_dir`], relative to the models root.
    pub stored: Model,
    /// Bytes `blobs/` occupied at publish, so the quota sweep can total a store
    /// from manifests instead of walking every payload file. Excludes
    /// `manifest.toml`, which every hit rewrites. When `None`, [`crate::quota`]
    /// measures the payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blobs_bytes: Option<u64>,
}

impl ModelManifest {
    /// `stored` with its paths joined under `models_dir`. Does not re-derive
    /// from `declared`.
    pub fn bind_under(&self, models_dir: &Path) -> Result<Model, ModelStoredError> {
        under_root(&self.stored, models_dir)
    }

    /// Payload paths this entry should have under `models_dir`, to tell a live
    /// entry from a husk whose bytes are gone. Only GGUF text, GGUF vision and
    /// MLX list paths; empty for the rest and when `stored` will not bind (the
    /// store reports that as corrupt).
    pub(crate) fn payload_paths(&self, models_dir: &Path) -> Vec<std::path::PathBuf> {
        use pipette_plan_types::{
            GgufText, GgufTextSource, GgufVision, GgufVisionAbsoluteFiles, GgufVisionSource, Mlx,
            ModelSource,
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
                source: GgufVisionSource::AbsoluteFiles(GgufVisionAbsoluteFiles { model, mmproj }),
            }) => vec![abs(model.as_ref()), abs(mmproj.as_ref())],
            Model::Mlx(Mlx {
                source: ModelSource::AbsoluteDir { dir },
            }) => vec![abs(dir.as_ref())],
            Model::GgufText(_)
            | Model::GgufVision(_)
            | Model::GgufVisionSpeculative(_)
            | Model::Mlx(_)
            | Model::MlxSpeculative(_)
            | Model::TorchSpeculative(_)
            | Model::Torch(_)
            | Model::Openvino(_)
            | Model::OnnxGenai(_)
            | Model::AppleFoundationText
            | Model::AppleCoreAi(_) => Vec::new(),
        }
    }

    pub fn fetched_at_rfc3339(&self) -> Result<String, time::error::Format> {
        self.fetched_at.format(&Rfc3339)
    }

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
        // Platform-shaped: `bind_under` picks its `Absolute*` arm off
        // `Path::is_absolute`, and `/ws/models` is relative on Windows.
        #[cfg(windows)]
        const MODELS_ROOT: &str = "C:/ws/models";
        #[cfg(not(windows))]
        const MODELS_ROOT: &str = "/ws/models";
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
            manifest.bind_under(Path::new(MODELS_ROOT))?,
            Model::GgufText(GgufText {
                source: GgufTextSource::AbsoluteFile {
                    path: AbsolutePath::try_new(format!(
                        "{MODELS_ROOT}/meta__llama__Q4.gguf/blobs/Q4.gguf"
                    ))?,
                },
            })
        );
        // A relative models_dir keeps the Relative arm.
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
