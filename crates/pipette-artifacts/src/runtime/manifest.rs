//! [`RuntimeManifest`] — the on-disk record for one installed runtime, the
//! runtime-side counterpart to [`crate::model::ModelManifest`].
//!
//! An installed runtime is one directory under the workspace `runtimes/` tree,
//! named by its [`RuntimeStorageKey`] segment — the same `<key>/manifest.toml` +
//! `<key>/blobs/` layout the model store uses:
//!
//! ```text
//! runtimes/<key>/
//! ├── manifest.toml         # this record, at the entry root
//! └── blobs/                # payload (empty for docker)
//! ```
//!
//! Like models, the manifest records the runtime twice: `declared` (pull
//! identity) and `stored` (effective form under the entry — `RelativeDir` /
//! `RelativePreinstalled` / docker clone). Tool binaries are discovered under
//! [`Self::install_dir`] at use time via [`Self::resolve_tool`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use pipette_plan_types::Runtime;

use super::key::{RuntimeStorageKey, RuntimeStorageKeyError};
use super::stored::{to_stored, RuntimeStoredError};

/// The manifest schema version (a validity gate, not a migration signal — old
/// entries under a different name/layout are simply not found and re-pulled).
pub const MANIFEST_VERSION: u32 = 2;

/// Why a [`RuntimeManifest`] couldn't be built or read.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeManifestError {
    /// The runtime has no storage key (on-device or OS-bundled — not installed).
    #[error(transparent)]
    NotStorable(#[from] RuntimeStorageKeyError),
    /// Could not build the effective `stored` form of `declared`.
    #[error(transparent)]
    Local(#[from] RuntimeStoredError),
    /// A parsed manifest is unusable — an unsupported version, a bad timestamp,
    /// or a drift / missing tool.
    #[error("installed runtime is corrupt: {0}")]
    Corrupt(String),
}

/// The declared runtime plus its effective on-disk form, recorded in
/// `runtimes/<key>/`.
///
/// This type is a **serde record** for the on-disk fields. Version / key /
/// `stored` drift checks run in the store read path (same pattern as
/// [`crate::model::ModelManifest`]). [`Self::new`] validates on write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub manifest_version: u32,
    /// When the pull started — stamped before the install runs, not after.
    #[serde(with = "time::serde::rfc3339")]
    pub fetched_at: OffsetDateTime,
    /// When an `ensure` last resolved this entry — the eviction order. Seeded to
    /// `fetched_at` on publish, rewritten best-effort on every hit.
    #[serde(with = "time::serde::rfc3339")]
    pub last_used_at: OffsetDateTime,
    /// The runtime as declared, i.e. the storage key + reporting identity.
    pub declared: Runtime,
    /// Effective install under the entry (`RelativeDir` / `RelativePreinstalled` / docker).
    pub stored: Runtime,
    /// Bytes `blobs/` occupied at publish — see
    /// [`crate::model::ModelManifest::blobs_bytes`] for why this measures
    /// `blobs/` alone and when it is `None`. Docker records `Some(0)`: the image
    /// lives in the daemon, so the entry here really is manifest-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blobs_bytes: Option<u64>,
}

/// Host-absolute install root already on `stored`, if any.
fn absolute_install_root(stored: &Runtime) -> Option<&str> {
    use pipette_plan_types::{LlamacppCliStockToolsSource, UvRuntimeSource};
    match stored {
        Runtime::LlamacppCliStockTools(rt) => match &rt.source {
            LlamacppCliStockToolsSource::AbsoluteDir { dir } => Some(dir.as_ref()),
            _ => None,
        },
        Runtime::UvVllm(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            _ => None,
        },
        Runtime::UvSglang(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            _ => None,
        },
        Runtime::MlxMacosPipette(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            _ => None,
        },
        Runtime::UvOpenvino(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

/// Entry-relative install subtree from `stored`, or `None` when the entry root
/// is the install root (docker) or the root is already absolute on `stored`.
fn install_rel(stored: &Runtime) -> Result<Option<&str>, RuntimeManifestError> {
    use pipette_plan_types::{LlamacppCliStockToolsSource, UvRuntimeSource};
    if absolute_install_root(stored).is_some() {
        return Ok(None);
    }
    match stored {
        Runtime::LlamacppCliStockTools(rt) => match &rt.source {
            LlamacppCliStockToolsSource::RelativeDir { dir } => Ok(Some(dir.as_ref())),
            other => Err(RuntimeManifestError::Corrupt(format!(
                "stored llama.cpp source is not RelativeDir: {other:?}"
            ))),
        },
        Runtime::UvVllm(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other => Err(RuntimeManifestError::Corrupt(format!(
                "stored uv-vllm source is not RelativePreinstalled: {other:?}"
            ))),
        },
        Runtime::UvSglang(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other => Err(RuntimeManifestError::Corrupt(format!(
                "stored uv-sglang source is not RelativePreinstalled: {other:?}"
            ))),
        },
        Runtime::MlxMacosPipette(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other => Err(RuntimeManifestError::Corrupt(format!(
                "stored mlx source is not RelativePreinstalled: {other:?}"
            ))),
        },
        Runtime::UvOpenvino(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other => Err(RuntimeManifestError::Corrupt(format!(
                "stored uv-openvino source is not RelativePreinstalled: {other:?}"
            ))),
        },
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => Ok(None),
        other => Err(RuntimeManifestError::Corrupt(format!(
            "stored runtime has no install dir: {other}"
        ))),
    }
}

/// Walk `root` for a file whose name is `expected_name` (or `expected_name.exe`).
fn find_tool_file(root: &Path, expected_name: &str) -> Option<PathBuf> {
    let exe = format!("{expected_name}.exe");
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name == expected_name || name == exe {
                return Some(path);
            }
        }
    }
    None
}

impl RuntimeManifest {
    /// Build the manifest for a freshly-installed `declared` runtime at
    /// `fetched_at`. Only an installable runtime has a storage key (else
    /// [`RuntimeManifestError::NotStorable`]).
    pub fn new(
        declared: Runtime,
        fetched_at: OffsetDateTime,
        blobs_bytes: u64,
    ) -> Result<Self, RuntimeManifestError> {
        RuntimeStorageKey::of(&declared)?;
        let stored = to_stored(&declared)?;
        Ok(Self {
            manifest_version: MANIFEST_VERSION,
            fetched_at,
            last_used_at: fetched_at,
            declared,
            stored,
            blobs_bytes: Some(blobs_bytes),
        })
    }

    /// This runtime's storage key.
    pub fn storage_key(&self) -> Result<RuntimeStorageKey, RuntimeStorageKeyError> {
        RuntimeStorageKey::of(&self.declared)
    }

    /// This runtime's entry directory under `runtimes_dir` (`runtimes_dir/<key>`)
    /// — where its manifest and any payload live.
    pub fn entry_dir(&self, runtimes_dir: &Path) -> Result<PathBuf, RuntimeManifestError> {
        Ok(runtimes_dir.join(self.storage_key()?.relative_dir()))
    }

    /// Absolute install root from `stored` under `runtimes_dir`.
    ///
    /// - already-absolute on `stored` (`AbsoluteDir` / `AbsolutePreinstalled`): as-is
    /// - file-tree / venv: `entry_dir` + relative install subtree
    /// - docker: the entry dir itself (manifest-only payload)
    pub fn install_dir(&self, runtimes_dir: &Path) -> Result<PathBuf, RuntimeManifestError> {
        if let Some(abs) = absolute_install_root(&self.stored) {
            return Ok(PathBuf::from(abs));
        }
        let entry = self.entry_dir(runtimes_dir)?;
        Ok(match install_rel(&self.stored)? {
            Some(rel) => entry.join(rel),
            None => entry,
        })
    }

    /// Resolve a tool by basename under [`Self::install_dir`] (walks the tree).
    pub fn resolve_tool(
        &self,
        runtimes_dir: &Path,
        name: &str,
    ) -> Result<PathBuf, RuntimeManifestError> {
        let root = self.install_dir(runtimes_dir)?;
        find_tool_file(&root, name).ok_or_else(|| {
            RuntimeManifestError::Corrupt(format!(
                "tool `{name}` not found under {}",
                root.display()
            ))
        })
    }

    /// Loader form: rewrite `stored` path fields to host-absolute under `runtimes_dir`.
    ///
    /// Llama → `AbsoluteDir`; UV/MLX → `AbsolutePreinstalled`; docker unchanged.
    pub fn bind_under(&self, runtimes_dir: &Path) -> Result<Runtime, RuntimeManifestError> {
        use pipette_plan_types::{
            AbsolutePath, LlamacppCliStockTools, LlamacppCliStockToolsSource, MlxMacosPipette,
            UvOpenvino, UvRuntimeSource, UvSglang, UvVllm,
        };

        let install = self.install_dir(runtimes_dir)?;
        let abs = |p: PathBuf| -> Result<AbsolutePath, RuntimeManifestError> {
            AbsolutePath::try_new(p.to_string_lossy().into_owned())
                .map_err(|e| RuntimeManifestError::Corrupt(e.to_string()))
        };
        match &self.stored {
            Runtime::LlamacppCliStockTools(rt) => {
                Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                    source: LlamacppCliStockToolsSource::AbsoluteDir { dir: abs(install)? },
                    flavor: rt.flavor.clone(),
                }))
            }
            Runtime::UvVllm(rt) => Ok(Runtime::UvVllm(UvVllm {
                server_version: rt.server_version.clone(),
                build: rt.build.clone(),
                python_version: rt.python_version.clone(),
                source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(install)? },
            })),
            Runtime::UvSglang(rt) => Ok(Runtime::UvSglang(UvSglang {
                server_version: rt.server_version.clone(),
                build: rt.build.clone(),
                python_version: rt.python_version.clone(),
                source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(install)? },
            })),
            Runtime::MlxMacosPipette(rt) => Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
                version: rt.version.clone(),
                flavor: rt.flavor,
                source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(install)? },
            })),
            Runtime::UvOpenvino(rt) => Ok(Runtime::UvOpenvino(UvOpenvino {
                server_version: rt.server_version.clone(),
                python_version: rt.python_version.clone(),
                source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(install)? },
            })),
            other => Ok(other.clone()),
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
    use time::format_description::well_known::Rfc3339;

    use pipette_plan_types::{
        LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource, NonEmptyString,
        RepositoryUrl, SourceRepository,
    };

    use super::*;

    const FETCHED_AT_STR: &str = "2026-01-01T00:00:00Z";

    fn fetched_at() -> anyhow::Result<OffsetDateTime> {
        Ok(OffsetDateTime::parse(FETCHED_AT_STR, &Rfc3339)?)
    }

    fn llamacpp() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }

    #[test]
    fn round_trips_through_toml() -> anyhow::Result<()> {
        let manifest = RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?;
        let parsed: RuntimeManifest = toml::from_str(&toml::to_string(&manifest)?)?;
        assert_eq!(parsed, manifest);
        assert!(matches!(
            &parsed.stored,
            Runtime::LlamacppCliStockTools(rt)
                if matches!(rt.source, LlamacppCliStockToolsSource::RelativeDir { ref dir } if dir.as_ref() == "blobs")
        ));
        Ok(())
    }

    #[test]
    fn new_seeds_last_used_at_from_fetched_at() -> anyhow::Result<()> {
        let manifest = RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?;
        assert_eq!(manifest.last_used_at, fetched_at()?);
        Ok(())
    }

    /// No migration: a v1 record has no `last_used_at`, so it does not decode.
    #[test]
    fn a_manifest_without_last_used_at_does_not_deserialize() -> anyhow::Result<()> {
        let rendered = toml::to_string(&RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?)?;
        let mut table: toml::Table = toml::from_str(&rendered)?;
        table.remove("last_used_at");
        assert!(toml::from_str::<RuntimeManifest>(&toml::to_string(&table)?).is_err());
        Ok(())
    }

    #[test]
    fn only_installable_runtimes_are_stored() -> anyhow::Result<()> {
        assert!(matches!(
            RuntimeManifest::new(
                Runtime::AppleFoundation(Default::default()),
                fetched_at()?,
                0
            ),
            Err(RuntimeManifestError::NotStorable(_))
        ));
        Ok(())
    }

    #[test]
    fn install_dir_uses_stored_local_dir() -> anyhow::Result<()> {
        let manifest = RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?;
        let root = Path::new("/ws/runtimes");
        assert_eq!(
            manifest.install_dir(root)?,
            manifest.entry_dir(root)?.join("blobs")
        );
        Ok(())
    }

    #[test]
    fn bind_under_makes_llama_relative_dir_absolute() -> anyhow::Result<()> {
        let manifest = RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?;
        let root = Path::new("/ws/runtimes");
        let local = manifest.bind_under(root)?;
        let Runtime::LlamacppCliStockTools(rt) = local else {
            anyhow::bail!("expected llama runtime");
        };
        let LlamacppCliStockToolsSource::AbsoluteDir { dir } = rt.source else {
            anyhow::bail!("expected AbsoluteDir");
        };
        assert_eq!(
            Path::new(dir.as_ref()),
            manifest.install_dir(root)?.as_path()
        );
        Ok(())
    }

    #[test]
    fn resolve_tool_finds_nested_binary() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let manifest = RuntimeManifest::new(llamacpp()?, fetched_at()?, 4096)?;
        let install = manifest.install_dir(tmp.path())?;
        std::fs::create_dir_all(install.join("build/bin"))?;
        std::fs::write(install.join("build/bin/llama-server"), b"")?;
        assert_eq!(
            manifest.resolve_tool(tmp.path(), "llama-server")?,
            install.join("build/bin/llama-server")
        );
        assert!(manifest.resolve_tool(tmp.path(), "nope").is_err());
        Ok(())
    }
}
