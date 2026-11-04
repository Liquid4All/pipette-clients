//! [`RuntimeManifest`]: `runtimes/<key>/manifest.toml`, the record for one
//! installed runtime, laid out like [`crate::model::ModelManifest`] with the
//! payload in `blobs/` (absent for docker).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use pipette_plan_types::Runtime;

use super::key::{RuntimeStorageKey, RuntimeStorageKeyError};
use super::stored::{RuntimeStoredError, to_stored};

/// A validity gate, not a migration signal: a manifest that parses at another
/// version reads as `Corrupt`.
pub const MANIFEST_VERSION: u32 = 2;

/// Why a [`RuntimeManifest`] couldn't be built or read.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeManifestError {
    #[error(transparent)]
    NotStorable(#[from] RuntimeStorageKeyError),
    #[error(transparent)]
    Local(#[from] RuntimeStoredError),
    #[error("installed runtime is corrupt: {0}")]
    Corrupt(String),
}

/// Version, key and `stored` drift checks run in the store read path;
/// [`Self::new`] validates on write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub manifest_version: u32,
    /// When the pull started, stamped before the install runs.
    #[serde(with = "time::serde::rfc3339")]
    pub fetched_at: OffsetDateTime,
    /// When an `ensure` last resolved this entry: the eviction order. Seeded to
    /// `fetched_at` on publish, rewritten best-effort on every hit.
    #[serde(with = "time::serde::rfc3339")]
    pub last_used_at: OffsetDateTime,
    /// The storage key and reporting identity.
    pub declared: Runtime,
    /// Entry-relative installed form (see [`to_stored`]).
    pub stored: Runtime,
    /// Bytes `blobs/` occupied at publish; see
    /// [`crate::model::ModelManifest::blobs_bytes`]. Docker records `Some(0)`:
    /// the image lives in the daemon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blobs_bytes: Option<u64>,
}

fn absolute_install_root(stored: &Runtime) -> Option<&str> {
    use pipette_plan_types::{
        LlamacppCliStockToolsSource, OvmsSource, TrtEdgellmSource, UvRuntimeSource,
    };
    match stored {
        Runtime::LlamacppCliStockTools(rt) => match &rt.source {
            LlamacppCliStockToolsSource::AbsoluteDir { dir } => Some(dir.as_ref()),
            LlamacppCliStockToolsSource::GithubRelease(_)
            | LlamacppCliStockToolsSource::RemoteArchive { .. }
            | LlamacppCliStockToolsSource::VaultRemoteArchive { .. }
            | LlamacppCliStockToolsSource::RelativeDir { .. } => None,
        },
        Runtime::Ovms(rt) => match &rt.source {
            OvmsSource::AbsoluteDir { dir } => Some(dir.as_ref()),
            OvmsSource::GithubRelease(_)
            | OvmsSource::RemoteArchive { .. }
            | OvmsSource::VaultRemoteArchive { .. }
            | OvmsSource::RelativeDir { .. } => None,
        },
        Runtime::TrtEdgellm(rt) => match &rt.source {
            TrtEdgellmSource::AbsoluteDir { dir } => Some(dir.as_ref()),
            TrtEdgellmSource::RemoteArchive { .. }
            | TrtEdgellmSource::VaultRemoteArchive { .. }
            | TrtEdgellmSource::RelativeDir { .. } => None,
        },
        Runtime::UvVllm(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::RelativePreinstalled { .. } => None,
        },
        Runtime::UvSglang(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::RelativePreinstalled { .. } => None,
        },
        Runtime::MlxMacosPipette(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::RelativePreinstalled { .. } => None,
        },
        Runtime::UvOpenvino(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::RelativePreinstalled { .. } => None,
        },
        Runtime::UvOnnxGenai(rt) => match &rt.source {
            UvRuntimeSource::AbsolutePreinstalled { dir } => Some(dir.as_ref()),
            UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::RelativePreinstalled { .. } => None,
        },
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_) => None,
    }
}

/// Entry-relative install subtree from `stored`, or `None` when the entry root
/// is the install root (docker) or the root is already absolute on `stored`.
fn install_rel(stored: &Runtime) -> Result<Option<&str>, RuntimeManifestError> {
    use pipette_plan_types::{
        LlamacppCliStockToolsSource, OvmsSource, TrtEdgellmSource, UvRuntimeSource,
    };
    if absolute_install_root(stored).is_some() {
        return Ok(None);
    }
    match stored {
        Runtime::LlamacppCliStockTools(rt) => match &rt.source {
            LlamacppCliStockToolsSource::RelativeDir { dir } => Ok(Some(dir.as_ref())),
            other @ (LlamacppCliStockToolsSource::GithubRelease(_)
            | LlamacppCliStockToolsSource::RemoteArchive { .. }
            | LlamacppCliStockToolsSource::VaultRemoteArchive { .. }
            | LlamacppCliStockToolsSource::AbsoluteDir { .. }) => {
                Err(RuntimeManifestError::Corrupt(format!(
                    "stored llama.cpp source is not RelativeDir: {other:?}"
                )))
            }
        },
        Runtime::Ovms(rt) => match &rt.source {
            OvmsSource::RelativeDir { dir } => Ok(Some(dir.as_ref())),
            other @ (OvmsSource::GithubRelease(_)
            | OvmsSource::RemoteArchive { .. }
            | OvmsSource::VaultRemoteArchive { .. }
            | OvmsSource::AbsoluteDir { .. }) => Err(RuntimeManifestError::Corrupt(format!(
                "stored ovms source is not RelativeDir: {other:?}"
            ))),
        },
        Runtime::TrtEdgellm(rt) => match &rt.source {
            TrtEdgellmSource::RelativeDir { dir } => Ok(Some(dir.as_ref())),
            other @ (TrtEdgellmSource::RemoteArchive { .. }
            | TrtEdgellmSource::VaultRemoteArchive { .. }
            | TrtEdgellmSource::AbsoluteDir { .. }) => Err(RuntimeManifestError::Corrupt(format!(
                "stored trt_edgellm source is not RelativeDir: {other:?}"
            ))),
        },
        Runtime::UvVllm(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other @ (UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::AbsolutePreinstalled { .. }) => Err(RuntimeManifestError::Corrupt(
                format!("stored uv-vllm source is not RelativePreinstalled: {other:?}"),
            )),
        },
        Runtime::UvSglang(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other @ (UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::AbsolutePreinstalled { .. }) => Err(RuntimeManifestError::Corrupt(
                format!("stored uv-sglang source is not RelativePreinstalled: {other:?}"),
            )),
        },
        Runtime::MlxMacosPipette(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other @ (UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::AbsolutePreinstalled { .. }) => Err(RuntimeManifestError::Corrupt(
                format!("stored mlx source is not RelativePreinstalled: {other:?}"),
            )),
        },
        Runtime::UvOpenvino(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other @ (UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::AbsolutePreinstalled { .. }) => Err(RuntimeManifestError::Corrupt(
                format!("stored uv-openvino source is not RelativePreinstalled: {other:?}"),
            )),
        },
        Runtime::UvOnnxGenai(rt) => match &rt.source {
            UvRuntimeSource::RelativePreinstalled { dir } => Ok(Some(dir.as_ref())),
            other @ (UvRuntimeSource::PipRequirementsText { .. }
            | UvRuntimeSource::VaultWheelhouse { .. }
            | UvRuntimeSource::AbsolutePreinstalled { .. }) => Err(RuntimeManifestError::Corrupt(
                format!("stored uv-onnx-genai source is not RelativePreinstalled: {other:?}"),
            )),
        },
        Runtime::DockerVllm(_) | Runtime::DockerSglang(_) => Ok(None),
        other @ (Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleCoreAiMacosPipette(_)
        | Runtime::AppleFoundation(_)) => Err(RuntimeManifestError::Corrupt(format!(
            "stored runtime has no install dir: {other}"
        ))),
    }
}

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

    pub fn storage_key(&self) -> Result<RuntimeStorageKey, RuntimeStorageKeyError> {
        RuntimeStorageKey::of(&self.declared)
    }

    /// `runtimes_dir/<key>`.
    pub fn entry_dir(&self, runtimes_dir: &Path) -> Result<PathBuf, RuntimeManifestError> {
        Ok(runtimes_dir.join(self.storage_key()?.relative_dir()))
    }

    /// Absolute install root; for docker, the entry dir itself.
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

    /// Finds `name` (or `name.exe`) anywhere under [`Self::install_dir`].
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

    /// Loader form: `stored` with its path made host-absolute under `runtimes_dir`.
    pub fn bind_under(&self, runtimes_dir: &Path) -> Result<Runtime, RuntimeManifestError> {
        use pipette_plan_types::{
            AbsolutePath, LlamacppCliStockTools, LlamacppCliStockToolsSource, MlxMacosPipette,
            Ovms, OvmsSource, TrtEdgellm, TrtEdgellmSource, UvOnnxGenai, UvOpenvino,
            UvRuntimeSource, UvSglang, UvVllm,
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
            Runtime::Ovms(rt) => Ok(Runtime::Ovms(Ovms {
                source: OvmsSource::AbsoluteDir { dir: abs(install)? },
                flavor: rt.flavor.clone(),
            })),
            Runtime::TrtEdgellm(rt) => Ok(Runtime::TrtEdgellm(TrtEdgellm {
                source: TrtEdgellmSource::AbsoluteDir { dir: abs(install)? },
                flavor: rt.flavor.clone(),
            })),
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
            Runtime::UvOnnxGenai(rt) => Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
                server_version: rt.server_version.clone(),
                execution_provider: rt.execution_provider.clone(),
                python_version: rt.python_version.clone(),
                source: UvRuntimeSource::AbsolutePreinstalled { dir: abs(install)? },
            })),
            other @ (Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::DockerVllm(_)
            | Runtime::DockerSglang(_)
            | Runtime::AppleCoreAiMacosPipette(_)
            | Runtime::AppleFoundation(_)) => Ok(other.clone()),
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
    use time::format_description::well_known::Rfc3339;

    use pipette_plan_types::{
        LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsRelease,
        LlamacppCliStockToolsSource, NonEmptyString, OnnxExecutionProvider, RepositoryUrl,
        UvOnnxGenai, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    };

    use super::*;

    const FETCHED_AT_STR: &str = "2026-01-01T00:00:00Z";

    fn fetched_at() -> anyhow::Result<OffsetDateTime> {
        Ok(OffsetDateTime::parse(FETCHED_AT_STR, &Rfc3339)?)
    }

    fn llamacpp() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
                asset: None,
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

    fn onnx_genai(execution_provider: OnnxExecutionProvider) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: UvServerVersion::try_new("0.16.0".to_owned())?,
            execution_provider,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("onnxruntime-genai==0.16.0\n".to_owned())?,
                install_flags: None,
            },
        }))
    }

    /// A variant moved into `bind_under`'s pass-through arm would still compile
    /// and return a relative source that fails later in `pipette-venv`.
    #[rstest::rstest]
    #[case(OnnxExecutionProvider::Cpu)]
    #[case(OnnxExecutionProvider::Cuda)]
    fn bind_under_rewrites_onnx_genai_source_and_keeps_the_provider(
        #[case] execution_provider: OnnxExecutionProvider,
    ) -> anyhow::Result<()> {
        let manifest =
            RuntimeManifest::new(onnx_genai(execution_provider.clone())?, fetched_at()?, 4096)?;
        let root = Path::new("/ws/runtimes");
        let Runtime::UvOnnxGenai(rt) = manifest.bind_under(root)? else {
            anyhow::bail!("expected a UvOnnxGenai runtime");
        };
        let UvRuntimeSource::AbsolutePreinstalled { dir } = &rt.source else {
            anyhow::bail!("expected AbsolutePreinstalled, got {:?}", rt.source);
        };
        assert_eq!(Path::new(dir.as_ref()), manifest.install_dir(root)?);
        assert_eq!(rt.execution_provider, execution_provider);
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
