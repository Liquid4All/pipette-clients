//! [`RuntimeStorageKey`]: the flat `runtimes/<key>/` directory name for a
//! pullable [`Runtime`], built like [`crate::model`]'s keys. The leading type
//! prefix keeps the flat dir collision-free across runtime kinds.

use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use pipette_plan_types::{
    LlamacppCliStockToolsSource, OvmsSource, Runtime, TrtEdgellmSource, UvRuntimeSource,
};

use crate::model::{bound_to, slug_from};

/// The whole SHA-256: the requirements body is what a uv entry *is*, so the
/// on-disk name carries it in full.
const PIP_DIGEST_HEX_LEN: usize = 64;

/// Above [`crate::model`]'s cap so a full requirements digest is not folded
/// away; well under the 255-byte filename limit.
const RUNTIME_MAX_LEN: usize = 128;

/// Why a [`Runtime`] has no [`RuntimeStorageKey`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeStorageKeyError {
    /// On-device / OS-bundled, or an already-installed source arm.
    #[error("runtime `{0}` has no storage key (not pullable / effective-only)")]
    NotStorable(String),
}

/// Distinct from [`Runtime`]'s `Display` and `cli_ref`, the `:`-bearing
/// canonical and `--runtime` references.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeStorageKey(String);

impl RuntimeStorageKey {
    /// Never longer than `RUNTIME_MAX_LEN`.
    pub fn of(runtime: &Runtime) -> Result<Self, RuntimeStorageKeyError> {
        let not_storable = || RuntimeStorageKeyError::NotStorable(runtime.to_string());
        let segments: Vec<String> = match runtime {
            Runtime::LlamacppCliStockTools(rt) => {
                let mut segs = vec!["llama-cpp".to_owned()];
                match &rt.source {
                    LlamacppCliStockToolsSource::GithubRelease(release) => {
                        segs.push(release.repository_url.org_repo().to_owned());
                        segs.push(release.repository_version.to_string());
                        // Only when named, so an entry pulled before `asset`
                        // existed keeps its key.
                        if let Some(asset) = &release.asset {
                            segs.push(asset.to_string());
                        }
                    }
                    LlamacppCliStockToolsSource::RemoteArchive { url } => {
                        segs.push("remote-archive".to_owned());
                        segs.push(url.to_string());
                    }
                    LlamacppCliStockToolsSource::VaultRemoteArchive { url, .. } => {
                        segs.push("vault".to_owned());
                        segs.push(url.resource_id().to_string());
                    }
                    LlamacppCliStockToolsSource::RelativeDir { .. }
                    | LlamacppCliStockToolsSource::AbsoluteDir { .. } => return Err(not_storable()),
                }
                segs.push(rt.flavor.to_string());
                segs
            }
            Runtime::Ovms(rt) => {
                let mut segs = vec!["ovms".to_owned()];
                match &rt.source {
                    OvmsSource::GithubRelease(repo) => {
                        segs.push(repo.repository_url.org_repo().to_owned());
                        segs.push(repo.repository_version.to_string());
                    }
                    OvmsSource::RemoteArchive { url } => {
                        segs.push("remote-archive".to_owned());
                        segs.push(url.to_string());
                    }
                    OvmsSource::VaultRemoteArchive { url, .. } => {
                        segs.push("vault".to_owned());
                        segs.push(url.resource_id().to_string());
                    }
                    OvmsSource::RelativeDir { .. } | OvmsSource::AbsoluteDir { .. } => {
                        return Err(not_storable());
                    }
                }
                segs.push(rt.flavor.to_string());
                segs
            }
            Runtime::TrtEdgellm(rt) => {
                let mut segs = vec!["trt-edgellm".to_owned()];
                match &rt.source {
                    TrtEdgellmSource::RemoteArchive { url } => {
                        segs.push("remote-archive".to_owned());
                        segs.push(url.to_string());
                    }
                    TrtEdgellmSource::VaultRemoteArchive { url, .. } => {
                        segs.push("vault".to_owned());
                        segs.push(url.resource_id().to_string());
                    }
                    TrtEdgellmSource::RelativeDir { .. } | TrtEdgellmSource::AbsoluteDir { .. } => {
                        return Err(not_storable());
                    }
                }
                segs.push(rt.flavor.to_string());
                segs
            }
            Runtime::MlxMacosPipette(rt) => {
                // `version` is a label, not identity: nothing checks it against
                // the requirements body, so keying on it would build one
                // environment twice.
                let mut segs = vec!["mlx".to_owned(), flavor_tag(&rt.flavor)];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            Runtime::DockerVllm(rt) => vec![
                "docker-vllm".to_owned(),
                rt.image_name.to_string(),
                rt.image_tag.to_string(),
                flavor_tag(&rt.flavor),
            ],
            Runtime::DockerSglang(rt) => vec![
                "docker-sglang".to_owned(),
                rt.image_name.to_string(),
                rt.image_tag.to_string(),
                flavor_tag(&rt.flavor),
            ],
            // `server_version` is dropped for the same reason as MLX's `version`.
            // `python_version` stays: two interpreters give two venvs from one
            // body. `build` names the hardware target. install_flags are not
            // identity: they change how a body installs, not which body it is.
            Runtime::UvVllm(rt) => {
                let mut segs = vec![
                    "uv-vllm".to_owned(),
                    rt.build.to_string(),
                    rt.python_version.to_string(),
                ];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            // No hardware segment: one wheel serves CPU, GPU and NPU, and the
            // device is a launch flag.
            Runtime::UvOpenvino(rt) => {
                let mut segs = vec!["uv-openvino".to_owned(), rt.python_version.to_string()];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            Runtime::UvSglang(rt) => {
                let mut segs = vec![
                    "uv-sglang".to_owned(),
                    rt.build.to_string(),
                    rt.python_version.to_string(),
                ];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            // Each execution provider is a separate wheel, so it is identity like
            // `build`. `to_string()` rather than `flavor_tag`: the provider set
            // is open, not a closed enum.
            Runtime::UvOnnxGenai(rt) => {
                let mut segs = vec![
                    "uv-onnx-genai".to_owned(),
                    rt.execution_provider.to_string(),
                    rt.python_version.to_string(),
                ];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            Runtime::AppleCoreAiMacosPipette(rt) => {
                if !rt.is_bundled() || !rt.source.is_bundled() {
                    return Err(not_storable());
                }
                vec!["core-ai".to_owned(), super::coreai::source_digest(rt)]
            }
            Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::AppleFoundation(_) => return Err(not_storable()),
        };
        Ok(RuntimeStorageKey(bound_to(
            slug_from(&segments),
            RUNTIME_MAX_LEN,
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// This runtime's store subdirectory, relative to the `runtimes/` root. A
    /// single flat component (the key), never a nested path.
    pub fn relative_dir(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }
}

impl std::fmt::Display for RuntimeStorageKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Preinstalled sources are `Err`. The tail is a digest of the requirements
/// body, never a name: a name would let two bodies share an entry and the
/// first installed would silently serve the other.
fn uv_declared_key_tail(source: &UvRuntimeSource) -> Result<Vec<String>, ()> {
    match source {
        UvRuntimeSource::PipRequirementsText { contents, .. } => Ok(vec![
            "pip".to_owned(),
            pip_requirements_digest(contents.as_ref()),
        ]),
        // The wheelhouse is part of what lands in the venv, so the resource
        // sits beside the body: the same body over other wheels is another
        // environment.
        UvRuntimeSource::VaultWheelhouse { contents, url, .. } => Ok(vec![
            "vault".to_owned(),
            url.resource_id().to_string(),
            pip_requirements_digest(contents.as_ref()),
        ]),
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => Err(()),
    }
}

fn pip_requirements_digest(contents: &str) -> String {
    let mut digest = hex::encode(Sha256::digest(contents.as_bytes()));
    digest.truncate(PIP_DIGEST_HEX_LEN);
    digest
}

/// JSON string tag of a closed flavor enum; `unknown` if the shape drifts.
fn flavor_tag<T: Serialize>(flavor: &T) -> String {
    serde_json::to_value(flavor)
        .ok()
        .as_ref()
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_artifacts_catalog::runtime::ovms::upstream_repository_url;
    use pipette_plan_types::{
        DockerSglang, DockerVllm, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString,
        OnnxExecutionProvider, Ovms, OvmsFlavor, OvmsSourceRepository, RemoteArchiveUrl,
        SglangFlavor, TrtEdgellm, TrtEdgellmFlavor, UvBuild, UvOnnxGenai, UvOpenvino,
        UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang, UvVllm, VllmFlavor,
    };

    use super::*;
    use crate::runtime::llamacpp::upstream_release;
    use crate::test_support::labelled;

    fn llamacpp_repo(version: &str, flavor: LlamaCppFlavor) -> anyhow::Result<Runtime> {
        llamacpp_release(version, None, flavor)
    }

    fn llamacpp_release(
        version: &str,
        asset: Option<&str>,
        flavor: LlamaCppFlavor,
    ) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(upstream_release(version, asset)?),
            flavor,
        }))
    }

    fn uv_source(requirements: &str) -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new(requirements.to_owned())?,
            install_flags: None,
        })
    }

    #[test]
    fn short_mlx_key_is_exact() -> anyhow::Result<()> {
        let runtime = Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.1".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: uv_source("vllm==0.10.0\n")?,
        });
        let expected = "mlx__macos-arm64__pip__a35f54f0c6df73cb9de9aae4d375d5578e4d4e4da99bdbce6dd8869347076865";
        let key = RuntimeStorageKey::of(&runtime)?;
        assert_eq!(key.as_str(), expected);
        assert!(!key.as_str().contains('/'));
        assert_eq!(key.relative_dir(), PathBuf::from(expected));
        assert_eq!(RuntimeStorageKey::of(&runtime)?, key);
        Ok(())
    }

    /// The last case is long enough to fold at [`RUNTIME_MAX_LEN`].
    #[rstest]
    #[case(llamacpp_repo("b9305", LlamaCppFlavor::MacosArm64)?, "llama-cpp__")]
    #[case(
        Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.10.0".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        }),
        "docker-vllm__"
    )]
    #[case(
        Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(format!(
                    "https://example.com/{}/llama-b9305.tar.gz",
                    "a-very-long-path-segment".repeat(6)
                ))?,
            },
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        }),
        "llama-cpp__"
    )]
    fn runtime_keys_stay_bounded_and_prefixed(
        #[case] runtime: Runtime,
        #[case] prefix: &str,
    ) -> anyhow::Result<()> {
        let key = RuntimeStorageKey::of(&runtime)?;
        assert!(key.as_str().starts_with(prefix), "got {key}");
        assert!(key.as_str().len() <= RUNTIME_MAX_LEN);
        assert!(!key.as_str().contains('/'));
        assert_eq!(RuntimeStorageKey::of(&runtime)?, key);
        Ok(())
    }

    /// Two files of one release are two builds.
    #[rstest]
    #[case::flavor(
        llamacpp_repo("b9305", LlamaCppFlavor::MacosArm64)?,
        llamacpp_repo("b9305", LlamaCppFlavor::MacosX64)?
    )]
    #[case::asset_and_default(
        llamacpp_release("b11399", Some(WIN_CUDA), LlamaCppFlavor::WindowsX64Cuda131)?,
        llamacpp_repo("b11399", LlamaCppFlavor::WindowsX64Cuda131)?
    )]
    #[case::two_assets(
        llamacpp_release("b11399", Some(WIN_CUDA), LlamaCppFlavor::WindowsX64Cuda131)?,
        llamacpp_release("b11399", Some(WIN_CUDART), LlamaCppFlavor::WindowsX64Cuda131)?
    )]
    fn distinct_llamacpp_builds_yield_distinct_keys(
        #[case] a: Runtime,
        #[case] b: Runtime,
    ) -> anyhow::Result<()> {
        assert_ne!(RuntimeStorageKey::of(&a)?, RuntimeStorageKey::of(&b)?);
        Ok(())
    }

    const WIN_CUDA: &str = "llama-b11399-bin-win-cuda-13.4-x64.zip";
    const WIN_CUDART: &str = "cudart-llama-bin-win-cuda-13.4-x64.zip";

    fn uv_vllm(source: UvRuntimeSource) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source,
        }))
    }

    #[rstest]
    #[case::afm(Runtime::AppleFoundation(Default::default()))]
    #[case::llama_relative_dir(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source: LlamacppCliStockToolsSource::RelativeDir {
            dir: pipette_plan_types::RelativePath::try_new("blobs".to_owned())?,
        },
        flavor: LlamaCppFlavor::MacosArm64,
    }))]
    #[case::llama_absolute_dir(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source: LlamacppCliStockToolsSource::AbsoluteDir {
            dir: pipette_plan_types::AbsolutePath::try_new("/ws/r/blobs".to_owned())?,
        },
        flavor: LlamaCppFlavor::MacosArm64,
    }))]
    #[case::ovms_relative_dir(Runtime::Ovms(Ovms {
        source: OvmsSource::RelativeDir {
            dir: pipette_plan_types::RelativePath::try_new("blobs".to_owned())?,
        },
        flavor: OvmsFlavor::WindowsX64,
    }))]
    #[case::trt_edgellm_absolute_dir(Runtime::TrtEdgellm(TrtEdgellm {
        source: TrtEdgellmSource::AbsoluteDir {
            dir: pipette_plan_types::AbsolutePath::try_new("/ws/r/blobs".to_owned())?,
        },
        flavor: TrtEdgellmFlavor::JetsonOrinCuda13,
    }))]
    #[case::uv_relative_preinstalled(uv_vllm(UvRuntimeSource::RelativePreinstalled {
        dir: pipette_plan_types::RelativePath::try_new("blobs/venv".to_owned())?,
    })?)]
    #[case::uv_absolute_preinstalled(uv_vllm(UvRuntimeSource::AbsolutePreinstalled {
        dir: pipette_plan_types::AbsolutePath::try_new("/ws/venv".to_owned())?,
    })?)]
    fn no_storage_key_for_non_pullable(#[case] runtime: Runtime) -> anyhow::Result<()> {
        assert!(matches!(
            RuntimeStorageKey::of(&runtime),
            Err(RuntimeStorageKeyError::NotStorable(_))
        ));
        Ok(())
    }

    /// Nothing validates a `version` label against the requirements body, so
    /// keying on it would build one environment twice.
    #[test]
    fn a_version_label_does_not_change_the_entry() -> anyhow::Result<()> {
        fn mlx(version: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
                version: NonEmptyString::try_new(version.to_owned())?,
                flavor: MlxMacosPipetteFlavor::MacosArm64,
                source: uv_source("mlx-lm==0.31.3\n")?,
            }))
        }
        fn vllm(server_version: &str, python: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new(server_version.to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new(python.to_owned())?,
                source: uv_source("vllm==0.10.0\n")?,
            }))
        }
        fn openvino(server_version: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::UvOpenvino(UvOpenvino {
                server_version: UvServerVersion::try_new(server_version.to_owned())?,
                python_version: UvPythonVersion::try_new("3.11".to_owned())?,
                source: uv_source("openvino-genai==2026.4.1.0\n")?,
            }))
        }

        assert_eq!(
            RuntimeStorageKey::of(&mlx("0.31.3")?)?,
            RuntimeStorageKey::of(&mlx("99.0.0")?)?,
            "one requirements body is one venv, whatever it is labelled"
        );
        assert_eq!(
            RuntimeStorageKey::of(&vllm("0.10.0", "3.12")?)?,
            RuntimeStorageKey::of(&vllm("9.9.9", "3.12")?)?,
            "`server_version` is the same kind of label"
        );
        assert_ne!(
            RuntimeStorageKey::of(&vllm("0.10.0", "3.12")?)?,
            RuntimeStorageKey::of(&vllm("0.10.0", "3.11")?)?,
            "`python_version` builds a different venv, so it stays identity"
        );

        assert_eq!(
            RuntimeStorageKey::of(&openvino("2026.4.1")?)?,
            RuntimeStorageKey::of(&openvino("9.9.9")?)?,
            "`server_version` is a label here too"
        );
        Ok(())
    }

    #[test]
    fn uv_source_is_storage_identity() -> anyhow::Result<()> {
        fn base_fields(source: UvRuntimeSource) -> anyhow::Result<UvVllm> {
            Ok(UvVllm {
                server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source,
            })
        }
        fn uv(body: &str, install_flags: Option<Vec<String>>) -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(base_fields(
                UvRuntimeSource::PipRequirementsText {
                    contents: NonEmptyString::try_new(body.to_owned())?,
                    install_flags,
                },
            )?))
        }

        let pip_body = "vllm==0.10.0\n";
        let key = RuntimeStorageKey::of(&uv(pip_body, None)?)?;
        assert!(key.as_str().starts_with("uv-vllm__"), "got {key}");
        assert_eq!(
            RuntimeStorageKey::of(&uv(pip_body, None)?)?,
            key,
            "one body, one entry"
        );
        assert_eq!(
            RuntimeStorageKey::of(&uv(pip_body, Some(vec!["--quiet".into()]))?)?,
            key,
            "install flags are not identity"
        );
        assert_ne!(
            RuntimeStorageKey::of(&uv("vllm==0.11.0\n", None)?)?,
            key,
            "a different environment must not reuse another's entry"
        );

        let pre = Runtime::UvVllm(base_fields(UvRuntimeSource::RelativePreinstalled {
            dir: pipette_plan_types::RelativePath::try_new("blobs/venv".to_owned())?,
        })?);
        assert!(
            matches!(
                RuntimeStorageKey::of(&pre),
                Err(RuntimeStorageKeyError::NotStorable(_))
            ),
            "preinstalled must not be storable"
        );
        // SHA-256, not DefaultHasher: the key must match across builds.
        let digest = pip_requirements_digest(pip_body);
        assert_eq!(digest.len(), PIP_DIGEST_HEX_LEN);
        assert!(hex::encode(sha2::Sha256::digest(pip_body.as_bytes())).starts_with(&digest));
        Ok(())
    }

    /// A key is the `runtimes/<key>/` directory on every machine that has
    /// pulled. Changing one fails nothing at runtime: the lookup misses, the
    /// entry re-downloads and the old tree is orphaned. Only these literals
    /// catch that. A diff here is a store migration, not a test update.
    #[rstest]
    #[case::llamacpp_release(
        llamacpp_repo("b9305", LlamaCppFlavor::MacosArm64)?,
        "llama-cpp__ggml-org_llama.cpp__b9305__macos-arm64"
    )]
    #[case::llamacpp_release_asset(
        llamacpp_release(
            "b11399",
            Some("llama-b11399-bin-ubuntu-rocm-10.0-x64.tar.gz"),
            LlamaCppFlavor::LinuxX64Rocm,
        )?,
        "llama-cpp__ggml-org_llama.cpp__b11399__llama-b11399-bin-ubuntu-rocm-10.0-x64.tar.gz__linux-x64-rocm"
    )]
    #[case::llamacpp_archive(
        Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(
                    "https://example.com/llama-b9305.tar.gz".to_owned())?,
            },
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        }),
        "llama-cpp__remote-archive__https_example.com_llama-b9305.tar.gz__linux-x64-cpu"
    )]
    #[case::llamacpp_vault_archive(
        Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::VaultRemoteArchive {
                url: "/vault/resources/vault_0195c3e2/content".parse()?,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        }),
        "llama-cpp__vault__vault_0195c3e2__linux-x64-cpu"
    )]
    #[case::ovms_release(
        Runtime::Ovms(Ovms {
            source: OvmsSource::GithubRelease(OvmsSourceRepository {
                repository_url: upstream_repository_url(),
                repository_version: NonEmptyString::try_new("v2026.4.1".to_owned())?,
            }),
            flavor: OvmsFlavor::WindowsX64,
        }),
        "ovms__openvinotoolkit_model_server__v2026.4.1__windows-x64"
    )]
    #[case::ovms_archive(
        Runtime::Ovms(Ovms {
            source: OvmsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new("https://example.com/ovms.tar.gz".to_owned())?,
            },
            flavor: OvmsFlavor::Ubuntu24X64,
        }),
        "ovms__remote-archive__https_example.com_ovms.tar.gz__ubuntu24-x64"
    )]
    #[case::ovms_vault_archive(
        Runtime::Ovms(Ovms {
            source: OvmsSource::VaultRemoteArchive {
                url: "/vault/resources/vault_0195c3e2/content".parse()?,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
            flavor: OvmsFlavor::Custom("acme-x64".to_owned()),
        }),
        "ovms__vault__vault_0195c3e2__acme-x64"
    )]
    #[case::trt_edgellm_vault_archive(
        Runtime::TrtEdgellm(TrtEdgellm {
            source: TrtEdgellmSource::VaultRemoteArchive {
                url: "/vault/resources/vault_0195c3e2/content".parse()?,
                principal_id: "org_1".parse()?,
                custom_label: None,
            },
            flavor: TrtEdgellmFlavor::JetsonOrinCuda13,
        }),
        "trt-edgellm__vault__vault_0195c3e2__jetson-orin-cuda-13"
    )]
    #[case::uv_vllm_wheelhouse(
        uv_vllm(UvRuntimeSource::VaultWheelhouse {
            contents: NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?,
            url: "/vault/resources/vault_0195c3e2/content".parse()?,
            principal_id: "org_1".parse()?,
            custom_label: None,
            install_flags: None,
        })?,
        "uv-vllm__cu121__3.12__vault__vault_0195c3e2__a35f54f0c6df73cb9de9aae4d375d5578e4d4e4da99bdbce6dd8869347076865"
    )]
    #[case::docker_vllm(
        Runtime::DockerVllm(DockerVllm {
            image_name: NonEmptyString::try_new("vllm/vllm-openai".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.10.0".to_owned())?,
            flavor: VllmFlavor::NvidiaGpu,
        }),
        "docker-vllm__vllm_vllm-openai__v0.10.0__nvidia_gpu"
    )]
    #[case::docker_sglang(
        Runtime::DockerSglang(DockerSglang {
            image_name: NonEmptyString::try_new("lmsysorg/sglang".to_owned())?,
            image_tag: NonEmptyString::try_new("v0.4.0".to_owned())?,
            flavor: SglangFlavor::AmdGpu,
        }),
        "docker-sglang__lmsysorg_sglang__v0.4.0__amd_gpu"
    )]
    #[case::uv_vllm_catalog(
        Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.22.0".to_owned())?,
            build: UvBuild::try_new("cu129".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: uv_source("vllm==0.22.0\n")?,
        }),
        "uv-vllm__cu129__3.12__pip__1795b49957c5fd13432458df877dce586cfd52777f46db76544ba392bf104b13"
    )]
    #[case::uv_sglang_catalog(
        Runtime::UvSglang(UvSglang {
            server_version: UvServerVersion::try_new("0.5.12".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: uv_source("sglang==0.5.12\n")?,
        }),
        "uv-sglang__cu121__3.12__pip__08353b22c0dd876223139aca1a58f6445092265eca512843c675db6c52a1d2f8"
    )]
    #[case::uv_onnx_genai_cpu(
        Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: UvServerVersion::try_new("0.16.0".to_owned())?,
            execution_provider: OnnxExecutionProvider::Cpu,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: uv_source("onnxruntime-genai==0.16.0\n")?,
        }),
        "uv-onnx-genai__cpu__3.11__pip__5ff6726d041f40f340b42d683cb470a3fd1b741c83d0961401ae1c3d4bb8cb2b"
    )]
    // Two providers are two wheels, so they must not share an entry.
    #[case::uv_onnx_genai_cuda(
        Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: UvServerVersion::try_new("0.16.0".to_owned())?,
            execution_provider: OnnxExecutionProvider::Cuda,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: uv_source("onnxruntime-genai-cuda==0.16.0\n")?,
        }),
        "uv-onnx-genai__cuda__3.11__pip__37bcb9514f8300f1f86de9f8140c73d688611ef9ebb3d44100bb1d53927b005f"
    )]
    #[case::mlx_catalog_folded(
        Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: uv_source("mlx-lm==0.31.3\n")?,
        }),
        "mlx__macos-arm64__pip__5dc4a038a260c2db6e72a1025842f2d8d229bd4e87f95f2f757ac61ec49aaa40"
    )]
    // Same fields as `uv_vllm_catalog`, different body: must not collide.
    #[case::uv_vllm_other_body(
        Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.22.0".to_owned())?,
            build: UvBuild::try_new("cu129".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: uv_source("vllm==0.22.0\n--extra-index-url https://example/cu129\n")?,
        }),
        "uv-vllm__cu129__3.12__pip__622ffba411e11b017d82ea1238fa3e9c7cf0f79c9bdd580eb3ef5c74e385edc8"
    )]
    fn storage_key_is_stable_on_disk(
        #[case] runtime: Runtime,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(
            RuntimeStorageKey::of(&runtime)?.as_str(),
            expected,
            "storage key changed — existing workspaces would re-pull and orphan \
             the old `runtimes/` entry"
        );
        // A vault source's `custom_label` names no entry.
        let labelled = labelled(&runtime)?;
        assert_eq!(labelled != runtime, expected.contains("__vault__"));
        assert_eq!(RuntimeStorageKey::of(&labelled)?.as_str(), expected);
        Ok(())
    }
}
