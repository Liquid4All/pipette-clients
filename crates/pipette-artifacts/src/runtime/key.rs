//! [`RuntimeStorageKey`] — a filesystem-safe, flat storage identity for a
//! [`Runtime`], the runtime-side counterpart to [`crate::model`].
//!
//! Every runtime pipette can pull gets a `runtimes/<key>/` **manifest record**
//! (like the model store records each model). The key is the runtime's identity
//! segments — a short type prefix plus the fields that distinguish one build
//! from another — normalized to a filesystem-safe token and `__`-joined, reusing
//! [`crate::model`]'s `slug_from`/`bound`. The leading type prefix
//! keeps the flat dir collision-free across runtime kinds.
//!
//! For llama.cpp and the uv/Python envs the entry also holds the extracted
//! files; a **docker** entry is manifest-only (the image lives in the docker
//! daemon).
//!
//! Keys are built only for **declared / pullable** forms. Effective-only arms
//! (`LlamacppCliStockToolsSource::RelativeDir` / `AbsoluteDir`, UV/MLX preinstalled) and
//! on-device / OS-bundled runtimes are [`RuntimeStorageKeyError::NotStorable`]
//! — no key is constructed for them.

use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use pipette_plan_types::{LlamacppCliStockToolsSource, Runtime, UvRuntimeSource};

use crate::model::{bound_to, slug_from};

/// Hex chars kept from SHA-256 for a uv requirements-body key segment: the
/// whole digest. The body is what a uv entry *is*, so the on-disk name says so
/// in full rather than making an operator re-derive it from the manifest.
const PIP_DIGEST_HEX_LEN: usize = 64;

/// Key-length ceiling for runtimes, above [`crate::model`]'s tighter cap so a
/// full requirements digest survives instead of being folded away. Well under
/// the 255-byte component limit every target filesystem allows.
const RUNTIME_MAX_LEN: usize = 128;

/// Why a [`Runtime`] has no [`RuntimeStorageKey`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeStorageKeyError {
    /// Not a pullable store identity: on-device / OS-bundled, or an effective
    /// (already-installed) source arm that is never used as a storage key.
    #[error("runtime `{0}` has no storage key (not pullable / effective-only)")]
    NotStorable(String),
}

/// Flat storage key for a recorded [`Runtime`] — its identity segments,
/// normalized and `__`-joined (see the module docs). Distinct from
/// [`Runtime`]'s `Display`/`cli_ref` (the `:`-bearing reference used on the CLI);
/// this is its flattened on-disk form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeStorageKey(String);

impl RuntimeStorageKey {
    /// Key for a **pullable declared** `runtime`, or
    /// [`RuntimeStorageKeyError::NotStorable`] when no store entry should exist.
    /// Never longer than the shared cap in [`crate::model`].
    ///
    /// Exhaustive match on kind + source; effective-only / on-device → `NotStorable`.
    pub fn of(runtime: &Runtime) -> Result<Self, RuntimeStorageKeyError> {
        let not_storable = || RuntimeStorageKeyError::NotStorable(runtime.to_string());
        let segments: Vec<String> = match runtime {
            Runtime::LlamacppCliStockTools(rt) => {
                // Declared only: github release / remote archive. Installed
                // RelativeDir / AbsoluteDir are bind-time forms, not keys.
                let mut segs = vec!["llama-cpp".to_owned()];
                match &rt.source {
                    LlamacppCliStockToolsSource::GithubRelease(repo) => {
                        segs.push(repo.repository_url.org_repo().to_owned());
                        segs.push(repo.repository_version.to_string());
                    }
                    LlamacppCliStockToolsSource::RemoteArchive { url } => {
                        segs.push("remote-archive".to_owned());
                        segs.push(url.to_string());
                    }
                    LlamacppCliStockToolsSource::RelativeDir { .. }
                    | LlamacppCliStockToolsSource::AbsoluteDir { .. } => return Err(not_storable()),
                }
                segs.push(rt.flavor.to_string());
                segs
            }
            Runtime::MlxMacosPipette(rt) => {
                // `version` is a label, not identity: nothing checks it against
                // the requirements body, so declaring one environment under two
                // versions would otherwise build it twice. See the uv arms.
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
            // `python_version` stays: it is passed to `uv venv --python`, so two
            // interpreters give two different venvs from identical requirements.
            // `build` stays as the hardware target the entry was created for.
            // install_flags are not key identity; the requirements body is.
            Runtime::UvVllm(rt) => {
                let mut segs = vec![
                    "uv-vllm".to_owned(),
                    rt.build.to_string(),
                    rt.python_version.to_string(),
                ];
                segs.extend(uv_declared_key_tail(&rt.source).map_err(|_| not_storable())?);
                segs
            }
            // `server_version` is a label like the others above, so a segment
            // earns its place only by changing what lands in the venv. That
            // leaves `python_version` and the requirements body — one wheel
            // serves CPU, GPU and NPU, and which one a cell used is a flag.
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
            Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxIosPipette(_)
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

/// Declared UV/MLX source identity tail. Preinstalled → `Err` (caller → NotStorable).
///
/// The tail is a digest of the requirements body and nothing else, so the key
/// answers "which environment is this?" rather than "what was it called". A
/// name here would let two different bodies share an entry — whichever
/// installed first would silently serve the other.
fn uv_declared_key_tail(source: &UvRuntimeSource) -> Result<Vec<String>, ()> {
    match source {
        UvRuntimeSource::PipRequirementsText { contents, .. } => Ok(vec![
            "pip".to_owned(),
            pip_requirements_digest(contents.as_ref()),
        ]),
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => Err(()),
    }
}

/// SHA-256(`contents`) as hex, truncated to [`PIP_DIGEST_HEX_LEN`].
fn pip_requirements_digest(contents: &str) -> String {
    let full = hex::encode(Sha256::digest(contents.as_bytes()));
    full[..PIP_DIGEST_HEX_LEN].to_owned()
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

    use pipette_plan_types::{
        DockerSglang, DockerVllm, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString,
        RemoteArchiveUrl, RepositoryUrl, SglangFlavor, SourceRepository, UvBuild, UvOpenvino,
        UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang, UvVllm, VllmFlavor,
    };

    use super::*;

    fn llamacpp_repo(version: &str, flavor: LlamaCppFlavor) -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: RepositoryUrl::new("github.com/ggml-org/llama.cpp"),
                repository_version: NonEmptyString::try_new(version.to_owned())?,
            }),
            flavor,
        }))
    }

    /// A uv source pinned to `requirements` — the only thing that distinguishes
    /// one uv entry from another.
    fn uv_source(requirements: &str) -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new(requirements.to_owned())?,
            install_flags: None,
        })
    }

    /// Exact key for the short (unfolded) kinds — the type prefix + identity
    /// segments, `__`-joined. (llama.cpp and docker keys exceed the 32-char cap
    /// and fold to a hash tail; covered structurally below.)
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

    /// Every key keeps its type-prefixed head, stays under the cap, and stays
    /// deterministic — whether or not it had to fold. Docker is keyed like the
    /// rest (the entry is manifest-only). The last case is long enough to
    /// actually fold at [`RUNTIME_MAX_LEN`], so the fold path stays covered now
    /// that ordinary keys fit unfolded.
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
        assert_eq!(RuntimeStorageKey::of(&runtime)?, key); // deterministic
        Ok(())
    }

    /// Two llama.cpp builds differing only in flavor get distinct keys (the hash
    /// tail disambiguates even when both fold).
    #[test]
    fn distinct_flavors_yield_distinct_keys() -> anyhow::Result<()> {
        let a = RuntimeStorageKey::of(&llamacpp_repo("b9305", LlamaCppFlavor::MacosArm64)?)?;
        let b = RuntimeStorageKey::of(&llamacpp_repo("b9305", LlamaCppFlavor::MacosX64)?)?;
        assert_ne!(a, b);
        Ok(())
    }

    fn uv_vllm(source: UvRuntimeSource) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source,
        }))
    }

    /// OS-bundled, bind-time llama dirs, and UV preinstalled never get a key.
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

    /// A `version` label is not identity. Nothing validates it against the
    /// requirements body — a runtime may declare `99.0.0` while installing
    /// `mlx-lm==0.31.3` — so keying on it would build one environment twice and
    /// split its warehouse grouping, while still not making the label true.
    #[test]
    fn a_version_label_does_not_change_the_entry() -> anyhow::Result<()> {
        fn mlx(version: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
                version: NonEmptyString::try_new(version.to_owned())?,
                flavor: MlxMacosPipetteFlavor::MacosArm64,
                source: uv_source("mlx-lm==0.31.3\n")?,
            }))
        }
        assert_eq!(
            RuntimeStorageKey::of(&mlx("0.31.3")?)?,
            RuntimeStorageKey::of(&mlx("99.0.0")?)?,
            "one requirements body is one venv, whatever it is labelled"
        );

        fn vllm(server_version: &str, python: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new(server_version.to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new(python.to_owned())?,
                source: uv_source("vllm==0.10.0\n")?,
            }))
        }
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

        // OpenVINO has a second label-shaped field the others lack: `device` is
        // like `server_version` it must not fork the store.
        fn openvino(server_version: &str) -> anyhow::Result<Runtime> {
            Ok(Runtime::UvOpenvino(UvOpenvino {
                server_version: UvServerVersion::try_new(server_version.to_owned())?,
                python_version: UvPythonVersion::try_new("3.11".to_owned())?,
                source: uv_source("openvino-genai==2026.2.1.0\n")?,
            }))
        }
        assert_eq!(
            RuntimeStorageKey::of(&openvino("2026.2.1")?)?,
            RuntimeStorageKey::of(&openvino("9.9.9")?)?,
            "`server_version` is a label here too"
        );
        Ok(())
    }

    /// Identity is the requirements body: the same body is one entry, a
    /// different body is a different entry, and `install_flags` — which change
    /// how a body installs, not which body it is — stay out of it.
    ///
    /// The `assert_ne!` on differing bodies is the one that matters. Identity
    /// used to be the catalog name, so two unrelated environments sharing a name
    /// collapsed onto one entry and whichever installed first silently served
    /// the other.
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
        // Product identity may fold under the key length cap; compare by equality.
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
        // Digest is a fixed hex prefix of SHA-256, not a DefaultHasher value.
        assert_eq!(
            pip_requirements_digest(pip_body),
            &hex::encode(sha2::Sha256::digest(pip_body.as_bytes()))[..PIP_DIGEST_HEX_LEN]
        );
        Ok(())
    }

    /// Golden storage keys — one per storable kind, including the folded forms.
    ///
    /// The uv/MLX keys were re-based twice in quick succession: once when
    /// `CatalogDefined` was retired, and again when the `version` /
    /// `server_version` labels were dropped from identity. Both moves take a
    /// name that nothing validates out of the key and leave the requirements
    /// digest to say which environment an entry holds. Each costs one
    /// re-install; neither can be undone by a later rename.
    ///
    /// A key **is** the `runtimes/<key>/` directory name on every machine that
    /// has ever pulled. Changing one does not fail anything at runtime: the
    /// lookup simply misses, the entry is re-downloaded, and the old tree is
    /// orphaned on disk. The other tests here assert shape (prefix, length cap,
    /// determinism *within one build*), which a segment-order or hash-input
    /// change would satisfy while still repointing every workspace. These
    /// literals are the only thing that catches that.
    ///
    /// Deliberately covers both the short path (`mlx` with a slug that fits)
    /// and the folded path (everything else), since folding is where the hash
    /// tail — the part most sensitive to refactoring — is produced.
    ///
    /// A diff here is a store migration, not a test update.
    #[rstest]
    #[case::llamacpp_release(
        llamacpp_repo("b9305", LlamaCppFlavor::MacosArm64)?,
        "llama-cpp__ggml-org_llama.cpp__b9305__macos-arm64"
    )]
    #[case::llamacpp_archive(
        Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new(
                    "https://example.com/llama-b9305.tar.gz".to_owned())?,
            },
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        }),
        "llama-cpp__remote-archive__example.com_llama-b9305.tar.gz__linux-x64-cpu"
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
    #[case::mlx_catalog_folded(
        Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: uv_source("mlx-lm==0.31.3\n")?,
        }),
        "mlx__macos-arm64__pip__5dc4a038a260c2db6e72a1025842f2d8d229bd4e87f95f2f757ac61ec49aaa40"
    )]
    // Same product fields as `uv_vllm_catalog`, different body. The digest tail
    // is what keeps two environments apart on disk, so this must not collide
    // with it.
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
        Ok(())
    }
}
