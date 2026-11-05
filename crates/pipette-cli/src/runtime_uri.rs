//! The compact `--runtime` URI: [`parse_runtime_uri`] and its inverse
//! [`runtime_to_uri`], over shared key constants. The grammar and keys are in
//! `docs/pipette-cli/models-and-runtimes.md`.
//!
//! The two directions agree on the runtime a URI names, not on its spelling:
//! the renderer writes keys the catalog leaves defaulted. A uv or MLX source
//! contributes no keys of its own, so a URI can only name a bundled catalog
//! preset; anything else travels as JSON. `runtime://sha256=` is resolved a
//! layer up, in [`crate::artifact_ref`].

use std::fmt;
use std::str::FromStr;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use pipette_artifacts_catalog::runtime::llamacpp::upstream_repository_url as llamacpp_upstream;
use pipette_artifacts_catalog::runtime::ovms::upstream_repository_url as ovms_upstream;
use pipette_artifacts_catalog::runtime::{mlx, onnx_genai, openvino, torch_oai};
use pipette_plan_types::{
    AppleCoreAiMacosPipette, DockerSglang, DockerVllm, FileName, LlamaCppFlavor,
    LlamacppCliStockTools, LlamacppCliStockToolsRelease, LlamacppCliStockToolsSource,
    MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString, Ovms, OvmsFlavor, OvmsSource,
    OvmsSourceRepository, RemoteArchiveUrl, RepositoryUrl, Runtime, SglangFlavor, TrtEdgellm,
    TrtEdgellmFlavor, TrtEdgellmSource, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion,
    UvSglang, UvVllm, VllmFlavor,
};

// Shared by the parser and [`runtime_to_uri`], so a rename cannot desync them.
const KEY_REPO: &str = "repo";
const KEY_VERSION: &str = "version";
const KEY_URL: &str = "url";
const KEY_FLAVOR: &str = "flavor";
const KEY_ASSET: &str = "asset";
const KEY_IMAGE: &str = "image";
const KEY_TAG: &str = "tag";
const KEY_SERVER: &str = "server";
const KEY_BUILD: &str = "build";
const KEY_PYTHON: &str = "python";
const KEY_EP: &str = "ep";

/// The importable runtime kinds, one per URI scheme. Both directions match on
/// it exhaustively, so neither can drop a kind.
///
/// A scheme is the runtime's type word with `-` for `_`, never a shorthand
/// (`a_schemes_prefix_is_its_runtime_type` enforces it). Iterable so a test can
/// hold `runtimes pull --help` to documenting every scheme.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter, strum::EnumString, strum::IntoStaticStr,
)]
#[strum(serialize_all = "kebab-case")]
pub(crate) enum Scheme {
    LlamacppCliStockTools,
    MlxMacosPipette,
    /// The wire scheme is `core-ai-macos-pipette`. The authored URI shape is
    /// public API, and the variant name follows the plan type.
    #[strum(serialize = "core-ai-macos-pipette")]
    AppleCoreAiMacosPipette,
    DockerVllm,
    DockerSglang,
    UvVllm,
    UvSglang,
    UvOpenvino,
    UvOnnxGenai,
    Ovms,
    TrtEdgellm,
}

impl Scheme {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }

    /// `apple-foundation` and the on-device app runtimes are real but not
    /// importable, so they get [`RuntimeUriError::NotRepresentable`] rather than
    /// "unknown".
    fn parse(s: &str) -> Result<Self, RuntimeUriError> {
        Scheme::from_str(s).map_err(|_| match s {
            "apple-foundation"
            | "llamacpp-apk-pipette"
            | "llamacpp-ios-pipette"
            | "mlx-ios-pipette"
            | "onnx-genai-ios-pipette" => RuntimeUriError::NotRepresentable(s.to_owned()),
            // `crate::artifact_ref` intercepts it before this parser.
            "runtime" => RuntimeUriError::NotRepresentable(
                "a `runtime://sha256=` digest here; it resolves against the local store".to_owned(),
            ),
            _ => RuntimeUriError::UnknownScheme(s.to_owned()),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeUriError {
    #[error("runtime URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`")]
    MissingSchemeSeparator,

    #[error(
        "unknown runtime URI scheme `{0}` (expected `llamacpp-cli-stock-tools`, \
         `mlx-macos-pipette`, `core-ai-macos-pipette`, `docker-vllm`, `docker-sglang`, `uv-vllm`, `uv-sglang`, \
         `uv-openvino`, `uv-onnx-genai`, `ovms`, or `trt-edgellm`; a scheme is the runtime \
         type with `-` for `_`)"
    )]
    UnknownScheme(String),

    #[error("runtime `{0}` is not representable as a URI; pass a JSON `--runtime` object instead")]
    NotRepresentable(String),

    #[error("malformed `key=value` pair `{0}` in runtime URI")]
    MalformedPair(String),

    #[error("empty key in runtime URI")]
    EmptyKey,

    #[error("duplicate key `{0}` in runtime URI")]
    DuplicateKey(String),

    #[error("unknown key `{key}` for scheme `{scheme}`")]
    UnknownKey { scheme: &'static str, key: String },

    #[error("missing required key `{key}` for scheme `{scheme}`")]
    MissingKey {
        scheme: &'static str,
        key: &'static str,
    },

    #[error("scheme `{scheme}` requires exactly one of {expected}")]
    MutuallyExclusive {
        scheme: &'static str,
        expected: &'static str,
    },

    #[error(
        "query-string URL in key `{key}` is unsupported in the URI form \
         (the `&` separator can't carry query params). Pass a JSON `--runtime` object instead"
    )]
    QueryUrl { key: &'static str },

    #[error("invalid value for key `{key}`: {message}")]
    InvalidValue { key: &'static str, message: String },

    #[error(
        "value `{value}` for key `{key}` contains the `&` pair separator and has no URI form. \
         Pass a JSON `--runtime` object instead"
    )]
    UnrepresentableValue { key: &'static str, value: String },
}

/// The `key=value` pairs of a URI body, tokenized as in [`crate::model_uri`].
struct Pairs<'a> {
    scheme: Scheme,
    items: Vec<(&'a str, &'a str)>,
}

impl<'a> Pairs<'a> {
    fn parse(scheme: Scheme, body: &'a str) -> Result<Self, RuntimeUriError> {
        let items = if body.is_empty() {
            Vec::new()
        } else {
            body.split('&')
                .try_fold(Vec::new(), |mut items: Vec<(&str, &str)>, seg| {
                    let (key, value) = seg
                        .split_once('=')
                        .ok_or_else(|| RuntimeUriError::MalformedPair(seg.to_owned()))?;
                    if key.is_empty() {
                        return Err(RuntimeUriError::EmptyKey);
                    }
                    if items.iter().any(|(existing, _)| *existing == key) {
                        return Err(RuntimeUriError::DuplicateKey(key.to_owned()));
                    }
                    items.push((key, value));
                    Ok(items)
                })?
        };
        Ok(Self { scheme, items })
    }

    fn take(&mut self, key: &str) -> Option<&'a str> {
        self.items
            .iter()
            .position(|(k, _)| *k == key)
            .map(|pos| self.items.remove(pos).1)
    }

    fn require(&mut self, key: &'static str) -> Result<&'a str, RuntimeUriError> {
        self.take(key).ok_or(RuntimeUriError::MissingKey {
            scheme: self.scheme.as_str(),
            key,
        })
    }

    /// Reject any key the scheme didn't claim.
    fn finish(self) -> Result<(), RuntimeUriError> {
        match self.items.first() {
            None => Ok(()),
            Some((key, _)) => Err(RuntimeUriError::UnknownKey {
                scheme: self.scheme.as_str(),
                key: (*key).to_owned(),
            }),
        }
    }
}

fn non_empty(key: &'static str, v: &str) -> Result<NonEmptyString, RuntimeUriError> {
    NonEmptyString::try_new(v).map_err(|e| RuntimeUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

fn release_asset_name(v: &str) -> Result<FileName, RuntimeUriError> {
    FileName::try_new(v.to_owned()).map_err(|e| RuntimeUriError::InvalidValue {
        key: KEY_ASSET,
        message: e.to_string(),
    })
}

fn remote_archive_url(key: &'static str, v: &str) -> Result<RemoteArchiveUrl, RuntimeUriError> {
    // A query string can't survive the `&` split; signed URLs go through JSON.
    if v.contains('?') {
        return Err(RuntimeUriError::QueryUrl { key });
    }
    RemoteArchiveUrl::try_new(v.to_owned()).map_err(|e| RuntimeUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

/// Resolve a torch-oai catalog slug into its requirements. The slug is composed
/// from the URI's own keys rather than accepted as one, so a URI cannot name a
/// row whose body it then misreports.
fn uv_catalog_source_from_uri(slug: &str) -> Result<UvRuntimeSource, RuntimeUriError> {
    let uv_slug =
        torch_oai::slug::UvSlug::try_new(slug).map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: e.to_string(),
        })?;
    let entry = torch_oai::lookup(&uv_slug)
        .map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: e.to_string(),
        })?
        .ok_or_else(|| {
            RuntimeUriError::NotRepresentable(format!(
                "`{slug}` is not in the bundled uv catalog; pass a JSON `--runtime` \
                 with a full pip_requirements_text source"
            ))
        })?;
    let runtime = entry
        .to_runtime()
        .map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: e.to_string(),
        })?;
    match runtime {
        Runtime::UvVllm(rt) => Ok(rt.source),
        Runtime::UvSglang(rt) => Ok(rt.source),
        other @ (Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::DockerVllm(_)
        | Runtime::DockerSglang(_)
        | Runtime::UvOpenvino(_)
        | Runtime::UvOnnxGenai(_)
        | Runtime::Ovms(_)
        | Runtime::TrtEdgellm(_)
        | Runtime::AppleFoundation(_)
        | Runtime::AppleCoreAiMacosPipette(_)) => Err(RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: format!("catalog row produced unexpected runtime {other}"),
        }),
    }
}

/// Resolve `version` against the bundled MLX catalog into its requirements.
fn mlx_catalog_source_from_uri(version: &str) -> Result<UvRuntimeSource, RuntimeUriError> {
    mlx::declared_from_catalog(version)
        .map(|rt| rt.source)
        .map_err(|e| {
            RuntimeUriError::NotRepresentable(format!(
                "mlx version `{version}` is not in the bundled catalog ({e}); pass a JSON \
                 `--runtime` with a full pip_requirements_text source"
            ))
        })
}

// The closed flavor enums have serde renames but no `Display`, so explicit maps
// keep a wrong flavor a precise `InvalidValue` and the spelling pinned.

fn vllm_flavor_str(flavor: &VllmFlavor) -> &'static str {
    match flavor {
        VllmFlavor::NvidiaGpu => "nvidia_gpu",
        VllmFlavor::AmdGpu => "amd_gpu",
        VllmFlavor::Cpu => "cpu",
    }
}

fn parse_vllm_flavor(v: &str) -> Result<VllmFlavor, RuntimeUriError> {
    match v {
        "nvidia_gpu" => Ok(VllmFlavor::NvidiaGpu),
        "amd_gpu" => Ok(VllmFlavor::AmdGpu),
        "cpu" => Ok(VllmFlavor::Cpu),
        other => Err(RuntimeUriError::InvalidValue {
            key: KEY_FLAVOR,
            message: format!("expected `nvidia_gpu`, `amd_gpu`, or `cpu`, got `{other}`"),
        }),
    }
}

fn sglang_flavor_str(flavor: &SglangFlavor) -> &'static str {
    match flavor {
        SglangFlavor::NvidiaGpu => "nvidia_gpu",
        SglangFlavor::AmdGpu => "amd_gpu",
        SglangFlavor::Cpu => "cpu",
    }
}

fn parse_sglang_flavor(v: &str) -> Result<SglangFlavor, RuntimeUriError> {
    match v {
        "nvidia_gpu" => Ok(SglangFlavor::NvidiaGpu),
        "amd_gpu" => Ok(SglangFlavor::AmdGpu),
        "cpu" => Ok(SglangFlavor::Cpu),
        other => Err(RuntimeUriError::InvalidValue {
            key: KEY_FLAVOR,
            message: format!("expected `nvidia_gpu`, `amd_gpu`, or `cpu`, got `{other}`"),
        }),
    }
}

fn parse_mlx_flavor(v: &str) -> Result<MlxMacosPipetteFlavor, RuntimeUriError> {
    match v {
        "macos-arm64" => Ok(MlxMacosPipetteFlavor::MacosArm64),
        other => Err(RuntimeUriError::InvalidValue {
            key: KEY_FLAVOR,
            message: format!("expected `macos-arm64`, got `{other}`"),
        }),
    }
}

fn parse_llama_cpp(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let repo = p.take(KEY_REPO);
    let version = p.take(KEY_VERSION);
    let asset = p.take(KEY_ASSET);
    let url = p.take(KEY_URL);
    let flavor = LlamaCppFlavor::parse(p.require(KEY_FLAVOR)?);
    let exclusive = || RuntimeUriError::MutuallyExclusive {
        scheme: p.scheme.as_str(),
        expected: "`version` (optionally with `repo` and `asset`) or `url`",
    };
    let source = match (version, url) {
        (Some(version), None) => {
            LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: repo.map_or_else(llamacpp_upstream, RepositoryUrl::new),
                repository_version: non_empty(KEY_VERSION, version)?,
                asset: asset.map(release_asset_name).transpose()?,
            })
        }
        // The URL is the file, so `repo` and `asset` don't apply.
        (None, Some(url)) => {
            if repo.is_some() || asset.is_some() {
                return Err(exclusive());
            }
            LlamacppCliStockToolsSource::RemoteArchive {
                url: remote_archive_url(KEY_URL, url)?,
            }
        }
        _ => return Err(exclusive()),
    };
    p.finish()?;
    Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source,
        flavor,
    }))
}

/// `ovms://`: llama.cpp's keys, with OVMS's upstream as the default `repo`.
/// `version` is the release tag verbatim (`v2026.4.1`).
fn parse_ovms(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let repo = p.take(KEY_REPO);
    let version = p.take(KEY_VERSION);
    let url = p.take(KEY_URL);
    let flavor = parse_ovms_flavor(p.require(KEY_FLAVOR)?)?;
    let source = match (version, url) {
        (Some(version), None) => OvmsSource::GithubRelease(OvmsSourceRepository {
            repository_url: repo.map_or_else(ovms_upstream, RepositoryUrl::new),
            repository_version: non_empty(KEY_VERSION, version)?,
        }),
        (None, Some(url)) if repo.is_none() => OvmsSource::RemoteArchive {
            url: remote_archive_url(KEY_URL, url)?,
        },
        _ => {
            return Err(RuntimeUriError::MutuallyExclusive {
                scheme: p.scheme.as_str(),
                expected: "`version` (optionally with `repo`) or `url`",
            });
        }
    };
    p.finish()?;
    Ok(Runtime::Ovms(Ovms { source, flavor }))
}

/// Through the flavor's own deserializer, so a URI is held to the JSON slug rule.
fn parse_ovms_flavor(v: &str) -> Result<OvmsFlavor, RuntimeUriError> {
    use serde::de::IntoDeserializer as _;
    OvmsFlavor::deserialize(v.into_deserializer()).map_err(|e: de::value::Error| {
        RuntimeUriError::InvalidValue {
            key: KEY_FLAVOR,
            message: e.to_string(),
        }
    })
}

/// `trt-edgellm://url=<archive>&flavor=<f>`: upstream publishes no server
/// archive, so a URI names one by URL; a vault archive travels as JSON.
fn parse_trt_edgellm(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let url = remote_archive_url(KEY_URL, p.require(KEY_URL)?)?;
    let flavor = parse_trt_edgellm_flavor(p.require(KEY_FLAVOR)?)?;
    p.finish()?;
    Ok(Runtime::TrtEdgellm(TrtEdgellm {
        source: TrtEdgellmSource::RemoteArchive { url },
        flavor,
    }))
}

/// As [`parse_ovms_flavor`].
fn parse_trt_edgellm_flavor(v: &str) -> Result<TrtEdgellmFlavor, RuntimeUriError> {
    use serde::de::IntoDeserializer as _;
    TrtEdgellmFlavor::deserialize(v.into_deserializer()).map_err(|e: de::value::Error| {
        RuntimeUriError::InvalidValue {
            key: KEY_FLAVOR,
            message: e.to_string(),
        }
    })
}

fn parse_mlx(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let version = non_empty(KEY_VERSION, p.require(KEY_VERSION)?)?;
    let flavor = match p.take(KEY_FLAVOR) {
        Some(f) => parse_mlx_flavor(f)?,
        None => MlxMacosPipetteFlavor::MacosArm64,
    };
    p.finish()?;
    let source = mlx_catalog_source_from_uri(version.as_ref())?;
    Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
        version,
        flavor,
        source,
    }))
}

fn parse_coreai(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let rt = AppleCoreAiMacosPipette::bundled();
    if let Some(version) = p.take(KEY_VERSION) {
        let version = non_empty(KEY_VERSION, version)?;
        let bundled = rt.packages.coreai_models.repository_version.as_ref();
        if version.as_ref() != bundled {
            return Err(RuntimeUriError::InvalidValue {
                key: KEY_VERSION,
                message: format!(
                    "`{version}` is not the bundled pin `{bundled}`. \
                     The sidecar uses the bundled pin only."
                ),
            });
        }
    }
    p.finish()?;
    Ok(Runtime::AppleCoreAiMacosPipette(rt))
}

fn parse_docker_vllm(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let image_name = non_empty(KEY_IMAGE, p.require(KEY_IMAGE)?)?;
    let image_tag = non_empty(KEY_TAG, p.require(KEY_TAG)?)?;
    let flavor = match p.take(KEY_FLAVOR) {
        Some(f) => parse_vllm_flavor(f)?,
        None => VllmFlavor::NvidiaGpu,
    };
    p.finish()?;
    Ok(Runtime::DockerVllm(DockerVllm {
        image_name,
        image_tag,
        flavor,
    }))
}

fn parse_docker_sglang(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let image_name = non_empty(KEY_IMAGE, p.require(KEY_IMAGE)?)?;
    let image_tag = non_empty(KEY_TAG, p.require(KEY_TAG)?)?;
    let flavor = match p.take(KEY_FLAVOR) {
        Some(f) => parse_sglang_flavor(f)?,
        None => SglangFlavor::NvidiaGpu,
    };
    p.finish()?;
    Ok(Runtime::DockerSglang(DockerSglang {
        image_name,
        image_tag,
        flavor,
    }))
}

fn parse_uv_fields(
    p: &mut Pairs,
    server_label: &str,
) -> Result<(UvServerVersion, UvBuild, UvPythonVersion, UvRuntimeSource), RuntimeUriError> {
    let server = p.require(KEY_SERVER)?;
    let build = p.require(KEY_BUILD)?;
    let python = p.require(KEY_PYTHON)?;
    let server_version =
        UvServerVersion::try_new(server).map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: e.to_string(),
        })?;
    let build = UvBuild::try_new(build).map_err(|e| RuntimeUriError::InvalidValue {
        key: KEY_BUILD,
        message: e.to_string(),
    })?;
    let python_version =
        UvPythonVersion::try_new(python).map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_PYTHON,
            message: e.to_string(),
        })?;
    let slug = format!(
        "{server_label}@{}+{}.py{}",
        server_version.as_ref(),
        build.as_ref(),
        python_version.as_ref()
    );
    let source = uv_catalog_source_from_uri(&slug)?;
    Ok((server_version, build, python_version, source))
}

fn parse_uv_vllm(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let (server_version, build, python_version, source) = parse_uv_fields(&mut p, "vllm")?;
    p.finish()?;
    Ok(Runtime::UvVllm(UvVllm {
        server_version,
        build,
        python_version,
        source,
    }))
}

fn parse_uv_sglang(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let (server_version, build, python_version, source) = parse_uv_fields(&mut p, "sglang")?;
    p.finish()?;
    Ok(Runtime::UvSglang(UvSglang {
        server_version,
        build,
        python_version,
        source,
    }))
}

pub fn parse_runtime_uri(input: &str) -> Result<Runtime, RuntimeUriError> {
    let (scheme, body) = input
        .split_once("://")
        .ok_or(RuntimeUriError::MissingSchemeSeparator)?;
    let scheme = Scheme::parse(scheme)?;
    let pairs = Pairs::parse(scheme, body)?;
    match scheme {
        Scheme::LlamacppCliStockTools => parse_llama_cpp(pairs),
        Scheme::MlxMacosPipette => parse_mlx(pairs),
        Scheme::AppleCoreAiMacosPipette => parse_coreai(pairs),
        Scheme::DockerVllm => parse_docker_vllm(pairs),
        Scheme::DockerSglang => parse_docker_sglang(pairs),
        Scheme::UvVllm => parse_uv_vllm(pairs),
        Scheme::UvSglang => parse_uv_sglang(pairs),
        Scheme::UvOpenvino => parse_openvino(pairs),
        Scheme::UvOnnxGenai => parse_onnx_genai(pairs),
        Scheme::Ovms => parse_ovms(pairs),
        Scheme::TrtEdgellm => parse_trt_edgellm(pairs),
    }
}

/// `uv-openvino://version=<v>`. No `device` key: one wheel serves CPU, GPU and
/// NPU, so the device is a flag on the cell.
fn parse_openvino(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let version = non_empty(KEY_VERSION, p.require(KEY_VERSION)?)?;
    p.finish()?;
    openvino::declared_from_catalog(version.as_ref()).map_err(|e| {
        RuntimeUriError::NotRepresentable(format!(
            "openvino version `{version}` is not in the bundled catalog ({e}); pass a \
             JSON `--runtime` with a full pip_requirements_text source"
        ))
    })
}

/// `uv-onnx-genai://version=<v>&ep=<provider>`. Each execution provider is a
/// separate wheel, so it is part of the install, not a flag on the cell.
fn parse_onnx_genai(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let version = non_empty(KEY_VERSION, p.require(KEY_VERSION)?)?;
    let ep = non_empty(KEY_EP, p.require(KEY_EP)?)?;
    p.finish()?;
    onnx_genai::declared_from_catalog(version.as_ref(), ep.as_ref()).map_err(|e| {
        RuntimeUriError::NotRepresentable(format!(
            "onnx-genai version `{version}` on `{ep}` is not in the bundled catalog ({e}); \
             pass a JSON `--runtime` with a full pip_requirements_text source"
        ))
    })
}

struct Body {
    scheme: Scheme,
    pairs: Vec<String>,
}

impl Body {
    fn new(scheme: Scheme) -> Self {
        Self {
            scheme,
            pairs: Vec::new(),
        }
    }

    /// Rejects a value containing `&`, which would fracture on re-parse, so
    /// render then parse stays total even for a free-form `Custom` flavor.
    fn push(&mut self, key: &'static str, value: &str) -> Result<(), RuntimeUriError> {
        if value.contains('&') {
            return Err(RuntimeUriError::UnrepresentableValue {
                key,
                value: value.to_owned(),
            });
        }
        self.pairs.push(format!("{key}={value}"));
        Ok(())
    }

    /// A URL with a query string has no URI form, as on the parse side.
    fn url(&mut self, key: &'static str, url: impl AsRef<str>) -> Result<(), RuntimeUriError> {
        let value = url.as_ref();
        if value.contains('?') {
            return Err(RuntimeUriError::QueryUrl { key });
        }
        self.push(key, value)
    }

    fn finish(self) -> String {
        format!("{}://{}", self.scheme.as_str(), self.pairs.join("&"))
    }
}

/// Check that `source` is exactly what this scheme's keys resolve to: any other
/// body would render a URI that re-parses into a different environment.
fn ensure_catalog_backed(
    source: &UvRuntimeSource,
    resolve: impl FnOnce() -> Result<UvRuntimeSource, RuntimeUriError>,
) -> Result<(), RuntimeUriError> {
    match source {
        UvRuntimeSource::PipRequirementsText { .. } if resolve().ok().as_ref() == Some(source) => {
            Ok(())
        }
        UvRuntimeSource::PipRequirementsText { .. } => Err(RuntimeUriError::NotRepresentable(
            "a uv runtime whose requirements are not the bundled catalog's for these keys \
             (pass it as a JSON `--runtime` object)"
                .to_owned(),
        )),
        // No key carries the principal, and a URI that dropped it would name
        // a public install of a private wheelhouse.
        UvRuntimeSource::VaultWheelhouse { .. } => Err(RuntimeUriError::NotRepresentable(
            "a uv runtime installed from a vault wheelhouse (pass it as a JSON `--runtime` \
             object)"
                .to_owned(),
        )),
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => Err(RuntimeUriError::NotRepresentable(
            "a preinstalled (local) uv/MLX runtime".to_owned(),
        )),
    }
}

/// The inverse of [`parse_runtime_uri`], with a fixed key order for stable
/// output.
pub fn runtime_to_uri(runtime: &Runtime) -> Result<String, RuntimeUriError> {
    match runtime {
        Runtime::LlamacppCliStockTools(rt) => {
            let mut body = Body::new(Scheme::LlamacppCliStockTools);
            match &rt.source {
                LlamacppCliStockToolsSource::GithubRelease(repo) => {
                    // The parser fills in the upstream default.
                    if repo.repository_url != llamacpp_upstream() {
                        body.push(KEY_REPO, repo.repository_url.as_ref())?;
                    }
                    body.push(KEY_VERSION, repo.repository_version.as_ref())?;
                    if let Some(asset) = &repo.asset {
                        body.push(KEY_ASSET, asset.as_ref())?;
                    }
                }
                LlamacppCliStockToolsSource::RemoteArchive { url } => body.url(KEY_URL, url)?,
                LlamacppCliStockToolsSource::VaultRemoteArchive { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a llama.cpp runtime from a vault archive (pass it as a JSON \
                         `--runtime` object)"
                            .to_owned(),
                    ));
                }
                // The installed form, not a fetch coordinate.
                LlamacppCliStockToolsSource::RelativeDir { .. }
                | LlamacppCliStockToolsSource::AbsoluteDir { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a local (installed) llama.cpp runtime".to_owned(),
                    ));
                }
            }
            body.push(KEY_FLAVOR, rt.flavor.as_ref())?;
            Ok(body.finish())
        }
        Runtime::Ovms(rt) => {
            let mut body = Body::new(Scheme::Ovms);
            match &rt.source {
                OvmsSource::GithubRelease(repo) => {
                    if repo.repository_url != ovms_upstream() {
                        body.push(KEY_REPO, repo.repository_url.as_ref())?;
                    }
                    body.push(KEY_VERSION, repo.repository_version.as_ref())?;
                }
                OvmsSource::RemoteArchive { url } => body.url(KEY_URL, url)?,
                OvmsSource::VaultRemoteArchive { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "an ovms runtime from a vault archive (pass it as a JSON `--runtime` \
                         object)"
                            .to_owned(),
                    ));
                }
                OvmsSource::RelativeDir { .. } | OvmsSource::AbsoluteDir { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a local (installed) ovms runtime".to_owned(),
                    ));
                }
            }
            body.push(KEY_FLAVOR, rt.flavor.as_str())?;
            Ok(body.finish())
        }
        Runtime::TrtEdgellm(rt) => {
            let mut body = Body::new(Scheme::TrtEdgellm);
            match &rt.source {
                TrtEdgellmSource::RemoteArchive { url } => body.url(KEY_URL, url)?,
                TrtEdgellmSource::VaultRemoteArchive { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a trt_edgellm runtime from a vault archive (pass it as a JSON \
                         `--runtime` object)"
                            .to_owned(),
                    ));
                }
                TrtEdgellmSource::RelativeDir { .. } | TrtEdgellmSource::AbsoluteDir { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a local (installed) trt_edgellm runtime".to_owned(),
                    ));
                }
            }
            body.push(KEY_FLAVOR, rt.flavor.as_str())?;
            Ok(body.finish())
        }
        Runtime::MlxMacosPipette(rt) => {
            let mut body = Body::new(Scheme::MlxMacosPipette);
            body.push(KEY_VERSION, rt.version.as_ref())?;
            body.push(KEY_FLAVOR, mlx_flavor_str(&rt.flavor))?;
            ensure_catalog_backed(&rt.source, || {
                mlx_catalog_source_from_uri(rt.version.as_ref())
            })?;
            Ok(body.finish())
        }
        Runtime::UvOpenvino(rt) => {
            let mut body = Body::new(Scheme::UvOpenvino);
            body.push(KEY_VERSION, rt.server_version.as_ref())?;
            ensure_catalog_backed(&rt.source, || {
                match openvino::declared_from_catalog(rt.server_version.as_ref()) {
                    Ok(Runtime::UvOpenvino(catalog_rt)) => Ok(catalog_rt.source),
                    _ => Err(RuntimeUriError::NotRepresentable(
                        "an openvino runtime outside the bundled catalog".to_owned(),
                    )),
                }
            })?;
            Ok(body.finish())
        }
        Runtime::UvOnnxGenai(rt) => {
            let mut body = Body::new(Scheme::UvOnnxGenai);
            body.push(KEY_VERSION, rt.server_version.as_ref())?;
            body.push(KEY_EP, rt.execution_provider.as_str())?;
            ensure_catalog_backed(&rt.source, || {
                match onnx_genai::declared_from_catalog(
                    rt.server_version.as_ref(),
                    rt.execution_provider.as_str(),
                ) {
                    Ok(Runtime::UvOnnxGenai(catalog_rt)) => Ok(catalog_rt.source),
                    _ => Err(RuntimeUriError::NotRepresentable(
                        "an onnx-genai runtime outside the bundled catalog".to_owned(),
                    )),
                }
            })?;
            Ok(body.finish())
        }
        Runtime::DockerVllm(rt) => {
            let mut body = Body::new(Scheme::DockerVllm);
            body.push(KEY_IMAGE, rt.image_name.as_ref())?;
            body.push(KEY_TAG, rt.image_tag.as_ref())?;
            body.push(KEY_FLAVOR, vllm_flavor_str(&rt.flavor))?;
            Ok(body.finish())
        }
        Runtime::DockerSglang(rt) => {
            let mut body = Body::new(Scheme::DockerSglang);
            body.push(KEY_IMAGE, rt.image_name.as_ref())?;
            body.push(KEY_TAG, rt.image_tag.as_ref())?;
            body.push(KEY_FLAVOR, sglang_flavor_str(&rt.flavor))?;
            Ok(body.finish())
        }
        Runtime::UvVllm(rt) => {
            let mut body = Body::new(Scheme::UvVllm);
            body.push(KEY_SERVER, rt.server_version.as_ref())?;
            body.push(KEY_BUILD, rt.build.as_ref())?;
            body.push(KEY_PYTHON, rt.python_version.as_ref())?;
            let slug = format!(
                "vllm@{}+{}.py{}",
                rt.server_version.as_ref(),
                rt.build.as_ref(),
                rt.python_version.as_ref()
            );
            ensure_catalog_backed(&rt.source, || uv_catalog_source_from_uri(&slug))?;
            Ok(body.finish())
        }
        Runtime::UvSglang(rt) => {
            let mut body = Body::new(Scheme::UvSglang);
            body.push(KEY_SERVER, rt.server_version.as_ref())?;
            body.push(KEY_BUILD, rt.build.as_ref())?;
            body.push(KEY_PYTHON, rt.python_version.as_ref())?;
            let slug = format!(
                "sglang@{}+{}.py{}",
                rt.server_version.as_ref(),
                rt.build.as_ref(),
                rt.python_version.as_ref()
            );
            ensure_catalog_backed(&rt.source, || uv_catalog_source_from_uri(&slug))?;
            Ok(body.finish())
        }
        Runtime::AppleCoreAiMacosPipette(rt) => {
            if !rt.is_bundled() {
                return Err(RuntimeUriError::NotRepresentable(
                    "non-bundled Core AI Swift stack".to_owned(),
                ));
            }
            let mut body = Body::new(Scheme::AppleCoreAiMacosPipette);
            body.push(
                KEY_VERSION,
                rt.packages.coreai_models.repository_version.as_ref(),
            )?;
            Ok(body.finish())
        }
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
        | Runtime::OnnxGenaiIosPipette(_)
        | Runtime::AppleFoundation(_) => Err(RuntimeUriError::NotRepresentable(
            runtime.headless_token().to_owned(),
        )),
    }
}

fn mlx_flavor_str(flavor: &MlxMacosPipetteFlavor) -> &'static str {
    match flavor {
        MlxMacosPipetteFlavor::MacosArm64 => "macos-arm64",
    }
}

/// The `--runtime` dispatch: a JSON object (reusing `Runtime`'s `Deserialize`)
/// when the trimmed arg starts with `{`, else the compact URI grammar.
pub fn parse_runtime_arg(arg: &str) -> anyhow::Result<Runtime> {
    let trimmed = arg.trim();
    let runtime = if trimmed.starts_with('{') {
        serde_json::from_str::<Runtime>(trimmed)?
    } else {
        parse_runtime_uri(trimmed)?
    };
    // A JSON/TOML `Runtime` object bypasses the URI grammar, so re-check the
    // Core AI pin here: the sidecar is compiled against the bundled stack only,
    // and a non-bundled `packages` would record a pin the binary does not have.
    if let Runtime::AppleCoreAiMacosPipette(rt) = &runtime
        && !rt.is_bundled()
    {
        anyhow::bail!(
            "the core-ai-macos-pipette runtime pins a Swift stack that this client \
             does not contain. Use the bundled pin."
        );
    }
    Ok(runtime)
}

/// A [`Runtime`] that reads a URI string or the structured object, and writes
/// the URI when the runtime has one, else the structured form.
#[derive(Debug, Clone)]
pub struct RuntimeUri(pub Runtime);

impl<'de> Deserialize<'de> for RuntimeUri {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RuntimeUriVisitor;

        impl<'de> Visitor<'de> for RuntimeUriVisitor {
            type Value = Runtime;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a runtime URI string or a Runtime object")
            }

            fn visit_str<E>(self, uri: &str) -> Result<Runtime, E>
            where
                E: de::Error,
            {
                parse_runtime_uri(uri).map_err(de::Error::custom)
            }

            fn visit_map<A>(self, map: A) -> Result<Runtime, A::Error>
            where
                A: MapAccess<'de>,
            {
                Runtime::deserialize(de::value::MapAccessDeserializer::new(map))
            }
        }

        deserializer
            .deserialize_any(RuntimeUriVisitor)
            .map(RuntimeUri)
    }
}

impl Serialize for RuntimeUri {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match runtime_to_uri(&self.0) {
            Ok(uri) => serializer.serialize_str(&uri),
            Err(_) => self.0.serialize(serializer),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn llama_cpp_repo_default_upstream() -> anyhow::Result<()> {
        let runtime =
            parse_runtime_uri("llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64")?;
        let Runtime::LlamacppCliStockTools(LlamacppCliStockTools { source, flavor }) = &runtime
        else {
            anyhow::bail!("expected llamacpp, got {runtime:?}");
        };
        let LlamacppCliStockToolsSource::GithubRelease(repo) = source else {
            anyhow::bail!("expected a GitHubRelease source");
        };
        assert_eq!(repo.repository_url, llamacpp_upstream());
        assert_eq!(repo.repository_version.as_ref(), "b9305");
        assert_eq!(flavor.to_string(), "macos-arm64");
        Ok(())
    }

    #[test]
    fn llama_cpp_repo_explicit_and_custom_flavor() -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(
            "llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=my-fork",
        )?;
        let Runtime::LlamacppCliStockTools(LlamacppCliStockTools { source, flavor }) = &runtime
        else {
            anyhow::bail!("expected llamacpp");
        };
        let LlamacppCliStockToolsSource::GithubRelease(repo) = source else {
            anyhow::bail!("expected Repository");
        };
        assert_eq!(repo.repository_url.as_ref(), "github.com/acme/llama.cpp");
        assert_eq!(flavor.to_string(), "my-fork");
        Ok(())
    }

    #[test]
    fn llama_cpp_archive() -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(
            "llamacpp-cli-stock-tools://url=https://ex.com/llama-b1.tar.gz&flavor=macos-arm64",
        )?;
        assert!(matches!(
            runtime,
            Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::RemoteArchive { .. },
                ..
            })
        ));
        Ok(())
    }

    #[rstest]
    #[case::version(
        "llamacpp-cli-stock-tools://version=b1&url=https://x/a.tar.gz&flavor=macos-arm64"
    )]
    #[case::repo(
        "llamacpp-cli-stock-tools://repo=github.com/a/b&url=https://x/a.tgz&flavor=macos-arm64"
    )]
    #[case::asset("llamacpp-cli-stock-tools://url=https://x/a.tgz&asset=a.tgz&flavor=macos-arm64")]
    fn llama_cpp_release_keys_and_url_are_exclusive(#[case] uri: &str) {
        assert!(matches!(
            parse_runtime_uri(uri),
            Err(RuntimeUriError::MutuallyExclusive { .. })
        ));
    }

    /// OVMS's upstream, not llama.cpp's.
    #[test]
    fn an_ovms_release_defaults_to_the_upstream_repo() -> anyhow::Result<()> {
        let runtime = parse_runtime_uri("ovms://version=v2026.4.1&flavor=windows-x64")?;
        let Runtime::Ovms(Ovms {
            source: OvmsSource::GithubRelease(repo),
            flavor,
        }) = &runtime
        else {
            anyhow::bail!("expected an ovms release, got {runtime:?}");
        };
        assert_eq!(repo.repository_url, ovms_upstream());
        assert_eq!(repo.repository_version.as_ref(), "v2026.4.1");
        assert_eq!(*flavor, OvmsFlavor::WindowsX64);
        Ok(())
    }

    #[rstest]
    #[case::both("ovms://version=v2026.4.1&url=https://x/a.zip&flavor=windows-x64")]
    #[case::repo_with_url("ovms://repo=github.com/a/b&url=https://x/a.zip&flavor=windows-x64")]
    #[case::neither("ovms://flavor=windows-x64")]
    fn an_ovms_ref_names_a_release_or_an_archive(#[case] uri: &str) {
        assert!(
            matches!(
                parse_runtime_uri(uri),
                Err(RuntimeUriError::MutuallyExclusive { .. })
            ),
            "{uri}"
        );
    }

    #[test]
    fn mlx_minimal_defaults_flavor_and_catalog() -> anyhow::Result<()> {
        let runtime = parse_runtime_uri("mlx-macos-pipette://version=0.31.3")?;
        let Runtime::MlxMacosPipette(rt) = &runtime else {
            anyhow::bail!("expected mlx");
        };
        assert_eq!(rt.version.as_ref(), "0.31.3");
        assert_eq!(rt.flavor, MlxMacosPipetteFlavor::MacosArm64);
        let UvRuntimeSource::PipRequirementsText { contents, .. } = &rt.source else {
            anyhow::bail!("expected PipRequirementsText, got {:?}", rt.source);
        };
        assert!(
            contents.as_ref().contains("mlx-lm==0.31.3"),
            "expected catalog requirements, got {contents}"
        );
        Ok(())
    }

    /// Rendering it would emit `mlx-macos-pipette://version=0.31.3`, which
    /// re-parses into the catalog's environment.
    #[test]
    fn a_uv_runtime_off_the_catalog_has_no_uri_form() -> anyhow::Result<()> {
        let runtime = Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.31.3\n".to_owned())?,
                install_flags: None,
            },
        });
        assert!(
            matches!(
                runtime_to_uri(&runtime),
                Err(RuntimeUriError::NotRepresentable(_))
            ),
            "a one-line body is not the catalog's locked body for 0.31.3"
        );
        Ok(())
    }

    #[rstest]
    #[case("docker-vllm://image=vllm/vllm-openai&tag=v0.10.0")]
    #[case("docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=amd_gpu")]
    fn docker_parses(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri(uri)?,
            Runtime::DockerVllm(_) | Runtime::DockerSglang(_)
        ));
        Ok(())
    }

    #[rstest]
    #[case("uv-vllm://server=0.21.0&build=cu121&python=3.12", "vllm==")]
    #[case("uv-sglang://server=0.5.12.post1&build=cu121&python=3.12", "sglang")]
    fn uv_parses_fills_catalog_requirements(
        #[case] uri: &str,
        #[case] req_needle: &str,
    ) -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(uri)?;
        let source = match &runtime {
            Runtime::UvVllm(rt) => &rt.source,
            Runtime::UvSglang(rt) => &rt.source,
            other @ (Runtime::LlamacppCliStockTools(_)
            | Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxMacosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::DockerVllm(_)
            | Runtime::DockerSglang(_)
            | Runtime::UvOpenvino(_)
            | Runtime::UvOnnxGenai(_)
            | Runtime::Ovms(_)
            | Runtime::TrtEdgellm(_)
            | Runtime::AppleFoundation(_)
            | Runtime::AppleCoreAiMacosPipette(_)) => {
                anyhow::bail!("expected uv runtime, got {other:?}")
            }
        };
        let UvRuntimeSource::PipRequirementsText { contents, .. } = source else {
            anyhow::bail!("expected PipRequirementsText, got {source:?}");
        };
        assert!(
            contents.as_ref().contains(req_needle),
            "expected catalog requirements containing {req_needle:?}, got {contents}"
        );
        Ok(())
    }

    #[rstest]
    #[case(
        r#"{"type":"docker_vllm","image_name":"vllm","image_tag":"v0.10.0","flavor":"nvidia_gpu"}"#
    )]
    #[case(r#"  {"type":"docker_vllm","image_name":"vllm","image_tag":"v0.10.0","flavor":"nvidia_gpu"}  "#)]
    fn dispatch_json_object(#[case] arg: &str) -> anyhow::Result<()> {
        assert!(matches!(parse_runtime_arg(arg)?, Runtime::DockerVllm(_)));
        Ok(())
    }

    #[rstest]
    #[case("mlx-macos-pipette://version=0.31.3")]
    #[case("   mlx-macos-pipette://version=0.31.3   ")]
    fn dispatch_uri(#[case] arg: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_arg(arg)?,
            Runtime::MlxMacosPipette(_)
        ));
        Ok(())
    }

    #[test]
    fn unknown_scheme() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("foo://version=b1"),
            Err(RuntimeUriError::UnknownScheme(_))
        ));
        Ok(())
    }

    #[rstest]
    #[case("apple-foundation://")]
    #[case("llamacpp-apk-pipette://version=b1&flavor=android-arm64-v8")]
    #[case("mlx-ios-pipette://version=0.31.3")]
    fn not_representable_schemes(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri(uri),
            Err(RuntimeUriError::NotRepresentable(_))
        ));
        Ok(())
    }

    #[rstest]
    #[case("mlx:version=0.31.3")]
    #[case("just-text")]
    fn missing_scheme_separator(#[case] uri: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri(uri),
            Err(RuntimeUriError::MissingSchemeSeparator)
        ));
        Ok(())
    }

    // Keys required unconditionally; llama.cpp's version/url is MutuallyExclusive.
    #[rstest]
    #[case("docker-vllm://image=vllm", "tag")]
    #[case("uv-vllm://server=0.21.0&build=cu121", "python")]
    // `ep` selects half the catalog key, so a ref without it names no row.
    #[case("uv-onnx-genai://version=0.16.0", "ep")]
    #[case("uv-onnx-genai://ep=cpu", "version")]
    #[case("ovms://version=v2026.4.1", "flavor")]
    fn missing_required_key(#[case] uri: &str, #[case] key: &str) -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri(uri),
            Err(RuntimeUriError::MissingKey { key: got, .. }) if got == key
        ));
        Ok(())
    }

    #[test]
    fn duplicate_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("mlx-macos-pipette://version=a&version=b"),
            Err(RuntimeUriError::DuplicateKey(_))
        ));
        Ok(())
    }

    #[test]
    fn unknown_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("mlx-macos-pipette://version=0.31.3&foo=bar"),
            Err(RuntimeUriError::UnknownKey { key, .. }) if key == "foo"
        ));
        Ok(())
    }

    #[test]
    fn empty_key() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("mlx-macos-pipette://=abc"),
            Err(RuntimeUriError::EmptyKey)
        ));
        Ok(())
    }

    #[test]
    fn malformed_pair() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("mlx-macos-pipette://version"),
            Err(RuntimeUriError::MalformedPair(_))
        ));
        Ok(())
    }

    #[test]
    fn query_url_rejected() -> anyhow::Result<()> {
        let err = parse_runtime_uri(
            "llamacpp-cli-stock-tools://url=https://x/a.tgz?sig=1&flavor=macos-arm64",
        )
        .err()
        .ok_or_else(|| anyhow::anyhow!("expected rejection"))?;
        assert!(matches!(err, RuntimeUriError::QueryUrl { key: "url" }));
        Ok(())
    }

    #[rstest]
    #[case("llamacpp-cli-stock-tools://url=not-a-url&flavor=macos-arm64", "url")]
    #[case(
        "llamacpp-cli-stock-tools://url=file:///tmp/llama-b1.tar.gz&flavor=macos-arm64",
        "url"
    )]
    #[case(
        "llamacpp-cli-stock-tools://version=b1&asset=bin/a.tgz&flavor=macos-arm64",
        "asset"
    )]
    #[case("uv-vllm://server=0.21.0&build=notabuild&python=3.12", "build")]
    // The ovms flavor is held to the wire's slug rule: it lands in the storage key.
    #[case("ovms://version=v2026.4.1&flavor=acme/arm64", "flavor")]
    #[case("ovms://version=v2026.4.1&flavor=", "flavor")]
    #[case("trt-edgellm://url=https://ex.com/t.tar.gz&flavor=jetpack/7", "flavor")]
    fn invalid_value(#[case] uri: &str, #[case] key: &str) -> anyhow::Result<()> {
        let err = parse_runtime_uri(uri)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{uri}` to be rejected"))?;
        let RuntimeUriError::InvalidValue { key: got, .. } = err else {
            anyhow::bail!("expected InvalidValue, got {err:?}");
        };
        assert_eq!(got, key);
        Ok(())
    }

    /// The message points at the JSON form.
    #[rstest]
    #[case::off_catalog_version("uv-onnx-genai://version=0.0.0&ep=cpu")]
    #[case::off_catalog_provider("uv-onnx-genai://version=0.16.0&ep=directml")]
    fn an_unlisted_onnx_genai_pair_is_not_representable(#[case] uri: &str) -> anyhow::Result<()> {
        let Err(RuntimeUriError::NotRepresentable(msg)) = parse_runtime_uri(uri) else {
            anyhow::bail!("expected `{uri}` to be refused as unlisted");
        };
        assert!(msg.contains("not in the bundled catalog"), "got {msg}");
        assert!(msg.contains("pip_requirements_text"), "got {msg}");
        Ok(())
    }

    #[test]
    fn render_rejects_on_device_and_apple_foundation() -> anyhow::Result<()> {
        assert!(matches!(
            runtime_to_uri(&Runtime::AppleFoundation(Default::default())),
            Err(RuntimeUriError::NotRepresentable(_))
        ));
        Ok(())
    }

    #[test]
    fn render_rejects_value_containing_the_pair_separator() -> anyhow::Result<()> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(LlamacppCliStockToolsRelease {
                repository_url: llamacpp_upstream(),
                repository_version: NonEmptyString::try_new("b1")?,
                asset: None,
            }),
            flavor: LlamaCppFlavor::parse("weird&flavor"),
        });
        assert!(matches!(
            runtime_to_uri(&runtime),
            Err(RuntimeUriError::UnrepresentableValue { key: "flavor", .. })
        ));
        Ok(())
    }

    // Canonical URIs (keys in render order), so the render equals the input.
    #[rstest]
    #[case("llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64")]
    #[case("llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=macos-x64")]
    #[case(
        "llamacpp-cli-stock-tools://version=b11399&asset=llama-b11399-bin-win-cuda-13.4-x64.zip\
         &flavor=windows-x64-cuda-13-1"
    )]
    #[case("llamacpp-cli-stock-tools://url=https://ex.com/llama-b1.tar.gz&flavor=macos-arm64")]
    #[case("mlx-macos-pipette://version=0.31.3&flavor=macos-arm64")]
    #[case("mlx-macos-pipette://version=0.32.2&flavor=macos-arm64")]
    #[case("mlx-macos-pipette://version=0.32.0&flavor=macos-arm64")]
    #[case("mlx-macos-pipette://version=0.7.4&flavor=macos-arm64")]
    #[case("docker-vllm://image=vllm/vllm-openai&tag=v0.10.0&flavor=nvidia_gpu")]
    #[case("docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=amd_gpu")]
    #[case("uv-vllm://server=0.21.0&build=cu121&python=3.12")]
    #[case("uv-sglang://server=0.5.12.post1&build=cu121&python=3.12")]
    #[case("uv-openvino://version=2026.4.1")]
    #[case("uv-onnx-genai://version=0.16.0&ep=cpu")]
    #[case("uv-onnx-genai://version=0.16.0&ep=cuda")]
    #[case("ovms://version=v2026.4.1&flavor=windows-x64")]
    #[case("ovms://repo=github.com/acme/model_server&version=v2026.4.1&flavor=ubuntu24-x64")]
    #[case("ovms://url=https://ex.com/ovms.zip&flavor=acme-x64")]
    #[case(
        "trt-edgellm://url=https://ex.com/trt-edgellm-server-4e3f9d23f052.tar.gz&flavor=jetson-orin-cuda-13"
    )]
    fn round_trips(#[case] uri: &str) -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(uri)?;
        let rendered = runtime_to_uri(&runtime)?;
        assert_eq!(rendered, uri, "render of a canonical URI is byte-identical");
        assert_eq!(
            parse_runtime_uri(&rendered)?,
            runtime,
            "re-parse recovers the runtime"
        );
        Ok(())
    }

    /// ONNX Runtime's own API spells providers in capitals, and a JSON
    /// `--runtime` accepts them too.
    #[rstest]
    #[case::upper("uv-onnx-genai://version=0.16.0&ep=CUDA")]
    #[case::mixed("uv-onnx-genai://version=0.16.0&ep=Cuda")]
    fn a_provider_parses_in_any_case_and_renders_canonically(
        #[case] uri: &str,
    ) -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(uri)?;
        assert_eq!(
            runtime,
            parse_runtime_uri("uv-onnx-genai://version=0.16.0&ep=cuda")?
        );
        assert_eq!(
            runtime_to_uri(&runtime)?,
            "uv-onnx-genai://version=0.16.0&ep=cuda"
        );
        Ok(())
    }

    #[test]
    fn runtime_uri_serde_round_trip_via_uri_string() -> anyhow::Result<()> {
        let RuntimeUri(runtime) =
            serde_json::from_str(r#""mlx-macos-pipette://version=0.31.3&flavor=macos-arm64""#)?;
        assert!(matches!(runtime, Runtime::MlxMacosPipette(_)));
        let back = serde_json::to_string(&RuntimeUri(runtime))?;
        assert_eq!(
            back,
            r#""mlx-macos-pipette://version=0.31.3&flavor=macos-arm64""#
        );
        Ok(())
    }

    #[test]
    fn runtime_uri_serializes_non_representable_structurally() -> anyhow::Result<()> {
        let json =
            serde_json::to_string(&RuntimeUri(Runtime::AppleFoundation(Default::default())))?;
        assert!(json.starts_with('{'));
        let RuntimeUri(back) = serde_json::from_str(&json)?;
        assert_eq!(back, Runtime::AppleFoundation(Default::default()));
        Ok(())
    }

    /// Reaching this parser means the caller skipped `artifact_ref`.
    #[test]
    fn a_digest_ref_names_where_it_resolves() -> anyhow::Result<()> {
        let Err(RuntimeUriError::NotRepresentable(msg)) =
            parse_runtime_uri("runtime://sha256=abcdef")
        else {
            anyhow::bail!("expected the digest scheme to be refused");
        };
        assert!(msg.contains("local store"), "got {msg}");
        Ok(())
    }

    #[test]
    fn a_runtime_uri_rejects_a_non_string_non_object() -> anyhow::Result<()> {
        let err = serde_json::from_str::<RuntimeUri>("42")
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected a number to be refused"))?;
        assert!(
            err.to_string()
                .contains("a runtime URI string or a Runtime object"),
            "got {err}"
        );
        Ok(())
    }

    #[rstest]
    #[case::vllm_amd("docker-vllm://image=i&tag=t&flavor=amd_gpu", "amd_gpu")]
    #[case::vllm_cpu("docker-vllm://image=i&tag=t&flavor=cpu", "cpu")]
    #[case::vllm_default("docker-vllm://image=i&tag=t", "nvidia_gpu")]
    #[case::sglang_nvidia("docker-sglang://image=i&tag=t&flavor=nvidia_gpu", "nvidia_gpu")]
    #[case::sglang_cpu("docker-sglang://image=i&tag=t&flavor=cpu", "cpu")]
    #[case::sglang_default("docker-sglang://image=i&tag=t", "nvidia_gpu")]
    fn docker_flavors_parse_and_render(
        #[case] uri: &str,
        #[case] flavor: &str,
    ) -> anyhow::Result<()> {
        let runtime = parse_runtime_uri(uri)?;
        let rendered = if let Runtime::DockerVllm(rt) = &runtime {
            vllm_flavor_str(&rt.flavor)
        } else if let Runtime::DockerSglang(rt) = &runtime {
            sglang_flavor_str(&rt.flavor)
        } else {
            anyhow::bail!("expected a docker runtime, got {runtime:?}");
        };
        assert_eq!(rendered, flavor);
        assert!(runtime_to_uri(&runtime)?.ends_with(&format!("&flavor={flavor}")));
        Ok(())
    }

    #[rstest]
    #[case::vllm("docker-vllm://image=i&tag=t&flavor=tpu")]
    #[case::sglang("docker-sglang://image=i&tag=t&flavor=tpu")]
    #[case::mlx("mlx-macos-pipette://version=0.31.3&flavor=macos-x64")]
    #[case::empty_image("docker-vllm://image=&tag=t")]
    fn an_unknown_flavor_or_empty_value_is_an_invalid_value(
        #[case] uri: &str,
    ) -> anyhow::Result<()> {
        let err = parse_runtime_uri(uri)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{uri}` to be rejected"))?;
        assert!(
            matches!(err, RuntimeUriError::InvalidValue { .. }),
            "got {err:?}"
        );
        Ok(())
    }

    #[rstest]
    #[case::server("uv-vllm://server=not a version&build=cu121&python=3.12", "server")]
    #[case::python("uv-vllm://server=0.21.0&build=cu121&python=three", "python")]
    fn a_malformed_uv_field_names_its_key(
        #[case] uri: &str,
        #[case] key: &str,
    ) -> anyhow::Result<()> {
        let err = parse_runtime_uri(uri)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected `{uri}` to be rejected"))?;
        let RuntimeUriError::InvalidValue { key: got, .. } = err else {
            anyhow::bail!("expected InvalidValue, got {err:?}");
        };
        assert_eq!(got, key);
        Ok(())
    }

    /// A version-keyed miss lists what is bundled.
    #[rstest]
    #[case::vllm("uv-vllm://server=0.0.1&build=cu121&python=3.12", None)]
    #[case::sglang("uv-sglang://server=0.0.1&build=cu121&python=3.12", None)]
    #[case::openvino("uv-openvino://version=0.0.0", Some("2026.4.0"))]
    #[case::mlx("mlx-macos-pipette://version=0.0.0", Some("0.31.3"))]
    fn an_unlisted_uv_row_is_not_representable(
        #[case] uri: &str,
        #[case] bundled: Option<&str>,
    ) -> anyhow::Result<()> {
        let Err(RuntimeUriError::NotRepresentable(msg)) = parse_runtime_uri(uri) else {
            anyhow::bail!("expected `{uri}` to be refused as unlisted");
        };
        assert!(msg.contains("pip_requirements_text"), "got {msg}");
        assert!(
            bundled.is_none_or(|version| msg.contains(version)),
            "should list the bundled versions: {msg}"
        );
        Ok(())
    }

    #[rstest]
    #[case::llama_cpp(
        r#"{"type":"llamacpp_cli_stock_tools","source":"relative_dir","dir":"blobs","flavor":"macos-arm64"}"#
    )]
    #[case::ovms(r#"{"type":"ovms","source":"relative_dir","dir":"blobs","flavor":"windows-x64"}"#)]
    fn render_rejects_an_installed_layout(#[case] json: &str) -> anyhow::Result<()> {
        let runtime: Runtime = serde_json::from_str(json)?;
        let Err(RuntimeUriError::NotRepresentable(msg)) = runtime_to_uri(&runtime) else {
            anyhow::bail!("expected the installed layout to be refused: {json}");
        };
        assert!(msg.contains("installed"), "got {msg}");
        Ok(())
    }

    #[test]
    fn render_rejects_a_query_string_archive_url() -> anyhow::Result<()> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::RemoteArchive {
                url: RemoteArchiveUrl::try_new("https://ex.com/a.tgz?sig=1".to_owned())?,
            },
            flavor: LlamaCppFlavor::parse("macos-arm64"),
        });
        assert!(matches!(
            runtime_to_uri(&runtime),
            Err(RuntimeUriError::QueryUrl { key: "url" })
        ));
        Ok(())
    }

    #[test]
    fn render_rejects_a_preinstalled_uv_source() -> anyhow::Result<()> {
        let runtime = Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31.3".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::RelativePreinstalled {
                dir: pipette_plan_types::RelativePath::try_new("venv")?,
            },
        });
        let Err(RuntimeUriError::NotRepresentable(msg)) = runtime_to_uri(&runtime) else {
            anyhow::bail!("expected the preinstalled venv to be refused");
        };
        assert!(msg.contains("preinstalled"), "got {msg}");
        Ok(())
    }

    #[rstest]
    #[case::llamacpp(
        r#"{"type":"llamacpp_cli_stock_tools","source":"vault_remote_archive","url":"/vault/resources/vault_1/content","principal_id":"org_1","flavor":"macos-arm64"}"#
    )]
    #[case::uv_vllm(
        r#"{"type":"uv_vllm","server_version":"0.10.0","build":"cu121","python_version":"3.12","source":{"type":"vault_wheelhouse","contents":"vllm==0.10.0\n","url":"/vault/resources/vault_1/content","principal_id":"org_1"}}"#
    )]
    #[case::ovms(
        r#"{"type":"ovms","source":"vault_remote_archive","url":"/vault/resources/vault_1/content","principal_id":"org_1","flavor":"windows-x64"}"#
    )]
    fn render_rejects_a_vault_runtime(#[case] json: &str) -> anyhow::Result<()> {
        let runtime: Runtime = serde_json::from_str(json)?;
        let Err(RuntimeUriError::NotRepresentable(msg)) = runtime_to_uri(&runtime) else {
            anyhow::bail!("expected the vault runtime to be refused");
        };
        assert!(msg.contains("vault"), "got {msg}");
        assert!(msg.contains("JSON"), "got {msg}");
        Ok(())
    }

    /// The resolver has no body to compare against, so there is no URI form.
    #[rstest]
    #[case::openvino("uv-openvino://version=2026.4.1")]
    #[case::onnx_genai("uv-onnx-genai://version=0.16.0&ep=cpu")]
    fn render_rejects_a_catalog_runtime_off_its_catalog(#[case] uri: &str) -> anyhow::Result<()> {
        let off_catalog = UvServerVersion::try_new("0.0.1")?;
        let runtime = match parse_runtime_uri(uri)? {
            Runtime::UvOpenvino(mut rt) => {
                rt.server_version = off_catalog;
                Runtime::UvOpenvino(rt)
            }
            Runtime::UvOnnxGenai(mut rt) => {
                rt.server_version = off_catalog;
                Runtime::UvOnnxGenai(rt)
            }
            other @ (Runtime::LlamacppCliStockTools(_)
            | Runtime::LlamacppApkPipette(_)
            | Runtime::LlamacppIosPipette(_)
            | Runtime::MlxMacosPipette(_)
            | Runtime::MlxIosPipette(_)
            | Runtime::OnnxGenaiIosPipette(_)
            | Runtime::DockerVllm(_)
            | Runtime::DockerSglang(_)
            | Runtime::UvVllm(_)
            | Runtime::UvSglang(_)
            | Runtime::Ovms(_)
            | Runtime::TrtEdgellm(_)
            | Runtime::AppleFoundation(_)
            | Runtime::AppleCoreAiMacosPipette(_)) => {
                anyhow::bail!("expected a catalog runtime, got {other:?}")
            }
        };
        assert!(
            matches!(
                runtime_to_uri(&runtime),
                Err(RuntimeUriError::NotRepresentable(_))
            ),
            "the body is the catalog's for a version the catalog lacks"
        );
        Ok(())
    }
}

#[cfg(test)]
mod core_ai_port_tests {
    use super::*;

    #[test]
    fn bundled_core_ai_uri_round_trips_and_rejects_false_pins() -> anyhow::Result<()> {
        let bundled = Runtime::AppleCoreAiMacosPipette(AppleCoreAiMacosPipette::bundled());
        let uri = runtime_to_uri(&bundled)?;
        assert!(uri.starts_with("core-ai-macos-pipette://version="));
        assert_eq!(parse_runtime_uri(&uri)?, bundled);
        assert!(matches!(
            parse_runtime_uri("core-ai-macos-pipette://version=9.9.9"),
            Err(RuntimeUriError::InvalidValue {
                key: KEY_VERSION,
                ..
            })
        ));
        let mut wrong = AppleCoreAiMacosPipette::bundled();
        wrong.packages.coreai_models.repository_version = NonEmptyString::try_new("9.9.9")?;
        assert!(runtime_to_uri(&Runtime::AppleCoreAiMacosPipette(wrong)).is_err());
        Ok(())
    }

    /// The JSON form skips the URI grammar, so `parse_runtime_arg` checks the
    /// whole Swift stack itself.
    #[test]
    fn a_json_core_ai_runtime_with_a_foreign_stack_is_refused() -> anyhow::Result<()> {
        let bundled = Runtime::AppleCoreAiMacosPipette(AppleCoreAiMacosPipette::bundled());
        assert_eq!(
            parse_runtime_arg(&serde_json::to_string(&bundled)?)?,
            bundled
        );

        let mut foreign = AppleCoreAiMacosPipette::bundled();
        foreign.packages.swift_transformers.repository_version = NonEmptyString::try_new("9.9.9")?;
        let json = serde_json::to_string(&Runtime::AppleCoreAiMacosPipette(foreign))?;
        let Err(err) = parse_runtime_arg(&json) else {
            anyhow::bail!("a foreign Swift stack was accepted");
        };
        anyhow::ensure!(format!("{err:#}").contains("bundled pin"), "{err:#}");
        Ok(())
    }
}
