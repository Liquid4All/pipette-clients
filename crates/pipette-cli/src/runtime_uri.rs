//! The compact `--runtime` URI: [`parse_runtime_uri`] and its inverse
//! [`runtime_to_uri`], over a shared key vocabulary. The two agree on the
//! runtime a URI names, not on its spelling — the renderer writes keys the
//! catalog leaves defaulted, so a rendered URI may be longer than the one that
//! produced it. `runtimes list --format uri` covers the round trip. The
//! runtime-side counterpart to [`crate::model_uri`]: `runtimes pull` takes one
//! self-contained argument — a JSON `Runtime` object or the URI below.
//!
//! # Grammar
//!
//! ```text
//! uri    ::= scheme "://" body            ; split on the FIRST "://"
//! scheme ::= "llamacpp-cli-stock-tools" | "mlx-macos-pipette"
//!          | "docker-vllm" | "docker-sglang"
//!          | "uv-vllm" | "uv-sglang" | "uv-openvino"
//! body   ::= "" | pair ("&" pair)*        ; keys unordered, each at most once
//! pair   ::= key "=" value                ; first "=" splits; value may contain "="
//! ```
//! `&` separates pairs, so no value may contain `&`; a URL value additionally may
//! not contain `?` (a query string can't survive the split — use the JSON form).
//!
//! # Keys → variant (parsing is deterministic on key *presence*, not order)
//!
//! | scheme                        | keys (required; *optional*)                                |
//! |-------------------------------|------------------------------------------------------------|
//! | `llamacpp-cli-stock-tools`    | `version`(+*`repo`*) **xor** `url`; `flavor`               |
//! | `mlx-macos-pipette`           | `version`; *`flavor`* (default `macos-arm64`)               |
//! | `docker-vllm` / `docker-sglang` | `image`, `tag`; *`flavor`* (default `nvidia_gpu`)        |
//! | `uv-vllm` / `uv-sglang`       | `server`, `build`, `python`                                |
//! | `uv-openvino`                 | `version` (the device is a per-cell runtime flag)          |
//!
//! # Not representable (both directions error)
//!
//! - `apple-foundation` and the on-device app runtimes (`llamacpp-apk-pipette`,
//!   `llamacpp-ios-pipette`, `mlx-ios-pipette`) →
//!   [`RuntimeUriError::NotRepresentable`]; address them via a JSON `--runtime`
//!   object instead.
//! - a URL carrying a query string → [`RuntimeUriError::QueryUrl`].
//! - `runtime://sha256=<hex>`, which *refers to* a descriptor rather than
//!   describing one and so needs the local store to resolve; it is handled a
//!   layer up, in [`crate::artifact_ref`].
//! - a uv/MLX runtime whose `requirements_text` is not the bundled catalog's for
//!   its scheme keys, and `RelativePreinstalled` →
//!   [`RuntimeUriError::NotRepresentable`]; pass a JSON `--runtime` object. The
//!   uv source contributes no keys of its own, so a URI can only name a preset.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use pipette_plan_types::{
    default_repository_url, DockerSglang, DockerVllm, LlamaCppFlavor, LlamacppCliStockTools,
    LlamacppCliStockToolsSource, MlxMacosPipette, MlxMacosPipetteFlavor, NonEmptyString,
    RemoteArchiveUrl, RepositoryUrl, Runtime, SglangFlavor, SourceRepository, UvBuild,
    UvPythonVersion, UvRuntimeSource, UvServerVersion, UvSglang, UvVllm, VllmFlavor,
};

// Key names, shared by the parser and [`runtime_to_uri`] so the two directions
// reference one spelling each. Schemes get the same guarantee, plus
// exhaustiveness, from the [`Scheme`] enum below.
const KEY_REPO: &str = "repo";
const KEY_VERSION: &str = "version";
const KEY_URL: &str = "url";
const KEY_FLAVOR: &str = "flavor";
const KEY_IMAGE: &str = "image";
const KEY_TAG: &str = "tag";
const KEY_SERVER: &str = "server";
const KEY_BUILD: &str = "build";
const KEY_PYTHON: &str = "python";

/// The importable runtime kinds, one per URI scheme. Both parsing and
/// [`runtime_to_uri`] route through this; the exhaustive match means neither
/// direction can silently drop a kind.
///
/// Variants are named for the plan [`Runtime`] kinds they select, so the
/// kebab-case strum spelling *is* the runtime's type word with `-` for `_`
/// (`UvOpenvino` → `uv-openvino`, matching `uv_openvino`). That is the whole
/// naming rule, and `a_schemes_prefix_is_its_runtime_type` enforces it — a
/// shorthand here would be a second name for a type that already has one.
///
/// Iterable so `runtimes pull --help` can be held to documenting every scheme:
/// a kind added here and left out of the help is a kind nobody can discover.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter, strum::EnumString, strum::IntoStaticStr,
)]
#[strum(serialize_all = "kebab-case")]
pub(crate) enum Scheme {
    LlamacppCliStockTools,
    MlxMacosPipette,
    DockerVllm,
    DockerSglang,
    UvVllm,
    UvSglang,
    UvOpenvino,
}

impl Scheme {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }

    /// The scheme for `s`. `apple-foundation` and the on-device app runtimes name
    /// real but non-importable runtimes (OS-bundled or addressed via their app
    /// transports), so they get a distinct [`RuntimeUriError::NotRepresentable`]
    /// rather than "unknown".
    fn parse(s: &str) -> Result<Self, RuntimeUriError> {
        Scheme::from_str(s).map_err(|_| match s {
            // Same naming rule as the importable schemes, so a plan author who
            // knows the runtime type gets told it is not addressable here
            // rather than that the word is unknown.
            "apple-foundation"
            | "llamacpp-apk-pipette"
            | "llamacpp-ios-pipette"
            | "mlx-ios-pipette" => RuntimeUriError::NotRepresentable(s.to_owned()),
            // Resolved against the local store by `crate::artifact_ref`, which
            // intercepts it before this parser ever sees it.
            "runtime" => RuntimeUriError::NotRepresentable(
                "a `runtime://sha256=` digest here; it resolves against the local store".to_owned(),
            ),
            _ => RuntimeUriError::UnknownScheme(s.to_owned()),
        })
    }
}

/// Everything that can go wrong turning a compact runtime URI into a typed
/// [`Runtime`]. Converts into `anyhow` at the CLI boundary via `?`.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeUriError {
    #[error("runtime URI must be `<scheme>://<key>=<value>[&<key>=<value>...]`")]
    MissingSchemeSeparator,

    #[error(
        "unknown runtime URI scheme `{0}` (expected `llamacpp-cli-stock-tools`, \
         `mlx-macos-pipette`, `docker-vllm`, `docker-sglang`, `uv-vllm`, `uv-sglang`, \
         or `uv-openvino`; a scheme is the runtime type with `-` for `_`)"
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

/// The parsed `key=value` pairs of a URI body. Identical tokenizer to
/// [`crate::model_uri`]: `&` is the pair separator; the first `=` splits key
/// from value, so a value keeps any later `=`. URL values are barred from
/// carrying a query string, so a raw `&` can never be part of a value.
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

    /// Consume the tokenizer, rejecting any key the scheme didn't claim.
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

// ---------------------------------------------------------------------------
// Leaf-value constructors (newtype validation → InvalidValue)
// ---------------------------------------------------------------------------

fn non_empty(key: &'static str, v: &str) -> Result<NonEmptyString, RuntimeUriError> {
    NonEmptyString::try_new(v).map_err(|e| RuntimeUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

/// Remote archive coordinate: scheme-less host/path (or `http(s)://` paste,
/// stripped). `file://` and other schemes are rejected.
fn remote_archive_url(key: &'static str, v: &str) -> Result<RemoteArchiveUrl, RuntimeUriError> {
    // A query string can't survive the `&` split — signed / query URLs go
    // through the JSON `--runtime` form.
    if v.contains('?') {
        return Err(RuntimeUriError::QueryUrl { key });
    }
    RemoteArchiveUrl::try_new(v.to_owned()).map_err(|e| RuntimeUriError::InvalidValue {
        key,
        message: e.to_string(),
    })
}

/// Resolve the `server`/`build`/`python` triple against the bundled torch-oai
/// catalog into a plain [`UvRuntimeSource::PipRequirementsText`].
///
/// The slug is composed from the URI's own keys rather than accepted as one, so
/// a URI cannot name a row whose body it then misreports: the three keys that
/// select the row are the same three the rendered runtime carries.
fn uv_catalog_source_from_uri(slug: &str) -> Result<UvRuntimeSource, RuntimeUriError> {
    let uv_slug = pipette_torch_oai::slug::UvSlug::try_new(slug).map_err(|e| {
        RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: e.to_string(),
        }
    })?;
    let entry = pipette_torch_oai::catalog::lookup(&uv_slug)
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
        other => Err(RuntimeUriError::InvalidValue {
            key: KEY_SERVER,
            message: format!("catalog row produced unexpected runtime {other}"),
        }),
    }
}

/// MLX bundled catalog (compile-time). Parsed only to fill
/// `PipRequirementsText.requirements_text` for URIs so the declared form is
/// self-contained for the shared store fetcher.
const MLX_CATALOG_TOML: &str = include_str!("../../pipette-mlx/bundled-catalog/catalog.toml");

#[derive(serde::Deserialize)]
struct MlxCatalogFile {
    mlx: Vec<MlxCatalogRow>,
}

#[derive(serde::Deserialize)]
struct MlxCatalogRow {
    version: String,
    requirements: String,
}

/// Resolve `version` against the bundled MLX catalog into a plain
/// [`UvRuntimeSource::PipRequirementsText`].
///
/// The version is the whole selector: a URI names a preset, and what comes back
/// is an ordinary uv-defined runtime carrying that preset's body. There is no
/// way to spell a name that disagrees with the body it resolved to.
fn mlx_catalog_source_from_uri(version: &str) -> Result<UvRuntimeSource, RuntimeUriError> {
    let file: MlxCatalogFile =
        toml::from_str(MLX_CATALOG_TOML).map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_VERSION,
            message: format!("bundled mlx catalog is unreadable: {e}"),
        })?;
    let row = file
        .mlx
        .into_iter()
        .find(|row| row.version == version)
        .ok_or_else(|| {
            RuntimeUriError::NotRepresentable(format!(
                "mlx version `{version}` is not in the bundled catalog; pass a JSON \
                 `--runtime` with a full pip_requirements_text source"
            ))
        })?;
    let requirements_text =
        NonEmptyString::try_new(row.requirements).map_err(|e| RuntimeUriError::InvalidValue {
            key: KEY_VERSION,
            message: e.to_string(),
        })?;
    Ok(UvRuntimeSource::PipRequirementsText {
        contents: requirements_text,
        install_flags: None,
    })
}

// The closed flavor enums have serde renames but no `Display`, so render/parse
// go through explicit maps here — this keeps a wrong flavor a precise
// `InvalidValue` and the rendered spelling pinned.

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

// ---------------------------------------------------------------------------
// Parse: URI → Runtime
// ---------------------------------------------------------------------------

fn parse_llama_cpp(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let repo = p.take(KEY_REPO);
    let version = p.take(KEY_VERSION);
    let url = p.take(KEY_URL);
    let flavor = LlamaCppFlavor::parse(p.require(KEY_FLAVOR)?);
    let source = match (version, url) {
        // Git build: `version` (+ optional `repo`, default upstream).
        (Some(version), None) => LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
            repository_url: repo.map_or_else(default_repository_url, RepositoryUrl::new),
            repository_version: non_empty(KEY_VERSION, version)?,
        }),
        // Prebuilt archive: a single URL. `repo` has no meaning here.
        (None, Some(url)) => {
            if repo.is_some() {
                return Err(RuntimeUriError::MutuallyExclusive {
                    scheme: p.scheme.as_str(),
                    expected: "`version` (optionally with `repo`) or `url`",
                });
            }
            LlamacppCliStockToolsSource::RemoteArchive {
                url: remote_archive_url(KEY_URL, url)?,
            }
        }
        _ => {
            return Err(RuntimeUriError::MutuallyExclusive {
                scheme: p.scheme.as_str(),
                expected: "`version` (optionally with `repo`) or `url`",
            })
        }
    };
    p.finish()?;
    Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
        source,
        flavor,
    }))
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

/// The three uv fields shared by vllm and sglang, plus a catalog source.
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

/// Parse a compact runtime URI into a typed [`Runtime`]. Only importable desktop
/// runtimes have a scheme; on-device and Apple-Foundation runtimes are rejected
/// (see [`Scheme::parse`]).
pub fn parse_runtime_uri(input: &str) -> Result<Runtime, RuntimeUriError> {
    let (scheme, body) = input
        .split_once("://")
        .ok_or(RuntimeUriError::MissingSchemeSeparator)?;
    let scheme = Scheme::parse(scheme)?;
    let pairs = Pairs::parse(scheme, body)?;
    match scheme {
        Scheme::LlamacppCliStockTools => parse_llama_cpp(pairs),
        Scheme::MlxMacosPipette => parse_mlx(pairs),
        Scheme::DockerVllm => parse_docker_vllm(pairs),
        Scheme::DockerSglang => parse_docker_sglang(pairs),
        Scheme::UvVllm => parse_uv_vllm(pairs),
        Scheme::UvSglang => parse_uv_sglang(pairs),
        Scheme::UvOpenvino => parse_openvino(pairs),
    }
}

/// `uv-openvino://version=<v>&device=<cpu|gpu|npu>`.
///
/// No `device` key: one wheel serves CPU, GPU and NPU, so a runtime ref names
/// an installable artifact and the device is a flag on the cell that uses it.
/// The version resolves against the bundled catalog, which owns the pinned
/// requirements — same contract as `mlx-macos-pipette://`.
fn parse_openvino(mut p: Pairs) -> Result<Runtime, RuntimeUriError> {
    let version = non_empty(KEY_VERSION, p.require(KEY_VERSION)?)?;
    p.finish()?;
    pipette_openvino::catalog::declared_from_catalog(version.as_ref()).map_err(|e| {
        RuntimeUriError::NotRepresentable(format!(
            "openvino version `{version}` is not in the bundled catalog ({e}); pass a \
             JSON `--runtime` with a full pip_requirements_text source"
        ))
    })
}

// ---------------------------------------------------------------------------
// Render: Runtime → URI (inverse of the parse, arm-for-arm)
// ---------------------------------------------------------------------------

/// Accumulates a rendered URI's `key=value` pairs in push order, then joins them.
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

    /// Push a `key=value` pair, rejecting a value that contains the `&` pair
    /// separator — it would fracture the URI on re-parse. (`=` is fine: the
    /// parser splits on the first `=` only; `?` matters only for URL values,
    /// guarded in [`Body::url`].) This keeps render → parse a total round-trip
    /// even for free-form values like a `Custom` flavor or a `req-path`.
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

    /// A URL value that would fracture on re-parse — `&` (the separator) or `?`
    /// (a query string) — can't be represented; reject it symmetrically with
    /// the parser's URL checks.
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

/// Check that `source` is exactly what this scheme's keys resolve to.
///
/// A uv source contributes no URI keys of its own — the scheme's keys pick a
/// catalog preset and the preset supplies the body. So a runtime is
/// representable only when its body *is* that preset's. Rendering one that
/// differs would produce a URI that looks right and re-parses into a different
/// environment; such a runtime has to travel as JSON instead.
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
        UvRuntimeSource::RelativePreinstalled { .. }
        | UvRuntimeSource::AbsolutePreinstalled { .. } => Err(RuntimeUriError::NotRepresentable(
            "a preinstalled (local) uv/MLX runtime".to_owned(),
        )),
    }
}

/// Render a [`Runtime`] as its compact URI — the inverse of [`parse_runtime_uri`],
/// mirroring it arm-for-arm with a fixed key order for stable output.
/// Non-representable runtimes error: on-device / Apple-Foundation →
/// [`RuntimeUriError::NotRepresentable`]; a URL a query string or `&` would
/// fracture → [`RuntimeUriError::QueryUrl`].
pub fn runtime_to_uri(runtime: &Runtime) -> Result<String, RuntimeUriError> {
    match runtime {
        Runtime::LlamacppCliStockTools(rt) => {
            let mut body = Body::new(Scheme::LlamacppCliStockTools);
            match &rt.source {
                LlamacppCliStockToolsSource::GithubRelease(repo) => {
                    // `repo` is emitted only when it differs from the upstream
                    // default, which the parser fills in when absent.
                    if repo.repository_url != default_repository_url() {
                        body.push(KEY_REPO, repo.repository_url.as_ref())?;
                    }
                    body.push(KEY_VERSION, repo.repository_version.as_ref())?;
                }
                LlamacppCliStockToolsSource::RemoteArchive { url } => body.url(KEY_URL, url)?,
                // Effective (installed) form only — not a fetch coordinate.
                LlamacppCliStockToolsSource::RelativeDir { .. }
                | LlamacppCliStockToolsSource::AbsoluteDir { .. } => {
                    return Err(RuntimeUriError::NotRepresentable(
                        "a local (installed) llama.cpp runtime".to_owned(),
                    ));
                }
            }
            body.push(KEY_FLAVOR, &rt.flavor.to_string())?;
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
                match pipette_openvino::catalog::declared_from_catalog(rt.server_version.as_ref()) {
                    Ok(Runtime::UvOpenvino(catalog_rt)) => Ok(catalog_rt.source),
                    _ => Err(RuntimeUriError::NotRepresentable(
                        "an openvino runtime outside the bundled catalog".to_owned(),
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
        // On-device app runtimes + Apple Foundation aren't addressable via the
        // desktop CLI — the same reject-list `refs.rs` enforces.
        Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxIosPipette(_)
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
    if trimmed.starts_with('{') {
        Ok(serde_json::from_str::<Runtime>(trimmed)?)
    } else {
        Ok(parse_runtime_uri(trimmed)?)
    }
}

/// A [`Runtime`] wrapper that serde-round-trips through the compact URI. It reads
/// the URI string *or* the structured object, and writes the URI when the runtime
/// is representable (else the structured form) — so a serde field (config/plan
/// JSON or TOML) can use either.
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

            // A map is the structured `Runtime` — defer to its derived Deserialize.
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
        // Emit the compact URI when the runtime is representable, else the
        // structured form — the mirror of the URI-or-struct Deserialize.
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
        assert_eq!(repo.repository_url, default_repository_url());
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
        // An unknown flavor round-trips through `Custom`.
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

    #[test]
    fn llama_cpp_version_and_url_are_exclusive() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri(
                "llamacpp-cli-stock-tools://version=b1&url=https://x/a.tar.gz&flavor=macos-arm64"
            ),
            Err(RuntimeUriError::MutuallyExclusive { .. })
        ));
        // `repo` alongside `url` is also rejected.
        assert!(matches!(
            parse_runtime_uri(
                "llamacpp-cli-stock-tools://repo=github.com/a/b&url=https://x/a.tgz&flavor=macos-arm64"
            ),
            Err(RuntimeUriError::MutuallyExclusive { .. })
        ));
        Ok(())
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
        // Real frozen catalog body — not a URI stub.
        assert!(
            contents.as_ref().contains("mlx-lm==0.31.3"),
            "expected catalog requirements, got {contents}"
        );
        Ok(())
    }

    #[test]
    fn mlx_unknown_version_is_not_representable() -> anyhow::Result<()> {
        assert!(matches!(
            parse_runtime_uri("mlx-macos-pipette://version=0.0.0-not-in-catalog"),
            Err(RuntimeUriError::NotRepresentable(_))
        ));
        Ok(())
    }

    /// A URI names a preset and nothing else, so a runtime whose requirements
    /// are not that preset's has no URI form. Rendering one anyway would emit
    /// `mlx-macos-pipette://version=0.31.3`, which re-parses into the catalog's environment —
    /// a different runtime wearing the same spelling.
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
            other => anyhow::bail!("expected uv runtime, got {other:?}"),
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

    // `llama-cpp` with neither version nor url is a MutuallyExclusive (covered
    // above), so this table only covers keys required unconditionally.
    #[rstest]
    #[case("docker-vllm://image=vllm", "tag")]
    #[case("uv-vllm://server=0.21.0&build=cu121", "python")]
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
    #[case("uv-vllm://server=0.21.0&build=notabuild&python=3.12", "build")]
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
        // A free-form `Custom` flavor with `&` would fracture the URI on
        // re-parse, so render refuses it rather than emit an unparseable string.
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b1")?,
            }),
            flavor: LlamaCppFlavor::parse("weird&flavor"),
        });
        assert!(matches!(
            runtime_to_uri(&runtime),
            Err(RuntimeUriError::UnrepresentableValue { key: "flavor", .. })
        ));
        Ok(())
    }

    // Canonical URIs (keys already in render order) — parse → render → parse is
    // an identity, and the rendered string equals the input.
    #[rstest]
    #[case("llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64")]
    #[case("llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=macos-x64")]
    #[case("llamacpp-cli-stock-tools://url=ex.com/llama-b1.tar.gz&flavor=macos-arm64")]
    #[case("mlx-macos-pipette://version=0.31.3&flavor=macos-arm64")]
    #[case("docker-vllm://image=vllm/vllm-openai&tag=v0.10.0&flavor=nvidia_gpu")]
    #[case("docker-sglang://image=lmsysorg/sglang&tag=v0.4.0&flavor=amd_gpu")]
    #[case("uv-vllm://server=0.21.0&build=cu121&python=3.12")]
    #[case("uv-sglang://server=0.5.12.post1&build=cu121&python=3.12")]
    #[case("uv-openvino://version=2026.2.1")]
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
        // Apple Foundation has no URI form, so serde falls back to the object.
        let json =
            serde_json::to_string(&RuntimeUri(Runtime::AppleFoundation(Default::default())))?;
        assert!(json.starts_with('{'));
        let RuntimeUri(back) = serde_json::from_str(&json)?;
        assert_eq!(back, Runtime::AppleFoundation(Default::default()));
        Ok(())
    }
}
