//! The Runtime family: [`Runtime`] and its per-kind variant structs,
//! runtime source coordinates, and the closed/open build-flavor enums.
//! Re-exported flat from `lib.rs`, so consumers reference these as
//! `pipette_plan_types::Runtime` etc. without seeing the submodule.

use nutype::nutype;
use serde::{Deserialize, Serialize};

use crate::{
    AbsolutePath, NonEmptyString, RelativePath, RemoteArchiveUrl, UvBuild, UvPythonVersion,
    UvServerVersion,
};

/// Runtime declaration on a variant.
///
/// vLLM-/sglang-hosted runtimes are split per-server (both for
/// `docker` and `uv` deployment paths) so server-specific flavor sets
/// stay precise: vllm may grow `apple_gpu`, sglang may grow `tpu`,
/// and neither contaminates the other.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Runtime {
    /// llama.cpp driven through the stock upstream CLI tools
    /// (`llama-bench`/`llama-server`) via the desktop `pipette` client.
    LlamacppCliStockTools(LlamacppCliStockTools),
    /// llama.cpp running in-process inside the Android pipette app —
    /// a distinct execution surface from the pushed CLI.
    LlamacppApkPipette(LlamacppApkPipette),
    /// llama.cpp running in-process inside the iOS pipette app.
    LlamacppIosPipette(LlamacppIosPipette),
    MlxMacosPipette(MlxMacosPipette),
    /// MLX running in-process inside the iOS pipette app (mlx-swift) — the
    /// on-device counterpart to the desktop `MlxMacosPipette` (Python/uv) runtime.
    MlxIosPipette(MlxIosPipette),
    DockerVllm(DockerVllm),
    DockerSglang(DockerSglang),
    UvVllm(UvVllm),
    UvSglang(UvSglang),
    /// OpenVINO IR served by `openvino-genai` in a uv venv. The only runtime
    /// with an Intel NPU path, and the only venv-backed one that runs on
    /// Windows — which is where Intel NPU hardware lives.
    UvOpenvino(UvOpenvino),
    /// Apple Foundation Models runtime (iOS) — loads only
    /// `Model::AppleFoundationText`. Wire tag `apple_foundation` matches
    /// the submission `runtime_name`; `runtime_version` is the OS version,
    /// resolved on-device.
    AppleFoundation(AppleFoundation),
}

/// The Apple Foundation runtime carries no build of ours — the weights and the engine
/// ship with the OS — so its only identity is the app it runs inside, and the one thing
/// about that app which changes what a number means is the readiness gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct AppleFoundation {
    /// As [`LlamacppIosPipette::private_thermal`]. AFM cells wait on the same gate as the
    /// other two iOS runtimes — it belongs to the app, not to the engine — so a plan can
    /// require the gated build here for the same reason.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub private_thermal: bool,
}

/// Normalize any repository coordinate a user might paste to the one
/// host-qualified, scheme-less form `<host>/<org>/<repo>`. Host-agnostic (it does
/// not assume GitHub) and infallible. All of these reduce to
/// `github.com/ggml-org/llama.cpp`:
///
/// - `https://github.com/ggml-org/llama.cpp` (and `http://`, trailing slash)
/// - `https://github.com/ggml-org/llama.cpp.git` (the clone-URL `.git` suffix)
/// - `git@github.com:ggml-org/llama.cpp.git` (SSH scp-like syntax)
/// - `ssh://git@github.com/ggml-org/llama.cpp` (SSH URL syntax)
///
/// This is the single source of truth for repo identity, so a coordinate copied
/// from GitHub's "Code" dropdown in any form stores and compares identically.
fn strip_url_scheme(raw: String) -> String {
    let mut s = raw.trim();

    // Drop a leading URL scheme.
    if let Some(rest) = ["https://", "http://", "ssh://", "git://"]
        .into_iter()
        .find_map(|scheme| s.strip_prefix(scheme))
    {
        s = rest;
    }

    // Drop userinfo (`git@…`, `user:token@…`): a `@` that precedes the host,
    // i.e. before the first path `/`.
    if let Some(at) = s.find('@') {
        if !s[..at].contains('/') {
            s = &s[at + 1..];
        }
    }

    // Normalize the SSH scp-like `host:org/repo` separator to `host/org/repo` —
    // the first `:` when no `/` precedes it (a scheme-stripped URL path is left
    // untouched).
    let colon_normalized = match s.find(':') {
        Some(colon) if !s[..colon].contains('/') => format!("{}/{}", &s[..colon], &s[colon + 1..]),
        _ => s.to_owned(),
    };

    // Trim a trailing slash, then the `.git` clone suffix, then any slash it
    // exposed.
    let trimmed = colon_normalized.trim_end_matches('/');
    trimmed
        .strip_suffix(".git")
        .unwrap_or(trimmed)
        .trim_end_matches('/')
        .to_owned()
}

/// The canonical upstream llama.cpp repo, used when a plan/manifest omits
/// `repository_url`. Public so the CLI ref-reverser resolves the same default
/// `cli_ref` drops, rather than duplicating the URL string.
pub fn default_repository_url() -> RepositoryUrl {
    RepositoryUrl::new("github.com/ggml-org/llama.cpp")
}

/// A source repository, exposed host-qualified and scheme-less as
/// `<host>/<org>/<repo>` (`github.com/ggml-org/llama.cpp`,
/// `gitlab.com/…`, …) so the host names the provider and provenance is a full
/// coordinate. Any pasted form — HTTPS/SSH URL, `.git` clone suffix, scp-like
/// `git@host:org/repo` — is normalized down to this form (see
/// `strip_url_scheme`); the sanitizer never fails, so construction is
/// infallible.
#[nutype(
    sanitize(with = strip_url_scheme),
    derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, AsRef, Display)
)]
pub struct RepositoryUrl(String);

/// Where the llama.cpp CLI runtime's *prebuilt* binaries come from — there is no
/// git build. `GitHubRelease` is a GitHub release identified by its source repo +
/// release tag; the fetcher resolves it to the release-asset URL
/// `https://<repo>/releases/download/<tag>/<asset>`. `RemoteArchive` is a
/// prebuilt archive download coordinate stored **without** a URL scheme
/// (`host/path/…`); fetchers download via `https://`. `file://` and other
/// schemes are rejected. `RelativeDir` is the **effective** (installed) form only:
/// a directory under the store entry that holds the unpacked tree. Plans and
/// URIs use only `GitHubRelease` / `RemoteArchive`.
///
/// Tagged `source` (same dialect as model sources): `github_release` (repo +
/// version; `version` aliases `repository_version`, `repository_url` defaults to
/// upstream llama.cpp), `remote_archive` (`url`), `relative_dir` / `absolute_dir`
/// (`dir`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum LlamacppCliStockToolsSource {
    GithubRelease(SourceRepository),
    /// Prebuilt archive at a remote host/path. See [`RemoteArchiveUrl`].
    RemoteArchive {
        url: RemoteArchiveUrl,
    },
    /// Portable install layout under the store entry (e.g. `blobs`).
    RelativeDir {
        dir: RelativePath,
    },
    /// Absolute install root on the host after bind.
    AbsoluteDir {
        dir: AbsolutePath,
    },
}

impl LlamacppCliStockToolsSource {
    /// The ref-ish token used in the `--runtime` display form and identity
    /// markers: the release tag for a `GitHubRelease`, the remote archive
    /// coordinate, or the install directory for `RelativeDir`.
    pub fn reference(&self) -> &str {
        match self {
            LlamacppCliStockToolsSource::GithubRelease(repo) => repo.repository_version.as_ref(),
            LlamacppCliStockToolsSource::RemoteArchive { url } => url.as_ref(),
            LlamacppCliStockToolsSource::RelativeDir { dir } => dir.as_ref(),
            LlamacppCliStockToolsSource::AbsoluteDir { dir } => dir.as_ref(),
        }
    }

    /// The submission `runtime_name`: the host-qualified repo URL for a
    /// `GitHubRelease`, the remote archive coordinate, or the local dir.
    pub fn runtime_name(&self) -> String {
        match self {
            LlamacppCliStockToolsSource::GithubRelease(repo) => repo.repository_url.to_string(),
            LlamacppCliStockToolsSource::RemoteArchive { url } => url.to_string(),
            LlamacppCliStockToolsSource::RelativeDir { dir } => dir.to_string(),
            LlamacppCliStockToolsSource::AbsoluteDir { dir } => dir.to_string(),
        }
    }

    /// A coarse, stable token for the build's origin, used as the leading
    /// segment of the eval-checkpoint marker: the bare `<org>/<repo>` slug for a
    /// `GitHubRelease`, `remote-archive` for a remote prebuilt archive, `local`
    /// for an installed tree.
    pub fn origin_slug(&self) -> &str {
        match self {
            LlamacppCliStockToolsSource::GithubRelease(repo) => repo.repository_url.org_repo(),
            LlamacppCliStockToolsSource::RemoteArchive { .. } => "remote-archive",
            LlamacppCliStockToolsSource::RelativeDir { .. } => "local",
            LlamacppCliStockToolsSource::AbsoluteDir { .. } => "local",
        }
    }
}

impl RepositoryUrl {
    /// The `<org>/<repo>` portion, dropping the leading `<host>/` segment.
    pub fn org_repo(&self) -> &str {
        let s: &str = self.as_ref();
        s.split_once('/').map(|(_, rest)| rest).unwrap_or(s)
    }
}

/// A GitHub source checkout — the repo URL plus the ref built from it. The two
/// travel together: `repository_version` (a release tag like `b5000` or a
/// commit) is only meaningful relative to `repository_url`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct SourceRepository {
    #[serde(default = "default_repository_url", alias = "source_repo")]
    pub repository_url: RepositoryUrl,
    #[serde(alias = "version")]
    pub repository_version: NonEmptyString,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct LlamacppCliStockTools {
    #[serde(flatten)]
    pub source: LlamacppCliStockToolsSource,
    pub flavor: LlamaCppFlavor,
}

/// The Android pipette app's in-process llama.cpp. Carries the source
/// checkout and its (single, today) build flavor. There's no local HTTP
/// server, so the CLI's `http_timeout_seconds` knob doesn't apply, and it
/// has no raw `flags` field. Always a git checkout — no archive source.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct LlamacppApkPipette {
    #[serde(flatten)]
    pub source: SourceRepository,
    pub flavor: LlamacppApkPipetteFlavor,
}

/// Build targets for the llama.cpp APK. A closed set — one member today
/// (the app ships a single ABI) — kept as its own enum so the APK runtime
/// carries a typed flavor rather than a bare marker, and so new ABIs are an
/// additive, compile-checked change. `Display`/`FromStr` (strum) and serde
/// all render the same kebab spelling.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::Display,
    strum::EnumString,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum LlamacppApkPipetteFlavor {
    AndroidArm64V8,
}

/// The iOS pipette app's in-process llama.cpp — `LlamacppApkPipette`'s
/// counterpart on iOS (same shape, same reasons).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct LlamacppIosPipette {
    #[serde(flatten)]
    pub source: SourceRepository,
    pub flavor: LlamacppIosPipetteFlavor,
    /// Whether the build reads the SoC die temperature through the private IOHID sensors
    /// (`PIPETTE_PRIVATE_THERMAL`). Part of the runtime's identity, not a flag: it decides
    /// what the readiness gate can see. Without it the gate has only
    /// `ProcessInfo.thermalState`, which stays `.nominal` well into GPU throttling, so two
    /// runs of one cell are not comparable across builds — the ungated one measures a device
    /// that was allowed to start hot.
    ///
    /// Defaults to `false`, which is what a stock build is, so a plan that says nothing keeps
    /// meaning the same thing. A plan that needs the gated build says so and a stock device
    /// refuses the cell rather than quietly producing weaker numbers.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub private_thermal: bool,
}

/// Build targets for the llama.cpp iOS app. One member today (a single
/// ABI); its own enum for the same reasons as `LlamacppApkPipetteFlavor`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::Display,
    strum::EnumString,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum LlamacppIosPipetteFlavor {
    IosArm64,
}

/// The iOS pipette app's in-process MLX (Swift) runtime — the mlx-swift
/// counterpart to `LlamacppIosPipette`. Unlike the desktop `MlxMacosPipette` (Python/uv),
/// its version is the set of pinned Swift packages, each a `SourceRepository`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct MlxIosPipette {
    pub packages: MlxSwiftStack,
    pub flavor: MlxIosPipetteFlavor,
    /// As [`LlamacppIosPipette::private_thermal`] — the gate is the app's, not the
    /// engine's, so both iOS runtimes carry it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub private_thermal: bool,
}

/// The pinned Swift-package stack the iOS MLX build was compiled against
/// (from the app's `Package.resolved`). Each is a repo + ref.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct MlxSwiftStack {
    pub mlx_swift: SourceRepository,
    pub mlx_swift_lm: SourceRepository,
    pub swift_transformers: SourceRepository,
}

/// Build targets for the MLX iOS app. One member today (a single ABI); its
/// own enum for the same reasons as `LlamacppIosPipetteFlavor`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::Display,
    strum::EnumString,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MlxIosPipetteFlavor {
    IosArm64,
}

/// How a UV (or MLX-uv) environment is obtained — the UV counterpart to
/// [`LlamacppCliStockToolsSource`] for llama.cpp.
///
/// Plans and URIs use [`Self::PipRequirementsText`]. [`Self::RelativePreinstalled`] is the
/// **effective** form only (after ensure/pull): a venv directory under the store
/// entry; tools are found by backend convention under `dir` (`bin/python`,
/// `bin/vllm`, …).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum UvRuntimeSource {
    /// An installable environment, defined by the exact `requirements.txt` body
    /// that builds it. The body **is** the identity: the storage key is a digest
    /// over it (`pipette_artifacts::RuntimeStorageKey`), so two declarations
    /// install to one entry precisely when they install the same thing.
    ///
    /// The bundled catalogs (`pipette_mlx::catalog`, `pipette_torch_oai::catalog`)
    /// are authoring shortcuts that resolve *into* this variant — a preset name
    /// picks a body, then plays no further part. Nothing downstream can tell a
    /// resolved preset from a hand-authored one, which is the point: a name that
    /// survived into the value would be a claim about content that nothing
    /// verifies, and the store would alias two different environments onto one
    /// entry whenever it was wrong.
    PipRequirementsText {
        contents: NonEmptyString,
        /// `uv pip install`/`compile` flags. Not identity: they alter how the
        /// body is installed, not which body it is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        install_flags: Option<Vec<String>>,
    },
    /// Venv already present under the store entry (entry-relative). Do not install.
    #[serde(alias = "preinstalled")]
    RelativePreinstalled { dir: RelativePath },
    /// Absolute venv root on the host after bind.
    #[serde(alias = "abs_preinstalled")]
    AbsolutePreinstalled { dir: AbsolutePath },
}

impl UvRuntimeSource {
    /// Install-time `uv` flags; empty for [`Self::RelativePreinstalled`].
    pub fn install_flags(&self) -> &[String] {
        match self {
            Self::PipRequirementsText { install_flags, .. } => {
                install_flags.as_deref().unwrap_or(&[])
            }
            Self::RelativePreinstalled { .. } | Self::AbsolutePreinstalled { .. } => &[],
        }
    }

    /// Requirements body for the installable arm; `None` for preinstalled.
    pub fn requirements_text(&self) -> Option<&str> {
        match self {
            Self::PipRequirementsText { contents, .. } => Some(contents.as_ref()),
            Self::RelativePreinstalled { .. } | Self::AbsolutePreinstalled { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct MlxMacosPipette {
    pub version: NonEmptyString,
    pub flavor: MlxMacosPipetteFlavor,
    pub source: UvRuntimeSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct DockerVllm {
    pub image_name: NonEmptyString,
    pub image_tag: NonEmptyString,
    pub flavor: VllmFlavor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct DockerSglang {
    pub image_name: NonEmptyString,
    pub image_tag: NonEmptyString,
    pub flavor: SglangFlavor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct UvVllm {
    pub server_version: UvServerVersion,
    pub build: UvBuild,
    pub python_version: UvPythonVersion,
    pub source: UvRuntimeSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct UvSglang {
    pub server_version: UvServerVersion,
    pub build: UvBuild,
    pub python_version: UvPythonVersion,
    pub source: UvRuntimeSource,
}

/// `openvino-genai` in a uv venv, pinned to one compute device.
///
/// `device` is a required field rather than a runtime flag on purpose: the
/// OpenVINO default is CPU, so an omitted device would silently measure the CPU
/// and file the row as whatever the author meant. It is read from the
/// **declared** runtime at dispatch, never from the bound one, so the cell's
/// choice always wins over whatever device first populated the store entry.
///
/// Unlike [`UvVllm`]'s `build`, `device` is not a build target — one wheel
/// serves all three — so it is deliberately left out of the storage key and the
/// three devices share a single installed venv.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UvOpenvino {
    /// The `openvino-genai` version, which must match the `openvino` version
    /// that produced the IR.
    pub server_version: UvServerVersion,
    pub python_version: UvPythonVersion,
    pub source: UvRuntimeSource,
}

/// The OpenVINO compute device a cell runs on.
///
/// The three devices pipette targets are named; `Custom` is the escape hatch
/// for anything else OpenVINO accepts in that slot — a per-index GPU (`GPU.1`)
/// on a multi-GPU host, or a virtual device (`AUTO`, `HETERO:GPU,CPU`). Same
/// shape as [`LlamaCppFlavor`] and for the same reason: the set is curated, not
/// closed. `ov.Core().available_devices` reports more than pipette names, and a
/// closed enum could only refuse the rest.
///
/// Wire form is a single string, and [`Self::as_str`]/[`Self::parse`] are the
/// spelling — they live here rather than in the backend crate so the URI
/// grammar spells a plan-types value without depending on the runtime that
/// executes it. Named devices use their canonical lowercase spelling (`cpu`,
/// `gpu`, `npu`); `Custom(s)` emits `s` verbatim, because
/// OpenVINO's device names are case-sensitive and the author's spelling is the
/// one the plugin has to accept.
///
/// A `Custom` device is *not* treated as an NPU by the static-shape and
/// warm-up rules, even when its string mentions one: those rules exist for the
/// NPU pipeline specifically, and guessing at `HETERO:NPU,CPU` would be a
/// support-matrix belief of the kind `pipette-openvino`'s `models` module
/// argues against. Such a cell gets no static-shape properties and refuses at
/// compile if its prompt exceeds the default bound.
#[derive(Debug, Clone, PartialEq, Eq, Hash, strum::EnumIter)]
pub enum OpenvinoDevice {
    Cpu,
    Gpu,
    /// Intel AI Boost. Static-shape only — see `docs/openvino-ir.md`.
    Npu,
    /// A device OpenVINO accepts that pipette does not name.
    Custom(String),
}

impl OpenvinoDevice {
    /// Every device pipette names, for a listing or a validator. Excludes
    /// [`Self::Custom`], which is open by construction.
    pub fn known() -> impl Iterator<Item = Self> {
        use strum::IntoEnumIterator as _;
        Self::iter().filter(|device| !device.is_custom())
    }

    /// Whether this is a string outside the named set. [`Self::parse`] never
    /// fails, so this is how a caller tells "a device we target" from "a device
    /// the operator spelled".
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Wire form: the canonical name, or the custom string verbatim.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
            Self::Custom(device) => device,
        }
    }

    /// Parse a wire form. Matching is case-insensitive, but a `Custom` keeps
    /// the original spelling rather than the lowercased match key — OpenVINO
    /// resolves `GPU.1`, not `gpu.1`.
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "gpu" => Self::Gpu,
            "npu" => Self::Npu,
            _ => Self::Custom(s.to_owned()),
        }
    }
}

impl std::fmt::Display for OpenvinoDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for OpenvinoDevice {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for OpenvinoDevice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::parse(&s))
    }
}

impl UvVllm {
    pub fn runtime_version(&self) -> String {
        uv_runtime_version(&self.server_version, &self.build, &self.python_version)
    }
}

impl UvSglang {
    pub fn runtime_version(&self) -> String {
        uv_runtime_version(&self.server_version, &self.build, &self.python_version)
    }
}

impl UvOpenvino {
    /// `<version>.py<python>` — the other uv runtimes carry a build tag here,
    /// but one OpenVINO wheel serves every device, so there is no hardware
    /// target to name. Which device a cell used is a flag, not part of the
    /// runtime it used.
    pub fn runtime_version(&self) -> String {
        format!("{}.py{}", self.server_version, self.python_version)
    }
}

pub fn uv_runtime_version(
    server_version: &UvServerVersion,
    build: &UvBuild,
    python_version: &UvPythonVersion,
) -> String {
    format!("{server_version}+{build}.py{python_version}")
}

/// The GPU target a uv build tag (`cu121`, `rocm624`, `cpu`) names.
///
/// Lives beside the tag it reads because two crates classify it — the installer
/// to pick the in-venv torch probe, the client to pick the host GPU preflight —
/// and two spellings could send one runtime past the wrong check. Returns
/// [`VllmFlavor`]; sglang callers translate.
pub fn flavor_from_uv_build(build: &UvBuild) -> VllmFlavor {
    let build = build.as_ref();
    if build == "cpu" {
        VllmFlavor::Cpu
    } else if build.starts_with("rocm") {
        VllmFlavor::AmdGpu
    } else {
        VllmFlavor::NvidiaGpu
    }
}

impl Runtime {
    /// The iOS `headlessrun runtime=<token>` value for this runtime — the
    /// plan `Runtime` `type` tag (serde snake_case discriminant), so the
    /// launch arg names the exact variant the plan authored. Kept in sync
    /// with the serde tags by `headless_token_matches_plan_type_tag`. Only
    /// the on-device runtimes ever reach an iOS transport; the server-hosted
    /// kinds still carry their tag, but a plan pairing one with an iOS
    /// transport is a misconfiguration the device rejects.
    pub fn headless_token(&self) -> &'static str {
        match self {
            Runtime::LlamacppCliStockTools(_) => "llamacpp_cli_stock_tools",
            Runtime::LlamacppApkPipette(_) => "llamacpp_apk_pipette",
            Runtime::LlamacppIosPipette(_) => "llamacpp_ios_pipette",
            Runtime::MlxMacosPipette(_) => "mlx_macos_pipette",
            Runtime::MlxIosPipette(_) => "mlx_ios_pipette",
            Runtime::DockerVllm(_) => "docker_vllm",
            Runtime::DockerSglang(_) => "docker_sglang",
            Runtime::UvVllm(_) => "uv_vllm",
            Runtime::UvSglang(_) => "uv_sglang",
            Runtime::UvOpenvino(_) => "uv_openvino",
            Runtime::AppleFoundation(_) => "apple_foundation",
        }
    }

    /// The value passed as `--runtime <ref>` to the runtime binary — the
    /// binary's own addressing grammar: llamacpp `version:flavor`, mlx a bare
    /// version, `docker://…`, `uv://…`. Narrower than [`Display`](std::fmt::Display), which also
    /// carries the source repo for identity; the binary resolves an installed
    /// runtime by version+flavor, so the repo would break that lookup.
    pub fn cli_ref(&self) -> String {
        match self {
            Runtime::LlamacppCliStockTools(rt) => {
                format!("{}:{}", rt.source.reference(), rt.flavor)
            }
            Runtime::LlamacppApkPipette(rt) => {
                format!("{}:{}", rt.source.repository_version, rt.flavor)
            }
            Runtime::LlamacppIosPipette(rt) => {
                format!("{}:{}", rt.source.repository_version, rt.flavor)
            }
            // The remaining variants' `Display` already is the binary ref form.
            other => other.to_string(),
        }
    }
}

/// Reduce a CLI `--runtime <arg>` to its [`Runtime::cli_ref`] addressing
/// grammar, accepting either form the plan runner might emit: canonical
/// `Runtime` JSON (what `build_argv` ships) or the flat ref a human types. A
/// `{`-prefixed arg is the JSON form — deserialize it and take its `cli_ref`;
/// anything else is already the flat ref and passes through. The JSON path
/// round-trips by construction: `build_argv` serialized the same `Runtime`
/// whose `cli_ref` the client would have parsed directly.
pub fn cli_ref_from_runtime_arg(arg: &str) -> anyhow::Result<String> {
    if arg.trim_start().starts_with('{') {
        let runtime: Runtime = serde_json::from_str(arg)?;
        Ok(runtime.cli_ref())
    } else {
        Ok(arg.to_string())
    }
}

/// `<repository_url>@<repository_version>`, e.g.
/// `github.com/ggml-org/llama.cpp@b9050`. Preserves the full coordinate (repo
/// *and* ref) so it disambiguates forks/hosts, not just the ref.
impl std::fmt::Display for SourceRepository {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.repository_url, self.repository_version)
    }
}

/// A `GitHubRelease` renders as its `SourceRepository` coordinate (`repo@tag`);
/// an archive as its URL; a local install as `local-dir:<entry-relative path>`.
impl std::fmt::Display for LlamacppCliStockToolsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlamacppCliStockToolsSource::GithubRelease(repo) => write!(f, "{repo}"),
            LlamacppCliStockToolsSource::RemoteArchive { url } => write!(f, "{url}"),
            LlamacppCliStockToolsSource::RelativeDir { dir } => write!(f, "relative-dir:{dir}"),
            LlamacppCliStockToolsSource::AbsoluteDir { dir } => write!(f, "absolute-dir:{dir}"),
        }
    }
}

impl std::fmt::Display for LlamacppCliStockTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.source, self.flavor)
    }
}

impl std::fmt::Display for LlamacppApkPipette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.source, self.flavor)
    }
}

impl std::fmt::Display for LlamacppIosPipette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.source, self.flavor)
    }
}

impl std::fmt::Display for MlxMacosPipette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.version, f)
    }
}

impl std::fmt::Display for MlxIosPipette {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.packages, self.flavor)
    }
}

/// The pinned Swift packages joined space-separated, each as its
/// `SourceRepository` coordinate (`repo@version`), in field order.
impl std::fmt::Display for MlxSwiftStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mlx-swift={} mlx-swift-lm={} swift-transformers={}",
            self.mlx_swift, self.mlx_swift_lm, self.swift_transformers
        )
    }
}

impl std::fmt::Display for DockerVllm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "docker://{}:{}", self.image_name, self.image_tag)
    }
}

impl std::fmt::Display for DockerSglang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "docker://{}:{}", self.image_name, self.image_tag)
    }
}

/// `uv://vllm@<version>` — wire form for torch-oai's `--runtime` arg.
impl std::fmt::Display for UvVllm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "uv://vllm@{}", self.runtime_version())
    }
}

/// `uv://sglang@<version>` — wire form for torch-oai's `--runtime` arg.
impl std::fmt::Display for UvSglang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "uv://sglang@{}", self.runtime_version())
    }
}

/// `uv://openvino@<version>` — wire form for the openvino `--runtime` arg.
impl std::fmt::Display for UvOpenvino {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "uv://openvino@{}", self.runtime_version())
    }
}

/// `{source}:{flavor}` for llamacpp — where `{source}` is the full origin
/// (`<repository_url>@<version>` for a git build, or the archive URL), so the
/// string uniquely identifies the build, not just its ref; `docker://{image}:{tag}`
/// for docker, bare `{version}` for mlx, `uv://<name>@<version>` for uv. This
/// is the canonical string identifier for a runtime: the state `runtime_ref`
/// and cell-hash key, and what shows up in logs/errors. Distinct from this
/// type's TOML serialization (serde's `{ type = "...", ... }` tagged-table plan
/// form). Each `Runtime` variant struct has its own `Display` impl; this enum
/// just dispatches.
impl std::fmt::Display for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Runtime::LlamacppCliStockTools(rt) => rt.fmt(f),
            Runtime::LlamacppApkPipette(rt) => rt.fmt(f),
            Runtime::LlamacppIosPipette(rt) => rt.fmt(f),
            Runtime::MlxMacosPipette(rt) => rt.fmt(f),
            Runtime::MlxIosPipette(rt) => rt.fmt(f),
            Runtime::DockerVllm(rt) => rt.fmt(f),
            Runtime::DockerSglang(rt) => rt.fmt(f),
            Runtime::UvVllm(rt) => rt.fmt(f),
            Runtime::UvSglang(rt) => rt.fmt(f),
            Runtime::UvOpenvino(rt) => rt.fmt(f),
            Runtime::AppleFoundation(_) => write!(f, "apple_foundation"),
        }
    }
}

/// Closed set of vllm build targets.  Identical to `SglangFlavor`
/// today; the two enums exist separately so server-specific targets
/// (e.g. vllm's `apple_gpu`, sglang's `tpu`) can land on one side
/// without polluting the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "clap", clap(rename_all = "snake_case"))]
pub enum VllmFlavor {
    NvidiaGpu,
    AmdGpu,
    Cpu,
}

/// Closed set of sglang build targets.  See `VllmFlavor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "clap", clap(rename_all = "snake_case"))]
pub enum SglangFlavor {
    NvidiaGpu,
    AmdGpu,
    Cpu,
}

/// Closed set of MLX build targets pipette tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "clap", clap(rename_all = "kebab-case"))]
pub enum MlxMacosPipetteFlavor {
    MacosArm64,
}

/// llama.cpp build targets pipette tracks. The known variants
/// mirror canonical pipette/warehouse naming; `Custom(String)` is an
/// escape hatch for operator-supplied builds that aren't in the
/// canonical set (e.g. a private fork or a vendor build).
///
/// Plans may declare any flavor — `Custom` strings work but are
/// inherently local: only hosts that have an install matching the
/// exact string will run them. Stick to the known variants for plans
/// that need to fan out across heterogeneous hosts.
///
/// Wire form is a single kebab-case string. Known variants use their
/// canonical kebab name (`macos-arm64`, `linux-x64-cpu`, …, plus the
/// Android NDK ABI name `android-arm64-v8a`); `Custom(s)` serializes
/// as `s` verbatim. Deserialize tries the known set first and falls
/// through to `Custom` for any string outside it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, strum::EnumIter)]
pub enum LlamaCppFlavor {
    MacosX64,
    MacosArm64,
    MacosArm64Kleidiai,
    LinuxX64Openvino,
    LinuxX64Rocm,
    LinuxX64Cpu,
    LinuxX64SyclFp16,
    LinuxX64SyclFp32,
    LinuxX64Vulkan,
    LinuxArm64Cpu,
    LinuxArm64Vulkan,
    LinuxS390xCpu,
    LinuxOpenEulerAarch64Ascend310p,
    LinuxOpenEulerX86Ascend310p,
    LinuxOpenEulerAarch64Ascend910bAclgraph,
    LinuxOpenEulerX86Ascend910bAclgraph,
    WindowsX64Cpu,
    WindowsArm64Cpu,
    WindowsArm64OpenclAdreno,
    WindowsX64Cuda124,
    WindowsX64Cuda131,
    WindowsX64Vulkan,
    WindowsX64Sycl,
    WindowsX64Hip,
    /// Wire form matches the Android NDK ABI name (`android-arm64-v8a`),
    /// not a kebab-cased rendering of the variant name.
    AndroidArm64Cpu,
    /// Operator-supplied build outside the known set. Round-trips
    /// through serde as the raw string.
    Custom(String),
}

impl LlamaCppFlavor {
    /// Every known variant, in declaration order — the vocabulary a caller may
    /// choose from. Excludes [`Self::Custom`], which is an open escape hatch
    /// with no enumerable membership.
    ///
    /// Exists so a consumer can *offer* the set rather than make the operator
    /// guess it: `runtimes catalog llamacpp_cli_stock_tools` lists these, and
    /// [`Self::parse`] silently yields `Custom` for anything else, so an
    /// unlisted spelling is otherwise indistinguishable from a typo.
    ///
    /// Derived from the enum via `EnumIter` rather than hand-listed, so a new
    /// variant joins the vocabulary by existing — the same reason [`RuntimeType`]
    /// derives it.
    pub fn known() -> impl Iterator<Item = Self> {
        use strum::IntoEnumIterator as _;
        Self::iter().filter(|flavor| !flavor.is_custom())
    }

    /// Whether this is an operator-supplied string outside the known set.
    /// [`Self::parse`] never fails, so this is how a caller tells "the
    /// operator named a flavor pipette tracks" from "the operator typed
    /// something pipette will treat as a private build".
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(_))
    }

    /// Wire form. Known variants emit their canonical
    /// kebab name; `Custom(s)` emits `s` verbatim.
    pub fn as_str(&self) -> &str {
        match self {
            Self::MacosX64 => "macos-x64",
            Self::MacosArm64 => "macos-arm64",
            Self::MacosArm64Kleidiai => "macos-arm64-kleidiai",
            Self::LinuxX64Openvino => "linux-x64-openvino",
            Self::LinuxX64Rocm => "linux-x64-rocm",
            Self::LinuxX64Cpu => "linux-x64-cpu",
            Self::LinuxX64SyclFp16 => "linux-x64-sycl-fp16",
            Self::LinuxX64SyclFp32 => "linux-x64-sycl-fp32",
            Self::LinuxX64Vulkan => "linux-x64-vulkan",
            Self::LinuxArm64Cpu => "linux-arm64-cpu",
            Self::LinuxArm64Vulkan => "linux-arm64-vulkan",
            Self::LinuxS390xCpu => "linux-s390x-cpu",
            Self::LinuxOpenEulerAarch64Ascend310p => "linux-openeuler-aarch64-ascend-310p",
            Self::LinuxOpenEulerX86Ascend310p => "linux-openeuler-x86-ascend-310p",
            Self::LinuxOpenEulerAarch64Ascend910bAclgraph => {
                "linux-openeuler-aarch64-ascend-910b-aclgraph"
            }
            Self::LinuxOpenEulerX86Ascend910bAclgraph => "linux-openeuler-x86-ascend-910b-aclgraph",
            Self::WindowsX64Cpu => "windows-x64-cpu",
            Self::WindowsArm64Cpu => "windows-arm64-cpu",
            Self::WindowsArm64OpenclAdreno => "windows-arm64-opencl-adreno",
            Self::WindowsX64Cuda124 => "windows-x64-cuda-12-4",
            Self::WindowsX64Cuda131 => "windows-x64-cuda-13-1",
            Self::WindowsX64Vulkan => "windows-x64-vulkan",
            Self::WindowsX64Sycl => "windows-x64-sycl",
            Self::WindowsX64Hip => "windows-x64-hip",
            Self::AndroidArm64Cpu => "android-arm64-v8a",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// The upstream `ggml-org/llama.cpp` release asset name for this flavor at
    /// `version`, or `None` for [`Self::Custom`] — an operator-supplied build
    /// has no upstream asset to name.
    ///
    /// Deliberately not derived from [`Self::as_str`]: that is pipette's
    /// canonical vocabulary, this is upstream's, and the two disagree on
    /// several flavors (`linux-x64-openvino` ships as
    /// `ubuntu-openvino-2026.0-x64`, `android-arm64-v8a` as `android-arm64`).
    ///
    /// The extension is derived from the suffix rather than named per variant,
    /// keyed on upstream prefixing every Windows asset with `win-` — the one
    /// family they ship as `.zip`. Deriving it keeps `tar.gz` in a single place
    /// instead of repeating it across 17 arms, at the cost of assuming that
    /// prefix holds: a Windows flavor whose upstream suffix breaks it would
    /// silently resolve to `.tar.gz`. The test table pins the rule for every
    /// variant, so adding one there catches it.
    ///
    /// Lives here rather than in a consumer because both the runtime store
    /// (which downloads the asset) and the CLI's release listing (which checks
    /// one exists) need it, and neither may depend on the other.
    pub fn release_asset_name(&self, version: &str) -> Option<String> {
        let suffix = match self {
            Self::MacosX64 => "macos-x64",
            Self::MacosArm64 => "macos-arm64",
            Self::MacosArm64Kleidiai => "macos-arm64-kleidiai",
            Self::LinuxX64Openvino => "ubuntu-openvino-2026.2.1-x64",
            Self::LinuxX64Rocm => "ubuntu-rocm-7.2-x64",
            Self::LinuxX64Cpu => "ubuntu-x64",
            Self::LinuxX64SyclFp16 => "ubuntu-sycl-fp16-x64",
            Self::LinuxX64SyclFp32 => "ubuntu-sycl-fp32-x64",
            Self::LinuxX64Vulkan => "ubuntu-vulkan-x64",
            Self::LinuxArm64Cpu => "ubuntu-arm64",
            Self::LinuxArm64Vulkan => "ubuntu-vulkan-arm64",
            Self::LinuxS390xCpu => "ubuntu-s390x",
            Self::LinuxOpenEulerAarch64Ascend310p => "310p-openEuler-aarch64",
            Self::LinuxOpenEulerX86Ascend310p => "310p-openEuler-x86",
            Self::LinuxOpenEulerAarch64Ascend910bAclgraph => "910b-openEuler-aarch64-aclgraph",
            Self::LinuxOpenEulerX86Ascend910bAclgraph => "910b-openEuler-x86-aclgraph",
            Self::WindowsX64Cpu => "win-cpu-x64",
            Self::WindowsArm64Cpu => "win-cpu-arm64",
            Self::WindowsArm64OpenclAdreno => "win-opencl-adreno-arm64",
            Self::WindowsX64Cuda124 => "win-cuda-12.4-x64",
            Self::WindowsX64Cuda131 => "win-cuda-13.1-x64",
            Self::WindowsX64Vulkan => "win-vulkan-x64",
            Self::WindowsX64Sycl => "win-sycl-x64",
            Self::WindowsX64Hip => "win-hip-radeon-x64",
            Self::AndroidArm64Cpu => "android-arm64",
            Self::Custom(_) => return None,
        };
        let ext = if suffix.starts_with("win-") {
            "zip"
        } else {
            "tar.gz"
        };
        Some(format!("llama-{version}-bin-{suffix}.{ext}"))
    }

    /// Parse a kebab-case flavor string. Case-insensitive on the
    /// known set; falls through to `Custom(s)` (preserving caller
    /// case) for any string outside the known kebab vocabulary.
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "macos-x64" => Self::MacosX64,
            "macos-arm64" => Self::MacosArm64,
            "macos-arm64-kleidiai" => Self::MacosArm64Kleidiai,
            "linux-x64-openvino" => Self::LinuxX64Openvino,
            "linux-x64-rocm" => Self::LinuxX64Rocm,
            "linux-x64-cpu" => Self::LinuxX64Cpu,
            "linux-x64-sycl-fp16" => Self::LinuxX64SyclFp16,
            "linux-x64-sycl-fp32" => Self::LinuxX64SyclFp32,
            "linux-x64-vulkan" => Self::LinuxX64Vulkan,
            "linux-arm64-cpu" => Self::LinuxArm64Cpu,
            "linux-arm64-vulkan" => Self::LinuxArm64Vulkan,
            "linux-s390x-cpu" => Self::LinuxS390xCpu,
            "linux-openeuler-aarch64-ascend-310p" => Self::LinuxOpenEulerAarch64Ascend310p,
            "linux-openeuler-x86-ascend-310p" => Self::LinuxOpenEulerX86Ascend310p,
            "linux-openeuler-aarch64-ascend-910b-aclgraph" => {
                Self::LinuxOpenEulerAarch64Ascend910bAclgraph
            }
            "linux-openeuler-x86-ascend-910b-aclgraph" => Self::LinuxOpenEulerX86Ascend910bAclgraph,
            "windows-x64-cpu" => Self::WindowsX64Cpu,
            "windows-arm64-cpu" => Self::WindowsArm64Cpu,
            "windows-arm64-opencl-adreno" => Self::WindowsArm64OpenclAdreno,
            "windows-x64-cuda-12-4" => Self::WindowsX64Cuda124,
            "windows-x64-cuda-13-1" => Self::WindowsX64Cuda131,
            "windows-x64-vulkan" => Self::WindowsX64Vulkan,
            "windows-x64-sycl" => Self::WindowsX64Sycl,
            "windows-x64-hip" => Self::WindowsX64Hip,
            "android-arm64-v8a" => Self::AndroidArm64Cpu,
            _ => Self::Custom(s.to_string()),
        }
    }
}

impl std::fmt::Display for LlamaCppFlavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for LlamaCppFlavor {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LlamaCppFlavor {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self::parse(&s))
    }
}

/// Runtime-type discriminant — 1:1 with the [`Runtime`] enum's variants, so a
/// flag entry names its runtime axis by the specific runtime
/// (`runtime_type = "llamacpp_cli_stock_tools"`).
///
/// `EnumIter` lets consumers enumerate every kind without hand-maintaining a
/// parallel list — used by `pipette-plan`'s capability-rules drift guard, which
/// must visit each runtime's policy and would otherwise go stale exactly when a
/// new runtime is added.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, strum::Display, strum::EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum RuntimeType {
    LlamacppCliStockTools,
    LlamacppApkPipette,
    LlamacppIosPipette,
    MlxMacosPipette,
    MlxIosPipette,
    DockerVllm,
    DockerSglang,
    UvVllm,
    UvSglang,
    UvOpenvino,
    AppleFoundation,
}

impl RuntimeType {
    /// The type of a concrete [`Runtime`]. Exhaustive, so adding a `Runtime`
    /// variant fails to compile until this is updated — no silent drift.
    pub fn of(runtime: &Runtime) -> Self {
        match runtime {
            Runtime::LlamacppCliStockTools(_) => Self::LlamacppCliStockTools,
            Runtime::LlamacppApkPipette(_) => Self::LlamacppApkPipette,
            Runtime::LlamacppIosPipette(_) => Self::LlamacppIosPipette,
            Runtime::MlxMacosPipette(_) => Self::MlxMacosPipette,
            Runtime::MlxIosPipette(_) => Self::MlxIosPipette,
            Runtime::DockerVllm(_) => Self::DockerVllm,
            Runtime::DockerSglang(_) => Self::DockerSglang,
            Runtime::UvVllm(_) => Self::UvVllm,
            Runtime::UvSglang(_) => Self::UvSglang,
            Runtime::UvOpenvino(_) => Self::UvOpenvino,
            Runtime::AppleFoundation(_) => Self::AppleFoundation,
        }
    }
}

/// The in-process app variants' source (`SourceRepository`) at the default
/// repo and the given ref. Shared with `plan.rs`'s runtime fixtures.
#[cfg(test)]
pub(crate) fn app_source(version: &str) -> anyhow::Result<SourceRepository> {
    Ok(SourceRepository {
        repository_url: default_repository_url(),
        repository_version: NonEmptyString::try_new(version.to_owned())?,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::*;

    /// The gate's fidelity is part of the runtime's identity, so two builds that differ
    /// only in it are different runtimes. The device compares the declared `Runtime`
    /// against its own and refuses a mismatch, which is what keeps a plan that needs the
    /// gated build from silently getting a stock one's numbers.
    #[test]
    fn private_thermal_separates_two_otherwise_identical_ios_runtimes() -> anyhow::Result<()> {
        let stock = LlamacppIosPipette {
            source: app_source("b10216")?,
            flavor: LlamacppIosPipetteFlavor::IosArm64,
            private_thermal: false,
        };
        let gated = LlamacppIosPipette {
            private_thermal: true,
            ..stock.clone()
        };

        assert_ne!(stock, gated);
        // Absent means stock, so every plan written before this field keeps its meaning.
        let absent: LlamacppIosPipette = serde_json::from_str(
            r#"{"repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10216","flavor":"ios-arm64"}"#,
        )?;
        assert_eq!(absent, stock);
        // And a stock runtime serializes without the field, so the wire form is unchanged.
        let wire = serde_json::to_value(&stock)?;
        assert!(wire.get("private_thermal").is_none(), "{wire}");
        assert_eq!(
            serde_json::to_value(&gated)?.get("private_thermal"),
            Some(&serde_json::Value::Bool(true))
        );
        Ok(())
    }

    /// A plan written before the device moved onto the cell's flags has to be
    /// refused, not quietly stripped.
    ///
    /// Serde ignores unknown fields by default, so without the `deny` the
    /// device an author wrote here is dropped, every cell reaches the client
    /// naming no device, and nothing between the two says why. This is the one
    /// field whose removal was breaking, so it is the one that has to fail
    /// loudly.
    #[test]
    fn a_device_on_the_runtime_is_refused() -> anyhow::Result<()> {
        let json = serde_json::json!({
            "type": "uv_openvino",
            "server_version": "2026.2.1",
            "device": "npu",
            "python_version": "3.11",
            "source": { "type": "pip_requirements_text", "contents": "openvino-genai==2026.2.1.0\n" },
        });
        let Err(err) = serde_json::from_value::<Runtime>(json) else {
            anyhow::bail!("expected a pre-move runtime to be refused");
        };
        // Names the field, so the reader knows which spelling went stale.
        assert!(err.to_string().contains("device"), "got {err}");
        Ok(())
    }

    /// A device outside the named set survives as the operator wrote it —
    /// OpenVINO resolves `GPU.1`, not the lowercased match key, so a `Custom`
    /// that normalized its own spelling would name a device that does not
    /// exist.
    #[rstest]
    #[case::named_lowercase("npu", OpenvinoDevice::Npu)]
    #[case::named_uppercase("NPU", OpenvinoDevice::Npu)]
    #[case::indexed_gpu("GPU.1", OpenvinoDevice::Custom("GPU.1".to_owned()))]
    #[case::virtual_device(
        "HETERO:GPU,CPU",
        OpenvinoDevice::Custom("HETERO:GPU,CPU".to_owned())
    )]
    fn openvino_device_parses_named_and_custom(
        #[case] wire: &str,
        #[case] expected: OpenvinoDevice,
    ) {
        assert_eq!(OpenvinoDevice::parse(wire), expected);
    }

    /// Round trip through the wire form. A named device normalizes to its
    /// canonical spelling; a custom one comes back byte-identical.
    #[rstest]
    #[case("cpu")]
    #[case("gpu")]
    #[case("npu")]
    #[case("GPU.1")]
    #[case("AUTO")]
    fn openvino_device_round_trips(#[case] wire: &str) {
        assert_eq!(OpenvinoDevice::parse(wire).as_str(), wire);
    }

    /// `known()` is what a listing offers, so it must never include the escape
    /// hatch — a catalog row spelling `custom` would parse back to a device
    /// literally named "custom".
    #[test]
    fn openvino_known_devices_exclude_the_escape_hatch() {
        let known: Vec<String> = OpenvinoDevice::known().map(|d| d.to_string()).collect();
        assert_eq!(known, vec!["cpu", "gpu", "npu"]);
    }

    /// Every paste form of one repo normalizes to the single canonical
    /// scheme-less `<host>/<org>/<repo>` — so a coordinate copied from GitHub's
    /// "Code" dropdown (HTTPS, `.git` clone URL, or SSH) stores and compares
    /// identically.
    #[rstest]
    #[case("github.com/ggml-org/llama.cpp")]
    #[case("https://github.com/ggml-org/llama.cpp")]
    #[case("http://github.com/ggml-org/llama.cpp")]
    #[case("https://github.com/ggml-org/llama.cpp/")]
    #[case("https://github.com/ggml-org/llama.cpp.git")]
    #[case("  https://github.com/ggml-org/llama.cpp.git/  ")]
    #[case("git@github.com:ggml-org/llama.cpp.git")]
    #[case("git@github.com:ggml-org/llama.cpp")]
    #[case("ssh://git@github.com/ggml-org/llama.cpp.git")]
    fn repository_url_normalizes_every_paste_form(#[case] raw: &str) {
        assert_eq!(
            RepositoryUrl::new(raw).as_ref(),
            "github.com/ggml-org/llama.cpp",
            "`{raw}` should normalize to the canonical form"
        );
    }

    /// Host-agnostic: a non-GitHub SSH coordinate keeps its host, and `org_repo`
    /// drops only the leading host segment.
    #[test]
    fn repository_url_is_host_agnostic() {
        let repo = RepositoryUrl::new("git@gitlab.com:acme/fork.git");
        assert_eq!(repo.as_ref(), "gitlab.com/acme/fork");
        assert_eq!(repo.org_repo(), "acme/fork");
    }

    #[test]
    fn llamacpp_cli_from_archive_url_parses() -> anyhow::Result<()> {
        // A CLI runtime sourced from a prebuilt archive URL, not a git repo:
        // exercises `LlamacppCliStockToolsSource::RemoteArchive` and its origin slug.
        let runtime: Runtime = toml::from_str(
            r#"type = "llamacpp_cli_stock_tools"
source = "remote_archive"
url = "https://example.com/llama.tar.gz"
flavor = "macos-arm64""#,
        )
        .context("archive runtime should parse")?;
        let Runtime::LlamacppCliStockTools(rt) = &runtime else {
            anyhow::bail!("expected LlamacppCliStockTools");
        };
        assert!(matches!(
            rt.source,
            LlamacppCliStockToolsSource::RemoteArchive { .. }
        ));
        assert_eq!(rt.source.origin_slug(), "remote-archive");
        // Scheme is stripped on construction; download uses https://.
        assert_eq!(rt.source.reference(), "example.com/llama.tar.gz");
        Ok(())
    }

    /// Upstream asset names, which are *not* the wire names above — the two
    /// vocabularies diverge, so this table is maintained against ggml-org's
    /// release listing rather than derived from `as_str`. Every variant is
    /// listed: this is the only definition of the mapping, and the enum carries
    /// `Custom(String)` so `strum` can't enumerate it for us.
    ///
    /// What this pins is the assembly (prefix, version, extension) per variant.
    /// Whether a suffix still matches a real upstream asset is checked at run
    /// time by `release_asset_available`.
    #[rstest::rstest]
    #[case(LlamaCppFlavor::MacosX64, Some("llama-b9305-bin-macos-x64.tar.gz"))]
    #[case(LlamaCppFlavor::MacosArm64, Some("llama-b9305-bin-macos-arm64.tar.gz"))]
    #[case(
        LlamaCppFlavor::MacosArm64Kleidiai,
        Some("llama-b9305-bin-macos-arm64-kleidiai.tar.gz")
    )]
    // Diverges from the wire name `linux-x64-openvino`.
    #[case(
        LlamaCppFlavor::LinuxX64Openvino,
        Some("llama-b9305-bin-ubuntu-openvino-2026.2.1-x64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxX64Rocm,
        Some("llama-b9305-bin-ubuntu-rocm-7.2-x64.tar.gz")
    )]
    #[case(LlamaCppFlavor::LinuxX64Cpu, Some("llama-b9305-bin-ubuntu-x64.tar.gz"))]
    #[case(
        LlamaCppFlavor::LinuxX64SyclFp16,
        Some("llama-b9305-bin-ubuntu-sycl-fp16-x64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxX64SyclFp32,
        Some("llama-b9305-bin-ubuntu-sycl-fp32-x64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxX64Vulkan,
        Some("llama-b9305-bin-ubuntu-vulkan-x64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxArm64Cpu,
        Some("llama-b9305-bin-ubuntu-arm64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxArm64Vulkan,
        Some("llama-b9305-bin-ubuntu-vulkan-arm64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxS390xCpu,
        Some("llama-b9305-bin-ubuntu-s390x.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerAarch64Ascend310p,
        Some("llama-b9305-bin-310p-openEuler-aarch64.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerX86Ascend310p,
        Some("llama-b9305-bin-310p-openEuler-x86.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerAarch64Ascend910bAclgraph,
        Some("llama-b9305-bin-910b-openEuler-aarch64-aclgraph.tar.gz")
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerX86Ascend910bAclgraph,
        Some("llama-b9305-bin-910b-openEuler-x86-aclgraph.tar.gz")
    )]
    #[case(LlamaCppFlavor::WindowsX64Cpu, Some("llama-b9305-bin-win-cpu-x64.zip"))]
    #[case(
        LlamaCppFlavor::WindowsArm64Cpu,
        Some("llama-b9305-bin-win-cpu-arm64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsArm64OpenclAdreno,
        Some("llama-b9305-bin-win-opencl-adreno-arm64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsX64Cuda124,
        Some("llama-b9305-bin-win-cuda-12.4-x64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsX64Cuda131,
        Some("llama-b9305-bin-win-cuda-13.1-x64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsX64Vulkan,
        Some("llama-b9305-bin-win-vulkan-x64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsX64Sycl,
        Some("llama-b9305-bin-win-sycl-x64.zip")
    )]
    #[case(
        LlamaCppFlavor::WindowsX64Hip,
        Some("llama-b9305-bin-win-hip-radeon-x64.zip")
    )]
    // Diverges from the wire name `android-arm64-v8a`.
    #[case(
        LlamaCppFlavor::AndroidArm64Cpu,
        Some("llama-b9305-bin-android-arm64.tar.gz")
    )]
    // An operator-supplied build has no upstream asset.
    #[case(LlamaCppFlavor::Custom("weird".to_owned()), None)]
    fn llamacpp_flavor_release_asset_names(
        #[case] variant: LlamaCppFlavor,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(variant.release_asset_name("b9305").as_deref(), expected);
    }

    #[rstest::rstest]
    #[case(LlamaCppFlavor::MacosX64, "macos-x64")]
    #[case(LlamaCppFlavor::MacosArm64, "macos-arm64")]
    #[case(LlamaCppFlavor::MacosArm64Kleidiai, "macos-arm64-kleidiai")]
    #[case(LlamaCppFlavor::LinuxX64Openvino, "linux-x64-openvino")]
    #[case(LlamaCppFlavor::LinuxX64Rocm, "linux-x64-rocm")]
    #[case(LlamaCppFlavor::LinuxX64Cpu, "linux-x64-cpu")]
    #[case(LlamaCppFlavor::LinuxX64SyclFp16, "linux-x64-sycl-fp16")]
    #[case(LlamaCppFlavor::LinuxX64SyclFp32, "linux-x64-sycl-fp32")]
    #[case(LlamaCppFlavor::LinuxX64Vulkan, "linux-x64-vulkan")]
    #[case(LlamaCppFlavor::LinuxArm64Cpu, "linux-arm64-cpu")]
    #[case(LlamaCppFlavor::LinuxArm64Vulkan, "linux-arm64-vulkan")]
    #[case(LlamaCppFlavor::LinuxS390xCpu, "linux-s390x-cpu")]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerAarch64Ascend310p,
        "linux-openeuler-aarch64-ascend-310p"
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerX86Ascend310p,
        "linux-openeuler-x86-ascend-310p"
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerAarch64Ascend910bAclgraph,
        "linux-openeuler-aarch64-ascend-910b-aclgraph"
    )]
    #[case(
        LlamaCppFlavor::LinuxOpenEulerX86Ascend910bAclgraph,
        "linux-openeuler-x86-ascend-910b-aclgraph"
    )]
    #[case(LlamaCppFlavor::WindowsX64Cpu, "windows-x64-cpu")]
    #[case(LlamaCppFlavor::WindowsArm64Cpu, "windows-arm64-cpu")]
    #[case(
        LlamaCppFlavor::WindowsArm64OpenclAdreno,
        "windows-arm64-opencl-adreno"
    )]
    #[case(LlamaCppFlavor::WindowsX64Cuda124, "windows-x64-cuda-12-4")]
    #[case(LlamaCppFlavor::WindowsX64Cuda131, "windows-x64-cuda-13-1")]
    #[case(LlamaCppFlavor::WindowsX64Vulkan, "windows-x64-vulkan")]
    #[case(LlamaCppFlavor::WindowsX64Sycl, "windows-x64-sycl")]
    #[case(LlamaCppFlavor::WindowsX64Hip, "windows-x64-hip")]
    #[case(LlamaCppFlavor::AndroidArm64Cpu, "android-arm64-v8a")]
    fn llamacpp_flavor_wire_names(
        #[case] variant: LlamaCppFlavor,
        #[case] wire: &str,
    ) -> anyhow::Result<()> {
        #[derive(Deserialize, Serialize)]
        struct Row {
            flavor: LlamaCppFlavor,
        }
        let parsed: Row = toml::from_str(&format!(r#"flavor = "{wire}""#))?;
        assert_eq!(parsed.flavor, variant);
        let emitted = toml::to_string(&Row { flavor: variant })?;
        assert!(emitted.contains(wire), "emit missing {wire:?}: {emitted:?}");
        Ok(())
    }

    #[test]
    fn llamacpp_flavor_custom_round_trips_as_bare_string() -> anyhow::Result<()> {
        #[derive(Deserialize, Serialize)]
        struct Row {
            flavor: LlamaCppFlavor,
        }
        // Unknown kebab string → Custom, original case preserved.
        let parsed: Row = toml::from_str(r#"flavor = "acme-cuda12""#)?;
        assert_eq!(
            parsed.flavor,
            LlamaCppFlavor::Custom("acme-cuda12".to_string()),
        );
        let emitted = toml::to_string(&parsed)?;
        assert!(
            emitted.contains("acme-cuda12"),
            "Custom must emit bare string; got {emitted:?}",
        );

        // Strings that DO match the known set deserialize as the
        // known variant, not Custom. This matters: a `Custom` whose
        // inner string happens to be a known kebab is ill-formed by
        // construction (the parser is its only constructor), and the
        // round-trip preserves the canonical form.
        let parsed: Row = toml::from_str(r#"flavor = "macos-arm64""#)?;
        assert_eq!(parsed.flavor, LlamaCppFlavor::MacosArm64);
        Ok(())
    }

    /// `known()` is the flavor set the CLI offers, so every member has to be
    /// one an operator can actually use. Completeness needs no assertion — the
    /// set is the enum, less `Custom` — but being *offerable* does: a variant
    /// that names no upstream asset would list as a choice that fetches nothing.
    #[test]
    fn every_known_flavor_is_offerable() -> anyhow::Result<()> {
        use strum::IntoEnumIterator as _;

        let known: Vec<_> = LlamaCppFlavor::known().collect();
        assert_eq!(
            known.len(),
            LlamaCppFlavor::iter().count() - 1,
            "known() must drop exactly `Custom`",
        );
        known.iter().for_each(|flavor| {
            assert_eq!(&LlamaCppFlavor::parse(flavor.as_str()), flavor);
            assert!(
                flavor.release_asset_name("b9305").is_some(),
                "{flavor} must name an upstream asset",
            );
        });
        Ok(())
    }

    #[rstest::rstest]
    #[case(VllmFlavor::NvidiaGpu, "nvidia_gpu")]
    #[case(VllmFlavor::AmdGpu, "amd_gpu")]
    #[case(VllmFlavor::Cpu, "cpu")]
    fn vllm_flavor_wire_names(
        #[case] variant: VllmFlavor,
        #[case] wire: &str,
    ) -> anyhow::Result<()> {
        #[derive(Deserialize, Serialize)]
        struct Row {
            flavor: VllmFlavor,
        }
        let parsed: Row = toml::from_str(&format!(r#"flavor = "{wire}""#))?;
        assert_eq!(parsed.flavor, variant);
        let emitted = toml::to_string(&Row { flavor: variant })?;
        assert!(emitted.contains(wire), "emit missing {wire:?}: {emitted:?}");
        Ok(())
    }

    // A wrong mapping here routes a runtime through the other vendor's GPU
    // preflight, so every branch is pinned.
    #[rstest]
    #[case::cuda("cu121", VllmFlavor::NvidiaGpu)]
    #[case::cpu("cpu", VllmFlavor::Cpu)]
    #[case::rocm("rocm624", VllmFlavor::AmdGpu)]
    fn flavor_from_uv_build_cases(
        #[case] build: &str,
        #[case] expected: VllmFlavor,
    ) -> anyhow::Result<()> {
        let build = UvBuild::try_new(build.to_string())?;
        assert_eq!(flavor_from_uv_build(&build), expected);
        Ok(())
    }

    #[rstest::rstest]
    #[case(SglangFlavor::NvidiaGpu, "nvidia_gpu")]
    #[case(SglangFlavor::AmdGpu, "amd_gpu")]
    #[case(SglangFlavor::Cpu, "cpu")]
    fn sglang_flavor_wire_names(
        #[case] variant: SglangFlavor,
        #[case] wire: &str,
    ) -> anyhow::Result<()> {
        #[derive(Deserialize, Serialize)]
        struct Row {
            flavor: SglangFlavor,
        }
        let parsed: Row = toml::from_str(&format!(r#"flavor = "{wire}""#))?;
        assert_eq!(parsed.flavor, variant);
        let emitted = toml::to_string(&Row { flavor: variant })?;
        assert!(emitted.contains(wire), "emit missing {wire:?}: {emitted:?}");
        Ok(())
    }

    #[rstest::rstest]
    #[case(MlxMacosPipetteFlavor::MacosArm64, "macos-arm64")]
    fn mlx_flavor_wire_names(
        #[case] variant: MlxMacosPipetteFlavor,
        #[case] wire: &str,
    ) -> anyhow::Result<()> {
        #[derive(Deserialize, Serialize)]
        struct Row {
            flavor: MlxMacosPipetteFlavor,
        }
        let parsed: Row = toml::from_str(&format!(r#"flavor = "{wire}""#))?;
        assert_eq!(parsed.flavor, variant);
        let emitted = toml::to_string(&Row { flavor: variant })?;
        assert!(emitted.contains(wire), "emit missing {wire:?}: {emitted:?}");
        Ok(())
    }

    #[test]
    fn closed_flavor_enums_reject_unknown_values() {
        let cases = [
            // mlx — anything that isn't a known Apple target
            r#"type = "mlx_macos_pipette"
version = "0.20.0"
source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" }
flavor = "acme-arm64""#,
            // vllm — unknown flavor string
            r#"type = "docker_vllm"
image_name = "vllm/vllm-openai"
image_tag = "v0.20.2"
flavor = "tpu""#,
            // sglang — unknown flavor string
            r#"type = "docker_sglang"
image_name = "lmsysorg/sglang"
image_tag = "v0.5.15-cu130"
flavor = "tpu""#,
        ];
        for runtime_toml in cases {
            assert!(
                toml::from_str::<Runtime>(runtime_toml).is_err(),
                "should reject: {runtime_toml}"
            );
        }
    }

    #[test]
    fn llamacpp_unknown_flavor_parses_as_custom() -> anyhow::Result<()> {
        // llama.cpp's flavor was previously closed; opening it (with
        // `Custom(String)`) means a non-canonical kebab now parses
        // through. Operators get plans that reference operator-local
        // builds. The fanout cost is theirs; the parser accepts.
        let runtime: Runtime = toml::from_str(
            r#"type = "llamacpp_cli_stock_tools"
source = "github_release"
version = "b5000"
flavor = "acme-cuda12""#,
        )
        .context("custom flavor should parse")?;
        match runtime {
            Runtime::LlamacppCliStockTools(lc) => {
                assert_eq!(lc.flavor, LlamaCppFlavor::Custom("acme-cuda12".to_string()));
            }
            other => anyhow::bail!("expected LlamacppCliStockTools, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mlx_ios_flavor_parses_as_known_variant() -> anyhow::Result<()> {
        // `ios-arm64` now belongs to the on-device MLX runtime's flavor enum,
        // not the desktop `MlxMacosPipetteFlavor` (which dropped it).
        #[derive(Deserialize)]
        struct Row {
            flavor: MlxIosPipetteFlavor,
        }
        let parsed: Row = toml::from_str(r#"flavor = "ios-arm64""#)?;
        assert_eq!(parsed.flavor, MlxIosPipetteFlavor::IosArm64);
        Ok(())
    }

    fn sample_catalog_source() -> anyhow::Result<UvRuntimeSource> {
        Ok(UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?,
            install_flags: None,
        })
    }

    #[test]
    fn uv_runtime_source_matrix() {
        let cases: &[(Option<&str>, bool)] = &[
            (
                Some(r#"type = "pip_requirements_text", contents = "vllm==0.10.0""#),
                true,
            ),
            (
                Some(r#"type = "pip_requirements_text", contents = "vllm==0.10.0""#),
                true,
            ),
            (Some(r#"type = "preinstalled", dir = "blobs/venv""#), true),
            // A body is required, and the retired `catalog_slug` is now simply
            // an unknown field rather than something silently accepted.
            (Some(r#"type = "pip_requirements_text""#), false),
            (
                Some(r#"type = "pip_requirements_text", catalog_slug = "x""#),
                false,
            ),
            (Some(r#"type = "path", file = "/host/req.txt""#), false),
            (Some(r#"type = "catalog""#), false),
            (None, false),
            (Some(r#"type = "bogus""#), false),
        ];
        cases.iter().for_each(|(body, expect_ok)| {
            let runtime_toml = match body {
                None => r#"type = "uv_vllm"
server_version = "0.10.0"
build = "cu121"
python_version = "3.12""#
                    .to_owned(),
                Some(body) => format!(
                    r#"type = "uv_vllm"
server_version = "0.10.0"
build = "cu121"
python_version = "3.12"
source = {{ {body} }}"#
                ),
            };
            let parsed = toml::from_str::<Runtime>(&runtime_toml);
            assert_eq!(parsed.is_ok(), *expect_ok, "uv source {body:?}");
        });
    }

    #[test]
    fn uv_vllm_display_emits_uv_scheme() -> anyhow::Result<()> {
        let rt = UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: sample_catalog_source()?,
        };
        assert_eq!(rt.to_string(), "uv://vllm@0.10.0+cu121.py3.12");
        Ok(())
    }

    #[test]
    fn uv_sglang_display_emits_uv_scheme() -> anyhow::Result<()> {
        let rt = UvSglang {
            server_version: UvServerVersion::try_new("0.4.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("sglang==0.4.0\n".to_owned())?,
                install_flags: None,
            },
        };
        assert_eq!(rt.to_string(), "uv://sglang@0.4.0+cu121.py3.12");
        Ok(())
    }

    #[test]
    fn uv_runtime_components_reject_malformed_fields() {
        [
            ("0.10.0", "xpu", "3.12"),
            ("0.10.0", "cu121.py3.12", "3.12"),
            ("0.10.0", "rocm6.py3.12", "3.12"),
            ("0.10.0+cu121", "cu121", "3.12"),
            ("0.10.0", "cu121", "py3.12"),
        ]
        .into_iter()
        .for_each(|(server_version, build, python_version)| {
            let raw = format!(
                r#"
                server_version = "{server_version}"
                build = "{build}"
                python_version = "{python_version}"
                source = {{ type = "pip_requirements_text", contents = "vllm==1" }}
                "#
            );
            assert!(
                toml::from_str::<UvVllm>(&raw).is_err(),
                "{server_version}+{build}.py{python_version} should reject"
            );
        });
    }

    #[test]
    fn uv_python_version_round_trips() -> anyhow::Result<()> {
        let rt = UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: sample_catalog_source()?,
        };
        let emitted = toml::to_string(&rt)?;
        assert!(emitted.contains("python_version = \"3.11\""));
        assert!(!emitted.contains("\nversion = "));
        let parsed: UvVllm = toml::from_str(&emitted)?;
        assert_eq!(parsed, rt);
        Ok(())
    }

    #[test]
    fn uv_source_install_flags_optional_omitted_and_round_trip() -> anyhow::Result<()> {
        let bare = UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: sample_catalog_source()?,
        };
        let bare_json = serde_json::to_value(&bare)?;
        assert!(
            bare_json["source"].get("install_flags").is_none(),
            "None install_flags must be omitted from the wire: {bare_json}"
        );

        let with_flags = UvVllm {
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?,
                install_flags: Some(vec!["--index-strategy".into(), "unsafe-best-match".into()]),
            },
            ..bare
        };
        assert_eq!(
            with_flags.source.install_flags(),
            &[
                "--index-strategy".to_owned(),
                "unsafe-best-match".to_owned()
            ][..]
        );
        let flagged_json = serde_json::to_value(&with_flags)?;
        assert_eq!(
            flagged_json["source"].get("install_flags"),
            Some(&serde_json::json!([
                "--index-strategy",
                "unsafe-best-match"
            ]))
        );
        let reparsed: UvVllm = serde_json::from_value(flagged_json)?;
        assert_eq!(reparsed, with_flags);
        Ok(())
    }

    #[test]
    fn runtime_source_relative_dir_round_trips_and_is_not_fetch_form() -> anyhow::Result<()> {
        let local = LlamacppCliStockToolsSource::RelativeDir {
            dir: RelativePath::try_new("blobs".to_owned())?,
        };
        assert!(matches!(
            local,
            LlamacppCliStockToolsSource::RelativeDir { .. }
        ));
        assert_eq!(local.reference(), "blobs");
        assert_eq!(local.origin_slug(), "local");
        assert_eq!(local.to_string(), "relative-dir:blobs");

        let json = serde_json::to_value(&local)?;
        assert_eq!(
            json,
            serde_json::json!({"source": "relative_dir", "dir": "blobs"})
        );
        let reparsed: LlamacppCliStockToolsSource = serde_json::from_value(json)?;
        assert_eq!(reparsed, local);

        // Plan form (GitHubRelease / RemoteArchive) is not an install dir.
        let release = LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
            repository_url: default_repository_url(),
            repository_version: NonEmptyString::try_new("b1".to_owned())?,
        });
        assert!(!matches!(
            release,
            LlamacppCliStockToolsSource::RelativeDir { .. }
        ));
        Ok(())
    }

    #[test]
    fn runtime_source_absolute_dir_round_trips() -> anyhow::Result<()> {
        let abs = LlamacppCliStockToolsSource::AbsoluteDir {
            dir: AbsolutePath::try_new("/ws/runtimes/key/blobs".to_owned())?,
        };
        assert_eq!(abs.reference(), "/ws/runtimes/key/blobs");
        assert_eq!(abs.to_string(), "absolute-dir:/ws/runtimes/key/blobs");
        let json = serde_json::to_value(&abs)?;
        assert_eq!(
            json,
            serde_json::json!({"source": "absolute_dir", "dir": "/ws/runtimes/key/blobs"})
        );
        assert_eq!(
            serde_json::from_value::<LlamacppCliStockToolsSource>(json)?,
            abs
        );
        Ok(())
    }

    /// Tagged `source` + path-nature validation on `dir`.
    #[rstest::rstest]
    #[case::relative_ok(serde_json::json!({"source": "relative_dir", "dir": "blobs"}), true)]
    #[case::absolute_ok(serde_json::json!({"source": "absolute_dir", "dir": "/ws/r/blobs"}), true)]
    #[case::github_ok(serde_json::json!({"source": "github_release", "repository_version": "b1"}), true)]
    #[case::archive_ok(serde_json::json!({"source": "remote_archive", "url": "ex.com/a.tgz"}), true)]
    #[case::relative_rejects_abs(serde_json::json!({"source": "relative_dir", "dir": "/abs"}), false)]
    #[case::absolute_rejects_rel(serde_json::json!({"source": "absolute_dir", "dir": "blobs"}), false)]
    #[case::untagged_dir_rejected(serde_json::json!({"dir": "blobs"}), false)]
    #[case::missing_source_rejected(serde_json::json!({"repository_version": "b1"}), false)]
    fn runtime_source_path_wire(#[case] v: serde_json::Value, #[case] ok: bool) {
        assert_eq!(
            serde_json::from_value::<LlamacppCliStockToolsSource>(v).is_ok(),
            ok
        );
    }

    #[test]
    fn uv_runtime_source_catalog_pip_and_preinstalled_round_trip() -> anyhow::Result<()> {
        let catalog = sample_catalog_source()?;
        let catalog_json = serde_json::to_value(&catalog)?;
        assert_eq!(catalog_json["type"], "pip_requirements_text");
        // Nothing names where the body came from: a resolved catalog row and a
        // hand-authored one are the same value on the wire.
        assert!(catalog_json.get("catalog_slug").is_none());
        assert!(catalog_json.get("install_flags").is_none());
        assert_eq!(
            serde_json::from_value::<UvRuntimeSource>(catalog_json)?,
            catalog
        );

        let pip = UvRuntimeSource::PipRequirementsText {
            contents: NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?,
            install_flags: None,
        };
        assert_eq!(pip.requirements_text(), Some("vllm==0.10.0\n"));
        let pip_json = serde_json::to_value(&pip)?;
        assert_eq!(pip_json["type"], "pip_requirements_text");
        assert_eq!(serde_json::from_value::<UvRuntimeSource>(pip_json)?, pip);

        let pre = UvRuntimeSource::RelativePreinstalled {
            dir: RelativePath::try_new("blobs/venv".to_owned())?,
        };
        assert!(pre.requirements_text().is_none());
        let pre_json = serde_json::to_value(&pre)?;
        assert_eq!(pre_json["type"], "relative_preinstalled");
        assert_eq!(pre_json["dir"], "blobs/venv");
        assert_eq!(serde_json::from_value::<UvRuntimeSource>(pre_json)?, pre);
        // Legacy wire tag still deserializes.
        let legacy = serde_json::json!({"type": "preinstalled", "dir": "blobs/venv"});
        assert_eq!(serde_json::from_value::<UvRuntimeSource>(legacy)?, pre);

        let mlx = MlxMacosPipette {
            version: NonEmptyString::try_new("0.31".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: pre,
        };
        let mlx_json = serde_json::to_value(&mlx)?;
        assert_eq!(mlx_json["source"]["type"], "relative_preinstalled");
        assert_eq!(serde_json::from_value::<MlxMacosPipette>(mlx_json)?, mlx);
        Ok(())
    }

    #[test]
    fn uv_python_version_is_required() {
        let raw = r#"
            server_version = "0.10.0"
            build = "cu121"
            source = { type = "pip_requirements_text", contents = "vllm==1" }
        "#;
        assert!(toml::from_str::<UvVllm>(raw).is_err());
    }

    // The in-process app runtimes are their own variants with a single-entry
    // flavor; pins each wire tag and flavor spelling. Asserted through the
    // serde round-trip so the `type`/`flavor` wire form is what's checked.
    #[rstest::rstest]
    #[case::apk("llamacpp_apk_pipette", "android-arm64-v8")]
    #[case::ios("llamacpp_ios_pipette", "ios-arm64")]
    fn in_process_pipette_runtime_parses(
        #[case] type_tag: &str,
        #[case] flavor: &str,
    ) -> anyhow::Result<()> {
        let runtime: Runtime = toml::from_str(&format!(
            "type = \"{type_tag}\"\nversion = \"b5000\"\nflavor = \"{flavor}\""
        ))
        .context("in-process runtime should parse")?;
        let value = serde_json::to_value(&runtime)?;
        assert_eq!(value["type"], type_tag);
        assert_eq!(value["flavor"], flavor);
        assert_eq!(value["repository_version"], "b5000");
        Ok(())
    }

    #[test]
    fn mlx_ios_pipette_runtime_parses() -> anyhow::Result<()> {
        // The on-device MLX runtime: three pinned Swift packages (each a
        // repo + ref) plus the iOS flavor.
        let runtime: Runtime = toml::from_str(
            r#"type = "mlx_ios_pipette"
flavor = "ios-arm64"

[packages]
mlx_swift = { repository_url = "github.com/ml-explore/mlx-swift", repository_version = "0.25.6" }
mlx_swift_lm = { repository_url = "github.com/ml-explore/mlx-swift-examples", repository_version = "2.25.6" }
swift_transformers = { repository_url = "github.com/huggingface/swift-transformers", repository_version = "0.1.22" }"#,
        )
        .context("mlx ios pipette runtime should parse")?;
        let Runtime::MlxIosPipette(rt) = &runtime else {
            anyhow::bail!("expected MlxIosPipette, got {runtime:?}");
        };
        assert_eq!(rt.flavor, MlxIosPipetteFlavor::IosArm64);
        assert_eq!(rt.packages.mlx_swift.repository_version.as_ref(), "0.25.6");
        Ok(())
    }

    #[test]
    fn headless_token_matches_plan_type_tag() -> anyhow::Result<()> {
        // `headless_token()` is the plan `type` discriminant; it must equal
        // serde's tag for every variant or the iOS launch arg drifts from the
        // wire form the device parses.
        let runtimes = [
            Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                    repository_url: default_repository_url(),
                    repository_version: NonEmptyString::try_new("b9050".to_owned())?,
                }),
                flavor: LlamaCppFlavor::MacosArm64,
            }),
            Runtime::LlamacppApkPipette(LlamacppApkPipette {
                source: app_source("b1")?,
                flavor: LlamacppApkPipetteFlavor::AndroidArm64V8,
            }),
            Runtime::LlamacppIosPipette(LlamacppIosPipette {
                source: app_source("b1")?,
                flavor: LlamacppIosPipetteFlavor::IosArm64,
                private_thermal: false,
            }),
            Runtime::MlxMacosPipette(MlxMacosPipette {
                version: NonEmptyString::try_new("0.31".to_owned())?,
                flavor: MlxMacosPipetteFlavor::MacosArm64,
                source: sample_catalog_source()?,
            }),
            Runtime::MlxIosPipette(MlxIosPipette {
                packages: MlxSwiftStack {
                    mlx_swift: app_source("0.25.6")?,
                    mlx_swift_lm: app_source("2.25.6")?,
                    swift_transformers: app_source("0.1.22")?,
                },
                flavor: MlxIosPipetteFlavor::IosArm64,
                private_thermal: true,
            }),
            Runtime::UvVllm(UvVllm {
                server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
                build: UvBuild::try_new("cu121".to_owned())?,
                python_version: UvPythonVersion::try_new("3.12".to_owned())?,
                source: sample_catalog_source()?,
            }),
            Runtime::AppleFoundation(Default::default()),
        ];
        for rt in runtimes {
            let tag = serde_json::to_value(&rt)?
                .get("type")
                .and_then(|t| t.as_str())
                .map(str::to_owned)
                .context("runtime serde value has a string `type` tag")?;
            assert_eq!(rt.headless_token(), tag, "drift for {rt:?}");
        }
        Ok(())
    }
}
