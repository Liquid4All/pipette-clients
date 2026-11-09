//! What this client advertises to the planner: the capability flags derived
//! from installed runtimes, the device profile, and the reindex wait that
//! follows a profile change.

use std::time::Duration;

use pipette_device::detect_device_info;
use pipette_http::HttpClient;
use pipette_mgmt_client::{
    ClientIdentity, ServerUrl, client_api,
    types::{ClientProfile, DeviceProfileFields, UpdateClientRequest},
};
use pipette_plan_types::{DeviceFormFactor, DeviceInfo, OvmsSource, Runtime, TrtEdgellmSource};
use pipette_wait::{WaitCallbacks, WaitStatus};

use crate::error::{Error, Result};
use crate::identity::{ClientRegistrationStore, DeviceLabels};

pub const REINDEX_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// How long to wait for reindex before entering the claim loop anyway; the
/// gate is server-side, this is only a local budget.
pub const REINDEX_WAIT_BUDGET: Duration = Duration::from_secs(5 * 60);

/// How the reindex poll paces itself. Only tests choose anything but the
/// [`Default`]. The budget is wall clock, so a short pace alone cannot end the
/// wait. Mirrors the iOS `ReindexWait` (`Client/Worker/Profile.swift`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReindexWait {
    pub poll_interval: Duration,
    pub budget: Duration,
}

impl Default for ReindexWait {
    fn default() -> Self {
        Self {
            poll_interval: REINDEX_POLL_INTERVAL,
            budget: REINDEX_WAIT_BUDGET,
        }
    }
}

/// Where a reindex wait stands at the poll that produced a [`ReindexStatus`].
///
/// Both endings hand the caller a profile, so the phase tells them apart. A
/// claim on a [`Self::BudgetSpent`] profile is answered `204` until the gate
/// lifts, which is intended. A cancelled wait has no phase: it reports nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReindexPhase {
    Waiting,
    Lifted,
    /// The gate still holds; the worker enters the claim loop anyway.
    BudgetSpent,
}

/// One poll of the reindex wait, structured so a caller that draws the wait
/// reads `elapsed` against the budget instead of parsing prose. Same split as
/// `pipette_readiness::ReadinessStatus`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexStatus {
    pub phase: ReindexPhase,
    /// `GET /clients/me` calls so far, counting this one (1-based).
    pub polls: usize,
    /// Measured from before the first sleep.
    pub elapsed: Duration,
    pub wait: ReindexWait,
}

impl std::fmt::Display for ReindexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.phase {
            ReindexPhase::Waiting => write!(
                f,
                "reindex still pending after {}s of {}s ({} poll(s)); next poll in {}s",
                self.elapsed.as_secs(),
                self.wait.budget.as_secs(),
                self.polls,
                self.wait.poll_interval.as_secs(),
            ),
            ReindexPhase::Lifted => f.write_str("reindex gate lifted"),
            ReindexPhase::BudgetSpent => write!(
                f,
                "reindex still pending after {}s; continuing (claim will idle until gate lifts)",
                self.wait.budget.as_secs(),
            ),
        }
    }
}

/// A spent budget is the one phase a reader has to notice, so it warns; the
/// others log at `info`.
impl WaitStatus for ReindexStatus {
    fn report_default(&self) {
        match self.phase {
            ReindexPhase::BudgetSpent => tracing::warn!(
                budget = ?self.wait.budget,
                "reindex still pending; continuing (claim will idle until the gate lifts)"
            ),
            ReindexPhase::Waiting => tracing::info!(
                elapsed = ?self.elapsed,
                budget = ?self.wait.budget,
                polls = self.polls,
                poll_interval = ?self.wait.poll_interval,
                "reindex still pending"
            ),
            ReindexPhase::Lifted => tracing::info!("reindex gate lifted"),
        }
    }
}

/// The cancel is the worker's stop flag, so a `^C` during a five-minute gate
/// leaves startup instead of being noticed only once the gate lifts.
pub type ReindexCallbacks<'a> = WaitCallbacks<'a, ReindexStatus>;

/// Capability flags for one installed runtime: the general `runtime:<name>`
/// flag plus a versioned `runtime:<name>:<build>` when a build id is known.
/// Matching is exact, so both levels must be advertised (the mgmt repo's
/// `docs/client-integration.md`, "Choosing a capability set").
fn runtime_capability_flags(runtime: &Runtime) -> Vec<String> {
    let (name, version) = match runtime {
        Runtime::LlamacppCliStockTools(rt) => {
            ("llama_cpp", Some(rt.source.reference().to_string()))
        }
        Runtime::LlamacppApkPipette(rt) => (
            "llama_cpp",
            Some(rt.source.repository_version.as_ref().to_string()),
        ),
        Runtime::LlamacppIosPipette(rt) => (
            "llama_cpp",
            Some(rt.source.repository_version.as_ref().to_string()),
        ),
        Runtime::MlxMacosPipette(rt) => ("mlx", Some(rt.version.as_ref().to_string())),
        // The device is not in the flag: it is a per-cell choice over one venv.
        Runtime::UvOpenvino(rt) => ("openvino", Some(rt.server_version.as_ref().to_string())),
        // The execution provider *is* in the flag: each provider is its own venv.
        Runtime::UvOnnxGenai(rt) => ("onnx_genai", Some(rt.runtime_version())),
        // As llama.cpp: the release tag, archive URL, vault resource id or dir.
        Runtime::Ovms(rt) => (
            "ovms",
            Some(match &rt.source {
                OvmsSource::GithubRelease(repo) => repo.repository_version.to_string(),
                OvmsSource::RemoteArchive { url } => url.as_ref().to_owned(),
                OvmsSource::VaultRemoteArchive { url, .. } => url.resource_id().to_string(),
                OvmsSource::RelativeDir { dir } => dir.to_string(),
                OvmsSource::AbsoluteDir { dir } => dir.to_string(),
            }),
        ),
        // As OVMS: the archive URL, vault resource id or dir. The archive is
        // named by its content hash, so the URL pins the build.
        Runtime::TrtEdgellm(rt) => (
            "trt_edgellm",
            Some(match &rt.source {
                TrtEdgellmSource::RemoteArchive { url } => url.as_ref().to_owned(),
                TrtEdgellmSource::VaultRemoteArchive { url, .. } => url.resource_id().to_string(),
                TrtEdgellmSource::RelativeDir { dir } => dir.to_string(),
                TrtEdgellmSource::AbsoluteDir { dir } => dir.to_string(),
            }),
        ),
        // Advertise the mlx-swift ref. The iOS app also reports per-package
        // flags (`runtime:mlx:mlx-swift=`, ...) that this function does not
        // derive, so a plan pinning those matches only iOS.
        Runtime::MlxIosPipette(rt) => (
            "mlx",
            Some(
                rt.packages
                    .mlx_swift
                    .repository_version
                    .as_ref()
                    .to_string(),
            ),
        ),
        // Advertise the onnxruntime-genai ref, the one the app calls into. As
        // with MLX, the per-tree flags the iOS app reports match only iOS.
        Runtime::OnnxGenaiIosPipette(rt) => (
            "onnx_genai",
            Some(
                rt.packages
                    .onnxruntime_genai
                    .repository_version
                    .as_ref()
                    .to_string(),
            ),
        ),
        Runtime::DockerVllm(rt) => ("docker_vllm", Some(rt.image_tag.as_ref().to_string())),
        Runtime::DockerSglang(rt) => ("docker_sglang", Some(rt.image_tag.as_ref().to_string())),
        Runtime::UvVllm(rt) => ("uv_vllm", Some(rt.runtime_version())),
        Runtime::UvSglang(rt) => ("uv_sglang", Some(rt.runtime_version())),
        Runtime::AppleFoundation(_) => ("apple_foundation", None),
        Runtime::AppleCoreAiMacosPipette(rt) => (
            "core_ai",
            Some(
                rt.packages
                    .coreai_models
                    .repository_version
                    .as_ref()
                    .to_string(),
            ),
        ),
    };
    let general = format!("runtime:{name}");
    let mut flags = match version {
        Some(v) if !v.is_empty() => vec![
            general.clone(),
            format!("{general}:{}", canonical_build(&v)),
        ],
        _ => vec![general.clone()],
    };
    // Core AI runs only its bundled Swift stack, so it also reports each
    // package. A plan that requires the whole stack then matches only a client
    // that runs it.
    if let Runtime::AppleCoreAiMacosPipette(rt) = runtime {
        let packages = &rt.packages;
        flags.extend(
            [
                ("coreai-models", &packages.coreai_models),
                ("swift-transformers", &packages.swift_transformers),
                ("xgrammar", &packages.xgrammar),
                ("swift-jinja", &packages.swift_jinja),
            ]
            .into_iter()
            .map(|(package, source)| {
                format!(
                    "{general}:{package}={}",
                    canonical_build(source.repository_version.as_ref())
                )
            }),
        );
    }
    flags
}

/// Capability flags must be lowercase with no whitespace.
fn canonical_build(build: &str) -> String {
    build.to_ascii_lowercase().replace(char::is_whitespace, "")
}

/// The literal [`crate::CLIENT_VERSION`] holds when the build names no release
/// (`crates/pipette-cli/build.rs`).
const LOCAL_BUILD_VERSION: &str = "dev";

/// The capability flag for a client build version, or `None` when the version
/// names no release.
///
/// The backstop for a client feature with no flag of its own: a plan pins the
/// build that introduced it. `dev` names no release, so it earns no flag. The
/// value comes from an environment variable and is canonicalized, because a
/// non-canonical flag fails the whole profile update with a `400`.
fn client_version_capability_flag(version: &str) -> Option<String> {
    let canonical = version
        .to_ascii_lowercase()
        .replace(char::is_whitespace, "");
    (!canonical.is_empty() && canonical != LOCAL_BUILD_VERSION)
        .then(|| format!("client_version:{canonical}"))
}

/// Union of the capability flags of the intact entries in the runtime store.
///
/// No `job_schema:<n>` flag is reported: no plan emits one yet (the mgmt repo's
/// `docs/plan-ingestion.md`, "Schema revisions ride on capability flags"). Add
/// it in the same change that starts emitting it.
///
/// An entry whose manifest does not parse is warned about and skipped (see
/// [`RuntimeArtifactStore::list_intact`]) until a job reinstalls it.
///
/// [`RuntimeArtifactStore::list_intact`]: pipette_artifacts::runtime::RuntimeArtifactStore::list_intact
pub fn installed_runtime_capabilities(
    runtimes: &pipette_artifacts::runtime::RuntimeArtifactStore,
) -> Result<Vec<String>> {
    let manifests = runtimes.list_intact().map_err(|e| Error::Other(e.into()))?;
    let flags: std::collections::BTreeSet<_> = manifests
        .iter()
        .flat_map(|m| runtime_capability_flags(&m.declared))
        .collect();
    Ok(flags.into_iter().collect())
}

/// Build a `PATCH /clients/me` body from the local device labels, detected
/// hardware, and installed-runtime capabilities.
fn build_profile_update(
    labels: &DeviceLabels,
    capabilities: Vec<String>,
) -> Result<UpdateClientRequest> {
    let device = detect_device_info(
        labels.device_name.as_ref().map(AsRef::as_ref),
        labels.device_form_factor,
    )
    .map_err(Error::Other)?;
    Ok(profile_update(&device, capabilities))
}

/// The mapping over a caller-supplied snapshot, so a test can pin it against a
/// fixture.
fn profile_update(device: &DeviceInfo, capabilities: Vec<String>) -> UpdateClientRequest {
    let hardware = pipette_device::hardware_capabilities(device);
    tracing::info!(flags = ?hardware, "hardware capability flags");
    let mut capabilities = capabilities;
    capabilities.extend(hardware);
    let client_version = client_version_capability_flag(crate::CLIENT_VERSION);
    match &client_version {
        Some(flag) => tracing::info!(%flag, "client build capability flag"),
        None => tracing::info!(
            version = %crate::CLIENT_VERSION,
            "build version names no release; no client_version capability flag"
        ),
    }
    capabilities.extend(client_version);
    capabilities.sort_unstable();
    capabilities.dedup();
    UpdateClientRequest {
        client_details: None,
        device: DeviceProfileFields {
            device_name: Some(device.device_name.as_ref().to_string()),
            device_form_factor: Some(form_factor_wire(device.device_form_factor).to_string()),
            device_os_name: Some(device.device_os_name.as_ref().to_string()),
            device_os_version: Some(device.device_os_version.as_ref().to_string()),
            device_cpu_arch: device
                .device_cpu_arch
                .as_ref()
                .map(|s| s.as_ref().to_string()),
            device_os_family: device
                .device_os_family
                .as_ref()
                .map(|s| s.as_ref().to_string()),
            device_chip_model: Some(device.device_chip_model.as_ref().to_string()),
            device_ram_bytes: Some(device.device_ram_bytes),
            device_gpu_model: device
                .device_gpu_model
                .as_ref()
                .map(|s| s.as_ref().to_string()),
            device_gpu_vram_bytes: device.device_gpu_vram_bytes,
            device_npu_model: device
                .device_npu_model
                .as_ref()
                .map(|s| s.as_ref().to_string()),
            device_npu_vram_bytes: device.device_npu_vram_bytes,
            capabilities: Some(capabilities),
        },
    }
}

fn form_factor_wire(ff: DeviceFormFactor) -> &'static str {
    match ff {
        DeviceFormFactor::Phone => "phone",
        DeviceFormFactor::Tablet => "tablet",
        DeviceFormFactor::Laptop => "laptop",
        DeviceFormFactor::Desktop => "desktop",
        DeviceFormFactor::Server => "server",
        DeviceFormFactor::Embedded => "embedded",
    }
}

/// Refresh the server-side device profile and capabilities at startup, then
/// wait out `reindex_pending` if the server reports it: a claim before the gate
/// lifts is answered `204` and idles the worker for a whole idle wait. `None`
/// if the caller cancelled the wait.
pub fn refresh_profile_at_startup(
    registration: &ClientRegistrationStore,
    labels: &DeviceLabels,
    http: &HttpClient,
    capabilities: Vec<String>,
    wait: ReindexWait,
    callbacks: &ReindexCallbacks<'_>,
) -> Result<Option<ClientProfile>> {
    let auth = registration.signing_identity()?;
    let server = &registration.server_url()?;
    let request = build_profile_update(labels, capabilities)?;
    tracing::info!(
        flag_count = request.device.capabilities.as_ref().map_or(0, Vec::len),
        cpu_arch = request.device.device_cpu_arch.as_deref(),
        os_family = request.device.device_os_family.as_deref(),
        "refreshing device profile"
    );
    let profile = client_api::update_me(server, http, &auth, request)?;
    if !profile.reindex_pending {
        return Ok(Some(profile));
    }
    tracing::info!(
        poll_interval = ?wait.poll_interval,
        budget = ?wait.budget,
        "reindex_pending=true after profile update; waiting for queue-maintenance"
    );
    wait_for_reindex(server, http, &auth, wait, callbacks)
}

/// Poll `GET /clients/me` until `reindex_pending` is false or the budget is
/// spent; on expiry the last, still-pending profile is returned. A stop returns
/// `None` without polling again. A slow `GET` spends the budget too.
///
/// Every poll is reported: this is a five-minute hold on startup, and a reader
/// needs to see it still going. The iOS `waitForReindex` takes no callbacks yet.
fn wait_for_reindex(
    server: &ServerUrl,
    http: &HttpClient,
    auth: &ClientIdentity,
    wait: ReindexWait,
    callbacks: &ReindexCallbacks<'_>,
) -> Result<Option<ClientProfile>> {
    let started = std::time::Instant::now();
    let deadline = started + wait.budget;
    let mut polls = 0usize;
    loop {
        // Sleep before the first `GET`: the `PATCH` that got us here has just
        // reported the gate. Sliced, so a stop is honored about a second in.
        if callbacks.sleep_cancellably(wait.poll_interval).is_break() {
            // No closing line and no last `GET`: this client sets no request
            // timeout, so that `GET` could outlast the `^C` it was answering.
            return Ok(None);
        }
        let profile = client_api::me(server, http, auth)?;
        polls += 1;
        let status = |phase| ReindexStatus {
            phase,
            polls,
            elapsed: started.elapsed(),
            wait,
        };
        if !profile.reindex_pending {
            callbacks.report(&status(ReindexPhase::Lifted));
            return Ok(Some(profile));
        }
        if std::time::Instant::now() >= deadline {
            callbacks.report(&status(ReindexPhase::BudgetSpent));
            return Ok(Some(profile));
        }
        callbacks.report(&status(ReindexPhase::Waiting));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use anyhow::Context;
    use httpmock::prelude::*;
    use rstest::rstest;
    use strum::IntoEnumIterator;

    use pipette_artifacts::runtime::RuntimeArtifactStore;
    use pipette_mgmt_client::PrivateKey;
    use pipette_plan_types::{
        CapabilityFlag, ClientId, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, RelativePath, RemoteArchiveUrl, RuntimeType, TrimmedString,
    };

    use super::*;
    use crate::identity::IdentityRegistration;

    #[rstest]
    #[case("2026.08.11-640d96e8", Some("client_version:2026.08.11-640d96e8"))]
    #[case("2026.08.11-640D96E8", Some("client_version:2026.08.11-640d96e8"))]
    #[case("2026.08.11 640d96e8", Some("client_version:2026.08.11640d96e8"))]
    #[case("dev", None)]
    #[case("DEV", None)]
    #[case("", None)]
    #[case("   ", None)]
    fn client_version_flag_drops_a_version_that_names_no_release(
        #[case] version: &str,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let flag = client_version_capability_flag(version);
        assert_eq!(flag.as_deref(), expected);
        if let Some(flag) = flag {
            CapabilityFlag::try_new(flag)?;
        }
        Ok(())
    }

    /// One runtime per kind, paired with the flags it must advertise. Spelled
    /// out because a wrong tag is still a canonical flag: the client silently
    /// matches zero jobs. `sample_runtimes_cover_every_runtime_kind` holds this
    /// list to the [`Runtime`] enum.
    const SAMPLE_RUNTIMES: &[(&str, &[&str])] = &[
        // The OS ships the engine and exposes no build id: no versioned level.
        (
            r#"type = "apple_foundation""#,
            &["runtime:apple_foundation"],
        ),
        (
            r#"type = "core_ai_macos_pipette"
packages = { coreai_models = { repository_url = "github.com/apple/coreai-models", repository_version = "27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60" }, swift_transformers = { repository_url = "github.com/huggingface/swift-transformers", repository_version = "1.2.0" }, xgrammar = { repository_url = "github.com/mlc-ai/xgrammar", repository_version = "0.2.2" }, swift_jinja = { repository_url = "github.com/huggingface/swift-jinja", repository_version = "2.3.2" } }"#,
            &[
                "runtime:core_ai",
                "runtime:core_ai:27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60",
                "runtime:core_ai:coreai-models=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60",
                "runtime:core_ai:swift-transformers=1.2.0",
                "runtime:core_ai:xgrammar=0.2.2",
                "runtime:core_ai:swift-jinja=2.3.2",
            ],
        ),
        (
            r#"type = "llamacpp_cli_stock_tools"
source = "github_release"
repository_url = "github.com/ggml-org/llama.cpp"
version = "b5000"
flavor = "macos-arm64""#,
            &["runtime:llama_cpp", "runtime:llama_cpp:b5000"],
        ),
        // The three llama.cpp surfaces are one `runtime:llama_cpp` vocabulary:
        // a plan pins the upstream tag and any of them can match it.
        (
            r#"type = "llamacpp_apk_pipette"
repository_url = "github.com/ggml-org/llama.cpp"
version = "b9050"
flavor = "android-arm64-v8""#,
            &["runtime:llama_cpp", "runtime:llama_cpp:b9050"],
        ),
        (
            r#"type = "llamacpp_ios_pipette"
repository_url = "github.com/ggml-org/llama.cpp"
version = "b10216"
flavor = "ios-arm64""#,
            &["runtime:llama_cpp", "runtime:llama_cpp:b10216"],
        ),
        (
            r#"type = "mlx_macos_pipette"
version = "0.31.3"
flavor = "macos-arm64"
source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.3" }"#,
            &["runtime:mlx", "runtime:mlx:0.31.3"],
        ),
        // The three package versions differ on purpose, so a version read off
        // the wrong package shows up.
        (
            r#"type = "mlx_ios_pipette"
flavor = "ios-arm64"
packages = { mlx_swift = { repository_url = "github.com/ml-explore/mlx-swift", version = "0.31.6" }, mlx_swift_lm = { repository_url = "github.com/ml-explore/mlx-swift-lm", version = "f5f18ed9d" }, swift_transformers = { repository_url = "github.com/huggingface/swift-transformers", version = "1.3.3" } }"#,
            &["runtime:mlx", "runtime:mlx:0.31.6"],
        ),
        // The versions differ on purpose, as in the MLX row above.
        (
            r#"type = "onnx_genai_ios_pipette"
flavor = "ios-arm64"
packages = { onnxruntime = { repository_url = "github.com/microsoft/onnxruntime", version = "1.26.0" }, onnxruntime_genai = { repository_url = "github.com/microsoft/onnxruntime-genai", version = "0.15.2" } }"#,
            &["runtime:onnx_genai", "runtime:onnx_genai:0.15.2"],
        ),
        (
            r#"type = "docker_vllm"
image_name = "vllm/vllm-openai"
image_tag = "v0.21.0"
flavor = "nvidia_gpu""#,
            &["runtime:docker_vllm", "runtime:docker_vllm:v0.21.0"],
        ),
        (
            r#"type = "docker_sglang"
image_name = "lmsysorg/sglang"
image_tag = "v0.5.15-cu130"
flavor = "nvidia_gpu""#,
            &[
                "runtime:docker_sglang",
                "runtime:docker_sglang:v0.5.15-cu130",
            ],
        ),
        // One venv per (version, build, python) triple, and that triple is the pin.
        (
            r#"type = "uv_vllm"
server_version = "0.10.0"
build = "cu121"
python_version = "3.12"
source = { type = "pip_requirements_text", contents = "vllm==0.10.0" }"#,
            &["runtime:uv_vllm", "runtime:uv_vllm:0.10.0+cu121.py3.12"],
        ),
        (
            r#"type = "uv_sglang"
server_version = "0.5.15"
build = "cu130"
python_version = "3.12"
source = { type = "pip_requirements_text", contents = "sglang==0.5.15" }"#,
            &["runtime:uv_sglang", "runtime:uv_sglang:0.5.15+cu130.py3.12"],
        ),
        // OpenVINO pins the bare `server_version`: the device is a per-cell flag
        // over one venv. The tag drops the `uv_` prefix, as `onnx_genai` does.
        (
            r#"type = "uv_openvino"
server_version = "2026.4.1"
python_version = "3.11"
source = { type = "pip_requirements_text", contents = "openvino-genai==2026.4.1.0" }"#,
            &["runtime:openvino", "runtime:openvino:2026.4.1"],
        ),
        // The pin carries the execution provider because two providers are two venvs.
        (
            r#"type = "uv_onnx_genai"
server_version = "0.16.0"
execution_provider = "cuda"
python_version = "3.11"
source = { type = "pip_requirements_text", contents = "onnxruntime-genai-cuda==0.16.0" }"#,
            &[
                "runtime:onnx_genai",
                "runtime:onnx_genai:0.16.0+cuda.py3.11",
            ],
        ),
        // As llama.cpp, the pin is the release tag.
        (
            r#"type = "ovms"
source = "github_release"
repository_url = "github.com/openvinotoolkit/model_server"
repository_version = "v2026.4.1"
flavor = "windows-x64""#,
            &["runtime:ovms", "runtime:ovms:v2026.4.1"],
        ),
        // A vault archive pins by its resource id, as llama.cpp's and OVMS's do.
        (
            r#"type = "trt_edgellm"
source = "vault_remote_archive"
url = "/vault/resources/vault_0195c3e2/content"
principal_id = "org_1"
flavor = "jetson-orin-cuda-13""#,
            &["runtime:trt_edgellm", "runtime:trt_edgellm:vault_0195c3e2"],
        ),
    ];

    /// [`SAMPLE_RUNTIMES`], parsed, each beside the flags it must advertise.
    fn sample_runtimes() -> anyhow::Result<Vec<(Runtime, &'static [&'static str])>> {
        SAMPLE_RUNTIMES
            .iter()
            .map(|(source, expected)| {
                let runtime = toml::from_str::<Runtime>(source)
                    .with_context(|| format!("parsing {source}"))?;
                Ok((runtime, *expected))
            })
            .collect()
    }

    #[test]
    fn runtime_capability_flags_are_the_strings_a_plan_pins() -> anyhow::Result<()> {
        for (runtime, expected) in sample_runtimes()? {
            assert_eq!(
                runtime_capability_flags(&runtime),
                expected,
                "flags for {runtime}"
            );
        }
        Ok(())
    }

    #[test]
    fn sample_runtimes_cover_every_runtime_kind() -> anyhow::Result<()> {
        let sampled: BTreeSet<String> = sample_runtimes()?
            .iter()
            .map(|(runtime, _)| RuntimeType::of(runtime).to_string())
            .collect();
        let missing: Vec<String> = RuntimeType::iter()
            .map(|kind| kind.to_string())
            .filter(|kind| !sampled.contains(kind))
            .collect();
        anyhow::ensure!(
            missing.is_empty(),
            "these runtime kinds have no sample runtime: {missing:?}"
        );
        Ok(())
    }

    /// The server rejects the whole profile update over one non-canonical flag.
    #[test]
    fn every_runtime_capability_flag_is_canonical() -> anyhow::Result<()> {
        for (runtime, _) in sample_runtimes()? {
            for flag in runtime_capability_flags(&runtime) {
                CapabilityFlag::try_new(flag.clone())
                    .with_context(|| format!("{flag} (from {runtime})"))?;
            }
        }
        Ok(())
    }

    /// Matching is whole-string, so the versioned flag does not imply the
    /// general one.
    #[test]
    fn every_runtime_reports_its_general_level() -> anyhow::Result<()> {
        for (runtime, _) in sample_runtimes()? {
            let flags = runtime_capability_flags(&runtime);
            let general = flags
                .first()
                .ok_or_else(|| anyhow::anyhow!("{runtime} advertises nothing"))?;
            assert!(
                general.matches(':').count() == 1,
                "{runtime}'s first flag {general} is not the general level"
            );
            assert!(
                flags[1..]
                    .iter()
                    .all(|f| f.starts_with(&format!("{general}:"))),
                "{runtime} reports a versioned flag outside its own namespace: {flags:?}"
            );
        }
        Ok(())
    }

    /// A remote archive or a local tree pins by URL or path instead of a tag.
    #[rstest]
    #[case::remote_archive(
        LlamacppCliStockToolsSource::RemoteArchive {
            url: RemoteArchiveUrl::try_new("https://Example.test/llama-B9050-macos.zip".to_owned())?
        },
        "runtime:llama_cpp:https://example.test/llama-b9050-macos.zip"
    )]
    #[case::local_tree(
        LlamacppCliStockToolsSource::RelativeDir { dir: RelativePath::try_new("blobs/build".to_owned())? },
        "runtime:llama_cpp:blobs/build"
    )]
    fn a_llama_cpp_source_that_names_no_release_still_reports_a_canonical_flag(
        #[case] source: LlamacppCliStockToolsSource,
        #[case] expected_versioned: &str,
    ) -> anyhow::Result<()> {
        let rt = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source,
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let flags = runtime_capability_flags(&rt);
        assert_eq!(
            flags,
            vec![
                "runtime:llama_cpp".to_string(),
                expected_versioned.to_string()
            ]
        );
        for flag in flags {
            CapabilityFlag::try_new(flag)?;
        }
        Ok(())
    }

    /// Installed with a no-op payload: `list` reads only `manifest.toml`.
    fn store_holding(
        runtimes_dir: &std::path::Path,
        declared: &[&str],
    ) -> anyhow::Result<RuntimeArtifactStore> {
        let store = RuntimeArtifactStore::new(runtimes_dir.to_path_buf());
        for source in declared {
            let runtime =
                toml::from_str::<Runtime>(source).with_context(|| format!("parsing {source}"))?;
            store.ensure(&runtime, |_, _| Ok(()))?;
        }
        Ok(store)
    }

    const LLAMA_B9050: &str = r#"type = "llamacpp_cli_stock_tools"
source = "github_release"
repository_url = "github.com/ggml-org/llama.cpp"
version = "b9050"
flavor = "macos-arm64""#;
    const LLAMA_B10216: &str = r#"type = "llamacpp_cli_stock_tools"
source = "github_release"
repository_url = "github.com/ggml-org/llama.cpp"
version = "b10216"
flavor = "macos-arm64""#;
    const MLX_0_31_3: &str = r#"type = "mlx_macos_pipette"
version = "0.31.3"
flavor = "macos-arm64"
source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.3" }"#;

    #[test]
    fn installed_runtime_capabilities_is_empty_when_nothing_is_installed() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let never_created = RuntimeArtifactStore::new(tmp.path().join("runtimes"));
        assert!(installed_runtime_capabilities(&never_created)?.is_empty());

        // An existing but empty `runtimes/` (after `runtimes remove`) is a
        // different code path.
        let emptied = tmp.path().join("emptied");
        std::fs::create_dir_all(&emptied)?;
        let store = RuntimeArtifactStore::new(emptied);
        assert!(installed_runtime_capabilities(&store)?.is_empty());
        Ok(())
    }

    /// Sorted and deduplicated: an unstable order would change the reported set
    /// on every launch, which the server reads as a profile change and answers
    /// by voiding the client's queue standing.
    #[test]
    fn installed_runtime_capabilities_unions_every_installed_runtime() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = store_holding(tmp.path(), &[LLAMA_B9050, MLX_0_31_3, LLAMA_B10216])?;
        assert_eq!(
            installed_runtime_capabilities(&store)?,
            vec![
                "runtime:llama_cpp",
                "runtime:llama_cpp:b10216",
                "runtime:llama_cpp:b9050",
                "runtime:mlx",
                "runtime:mlx:0.31.3",
            ]
        );
        Ok(())
    }

    /// The bundled Core AI runtime's flags, as its store manifest reports them.
    #[test]
    fn the_bundled_core_ai_runtime_reports_its_coreai_models_commit() {
        assert_eq!(
            runtime_capability_flags(&Runtime::AppleCoreAiMacosPipette(
                pipette_plan_types::AppleCoreAiMacosPipette::bundled()
            )),
            vec![
                "runtime:core_ai",
                "runtime:core_ai:27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60",
                "runtime:core_ai:coreai-models=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60",
                "runtime:core_ai:swift-transformers=1.2.0",
                "runtime:core_ai:xgrammar=0.2.2",
                "runtime:core_ai:swift-jinja=2.3.2",
            ]
        );
    }

    /// Pins the deliberate absence (see [`installed_runtime_capabilities`]).
    #[test]
    fn no_job_schema_flag_is_reported() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = store_holding(tmp.path(), &[LLAMA_B9050, MLX_0_31_3])?;
        assert!(
            installed_runtime_capabilities(&store)?
                .iter()
                .all(|flag| !flag.starts_with("job_schema:"))
        );
        Ok(())
    }

    #[test]
    fn installed_runtime_capabilities_skips_a_damaged_entry_and_logs_it() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = store_holding(tmp.path(), &[LLAMA_B9050, MLX_0_31_3])?;
        let mlx = toml::from_str::<Runtime>(MLX_0_31_3)?;
        let torn = store
            .find(&mlx)?
            .context("mlx is installed")?
            .entry_dir(tmp.path())?
            .join("manifest.toml");
        std::fs::write(&torn, [0u8; 444])?;

        let (caps, logged) =
            crate::test_support::logged_text(|| installed_runtime_capabilities(&store))?;

        assert_eq!(caps?, vec!["runtime:llama_cpp", "runtime:llama_cpp:b9050"]);
        assert!(
            logged.contains("WARN") && logged.contains(&torn.display().to_string()),
            "the skipped entry is logged with its path: {logged}"
        );
        Ok(())
    }

    /// Only an unparseable manifest is skipped; one that parses but matches no
    /// known schema stops the survey.
    #[test]
    fn installed_runtime_capabilities_surfaces_an_unknown_schema() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store = store_holding(tmp.path(), &[LLAMA_B9050])?;
        let unknown = tmp.path().join("llama-cpp__unknown");
        std::fs::create_dir_all(&unknown)?;
        std::fs::write(unknown.join("manifest.toml"), b"foo = 1\n")?;

        assert!(installed_runtime_capabilities(&store).is_err());
        Ok(())
    }

    /// Detected values can only be asserted for shape, so the mapping is pinned
    /// against this snapshot.
    fn device_info_fixture() -> DeviceInfo {
        DeviceInfo {
            device_name: TrimmedString::new("lab-07"),
            device_form_factor: DeviceFormFactor::Laptop,
            device_os_name: TrimmedString::new("macOS"),
            device_os_version: TrimmedString::new("26.4"),
            device_os_build: Some(TrimmedString::new("24F74")),
            device_os_security_patch: None,
            device_cpu_arch: Some(TrimmedString::new("aarch64")),
            device_os_family: Some(TrimmedString::new("macos")),
            device_chip_model: TrimmedString::new("Apple M3 Max"),
            device_ram_bytes: 137_438_953_472,
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
            device_npu_model: None,
            device_npu_vram_bytes: None,
        }
    }

    fn encoded(request: &UpdateClientRequest) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(request)?)
    }

    #[test]
    fn profile_update_maps_each_field_from_the_device_snapshot() -> anyhow::Result<()> {
        let json = encoded(&profile_update(&device_info_fixture(), Vec::new()))?;
        assert_eq!(json["device_name"], "lab-07");
        assert_eq!(json["device_form_factor"], "laptop");
        assert_eq!(json["device_os_name"], "macOS");
        assert_eq!(json["device_os_version"], "26.4");
        assert_eq!(json["device_cpu_arch"], "aarch64");
        assert_eq!(json["device_os_family"], "macos");
        assert_eq!(json["device_chip_model"], "Apple M3 Max");
        assert_eq!(json["device_ram_bytes"], 137_438_953_472u64);
        Ok(())
    }

    /// `client_details` is the operator's; a per-launch refresh would overwrite it.
    #[test]
    fn a_profile_update_sends_no_client_details() -> anyhow::Result<()> {
        let json = encoded(&profile_update(&device_info_fixture(), Vec::new()))?;
        assert!(json.get("client_details").is_none());
        Ok(())
    }

    /// A `*_vram_bytes` without its `*_model` is a `400`, so absent, never null.
    /// The OS build and patch ride on benchmark submissions instead.
    #[test]
    fn a_profile_update_omits_what_the_schema_has_no_column_for() -> anyhow::Result<()> {
        let json = encoded(&profile_update(&device_info_fixture(), Vec::new()))?;
        for key in [
            "device_gpu_model",
            "device_gpu_vram_bytes",
            "device_npu_model",
            "device_npu_vram_bytes",
            "device_os_build",
            "device_os_security_patch",
        ] {
            assert!(json.get(key).is_none(), "{key} must not be sent");
        }
        Ok(())
    }

    #[test]
    fn a_profile_update_sends_an_accelerator_it_detected() -> anyhow::Result<()> {
        let device = DeviceInfo {
            device_gpu_model: Some(TrimmedString::new("NVIDIA H100 80GB HBM3")),
            device_gpu_vram_bytes: Some(85_899_345_920),
            ..device_info_fixture()
        };
        let json = encoded(&profile_update(&device, Vec::new()))?;
        assert_eq!(json["device_gpu_model"], "NVIDIA H100 80GB HBM3");
        assert_eq!(json["device_gpu_vram_bytes"], 85_899_345_920u64);
        Ok(())
    }

    /// A present set replaces the stored one wholesale, so the body must carry
    /// the runtime flags *and* the hardware flags.
    #[test]
    fn a_profile_update_sends_the_full_merged_capability_set() -> anyhow::Result<()> {
        let device = device_info_fixture();
        let hardware = pipette_device::hardware_capabilities(&device);
        let mut capabilities = vec!["runtime:mlx".to_string(), "runtime:llama_cpp".to_string()];
        // This host's first hardware flag, handed in again: the merge must report it once.
        capabilities.extend(hardware.first().cloned());
        let request = profile_update(&device, capabilities);
        let sent = request
            .device
            .capabilities
            .context("a profile update always sends a capability set")?;

        assert!(sent.contains(&"runtime:mlx".to_string()));
        assert!(sent.contains(&"runtime:llama_cpp".to_string()));
        for flag in &hardware {
            assert!(
                sent.contains(flag),
                "{flag} was dropped from the merged set"
            );
        }
        let mut deduped = sent.clone();
        deduped.dedup();
        assert_eq!(sent, deduped, "a repeated flag reached the wire");
        let mut sorted = sent.clone();
        sorted.sort();
        assert_eq!(sent, sorted, "the reported order must be stable");
        for flag in sent {
            CapabilityFlag::try_new(flag)?;
        }
        Ok(())
    }

    /// `CLIENT_VERSION` is `dev` locally and a release in CI, so only the count
    /// is assertable.
    #[test]
    fn a_profile_update_carries_this_builds_client_version_flag() -> anyhow::Result<()> {
        let request = profile_update(&device_info_fixture(), Vec::new());
        let sent = request.device.capabilities.context("capability set")?;
        assert_eq!(
            sent.iter()
                .filter(|f| f.starts_with("client_version:"))
                .count(),
            usize::from(client_version_capability_flag(crate::CLIENT_VERSION).is_some())
        );
        Ok(())
    }

    #[test]
    fn the_operators_labels_name_the_device_on_the_detected_profile() -> anyhow::Result<()> {
        let labels = DeviceLabels {
            device_name: Some(TrimmedString::new("  lab-07  ")),
            device_form_factor: Some(DeviceFormFactor::Server),
        };

        let json = encoded(&build_profile_update(&labels, Vec::new())?)?;

        assert_eq!(json["device_name"], "lab-07");
        assert_eq!(json["device_form_factor"], "server");
        for key in [
            "device_os_name",
            "device_os_version",
            "device_cpu_arch",
            "device_os_family",
            "device_chip_model",
        ] {
            assert!(
                json[key].as_str().is_some_and(|v| !v.is_empty()),
                "{key} must be detected and non-empty"
            );
        }
        assert!(json["device_ram_bytes"].as_u64().is_some_and(|b| b > 0));
        Ok(())
    }

    /// Written out, not derived: a test derived from the same enum would agree
    /// with any typo.
    #[rstest]
    #[case(DeviceFormFactor::Phone, "phone")]
    #[case(DeviceFormFactor::Tablet, "tablet")]
    #[case(DeviceFormFactor::Laptop, "laptop")]
    #[case(DeviceFormFactor::Desktop, "desktop")]
    #[case(DeviceFormFactor::Server, "server")]
    #[case(DeviceFormFactor::Embedded, "embedded")]
    fn form_factor_wire_spells_each_form_factor_as_the_server_does(
        #[case] form_factor: DeviceFormFactor,
        #[case] expected: &str,
    ) {
        assert_eq!(form_factor_wire(form_factor), expected);
    }

    /// The table above could miss a new form factor; this walks the enum.
    #[test]
    fn every_form_factor_is_wired_as_the_type_serializes_it() -> anyhow::Result<()> {
        for form_factor in DeviceFormFactor::iter() {
            assert_eq!(
                serde_json::to_value(form_factor)?,
                serde_json::Value::String(form_factor_wire(form_factor).to_string()),
                "{form_factor} is wired as something other than its own serialization"
            );
        }
        Ok(())
    }

    fn test_http() -> anyhow::Result<HttpClient> {
        Ok(HttpClient::new("pipette-test/0")?)
    }

    /// Any 32 bytes are a valid Ed25519 seed.
    const KEY: &str = "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899";

    fn test_auth() -> anyhow::Result<ClientIdentity> {
        Ok(ClientIdentity {
            client_id: ClientId::try_new("ev1_test")?,
            private_key_hex: PrivateKey::try_new(KEY)?,
        })
    }

    /// A workspace registered against `server`.
    fn seed_registration(
        root: &std::path::Path,
        server: &MockServer,
    ) -> anyhow::Result<ClientRegistrationStore> {
        let registration = ClientRegistrationStore::new(root.join("identity"));
        registration.put_private_key(&PrivateKey::try_new(KEY)?)?;
        registration.put(&IdentityRegistration {
            client_id: ClientId::try_new("ev1_test")?,
            server_url: ServerUrl::try_new(server.base_url())?,
        })?;
        Ok(registration)
    }

    fn profile_body(pending: bool) -> String {
        format!(
            r#"{{"client_id": "ev1_test", "organization": "o", "client_details": "d",
                "contact_email": "a@b.com", "status": "approved",
                "reindex_pending": {pending}}}"#
        )
    }

    /// No pace, so only `budget` ends the poll loop, and callbacks that never cancel.
    fn refresh_at_startup(
        registration: &ClientRegistrationStore,
        http: &HttpClient,
        capabilities: Vec<String>,
        budget: Duration,
    ) -> Result<Option<ClientProfile>> {
        refresh_profile_at_startup(
            registration,
            &DeviceLabels::default(),
            http,
            capabilities,
            ReindexWait {
                poll_interval: Duration::ZERO,
                budget,
            },
            &ReindexCallbacks::default(),
        )
    }

    /// Separate from the call so a setup failure still fails a test that only
    /// asserts an error.
    fn wait_args(server: &MockServer) -> anyhow::Result<(ServerUrl, HttpClient, ClientIdentity)> {
        Ok((
            ServerUrl::try_new(server.base_url())?,
            test_http()?,
            test_auth()?,
        ))
    }

    /// What a wait reported, and what it would have slept.
    #[derive(Default)]
    struct Recorded {
        slept: std::cell::RefCell<Vec<Duration>>,
        statuses: std::cell::RefCell<Vec<ReindexStatus>>,
    }

    impl Recorded {
        /// Records a slice instead of spending it, and returns the total so far,
        /// so a test can say "three polls in" without counting slices.
        fn sleep(&self, slice: Duration) -> Duration {
            let mut slept = self.slept.borrow_mut();
            slept.push(slice);
            slept.iter().sum()
        }

        fn status(&self, status: &ReindexStatus) {
            self.statuses.borrow_mut().push(status.clone());
        }

        fn slept(&self) -> Vec<Duration> {
            self.slept.borrow().clone()
        }

        fn slept_total(&self) -> Duration {
            self.slept.borrow().iter().sum()
        }

        fn phases(&self) -> Vec<ReindexPhase> {
            self.statuses.borrow().iter().map(|s| s.phase).collect()
        }

        fn polls(&self) -> Vec<usize> {
            self.statuses.borrow().iter().map(|s| s.polls).collect()
        }

        fn waits(&self) -> Vec<ReindexWait> {
            self.statuses.borrow().iter().map(|s| s.wait).collect()
        }
    }

    #[test]
    fn refresh_profile_at_startup_patches_once_when_the_gate_is_clear() -> anyhow::Result<()> {
        let server = MockServer::start();
        let patch = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200).body(profile_body(false));
        });
        let polled = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(false));
        });
        let tmp = tempfile::tempdir()?;
        let registration = seed_registration(tmp.path(), &server)?;

        let profile = refresh_at_startup(
            &registration,
            &test_http()?,
            vec!["runtime:llama_cpp".to_string()],
            Duration::from_secs(60),
        )?
        .ok_or_else(|| anyhow::anyhow!("these callbacks never cancel"))?;

        assert!(!profile.reindex_pending);
        patch.assert();
        polled.assert_calls(0);
        Ok(())
    }

    #[test]
    fn refresh_profile_at_startup_sends_the_capability_set() -> anyhow::Result<()> {
        let server = MockServer::start();
        let patch = server.mock(|when, then| {
            when.method(httpmock::Method::PATCH)
                .path("/clients/me")
                .body_includes("runtime:llama_cpp:b9050");
            then.status(200).body(profile_body(false));
        });
        let tmp = tempfile::tempdir()?;
        let registration = seed_registration(tmp.path(), &server)?;

        refresh_at_startup(
            &registration,
            &test_http()?,
            vec![
                "runtime:llama_cpp".to_string(),
                "runtime:llama_cpp:b9050".to_string(),
            ],
            Duration::from_secs(60),
        )?;

        patch.assert();
        Ok(())
    }

    #[test]
    fn refresh_profile_at_startup_waits_out_the_reindex_gate() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200).body(profile_body(true));
        });
        let polled = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(false));
        });
        let tmp = tempfile::tempdir()?;
        let registration = seed_registration(tmp.path(), &server)?;

        let profile = refresh_at_startup(
            &registration,
            &test_http()?,
            Vec::new(),
            Duration::from_secs(60),
        )?
        .ok_or_else(|| anyhow::anyhow!("these callbacks never cancel"))?;

        assert!(!profile.reindex_pending, "the gate lifted");
        polled.assert();
        Ok(())
    }

    /// The still-pending profile comes back rather than holding startup open.
    #[test]
    fn a_spent_budget_ends_the_startup_wait() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(httpmock::Method::PATCH).path("/clients/me");
            then.status(200).body(profile_body(true));
        });
        let polled = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(true));
        });
        let tmp = tempfile::tempdir()?;
        let registration = seed_registration(tmp.path(), &server)?;

        let profile = refresh_at_startup(&registration, &test_http()?, Vec::new(), Duration::ZERO)?
            .ok_or_else(|| anyhow::anyhow!("these callbacks never cancel"))?;

        assert!(profile.reindex_pending, "the gate never lifted");
        polled.assert_calls(1);
        Ok(())
    }

    /// iOS answers this with `nil` instead: its UI runs unenrolled, the worker cannot.
    #[test]
    fn refresh_profile_at_startup_refuses_an_unregistered_workspace() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let registration = ClientRegistrationStore::new(tmp.path().join("identity"));

        let err = refresh_at_startup(&registration, &test_http()?, Vec::new(), Duration::ZERO)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a workspace with no registration cannot refresh"))?;
        assert!(matches!(err, Error::RegistrationMissing(_)), "got {err:?}");
        Ok(())
    }

    /// Every poll is reported, not only the last. Scripted through the injected
    /// sleep, since httpmock has no per-call sequencing: swapping the mock in the
    /// gap makes the next `GET` a different answer.
    #[test]
    fn the_wait_ends_when_the_gate_lifts_on_a_later_poll() -> anyhow::Result<()> {
        const LIFTS_ON: u32 = 3;
        let server = MockServer::start();
        let pending = std::cell::RefCell::new(server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(true));
        }));
        let recorded = Recorded::default();
        // Swap on the last slice of the third gap and no other: deleting an
        // already-deleted mock panics.
        let sleep = |slice: Duration| {
            if recorded.sleep(slice) != REINDEX_POLL_INTERVAL * LIFTS_ON {
                return;
            }
            pending.borrow_mut().delete();
            server.mock(|when, then| {
                when.method(GET).path("/clients/me");
                then.status(200).body(profile_body(false));
            });
        };
        let on_status = |status: &ReindexStatus| recorded.status(status);
        let callbacks = ReindexCallbacks::new()
            .with_status(&on_status)
            .with_sleep(&sleep);

        // A short wall-clock budget: a regression that stopped calling the sleep
        // would otherwise poll flat out for five minutes before failing.
        let wait = ReindexWait {
            poll_interval: REINDEX_POLL_INTERVAL,
            budget: Duration::from_secs(5),
        };
        let (url, http, auth) = wait_args(&server)?;

        let profile = wait_for_reindex(&url, &http, &auth, wait, &callbacks)?
            .ok_or_else(|| anyhow::anyhow!("nothing cancelled this wait"))?;

        assert!(
            !profile.reindex_pending,
            "the gate lifted on poll {LIFTS_ON}",
        );
        assert_eq!(
            recorded.slept_total(),
            REINDEX_POLL_INTERVAL * LIFTS_ON,
            "one production interval before each poll, none after the last",
        );
        assert!(
            recorded
                .slept()
                .iter()
                .all(|slice| *slice <= pipette_wait::CANCEL_CHECK_INTERVAL),
            "a gap is spent in slices, so a stop lands within one: {:?}",
            recorded.slept(),
        );
        assert_eq!(
            recorded.phases(),
            vec![
                ReindexPhase::Waiting,
                ReindexPhase::Waiting,
                ReindexPhase::Lifted,
            ],
            "a held gate has to say it is still going",
        );
        assert_eq!(
            recorded.polls(),
            vec![1, 2, 3],
            "the poll count is the wait's own, 1-based",
        );
        assert_eq!(
            recorded.waits(),
            vec![wait; 3],
            "every status carries the policy the wait was given, not a default",
        );
        Ok(())
    }

    /// Pinned with a budget already spent, which is the one-poll wait.
    #[test]
    fn the_wait_sleeps_before_its_first_poll() -> anyhow::Result<()> {
        let server = MockServer::start();
        let polled = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(true));
        });
        let recorded = Recorded::default();
        let sleep = |interval: Duration| {
            recorded.sleep(interval);
        };
        let on_status = |status: &ReindexStatus| recorded.status(status);
        let callbacks = ReindexCallbacks::new()
            .with_status(&on_status)
            .with_sleep(&sleep);

        let wait = ReindexWait {
            poll_interval: REINDEX_POLL_INTERVAL,
            budget: Duration::ZERO,
        };
        let (url, http, auth) = wait_args(&server)?;

        let profile = wait_for_reindex(&url, &http, &auth, wait, &callbacks)?
            .ok_or_else(|| anyhow::anyhow!("nothing cancelled this wait"))?;

        assert!(profile.reindex_pending, "the gate never lifted");
        assert_eq!(polled.calls(), 1);
        assert_eq!(recorded.slept_total(), REINDEX_POLL_INTERVAL);
        assert_eq!(
            recorded.phases(),
            vec![ReindexPhase::BudgetSpent],
            "a spent budget is reported as itself, not as a lifted gate",
        );
        Ok(())
    }

    /// Not at the next poll or the budget, and without a further `GET` or a
    /// closing line.
    #[test]
    fn a_raised_stop_ends_the_wait_inside_the_gap() -> anyhow::Result<()> {
        let server = MockServer::start();
        let polled = server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(200).body(profile_body(true));
        });
        let recorded = Recorded::default();
        // The `^C`, landing inside the first slice of the first gap.
        let stopped = std::cell::Cell::new(false);
        let sleep = |slice: Duration| {
            recorded.sleep(slice);
            stopped.set(true);
        };
        let cancelled = || stopped.get();
        let on_status = |status: &ReindexStatus| recorded.status(status);
        let callbacks = ReindexCallbacks::new()
            .with_status(&on_status)
            .with_cancel(&cancelled)
            .with_sleep(&sleep);

        // Short, so a wait that ignores the cancel fails fast.
        let wait = ReindexWait {
            poll_interval: REINDEX_POLL_INTERVAL,
            budget: Duration::from_secs(5),
        };
        let (url, http, auth) = wait_args(&server)?;

        let profile = wait_for_reindex(&url, &http, &auth, wait, &callbacks)?;

        assert!(profile.is_none(), "a cancelled wait hands back nothing");
        assert_eq!(
            recorded.slept_total(),
            pipette_wait::CANCEL_CHECK_INTERVAL,
            "the rest of the gap was spent after the stop",
        );
        assert!(
            recorded.phases().is_empty(),
            "the wait was called off, not given up on, so it says nothing: {:?}",
            recorded.phases(),
        );
        assert_eq!(polled.calls(), 0, "the stop lands before the next `GET`");
        Ok(())
    }

    /// Only `Waiting` is logged this way, as the debug `wait poll` line.
    #[rstest]
    #[case::waiting(
        ReindexPhase::Waiting,
        Duration::from_secs(15),
        "reindex still pending after 15s of 300s (3 poll(s)); next poll in 5s"
    )]
    #[case::lifted(ReindexPhase::Lifted, Duration::from_secs(15), "reindex gate lifted")]
    #[case::budget_spent(
        ReindexPhase::BudgetSpent,
        Duration::from_secs(300),
        "reindex still pending after 300s; continuing (claim will idle until gate lifts)"
    )]
    fn a_reindex_status_renders_as_one_line(
        #[case] phase: ReindexPhase,
        #[case] elapsed: Duration,
        #[case] expected: &str,
    ) {
        let status = ReindexStatus {
            phase,
            polls: 3,
            elapsed,
            wait: ReindexWait::default(),
        };
        assert_eq!(status.to_string(), expected);
    }

    #[test]
    fn the_wait_surfaces_a_server_error() -> anyhow::Result<()> {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/clients/me");
            then.status(503).body("down");
        });

        let (url, http, auth) = wait_args(&server)?;

        assert!(
            wait_for_reindex(
                &url,
                &http,
                &auth,
                ReindexWait {
                    poll_interval: Duration::ZERO,
                    budget: Duration::from_secs(60)
                },
                &ReindexCallbacks::default(),
            )
            .is_err()
        );
        Ok(())
    }
}

#[cfg(test)]
mod coreai_store_tests {
    use super::*;
    #[test]
    fn coreai_capability_requires_an_installed_manifest() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let store =
            pipette_artifacts::runtime::RuntimeArtifactStore::new(tmp.path().join("runtimes"));
        assert!(installed_runtime_capabilities(&store)?.is_empty());
        let runtime = Runtime::AppleCoreAiMacosPipette(Default::default());
        store.ensure(&runtime, |_, blobs| {
            std::fs::write(blobs.join("pipette-coreai-sidecar"), b"test")?;
            Ok(())
        })?;
        assert_eq!(installed_runtime_capabilities(&store)?, {
            let mut flags = runtime_capability_flags(&runtime);
            flags.sort();
            flags
        });
        store.remove(&runtime)?;
        assert!(installed_runtime_capabilities(&store)?.is_empty());
        Ok(())
    }
}
