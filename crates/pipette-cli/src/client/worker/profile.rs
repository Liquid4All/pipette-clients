//! What this client advertises to the planner: the capability flags derived
//! from installed runtimes, the device profile, and the reindex wait that
//! follows a profile change.

use std::time::Duration;

use pipette_device::detect_device_info;
use pipette_http::HttpClient;
use pipette_mgmt_client::{
    types::{ClientProfile, DeviceProfileFields, UpdateClientRequest},
    AuthIdentity, MgmtClient,
};
use pipette_plan_types::device::DeviceFormFactor;
use pipette_plan_types::Runtime;

use crate::error::{Error, Result};
use crate::identity::IdentityStore;

pub const REINDEX_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Cap on how long we'll wait for reindex before giving up and retrying the
/// claim loop anyway (the gate is server-side; this is just a local budget).
pub const REINDEX_WAIT_BUDGET: Duration = Duration::from_secs(5 * 60);

/// Capability flags for one installed runtime: the general `runtime:<name>`
/// flag plus a versioned `runtime:<name>:<build>` when a build id is known.
/// Matching is exact, so both levels must be advertised (client-integration §2).
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
        // The device is not in the flag: it is a per-cell choice over one
        // installed venv, so what this box has is `openvino:<version>` and the
        // plan picks CPU/GPU/NPU on top of it.
        Runtime::UvOpenvino(rt) => ("openvino", Some(rt.server_version.as_ref().to_string())),
        // iOS MLX is pinned by a Swift-package stack; advertise the mlx-swift ref.
        // The iOS app also reports package-level flags this function does not
        // derive — `runtime:mlx:mlx-swift=`, `:mlx-swift-lm=`,
        // `:swift-transformers=` (`Capabilities.swift`). A plan may pin to
        // those; only iOS will match.
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
        Runtime::DockerVllm(rt) => ("docker_vllm", Some(rt.image_tag.as_ref().to_string())),
        Runtime::DockerSglang(rt) => ("docker_sglang", Some(rt.image_tag.as_ref().to_string())),
        Runtime::UvVllm(rt) => ("uv_vllm", Some(rt.runtime_version())),
        Runtime::UvSglang(rt) => ("uv_sglang", Some(rt.runtime_version())),
        Runtime::AppleFoundation(_) => ("apple_foundation", None),
    };
    let general = format!("runtime:{name}");
    match version {
        Some(v) if !v.is_empty() => {
            // Capability flags must be lowercase with no whitespace.
            let build = v.to_ascii_lowercase().replace(char::is_whitespace, "");
            vec![general.clone(), format!("{general}:{build}")]
        }
        _ => vec![general],
    }
}

/// Union of capability flags across every installed runtime in the workspace.
///
/// No `job_schema:<n>` flag is reported. The mechanism exists — a job written to
/// revision *n* carries `job_schema:<n>` in its `requires`, so a client that
/// cannot parse that revision never matches it (`pipette-mgmt`
/// `docs/plan-ingestion.md` §7) — but no plan emits one, and a flag no job names
/// does nothing (`client-integration.md`, "What earns a flag"). Add it on both
/// sides in the same change that starts emitting it.
pub fn installed_runtime_capabilities(
    runtimes: &pipette_artifacts::runtime::RuntimeArtifactStore,
) -> Result<Vec<String>> {
    let manifests = runtimes.list().map_err(|e| Error::Other(e.into()))?;
    let flags: std::collections::BTreeSet<_> = manifests
        .iter()
        .flat_map(|m| runtime_capability_flags(&m.declared))
        .collect();
    Ok(flags.into_iter().collect())
}

/// Build a `PATCH /clients/me` body from the local device labels + detected
/// hardware + installed-runtime capabilities.
fn build_profile_update(
    identity: &IdentityStore,
    capabilities: Vec<String>,
) -> Result<UpdateClientRequest> {
    let labels = identity.get_device_labels()?;
    let device = detect_device_info(
        labels.device_name.as_ref().map(AsRef::as_ref),
        labels.device_form_factor,
    )
    .map_err(Error::Other)?;
    Ok(UpdateClientRequest {
        client_details: None,
        device: DeviceProfileFields {
            device_name: Some(device.device_name.as_ref().to_string()),
            device_form_factor: Some(form_factor_wire(device.device_form_factor).to_string()),
            device_os_name: Some(device.device_os_name.as_ref().to_string()),
            device_os_version: Some(device.device_os_version.as_ref().to_string()),
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
    })
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

/// Refresh the server-side device profile + capabilities at startup. When the
/// response has `reindex_pending: true`, poll `GET /clients/me` until the gate
/// lifts (or the budget expires). Returns the final profile.
pub fn refresh_profile_at_startup(
    identity: &IdentityStore,
    client: &MgmtClient,
    http: &HttpClient,
    capabilities: Vec<String>,
) -> Result<ClientProfile> {
    let auth = identity.signing_identity()?;
    let request = build_profile_update(identity, capabilities)?;
    log::info!(
        "refreshing device profile ({} capability flag(s))",
        request
            .device
            .capabilities
            .as_ref()
            .map(|c| c.len())
            .unwrap_or(0)
    );
    let profile = client.update_me(http, &auth, request)?;
    if !profile.reindex_pending {
        return Ok(profile);
    }
    log::info!(
        "reindex_pending=true after profile update; waiting for queue-maintenance \
         (poll every {}s, budget {}s)",
        REINDEX_POLL_INTERVAL.as_secs(),
        REINDEX_WAIT_BUDGET.as_secs()
    );
    wait_for_reindex(client, http, &auth)
}

/// Poll `GET /clients/me` until `reindex_pending` is false or the budget is
/// exhausted. On budget expiry returns the last profile (still pending) so the
/// caller can decide — claim will simply return `204` until the gate lifts.
fn wait_for_reindex(
    client: &MgmtClient,
    http: &HttpClient,
    auth: &AuthIdentity,
) -> Result<ClientProfile> {
    let deadline = std::time::Instant::now() + REINDEX_WAIT_BUDGET;
    loop {
        std::thread::sleep(REINDEX_POLL_INTERVAL);
        let profile = client.me(http, auth)?;
        if !profile.reindex_pending {
            log::info!("reindex gate lifted");
            return Ok(profile);
        }
        if std::time::Instant::now() >= deadline {
            log::warn!(
                "reindex still pending after {}s; continuing (claim will idle until gate lifts)",
                REINDEX_WAIT_BUDGET.as_secs()
            );
            return Ok(profile);
        }
    }
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        default_repository_url, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource,
        NonEmptyString, SourceRepository,
    };

    use super::*;

    #[test]
    fn runtime_capability_flags_llama_general_and_versioned() -> anyhow::Result<()> {
        let rt = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b9050")?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let flags = runtime_capability_flags(&rt);
        assert_eq!(
            flags,
            vec![
                "runtime:llama_cpp".to_string(),
                "runtime:llama_cpp:b9050".to_string()
            ]
        );
        Ok(())
    }
}
