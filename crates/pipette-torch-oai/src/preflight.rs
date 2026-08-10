//! Cheap host-side checks that run before any docker pull / uv install.
//!
//! Dispatches on the runtime's per-server flavor so a host with a
//! broken driver fails fast — before the 10 GB image pull or
//! multi-minute `uv pip install` burns on a runtime that won't be able
//! to launch. Cost: one `nvidia-smi` / `rocm-smi` invocation (~50 ms
//! when healthy) plus a couple of fstat calls on AMD.
//!
//! * `NvidiaGpu` — `nvidia-smi -L` + NVML probe.
//! * `AmdGpu`    — `rocm-smi --version` + `/dev/kfd` +
//!   `/dev/dri/renderD*` accessibility check.
//! * `Cpu`       — no-op (no GPU to probe).
//!
//! Host-side only. Whether the *installed* torch can reach the GPU it found is
//! the uv installer's question, asked once per install rather than per cell.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::{flavor_from_uv_build, Runtime, SglangFlavor, VllmFlavor};

use crate::flavor;

/// Confirm the host GPU stack is healthy for the runtime's recorded
/// flavor. Called before `docker pull` / `uv pip install` so a broken
/// driver fails before any heavy I/O. Matches on plan [`Runtime`]
/// flavor fields — no parallel engine type.
///
/// The runtimes with no torch flavor to check are listed rather than caught by
/// a catch-all, so a new GPU-backed runtime has to answer here instead of
/// silently acquiring a no-op. Total so the ensure path can call it without
/// re-deciding which runtimes are torch-shaped.
pub fn assert_gpu_ready(runtime: &Runtime) -> anyhow::Result<()> {
    match runtime {
        Runtime::DockerVllm(d) => assert_gpu_ready_vllm(d.flavor),
        Runtime::DockerSglang(d) => assert_gpu_ready_sglang(d.flavor),
        Runtime::UvVllm(u) => assert_gpu_ready_vllm(flavor_from_uv_build(&u.build)),
        Runtime::UvSglang(u) => assert_gpu_ready_sglang(flavor::vllm_to_sglang_flavor(
            flavor_from_uv_build(&u.build),
        )),
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        // OpenVINO talks to Intel CPU/iGPU/NPU through its own runtime, not
        // through torch, so none of the CUDA/ROCm probes apply.
        | Runtime::UvOpenvino(_)
        | Runtime::AppleFoundation(_) => Ok(()),
    }
}

fn assert_gpu_ready_vllm(flavor: VllmFlavor) -> anyhow::Result<()> {
    match flavor {
        VllmFlavor::NvidiaGpu => assert_nvidia_ready(),
        VllmFlavor::AmdGpu => assert_amd_ready(),
        VllmFlavor::Cpu => Ok(()),
    }
}

fn assert_gpu_ready_sglang(flavor: SglangFlavor) -> anyhow::Result<()> {
    match flavor {
        SglangFlavor::NvidiaGpu => assert_nvidia_ready(),
        SglangFlavor::AmdGpu => assert_amd_ready(),
        SglangFlavor::Cpu => Ok(()),
    }
}

/// Run `nvidia-smi -L` to confirm NVML is healthy.
///
/// - `nvidia-smi` missing from `PATH` → warn-but-continue. Docker will
///   surface its own "couldn't find nvidia hook" error if it later
///   matters. Some hosts intentionally don't ship the CLI.
/// - `nvidia-smi -L` exits non-zero → bail with the verbatim stderr
///   plus a concrete fix.
pub fn assert_nvidia_ready() -> anyhow::Result<()> {
    let nvidia_smi = match pipette_subprocess::which("nvidia-smi") {
        Ok(p) => p,
        Err(_) => {
            log::warn!(
                "nvidia-smi not on PATH; the container's NVIDIA prestart hook may fail. \
                 Install the NVIDIA driver or use a CPU-only runtime."
            );
            return Ok(());
        }
    };
    let mut cmd = Command::new(&nvidia_smi);
    cmd.arg("-L");
    pipette_subprocess::echo_debug(&cmd);
    let output = match cmd.output() {
        Ok(o) => o,
        Err(err) => {
            log::warn!("couldn't spawn `nvidia-smi -L` ({err}); skipping GPU pre-flight");
            return Ok(());
        }
    };
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    anyhow::bail!("{}", nvml_failure_message(&stderr));
}

/// Render the actionable NVML failure error. Factored out for unit
/// testing independent of an actual nvidia-smi failure.
fn nvml_failure_message(nvidia_smi_stderr: &str) -> String {
    format!(
        "GPU preflight failed: `nvidia-smi -L` fails on this host:\n\n\
         {nvidia_smi_stderr}\n\n\
         This usually means the loaded kernel module and the installed userspace\n\
         library are out of sync (a routine package update bumped the userspace\n\
         lib past the still-loaded module). Fix on the host:\n\n  \
           sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia\n  \
           sudo modprobe nvidia && sudo modprobe nvidia_uvm\n  \
           nvidia-smi -L     # should now list the GPU\n\n\
         …or `sudo reboot`. For CPU-only execution, install a `+cpu` runtime."
    )
}

/// AMD ROCm host check: `rocm-smi --version` succeeds + the kernel
/// fusion driver (`/dev/kfd`) and at least one render node
/// (`/dev/dri/renderD*`) exist and are accessible.
///
/// - `rocm-smi` missing from `PATH` → warn-but-continue. The user may
///   have ROCm installed without the CLI bundled; the `/dev/kfd` and
///   render-node checks below still have to pass.
/// - `/dev/kfd` missing → bail (kernel driver isn't loaded; nothing
///   else will work). Include the standard ROCm install URL in the
///   message so operators have a one-link path to a fix.
/// - `/dev/kfd` exists but not accessible (no `video`/`render` group
///   membership) → bail with the `usermod -a -G` hint.
pub fn assert_amd_ready() -> anyhow::Result<()> {
    // rocm-smi version probe — warn-and-continue on missing/broken so
    // headless servers without the CLI installed don't bail just to
    // print a version banner.
    match pipette_subprocess::which("rocm-smi") {
        Ok(rocm_smi) => {
            let mut cmd = Command::new(&rocm_smi);
            cmd.arg("--version");
            pipette_subprocess::echo_debug(&cmd);
            match cmd.output() {
                Ok(out) if out.status.success() => {
                    log::debug!(
                        "rocm-smi --version: {}",
                        String::from_utf8_lossy(&out.stdout).trim()
                    );
                }
                Ok(out) => {
                    log::warn!(
                        "rocm-smi --version failed ({}): {}",
                        out.status,
                        String::from_utf8_lossy(&out.stderr).trim()
                    );
                }
                Err(err) => {
                    log::warn!(
                        "couldn't spawn `rocm-smi --version` ({err}); skipping version probe"
                    );
                }
            }
        }
        Err(_) => {
            log::warn!(
                "rocm-smi not on PATH; ROCm CLI utilities may be missing. \
                 The /dev/kfd and render-node checks still run."
            );
        }
    }

    let kfd = Path::new("/dev/kfd");
    if !kfd.exists() {
        anyhow::bail!("{}", kfd_missing_message());
    }
    // Existence isn't enough — the process needs read/write access. The
    // most common failure on a fresh host is "user not in the `video` /
    // `render` group"; check via O_RDWR open and surface the fix.
    if let Err(err) = std::fs::OpenOptions::new().read(true).write(true).open(kfd) {
        anyhow::bail!("{}", kfd_access_message(&err.to_string()));
    }

    let renders = find_render_nodes()?;
    if renders.is_empty() {
        anyhow::bail!("{}", render_node_missing_message());
    }
    log::debug!(
        "amd preflight: /dev/kfd ok; {} render node(s) at {}",
        renders.len(),
        renders
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn find_render_nodes() -> anyhow::Result<Vec<PathBuf>> {
    let dri = Path::new("/dev/dri");
    if !dri.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in
        std::fs::read_dir(dri).with_context(|| format!("failed to read {}", dri.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        if name_str.starts_with("renderD") {
            out.push(entry.path());
        }
    }
    out.sort();
    Ok(out)
}

fn kfd_missing_message() -> String {
    "GPU preflight failed: /dev/kfd is missing.\n\n\
     The amdkfd kernel driver isn't loaded. Install ROCm and reboot:\n\n  \
       https://rocm.docs.amd.com/projects/install-on-linux/en/latest/\n\n\
     For CPU-only execution, install a `+cpu` runtime."
        .to_string()
}

fn kfd_access_message(open_error: &str) -> String {
    format!(
        "GPU preflight failed: /dev/kfd exists but is not accessible:\n\n\
         {open_error}\n\n\
         This is almost always a group-membership problem. Add your user\n\
         to the `video` and `render` groups, then log out and back in:\n\n  \
           sudo usermod -a -G video,render $USER\n\n\
         (`render` is required on most modern distros; `video` is the\n\
         legacy group some distros still use.)"
    )
}

fn render_node_missing_message() -> String {
    "GPU preflight failed: no /dev/dri/renderD* nodes exist.\n\n\
     The ROCm DRM driver isn't exposing render nodes. Confirm `lsmod | grep amdgpu`\n\
     shows the driver loaded; re-run `sudo modprobe amdgpu` if not. If the GPU\n\
     model is unsupported (consumer Navi/Vega without ROCm support), use a\n\
     `+cpu` runtime instead."
        .to_string()
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{
        AbsolutePath, UvBuild, UvPythonVersion, UvRuntimeSource, UvServerVersion, UvVllm,
    };

    use super::*;

    fn uv_vllm_runtime(build: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.21.0".to_string())?,
            build: UvBuild::try_new(build.to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_string())?,
            },
        }))
    }

    #[test]
    fn nvml_failure_message_quotes_stderr() {
        let stderr = "Failed to initialize NVML: Driver/library version mismatch\n\
                      NVML library version: 570.207";
        let msg = nvml_failure_message(stderr);
        assert!(msg.contains("Failed to initialize NVML: Driver/library version mismatch"));
        assert!(msg.contains("NVML library version: 570.207"));
    }

    #[test]
    fn nvml_failure_message_includes_fix_commands() {
        let msg = nvml_failure_message("any");
        assert!(msg.contains("sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia"));
        assert!(msg.contains("sudo modprobe nvidia"));
        assert!(msg.contains("nvidia-smi -L"));
    }

    #[test]
    fn nvml_failure_message_explains_root_cause() {
        let msg = nvml_failure_message("any");
        assert!(msg.contains("kernel module"));
        assert!(msg.contains("userspace"));
    }

    #[test]
    fn kfd_missing_message_points_at_install_docs() {
        let msg = kfd_missing_message();
        assert!(msg.contains("/dev/kfd"));
        assert!(msg.contains("rocm.docs.amd.com"));
    }

    #[test]
    fn kfd_access_message_suggests_usermod() {
        let msg = kfd_access_message("Permission denied");
        assert!(msg.contains("Permission denied"));
        assert!(msg.contains("usermod"));
        assert!(msg.contains("video"));
        assert!(msg.contains("render"));
    }

    #[test]
    fn render_node_missing_message_suggests_amdgpu() {
        let msg = render_node_missing_message();
        assert!(msg.contains("renderD"));
        assert!(msg.contains("amdgpu"));
    }

    #[test]
    fn assert_gpu_ready_cpu_is_noop() -> anyhow::Result<()> {
        // CPU runtimes must never touch GPU CLIs — this test would hang
        // (or error) if assert_gpu_ready accidentally fell into the
        // nvidia/amd branch.
        assert_gpu_ready(&uv_vllm_runtime("cpu")?)?;
        Ok(())
    }

    // The ensure path calls this for whatever runtime it was handed, so a
    // non-torch one has to pass rather than fail the install it precedes.
    #[test]
    fn assert_gpu_ready_passes_a_non_torch_runtime() -> anyhow::Result<()> {
        assert_gpu_ready(&Runtime::AppleFoundation(Default::default()))
    }
}
