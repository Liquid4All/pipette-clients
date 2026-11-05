//! Cheap host-side checks that run before any docker pull / uv install, so a
//! broken driver fails before a 10 GB image pull or a multi-minute
//! `uv pip install`.
//!
//! * NVIDIA: `nvidia-smi -L` (NVML); the ONNX GenAI CUDA provider also needs a
//!   CUDA 13 driver.
//! * AMD: `rocm-smi --version`, plus `/dev/kfd` and `/dev/dri/renderD*` access.
//! * TensorRT Edge-LLM: a Jetson with the flavor's SoC, by its device tree.
//! * Everything else: no-op.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Context;

use pipette_plan_types::{
    OnnxExecutionProvider, Runtime, SglangFlavor, TrtEdgellmFlavor, VllmFlavor,
    flavor_from_uv_build,
};
use pipette_programs::{ExternalPrograms, ResolvedProgram};

/// Confirm the host GPU stack is healthy for the runtime. No catch-all arm, so
/// a new GPU-backed runtime has to answer here instead of acquiring a no-op.
pub fn assert_gpu_ready(runtime: &Runtime, programs: &ExternalPrograms) -> anyhow::Result<()> {
    match runtime {
        Runtime::DockerVllm(d) => assert_vllm_flavor_ready(d.flavor, programs),
        Runtime::DockerSglang(d) => assert_sglang_flavor_ready(d.flavor, programs),
        // A uv build tag reads as a `VllmFlavor` for either server.
        Runtime::UvVllm(u) => assert_vllm_flavor_ready(flavor_from_uv_build(&u.build), programs),
        Runtime::UvSglang(u) => assert_vllm_flavor_ready(flavor_from_uv_build(&u.build), programs),
        Runtime::UvOnnxGenai(u) => match u.execution_provider {
            OnnxExecutionProvider::Cuda => {
                assert_nvidia_ready(&programs.nvidia_smi)?;
                assert_cuda_13_driver(&programs.nvidia_smi)
            }
            OnnxExecutionProvider::Cpu
            | OnnxExecutionProvider::DirectMl
            | OnnxExecutionProvider::WinMl
            | OnnxExecutionProvider::Custom(_) => Ok(()),
        },
        // The archive is a JetPack build; nvidia-smi on a Jetson reports no
        // driver version to check.
        Runtime::TrtEdgellm(t) => assert_jetson(&t.flavor, Path::new(DEVICE_TREE_COMPATIBLE)),
        Runtime::LlamacppCliStockTools(_)
        | Runtime::LlamacppApkPipette(_)
        | Runtime::LlamacppIosPipette(_)
        | Runtime::MlxMacosPipette(_)
        | Runtime::MlxIosPipette(_)
        // Runs on the phone's CPU: no host GPU stack to probe.
        | Runtime::OnnxGenaiIosPipette(_)
        // OpenVINO (and OVMS, which bundles it) reaches Intel devices through
        // its own runtime, so no CUDA/ROCm probe applies.
        | Runtime::UvOpenvino(_)
        | Runtime::Ovms(_)
        | Runtime::AppleFoundation(_)
        | Runtime::AppleCoreAiMacosPipette(_) => Ok(()),
    }
}

/// The SoC list a Jetson's device tree carries, NUL-separated
/// (`nvidia,p3737-0000+p3701-0005\0nvidia,p3701-0005\0nvidia,tegra234\0` on an
/// AGX Orin).
const DEVICE_TREE_COMPATIBLE: &str = "/proc/device-tree/compatible";

/// A host whose device tree names the flavor's SoC, or any `nvidia,tegra*` one
/// for a custom flavor: the engines build against JetPack's TensorRT and the
/// Jetson's iGPU, so a cell refused here installs nothing.
fn assert_jetson(flavor: &TrtEdgellmFlavor, compatible: &Path) -> anyhow::Result<()> {
    // A Thor (`nvidia,tegra264`) is refused an Orin flavor.
    let (soc, exact) = match flavor {
        TrtEdgellmFlavor::JetsonOrinCuda13 => ("nvidia,tegra234", true),
        TrtEdgellmFlavor::Custom(_) => ("nvidia,tegra", false),
    };
    let admits = |entry: &[u8]| {
        if exact {
            entry == soc.as_bytes()
        } else {
            entry.starts_with(soc.as_bytes())
        }
    };
    let refused = |why: &str| {
        anyhow::anyhow!(
            "TensorRT Edge-LLM preflight failed: {why}; its archives are JetPack builds for a \
             Jetson (NVIDIA Tegra) board"
        )
    };
    match std::fs::read(compatible) {
        Ok(bytes) if bytes.split(|&b| b == 0).any(admits) => Ok(()),
        Ok(_) => Err(refused(&format!(
            "{} names no {soc} SoC, which flavor `{flavor}` needs",
            compatible.display()
        ))),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Err(refused(&format!(
            "{} does not exist, so this host has no device tree",
            compatible.display()
        ))),
        Err(err) => Err(err).with_context(|| format!("reading {}", compatible.display())),
    }
}

fn assert_vllm_flavor_ready(flavor: VllmFlavor, programs: &ExternalPrograms) -> anyhow::Result<()> {
    match flavor {
        VllmFlavor::NvidiaGpu => assert_nvidia_ready(&programs.nvidia_smi),
        VllmFlavor::AmdGpu => assert_amd_ready(&programs.rocm_smi),
        VllmFlavor::Cpu => Ok(()),
    }
}

fn assert_sglang_flavor_ready(
    flavor: SglangFlavor,
    programs: &ExternalPrograms,
) -> anyhow::Result<()> {
    match flavor {
        SglangFlavor::NvidiaGpu => assert_nvidia_ready(&programs.nvidia_smi),
        SglangFlavor::AmdGpu => assert_amd_ready(&programs.rocm_smi),
        SglangFlavor::Cpu => Ok(()),
    }
}

/// Run `nvidia-smi -L` to confirm NVML is healthy. A missing `nvidia-smi` only
/// warns: some hosts ship no CLI, and Docker reports its own missing-hook error.
fn assert_nvidia_ready(nvidia_smi: &ResolvedProgram) -> anyhow::Result<()> {
    let Ok(nvidia_smi) = nvidia_smi.path() else {
        tracing::warn!(
            "nvidia-smi not on PATH; the container's NVIDIA prestart hook may fail. \
             Install the NVIDIA driver or use a CPU-only runtime"
        );
        return Ok(());
    };
    let mut cmd = Command::new(nvidia_smi);
    cmd.arg("-L");
    pipette_subprocess::echo_debug(&cmd);
    let output = match cmd.output() {
        Ok(o) => o,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "couldn't spawn `nvidia-smi -L`; skipping GPU pre-flight"
            );
            return Ok(());
        }
    };
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    anyhow::bail!("{}", nvml_failure_message(&stderr));
}

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

/// The oldest NVIDIA driver the bundled CUDA catalog row runs on: it pins CUDA 13
/// libraries, which need R580. The host's toolkit does not matter (the venv
/// carries the libraries): the verified H100 ran 580.105.08 over toolkit 12.8.
const MIN_CUDA_13_DRIVER_MAJOR: u32 = 580;

/// Confirm the host driver reaches CUDA 13, for the ONNX GenAI CUDA provider.
/// Not in [`assert_nvidia_ready`]: a vLLM `cu121` build runs on an older driver.
///
/// An absent `nvidia-smi` is fatal here: no Docker hook fails later, and
/// `pipette-device` advertises `accel:gpu` from `lspci` alone, so a driverless
/// host would install about 1.2 GB and fail at model load. An unreadable
/// version only warns: the floor just moves that failure earlier.
fn assert_cuda_13_driver(nvidia_smi: &ResolvedProgram) -> anyhow::Result<()> {
    let Ok(nvidia_smi) = nvidia_smi.path() else {
        anyhow::bail!("{}", nvidia_smi_missing_message());
    };
    let mut cmd = Command::new(nvidia_smi);
    cmd.args(["--query-gpu=driver_version", "--format=csv,noheader"]);
    pipette_subprocess::echo_debug(&cmd);
    let reported = match cmd.output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => {
            tracing::warn!(
                exit_status = %output.status,
                "`nvidia-smi --query-gpu=driver_version` failed; leaving the CUDA 13 driver \
                 floor unchecked"
            );
            return Ok(());
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "couldn't spawn `nvidia-smi --query-gpu=driver_version`; leaving the \
                 CUDA 13 driver floor unchecked"
            );
            return Ok(());
        }
    };
    let Some(major) = driver_major(&reported) else {
        tracing::warn!(
            stdout = reported.as_str(),
            "couldn't read a driver version out of nvidia-smi's output; leaving the CUDA 13 \
             driver floor unchecked"
        );
        return Ok(());
    };
    anyhow::ensure!(
        major >= MIN_CUDA_13_DRIVER_MAJOR,
        "{}",
        cuda_13_driver_message(reported.trim())
    );
    tracing::debug!(driver_major = major, "driver reaches CUDA 13");
    Ok(())
}

/// The major version of the first card's driver: a mixed-driver host is not a
/// configuration pipette targets.
fn driver_major(reported: &str) -> Option<u32> {
    reported
        .lines()
        .next()?
        .trim()
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn nvidia_smi_missing_message() -> String {
    format!(
        "GPU preflight failed: `nvidia-smi` is not on PATH, so this host cannot prove it\n\
         has a working NVIDIA driver.\n\n\
         The ONNX GenAI CUDA runtime needs driver {MIN_CUDA_13_DRIVER_MAJOR} or newer. A host \
         that reports a GPU\n\
         through `lspci` alone still fails at model load, after about 1.2 GB of install.\n\
         Fix on the host:\n\n  \
           sudo apt install -y nvidia-driver-{MIN_CUDA_13_DRIVER_MAJOR}   # or your distro's equivalent\n  \
           sudo reboot\n  \
           nvidia-smi -L     # should now list the GPU\n\n\
         For CPU-only execution, use the `ep=cpu` catalog row."
    )
}

fn cuda_13_driver_message(reported: &str) -> String {
    format!(
        "GPU preflight failed: this host's NVIDIA driver is {reported}, and the ONNX GenAI\n\
         CUDA runtime needs {MIN_CUDA_13_DRIVER_MAJOR} or newer.\n\n\
         The bundled CUDA catalog row installs CUDA 13 libraries, which an older driver\n\
         cannot load. Without this check the install downloads about 1.2 GB and then\n\
         fails at model load. Fix on the host:\n\n  \
           sudo apt install -y nvidia-driver-{MIN_CUDA_13_DRIVER_MAJOR}   # or your distro's equivalent\n  \
           sudo reboot\n  \
           nvidia-smi --query-gpu=driver_version --format=csv,noheader\n\n\
         The host's own CUDA toolkit version does not matter; the venv carries the\n\
         libraries it loads. For CPU-only execution, use the `ep=cpu` catalog row."
    )
}

/// AMD ROCm: `/dev/kfd` openable and at least one `/dev/dri/renderD*` node. A
/// missing or broken `rocm-smi` only warns: ROCm can be installed without it.
fn assert_amd_ready(rocm_smi: &ResolvedProgram) -> anyhow::Result<()> {
    match rocm_smi.path() {
        Ok(rocm_smi) => {
            let mut cmd = Command::new(rocm_smi);
            cmd.arg("--version");
            pipette_subprocess::echo_debug(&cmd);
            match cmd.output() {
                Ok(out) if out.status.success() => {
                    tracing::debug!(
                        stdout = String::from_utf8_lossy(&out.stdout).trim(),
                        "rocm-smi --version"
                    );
                }
                Ok(out) => {
                    tracing::warn!(
                        exit_status = %out.status,
                        stderr = String::from_utf8_lossy(&out.stderr).trim(),
                        "rocm-smi --version failed"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "couldn't spawn `rocm-smi --version`; skipping version probe"
                    );
                }
            }
        }
        Err(_) => {
            tracing::warn!(
                "rocm-smi not on PATH; ROCm CLI utilities may be missing. \
                 The /dev/kfd and render-node checks still run"
            );
        }
    }

    let kfd = Path::new("/dev/kfd");
    if !kfd.exists() {
        anyhow::bail!("{}", kfd_missing_message());
    }
    // A fresh host's usual failure is a user outside the `video` / `render`
    // groups, which only a read/write open shows.
    if let Err(err) = std::fs::OpenOptions::new().read(true).write(true).open(kfd) {
        anyhow::bail!("{}", kfd_access_message(&err.to_string()));
    }

    let renders = find_render_nodes()?;
    if renders.is_empty() {
        anyhow::bail!("{}", render_node_missing_message());
    }
    tracing::debug!(render_nodes = ?renders, "/dev/kfd ok; render nodes present");
    Ok(())
}

fn find_render_nodes() -> anyhow::Result<Vec<PathBuf>> {
    let dri = Path::new("/dev/dri");
    if !dri.exists() {
        return Ok(Vec::new());
    }
    let mut out = std::fs::read_dir(dri)
        .with_context(|| format!("failed to read {}", dri.display()))?
        .filter_map(|entry| {
            entry
                .map(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("renderD"))
                        .then(|| e.path())
                })
                .transpose()
        })
        .collect::<std::io::Result<Vec<_>>>()?;
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
    use rstest::rstest;

    use pipette_plan_types::{
        AbsolutePath, UvBuild, UvOnnxGenai, UvPythonVersion, UvRuntimeSource, UvServerVersion,
        UvSglang, UvVllm,
    };

    use super::*;

    const AGX_ORIN: &[u8] = b"nvidia,p3737-0000+p3701-0005\0nvidia,p3701-0005\0nvidia,tegra234\0";
    /// Its SoC entry alone: the board entries before it do not matter here.
    const THOR: &[u8] = b"nvidia,tegra264\0";

    fn custom() -> TrtEdgellmFlavor {
        TrtEdgellmFlavor::Custom("private".to_owned())
    }

    #[rstest]
    #[case::orin_flavor_on_orin(TrtEdgellmFlavor::JetsonOrinCuda13, Some(AGX_ORIN), None)]
    #[case::orin_flavor_on_thor(
        TrtEdgellmFlavor::JetsonOrinCuda13,
        Some(THOR),
        Some("names no nvidia,tegra234 SoC, which flavor `jetson-orin-cuda-13` needs")
    )]
    #[case::custom_on_thor(custom(), Some(THOR), None)]
    #[case::raspberry_pi(
        custom(),
        Some(&b"raspberrypi,5-model-b\0brcm,bcm2712\0"[..]),
        Some("names no nvidia,tegra SoC")
    )]
    #[case::no_device_tree(TrtEdgellmFlavor::JetsonOrinCuda13, None, Some("does not exist"))]
    fn only_the_flavors_jetson_passes_the_trt_preflight(
        #[case] flavor: TrtEdgellmFlavor,
        #[case] compatible: Option<&[u8]>,
        #[case] refusal: Option<&str>,
    ) -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("compatible");
        if let Some(bytes) = compatible {
            std::fs::write(&path, bytes)?;
        }
        match (assert_jetson(&flavor, &path), refusal) {
            (Ok(()), None) => Ok(()),
            (Err(err), Some(refusal)) => {
                anyhow::ensure!(err.to_string().contains(refusal), "{err}");
                Ok(())
            }
            (got, want) => anyhow::bail!("expected {want:?}, got {got:?}"),
        }
    }

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

    fn uv_sglang_runtime(build: &str) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvSglang(UvSglang {
            server_version: UvServerVersion::try_new("0.5.12".to_string())?,
            build: UvBuild::try_new(build.to_string())?,
            python_version: UvPythonVersion::try_new("3.12".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_string())?,
            },
        }))
    }

    fn uv_onnx_genai_runtime(execution_provider: OnnxExecutionProvider) -> anyhow::Result<Runtime> {
        Ok(Runtime::UvOnnxGenai(UvOnnxGenai {
            server_version: UvServerVersion::try_new("0.16.0".to_string())?,
            execution_provider,
            python_version: UvPythonVersion::try_new("3.11".to_string())?,
            source: UvRuntimeSource::AbsolutePreinstalled {
                dir: AbsolutePath::try_new("/venv".to_string())?,
            },
        }))
    }

    #[rstest]
    #[case::nvml_quotes_stderr(
        nvml_failure_message(
            "Failed to initialize NVML: Driver/library version mismatch\n\
             NVML library version: 570.207"
        ),
        &[
            "Failed to initialize NVML: Driver/library version mismatch",
            "NVML library version: 570.207",
        ],
    )]
    #[case::nvml_fix_commands(
        nvml_failure_message("any"),
        &[
            "sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia",
            "sudo modprobe nvidia",
            "nvidia-smi -L",
        ],
    )]
    #[case::nvml_root_cause(nvml_failure_message("any"), &["kernel module", "userspace"])]
    #[case::nvidia_smi_missing(nvidia_smi_missing_message(), &["nvidia-smi", "580", "ep=cpu"])]
    #[case::cuda_13_driver(cuda_13_driver_message("570.207"), &["570.207", "580", "ep=cpu"])]
    #[case::kfd_missing(kfd_missing_message(), &["/dev/kfd", "rocm.docs.amd.com"])]
    #[case::kfd_access(
        kfd_access_message("Permission denied"),
        &["Permission denied", "usermod", "video", "render"],
    )]
    #[case::render_node_missing(render_node_missing_message(), &["renderD", "amdgpu"])]
    fn failure_message_names_the_cause_and_the_fix(#[case] msg: String, #[case] needles: &[&str]) {
        needles
            .iter()
            .for_each(|needle| assert!(msg.contains(needle), "{needle:?} missing from {msg:?}"));
    }

    #[rstest]
    #[case::single_card("580.105.08\n", Some(580))]
    #[case::two_cards("580.105.08\n580.105.08\n", Some(580))]
    #[case::older("570.207\n", Some(570))]
    #[case::padded("  580.105.08  \n", Some(580))]
    #[case::not_a_version("Failed to initialize NVML\n", None)]
    #[case::empty("", None)]
    fn driver_major_reads_the_first_reported_card(
        #[case] reported: &str,
        #[case] expected: Option<u32>,
    ) {
        assert_eq!(driver_major(reported), expected);
    }

    /// Only a wrong turn into the AMD probe fails here (no `/dev/kfd`); with
    /// `nvidia-smi` at a missing path the NVIDIA probe warns and passes.
    #[rstest]
    #[case::vllm_cpu(uv_vllm_runtime("cpu")?)]
    #[case::sglang_cpu(uv_sglang_runtime("cpu")?)]
    #[case::onnx_genai_cpu(uv_onnx_genai_runtime(OnnxExecutionProvider::Cpu)?)]
    #[case::apple_foundation(Runtime::AppleFoundation(Default::default()))]
    fn assert_gpu_ready_is_a_noop_without_a_gpu(#[case] runtime: Runtime) -> anyhow::Result<()> {
        assert_gpu_ready(&runtime, &ExternalPrograms::at_missing_paths())
    }
}
