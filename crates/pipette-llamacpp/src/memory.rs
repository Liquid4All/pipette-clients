//! How the peak-memory benchmarks run a llama.cpp tool: the watch for
//! `pipette_memprobe`'s sampler on each flavor, and the run's deadline.

use std::ffi::OsString;
use std::time::Duration;

use pipette_memprobe::{SupportedWatch, Watch, WatchError};
use pipette_plan_types::{LlamaCppFlavor, PositiveSeconds};

/// A peak-memory run, text or VL, when the cell sets no `bench_timeout_seconds`:
/// one prefill and its decode, under a minute on a small model.
const MAX_MEMORY_USAGE_TIMEOUT: Duration = Duration::from_secs(600);

/// Picks the OpenVINO backend's device; upstream runs on the CPU when it is
/// unset or empty.
const OPENVINO_DEVICE: &str = "GGML_OPENVINO_DEVICE";

/// Why a flavor's peak cannot be watched here.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum NoPeakWatchError {
    #[error(
        "flavor {flavor} is an operator's build, whose devices pipette does not know, so no watch \
         reads them; run this memory cell on a known flavor"
    )]
    Custom { flavor: LlamaCppFlavor },
    #[error(
        "nothing reads an Ascend NPU's memory (its driver bills none through DRM fdinfo), so the \
         peak of flavor {flavor} would be the host's alone; run this memory cell on a CPU or GPU \
         flavor"
    )]
    Ascend { flavor: LlamaCppFlavor },
    #[error(
        "{env}={device}: the OpenVINO backend falls back to the CPU on a device name OpenVINO does \
         not list, which a memory reading cannot tell from a GPU run; run this memory cell with \
         {env} unset or CPU",
        env = OPENVINO_DEVICE
    )]
    OpenvinoDevice { device: String },
    #[error("flavor {flavor}'s devices cannot be read here")]
    Unsupported {
        flavor: LlamaCppFlavor,
        source: WatchError,
    },
}

pub fn bench_timeout(pinned: Option<PositiveSeconds>) -> Duration {
    pinned.map_or(MAX_MEMORY_USAGE_TIMEOUT, |secs| {
        Duration::from_secs(secs.into_inner())
    })
}

/// How to watch a peak on `flavor`, whose tool launches with `env`, refused
/// before anything spawns when this host cannot read it. A Linux GPU flavor
/// reads what the DRM drivers bill the tool, which on an Intel iGPU sits
/// outside `VmHWM` (`docs/methodology/peak-memory-linux.md`), and what NVML
/// attributes to it on an NVIDIA GPU; a Windows one what WDDM bills it on its
/// GPU adapters.
pub fn peak(
    flavor: &LlamaCppFlavor,
    env: impl FnOnce(&str) -> Option<OsString>,
) -> Result<SupportedWatch, NoPeakWatchError> {
    let watch = match flavor {
        LlamaCppFlavor::MacosX64
        | LlamaCppFlavor::MacosArm64
        | LlamaCppFlavor::MacosArm64Kleidiai
        | LlamaCppFlavor::LinuxArm64Cpu
        | LlamaCppFlavor::LinuxX64Cpu
        | LlamaCppFlavor::LinuxS390xCpu
        | LlamaCppFlavor::AndroidArm64Cpu
        | LlamaCppFlavor::WindowsX64Cpu
        | LlamaCppFlavor::WindowsArm64Cpu => Watch::Peak,
        LlamaCppFlavor::LinuxX64Openvino | LlamaCppFlavor::WindowsX64Openvino => {
            openvino_device(env(OPENVINO_DEVICE))?;
            Watch::Peak
        }
        // Even at `-ngl 0` the package holds Adreno and Hexagon memory.
        LlamaCppFlavor::AndroidArm64Snapdragon | LlamaCppFlavor::LinuxArm64Snapdragon => {
            Watch::PeakWithAccelerators
        }
        // On Windows, PDH `\GPU Process Memory\Total Committed` is API-agnostic:
        // whatever the WDDM driver bills the process.
        LlamaCppFlavor::LinuxX64Vulkan
        | LlamaCppFlavor::LinuxArm64Vulkan
        | LlamaCppFlavor::LinuxX64Rocm
        | LlamaCppFlavor::LinuxX64SyclFp16
        | LlamaCppFlavor::LinuxX64SyclFp32
        | LlamaCppFlavor::LinuxX64Cuda128
        | LlamaCppFlavor::LinuxX64Cuda13
        | LlamaCppFlavor::LinuxArm64Cuda13
        | LlamaCppFlavor::WindowsX64Vulkan
        | LlamaCppFlavor::WindowsArm64Vulkan
        | LlamaCppFlavor::WindowsX64Hip
        | LlamaCppFlavor::WindowsX64Sycl
        | LlamaCppFlavor::WindowsX64Cuda124
        | LlamaCppFlavor::WindowsX64Cuda131
        | LlamaCppFlavor::WindowsX64Cuda13
        | LlamaCppFlavor::WindowsArm64Cuda13
        | LlamaCppFlavor::WindowsArm64OpenclAdreno => Watch::PeakWithGpu,
        LlamaCppFlavor::LinuxOpenEulerAarch64Ascend310p
        | LlamaCppFlavor::LinuxOpenEulerX86Ascend310p
        | LlamaCppFlavor::LinuxOpenEulerAarch64Ascend910bAclgraph
        | LlamaCppFlavor::LinuxOpenEulerX86Ascend910bAclgraph => {
            return Err(NoPeakWatchError::Ascend {
                flavor: flavor.clone(),
            });
        }
        LlamaCppFlavor::Custom(_) => {
            return Err(NoPeakWatchError::Custom {
                flavor: flavor.clone(),
            });
        }
    };
    watch
        .supported()
        .map_err(|source| NoPeakWatchError::Unsupported {
            flavor: flavor.clone(),
            source,
        })
}

/// Only the CPU, which upstream runs when `device` is unset or empty. On a name
/// OpenVINO does not list (a lone GPU is `GPU`, two are `GPU.0` and `GPU.1`)
/// the backend falls back to the CPU after its device query opened the GPU,
/// holding ~12 KiB there, so no reading tells that run from a GPU one.
fn openvino_device(device: Option<OsString>) -> Result<(), NoPeakWatchError> {
    let device = device.unwrap_or_default();
    if device.is_empty() || device == "CPU" {
        Ok(())
    } else {
        Err(NoPeakWatchError::OpenvinoDevice {
            device: device.to_string_lossy().into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `watch` where this OS's sampler reads it.
    fn on(flavor: &LlamaCppFlavor, watch: Watch) -> Result<SupportedWatch, NoPeakWatchError> {
        watch
            .supported()
            .map_err(|source| NoPeakWatchError::Unsupported {
                flavor: flavor.clone(),
                source,
            })
    }

    /// Each flavor's watch, the one that reads its devices, on every OS; a
    /// build for another OS fails when it launches.
    fn expected(flavor: &LlamaCppFlavor) -> Result<SupportedWatch, NoPeakWatchError> {
        match flavor {
            LlamaCppFlavor::MacosX64
            | LlamaCppFlavor::MacosArm64
            | LlamaCppFlavor::MacosArm64Kleidiai => on(flavor, Watch::Peak),
            LlamaCppFlavor::LinuxArm64Cpu
            | LlamaCppFlavor::LinuxX64Cpu
            | LlamaCppFlavor::LinuxS390xCpu
            | LlamaCppFlavor::AndroidArm64Cpu => on(flavor, Watch::Peak),
            LlamaCppFlavor::AndroidArm64Snapdragon | LlamaCppFlavor::LinuxArm64Snapdragon => {
                on(flavor, Watch::PeakWithAccelerators)
            }
            LlamaCppFlavor::LinuxX64Vulkan
            | LlamaCppFlavor::LinuxArm64Vulkan
            | LlamaCppFlavor::LinuxX64Rocm
            | LlamaCppFlavor::LinuxX64SyclFp16
            | LlamaCppFlavor::LinuxX64SyclFp32
            | LlamaCppFlavor::LinuxX64Cuda128
            | LlamaCppFlavor::LinuxX64Cuda13
            | LlamaCppFlavor::LinuxArm64Cuda13 => on(flavor, Watch::PeakWithGpu),
            // With `GGML_OPENVINO_DEVICE` unset.
            LlamaCppFlavor::LinuxX64Openvino | LlamaCppFlavor::WindowsX64Openvino => {
                on(flavor, Watch::Peak)
            }
            LlamaCppFlavor::LinuxOpenEulerAarch64Ascend310p
            | LlamaCppFlavor::LinuxOpenEulerX86Ascend310p
            | LlamaCppFlavor::LinuxOpenEulerAarch64Ascend910bAclgraph
            | LlamaCppFlavor::LinuxOpenEulerX86Ascend910bAclgraph => {
                Err(NoPeakWatchError::Ascend {
                    flavor: flavor.clone(),
                })
            }
            LlamaCppFlavor::WindowsX64Cpu | LlamaCppFlavor::WindowsArm64Cpu => {
                on(flavor, Watch::Peak)
            }
            LlamaCppFlavor::WindowsX64Vulkan
            | LlamaCppFlavor::WindowsArm64Vulkan
            | LlamaCppFlavor::WindowsX64Hip
            | LlamaCppFlavor::WindowsX64Sycl
            | LlamaCppFlavor::WindowsX64Cuda124
            | LlamaCppFlavor::WindowsX64Cuda131
            | LlamaCppFlavor::WindowsX64Cuda13
            | LlamaCppFlavor::WindowsArm64Cuda13
            | LlamaCppFlavor::WindowsArm64OpenclAdreno => on(flavor, Watch::PeakWithGpu),
            LlamaCppFlavor::Custom(_) => Err(NoPeakWatchError::Custom {
                flavor: flavor.clone(),
            }),
        }
    }

    #[test]
    fn a_peak_memory_cell_resolves_its_deadline() -> anyhow::Result<()> {
        assert_eq!(bench_timeout(None), MAX_MEMORY_USAGE_TIMEOUT);
        assert_eq!(
            bench_timeout(Some(PositiveSeconds::try_new(2400)?)),
            Duration::from_secs(2400)
        );
        Ok(())
    }

    #[test]
    fn each_flavor_has_its_watch() {
        LlamaCppFlavor::known()
            .chain([LlamaCppFlavor::Custom("my-build".to_owned())])
            .for_each(|flavor| {
                assert_eq!(peak(&flavor, |_| None), expected(&flavor), "{flavor}");
            });
    }

    #[rstest::rstest]
    #[case::unset(None, None)]
    #[case::empty(Some(""), None)]
    #[case::cpu(Some("CPU"), None)]
    #[case::gpu(Some("GPU"), Some("GPU"))]
    #[case::second_gpu(Some("GPU.1"), Some("GPU.1"))]
    #[case::npu(Some("NPU"), Some("NPU"))]
    #[case::auto(Some("AUTO"), Some("AUTO"))]
    fn the_openvino_flavor_watches_only_the_cpu(
        #[case] device: Option<&str>,
        #[case] refused: Option<&str>,
        #[values(LlamaCppFlavor::LinuxX64Openvino, LlamaCppFlavor::WindowsX64Openvino)]
        flavor: LlamaCppFlavor,
    ) {
        let env = |name: &str| {
            assert_eq!(name, OPENVINO_DEVICE);
            device.map(OsString::from)
        };
        let expected = match refused {
            Some(device) => Err(NoPeakWatchError::OpenvinoDevice {
                device: device.to_owned(),
            }),
            None => on(&flavor, Watch::Peak),
        };
        assert_eq!(peak(&flavor, env), expected, "{device:?}");
    }
}
