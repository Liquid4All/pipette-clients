//! Per-platform device identity, power and thermal probes.

// Thermal types go by crate path: each per-OS reader names a different subset,
// and importing them would need a `cfg` per name.
use pipette_plan_types::{DeviceFormFactor, DeviceInfo, PowerReading, TrimmedString};

#[cfg(any(target_os = "android", test))]
pub(crate) mod android;

struct DetectedFields {
    device_name: Option<String>,
    device_form_factor: Option<DeviceFormFactor>,
    device_os_name: Option<String>,
    device_os_version: Option<String>,
    /// Precise OS build id, finer-grained than `device_os_version`.
    device_os_build: Option<String>,
    /// OS security-patch level (Android-only; `None` elsewhere).
    device_os_security_patch: Option<String>,
    device_chip_model: Option<String>,
    device_ram_bytes: Option<u64>,
    device_gpu_model: Option<String>,
    device_gpu_vram_bytes: Option<u64>,
}

/// The CPU architecture of the target that Rust built this binary for, in
/// Rust's spelling (`x86_64`, `aarch64`). The server maps it to the `arch:`
/// flag.
pub const CPU_ARCH: &str = std::env::consts::ARCH;

/// The operating system (OS) family of the target that Rust built this binary
/// for, in Rust's spelling (`linux`, `macos`, `windows`). The server maps it to
/// the `os_family:` flag.
pub const OS_FAMILY: &str = std::env::consts::OS;

/// Build a [`DeviceInfo`] from the detected OS and hardware, with
/// `device_name` and `device_form_factor` overriding detection when given.
/// Errors listing every required field that could not be detected.
pub fn detect_device_info(
    device_name: Option<&str>,
    device_form_factor: Option<DeviceFormFactor>,
) -> anyhow::Result<DeviceInfo> {
    let d = PlatformDetector::detect();

    let device_name = device_name
        .map(TrimmedString::from)
        .or_else(|| d.device_name.map(TrimmedString::from))
        .filter(|s| !s.as_ref().is_empty());
    let device_form_factor = device_form_factor.or(d.device_form_factor);

    let mut missing = Vec::new();
    let device_name = require(&mut missing, "device_name", device_name);
    let device_os_name = require(
        &mut missing,
        "device_os_name",
        d.device_os_name
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
    );
    let device_os_version = require(
        &mut missing,
        "device_os_version",
        d.device_os_version
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
    );
    let device_chip_model = require(
        &mut missing,
        "device_chip_model",
        d.device_chip_model
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
    );
    let device_ram_bytes = require(&mut missing, "device_ram_bytes", d.device_ram_bytes);

    if !missing.is_empty() {
        anyhow::bail!(
            "device detection failed for: {}; \
             use --device-name / --device-form-factor or check platform support",
            missing.join(", ")
        );
    }

    Ok(DeviceInfo {
        device_name,
        device_form_factor: device_form_factor.unwrap_or(DeviceFormFactor::Desktop),
        device_os_name,
        device_os_version,
        device_os_build: d
            .device_os_build
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
        device_os_security_patch: d
            .device_os_security_patch
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
        device_cpu_arch: Some(TrimmedString::from(CPU_ARCH)),
        device_os_family: Some(TrimmedString::from(OS_FAMILY)),
        device_chip_model,
        device_ram_bytes,
        device_gpu_model: d
            .device_gpu_model
            .map(TrimmedString::from)
            .filter(|s| !s.as_ref().is_empty()),
        device_gpu_vram_bytes: d.device_gpu_vram_bytes,
        device_npu_model: None,
        device_npu_vram_bytes: None,
    })
}

fn require<T: Default>(missing: &mut Vec<&'static str>, name: &'static str, value: Option<T>) -> T {
    match value {
        Some(v) => v,
        None => {
            missing.push(name);
            T::default()
        }
    }
}

// `std::process::Command` is fully qualified at each call site: the catch-all
// `detect()` (iOS) does not shell out, and an unused import fails the build.
struct PlatformDetector;

#[cfg(target_os = "macos")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        let model_name = Self::system_profiler_model_name();
        let form_factor = model_name.as_deref().map(|n| {
            if n.to_lowercase().contains("book") {
                DeviceFormFactor::Laptop
            } else {
                DeviceFormFactor::Desktop
            }
        });

        DetectedFields {
            device_name: model_name,
            device_form_factor: form_factor,
            device_os_name: cmd_output("sw_vers", &["-productName"]),
            device_os_version: cmd_output("sw_vers", &["-productVersion"]),
            device_os_build: cmd_output("sw_vers", &["-buildVersion"]),
            device_os_security_patch: None,
            device_chip_model: cmd_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
            device_ram_bytes: cmd_output("sysctl", &["-n", "hw.memsize"])
                .and_then(|s| s.parse().ok()),
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
        }
    }

    fn system_profiler_model_name() -> Option<String> {
        let output = std::process::Command::new("system_profiler")
            .arg("SPHardwareDataType")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        text.lines()
            .find_map(|line| line.trim().strip_prefix("Model Name:"))
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }
}

#[cfg(target_os = "linux")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        let (gpu_model, gpu_vram) = match Self::detect_nvidia_gpu() {
            (Some(model), vram) => (Some(model), vram),
            (None, _) => amd_gpu_from_sysfs(std::path::Path::new("/sys/class/drm"))
                .map_or((None, None), |(model, vram)| (Some(model), vram)),
        };
        // e.g. `6.8.0-45-generic`: the build, and the version when the distro
        // has no `VERSION_ID`.
        let kernel_release = cmd_output("uname", &["-r"]);

        DetectedFields {
            device_name: Self::detect_device_name(),
            device_form_factor: Self::detect_form_factor(),
            device_os_name: Self::os_release_field("NAME").or_else(|| cmd_output("uname", &["-s"])),
            device_os_version: Self::os_release_field("VERSION_ID")
                .or_else(|| kernel_release.clone()),
            device_os_build: kernel_release.clone(),
            device_os_security_patch: None,
            device_chip_model: chip_model_or_board(cpuinfo_chip_model(), Self::device_tree_model())
                .or_else(Self::midr_chip_model)
                .or_else(Self::dmi_chip_model_fallback),
            device_ram_bytes: proc_memtotal_bytes(),
            device_gpu_model: gpu_model,
            device_gpu_vram_bytes: gpu_vram,
        }
    }

    fn detect_device_name() -> Option<String> {
        Self::dmi_field("product_name")
            .or_else(Self::device_tree_model)
            .or_else(|| cmd_output("hostname", &[]))
    }

    fn detect_form_factor() -> Option<DeviceFormFactor> {
        linux_form_factor(
            std::fs::read_to_string("/sys/devices/virtual/dmi/id/chassis_type")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok()),
            std::path::Path::new("/proc/device-tree").exists(),
            Self::dmi_field("product_name").as_deref(),
        )
    }

    fn dmi_field(field: &str) -> Option<String> {
        std::fs::read_to_string(format!("/sys/devices/virtual/dmi/id/{field}"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// The only chip identity GB10 (DGX Spark) boards expose. Silicon, so it
    /// outranks the DMI board name.
    fn midr_chip_model() -> Option<String> {
        let cores = parse_cpuinfo_arm_cores(&std::fs::read_to_string("/proc/cpuinfo").ok()?)?;
        tracing::debug!("device_chip_model resolved from /proc/cpuinfo MIDR fields {cores:?}");
        Some(cores)
    }

    /// Last resort, as a missing chip model fails [`detect_device_info`], so a SKU
    /// like `EdgeXpert (MS-C931)` here is expected. Placeholders are rejected so
    /// junk never becomes a fleet's chip model.
    fn dmi_chip_model_fallback() -> Option<String> {
        let board = Self::dmi_field("board_name")?;
        if is_dmi_placeholder(&board) {
            tracing::debug!("device_chip_model: ignoring DMI placeholder board_name {board:?}");
            return None;
        }
        tracing::debug!("device_chip_model resolved from DMI board_name {board:?}");
        Some(board)
    }

    /// The board name on hosts with no DMI (SBCs, most ARM boxes).
    /// Device-tree nodes are NUL-terminated.
    fn device_tree_model() -> Option<String> {
        let bytes = std::fs::read("/proc/device-tree/model").ok()?;
        let model = String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        (!model.is_empty()).then_some(model)
    }

    fn os_release_field(key: &str) -> Option<String> {
        let content = std::fs::read_to_string("/etc/os-release").ok()?;
        let prefix = format!("{key}=");
        for line in content.lines() {
            if let Some(value) = line.strip_prefix(&prefix) {
                let v = value.trim_matches('"').to_string();
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
        None
    }

    fn detect_nvidia_gpu() -> (Option<String>, Option<u64>) {
        if let Some((name, vram)) = Self::detect_nvidia_gpu_via_nvidia_smi() {
            return (Some(name), vram);
        }
        // nvidia-smi is often broken when NVML drifts out of sync with the
        // driver; lspci names the card without a working driver, but not VRAM.
        (Self::detect_nvidia_gpu_via_lspci(), None)
    }

    fn detect_nvidia_gpu_via_nvidia_smi() -> Option<(String, Option<u64>)> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let raw = String::from_utf8(output.stdout).ok()?;
        let first = raw.lines().next()?.trim();
        let (name, mib_str) = first.split_once(',')?;
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let vram = mib_str
            .trim()
            .parse::<u64>()
            .ok()
            .map(|mib| mib * 1024 * 1024);
        Some((name.to_string(), vram))
    }

    fn detect_nvidia_gpu_via_lspci() -> Option<String> {
        let output = std::process::Command::new("lspci")
            .args(["-d", "10de:", "-nn"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let raw = String::from_utf8(output.stdout).ok()?;
        parse_lspci_nvidia_gpu(&raw)
    }
}

/// The first AMD GPU under `drm` (`/sys/class/drm`), read from the files
/// `amd-smi` reads: `product_name`, or `AMD GPU [1002:<device>]` without one,
/// and `mem_info_vram_total`.
///
/// Only GPU-shaped PCI classes count, so AMD's HDMI audio does not; `1200`
/// (processing accelerator) is what the Instinct cards report.
#[cfg(any(target_os = "linux", test))]
fn amd_gpu_from_sysfs(drm: &std::path::Path) -> Option<(String, Option<u64>)> {
    let read = |path: std::path::PathBuf| {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let mut cards: Vec<(u32, std::path::PathBuf)> = std::fs::read_dir(drm)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let index = name.to_str()?.strip_prefix("card")?.parse::<u32>().ok()?;
            Some((index, entry.path().join("device")))
        })
        .collect();
    cards.sort_unstable_by_key(|(index, _)| *index);
    cards.into_iter().find_map(|(_, device)| {
        if read(device.join("vendor")).as_deref() != Some("0x1002") {
            return None;
        }
        let class = read(device.join("class"))?;
        let gpu_shaped = ["0x0300", "0x0302", "0x0380", "0x1200"]
            .iter()
            .any(|prefix| class.starts_with(prefix));
        if !gpu_shaped {
            return None;
        }
        let model = read(device.join("product_name")).unwrap_or_else(|| {
            let id = read(device.join("device")).unwrap_or_default();
            format!("AMD GPU [1002:{}]", id.trim_start_matches("0x"))
        });
        let vram = read(device.join("mem_info_vram_total"))
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 0);
        Some((model, vram))
    })
}

/// The first NVIDIA GPU in `lspci -d 10de: -nn` output:
/// `07:00.0 3D controller [0302]: NVIDIA Corporation GA102GL [A10] [10de:2236] (rev a1)`
/// gives `NVIDIA Corporation GA102GL [A10]`. Only display and 3D controllers
/// match, so NVIDIA's HDMI audio and PCI bridges don't.
#[cfg(target_os = "linux")]
fn parse_lspci_nvidia_gpu(raw: &str) -> Option<String> {
    for line in raw.lines() {
        let is_gpu = line.contains("[0300]")
            || line.contains("[0302]")
            || line.contains("[0380]")
            || line.contains("VGA")
            || line.contains("3D controller")
            || line.contains("Display controller");
        if !is_gpu {
            continue;
        }
        let Some((_, after_class)) = line.split_once("]: ") else {
            continue;
        };
        let mut name = after_class.to_string();
        if let Some(i) = name.rfind(" [10de:") {
            name.truncate(i);
        }
        if let Some(i) = name.rfind(" (rev") {
            name.truncate(i);
        }
        let name = name.trim().to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

#[cfg(target_os = "android")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        DetectedFields {
            device_name: Self::getprop_first(&["ro.product.marketname", "ro.product.model"]),
            device_form_factor: Self::detect_form_factor(),
            device_os_name: Some("Android".to_string()),
            device_os_version: Self::getprop("ro.build.version.release"),
            device_os_build: Self::getprop("ro.build.version.incremental"),
            device_os_security_patch: Self::getprop("ro.build.version.security_patch"),
            device_chip_model: Self::detect_chip_model(),
            device_ram_bytes: proc_memtotal_bytes(),
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
        }
    }

    fn detect_form_factor() -> Option<DeviceFormFactor> {
        match Self::getprop("ro.build.characteristics").as_deref() {
            Some(c) if c.contains("tablet") => Some(DeviceFormFactor::Tablet),
            Some(c) if c.contains("phone") || c.contains("default") => {
                Some(DeviceFormFactor::Phone)
            }
            _ => Some(DeviceFormFactor::Phone),
        }
    }

    fn detect_chip_model() -> Option<String> {
        cpuinfo_chip_model().or_else(|| Self::getprop("ro.soc.model"))
    }

    fn getprop(key: &str) -> Option<String> {
        let output = std::process::Command::new("getprop")
            .arg(key)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
        if text.is_empty() { None } else { Some(text) }
    }

    fn getprop_first(keys: &[&str]) -> Option<String> {
        keys.iter().find_map(|k| Self::getprop(k))
    }
}

#[cfg(target_os = "android")]
pub(crate) fn soc_manufacturer() -> Option<String> {
    PlatformDetector::getprop("ro.soc.manufacturer")
}

#[cfg(target_os = "android")]
pub fn boot_reason() -> Option<String> {
    PlatformDetector::getprop("sys.boot.reason")
}

#[cfg(target_os = "windows")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        if let Some(d) = Self::detect_batched() {
            return d;
        }
        tracing::warn!("batched PowerShell detection failed; falling back to per-field detection");
        Self::detect_individual()
    }

    fn detect_batched() -> Option<DetectedFields> {
        let script = r#"
$os  = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor
$cs  = Get-CimInstance Win32_ComputerSystem
$enc = Get-CimInstance Win32_SystemEnclosure
$cv  = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$ver = $cv.DisplayVersion
$build = if ($null -ne $cv.CurrentBuild -and $null -ne $cv.UBR) { "$($cv.CurrentBuild).$($cv.UBR)" } else { '' }
function Normalize-AdapterName($name) {
    if (-not $name) { return '' }
    $clean = [string]$name
    $clean = $clean -replace '\(TM\)', ''
    $clean = $clean -replace '\bAMD\b', ''
    return (($clean -replace '[^A-Za-z0-9]', '').ToLowerInvariant())
}
function Get-AdapterRegistryVram($gpuName) {
    $gpuNorm = Normalize-AdapterName $gpuName
    $candidates = @()
    foreach ($key in Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Video' -Recurse -ErrorAction SilentlyContinue) {
        if ($key.PSChildName -ne '0000') { continue }
        $props = Get-ItemProperty $key.PSPath -ErrorAction SilentlyContinue
        $bytes = $props.'HardwareInformation.qwMemorySize'
        if (-not $bytes -or $bytes -le 0) { continue }
        $candidateBytes = [UInt64]$bytes
        $candidates += $candidateBytes
        $adapter = ''
        if ($props.'HardwareInformation.AdapterString') {
            $rawAdapter = $props.'HardwareInformation.AdapterString'
            if ($rawAdapter -is [byte[]]) {
                $adapter = [Text.Encoding]::Unicode.GetString($rawAdapter) -replace "`0", ''
            } else {
                $adapter = [string]$rawAdapter
            }
        }
        $adapterNorm = Normalize-AdapterName $adapter
        if ($gpuNorm -and $adapterNorm -and ($gpuNorm.Contains($adapterNorm) -or $adapterNorm.Contains($gpuNorm))) {
            return $candidateBytes
        }
    }
    if ($candidates.Count -eq 1) { return [UInt64]$candidates[0] }
    return 0
}
$gpu = Get-CimInstance Win32_VideoController |
       Where-Object { $_.AdapterRAM -gt 0 -and $_.Name -notmatch 'Virtual|Basic' } |
       Select-Object -First 1
$gpuName = if ($gpu) { $gpu.Name } else { '' }
$gpuVram = if ($gpu) {
    $registryVram = Get-AdapterRegistryVram $gpu.Name
    if ($registryVram -gt 0) { $registryVram } else { $gpu.AdapterRAM }
} else { 0 }
$chassis = if ($enc.ChassisTypes) { $enc.ChassisTypes[0] } else { 0 }
"$($os.Caption)`t$ver`t$($cpu.Name)`t$($cs.TotalPhysicalMemory)`t$gpuName`t$gpuVram`t$($cs.Model)`t$chassis`t$build"
"#;
        let text = Self::powershell(script)?;
        let p: Vec<&str> = text.split('\t').collect();
        // An empty `$build` leaves a trailing tab that `powershell()` trims, so
        // the 9th field (OS build) may be missing.
        let [
            os_name,
            os_version,
            cpu,
            ram,
            gpu_name,
            gpu_vram,
            model,
            chassis,
            rest @ ..,
        ] = p.as_slice()
        else {
            return None;
        };
        let gpu_model = if gpu_name.is_empty() {
            None
        } else {
            Some(gpu_name.to_string())
        };
        let gpu_vram = if gpu_model.is_some() {
            gpu_vram.parse().ok().filter(|&v: &u64| v > 0)
        } else {
            None
        };
        Some(DetectedFields {
            device_name: Some(model.to_string()).filter(|s| !s.is_empty()),
            device_form_factor: chassis
                .trim()
                .parse::<u32>()
                .ok()
                .map(chassis_to_form_factor),
            device_os_name: Some(os_name.to_string()).filter(|s| !s.is_empty()),
            device_os_version: Some(os_version.to_string()).filter(|s| !s.is_empty()),
            // `CurrentBuild.UBR`, e.g. `26100.1234`.
            device_os_build: rest
                .first()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty()),
            device_os_security_patch: None,
            device_chip_model: Some(cpu.to_string()).filter(|s| !s.is_empty()),
            device_ram_bytes: ram.parse().ok(),
            device_gpu_model: gpu_model,
            device_gpu_vram_bytes: gpu_vram,
        })
    }

    fn detect_individual() -> DetectedFields {
        DetectedFields {
            device_name: Self::powershell("(Get-CimInstance Win32_ComputerSystem).Model"),
            device_form_factor: Self::powershell(
                "(Get-CimInstance Win32_SystemEnclosure).ChassisTypes[0]",
            )
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(chassis_to_form_factor),
            device_os_name: Self::powershell("(Get-CimInstance Win32_OperatingSystem).Caption"),
            device_os_version: Self::powershell(
                "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion').DisplayVersion",
            ),
            // Emits nothing unless both registry values are present, like the
            // batched path's `$build` guard.
            device_os_build: Self::powershell(
                "$cv = Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion'; if ($null -ne $cv.CurrentBuild -and $null -ne $cv.UBR) { \"$($cv.CurrentBuild).$($cv.UBR)\" }",
            ),
            device_os_security_patch: None,
            device_chip_model: Self::powershell("(Get-CimInstance Win32_Processor).Name")
                .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok()),
            device_ram_bytes: Self::powershell(
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            )
            .and_then(|s| s.parse().ok()),
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
        }
    }

    fn powershell(script: &str) -> Option<String> {
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
        if text.is_empty() { None } else { Some(text) }
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    target_os = "windows"
)))]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        DetectedFields {
            device_name: None,
            device_form_factor: None,
            device_os_name: None,
            device_os_version: None,
            device_os_build: None,
            device_os_security_patch: None,
            device_chip_model: None,
            device_ram_bytes: None,
            device_gpu_model: None,
            device_gpu_vram_bytes: None,
        }
    }
}

/// Run a command and return trimmed stdout, or `None` on failure/empty output.
#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_os = "android"
))]
fn cmd_output(cmd: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// What firmware ships in an unpopulated `board_name`, lower-case.
#[cfg(any(target_os = "linux", test))]
const DMI_PLACEHOLDERS: [&str; 12] = [
    "default string",
    "to be filled by o.e.m.",
    "to be filled by oem",
    "base board product name",
    "type2 - board product name",
    "not specified",
    "not applicable",
    "not available",
    "n/a",
    "none",
    "unknown",
    "undefined",
];

#[cfg(any(target_os = "linux", test))]
fn is_dmi_placeholder(value: &str) -> bool {
    DMI_PLACEHOLDERS.contains(&value.trim().to_ascii_lowercase().as_str())
}

/// Parse chip model from `/proc/cpuinfo` (`Hardware` or `model name` field).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn cpuinfo_chip_model() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for key in ["Hardware", "model name"] {
        for line in content.lines() {
            if let Some((k, v)) = line.split_once(':')
                && k.trim().eq_ignore_ascii_case(key)
            {
                let v = v.trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Jetson kernels print a generic `ARMv8 Processor` model that names no chip,
/// so the device-tree board name wins there.
#[cfg(any(target_os = "linux", test))]
fn chip_model_or_board(cpuinfo: Option<String>, device_tree: Option<String>) -> Option<String> {
    cpuinfo
        .filter(|model| !model.starts_with("ARMv8 Processor"))
        .or(device_tree)
}

/// Distinct core names in order of appearance, as `lscpu` lists them
/// (`Cortex-X925 + Cortex-A725`). `None` unless every core decodes, since a
/// partial list would misdescribe the chip.
#[cfg(any(target_os = "linux", test))]
fn parse_cpuinfo_arm_cores(cpuinfo: &str) -> Option<String> {
    let hex = |v: &str| u32::from_str_radix(v.strip_prefix("0x")?, 16).ok();
    let mut names: Vec<&str> = Vec::new();
    for block in cpuinfo.split("\n\n") {
        let field = |key: &str| {
            block.lines().find_map(|line| {
                let (k, v) = line.split_once(':')?;
                (k.trim() == key).then(|| v.trim())
            })
        };
        let (Some(implementer), Some(part)) = (field("CPU implementer"), field("CPU part")) else {
            continue;
        };
        let name = arm_core_name(hex(implementer)?, hex(part)?)?;
        if !names.contains(&name) {
            names.push(name);
        }
    }
    (!names.is_empty()).then(|| names.join(" + "))
}

/// Transcribed from util-linux `sys-utils/lscpu-arm.c` so names match `lscpu`.
/// ARM Ltd (`0x41`) only.
#[cfg(any(target_os = "linux", test))]
fn arm_core_name(implementer: u32, part: u32) -> Option<&'static str> {
    if implementer != 0x41 {
        return None;
    }
    Some(match part {
        0xd01 => "Cortex-A32",
        0xd02 => "Cortex-A34",
        0xd03 => "Cortex-A53",
        0xd04 => "Cortex-A35",
        0xd05 => "Cortex-A55",
        0xd06 => "Cortex-A65",
        0xd07 => "Cortex-A57",
        0xd08 => "Cortex-A72",
        0xd09 => "Cortex-A73",
        0xd0a => "Cortex-A75",
        0xd0b => "Cortex-A76",
        0xd0c => "Neoverse-N1",
        0xd0d => "Cortex-A77",
        0xd0e => "Cortex-A76AE",
        0xd40 => "Neoverse-V1",
        0xd41 => "Cortex-A78",
        0xd42 => "Cortex-A78AE",
        0xd43 => "Cortex-A65AE",
        0xd44 => "Cortex-X1",
        0xd46 => "Cortex-A510",
        0xd47 => "Cortex-A710",
        0xd48 => "Cortex-X2",
        0xd49 => "Neoverse-N2",
        0xd4a => "Neoverse-E1",
        0xd4b => "Cortex-A78C",
        0xd4c => "Cortex-X1C",
        0xd4d => "Cortex-A715",
        0xd4e => "Cortex-X3",
        0xd4f => "Neoverse-V2",
        0xd80 => "Cortex-A520",
        0xd81 => "Cortex-A720",
        0xd82 => "Cortex-X4",
        0xd83 => "Neoverse-V3AE",
        0xd84 => "Neoverse-V3",
        0xd85 => "Cortex-X925",
        0xd87 => "Cortex-A725",
        0xd88 => "Cortex-A520AE",
        0xd89 => "Cortex-A720AE",
        0xd8a => "C1-Nano",
        0xd8b => "C1-Pro",
        0xd8c => "C1-Ultra",
        0xd8e => "Neoverse-N3",
        0xd8f => "Cortex-A320",
        0xd90 => "C1-Premium",
        _ => return None,
    })
}

/// Parse `MemTotal` from `/proc/meminfo` and return bytes.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn proc_memtotal_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some((key, value)) = line.split_once(':')
            && key.trim() == "MemTotal"
        {
            let value = value.trim();
            let kb_str = value.strip_suffix("kB").unwrap_or(value).trim();
            if let Ok(kb) = kb_str.parse::<u64>() {
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// SMBIOS chassis 1 (`Other`) and 2 (`Unknown`) are the firmware declining to
/// say, so they yield to the device tree: a Jetson AGX Orin reports
/// `chassis_type = 1`. `None` falls back to `Desktop` in [`detect_device_info`].
///
/// The DGX Spark is a desktop unit, but its firmware declares chassis 17
/// (`Main Server Chassis`). For that unit, the DMI product name decides.
#[cfg(any(target_os = "linux", test))]
fn linux_form_factor(
    chassis: Option<u32>,
    has_device_tree: bool,
    product_name: Option<&str>,
) -> Option<DeviceFormFactor> {
    if chassis == Some(17) && product_name == Some("NVIDIA_DGX_Spark") {
        return Some(DeviceFormFactor::Desktop);
    }
    match chassis {
        None | Some(1 | 2) => has_device_tree.then_some(DeviceFormFactor::Embedded),
        Some(code) => Some(chassis_to_form_factor(code)),
    }
}

/// SMBIOS chassis type codes:
/// <https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-systemenclosure>
#[cfg(any(target_os = "windows", target_os = "linux", test))]
fn chassis_to_form_factor(chassis: u32) -> DeviceFormFactor {
    match chassis {
        8 | 9 | 10 | 14 => DeviceFormFactor::Laptop,
        30..=32 => DeviceFormFactor::Tablet,
        11 => DeviceFormFactor::Phone,
        17 | 23 | 28 => DeviceFormFactor::Server,
        33 | 34 => DeviceFormFactor::Embedded,
        _ => DeviceFormFactor::Desktop,
    }
}

/// The current power state, detected per submission: a laptop throttles on
/// battery or in low-power mode. Best-effort; every field degrades to `None`.
pub fn detect_power_state() -> PowerReading {
    #[cfg(target_os = "macos")]
    {
        macos_power_state()
    }
    #[cfg(target_os = "linux")]
    {
        linux_power_state()
    }
    #[cfg(target_os = "windows")]
    {
        windows_power_state()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        PowerReading::default()
    }
}

#[cfg(target_os = "macos")]
fn macos_power_state() -> PowerReading {
    let (battery_level, power_state) = cmd_output("pmset", &["-g", "batt"])
        .as_deref()
        .map(parse_pmset_batt)
        .unwrap_or((None, None));
    let power_save_mode = cmd_output("pmset", &["-g"])
        .as_deref()
        .map(macos_low_power_enabled);
    PowerReading {
        battery_level,
        power_state,
        power_save_mode,
    }
}

/// Parse `pmset -g batt`, whose battery line looks like
/// `" -InternalBattery-0 (id=…)\t83%; discharging; 3:11 remaining present: true"`.
/// A desktop Mac has no such line and is on external power
/// (`PluggedInNotCharging`), unlike a failed command (`None`).
#[cfg(target_os = "macos")]
fn parse_pmset_batt(out: &str) -> (Option<i32>, Option<pipette_plan_types::DevicePowerState>) {
    let Some(line) = out.lines().find(|l| l.contains('%')) else {
        return (
            None,
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging),
        );
    };
    let level = line
        .split('%')
        .next()
        .and_then(|p| {
            p.rsplit(|c: char| !c.is_ascii_digit())
                .find(|s| !s.is_empty())
        })
        .and_then(|d| d.parse::<i32>().ok())
        .filter(|n| (0..=100).contains(n));
    // "not charging" is plugged in but holding (Optimized Battery Charging).
    let power_state = line.split(';').nth(1).map(str::trim).and_then(|s| {
        if s.starts_with("discharging") {
            Some(pipette_plan_types::DevicePowerState::NotCharging)
        } else if s.starts_with("not charging") {
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
        } else if s.starts_with("charging") || s.starts_with("finishing charge") {
            Some(pipette_plan_types::DevicePowerState::Charging)
        } else if s.starts_with("charged") {
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
        } else {
            None
        }
    });
    (level, power_state)
}

/// `pmset -g` lists current settings; `lowpowermode 1` means Low Power Mode is on.
#[cfg(target_os = "macos")]
fn macos_low_power_enabled(pmset_g: &str) -> bool {
    pmset_g.lines().any(|l| {
        let l = l.trim();
        l.starts_with("lowpowermode") && l.split_whitespace().nth(1) == Some("1")
    })
}

#[cfg(target_os = "linux")]
fn linux_power_state() -> PowerReading {
    let base = std::path::Path::new("/sys/class/power_supply");
    let mut out = PowerReading::default();
    if let Ok(entries) = std::fs::read_dir(base) {
        let mut found_battery = false;
        for entry in entries.flatten() {
            let dir = entry.path();
            let is_battery = std::fs::read_to_string(dir.join("type"))
                .map(|t| t.trim() == "Battery")
                .unwrap_or(false);
            if !is_battery {
                continue;
            }
            found_battery = true;
            out.battery_level = std::fs::read_to_string(dir.join("capacity"))
                .ok()
                .and_then(|s| s.trim().parse::<i32>().ok())
                .filter(|n| (0..=100).contains(n));
            out.power_state = std::fs::read_to_string(dir.join("status"))
                .ok()
                .and_then(|s| map_linux_battery_status(s.trim()));
            break; // first battery wins
        }
        if !found_battery {
            // The supply dir exists but holds no battery: a desktop on AC,
            // distinct from `None` (couldn't read the supply dir at all).
            out.power_state = Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging);
        }
    }
    // Laptops only; `None` on desktops and older kernels.
    out.power_save_mode = std::fs::read_to_string("/sys/firmware/acpi/platform_profile")
        .ok()
        .and_then(|s| map_linux_platform_profile(s.trim()));
    out
}

#[cfg(target_os = "linux")]
fn map_linux_battery_status(status: &str) -> Option<pipette_plan_types::DevicePowerState> {
    match status {
        "Charging" => Some(pipette_plan_types::DevicePowerState::Charging),
        "Discharging" => Some(pipette_plan_types::DevicePowerState::NotCharging),
        "Full" | "Not charging" => Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging),
        _ => None, // "Unknown" etc.
    }
}

/// Ambiguous profiles (`quiet`, `cool`) and unknown ones stay `None` rather
/// than guess.
#[cfg(target_os = "linux")]
fn map_linux_platform_profile(profile: &str) -> Option<bool> {
    match profile {
        "low-power" => Some(true),
        "balanced" | "balanced-performance" | "performance" => Some(false),
        _ => None,
    }
}

#[cfg(target_os = "windows")]
fn windows_power_state() -> PowerReading {
    // `NOBATT` distinguishes a battery-less desktop (command succeeded, no
    // battery) from a failed query (`powershell` returns `None`).
    let (battery_level, power_state) = PlatformDetector::powershell(
        "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; \
         if ($b) { \"$($b.EstimatedChargeRemaining)`t$($b.BatteryStatus)\" } else { 'NOBATT' }",
    )
    .as_deref()
    .map(parse_windows_battery)
    .unwrap_or((None, None));
    let power_save_mode = cmd_output("powercfg", &["/getactivescheme"])
        .as_deref()
        .and_then(map_windows_active_scheme);
    PowerReading {
        battery_level,
        power_state,
        power_save_mode,
    }
}

/// Parse `"<EstimatedChargeRemaining>\t<BatteryStatus>"` or the `NOBATT`
/// sentinel.
#[cfg(target_os = "windows")]
fn parse_windows_battery(out: &str) -> (Option<i32>, Option<pipette_plan_types::DevicePowerState>) {
    let out = out.trim();
    if out.is_empty() {
        return (None, None);
    }
    if out == "NOBATT" {
        return (
            None,
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging),
        );
    }
    let mut fields = out.split('\t');
    let level = fields
        .next()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .filter(|n| (0..=100).contains(n));
    let power_state = fields
        .next()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .and_then(map_windows_battery_status);
    (level, power_state)
}

#[cfg(target_os = "windows")]
fn map_windows_battery_status(code: i32) -> Option<pipette_plan_types::DevicePowerState> {
    match code {
        // Discharging (1), Low (4) and Critical (5) all mean on battery.
        1 | 4 | 5 => Some(pipette_plan_types::DevicePowerState::NotCharging),
        // On AC (2) / fully (3) or partially (11) charged.
        2 | 3 | 11 => Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging),
        6..=9 => Some(pipette_plan_types::DevicePowerState::Charging), // charging variants
        _ => None,                                                     // 10 = undefined, etc.
    }
}

/// Matches the built-in Power saver scheme by GUID, which unlike its name does
/// not change with the locale.
#[cfg(target_os = "windows")]
fn map_windows_active_scheme(out: &str) -> Option<bool> {
    // Line: "Power Scheme GUID: a1841308-3541-4fab-bc81-f71556f20b4a  (Power saver)"
    const POWER_SAVER_GUID: &str = "a1841308-3541-4fab-bc81-f71556f20b4a";
    let guid = out
        .split_whitespace()
        .find(|tok| tok.len() == 36 && tok.contains('-'))?;
    Some(guid.eq_ignore_ascii_case(POWER_SAVER_GUID))
}

/// Best-effort thermal snapshot, sampled at each timed rep's two ends; only the
/// current OS's fields are filled. Empty on Windows, which reports through
/// [`crate::detect_device_readings`], and on iOS, which samples in the app.
pub fn detect_thermal() -> pipette_plan_types::ThermalReading {
    #[cfg(target_os = "macos")]
    {
        macos_thermal()
    }
    #[cfg(target_os = "linux")]
    {
        linux_thermal()
    }
    #[cfg(target_os = "android")]
    {
        android::android_thermal()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
    {
        pipette_plan_types::ThermalReading::default()
    }
}

#[cfg(target_os = "macos")]
fn macos_thermal() -> pipette_plan_types::ThermalReading {
    // `ProcessInfo.thermalState`, not the gate's pressure level (which has no
    // mapping to it), so this column stays comparable across gate changes. The
    // die temperature fills the field iOS does, so runs' start temperatures
    // can be compared from the data.
    pipette_plan_types::ThermalReading {
        apple_thermal_state: cmd_output(
            "/usr/bin/swift",
            &[
                "-e",
                "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
            ],
        )
        .as_deref()
        .and_then(parse_apple_thermal_state),
        apple_soc_temp_c: crate::die_temp::die_temp_max_c().map(recorded_die_temp_c),
        ..Default::default()
    }
}

/// Whole °C; [`pipette_plan_types::ThermalTelemetry::device_apple_soc_temp_c_before`]
/// says why.
#[cfg(any(target_os = "macos", test))]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a whole-degree reading fits an f32 exactly"
)]
fn recorded_die_temp_c(raw: f64) -> f32 {
    raw.round() as f32
}

/// `ProcessInfo.thermalState.rawValue` as `swift -e` prints it.
#[cfg(any(target_os = "macos", test))]
fn parse_apple_thermal_state(stdout: &str) -> Option<pipette_plan_types::AppleThermalState> {
    match stdout.trim().parse::<u8>().ok()? {
        0 => Some(pipette_plan_types::AppleThermalState::Nominal),
        1 => Some(pipette_plan_types::AppleThermalState::Fair),
        2 => Some(pipette_plan_types::AppleThermalState::Serious),
        3 => Some(pipette_plan_types::AppleThermalState::Critical),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn linux_thermal() -> pipette_plan_types::ThermalReading {
    pipette_plan_types::ThermalReading {
        linux_thermal_zones: read_thermal_zones(),
        ..Default::default()
    }
}

/// Every readable `/sys/class/thermal/thermal_zone*` in zone-number order
/// (sysfs lists them unordered), or `None` when none is readable.
#[cfg(target_os = "linux")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "i32 m°C / 1000 always fits an i32"
)]
fn read_thermal_zones() -> Option<Vec<pipette_plan_types::LinuxThermalZone>> {
    let mut zones: Vec<(u32, pipette_plan_types::LinuxThermalZone)> =
        std::fs::read_dir("/sys/class/thermal")
            .ok()?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let idx: u32 = entry
                    .file_name()
                    .to_str()?
                    .strip_prefix("thermal_zone")?
                    .parse()
                    .ok()?;
                let path = entry.path();
                let zone_type = std::fs::read_to_string(path.join("type"))
                    .ok()?
                    .trim()
                    .to_string();
                let millicelsius: i32 = std::fs::read_to_string(path.join("temp"))
                    .ok()?
                    .trim()
                    .parse()
                    .ok()?;
                Some((
                    idx,
                    pipette_plan_types::LinuxThermalZone {
                        // Stamped later by the per-run builder.
                        iteration: 0,
                        zone_type,
                        // Rounded, not truncated, like the Android sensors.
                        celsius: (millicelsius as f64 / 1000.0).round() as i32,
                    },
                ))
            })
            .collect();
    if zones.is_empty() {
        return None;
    }
    zones.sort_by_key(|(idx, _)| *idx);
    Some(zones.into_iter().map(|(_, zone)| zone).collect())
}

/// A `PowerManager` `THERMAL_STATUS_*` ordinal. Built on every target so the
/// host-checked `pipette-android` crate can call it.
pub fn android_thermal_status_from_code(
    code: i32,
) -> Option<pipette_plan_types::AndroidThermalStatus> {
    Some(match code {
        0 => pipette_plan_types::AndroidThermalStatus::None,
        1 => pipette_plan_types::AndroidThermalStatus::Light,
        2 => pipette_plan_types::AndroidThermalStatus::Moderate,
        3 => pipette_plan_types::AndroidThermalStatus::Severe,
        4 => pipette_plan_types::AndroidThermalStatus::Critical,
        5 => pipette_plan_types::AndroidThermalStatus::Emergency,
        6 => pipette_plan_types::AndroidThermalStatus::Shutdown,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    // -- Linux lspci parser --

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_a10_typical() {
        let raw =
            "07:00.0 3D controller [0302]: NVIDIA Corporation GA102GL [A10] [10de:2236] (rev a1)\n";
        assert_eq!(
            parse_lspci_nvidia_gpu(raw).as_deref(),
            Some("NVIDIA Corporation GA102GL [A10]")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_h100_no_rev() {
        let raw =
            "00:0d.0 3D controller [0302]: NVIDIA Corporation GH100 [H100 PCIe] [10de:2331]\n";
        assert_eq!(
            parse_lspci_nvidia_gpu(raw).as_deref(),
            Some("NVIDIA Corporation GH100 [H100 PCIe]")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_vga_class() {
        let raw = "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA104 [GeForce RTX 3070] [10de:2484] (rev a1)\n";
        assert_eq!(
            parse_lspci_nvidia_gpu(raw).as_deref(),
            Some("NVIDIA Corporation GA104 [GeForce RTX 3070]")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_skips_audio_controller() {
        let raw = "01:00.1 Audio device [0403]: NVIDIA Corporation GA102 High Definition Audio Controller [10de:1aef] (rev a1)\n";
        assert!(parse_lspci_nvidia_gpu(raw).is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_picks_first_gpu_when_multi() {
        let raw = "07:00.0 3D controller [0302]: NVIDIA Corporation GA100 [A100 SXM4 80GB] [10de:20b2] (rev a1)\n\
                   08:00.0 3D controller [0302]: NVIDIA Corporation GA100 [A100 SXM4 80GB] [10de:20b2] (rev a1)\n";
        assert_eq!(
            parse_lspci_nvidia_gpu(raw).as_deref(),
            Some("NVIDIA Corporation GA100 [A100 SXM4 80GB]")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parse_lspci_empty_returns_none() {
        assert!(parse_lspci_nvidia_gpu("").is_none());
    }

    // -- Linux AMD sysfs probe --

    /// A `/sys/class/drm` stand-in.
    fn drm_tree(cards: &[(&str, &[(&str, &str)])]) -> anyhow::Result<std::path::PathBuf> {
        let root = std::env::temp_dir().join(format!(
            "pipette-drm-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        for (card, files) in cards {
            let device = root.join(card).join("device");
            std::fs::create_dir_all(&device)?;
            for (name, value) in *files {
                std::fs::write(device.join(name), format!("{value}\n"))?;
            }
        }
        Ok(root)
    }

    #[test]
    fn amd_sysfs_reports_an_instinct_card() -> anyhow::Result<()> {
        const MI325X: &[(&str, &str)] = &[
            ("vendor", "0x1002"),
            ("device", "0x74a5"),
            ("class", "0x120000"),
            ("product_name", "AMD Instinct MI325DLC OAM"),
            ("mem_info_vram_total", "274861129728"),
        ];
        // A BMC's VGA on card0, cards out of name order, a connector and a
        // render node.
        let root = drm_tree(&[
            ("card0", &[("vendor", "0x1a03"), ("class", "0x030000")]),
            ("card9", MI325X),
            ("card1", MI325X),
            ("card1-DP-1", &[("vendor", "0x1002"), ("class", "0x030000")]),
            ("renderD128", MI325X),
        ])?;
        let found = amd_gpu_from_sysfs(&root);
        std::fs::remove_dir_all(&root)?;
        assert_eq!(
            found,
            Some((
                "AMD Instinct MI325DLC OAM".to_string(),
                Some(274_861_129_728)
            ))
        );
        Ok(())
    }

    #[test]
    fn amd_sysfs_names_a_card_by_its_id_without_a_product_name() -> anyhow::Result<()> {
        let root = drm_tree(&[
            ("card0", &[("vendor", "0x1002"), ("class", "0x040300")]),
            (
                "card1",
                &[
                    ("vendor", "0x1002"),
                    ("device", "0x1586"),
                    ("class", "0x030000"),
                ],
            ),
        ])?;
        let found = amd_gpu_from_sysfs(&root);
        std::fs::remove_dir_all(&root)?;
        assert_eq!(found, Some(("AMD GPU [1002:1586]".to_string(), None)));
        Ok(())
    }

    #[test]
    fn amd_sysfs_finds_nothing_without_an_amd_gpu() -> anyhow::Result<()> {
        let root = drm_tree(&[("card0", &[("vendor", "0x8086"), ("class", "0x030000")])])?;
        let found = amd_gpu_from_sysfs(&root);
        std::fs::remove_dir_all(&root)?;
        assert_eq!(found, None);
        assert_eq!(
            amd_gpu_from_sysfs(std::path::Path::new("/no/such/drm")),
            None
        );
        Ok(())
    }

    // -- Linux form factor --

    #[rstest]
    #[case::jetson_agx_orin(Some(1), true, None, Some(DeviceFormFactor::Embedded))]
    #[case::raspberry_pi_no_dmi(None, true, None, Some(DeviceFormFactor::Embedded))]
    #[case::unknown_chassis_with_device_tree(Some(2), true, None, Some(DeviceFormFactor::Embedded))]
    #[case::other_chassis_nothing_else(Some(1), false, None, None)]
    #[case::no_signal_at_all(None, false, None, None)]
    #[case::desktop_chassis(Some(3), false, None, Some(DeviceFormFactor::Desktop))]
    #[case::notebook_chassis(Some(10), false, None, Some(DeviceFormFactor::Laptop))]
    #[case::rack_mount_chassis(Some(23), false, None, Some(DeviceFormFactor::Server))]
    // A real chassis code wins even beside a device tree: an arm64 server board
    // that says `Rack Mount` is one.
    #[case::rack_mount_with_device_tree(Some(23), true, None, Some(DeviceFormFactor::Server))]
    #[case::main_server_chassis(Some(17), false, None, Some(DeviceFormFactor::Server))]
    #[case::dgx_spark(
        Some(17),
        false,
        Some("NVIDIA_DGX_Spark"),
        Some(DeviceFormFactor::Desktop)
    )]
    #[case::dgx_spark_notebook_chassis(
        Some(10),
        false,
        Some("NVIDIA_DGX_Spark"),
        Some(DeviceFormFactor::Laptop)
    )]
    #[case::dgx_spark_rack_mount_chassis(
        Some(23),
        false,
        Some("NVIDIA_DGX_Spark"),
        Some(DeviceFormFactor::Server)
    )]
    fn linux_form_factor_from_signals(
        #[case] chassis: Option<u32>,
        #[case] has_device_tree: bool,
        #[case] product_name: Option<&str>,
        #[case] want: Option<DeviceFormFactor>,
    ) {
        assert_eq!(
            linux_form_factor(chassis, has_device_tree, product_name),
            want
        );
    }

    #[rstest]
    #[case::desktop(3, DeviceFormFactor::Desktop)]
    #[case::hand_held(11, DeviceFormFactor::Phone)]
    #[case::docking_station(12, DeviceFormFactor::Desktop)]
    #[case::main_server_chassis(17, DeviceFormFactor::Server)]
    #[case::rack_mount(23, DeviceFormFactor::Server)]
    #[case::iot_gateway(33, DeviceFormFactor::Embedded)]
    #[case::embedded_pc(34, DeviceFormFactor::Embedded)]
    fn chassis_code_to_form_factor(#[case] chassis: u32, #[case] want: DeviceFormFactor) {
        assert_eq!(chassis_to_form_factor(chassis), want);
    }

    // -- Cross-platform --

    #[test]
    fn detect_returns_all_required_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert!(d.device_name.is_some(), "device_name not detected");
        assert!(d.device_os_name.is_some(), "device_os_name not detected");
        assert!(
            d.device_os_version.is_some(),
            "device_os_version not detected"
        );
        assert!(
            d.device_chip_model.is_some(),
            "device_chip_model not detected"
        );
        assert!(
            d.device_ram_bytes.is_some(),
            "device_ram_bytes not detected"
        );
        assert!(d.device_ram_bytes.context("device_ram_bytes")? > 0);
        Ok(())
    }

    #[test]
    fn detect_device_info_applies_overrides() -> anyhow::Result<()> {
        let info = detect_device_info(Some("custom-name"), Some(DeviceFormFactor::Server))?;
        assert_eq!(info.device_name.as_ref(), "custom-name");
        assert_eq!(info.device_form_factor, DeviceFormFactor::Server);
        assert!(info.device_ram_bytes > 0);
        Ok(())
    }

    #[test]
    fn detect_device_info_trims_user_provided_name() -> anyhow::Result<()> {
        let info = detect_device_info(Some("  custom-name  "), Some(DeviceFormFactor::Server))?;
        assert_eq!(info.device_name.as_ref(), "custom-name");
        Ok(())
    }

    #[test]
    fn detect_device_info_without_overrides_succeeds() -> anyhow::Result<()> {
        let info = detect_device_info(None, None)?;
        assert!(!info.device_name.as_ref().is_empty());
        assert!(info.device_ram_bytes > 0);
        Ok(())
    }

    #[test]
    fn detect_device_info_reports_the_build_target() -> anyhow::Result<()> {
        let info = detect_device_info(None, None)?;
        assert_eq!(
            info.device_cpu_arch.as_ref().map(AsRef::as_ref),
            Some(std::env::consts::ARCH)
        );
        assert_eq!(
            info.device_os_family.as_ref().map(AsRef::as_ref),
            Some(std::env::consts::OS)
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert_eq!(d.device_os_name.as_deref(), Some("macOS"));
        assert!(
            d.device_os_version
                .as_ref()
                .context("device_os_version")?
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        );
        assert!(
            d.device_chip_model
                .as_ref()
                .context("device_chip_model")?
                .contains("Apple")
        );
        assert!(d.device_ram_bytes.context("device_ram_bytes")? >= 4 * 1024 * 1024 * 1024);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert!(d.device_os_name.is_some());
        assert!(d.device_ram_bytes.context("device_ram_bytes")? >= 1024 * 1024 * 1024);
        Ok(())
    }

    #[rstest]
    #[case("Default string", true)]
    #[case("To Be Filled By O.E.M.", true)]
    #[case("Base Board Product Name", true)]
    #[case("Type2 - Board Product Name", true)]
    #[case("Not Specified", true)]
    #[case("  N/A  ", true)]
    #[case("EdgeXpert (MS-C931)", false)]
    #[case("Jetson", false)]
    fn dmi_placeholder_is_rejected(#[case] board: &str, #[case] placeholder: bool) {
        assert_eq!(is_dmi_placeholder(board), placeholder);
    }

    /// A Jetson AGX Orin Developer Kit (5.15 tegra kernel), two of twelve cores.
    const ORIN_CPUINFO: &str = "\
processor\t: 0
model name\t: ARMv8 Processor rev 1 (v8l)
BogoMIPS\t: 62.50
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp uscat ilrcpc flagm paca pacg
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0xd42
CPU revision\t: 1

processor\t: 1
model name\t: ARMv8 Processor rev 1 (v8l)
BogoMIPS\t: 62.50
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp uscat ilrcpc flagm paca pacg
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: 0xd42
CPU revision\t: 1
";

    /// A Raspberry Pi 5 (6.12 rpi kernel), one core plus the trailing board block
    /// with no MIDR fields. Serial zeroed.
    const PI5_CPUINFO: &str = "\
processor\t: 0
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x4
CPU part\t: 0xd0b
CPU revision\t: 1

Revision\t: e04171
Serial\t\t: 0000000000000000
Model\t\t: Raspberry Pi 5 Model B Rev 1.1
";

    /// The NVIDIA GB10 (MSI EdgeXpert) shape: no `model name`, big and little
    /// cores.
    const GB10_CPUINFO: &str = "\
processor\t: 0
CPU implementer\t: 0x41
CPU part\t: 0xd85

processor\t: 1
CPU implementer\t: 0x41
CPU part\t: 0xd87

processor\t: 2
CPU implementer\t: 0x41
CPU part\t: 0xd85
";

    #[rstest]
    #[case(ORIN_CPUINFO, Some("Cortex-A78AE"))]
    #[case(PI5_CPUINFO, Some("Cortex-A76"))]
    #[case(GB10_CPUINFO, Some("Cortex-X925 + Cortex-A725"))]
    // One core with a part the table does not know: no partial answer.
    #[case(
        "CPU implementer\t: 0x41\nCPU part\t: 0xd0b\n\nCPU implementer\t: 0x41\nCPU part\t: 0xfff\n",
        None
    )]
    // Another implementer (Qualcomm Oryon) is not decoded.
    #[case("CPU implementer\t: 0x51\nCPU part\t: 0x001\n", None)]
    #[case(
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) Ultra X7 358H\n",
        None
    )]
    #[case("", None)]
    fn cpuinfo_arm_cores_decode_midr(#[case] cpuinfo: &str, #[case] want: Option<&str>) {
        assert_eq!(parse_cpuinfo_arm_cores(cpuinfo).as_deref(), want);
    }

    #[rstest]
    // Jetson: the generic cpuinfo `model name` yields to the board name.
    #[case(
        Some("ARMv8 Processor rev 1 (v8l)"),
        Some("NVIDIA Jetson AGX Orin Developer Kit"),
        Some("NVIDIA Jetson AGX Orin Developer Kit")
    )]
    // Generic and no device tree: left to the MIDR fallback.
    #[case(Some("ARMv8 Processor rev 1 (v8l)"), None, None)]
    // Pi 5: cpuinfo has neither `Hardware` nor `model name`.
    #[case(
        None,
        Some("Raspberry Pi 5 Model B Rev 1.1"),
        Some("Raspberry Pi 5 Model B Rev 1.1")
    )]
    // A named cpuinfo chip outranks the board name.
    #[case(
        Some("BCM2711"),
        Some("Raspberry Pi 4 Model B Rev 1.4"),
        Some("BCM2711")
    )]
    #[case(
        Some("Intel(R) Core(TM) Ultra X7 358H"),
        None,
        Some("Intel(R) Core(TM) Ultra X7 358H")
    )]
    fn chip_model_skips_generic_armv8(
        #[case] cpuinfo: Option<&str>,
        #[case] device_tree: Option<&str>,
        #[case] want: Option<&str>,
    ) {
        let got = chip_model_or_board(cpuinfo.map(str::to_owned), device_tree.map(str::to_owned));
        assert_eq!(got.as_deref(), want);
    }

    // -- Windows --

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert!(
            d.device_os_name
                .as_ref()
                .context("device_os_name")?
                .contains("Windows")
        );
        assert!(d.device_os_version.is_some());
        assert!(d.device_chip_model.is_some());
        assert!(d.device_ram_bytes.context("device_ram_bytes")? > 0);
        Ok(())
    }

    // -- Power state (detection never panics; values are environment-dependent) --

    #[test]
    fn detect_power_state_is_consistent() {
        let p = detect_power_state();
        if let Some(level) = p.battery_level {
            assert!((0..=100).contains(&level));
        }
    }

    // -- macOS pmset parser --

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_pmset_batt_cases() {
        let on_battery = "Now drawing from 'Battery Power'\n \
            -InternalBattery-0 (id=4325123)\t72%; discharging; 3:11 remaining present: true";
        assert_eq!(
            parse_pmset_batt(on_battery),
            (
                Some(72),
                Some(pipette_plan_types::DevicePowerState::NotCharging)
            )
        );

        let charging = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t83%; charging; 0:42 remaining present: true";
        assert_eq!(
            parse_pmset_batt(charging),
            (
                Some(83),
                Some(pipette_plan_types::DevicePowerState::Charging)
            )
        );

        let charged = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t100%; charged; 0:00 remaining present: true";
        assert_eq!(
            parse_pmset_batt(charged),
            (
                Some(100),
                Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
            )
        );

        let holding = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t80%; not charging; 0:00 remaining present: true";
        assert_eq!(
            parse_pmset_batt(holding),
            (
                Some(80),
                Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
            )
        );

        assert_eq!(
            parse_pmset_batt("Now drawing from 'AC Power'\nNo batteries available."),
            (
                None,
                Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
            )
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_low_power_enabled_reads_setting() {
        assert!(macos_low_power_enabled(
            " lowpowermode         1\n hibernatemode 3"
        ));
        assert!(!macos_low_power_enabled(
            " lowpowermode         0\n hibernatemode 3"
        ));
        assert!(!macos_low_power_enabled(" hibernatemode 3"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn map_linux_battery_status_cases() {
        assert_eq!(
            map_linux_battery_status("Charging"),
            Some(pipette_plan_types::DevicePowerState::Charging)
        );
        assert_eq!(
            map_linux_battery_status("Discharging"),
            Some(pipette_plan_types::DevicePowerState::NotCharging)
        );
        assert_eq!(
            map_linux_battery_status("Full"),
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
        );
        assert_eq!(
            map_linux_battery_status("Not charging"),
            Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
        );
        assert_eq!(map_linux_battery_status("Unknown"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn map_linux_platform_profile_cases() {
        assert_eq!(map_linux_platform_profile("low-power"), Some(true));
        assert_eq!(map_linux_platform_profile("performance"), Some(false));
        assert_eq!(map_linux_platform_profile("balanced"), Some(false));
        assert_eq!(map_linux_platform_profile("quiet"), None);
        assert_eq!(map_linux_platform_profile(""), None);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn parse_windows_battery_cases() {
        assert_eq!(
            parse_windows_battery("72\t1"),
            (
                Some(72),
                Some(pipette_plan_types::DevicePowerState::NotCharging)
            )
        );
        assert_eq!(
            parse_windows_battery("90\t6"),
            (
                Some(90),
                Some(pipette_plan_types::DevicePowerState::Charging)
            )
        );
        assert_eq!(
            parse_windows_battery("100\t3"),
            (
                Some(100),
                Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
            )
        );
        assert_eq!(
            parse_windows_battery("8\t5"),
            (
                Some(8),
                Some(pipette_plan_types::DevicePowerState::NotCharging)
            )
        );
        assert_eq!(
            parse_windows_battery("NOBATT"),
            (
                None,
                Some(pipette_plan_types::DevicePowerState::PluggedInNotCharging)
            )
        );
        assert_eq!(parse_windows_battery(""), (None, None));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn map_windows_active_scheme_cases() {
        assert_eq!(
            map_windows_active_scheme(
                "Power Scheme GUID: a1841308-3541-4fab-bc81-f71556f20b4a  (Power saver)"
            ),
            Some(true)
        );
        assert_eq!(
            map_windows_active_scheme(
                "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
            ),
            Some(false)
        );
        assert_eq!(map_windows_active_scheme("no guid here"), None);
    }

    // -- Thermal parsers ----------------------------------------------------

    use rstest::rstest;

    #[rstest]
    #[case("0\n", Some(pipette_plan_types::AppleThermalState::Nominal))]
    #[case("2", Some(pipette_plan_types::AppleThermalState::Serious))]
    #[case("3\n", Some(pipette_plan_types::AppleThermalState::Critical))]
    #[case("nominal", None)] // non-numeric
    #[case("7", None)] // out of range
    fn apple_thermal_state_maps_raw_values(
        #[case] raw: &str,
        #[case] want: Option<pipette_plan_types::AppleThermalState>,
    ) {
        assert_eq!(parse_apple_thermal_state(raw), want);
    }

    /// The wire column is `f32`, so a fractional value would reach a reader as
    /// 46.79999923706055 rather than the number recorded.
    #[rstest]
    #[case(46.4, 46.0)]
    #[case(46.5, 47.0)] // half rounds away from zero, not to even
    #[case(46.764_129_638_671_875, 47.0)]
    #[case(0.0, 0.0)]
    fn recorded_die_temp_is_whole_degrees(#[case] raw: f64, #[case] want: f32) {
        let got = recorded_die_temp_c(raw);
        assert_eq!(got, want);
        assert_eq!(got.fract(), 0.0, "{got} is not a whole number of degrees");
    }

    #[rstest]
    #[case(0, Some(pipette_plan_types::AndroidThermalStatus::None))]
    #[case(1, Some(pipette_plan_types::AndroidThermalStatus::Light))]
    #[case(2, Some(pipette_plan_types::AndroidThermalStatus::Moderate))]
    #[case(3, Some(pipette_plan_types::AndroidThermalStatus::Severe))]
    #[case(4, Some(pipette_plan_types::AndroidThermalStatus::Critical))]
    #[case(5, Some(pipette_plan_types::AndroidThermalStatus::Emergency))]
    #[case(6, Some(pipette_plan_types::AndroidThermalStatus::Shutdown))]
    #[case(7, None)] // out of range
    #[case(-1, None)] // out of range (negative)
    fn android_thermal_status_from_code_maps_ordinals(
        #[case] code: i32,
        #[case] want: Option<pipette_plan_types::AndroidThermalStatus>,
    ) {
        assert_eq!(android_thermal_status_from_code(code), want);
    }

    /// Run with `cargo test detect_thermal_smoke -- --ignored --nocapture`.
    #[test]
    #[ignore = "shells out to platform thermal tools; run manually"]
    fn detect_thermal_smoke() {
        let reading = detect_thermal();
        println!("detect_thermal() => {reading:?}");
        #[cfg(target_os = "macos")]
        assert!(
            reading.apple_thermal_state.is_some(),
            "macOS should read a ProcessInfo thermal state"
        );
        // Absent is allowed (private API); present, it is the field iOS fills.
        #[cfg(target_os = "macos")]
        if let Some(celsius) = reading.apple_soc_temp_c {
            assert!(
                (-50.0..150.0).contains(&celsius),
                "implausible die temp {celsius}C",
            );
            assert_eq!(celsius.fract(), 0.0, "die temp {celsius}C was not rounded");
        }
    }
}
