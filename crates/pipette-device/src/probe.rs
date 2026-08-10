//! Cross-platform device detection.
//!
//! Each platform implements [`PlatformDetector::detect`] which runs all
//! necessary OS/hardware queries in one shot and returns [`DetectedFields`]
//! — a bag of `Option`s.  The public entry point [`detect_device_info`]
//! merges user overrides, validates that required fields are present, and
//! returns `Result<DeviceInfo>`.

// The module, not the types: each per-OS reader names a different subset, so
// importing them individually would need a `cfg` per name to stay unused-free.
use pipette_plan_types::thermal;

use pipette_plan_types::device::{DeviceFormFactor, DeviceInfo, TrimmedString};
use pipette_plan_types::thermal::PowerState;

/// Raw detection results — every field is `Option` so callers can
/// distinguish "not detected" from any particular value.
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

/// Build a [`DeviceInfo`] from auto-detected OS and hardware details.
///
/// `device_name` and `device_form_factor` override the auto-detected values
/// when provided (e.g. from registration config).  Returns an error listing
/// any required fields that could not be detected.
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

// ============================================================================
// Platform detector — one struct, one `detect()` call per platform
// ============================================================================
//
// `std::process::Command` is written fully-qualified at each call site below
// rather than imported at the top. Only the OS-specific detectors shell out;
// the iOS / catch-all `detect()` (`cfg(not(any(...)))`) does not. A top-level
// `use` would therefore be an unused import on iOS — and with deny-on-warnings
// (`[workspace.lints]`) that fails the build. Keeping the path inline ties the
// dependency to the OS-gated blocks that actually use it.

/// Namespace for platform-specific `detect()` implementations.
struct PlatformDetector;

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Linux
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        let (gpu_model, gpu_vram) = Self::detect_nvidia_gpu();
        // Kernel release (e.g. `6.8.0-45-generic`), read once: it is the
        // `device_os_build` value and also the `device_os_version` fallback when
        // the distro `VERSION_ID` is absent.
        let kernel_release = cmd_output("uname", &["-r"]);

        DetectedFields {
            device_name: Self::detect_device_name(),
            device_form_factor: Self::detect_form_factor(),
            device_os_name: Self::os_release_field("NAME").or_else(|| cmd_output("uname", &["-s"])),
            device_os_version: Self::os_release_field("VERSION_ID")
                .or_else(|| kernel_release.clone()),
            device_os_build: kernel_release.clone(),
            device_os_security_patch: None,
            device_chip_model: cpuinfo_chip_model().or_else(Self::device_tree_model),
            device_ram_bytes: proc_memtotal_bytes(),
            device_gpu_model: gpu_model,
            device_gpu_vram_bytes: gpu_vram,
        }
    }

    fn detect_device_name() -> Option<String> {
        std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(Self::device_tree_model)
            .or_else(|| cmd_output("hostname", &[]))
    }

    fn detect_form_factor() -> Option<DeviceFormFactor> {
        if let Some(chassis) = std::fs::read_to_string("/sys/devices/virtual/dmi/id/chassis_type")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            return Some(chassis_to_form_factor(chassis));
        }
        if std::path::Path::new("/proc/device-tree").exists() {
            return Some(DeviceFormFactor::Embedded);
        }
        None
    }

    /// `/proc/device-tree/model` — the board name on hosts with no DMI
    /// (SBCs, most ARM boxes). Device-tree nodes are NUL-terminated.
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
        // Fallback: nvidia-smi is missing or broken (common when the host's
        // NVML headers/library drift out of sync with the driver). lspci
        // identifies the card from the PCI device id even without a working
        // driver. VRAM isn't knowable this way; leave it None.
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

    /// Parse `lspci -d 10de: -nn` output for an NVIDIA display/3D
    /// controller. The first match wins.
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

/// Extract the NVIDIA GPU model from `lspci -d 10de: -nn` output.
///
/// Example input line:
///   `07:00.0 3D controller [0302]: NVIDIA Corporation GA102GL [A10] [10de:2236] (rev a1)`
/// Returns: `NVIDIA Corporation GA102GL [A10]`.
///
/// Restricts to GPU-shaped PCI classes (`0300` VGA, `0302` 3D, `0380` other
/// display) so NVIDIA-shipped audio controllers and PCI bridges don't match.
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
        let after_class = match line.find("]: ") {
            Some(i) => &line[i + 3..],
            None => continue,
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

// ---------------------------------------------------------------------------
// Android
// ---------------------------------------------------------------------------

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
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    fn getprop_first(keys: &[&str]) -> Option<String> {
        keys.iter().find_map(|k| Self::getprop(k))
    }
}

// ---------------------------------------------------------------------------
// Windows — single batched PowerShell call
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
impl PlatformDetector {
    fn detect() -> DetectedFields {
        if let Some(d) = Self::detect_batched() {
            return d;
        }
        log::warn!("batched PowerShell detection failed; falling back to per-field detection");
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
        // 8 required fields; the 9th (OS build) is optional — an empty `$build`
        // leaves a trailing tab that `powershell()`'s trim() drops, so accept 8.
        if p.len() < 8 {
            return None;
        }
        let gpu_model = if p[4].is_empty() {
            None
        } else {
            Some(p[4].to_string())
        };
        let gpu_vram = if gpu_model.is_some() {
            p[5].parse().ok().filter(|&v: &u64| v > 0)
        } else {
            None
        };
        Some(DetectedFields {
            device_name: Some(p[6].to_string()).filter(|s| !s.is_empty()),
            device_form_factor: p[7].trim().parse::<u32>().ok().map(chassis_to_form_factor),
            device_os_name: Some(p[0].to_string()).filter(|s| !s.is_empty()),
            device_os_version: Some(p[1].to_string()).filter(|s| !s.is_empty()),
            // `CurrentBuild.UBR` (e.g. `26100.1234`); a missing 9th field or an
            // empty value (either registry key absent — see the `$build` guard) → None.
            device_os_build: p.get(8).map(|s| s.to_string()).filter(|s| !s.is_empty()),
            device_os_security_patch: None,
            device_chip_model: Some(p[2].to_string()).filter(|s| !s.is_empty()),
            device_ram_bytes: p[3].parse().ok(),
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
            // Emits nothing (→ `None`) unless both registry values are present,
            // matching the batched path's `$build` guard.
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
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

// ---------------------------------------------------------------------------
// Unsupported platforms
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

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
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Parse chip model from `/proc/cpuinfo` (`Hardware` or `model name` field).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn cpuinfo_chip_model() -> Option<String> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for key in ["Hardware", "model name"] {
        for line in content.lines() {
            if let Some((k, v)) = line.split_once(':') {
                if k.trim().eq_ignore_ascii_case(key) {
                    let v = v.trim();
                    if !v.is_empty() {
                        return Some(v.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Parse `MemTotal` from `/proc/meminfo` and return bytes.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn proc_memtotal_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some((key, value)) = line.split_once(':') {
            if key.trim() == "MemTotal" {
                let value = value.trim();
                let kb_str = value.strip_suffix("kB").unwrap_or(value).trim();
                if let Ok(kb) = kb_str.parse::<u64>() {
                    return Some(kb * 1024);
                }
            }
        }
    }
    None
}

/// Map SMBIOS chassis type codes to [`DeviceFormFactor`].
/// Used by both Windows and Linux (DMI) form factor detection.
/// <https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-systemenclosure>
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn chassis_to_form_factor(chassis: u32) -> DeviceFormFactor {
    match chassis {
        8 | 9 | 10 | 14 => DeviceFormFactor::Laptop,
        30..=32 => DeviceFormFactor::Tablet,
        11 | 17 => DeviceFormFactor::Phone,
        23 | 28 | 12 => DeviceFormFactor::Server,
        _ => DeviceFormFactor::Desktop,
    }
}

// ---------------------------------------------------------------------------
// Run-environment power state
// ---------------------------------------------------------------------------
//
// Unlike [`DeviceInfo`] (static identity, detected once at registration), power
// state is volatile and detected fresh for each submission: a laptop throttles
// the CPU on battery / in low-power mode just like a phone, so each result
// records the state it ran under for later filtering. A desktop with no battery
// reports `None` for all fields. Mirrors the mobile `device_*` power fields and
// the `device_power_state` enum the management server expects
// (`charging` / `not_charging` / `plugged_in_not_charging`).

/// Detect the current power state (best-effort; every field degrades to `None`).
pub fn detect_power_state() -> PowerState {
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
        PowerState::default()
    }
}

#[cfg(target_os = "macos")]
fn macos_power_state() -> PowerState {
    let (battery_level, power_state) = cmd_output("pmset", &["-g", "batt"])
        .as_deref()
        .map(parse_pmset_batt)
        .unwrap_or((None, None));
    let power_save_mode = cmd_output("pmset", &["-g"])
        .as_deref()
        .map(macos_low_power_enabled);
    PowerState {
        battery_level,
        power_state,
        power_save_mode,
    }
}

/// Parse `pmset -g batt`. The battery line looks like
/// `" -InternalBattery-0 (id=…)\t83%; discharging; 3:11 remaining present: true"`.
/// A desktop Mac has no such line: the command still succeeds, so that's a
/// machine with no battery on external power (`PluggedInNotCharging`), which
/// the caller distinguishes from `None` (command failed / state unknown).
#[cfg(target_os = "macos")]
fn parse_pmset_batt(out: &str) -> (Option<i32>, Option<thermal::DevicePowerState>) {
    let Some(line) = out.lines().find(|l| l.contains('%')) else {
        return (None, Some(thermal::DevicePowerState::PluggedInNotCharging));
    };
    // Percentage = the trailing digit run immediately before the first '%'.
    let level = line
        .split('%')
        .next()
        .and_then(|p| {
            p.rsplit(|c: char| !c.is_ascii_digit())
                .find(|s| !s.is_empty())
        })
        .and_then(|d| d.parse::<i32>().ok())
        .filter(|n| (0..=100).contains(n));
    // State word follows the "<pct>%;" — one of "charging", "discharging",
    // "charged", "finishing charge", or "not charging" (plugged in but
    // holding, e.g. Optimized Battery Charging — still on external power).
    let power_state = line.split(';').nth(1).map(str::trim).and_then(|s| {
        if s.starts_with("discharging") {
            Some(thermal::DevicePowerState::NotCharging)
        } else if s.starts_with("not charging") {
            Some(thermal::DevicePowerState::PluggedInNotCharging)
        } else if s.starts_with("charging") || s.starts_with("finishing charge") {
            Some(thermal::DevicePowerState::Charging)
        } else if s.starts_with("charged") {
            Some(thermal::DevicePowerState::PluggedInNotCharging)
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
fn linux_power_state() -> PowerState {
    let base = std::path::Path::new("/sys/class/power_supply");
    let mut out = PowerState::default();
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
            out.power_state = Some(thermal::DevicePowerState::PluggedInNotCharging);
        }
    }
    // Low-power signal from the ACPI platform profile, where the kernel
    // exposes it (laptops; `None` on desktops / older kernels).
    out.power_save_mode = std::fs::read_to_string("/sys/firmware/acpi/platform_profile")
        .ok()
        .and_then(|s| map_linux_platform_profile(s.trim()));
    out
}

#[cfg(target_os = "linux")]
fn map_linux_battery_status(status: &str) -> Option<thermal::DevicePowerState> {
    match status {
        "Charging" => Some(thermal::DevicePowerState::Charging),
        "Discharging" => Some(thermal::DevicePowerState::NotCharging),
        "Full" | "Not charging" => Some(thermal::DevicePowerState::PluggedInNotCharging),
        _ => None, // "Unknown" etc.
    }
}

/// Map `/sys/firmware/acpi/platform_profile` to a low-power flag. Only the
/// explicit `low-power` profile counts as power-saving; the clearly-not ones
/// map to `false`; ambiguous profiles (`quiet`/`cool`) and anything unknown
/// stay `None` rather than guess.
#[cfg(target_os = "linux")]
fn map_linux_platform_profile(profile: &str) -> Option<bool> {
    match profile {
        "low-power" => Some(true),
        "balanced" | "balanced-performance" | "performance" => Some(false),
        _ => None,
    }
}

#[cfg(target_os = "windows")]
fn windows_power_state() -> PowerState {
    // `NOBATT` distinguishes a battery-less desktop (command succeeded, no
    // battery) from a failed query (`powershell` returns `None`).
    let (battery_level, power_state) = PlatformDetector::powershell(
        "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; \
         if ($b) { \"$($b.EstimatedChargeRemaining)`t$($b.BatteryStatus)\" } else { 'NOBATT' }",
    )
    .as_deref()
    .map(parse_windows_battery)
    .unwrap_or((None, None));
    // Low-power signal: the active power *plan*. The built-in "Power saver"
    // scheme has a stable, locale-independent GUID, so we match the GUID
    // rather than the localized scheme name.
    let power_save_mode = cmd_output("powercfg", &["/getactivescheme"])
        .as_deref()
        .and_then(map_windows_active_scheme);
    PowerState {
        battery_level,
        power_state,
        power_save_mode,
    }
}

/// Parse the tab-separated `"<EstimatedChargeRemaining>\t<BatteryStatus>"`
/// line, or the `NOBATT` sentinel emitted by a battery-less desktop.
#[cfg(target_os = "windows")]
fn parse_windows_battery(out: &str) -> (Option<i32>, Option<thermal::DevicePowerState>) {
    let out = out.trim();
    if out.is_empty() {
        return (None, None);
    }
    if out == "NOBATT" {
        // Command succeeded, no battery: a desktop on external power.
        return (None, Some(thermal::DevicePowerState::PluggedInNotCharging));
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

/// Map `Win32_Battery.BatteryStatus` codes to a power state.
#[cfg(target_os = "windows")]
fn map_windows_battery_status(code: i32) -> Option<thermal::DevicePowerState> {
    match code {
        // Discharging (1) / Low (4) / Critical (5) all mean "running on
        // battery", so report not-charging rather than dropping the signal.
        1 | 4 | 5 => Some(thermal::DevicePowerState::NotCharging),
        // On AC (2) / fully (3) or partially (11) charged.
        2 | 3 | 11 => Some(thermal::DevicePowerState::PluggedInNotCharging),
        6..=9 => Some(thermal::DevicePowerState::Charging), // charging variants
        _ => None,                                          // 10 = undefined, etc.
    }
}

/// Map `powercfg /getactivescheme` output to a low-power flag by matching the
/// built-in **Power saver** scheme GUID (stable across locales). Other schemes
/// (Balanced / High performance / custom) are not power-saving; a line with no
/// GUID yields `None`.
#[cfg(target_os = "windows")]
fn map_windows_active_scheme(out: &str) -> Option<bool> {
    // Line: "Power Scheme GUID: a1841308-3541-4fab-bc81-f71556f20b4a  (Power saver)"
    const POWER_SAVER_GUID: &str = "a1841308-3541-4fab-bc81-f71556f20b4a";
    let guid = out
        .split_whitespace()
        .find(|tok| tok.len() == 36 && tok.contains('-'))?;
    Some(guid.eq_ignore_ascii_case(POWER_SAVER_GUID))
}

// ---------------------------------------------------------------------------
// Thermal probe
//
// Sampled alongside `detect_power_state()`: the execute layer calls
// `detect_thermal()` around each measured repetition (`before` at its
// gate-pass, `after` at its completion) and accumulates the series into a
// `RunThermal`. Per-platform — this fills only the fields the current OS
// exposes; the rest stay `None`. The shapes themselves live in
// `pipette_plan_types::thermal`, with the leaf types they are built from.
// ---------------------------------------------------------------------------

/// Best-effort thermal snapshot for the current platform (every field degrades
/// to `None`). A no-op returning an empty reading on platforms with no reader
/// (Windows is blocked on a working sensor; iOS is handled in the app layer).
pub fn detect_thermal() -> thermal::ThermalReading {
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
        android_thermal()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
    {
        thermal::ThermalReading::default()
    }
}

#[cfg(target_os = "macos")]
fn macos_thermal() -> thermal::ThermalReading {
    // Two signals, deliberately from different sources.
    //
    // `ProcessInfo.thermalState` is kept verbatim so the recorded
    // `apple_thermal_state` series stays a fixed yardstick — the readiness
    // gate moved off it (see `pipette-readiness/src/macos.rs`), and the
    // point of not moving telemetry with it is that runs before and after
    // that change remain comparable on this column.
    //
    // Die temperature is the same field iOS already reports, so this adds a
    // platform to an existing series rather than a new one. The gate does not
    // consume it (see `pipette-readiness/src/macos.rs`); it is recorded so
    // that whether two runs started from comparable temperatures is
    // answerable from the data rather than assumed.
    thermal::ThermalReading {
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

/// The die reading as recorded: whole °C. See the field docs on
/// [`thermal::ThermalTelemetry::device_apple_soc_temp_c_before`] for why this is
/// not the raw sensor value, and `BenchmarkMeasurement.swift`, which rounds the
/// same field on iOS and must agree with this.
#[cfg(any(target_os = "macos", test))]
fn recorded_die_temp_c(raw: f64) -> f32 {
    raw.round() as f32
}

/// Map the `ProcessInfo.thermalState.rawValue` (0–3, coolest→hottest) printed
/// by `swift -e` onto [`thermal::AppleThermalState`].
#[cfg(any(target_os = "macos", test))]
fn parse_apple_thermal_state(stdout: &str) -> Option<thermal::AppleThermalState> {
    match stdout.trim().parse::<u8>().ok()? {
        0 => Some(thermal::AppleThermalState::Nominal),
        1 => Some(thermal::AppleThermalState::Fair),
        2 => Some(thermal::AppleThermalState::Serious),
        3 => Some(thermal::AppleThermalState::Critical),
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn linux_thermal() -> thermal::ThermalReading {
    thermal::ThermalReading {
        linux_thermal_zones: read_thermal_zones(),
        ..Default::default()
    }
}

/// Every readable `/sys/class/thermal/thermal_zone*` as a `{type, celsius}`
/// list, in zone-number order (sysfs enumerates them unordered), milli-°C
/// rounded to whole °C. `None` when no zone is readable.
#[cfg(target_os = "linux")]
fn read_thermal_zones() -> Option<Vec<thermal::LinuxThermalZone>> {
    let mut zones: Vec<(u32, thermal::LinuxThermalZone)> = std::fs::read_dir("/sys/class/thermal")
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
                thermal::LinuxThermalZone {
                    // Stamped with the real iteration by the per-run builder.
                    iteration: 0,
                    zone_type,
                    // Round to whole °C (matching the Android sensor path and
                    // the warehouse contract), not truncate: 44_900 m°C → 45.
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

#[cfg(target_os = "android")]
fn android_thermal() -> thermal::ThermalReading {
    // `dumpsys thermalservice` carries both the device-level status and the
    // thermal-HAL per-sensor block; `cmd thermalservice headroom` the forecast
    // headroom. All three require a shell-privileged context (adb / CLI) — a
    // sandboxed app can't reach them and gets `None` (the app-SDK PowerManager
    // path is handled separately in the Android app layer).
    let dumpsys = cmd_output("dumpsys", &["thermalservice"]);
    thermal::ThermalReading {
        android_thermal_status: dumpsys.as_deref().and_then(parse_android_thermal_status),
        android_thermal_headroom: cmd_output("cmd", &["thermalservice", "headroom", "0"])
            .as_deref()
            .and_then(parse_android_headroom),
        android_thermal_sensors: dumpsys.as_deref().and_then(parse_android_hal_sensors),
        ..Default::default()
    }
}

/// Read only the thermal-HAL per-sensor block from `dumpsys thermalservice` in
/// the current process. Requires `android.permission.DUMP` (grantable on lab
/// devices via `adb shell pm grant`); returns `None` when the permission is
/// absent, the exec is denied, or no sensor is reported. Used by the in-app
/// per-rep sampler — distinct from [`android_thermal`] (the CLI/desktop reader)
/// in that it forks only `dumpsys` and reads sensors alone: the app sources
/// status and headroom from the PowerManager SDK, not this path.
#[cfg(target_os = "android")]
pub fn android_hal_sensors() -> Option<Vec<thermal::AndroidTemperatureSensor>> {
    cmd_output("dumpsys", &["thermalservice"])
        .as_deref()
        .and_then(parse_android_hal_sensors)
}

/// Parse the `Thermal Status: N` line (`PowerManager` 0–6, coolest→hottest)
/// from `dumpsys thermalservice`.
#[cfg(any(target_os = "android", test))]
fn parse_android_thermal_status(stdout: &str) -> Option<thermal::AndroidThermalStatus> {
    let rest = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("Thermal Status:").map(str::trim))?;
    android_thermal_status_from_code(rest.parse::<i32>().ok()?)
}

/// Map a `PowerManager` `THERMAL_STATUS_*` ordinal (0–6, coolest→hottest) onto
/// [`thermal::AndroidThermalStatus`]; any out-of-range value → `None` (a defensive guard —
/// the in-app sampler reports `THERMAL_STATUS_NONE` (0), not a negative value,
/// when `getCurrentThermalStatus()` is unavailable). Shared by the `dumpsys`
/// parser and the in-app PowerManager sampler.
/// Compiled on all targets (pure logic) so the host-checked `pipette-android`
/// crate can call it from `JavaThermalSampler::status`.
pub fn android_thermal_status_from_code(code: i32) -> Option<thermal::AndroidThermalStatus> {
    Some(match code {
        0 => thermal::AndroidThermalStatus::None,
        1 => thermal::AndroidThermalStatus::Light,
        2 => thermal::AndroidThermalStatus::Moderate,
        3 => thermal::AndroidThermalStatus::Severe,
        4 => thermal::AndroidThermalStatus::Critical,
        5 => thermal::AndroidThermalStatus::Emergency,
        6 => thermal::AndroidThermalStatus::Shutdown,
        _ => return None,
    })
}

/// Parse the float from `cmd thermalservice headroom N` ("Headroom in N
/// seconds: 0.53"). `NaN` (rate-limited: the API refuses calls <~1 Hz) is
/// dropped to `None`.
#[cfg(any(target_os = "android", test))]
fn parse_android_headroom(stdout: &str) -> Option<f32> {
    let line = stdout.lines().find(|line| line.contains("Headroom"))?;
    let value: f32 = line.rsplit(':').next()?.trim().parse().ok()?;
    value.is_finite().then_some(value)
}

/// Parse the `Current temperatures from HAL:` block of `dumpsys
/// thermalservice` into per-sensor readings. Reads only the live block (a
/// stale `Cached temperatures:` block also appears). Sensors with no reading
/// report `-FLT_MAX` (~-3.4e38) and are dropped by the sane-band filter; the
/// `bcl_*` virtual sensors (mV/mA/%, not °C) are excluded via the type map.
#[cfg(any(target_os = "android", test))]
fn parse_android_hal_sensors(stdout: &str) -> Option<Vec<thermal::AndroidTemperatureSensor>> {
    let mut in_current = false;
    let sensors: Vec<thermal::AndroidTemperatureSensor> = stdout
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line == "Current temperatures from HAL:" {
                in_current = true;
                return None;
            }
            if !in_current {
                return None;
            }
            if !line.starts_with("Temperature{") {
                in_current = false; // left the live section
                return None;
            }
            let m_type: i32 = dumpsys_field(line, "mType=")?.parse().ok()?;
            let sensor_type = android_temperature_type(m_type)?; // skips bcl_*/unknown
            let value: f64 = dumpsys_field(line, "mValue=")?.parse().ok()?;
            if !value.is_finite() || !(-50.0..=200.0).contains(&value) {
                return None; // drops the -FLT_MAX no-reading sentinel
            }
            let m_status: i32 = dumpsys_field(line, "mStatus=")?.parse().ok()?;
            Some(thermal::AndroidTemperatureSensor {
                // A snapshot has no repetition of its own; the per-run builder
                // stamps the real iteration when flattening the series.
                iteration: 0,
                sensor_type: sensor_type.to_string(),
                name: dumpsys_field(line, "mName=")?.to_string(),
                celsius: value.round() as i32,
                throttling_status: android_throttling_severity(m_status),
            })
        })
        .collect();
    (!sensors.is_empty()).then_some(sensors)
}

/// Map an `android.hardware.thermal` `TemperatureType` ordinal onto its
/// snake_case wire name. The `BCL_*` virtual sensors (6/7/8 — voltage / current
/// / percentage, not °C) and `UNKNOWN` (-1) return `None` so they're excluded.
#[cfg(any(target_os = "android", test))]
fn android_temperature_type(m_type: i32) -> Option<&'static str> {
    Some(match m_type {
        0 => "cpu",
        1 => "gpu",
        2 => "battery",
        3 => "skin",
        4 => "usb_port",
        5 => "power_amplifier",
        9 => "npu",
        10 => "tpu",
        11 => "display",
        12 => "modem",
        13 => "soc",
        14 => "wifi",
        15 => "camera",
        16 => "flashlight",
        17 => "speaker",
        18 => "ambient",
        19 => "pogo",
        _ => return None,
    })
}

/// Map an `android.hardware.thermal` `ThrottlingSeverity` ordinal (0–6,
/// coolest→hottest) onto [`thermal::AndroidThrottlingSeverity`]; unknown → `None`.
#[cfg(any(target_os = "android", test))]
fn android_throttling_severity(
    m_status: i32,
) -> pipette_plan_types::thermal::AndroidThrottlingSeverity {
    use pipette_plan_types::thermal::AndroidThrottlingSeverity as S;
    match m_status {
        1 => S::Light,
        2 => S::Moderate,
        3 => S::Severe,
        4 => S::Critical,
        5 => S::Emergency,
        6 => S::Shutdown,
        _ => S::None,
    }
}

/// Extract the value of `key` (e.g. `"mType="`) from a `dumpsys`
/// `Temperature{..}` line, up to the next `,` or `}`.
#[cfg(any(target_os = "android", test))]
fn dumpsys_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        // NVIDIA also ships HDMI audio on the same PCI vendor id; skip it.
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

    // -- macOS --

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert_eq!(d.device_os_name.as_deref(), Some("macOS"));
        assert!(d
            .device_os_version
            .as_ref()
            .context("device_os_version")?
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit()));
        assert!(d
            .device_chip_model
            .as_ref()
            .context("device_chip_model")?
            .contains("Apple"));
        assert!(d.device_ram_bytes.context("device_ram_bytes")? >= 4 * 1024 * 1024 * 1024);
        Ok(())
    }

    // -- Linux --

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert!(d.device_os_name.is_some());
        assert!(d.device_ram_bytes.context("device_ram_bytes")? >= 1024 * 1024 * 1024);
        Ok(())
    }

    // -- Windows --

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_fields() -> anyhow::Result<()> {
        let d = PlatformDetector::detect();
        assert!(d
            .device_os_name
            .as_ref()
            .context("device_os_name")?
            .contains("Windows"));
        assert!(d.device_os_version.is_some());
        assert!(d.device_chip_model.is_some());
        assert!(d.device_ram_bytes.context("device_ram_bytes")? > 0);
        Ok(())
    }

    // -- Power state (detection never panics; values are environment-dependent) --

    #[test]
    fn detect_power_state_is_consistent() {
        // Detection must never panic; the battery level (if present) stays in
        // range. `power_state` validity is guaranteed by the typed enum.
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
            (Some(72), Some(thermal::DevicePowerState::NotCharging))
        );

        let charging = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t83%; charging; 0:42 remaining present: true";
        assert_eq!(
            parse_pmset_batt(charging),
            (Some(83), Some(thermal::DevicePowerState::Charging))
        );

        let charged = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t100%; charged; 0:00 remaining present: true";
        assert_eq!(
            parse_pmset_batt(charged),
            (
                Some(100),
                Some(thermal::DevicePowerState::PluggedInNotCharging)
            )
        );

        // Plugged in but holding charge (Optimized Battery Charging): the
        // state word is "not charging" — still on external power.
        let holding = "Now drawing from 'AC Power'\n \
            -InternalBattery-0 (id=4325123)\t80%; not charging; 0:00 remaining present: true";
        assert_eq!(
            parse_pmset_batt(holding),
            (
                Some(80),
                Some(thermal::DevicePowerState::PluggedInNotCharging)
            )
        );

        // Desktop: no battery line -> on AC power, not `None`.
        assert_eq!(
            parse_pmset_batt("Now drawing from 'AC Power'\nNo batteries available."),
            (None, Some(thermal::DevicePowerState::PluggedInNotCharging))
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

    // -- Linux /sys battery status mapping --

    #[cfg(target_os = "linux")]
    #[test]
    fn map_linux_battery_status_cases() {
        assert_eq!(
            map_linux_battery_status("Charging"),
            Some(thermal::DevicePowerState::Charging)
        );
        assert_eq!(
            map_linux_battery_status("Discharging"),
            Some(thermal::DevicePowerState::NotCharging)
        );
        assert_eq!(
            map_linux_battery_status("Full"),
            Some(thermal::DevicePowerState::PluggedInNotCharging)
        );
        assert_eq!(
            map_linux_battery_status("Not charging"),
            Some(thermal::DevicePowerState::PluggedInNotCharging)
        );
        assert_eq!(map_linux_battery_status("Unknown"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn map_linux_platform_profile_cases() {
        assert_eq!(map_linux_platform_profile("low-power"), Some(true));
        assert_eq!(map_linux_platform_profile("performance"), Some(false));
        assert_eq!(map_linux_platform_profile("balanced"), Some(false));
        // Ambiguous / unknown profiles don't guess.
        assert_eq!(map_linux_platform_profile("quiet"), None);
        assert_eq!(map_linux_platform_profile(""), None);
    }

    // -- Windows Win32_Battery parsing --

    #[cfg(target_os = "windows")]
    #[test]
    fn parse_windows_battery_cases() {
        assert_eq!(
            parse_windows_battery("72\t1"),
            (Some(72), Some(thermal::DevicePowerState::NotCharging))
        );
        assert_eq!(
            parse_windows_battery("90\t6"),
            (Some(90), Some(thermal::DevicePowerState::Charging))
        );
        assert_eq!(
            parse_windows_battery("100\t3"),
            (
                Some(100),
                Some(thermal::DevicePowerState::PluggedInNotCharging)
            )
        );
        // Low (4) / Critical (5) = running on battery.
        assert_eq!(
            parse_windows_battery("8\t5"),
            (Some(8), Some(thermal::DevicePowerState::NotCharging))
        );
        // Desktop sentinel -> on AC; a failed query yields empty -> None.
        assert_eq!(
            parse_windows_battery("NOBATT"),
            (None, Some(thermal::DevicePowerState::PluggedInNotCharging))
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
    #[case("0\n", Some(thermal::AppleThermalState::Nominal))]
    #[case("2", Some(thermal::AppleThermalState::Serious))]
    #[case("3\n", Some(thermal::AppleThermalState::Critical))]
    #[case("nominal", None)] // non-numeric
    #[case("7", None)] // out of range
    fn apple_thermal_state_maps_raw_values(
        #[case] raw: &str,
        #[case] want: Option<thermal::AppleThermalState>,
    ) {
        assert_eq!(parse_apple_thermal_state(raw), want);
    }

    /// Half-away-from-zero, and the result has to be *exactly* representable:
    /// the wire column is 32-bit, so a fractional value would reach a reader as
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
    #[case(
        "IsStatusOverride: false\nThermal Status: 2\nHAL Ready: true\n",
        Some(thermal::AndroidThermalStatus::Moderate)
    )]
    #[case("Thermal Status: 0\n", Some(thermal::AndroidThermalStatus::None))]
    #[case("no status line", None)]
    fn android_thermal_status_maps_levels(
        #[case] dump: &str,
        #[case] want: Option<thermal::AndroidThermalStatus>,
    ) {
        assert_eq!(parse_android_thermal_status(dump), want);
    }

    #[rstest]
    #[case(0, Some(thermal::AndroidThermalStatus::None))]
    #[case(1, Some(thermal::AndroidThermalStatus::Light))]
    #[case(2, Some(thermal::AndroidThermalStatus::Moderate))]
    #[case(3, Some(thermal::AndroidThermalStatus::Severe))]
    #[case(4, Some(thermal::AndroidThermalStatus::Critical))]
    #[case(5, Some(thermal::AndroidThermalStatus::Emergency))]
    #[case(6, Some(thermal::AndroidThermalStatus::Shutdown))]
    #[case(7, None)] // out of range
    #[case(-1, None)] // out of range (negative)
    fn android_thermal_status_from_code_maps_ordinals(
        #[case] code: i32,
        #[case] want: Option<thermal::AndroidThermalStatus>,
    ) {
        assert_eq!(android_thermal_status_from_code(code), want);
    }

    #[rstest]
    #[case("Headroom in 0 seconds: 0.53", Some(0.53))]
    #[case("Headroom in 5 seconds: 1.07\n", Some(1.07))]
    #[case("Headroom in 0 seconds: NaN", None)] // rate-limited
    #[case("unrelated", None)]
    fn android_headroom_parses_and_rejects_nan(#[case] out: &str, #[case] want: Option<f32>) {
        assert_eq!(parse_android_headroom(out), want);
    }

    /// Verbatim `dumpsys thermalservice` block from a Galaxy S25 Ultra
    /// (SM-S938W, Android 16 / API 36): a stale `Cached` block that must be
    /// ignored plus the live HAL block, including the `SUBBAT` sensor at the
    /// `0.0` floor and mixed sensor types.
    const S25_DUMPSYS: &str = "\
IsStatusOverride: false
Thermal Status: 1
Cached temperatures:
\tTemperature{mValue=99.0, mType=0, mName=AP, mStatus=3}
HAL Ready: true
Current temperatures from HAL:
\tTemperature{mValue=33.1, mType=0, mName=AP, mStatus=0}
\tTemperature{mValue=31.1, mType=2, mName=BAT, mStatus=1}
\tTemperature{mValue=34.3, mType=5, mName=PA, mStatus=0}
\tTemperature{mValue=32.7, mType=3, mName=SKIN, mStatus=0}
\tTemperature{mValue=-3.4028235E38, mType=1, mName=GPU, mStatus=0}
\tTemperature{mValue=0.0, mType=6, mName=BCL_V, mStatus=0}
\tTemperature{mValue=30.1, mType=4, mName=USB, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=1, mName=x}
";

    #[test]
    fn android_hal_sensors_parses_live_block_only() -> anyhow::Result<()> {
        let sensors = parse_android_hal_sensors(S25_DUMPSYS).context("live sensors present")?;
        // Live block only (cached 99°C AP dropped); GPU -FLT_MAX and the
        // BCL virtual sensor excluded; readings rounded to whole °C.
        assert_eq!(
            sensors,
            vec![
                thermal::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "cpu".into(),
                    name: "AP".into(),
                    celsius: 33,
                    throttling_status: thermal::AndroidThrottlingSeverity::None,
                },
                thermal::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "battery".into(),
                    name: "BAT".into(),
                    celsius: 31,
                    throttling_status: thermal::AndroidThrottlingSeverity::Light,
                },
                thermal::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "power_amplifier".into(),
                    name: "PA".into(),
                    celsius: 34,
                    throttling_status: thermal::AndroidThrottlingSeverity::None,
                },
                thermal::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "skin".into(),
                    name: "SKIN".into(),
                    celsius: 33,
                    throttling_status: thermal::AndroidThrottlingSeverity::None,
                },
                thermal::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "usb_port".into(),
                    name: "USB".into(),
                    celsius: 30,
                    throttling_status: thermal::AndroidThrottlingSeverity::None,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn android_hal_sensors_none_when_no_live_block() {
        assert_eq!(parse_android_hal_sensors("Thermal Status: 0\n"), None);
    }

    /// Live smoke test of the real `detect_thermal()` path (shells
    /// `swift`/`dumpsys`/`cmd`). Ignored by default so CI stays shell-free;
    /// run with `cargo test detect_thermal_smoke -- --ignored --nocapture`.
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
        // Die temp is private API and allowed to be absent, but on a Mac where
        // it resolves it must land in the *same* field iOS populates —
        // otherwise the two platforms' series diverge silently.
        #[cfg(target_os = "macos")]
        if let Some(celsius) = reading.apple_soc_temp_c {
            assert!(
                (-50.0..150.0).contains(&celsius),
                "implausible die temp {celsius}C",
            );
            // Whole degrees: a fractional reading here would not survive the
            // warehouse's f32 column intact (see the field's docs).
            assert_eq!(celsius.fract(), 0.0, "die temp {celsius}C was not rounded");
        }
    }
}
