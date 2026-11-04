//! Hardware capabilities for client matching.

use pipette_plan_types::DeviceInfo;

/// The canonical vendor token a CPU brand string names, case-insensitively.
/// The needles cover both marketing brands (`Snapdragon`, `Tensor G`) and the
/// manufacturer ids Android reports (`QTI`, `Google`): either can reach here.
fn cpu_vendor(brand: &str) -> Option<&'static str> {
    const CPU_VENDORS: &[(&str, &str)] = &[
        ("intel", "intel"),
        ("amd", "amd"),
        ("ryzen", "amd"),
        ("epyc", "amd"),
        ("threadripper", "amd"),
        ("apple", "apple"),
        ("qualcomm", "qualcomm"),
        ("snapdragon", "qualcomm"),
        ("qti", "qualcomm"),
        ("exynos", "samsung"),
        ("samsung", "samsung"),
        ("mediatek", "mediatek"),
        ("dimensity", "mediatek"),
        ("nvidia", "nvidia"),
        ("tegra", "nvidia"),
        ("google", "google"),
        ("tensor g", "google"),
        ("ampere", "ampere"),
        ("hisilicon", "hisilicon"),
        ("kirin", "hisilicon"),
        ("rockchip", "rockchip"),
    ];

    let brand = brand.to_ascii_lowercase();
    CPU_VENDORS
        .iter()
        .find_map(|(needle, vendor)| brand.contains(needle).then_some(*vendor))
}

fn hardware_flags(chip_model: &str, has_gpu: bool, has_avx512f: bool) -> Vec<String> {
    let mut flags = cpu_vendor(chip_model)
        .map(|vendor| format!("cpu_vendor:{vendor}"))
        .into_iter()
        .chain(has_gpu.then(|| "accel:gpu".to_string()))
        .chain(has_avx512f.then(|| "cpu_feature:avx512f".to_string()))
        .collect::<Vec<_>>();
    flags.sort_unstable();
    flags
}

/// The hardware capabilities this host advertises.
///
/// `accel:gpu` follows the probe, not the hardware: the probe finds an NVIDIA
/// or AMD card on Linux and a non-virtual display adapter on Windows, and no
/// GPU on macOS or Android, so a Mac advertises no `accel:gpu`.
pub fn hardware_capabilities(device: &DeviceInfo) -> Vec<String> {
    let has_gpu = device.device_gpu_model.is_some();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let has_avx512f = std::is_x86_feature_detected!("avx512f");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let has_avx512f = false;

    hardware_flags(&cpu_brand(device), has_gpu, has_avx512f)
}

/// A modern Android kernel drops the `Hardware` line from `/proc/cpuinfo`, so
/// `device_chip_model` is often a bare part number such as `SM8650`.
/// `ro.soc.manufacturer` comes first, but only when it names a known vendor:
/// some devices set it to a value outside the vocabulary.
#[cfg(target_os = "android")]
fn cpu_brand(device: &DeviceInfo) -> String {
    crate::probe::soc_manufacturer()
        .filter(|manufacturer| cpu_vendor(manufacturer).is_some())
        .unwrap_or_else(|| device.device_chip_model.as_ref().to_owned())
}

#[cfg(not(target_os = "android"))]
fn cpu_brand(device: &DeviceInfo) -> String {
    device.device_chip_model.as_ref().to_owned()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::CapabilityFlag;

    use super::*;

    #[rstest]
    #[case(
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
        false,
        false,
        vec!["cpu_vendor:intel"]
    )]
    #[case(
        "AMD EPYC 7763 64-Core Processor",
        false,
        false,
        vec!["cpu_vendor:amd"]
    )]
    #[case("Apple M4 Pro", false, false, vec!["cpu_vendor:apple"])]
    #[case(
        "Snapdragon(R) X Elite - X1E80100",
        false,
        false,
        vec!["cpu_vendor:qualcomm"]
    )]
    #[case(
        "Qualcomm Technologies, Inc SM8550",
        false,
        false,
        vec!["cpu_vendor:qualcomm"]
    )]
    #[case("QTI", false, false, vec!["cpu_vendor:qualcomm"])]
    #[case("Google", false, false, vec!["cpu_vendor:google"])]
    #[case("Tensor G3", false, false, vec!["cpu_vendor:google"])]
    #[case(
        "NVIDIA Jetson AGX Orin Developer Kit",
        false,
        false,
        vec!["cpu_vendor:nvidia"]
    )]
    #[case("Raspberry Pi 5 Model B Rev 1.1", false, false, vec![])]
    #[case("Some Unbranded CPU", false, false, vec![])]
    #[case(
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
        false,
        true,
        vec!["cpu_feature:avx512f", "cpu_vendor:intel"]
    )]
    #[case(
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
        true,
        false,
        vec!["accel:gpu", "cpu_vendor:intel"]
    )]
    #[case(
        "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
        true,
        true,
        vec!["accel:gpu", "cpu_feature:avx512f", "cpu_vendor:intel"]
    )]
    fn hardware_flags_are_canonical_and_sorted(
        #[case] chip_model: &str,
        #[case] has_gpu: bool,
        #[case] has_avx512f: bool,
        #[case] expected: Vec<&str>,
    ) {
        let expected = expected.into_iter().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(hardware_flags(chip_model, has_gpu, has_avx512f), expected);
    }

    #[test]
    fn every_hardware_flag_is_a_canonical_capability() -> anyhow::Result<()> {
        for chip_model in [
            "Intel Xeon",
            "AMD Ryzen",
            "Apple M4",
            "Qualcomm Snapdragon",
            "Samsung Exynos",
            "MediaTek Dimensity",
            "NVIDIA Tegra",
            "Google Tensor G",
            "Ampere Altra",
            "Hisilicon Kirin",
            "Rockchip RK3588",
        ] {
            for (has_gpu, has_avx512f) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                for flag in hardware_flags(chip_model, has_gpu, has_avx512f) {
                    CapabilityFlag::try_new(flag)?;
                }
            }
        }
        Ok(())
    }

    /// No `accel:gpu`: the macOS probe reports no GPU model.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_reports_apple_cpu_without_gpu_or_avx512() -> anyhow::Result<()> {
        let device = crate::detect_device_info(None, None)?;
        let flags = hardware_capabilities(&device);
        assert!(flags.iter().any(|flag| flag == "cpu_vendor:apple"));
        assert!(!flags.iter().any(|flag| flag == "accel:gpu"));
        assert!(!flags.iter().any(|flag| flag == "cpu_feature:avx512f"));
        Ok(())
    }

    /// A virtualized host reports a brand such as `QEMU Virtual CPU`, which
    /// names no vendor, and there the right answer is no `cpu_vendor:` flag.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn linux_x86_cpu_vendor_matches_the_brand_string() -> anyhow::Result<()> {
        let device = crate::detect_device_info(None, None)?;
        let flags = hardware_capabilities(&device);
        let reported = flags
            .iter()
            .find_map(|flag| flag.strip_prefix("cpu_vendor:"));
        assert_eq!(reported, cpu_vendor(device.device_chip_model.as_ref()));
        Ok(())
    }
}
