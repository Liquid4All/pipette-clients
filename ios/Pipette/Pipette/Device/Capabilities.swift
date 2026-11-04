import Foundation

// Hardware capabilities for client matching, as `pipette-device/src/capabilities.rs`. The
// runtime and client-version flags are added by `Client/Worker/Profile.swift`.

/// Case-insensitive; first match wins, in the crate's order.
nonisolated func cpuVendor(_ brand: String) -> String? {
    let vendors: [(needle: String, vendor: String)] = [
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
    ]
    let brand = brand.lowercased()
    return vendors.first { brand.contains($0.needle) }?.vendor
}

nonisolated func hardwareFlags(chipModel: String, hasGpu: Bool, hasAvx512f: Bool) -> [String] {
    var flags = cpuVendor(chipModel).map { ["cpu_vendor:\($0)"] } ?? []
    if hasGpu { flags.append("accel:gpu") }
    if hasAvx512f { flags.append("cpu_feature:avx512f") }
    return flags.sorted()
}

/// `accel:gpu` follows the probe, not the silicon: the probe reports no GPU model, so an
/// iPhone advertises none, as a Mac does under the crate.
nonisolated func hardwareCapabilities(device: DeviceInfo) -> [String] {
    hardwareFlags(chipModel: cpuBrand(device: device), hasGpu: device.deviceGpuModel != nil,
                  hasAvx512f: false)
}

/// `device_chip_model`, or `"Apple"` when it names no vendor. Unlike the crate's: iOS has
/// no SoC-name API, so a device newer than `detectChipModel`'s table reports `hw.machine`
/// (`"iPhone19,1"`), and dropping `cpu_vendor:apple` would stop it matching apple-gated plans
/// and void its queue standing. Every iOS device is Apple silicon.
private nonisolated func cpuBrand(device: DeviceInfo) -> String {
    cpuVendor(device.deviceChipModel) == nil ? "Apple" : device.deviceChipModel
}
