import Foundation
import UIKit

// What this device is and what state it is in, iOS's `pipette-device`. The payload shape
// lives in `PlanTypes/Device.swift`; this file only reads. No detector here can fail, so
// `detectDeviceInfo` returns a value where the crate's returns a `Result`.

/// Single-sourced so the profile patch and the submission payload cannot disagree.
///
/// The operator's labels win over the detected values when set. A blank name counts as
/// unset and falls back to the detected one, where the crate would fail detection.
nonisolated func detectDeviceInfo(deviceName: String?, deviceFormFactor: DeviceFormFactor?) -> DeviceInfo {
    DeviceInfo(
        deviceName: nonBlank(deviceName) ?? detectDeviceName(),
        deviceFormFactor: deviceFormFactor ?? detectFormFactor(),
        deviceOsName: detectOsName(),
        deviceOsVersion: detectOsVersion(),
        deviceOsBuild: detectOsBuild(),
        // No iOS API exposes a security-patch level; see `DeviceInfo`.
        deviceOsSecurityPatch: nil,
        deviceCpuArch: detectCpuArch(),
        deviceOsFamily: detectOsFamily(),
        deviceChipModel: detectChipModel(),
        deviceRamBytes: detectRamBytes(),
        deviceGpuModel: nil,
        deviceGpuVramBytes: nil,
        deviceNpuModel: nil,
        deviceNpuVramBytes: nil)
}

private nonisolated func nonBlank(_ value: String?) -> String? {
    guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
          !trimmed.isEmpty else { return nil }
    return trimmed
}

/// Separate from `detectDeviceInfo`: this changes between runs on one device, so it
/// describes a measurement, not the host, and rides on submissions only, never the profile.
@MainActor func detectPowerState() -> PowerReading {
    PowerReading(
        batteryLevel: detectBatteryLevel(),
        powerState: detectDevicePowerState(),
        powerSaveMode: detectPowerSaveMode())
}

// MARK: - Identity

private nonisolated let machineIdentifier: String = {
    var size = 0
    sysctlbyname("hw.machine", nil, &size, nil, 0)
    var machine = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.machine", &machine, &size, nil, 0)
    return String(cString: machine)
}()

/// E.g. "iPhone 16 Pro Max"; the raw machine identifier if unknown.
nonisolated func detectDeviceName() -> String {
    modelNameMapping[machineIdentifier] ?? machineIdentifier
}

/// E.g. "Apple A18 Pro"; the raw machine identifier if unknown.
nonisolated func detectChipModel() -> String {
    chipModelMapping[machineIdentifier] ?? machineIdentifier
}

nonisolated func detectFormFactor() -> DeviceFormFactor {
    machineIdentifier.hasPrefix("iPad") ? .tablet : .phone
}

/// Returns the CPU architecture of this build: `arm64` on a device and on an
/// Apple-silicon simulator, `x86_64` on an Intel simulator. A build for any other
/// architecture fails to compile here, so the probe never reports a value it did not
/// detect.
nonisolated func detectCpuArch() -> String {
    #if arch(arm64)
    "arm64"
    #elseif arch(x86_64)
    "x86_64"
    #endif
}

/// Returns the operating system (OS) family. iPhone and iPad share one family, `iOS`.
/// The field `deviceOsName` separates the two. The server maps `iOS` and `iPadOS` to
/// `os_family:ios`.
nonisolated func detectOsFamily() -> String {
    "iOS"
}

nonisolated func detectOsName() -> String {
    machineIdentifier.hasPrefix("iPad") ? "iPadOS" : "iOS"
}

/// E.g. "18.4"; the patch only when non-zero.
nonisolated func detectOsVersion() -> String {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    return v.patchVersion == 0
        ? "\(v.majorVersion).\(v.minorVersion)"
        : "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
}

/// E.g. "22F76", finer-grained than `detectOsVersion()`.
nonisolated func detectOsBuild() -> String? {
    var size = 0
    guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var build = [CChar](repeating: 0, count: size)
    guard sysctlbyname("kern.osversion", &build, &size, nil, 0) == 0 else { return nil }
    let value = String(cString: build)
    return value.isEmpty ? nil : value
}

nonisolated func detectRamBytes() -> UInt64 {
    ProcessInfo.processInfo.physicalMemory
}

// MARK: - Run-environment power state

// Captured with each result so on-battery runs (where the PMIC can cap CPU clocks to avoid
// voltage sag, apart from thermal throttling) and Low Power Mode runs can be filtered.

/// Percent, or nil when unavailable (the simulator reports -1). Enables battery
/// monitoring as a side effect.
@MainActor private func detectBatteryLevel() -> Int32? {
    UIDevice.current.isBatteryMonitoringEnabled = true
    let level = UIDevice.current.batteryLevel  // 0.0–1.0, or -1.0 if unknown
    guard level >= 0 else { return nil }
    return Int32((level * 100).rounded())
}

/// Mirrors Android's `DeviceInfo.powerState`.
@MainActor private func detectDevicePowerState() -> DevicePowerState? {
    UIDevice.current.isBatteryMonitoringEnabled = true
    switch UIDevice.current.batteryState {
    case .charging: return .charging
    case .unplugged: return .notCharging
    case .full: return .pluggedInNotCharging
    case .unknown: return nil
    @unknown default: return nil
    }
}

/// Low Power Mode can lower CPU/GPU clocks.
nonisolated func detectPowerSaveMode() -> Bool {
    ProcessInfo.processInfo.isLowPowerModeEnabled
}

// MARK: - Private mappings

private nonisolated let modelNameMapping: [String: String] = [
    // iPhone 18 series
    "iPhone19,2": "iPhone 18 Pro",
    "iPhone19,3": "iPhone 18 Pro Max",
    "iPhone19,4": "iPhone Duo",
    "iPhone19,7": "iPhone 18 Pro Max",
    // iPhone 17 series
    "iPhone18,1": "iPhone 17 Pro",
    "iPhone18,2": "iPhone 17 Pro Max",
    "iPhone18,3": "iPhone 17",
    "iPhone18,4": "iPhone Air",
    "iPhone18,5": "iPhone 17e",
    // iPhone 16 series
    "iPhone17,1": "iPhone 16 Pro",
    "iPhone17,2": "iPhone 16 Pro Max",
    "iPhone17,3": "iPhone 16",
    "iPhone17,4": "iPhone 16 Plus",
    "iPhone17,5": "iPhone 16e",
    // iPhone 15 series
    "iPhone16,1": "iPhone 15 Pro",
    "iPhone16,2": "iPhone 15 Pro Max",
    "iPhone15,4": "iPhone 15",
    "iPhone15,5": "iPhone 15 Plus",
    // iPhone 14 series
    "iPhone15,2": "iPhone 14 Pro",
    "iPhone15,3": "iPhone 14 Pro Max",
    "iPhone14,7": "iPhone 14",
    "iPhone14,8": "iPhone 14 Plus",
    // iPhone 13 series
    "iPhone14,2": "iPhone 13 Pro",
    "iPhone14,3": "iPhone 13 Pro Max",
    "iPhone14,5": "iPhone 13",
    "iPhone14,4": "iPhone 13 mini",
    // iPhone 12 series
    "iPhone13,3": "iPhone 12 Pro",
    "iPhone13,4": "iPhone 12 Pro Max",
    "iPhone13,2": "iPhone 12",
    "iPhone13,1": "iPhone 12 mini",
    // iPhone 11 series
    "iPhone12,3": "iPhone 11 Pro",
    "iPhone12,5": "iPhone 11 Pro Max",
    "iPhone12,1": "iPhone 11",
    // iPhone X series
    "iPhone11,2": "iPhone XS",
    "iPhone11,4": "iPhone XS Max",
    "iPhone11,6": "iPhone XS Max",
    "iPhone11,8": "iPhone XR",
    // iPhone SE
    "iPhone14,6": "iPhone SE (3rd gen)",
    "iPhone12,8": "iPhone SE (2nd gen)",
    // iPad, Apple A12 Bionic
    "iPad11,1": "iPad mini (5th gen)",
    "iPad11,2": "iPad mini (5th gen)",
    "iPad11,3": "iPad Air (3rd gen)",
    "iPad11,4": "iPad Air (3rd gen)",
    "iPad11,6": "iPad (8th gen)",
    "iPad11,7": "iPad (8th gen)",
    // iPad, Apple A12X Bionic
    "iPad8,1": "iPad Pro 11-inch (1st gen)",
    "iPad8,2": "iPad Pro 11-inch (1st gen)",
    "iPad8,3": "iPad Pro 11-inch (1st gen)",
    "iPad8,4": "iPad Pro 11-inch (1st gen)",
    "iPad8,5": "iPad Pro 12.9-inch (3rd gen)",
    "iPad8,6": "iPad Pro 12.9-inch (3rd gen)",
    "iPad8,7": "iPad Pro 12.9-inch (3rd gen)",
    "iPad8,8": "iPad Pro 12.9-inch (3rd gen)",
    // iPad, Apple A12Z Bionic
    "iPad8,9": "iPad Pro 11-inch (2nd gen)",
    "iPad8,10": "iPad Pro 11-inch (2nd gen)",
    "iPad8,11": "iPad Pro 12.9-inch (4th gen)",
    "iPad8,12": "iPad Pro 12.9-inch (4th gen)",
    // iPad, Apple A13 Bionic
    "iPad12,1": "iPad (9th gen)",
    "iPad12,2": "iPad (9th gen)",
    // iPad, Apple A14 Bionic
    "iPad13,1": "iPad Air (4th gen)",
    "iPad13,2": "iPad Air (4th gen)",
    "iPad13,18": "iPad (10th gen)",
    "iPad13,19": "iPad (10th gen)",
    // iPad, Apple A15 Bionic
    "iPad14,1": "iPad mini (6th gen)",
    "iPad14,2": "iPad mini (6th gen)",
    // iPad, Apple A16
    "iPad15,7": "iPad (A16)",
    "iPad15,8": "iPad (A16)",
    // iPad, Apple A17 Pro
    "iPad16,1": "iPad mini (A17 Pro)",
    "iPad16,2": "iPad mini (A17 Pro)",
    // iPad, Apple M1
    "iPad13,4": "iPad Pro 11-inch (3rd gen)",
    "iPad13,5": "iPad Pro 11-inch (3rd gen)",
    "iPad13,6": "iPad Pro 11-inch (3rd gen)",
    "iPad13,7": "iPad Pro 11-inch (3rd gen)",
    "iPad13,8": "iPad Pro 12.9-inch (5th gen)",
    "iPad13,9": "iPad Pro 12.9-inch (5th gen)",
    "iPad13,10": "iPad Pro 12.9-inch (5th gen)",
    "iPad13,11": "iPad Pro 12.9-inch (5th gen)",
    "iPad13,16": "iPad Air (5th gen)",
    "iPad13,17": "iPad Air (5th gen)",
    // iPad, Apple M2
    "iPad14,3": "iPad Pro 11-inch (4th gen)",
    "iPad14,4": "iPad Pro 11-inch (4th gen)",
    "iPad14,5": "iPad Pro 12.9-inch (6th gen)",
    "iPad14,6": "iPad Pro 12.9-inch (6th gen)",
    "iPad14,8": "iPad Air 11-inch (M2)",
    "iPad14,9": "iPad Air 11-inch (M2)",
    "iPad14,10": "iPad Air 13-inch (M2)",
    "iPad14,11": "iPad Air 13-inch (M2)",
    // iPad, Apple M3
    "iPad15,3": "iPad Air 11-inch (M3)",
    "iPad15,4": "iPad Air 11-inch (M3)",
    "iPad15,5": "iPad Air 13-inch (M3)",
    "iPad15,6": "iPad Air 13-inch (M3)",
    // iPad, Apple M4
    "iPad16,3": "iPad Pro 11-inch (M4)",
    "iPad16,4": "iPad Pro 11-inch (M4)",
    "iPad16,5": "iPad Pro 13-inch (M4)",
    "iPad16,6": "iPad Pro 13-inch (M4)",
    "iPad16,8": "iPad Air 11-inch (M4)",
    "iPad16,9": "iPad Air 11-inch (M4)",
    "iPad16,10": "iPad Air 13-inch (M4)",
    "iPad16,11": "iPad Air 13-inch (M4)",
    // iPad, Apple M5
    "iPad17,1": "iPad Pro 11-inch (M5)",
    "iPad17,2": "iPad Pro 11-inch (M5)",
    "iPad17,3": "iPad Pro 13-inch (M5)",
    "iPad17,4": "iPad Pro 13-inch (M5)",
]

private nonisolated let chipModelMapping: [String: String] = [
    // iPhone 18 series
    "iPhone19,2": "Apple A20 Pro",
    "iPhone19,3": "Apple A20 Pro",
    "iPhone19,4": "Apple A20 Pro",
    "iPhone19,7": "Apple A20 Pro",
    // iPhone 17 series
    "iPhone18,1": "Apple A19 Pro",
    "iPhone18,2": "Apple A19 Pro",
    "iPhone18,3": "Apple A19",
    "iPhone18,4": "Apple A19 Pro",
    "iPhone18,5": "Apple A19",
    // iPhone 16 series
    "iPhone17,1": "Apple A18 Pro",
    "iPhone17,2": "Apple A18 Pro",
    "iPhone17,3": "Apple A18",
    "iPhone17,4": "Apple A18",
    "iPhone17,5": "Apple A18",
    // iPhone 15 series
    "iPhone16,1": "Apple A17 Pro",
    "iPhone16,2": "Apple A17 Pro",
    "iPhone15,4": "Apple A16 Bionic",
    "iPhone15,5": "Apple A16 Bionic",
    // iPhone 14 series
    "iPhone15,2": "Apple A16 Bionic",
    "iPhone15,3": "Apple A16 Bionic",
    "iPhone14,7": "Apple A15 Bionic",
    "iPhone14,8": "Apple A15 Bionic",
    // iPhone 13 series
    "iPhone14,2": "Apple A15 Bionic",
    "iPhone14,3": "Apple A15 Bionic",
    "iPhone14,5": "Apple A15 Bionic",
    "iPhone14,4": "Apple A15 Bionic",
    // iPhone 12 series
    "iPhone13,3": "Apple A14 Bionic",
    "iPhone13,4": "Apple A14 Bionic",
    "iPhone13,2": "Apple A14 Bionic",
    "iPhone13,1": "Apple A14 Bionic",
    // iPhone 11 series
    "iPhone12,3": "Apple A13 Bionic",
    "iPhone12,5": "Apple A13 Bionic",
    "iPhone12,1": "Apple A13 Bionic",
    // iPhone X series
    "iPhone11,2": "Apple A12 Bionic",
    "iPhone11,4": "Apple A12 Bionic",
    "iPhone11,6": "Apple A12 Bionic",
    "iPhone11,8": "Apple A12 Bionic",
    // iPhone SE
    "iPhone14,6": "Apple A15 Bionic",
    "iPhone12,8": "Apple A13 Bionic",
    // iPad, Apple A12 Bionic
    "iPad11,1": "Apple A12 Bionic",
    "iPad11,2": "Apple A12 Bionic",
    "iPad11,3": "Apple A12 Bionic",
    "iPad11,4": "Apple A12 Bionic",
    "iPad11,6": "Apple A12 Bionic",
    "iPad11,7": "Apple A12 Bionic",
    // iPad, Apple A12X Bionic
    "iPad8,1": "Apple A12X Bionic",
    "iPad8,2": "Apple A12X Bionic",
    "iPad8,3": "Apple A12X Bionic",
    "iPad8,4": "Apple A12X Bionic",
    "iPad8,5": "Apple A12X Bionic",
    "iPad8,6": "Apple A12X Bionic",
    "iPad8,7": "Apple A12X Bionic",
    "iPad8,8": "Apple A12X Bionic",
    // iPad, Apple A12Z Bionic
    "iPad8,9": "Apple A12Z Bionic",
    "iPad8,10": "Apple A12Z Bionic",
    "iPad8,11": "Apple A12Z Bionic",
    "iPad8,12": "Apple A12Z Bionic",
    // iPad, Apple A13 Bionic
    "iPad12,1": "Apple A13 Bionic",
    "iPad12,2": "Apple A13 Bionic",
    // iPad, Apple A14 Bionic
    "iPad13,1": "Apple A14 Bionic",
    "iPad13,2": "Apple A14 Bionic",
    "iPad13,18": "Apple A14 Bionic",
    "iPad13,19": "Apple A14 Bionic",
    // iPad, Apple A15 Bionic
    "iPad14,1": "Apple A15 Bionic",
    "iPad14,2": "Apple A15 Bionic",
    // iPad, Apple A16
    "iPad15,7": "Apple A16",
    "iPad15,8": "Apple A16",
    // iPad, Apple A17 Pro
    "iPad16,1": "Apple A17 Pro",
    "iPad16,2": "Apple A17 Pro",
    // iPad, Apple M1
    "iPad13,4": "Apple M1",
    "iPad13,5": "Apple M1",
    "iPad13,6": "Apple M1",
    "iPad13,7": "Apple M1",
    "iPad13,8": "Apple M1",
    "iPad13,9": "Apple M1",
    "iPad13,10": "Apple M1",
    "iPad13,11": "Apple M1",
    "iPad13,16": "Apple M1",
    "iPad13,17": "Apple M1",
    // iPad, Apple M2
    "iPad14,3": "Apple M2",
    "iPad14,4": "Apple M2",
    "iPad14,5": "Apple M2",
    "iPad14,6": "Apple M2",
    "iPad14,8": "Apple M2",
    "iPad14,9": "Apple M2",
    "iPad14,10": "Apple M2",
    "iPad14,11": "Apple M2",
    // iPad, Apple M3
    "iPad15,3": "Apple M3",
    "iPad15,4": "Apple M3",
    "iPad15,5": "Apple M3",
    "iPad15,6": "Apple M3",
    // iPad, Apple M4
    "iPad16,3": "Apple M4",
    "iPad16,4": "Apple M4",
    "iPad16,5": "Apple M4",
    "iPad16,6": "Apple M4",
    "iPad16,8": "Apple M4",
    "iPad16,9": "Apple M4",
    "iPad16,10": "Apple M4",
    "iPad16,11": "Apple M4",
    // iPad, Apple M5
    "iPad17,1": "Apple M5",
    "iPad17,2": "Apple M5",
    "iPad17,3": "Apple M5",
    "iPad17,4": "Apple M5",
]
