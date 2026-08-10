import Foundation
import UIKit

/// What this device *is*, and what state it is in — iOS's `pipette-device`.
///
/// The crate splits detection from shape: `pipette_device::probe` reads the host and
/// returns `pipette_plan_types::device::DeviceInfo`. The same split holds here, which
/// is why the payload struct lives in `PlanTypes/Device.swift` and only the reading
/// lives in this file. Per-field detectors carry the crate's `detect*` verb.
///
/// Every value comes from a `sysctl`, `ProcessInfo`, or `UIDevice`, none of which can
/// fail — so `detectDeviceInfo()` returns a value where the crate's returns a `Result`,
/// and there is no `device.json` label-override layer.
///
/// MainActor-isolated by the project default, which `UIDevice` requires.
enum DeviceProbe {

    /// This device's full identity, the crate's `detect_device_info`. Single-sourced so
    /// the profile patch and the submission payload cannot disagree about the device
    /// they describe.
    static func detectDeviceInfo() -> DeviceInfo {
        DeviceInfo(
            deviceName: detectDeviceName(),
            deviceFormFactor: detectFormFactor(),
            deviceOsName: detectOsName(),
            deviceOsVersion: detectOsVersion(),
            deviceOsBuild: detectOsBuild(),
            // No iOS API exposes a security-patch level; see `DeviceInfo`.
            deviceOsSecurityPatch: nil,
            deviceChipModel: detectChipModel(),
            deviceRamBytes: detectRamBytes(),
            deviceGpuModel: nil,
            deviceGpuVramBytes: nil,
            deviceNpuModel: nil,
            deviceNpuVramBytes: nil)
    }

    /// The volatile run-environment state, the crate's `detect_power_state`. Separate
    /// from `detectDeviceInfo()` for the reason the crate separates them: this changes
    /// between two runs on one device, so it describes a measurement rather than the
    /// host, and rides on submissions only — never on the profile.
    static func detectPowerState() -> PowerState {
        PowerState(
            batteryLevel: detectBatteryLevel(),
            powerState: detectDevicePowerState(),
            powerSaveMode: detectPowerSaveMode())
    }

    // MARK: - Identity

    private static let machineIdentifier: String = {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }()

    /// The human-readable device model name (e.g. "iPhone 16 Pro Max"), the crate's
    /// `device_name`. Falls back to the raw machine identifier if unknown.
    static func detectDeviceName() -> String {
        modelNameMapping[machineIdentifier] ?? machineIdentifier
    }

    /// The SoC model (e.g. "Apple A18 Pro").
    /// Falls back to the raw machine identifier if unknown.
    static func detectChipModel() -> String {
        chipModelMapping[machineIdentifier] ?? machineIdentifier
    }

    /// `.tablet` for iPad, `.phone` otherwise — the only two arms this platform reaches.
    static func detectFormFactor() -> DeviceFormFactor {
        machineIdentifier.hasPrefix("iPad") ? .tablet : .phone
    }

    /// The OS name: "iPadOS" for iPad, "iOS" otherwise.
    static func detectOsName() -> String {
        machineIdentifier.hasPrefix("iPad") ? "iPadOS" : "iOS"
    }

    /// The OS version string (e.g. "18.4" or "26.4").
    static func detectOsVersion() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return v.patchVersion == 0
            ? "\(v.majorVersion).\(v.minorVersion)"
            : "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// The OS build identifier (e.g. "22F76") from `kern.osversion` — finer-grained
    /// than `detectOsVersion()`. Nil if the sysctl is unreadable or empty.
    static func detectOsBuild() -> String? {
        var size = 0
        guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var build = [CChar](repeating: 0, count: size)
        guard sysctlbyname("kern.osversion", &build, &size, nil, 0) == 0 else { return nil }
        let value = String(cString: build)
        return value.isEmpty ? nil : value
    }

    /// Total system RAM in bytes.
    static func detectRamBytes() -> UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    // MARK: - Run-environment power state

    // Captured with each result so on-battery runs (where the PMIC can cap CPU clocks
    // to avoid voltage sag — distinct from thermal throttling) and Low Power Mode runs
    // can be filtered/flagged after the fact.

    /// Battery charge percent (0–100), or nil when unavailable (e.g. the
    /// simulator reports -1). Enables battery monitoring as a side effect.
    static func detectBatteryLevel() -> Int32? {
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel  // 0.0–1.0, or -1.0 if unknown
        guard level >= 0 else { return nil }
        return Int32((level * 100).rounded())
    }

    /// Charging state, or nil when unknown. Mirrors Android's `DeviceInfo.powerState`.
    static func detectDevicePowerState() -> DevicePowerState? {
        UIDevice.current.isBatteryMonitoringEnabled = true
        switch UIDevice.current.batteryState {
        case .charging: return .charging
        case .unplugged: return .notCharging
        case .full: return .pluggedInNotCharging
        case .unknown: return nil
        @unknown default: return nil
        }
    }

    /// Whether Low Power Mode is enabled (can lower CPU/GPU clocks).
    static func detectPowerSaveMode() -> Bool {
        ProcessInfo.processInfo.isLowPowerModeEnabled
    }

    // MARK: - Private mappings

    private static let modelNameMapping: [String: String] = [
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
        // iPhone SE
        "iPhone14,6": "iPhone SE (3rd gen)",
        "iPhone12,8": "iPhone SE (2nd gen)",
    ]

    private static let chipModelMapping: [String: String] = [
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
        // iPhone SE
        "iPhone14,6": "Apple A15 Bionic",
        "iPhone12,8": "Apple A13 Bionic",
    ]
}
