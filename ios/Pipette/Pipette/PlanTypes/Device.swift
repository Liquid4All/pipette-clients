import Foundation

// The host identity fields every submission carries: pipette-mgmt's
// `crates/pipette-plan-types/src/device.rs`. `detectDeviceInfo()` produces the values.
//
// The crate's string fields are `TrimmedString` (for padded Windows WMI values); nothing
// here arrives padded (an operator label is trimmed in `detectDeviceInfo`), so the newtype
// is not ported.

/// The crate's `DeviceFormFactor`. All six arms are ported though a phone detects two:
/// it is the server's taxonomy. The crate's `desktop` default is not: a phone always
/// detects its own, and the operator label in `device.json` decodes as an optional.
nonisolated enum DeviceFormFactor: String, Sendable, Codable, CaseIterable {
    case phone, tablet, laptop, desktop, server, embedded
}

/// The crate's `DeviceInfo`. iOS leaves `deviceOsSecurityPatch` (no API exposes one) and
/// the GPU/NPU fields (unified memory has no separate VRAM) `nil`; a `*_vram_bytes`
/// without its `*_model` is a `400`.
nonisolated struct DeviceInfo: Codable, Hashable, Sendable {
    var deviceName: String
    var deviceFormFactor: DeviceFormFactor
    var deviceOsName: String
    var deviceOsVersion: String
    /// e.g. `22F76`.
    var deviceOsBuild: String?
    var deviceOsSecurityPatch: String?
    /// The CPU architecture as Swift names it (`arm64`). The server derives the
    /// `arch:` flag from it.
    var deviceCpuArch: String?
    /// The OS family (`iOS`). The server derives the `os_family:` flag from it.
    var deviceOsFamily: String?
    var deviceChipModel: String
    var deviceRamBytes: UInt64
    var deviceGpuModel: String?
    var deviceGpuVramBytes: UInt64?
    var deviceNpuModel: String?
    var deviceNpuVramBytes: UInt64?

    enum CodingKeys: String, CodingKey {
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceOsBuild = "device_os_build"
        case deviceOsSecurityPatch = "device_os_security_patch"
        case deviceCpuArch = "device_cpu_arch"
        case deviceOsFamily = "device_os_family"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
    }
}

/// The crate's `CapabilityFlag`: an eligibility token a plan's `requires` lists and a
/// client advertises (`os:ios`, `runtime:llama_cpp:b9999`, `job_retry`). The server
/// matches on this canonical form and answers a non-canonical one with a `400`; reserved
/// namespaces are the server's to check.
nonisolated struct CapabilityFlag: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        !value.isEmpty && !value.contains(where: \.isWhitespace) && value.lowercased() == value
    }
    static let rejection = ModelError.invalidCapabilityFlag
}
