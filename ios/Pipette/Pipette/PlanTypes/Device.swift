import Foundation

// What the host *is* — the identity fields every benchmark submission carries,
// mirroring `pipette-plan-types/src/device.rs`.
//
// Shape and wire contract only, as upstream: `DeviceProbe.detectDeviceInfo()` is
// what produces the values, matching the crate's split between this module and
// `pipette_device::detect_device_info`.
//
// The crate spells the string fields as `TrimmedString`, a newtype that strips
// surrounding whitespace on construction. That exists for the Windows detector,
// whose `Win32_VideoController.Name` arrives padded and would otherwise reach the
// server as a row distinct from its trimmed twin. Every value here comes from a
// `sysctl` or a compiled-in mapping table, so there is nothing to trim and the
// newtype is not ported.

/// Physical form factor of the device running the benchmark — the crate's
/// `DeviceFormFactor`, lowercase on the wire to match the management server.
///
/// All six arms are ported even though a phone can only ever detect two of them:
/// this is the server's taxonomy, not a capability claim, and narrowing it would
/// make the enum look like an iOS invention rather than the crate's type.
///
/// The crate defaults to `desktop` when an old payload omits the field. There is no
/// counterpart here because nothing on iOS decodes a form factor — `DeviceProbe`
/// always sets an explicit value.
///
/// TODO: review — mirrors `device.rs:46`; six arms and the lowercase wire spelling
/// checked against it.
nonisolated enum DeviceFormFactor: String, Sendable, Codable, CaseIterable {
    case phone, tablet, laptop, desktop, server, embedded
}

/// Device metadata sent alongside every benchmark submission — the crate's `DeviceInfo`.
///
/// Four fields iOS cannot source stay `nil` and elide on encode, rather than being
/// dropped from the type: `deviceOsSecurityPatch` (no iOS API exposes one; Android
/// reads `ro.build.version.security_patch`) and the GPU/NPU pairs (a unified-memory
/// device has no separately addressable VRAM to report, and a `*_vram_bytes` without
/// its matching `*_model` is a `400`). Keeping them means the struct is the crate's
/// struct, and a platform that later gains a source has a field to fill.
///
/// TODO: review — mirrors `device.rs:74`; all twelve fields and their wire names
/// checked against it.
nonisolated struct DeviceInfo: Codable, Hashable, Sendable {
    var deviceName: String
    var deviceFormFactor: DeviceFormFactor
    var deviceOsName: String
    var deviceOsVersion: String
    /// Precise OS build id, finer-grained than `deviceOsVersion` (e.g. `22F76`).
    var deviceOsBuild: String?
    var deviceOsSecurityPatch: String?
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
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
    }
}
