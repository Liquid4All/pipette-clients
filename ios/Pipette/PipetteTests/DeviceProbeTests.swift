import Foundation
import Testing

@testable import Pipette

/// `Device/Probe.swift` against `pipette-device/src/probe.rs`: the detected identity, the
/// operator's label overrides, and the die reading as recorded.
@MainActor
struct DeviceProbeTests {

    /// This platform reaches two of the six form factors, and the OS name must agree with
    /// the form factor: the server slugifies the pair into `device:` and `os:` matching flags.
    @Test func detectDeviceInfoWithoutOverridesSucceeds() {
        let device = detectDeviceInfo(deviceName: nil, deviceFormFactor: nil)

        #expect([.phone, .tablet].contains(device.deviceFormFactor))
        #expect(device.deviceOsName == (device.deviceFormFactor == .tablet ? "iPadOS" : "iOS"))
        #expect(!device.deviceName.isEmpty)
        #expect(!device.deviceChipModel.isEmpty)
        #expect(device.deviceRamBytes > 0)
        #expect(device.deviceCpuArch == detectCpuArch())
        #expect(device.deviceOsFamily == "iOS")
        // Nothing on iOS sources these; `DeviceInfo` carries them for the crate's shape.
        #expect(device.deviceOsSecurityPatch == nil)
        #expect(device.deviceGpuModel == nil)
        #expect(device.deviceNpuModel == nil)
    }

    @Test func detectsTheArchitectureThisBuildRunsAs() {
        #if arch(arm64)
        #expect(detectCpuArch() == "arm64")
        #elseif arch(x86_64)
        #expect(detectCpuArch() == "x86_64")
        #endif
    }

    @Test func detectDeviceInfoAppliesOverrides() {
        let info = detectDeviceInfo(deviceName: "custom-name", deviceFormFactor: .server)
        #expect(info.deviceName == "custom-name")
        #expect(info.deviceFormFactor == .server)
        #expect(info.deviceRamBytes > 0)
    }

    @Test func detectDeviceInfoTrimsUserProvidedName() {
        let info = detectDeviceInfo(deviceName: "  custom-name  ", deviceFormFactor: .server)
        #expect(info.deviceName == "custom-name")
    }

    /// A blank label is no label. The crate's empty filter would fail detection here;
    /// iOS always has a detected name to fall back to.
    @Test func aBlankNameLabelFallsBackToTheDetectedName() {
        let info = detectDeviceInfo(deviceName: " \n ", deviceFormFactor: nil)
        #expect(info.deviceName == detectDeviceName())
        #expect(info.deviceFormFactor == detectFormFactor())
    }
}
