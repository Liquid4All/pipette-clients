import Foundation
import Testing

@testable import Pipette

/// The device identity shape and its wire form, against
/// `pipette-plan-types::device::DeviceInfo`. The crate's own
/// `device_info_serializes_flat` / `device_form_factor_round_trip` are the counterparts.
struct DeviceInfoTests {

    private func encoded(_ device: DeviceInfo) throws -> [String: Any] {
        let data = try JSONEncoder().encode(device)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func deviceInfoEncodesTheCratesWireKeys() throws {
        let json = try encoded(deviceInfoFixture())

        #expect(json["device_name"] as? String == "iPhone 17 Pro")
        #expect(json["device_form_factor"] as? String == "phone")
        #expect(json["device_os_name"] as? String == "iOS")
        #expect(json["device_os_version"] as? String == "26.4")
        #expect(json["device_os_build"] as? String == "22F76")
        #expect(json["device_chip_model"] as? String == "Apple A19 Pro")
        #expect(json["device_ram_bytes"] as? UInt64 == 8_589_934_592)
    }

    /// Absent, not null: the crate's optional fields carry `skip_serializing_if`, and a
    /// literal `null` is a different value to the server than an omitted key.
    @Test func unsourceableFieldsElideRatherThanEncodeNull() throws {
        let json = try encoded(deviceInfoFixture(osBuild: nil))

        for key in [
            "device_os_build", "device_os_security_patch",
            "device_gpu_model", "device_gpu_vram_bytes",
            "device_npu_model", "device_npu_vram_bytes",
        ] {
            #expect(json[key] == nil, "\(key) should elide when nil")
        }
    }

    @Test func deviceInfoRoundTrips() throws {
        let device = deviceInfoFixture(formFactor: .tablet)
        let decoded = try JSONDecoder().decode(DeviceInfo.self, from: JSONEncoder().encode(device))
        #expect(decoded == device)
    }

    /// The lowercase spellings the management server's enum accepts. A typo here is a
    /// rejected submission, which is why the type exists at all.
    @Test func formFactorWireSpellingsMatchTheServer() {
        #expect(DeviceFormFactor.allCases.map(\.rawValue)
            == ["phone", "tablet", "laptop", "desktop", "server", "embedded"])
    }

    @Test func powerStateWireSpellingsMatchTheServer() {
        #expect(DevicePowerState.charging.rawValue == "charging")
        #expect(DevicePowerState.notCharging.rawValue == "not_charging")
        #expect(DevicePowerState.pluggedInNotCharging.rawValue == "plugged_in_not_charging")
    }

    /// This platform reaches exactly two of the six arms, and the OS name has to agree
    /// with the form factor — the pair is what the server slugifies into `device:` and
    /// `os:` matching flags.
    @Test @MainActor func theProbeReportsAPhoneOrATablet() {
        let device = DeviceProbe.detectDeviceInfo()

        #expect([.phone, .tablet].contains(device.deviceFormFactor))
        #expect(device.deviceOsName == (device.deviceFormFactor == .tablet ? "iPadOS" : "iOS"))
        #expect(!device.deviceName.isEmpty)
        #expect(!device.deviceChipModel.isEmpty)
        #expect(device.deviceRamBytes > 0)
        // Nothing on iOS sources these; `DeviceInfo` carries them for the crate's shape.
        #expect(device.deviceOsSecurityPatch == nil)
        #expect(device.deviceGpuModel == nil)
        #expect(device.deviceNpuModel == nil)
    }
}
