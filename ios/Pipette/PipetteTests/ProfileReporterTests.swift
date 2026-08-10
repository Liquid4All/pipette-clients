import Foundation
import Testing

@testable import Pipette

/// Pins the device-profile payload against the server's schema
/// (mgmt `httpapi.md` §2.2.1 / §2.4.1).
///
/// Every assertion here is about a `400` the server would return, or a field
/// that has no profile column to land in.
@MainActor
struct ProfileReporterTests {
    private func encoded(_ update: DeviceProfileUpdate) throws -> [String: Any] {
        let data = try JSONEncoder().encode(update)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func carriesTheProfileFieldsTheServerAccepts() throws {
        let json = try encoded(ProfileReporter.profile())
        #expect(json["device_os_name"] as? String == "iOS" || json["device_os_name"] as? String == "iPadOS")
        #expect((json["device_name"] as? String)?.isEmpty == false)
        #expect((json["device_os_version"] as? String)?.isEmpty == false)
        #expect((json["device_chip_model"] as? String)?.isEmpty == false)
        #expect(((json["device_ram_bytes"] as? NSNumber)?.int64Value ?? 0) > 0)
    }

    // `device_os_version` present without `device_os_name` is a 400.
    @Test func pairsOsVersionWithOsName() throws {
        let json = try encoded(ProfileReporter.profile())
        #expect(json["device_os_name"] != nil)
        #expect(json["device_os_version"] != nil)
    }

    // Anything outside this set is a 400.
    @Test func reportsAFormFactorFromTheServersEnum() throws {
        let json = try encoded(ProfileReporter.profile())
        let allowed = ["phone", "tablet", "laptop", "desktop", "server", "embedded"]
        #expect(allowed.contains(try #require(json["device_form_factor"] as? String)))
    }

    // A `*_vram_bytes` without its matching `*_model` is a 400, and a null would
    // leave the stored value unchanged anyway — so absent, never null.
    @Test func omitsGpuAndNpuFieldsEntirely() throws {
        let json = try encoded(ProfileReporter.profile())
        for key in ["device_gpu_model", "device_gpu_vram_bytes", "device_npu_model", "device_npu_vram_bytes"] {
            #expect(json[key] == nil, "\(key) must not be sent")
        }
    }

    // Detected by DeviceProbe and sent on benchmark submissions, but not profile
    // fields. The run-environment values additionally change minute to minute,
    // and a profile change voids the client's queue standing.
    @Test func omitsFieldsTheProfileSchemaHasNoColumnFor() throws {
        let json = try encoded(ProfileReporter.profile())
        for key in [
            "device_os_build", "device_os_security_patch",
            "device_battery_level", "device_power_state", "device_power_save_mode",
        ] {
            #expect(json[key] == nil, "\(key) is not a profile field")
        }
    }

    // Set-granular on the server: a present value replaces the stored set
    // wholesale, so the payload always carries the full current set.
    @Test func sendsTheFullCapabilitySet() throws {
        let json = try encoded(ProfileReporter.profile())
        #expect(json["capabilities"] as? [String] == Capabilities.flags())
    }

    // The payload always carries `client_details`. Registration has a field of
    // the same name and drops this copy in `RegisterRequest.encode(to:)`, so the
    // collision is handled there rather than by the caller choosing a variant.
    @Test func alwaysCarriesClientDetails() throws {
        #expect(try encoded(ProfileReporter.profile())["client_details"] != nil)
    }

    // The tests above run against the host, so they can only assert shape. This one
    // pins the DeviceInfo -> DeviceProfileUpdate mapping itself (the crate's
    // `build_profile_update`) against a fixed snapshot, so a field wired to the wrong
    // source is visible rather than merely "non-empty".
    @Test func mapsEachFieldFromTheDeviceSnapshot() throws {
        let json = try encoded(ProfileReporter.profile(device: deviceInfoFixture(formFactor: .tablet)))

        #expect(json["device_name"] as? String == "iPhone 17 Pro")
        #expect(json["client_details"] as? String == "iPhone 17 Pro")
        #expect(json["device_form_factor"] as? String == "tablet")
        #expect(json["device_os_name"] as? String == "iOS")
        #expect(json["device_os_version"] as? String == "26.4")
        #expect(json["device_chip_model"] as? String == "Apple A19 Pro")
        #expect((json["device_ram_bytes"] as? NSNumber)?.uint64Value == 8_589_934_592)
    }
}
