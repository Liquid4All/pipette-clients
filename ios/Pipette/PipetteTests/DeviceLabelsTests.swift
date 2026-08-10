import Foundation
import Testing

@testable import Pipette

/// `device.json` — operator-supplied labels, stored apart from the registration.
///
/// The split is the crate's: labels are what a run reports, the registration is what a
/// submission needs, and keeping them separate lets a device be named without being
/// registered. These tests pin that independence rather than the file's contents.
struct DeviceLabelsTests {
    private func store() throws -> (IdentityStore, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("labels-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (IdentityStore(root: root, privateKeySource: { nil }), root)
    }

    /// Absent is the normal state, not an error: a device reports its probed values until
    /// an operator names it.
    @Test func anAbsentFileReadsAsNoLabels() throws {
        let (identity, _) = try store()

        #expect(identity.getDeviceLabels() == .empty)
    }

    @Test func labelsRoundTrip() throws {
        let (identity, _) = try store()
        try identity.putDeviceLabels(DeviceLabels(deviceName: "boston-17-pro-1",
                                                  deviceFormFactor: .phone))

        let read = identity.getDeviceLabels()

        #expect(read.deviceName == "boston-17-pro-1")
        #expect(read.deviceFormFactor == .phone)
    }

    /// The point of the split: naming a device does not require, or create, a registration.
    @Test func labelsAreIndependentOfTheRegistration() throws {
        let (identity, _) = try store()
        try identity.putDeviceLabels(DeviceLabels(deviceName: "unregistered"))

        #expect(identity.getDeviceLabels().deviceName == "unregistered")
        #expect(identity.getRegistration() == nil)
    }
}
