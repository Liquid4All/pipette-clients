import Foundation
import Testing

@testable import Pipette

/// `device.json`: operator-supplied labels, stored apart from the registration (as the
/// crate does) so a device can be named without being registered.
struct DeviceLabelsTests {
    private func store() throws -> (ClientDeviceStore, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("labels-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return (ClientDeviceStore(root: root), root)
    }

    /// Absent is the normal state: a device reports its probed values until an operator
    /// names it.
    @Test func anAbsentFileReadsAsNoLabels() throws {
        let (device, _) = try store()

        #expect(try device.get() == .empty)
    }

    @Test func labelsReadWriteDeviceJson() throws {
        let (device, _) = try store()
        try device.put(DeviceLabels(deviceName: "boston-17-pro-1", deviceFormFactor: .phone))

        let read = try device.get()

        #expect(read.deviceName == "boston-17-pro-1")
        #expect(read.deviceFormFactor == .phone)
    }

    @Test func labelsAreIndependentOfTheRegistration() throws {
        let (device, root) = try store()
        try device.put(DeviceLabels(deviceName: "unregistered"))

        #expect(try device.get().deviceName == "unregistered")
        #expect(ClientRegistrationStore(root: root).get() == nil)
    }

    /// A blank name clears the name and must not erase the form factor.
    @Test(arguments: [
        ("renamed", nil, "renamed", .laptop),
        (nil, .server, "pinned", .server),
        ("   ", nil, nil, .laptop),
    ] as [(String?, DeviceFormFactor?, String?, DeviceFormFactor?)])
    func updateAppliesOnlyTheLabelsNamed(name: String?, formFactor: DeviceFormFactor?,
                                         wantName: String?,
                                         wantFormFactor: DeviceFormFactor?) throws {
        let (device, _) = try store()
        try device.put(DeviceLabels(deviceName: "pinned", deviceFormFactor: .laptop))

        let labels = try device.update(deviceName: name, deviceFormFactor: formFactor)

        #expect(labels.deviceName == wantName)
        #expect(labels.deviceFormFactor == wantFormFactor)
        #expect(try device.get() == labels)
    }

    @Test func updateNamingNoLabelIsRefused() throws {
        let (device, _) = try store()
        #expect(throws: IdentityError.setDeviceEmpty) {
            try device.update(deviceName: nil, deviceFormFactor: nil)
        }
        // Refused before any write, so an absent file stays absent.
        #expect(try device.get() == .empty)
    }

    /// Read as empty, a corrupt file would have `set-device` rewrite the one label it was
    /// asked to change and silently drop the other.
    @Test func labelsSurfaceParseErrorOnMalformedFile() throws {
        let (device, root) = try store()
        try Data("{not json".utf8).write(to: root.appendingPathComponent("device.json"))

        #expect(throws: DecodingError.self) { try device.get() }
    }

    /// Mirrors the crate's `DeviceLabels::supplied_name`.
    @Test(arguments: [("  lab 7 ", "lab 7"), ("", nil), ("   ", nil)] as [(String, String?)])
    func aSuppliedNameIsTrimmedAndBlankIsNone(raw: String, expected: String?) {
        #expect(DeviceLabels.suppliedName(raw) == expected)
    }
}
