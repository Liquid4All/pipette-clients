import Foundation

/// Operator-supplied labels for this device (the crate's `identity/device.json`), kept
/// apart from the registration so a device can be labelled without being registered. An
/// absent field leaves the probed value standing.
nonisolated struct DeviceLabels: Codable, Sendable, Equatable {
    var deviceName: String?
    var deviceFormFactor: DeviceFormFactor?

    enum CodingKeys: String, CodingKey {
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
    }

    static let empty = DeviceLabels()

    /// Trimmed, with blank meaning no value (the crate's `DeviceLabels::supplied_name`).
    /// `auth set-device` reads nil as "unset this label", `register` as "keep the prior one".
    static func suppliedName(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

/// `identity/device.json`: the crate's `ClientDeviceStore`.
nonisolated struct ClientDeviceStore: Sendable {
    private static let deviceName = "device.json"

    /// The workspace `identity/` directory.
    let root: URL

    /// Empty when the file is absent. A file that does not parse is an error rather than
    /// empty labels: `set-device` reads before it writes, and would otherwise silently drop
    /// the label it was not asked to change.
    func get() throws -> DeviceLabels {
        let labels: DeviceLabels? = try IdentityFiles.readJson(root, Self.deviceName)
        return labels ?? .empty
    }

    func put(_ labels: DeviceLabels) throws {
        try IdentityFiles.writeJson(root, Self.deviceName, labels)
    }

    /// Apply the labels named and return what `device.json` now holds. A blank `deviceName`
    /// unsets the label; naming neither throws ``IdentityError/setDeviceEmpty``.
    func update(deviceName: String?, deviceFormFactor: DeviceFormFactor?) throws -> DeviceLabels {
        guard deviceName != nil || deviceFormFactor != nil else { throw IdentityError.setDeviceEmpty }
        var labels = try get()
        if let deviceName { labels.deviceName = DeviceLabels.suppliedName(deviceName) }
        if let deviceFormFactor { labels.deviceFormFactor = deviceFormFactor }
        try put(labels)
        return labels
    }

    func clear() throws {
        try IdentityFiles.remove(root, Self.deviceName)
    }
}
