import Foundation

/// Operator-supplied labels for this device, stored apart from the registration —
/// the crate's `identity/device.json`.
///
/// Separate for the reason the crate gives: a benchmark run needs labels, submission
/// needs the registration, and splitting them lets a device be labelled without being
/// registered. Either field may be absent, in which case the auto-detected value from
/// `DeviceProbe` stands.
nonisolated struct DeviceLabels: Codable, Sendable, Equatable {
    var deviceName: String?
    var deviceFormFactor: DeviceFormFactor?

    enum CodingKeys: String, CodingKey {
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
    }

    /// Nothing set — what an absent file means, and what a device reports before an
    /// operator names it.
    static let empty = DeviceLabels()
}
