import Foundation

// `pipette-device/src/readings.rs`, with the iOS keys of the mgmt repo's
// `docs/design/device-telemetry.md`: each facility is a key holding what the OS returned,
// unconverted, and one that failed or does not exist is absent.

/// The crate's `serde_json::Value`, narrowed to the kinds the readers here produce.
nonisolated enum JsonValue: Encodable, Equatable, Sendable {
    case int(Int)
    case double(Double)
    case string(String)
    case array([JsonValue])
    case object([String: JsonValue])

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .int(value): try container.encode(value)
        case let .double(value): try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        }
    }
}

/// One sample point's readings, split as the submission splits them.
nonisolated struct DeviceReadings: Equatable, Sendable {
    /// What the device's sensors reported, by facility.
    var hardware: [String: JsonValue] = [:]
    /// What the OS reported about its own throttling, by facility.
    var osThrottling: [String: JsonValue] = [:]
}

/// This device's readings now: `ProcessInfo.thermalState` always, and every IOHID
/// temperature sensor on a `PIPETTE_PRIVATE_THERMAL` build. One IPC round trip per sensor,
/// so call it outside a timed window.
nonisolated func detectDeviceReadings() -> DeviceReadings {
    var readings = DeviceReadings()
    // The raw value: 0 nominal, 1 fair, 2 serious, 3 critical.
    readings.osThrottling["apple_process_info_thermal_state"] =
        .int(ProcessInfo.processInfo.thermalState.rawValue)
    if let sensors = pipette_temperature_sensors() {
        readings.hardware["apple_iohid_temperature_sensors"] = appleIohidTemperatureSensors(sensors)
    }
    return readings
}

/// Each IOHID temperature service under IOHID's own keys; a missing one stays absent.
nonisolated func appleIohidTemperatureSensors(_ sensors: [[String: Any]]) -> JsonValue {
    .array(sensors.map { sensor in
        var record: [String: JsonValue] = [:]
        if let product = sensor["Product"] as? String {
            record["Product"] = .string(product)
        }
        if let level = sensor["TemperatureLevel"] as? NSNumber {
            record["TemperatureLevel"] = .double(level.doubleValue)
        }
        return .object(record)
    })
}
