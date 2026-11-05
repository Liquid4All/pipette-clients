import Foundation

// `deny_unknown_fields` for the flag wire forms: serde gives it to the crate, `Decodable`
// does not.

/// A field the wire form does not declare. `group` is the spec key the group sat under,
/// since the caller reports refusals per group.
///
/// Stricter than the crate: a field the crate declares and an iOS ref does not is refused
/// whatever its value, so `"mmap": null` or `"raw": []` (unset to the crate) is unknown
/// here. Only a hand-written body spells an unset knob that way.
nonisolated struct UnknownFlagField: Error {
    let group: String?
    let name: String
}

nonisolated func rejectUnknownFields<
    Key: CodingKey & CaseIterable & RawRepresentable
>(
    _ decoder: Decoder,
    keys: Key.Type
) throws where Key.RawValue == String {
    let container = try decoder.container(keyedBy: AnyFlagKey.self)
    let known = Set(Key.allCases.map(\.rawValue))
    for key in container.allKeys where !known.contains(key.stringValue) {
        throw UnknownFlagField(
            group: decoder.codingPath.first?.stringValue,
            name: key.stringValue
        )
    }
}

private nonisolated struct AnyFlagKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue _: Int) { nil }
}
