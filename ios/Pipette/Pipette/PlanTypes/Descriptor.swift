import CryptoKit
import Foundation

// Canonical descriptor form and the digest that addresses it — the Swift counterpart of
// `crates/pipette-plan-types/src/descriptor.rs`.
//
// A descriptor is the full typed value rendered as JSON. `digest` hashes its *canonical*
// form — object keys sorted recursively, no insignificant whitespace — so the id survives
// a client formatting its payload differently, or a type reordering its fields.
//
// The rule is shared with `pipette-mgmt`'s `canonical_json`, which is what the warehouse
// stores as `model_descriptor_sha256`. Reproducing it here means a prefix read off this
// client addresses the same artifact the warehouse groups by, rather than a second,
// private id.

nonisolated enum Descriptor {
    /// How many leading hex chars a listing shows.
    static let digestDisplayLength = 12
    /// The shortest prefix accepted when addressing an artifact by digest. 32 bits is far
    /// past collision range for a local store, and shorter reads as a typo.
    static let digestMinPrefixLength = 8

    /// Canonical, compact JSON: object keys sorted recursively, no insignificant
    /// whitespace. Array order is preserved — it is semantically significant.
    static func canonicalize(_ value: Any) -> String {
        var out = ""
        write(value, into: &out)
        return out
    }

    /// Hex SHA-256 over the canonical form of `value`.
    static func digest(_ value: some Encodable) throws -> String {
        let data = try JSONEncoder().encode(value)
        let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        return sha256Hex(canonicalize(object))
    }

    /// Hex SHA-256 of a string — the hash `digest` applies, exposed so a test can anchor
    /// on it the way the crate's does.
    static func sha256Hex(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// `digest` truncated for display.
    static func shortDigest(_ digest: String) -> String {
        String(digest.prefix(digestDisplayLength))
    }

    private static func write(_ value: Any, into out: inout String) {
        switch value {
        case let object as [String: Any]:
            out += "{"
            for (index, key) in object.keys.sorted().enumerated() {
                if index > 0 { out += "," }
                out += jsonString(key) + ":"
                write(object[key] ?? NSNull(), into: &out)
            }
            out += "}"
        case let array as [Any]:
            out += "["
            for (index, item) in array.enumerated() {
                if index > 0 { out += "," }
                write(item, into: &out)
            }
            out += "]"
        case let string as String:
            out += jsonString(string)
        case is NSNull:
            out += "null"
        case let number as NSNumber:
            // `NSNumber` is how `JSONSerialization` hands back both bools and numbers, and
            // only its ObjC type tells them apart — a bool written as `1` would hash
            // differently from the same descriptor on the Rust side.
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                out += number.boolValue ? "true" : "false"
            } else {
                out += number.stringValue
            }
        default:
            out += jsonString(String(describing: value))
        }
    }

    private static func jsonString(_ value: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [value],
                                                     options: [.withoutEscapingSlashes]),
              let array = String(data: data, encoding: .utf8)
        else { return "\"\(value)\"" }
        // `["…"]` → `"…"`, so the escaping rules come from the encoder rather than here.
        return String(array.dropFirst().dropLast())
    }
}
