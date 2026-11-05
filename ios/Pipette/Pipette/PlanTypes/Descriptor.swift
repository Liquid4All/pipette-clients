import CryptoKit
import Foundation

// Canonical descriptor JSON and its digest: pipette-mgmt's
// `crates/pipette-plan-types/src/descriptor.rs`.
//
// Must match the crate's `canonical_json` byte for byte: the warehouse stores its hash as
// `model_descriptor_sha256`, so a digest from this client addresses the same artifact.
// The server hashes a model without a vault source's `custom_label`, so
// `modelDescriptorDigest` clears it first.

/// The crate's `DIGEST_DISPLAY_LEN`.
nonisolated let digestDisplayLen = 12

/// Compact JSON with object keys sorted recursively; array order is kept.
nonisolated func canonicalJson(_ value: Any) -> String {
    var out = ""
    write(value, into: &out)
    return out
}

nonisolated func descriptorDigest(_ value: some Encodable) throws -> String {
    let data = try JSONEncoder().encode(value)
    let object = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    return sha256Hex(canonicalJson(object))
}

/// Exposed so a test can anchor on it, as the crate's does.
nonisolated func sha256Hex(_ value: String) -> String {
    SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
}

nonisolated func shortDigest(_ digest: String) -> String {
    String(digest.prefix(digestDisplayLen))
}

private nonisolated func write(_ value: Any, into out: inout String) {
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
        // `JSONSerialization` returns bools as `NSNumber` too; only the CF type tells them
        // apart, and a bool written as `1` would hash differently from the Rust side.
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            out += number.boolValue ? "true" : "false"
        } else if CFNumberIsFloatType(number as CFNumber) {
            out += float(number.doubleValue)
        } else {
            out += number.stringValue
        }
    default:
        out += jsonString(String(describing: value))
    }
}

/// A float as `serde_json` writes one: `NSNumber.stringValue` would hash `1.0` as `1`.
/// Swift's shortest form keeps the `.0`; the exponent is respelled from `1e+20` / `1e-07`
/// to serde's `1e20` / `1e-7`. The libraries switch to an exponent at slightly different
/// magnitudes, which descriptors (almost entirely strings) do not reach.
private nonisolated func float(_ value: Double) -> String {
    // serde_json writes a non-finite float as `null`.
    guard value.isFinite else { return "null" }
    let shortest = "\(value)"
    guard let e = shortest.firstIndex(where: { $0 == "e" || $0 == "E" }) else { return shortest }
    var exponent = shortest[shortest.index(after: e)...]
    let negative = exponent.first == "-"
    if exponent.first == "+" || negative { exponent = exponent.dropFirst() }
    let digits = exponent.drop(while: { $0 == "0" })
    return shortest[..<e] + "e" + (negative ? "-" : "") + (digits.isEmpty ? "0" : digits)
}

private nonisolated func jsonString(_ value: String) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: [value],
                                                 options: [.withoutEscapingSlashes]),
          let array = String(data: data, encoding: .utf8)
    else { return "\"\(value)\"" }
    // Strip `[` `]` so the escaping rules come from the encoder rather than here.
    return String(array.dropFirst().dropLast())
}
