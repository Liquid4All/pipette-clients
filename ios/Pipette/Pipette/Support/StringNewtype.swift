import Foundation

// Validated newtypes (`ClientId`, `ServerUrl`, `PrivateKey`, ...) do not adopt these:
// their initializers throw, and a key must stay non-`Codable` and render redacted.

/// Wraps a single `String` and serializes as that bare string. Members are
/// `nonisolated` to satisfy `nonisolated` conformers (needed for `Sendable`).
protocol StringNewtype: Codable {
    nonisolated var value: String { get }
    nonisolated init(_ value: String)
}

extension StringNewtype {
    nonisolated init(from decoder: any Decoder) throws {
        self.init(try decoder.singleValueContainer().decode(String.self))
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

/// An opaque identifier. Only literals coerce (for tests and previews); a wrong-typed
/// variable is still a compile error, which is the point of these types.
protocol StringId: StringNewtype, Comparable, CustomStringConvertible, ExpressibleByStringLiteral {}

extension StringId {
    nonisolated init(stringLiteral value: String) { self.init(value) }
    nonisolated static func < (lhs: Self, rhs: Self) -> Bool { lhs.value < rhs.value }
    nonisolated var description: String { value }
}
