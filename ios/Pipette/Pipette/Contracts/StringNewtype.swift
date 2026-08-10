import Foundation

// Shared scaffolding for the single-property `String`-wrapper identifier types
// (`ServerURL`, `ClientID`, `JobId`, `CellId`). A conformer declares only
// `let value: String` and `init(_:)`; the transparent single-value `Codable`
// (encode/decode as the bare string, like `EvalId`) and — for id types — the
// ordering, printing, and literal-construction boilerplate live here, so the
// concrete types don't re-hand-write four identical copies.
//
// `PrivateKeyHex` deliberately does NOT adopt these: it must stay non-`Codable`
// and render redacted, so it keeps its own minimal definition.

/// A value type that wraps a single `String` and serializes transparently as
/// that bare string. Extension members are `nonisolated` so they satisfy the
/// requirements of `nonisolated` conformers (needed for `Sendable`).
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

/// A `StringNewtype` used as an opaque identifier: ordered, printable as its raw
/// value, and constructible from a string literal. Only *literals* coerce (handy
/// for tests and previews); passing a wrong-typed *variable* is still a compile
/// error, which is the whole point of these types.
protocol StringId: StringNewtype, Comparable, CustomStringConvertible, ExpressibleByStringLiteral {}

extension StringId {
    nonisolated init(stringLiteral value: String) { self.init(value) }
    nonisolated static func < (lhs: Self, rhs: Self) -> Bool { lhs.value < rhs.value }
    nonisolated var description: String { value }
}
