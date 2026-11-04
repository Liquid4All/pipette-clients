import Foundation

// The value layer's primitives: pipette-mgmt's
// `crates/pipette-plan-types/src/text.rs` and `numbers.rs`, plus `ValidatedString` (the
// stand-in for the crate's `nutype` wrappers) and `ModelError` (one enum where the crate
// declares an error per type).

// MARK: - Validated string newtypes
//
// Validates at construction and on decode, so an invalid value is unrepresentable. Unlike
// `StringNewtype`, the constructor throws. Extension members are `nonisolated` so they
// satisfy `nonisolated` conformers under main-actor default isolation.

protocol ValidatedString: Codable, Hashable, Sendable, CustomStringConvertible {
    nonisolated var value: String { get }
    /// Trusted constructor, only called after `validate` has passed.
    nonisolated init(validated value: String)
    nonisolated static func validate(_ value: String) -> Bool
    nonisolated static var rejection: (String) -> ModelError { get }
    /// Runs before `validate`, as nutype's `sanitize(with = …)` does. Identity by default.
    nonisolated static func sanitize(_ value: String) -> String
}

extension ValidatedString {
    nonisolated static func sanitize(_ value: String) -> String { value }

    nonisolated init(_ value: String) throws {
        let value = Self.sanitize(value)
        guard Self.validate(value) else { throw Self.rejection(value) }
        self.init(validated: value)
    }

    nonisolated var description: String { value }

    nonisolated init(from decoder: any Decoder) throws {
        try self.init(try decoder.singleValueContainer().decode(String.self))
    }

    nonisolated func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

/// Rust's `str::trim` set, which leaves U+200B in place, so a value refused by the crate
/// is refused here too.
nonisolated func rustTrim(_ value: String) -> String {
    value.trimmingCharacters(in: .whitespacesAndNewlines.subtracting(["\u{200B}"]))
}

/// The crate's `NonEmptyString`.
nonisolated struct NonEmptyString: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { !value.isEmpty }
    static let rejection = ModelError.emptyValue
}

/// The crate's `NonEmptyTrimmedString`: trimmed, then refused when nothing is left.
nonisolated struct NonEmptyTrimmedString: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { !value.isEmpty }
    static func sanitize(_ value: String) -> String { rustTrim(value) }
    static let rejection = ModelError.emptyValue
}

/// A count of at least one, in its warehouse column's `Int32` width; past the max it is
/// refused, not wrapped.
nonisolated struct PositiveCount: Hashable, Comparable, Sendable, Codable,
    CustomStringConvertible {
    let value: Int32

    init(_ value: Int) throws(PositiveCountError) {
        guard let narrowed = Int32(exactly: value), narrowed > 0 else {
            throw PositiveCountError(value: value)
        }
        self.value = narrowed
    }

    /// Trusted constructor, for a value already known to be in range.
    init(validated value: Int32) { self.value = value }

    var description: String { "\(value)" }

    static func < (lhs: PositiveCount, rhs: PositiveCount) -> Bool { lhs.value < rhs.value }

    init(from decoder: any Decoder) throws {
        try self.init(Int(try decoder.singleValueContainer().decode(Int32.self)))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

nonisolated struct PositiveCountError: Error, Equatable, LocalizedError {
    let value: Int

    var errorDescription: String? { "count must be between 1 and \(Int32.max), got \(value)" }
}

/// A count that may be zero, such as the tokens of an empty prompt; capped at `Int32.max`
/// like ``PositiveCount``.
nonisolated struct NonNegativeCount: Hashable, Comparable, Sendable, Codable,
    CustomStringConvertible {
    let value: Int32

    init(_ value: Int) throws(NonNegativeCountError) {
        guard let narrowed = Int32(exactly: value), narrowed >= 0 else {
            throw NonNegativeCountError(value: value)
        }
        self.value = narrowed
    }

    /// Trusted constructor, for a value already known to be in range.
    init(validated value: Int32) { self.value = value }

    /// The crate's `Default`.
    static let zero = NonNegativeCount(validated: 0)

    var description: String { "\(value)" }

    static func < (lhs: NonNegativeCount, rhs: NonNegativeCount) -> Bool {
        lhs.value < rhs.value
    }

    init(from decoder: any Decoder) throws {
        try self.init(Int(try decoder.singleValueContainer().decode(Int32.self)))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

nonisolated struct NonNegativeCountError: Error, Equatable, LocalizedError {
    let value: Int

    var errorDescription: String? { "count must be between 0 and \(Int32.max), got \(value)" }
}

// MARK: - Construction failures

/// Every `ValidatedString` rejection, plus a repo slug with no `/` and an unknown model
/// `type`/`source` tag, in one enum where the crate declares an error per type. Wraps the
/// offending string, not an underlying `Error`, so it stays `Equatable`.
nonisolated enum ModelError: Error, Equatable {
    case repoMissingSeparator(String)
    case invalidOrg(String)
    case invalidRepoName(String)
    case invalidSubpath(String)
    case invalidRelativePath(String)
    case invalidAbsolutePath(String)
    case invalidResourceUrl(String)
    case invalidRevision(String)
    case invalidSha256(String)
    case emptyValue(String)
    /// Payload-free: the rejected input is the token.
    case invalidAuthToken
    case unknownModelType(String)
    case invalidBenchmarkId(String)
    case invalidClientId(String)
    case invalidCapabilityFlag(String)
    case invalidGitOid(String)
    case invalidRepositoryUrl(String)
    case invalidDatasetPath(String)
    case invalidVaultResourceId(String)
    case invalidVaultPrincipalId(String)
    case invalidVaultPath(String)
}

extension ModelError: LocalizedError {
    var errorDescription: String? {
        switch self {
        // The crate's wording, which names the shape rather than echoing the input.
        case .repoMissingSeparator: "HF repo must be in the form `org/repo_name`"
        case let .invalidOrg(s): "invalid HF org: \(s)"
        case let .invalidRepoName(s): "invalid HF repo name: \(s)"
        case let .invalidSubpath(s): "invalid repo subpath: \(s)"
        case let .invalidRelativePath(s): "path must be store-relative: \(s)"
        case let .invalidAbsolutePath(s): "path must be absolute: \(s)"
        case let .invalidResourceUrl(s): "not an http(s)/file URL: \(s)"
        case let .invalidRevision(s): "invalid HF revision: \(s)"
        case let .invalidSha256(s): "invalid sha256 digest: \(s)"
        case .emptyValue: "expected a non-empty value"
        // No interpolation: the rejected value is the token.
        case .invalidAuthToken: "the HF access token is empty"
        case let .unknownModelType(s): "unknown model type: \(s)"
        case let .invalidBenchmarkId(s):
            "benchmark id must match [A-Za-z0-9][A-Za-z0-9_.-]*: \(escapedQuoted(s))"
        case let .invalidClientId(s):
            "client id must match [A-Za-z0-9_-]+: \(escapedQuoted(s))"
        case let .invalidCapabilityFlag(s):
            "capability flag must be non-empty, lowercase, with no whitespace: \(s)"
        case let .invalidGitOid(s): "not a full git object id: \(s)"
        case let .invalidRepositoryUrl(s): "not a `host/path` repository coordinate: \(s)"
        case let .invalidDatasetPath(s): "invalid dataset path: \(s)"
        case let .invalidVaultResourceId(s):
            "vault resource id must be `vault_` then [A-Za-z0-9_-]: \(escapedQuoted(s))"
        case let .invalidVaultPrincipalId(s):
            "principal id must start with `org_` or `user_`, then [A-Za-z0-9_-]: \(escapedQuoted(s))"
        case let .invalidVaultPath(s):
            "a vault path has the form `/vault/resources/{resource_id}/content`, relative to the server. Give a path in that form: \(escapedQuoted(s))"
        }
    }
}

/// `value` quoted with every control or non-ASCII scalar escaped, so a refused id can
/// neither forge a log line nor hide a zero-width character.
nonisolated func escapedQuoted(_ value: String) -> String {
    "\"" + value.unicodeScalars.map { $0.escaped(asASCII: true) }.joined() + "\""
}
