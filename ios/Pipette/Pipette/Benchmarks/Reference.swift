import Foundation

// A location-qualified catalog address, `local/<id>` or `remote/<id>`: `pipette-cli`'s
// `benchmarks/reference.rs`.

/// A reference that names no benchmark. The offending reference is echoed whole, escaped.
nonisolated enum SourcedBenchmarkIdError: Error, Equatable, LocalizedError {
    /// An unknown catalog side or a nested address (`local/remote/foo`), where the
    /// accepted forms are the useful part of the message.
    case shape(String)
    /// A well-shaped reference whose id the catalog grammar refuses.
    case id(reference: String, reason: BenchmarkIdError)

    var errorDescription: String? {
        switch self {
        case let .shape(reference):
            "\(escapedQuoted(reference)) is not a benchmark reference: expected `<id>`, "
                + "`local/<id>` or `remote/<id>`"
        case let .id(reference, reason):
            "\(escapedQuoted(reference)) is not a benchmark reference: "
                + (reason.errorDescription ?? "")
        }
    }
}

/// A location-qualified benchmark id (`local/<id>` or `remote/<id>`).
nonisolated enum SourcedBenchmarkId: Hashable, Sendable {
    case local(BenchmarkId)
    case remote(BenchmarkId)

    init(source: BenchmarkSource, id: BenchmarkId) {
        switch source {
        case .local: self = .local(id)
        case .remote: self = .remote(id)
        }
    }

    var id: BenchmarkId {
        switch self {
        case .local(let id), .remote(let id): return id
        }
    }

    var source: BenchmarkSource {
        switch self {
        case .local: return .local
        case .remote: return .remote
        }
    }

    /// Parse a bare id, `local/<id>`, or `remote/<id>`: the crate's `FromStr`.
    ///
    /// A bare id means the synced catalog: that is the form plans and claims carry, so the
    /// distributed case needs no prefix. `local/` stays as the explicit opt-in for a
    /// definition only this device has.
    init(parsing reference: String) throws(SourcedBenchmarkIdError) {
        let source: BenchmarkSource
        let raw: Substring
        if let slash = reference.firstIndex(of: "/") {
            switch reference[..<slash] {
            case "local": source = .local
            case "remote": source = .remote
            default: throw .shape(reference)
            }
            raw = reference[reference.index(after: slash)...]
        } else {
            source = .remote
            raw = reference[...]
        }
        guard !raw.contains("/") else { throw .shape(reference) }
        if let reason = BenchmarkId.check(String(raw)) {
            throw .id(reference: reference, reason: reason)
        }
        self.init(source: source, id: BenchmarkId(validated: String(raw)))
    }
}

nonisolated extension SourcedBenchmarkId: CustomStringConvertible {
    /// `local/<id>` or `remote/<id>`: round-trips through `init(parsing:)`.
    var description: String { "\(source.rawValue)/\(id)" }
}
