import Foundation

/// A location-qualified catalog address — `local/<id>` or `remote/<id>`, the crate's
/// `SourcedBenchmarkId` (`pipette-cli/src/benchmarks/reference.rs:22`).
///
/// A bare id means the *synced* catalog, because that is the form plans and claims
/// carry: the distributed case needs no prefix, and `local/` is the explicit opt-in for
/// a definition only this device has.
///
/// TODO: review — mirrors `reference.rs:22`; the parse table and the bare-id default
/// checked against its `from_str` tests.
nonisolated enum SourcedBenchmarkId: Hashable, Sendable {
    case local(String)
    case remote(String)

    init(source: BenchmarkSource, id: String) {
        switch source {
        case .local: self = .local(id)
        case .remote: self = .remote(id)
        }
    }

    /// The id without the location prefix.
    var id: String {
        switch self {
        case .local(let id), .remote(let id): return id
        }
    }

    /// The catalog side this address implies.
    var source: BenchmarkSource {
        switch self {
        case .local: return .local
        case .remote: return .remote
        }
    }

    /// Parse a bare id, `local/<id>`, or `remote/<id>`; nil for anything else.
    ///
    /// Rejects an empty or whitespace-bearing id, an unknown side, and a nested address
    /// (`local/remote/foo`) — the crate leans on `BenchmarkId` refusing a `/` for that
    /// last one, and this checks it directly since iOS spells ids as bare `String`.
    init?(reference: String) {
        let source: BenchmarkSource
        let raw: String
        if let slash = reference.firstIndex(of: "/") {
            switch String(reference[reference.startIndex..<slash]) {
            case "local": source = .local
            case "remote": source = .remote
            default: return nil
            }
            raw = String(reference[reference.index(after: slash)...])
        } else {
            source = .remote
            raw = reference
        }
        guard !raw.isEmpty,
              !raw.contains("/"),
              !raw.contains(where: \.isWhitespace)
        else { return nil }
        self.init(source: source, id: raw)
    }
}

extension SourcedBenchmarkId: CustomStringConvertible {
    /// `local/<id>` or `remote/<id>` — round-trips through `init(reference:)`.
    var description: String { "\(source.rawValue)/\(id)" }
}
