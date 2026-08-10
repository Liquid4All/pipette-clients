import Foundation

/// Canonical JSON coders. Every persisted artifact and request body is encoded —
/// and every artifact that carries a `Date` decoded — through these, so the
/// on-disk/on-wire format is single-sourced and encode/decode can't drift apart.
///
/// Dates are ISO-8601 in both directions, matching the app's convention (the
/// `String` timestamps elsewhere are built with `JobDateFormat.iso8601`). Types
/// that stringify their own timestamps carry no `Date`, so the strategy is a no-op
/// for them; it's load-bearing only for models with a real `Date` property (e.g.
/// `ModelManifest.fetchedAt`), where the encoder and decoder must agree.
///
/// Shared instances, not factories: `JSONEncoder`/`JSONDecoder` are `Sendable` and
/// their configuration is fixed after setup, so one cached coder reuses safely
/// across the MainActor UI and the nonisolated persistence layer while skipping a
/// per-call allocation at every serialization boundary.
nonisolated enum Coding {
    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
