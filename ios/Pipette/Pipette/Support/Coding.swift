import Foundation

/// The app's JSON coders, so a `Date` field (e.g. `ModelManifest.fetchedAt`) is
/// encoded and decoded with the same ISO-8601 strategy.
///
/// Shared instances: the coders are `Sendable` and never reconfigured after setup.
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
