import Foundation

// The headers of a `206` that answers `Range: bytes=N-` (the `pipette-http` crate's
// `range.rs`), which a resumed download reads to check the response continues its file.

/// The `ETag` value `value`, quotes included, if it is strong. A weak `ETag` does not
/// promise the same bytes, so it cannot prove that a `206` continues a file.
nonisolated func strongEtagValue(_ value: String) -> String? {
    let value = value.trimmingCharacters(in: .whitespaces)
    return value.hasPrefix("W/") ? nil : value
}

/// A `Content-Range` of the form `bytes first-last/total`.
nonisolated struct ContentRange: Equatable, Sendable {
    let first: UInt64
    let last: UInt64
    /// `nil` when the server sent `*`.
    let total: UInt64?
}

nonisolated extension ContentRange {
    /// `nil` unless it names a range of one byte or more.
    init?(headerValue value: String) {
        guard value.hasPrefix("bytes ") else { return nil }
        let parts = value.dropFirst("bytes ".count).split(
            separator: "/", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        let bounds = parts[0].split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        guard bounds.count == 2,
              let first = UInt64(bounds[0].trimmingCharacters(in: .whitespaces)),
              let last = UInt64(bounds[1].trimmingCharacters(in: .whitespaces)),
              first <= last
        else { return nil }
        self.init(
            first: first, last: last, total: UInt64(parts[1].trimmingCharacters(in: .whitespaces)))
    }
}
