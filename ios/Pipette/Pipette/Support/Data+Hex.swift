import Foundation

/// The one hex codec, so encode and decode cannot drift apart: the pair has to round-trip
/// a signing key exactly.
nonisolated extension Data {
    /// Ignores surrounding whitespace; `nil` for an odd length or a non-hex character, so a
    /// truncated or hand-edited key never decodes into a shorter one.
    ///
    /// Each pair is checked for hex digits first because `UInt8(_:radix:)` accepts a sign
    /// (`"-0"` is 0), so `"-0-0…"` would decode to 32 bytes and pass every length check.
    ///
    /// Failable rather than throwing so each caller reports a bad key in its own terms.
    init?(hexEncoded hex: String) {
        let trimmed = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count.isMultiple(of: 2) else { return nil }

        var bytes: [UInt8] = []
        bytes.reserveCapacity(trimmed.count / 2)

        var index = trimmed.startIndex
        while index < trimmed.endIndex {
            let next = trimmed.index(index, offsetBy: 2)
            let pair = trimmed[index..<next]
            guard pair.allSatisfy(\.isHexDigit),
                  let byte = UInt8(pair, radix: 16)
            else { return nil }
            bytes.append(byte)
            index = next
        }
        self.init(bytes)
    }

    func hexEncodedString() -> String {
        map { String(format: "%02x", $0) }.joined()
    }
}
