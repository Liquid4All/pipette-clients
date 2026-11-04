import Foundation

// Addressing an installed artifact by its descriptor digest, `model://sha256=<hex>`: the
// counterpart of `crates/pipette-cli/src/artifact_ref.rs`. Unlike every other `model=`
// spelling, a digest refers to a descriptor rather than describing one, so resolving it
// needs the local store; that is why this sits above the pure parser.
//
// A manifest holds whichever vault label its entry was first fetched under, so a digest
// resolves to the model without it: a run named by digest reports no label rather than a
// stale one. This client has no runtime store, so `runtimeScheme` is declared only so the
// vocabulary is the crate's.

nonisolated let runtimeScheme = "runtime://"
nonisolated let modelScheme = "model://"
private nonisolated let digestKey = "sha256="
/// 32 bits is far past collision range for a local store; shorter reads as a typo.
private nonisolated let digestMinPrefixLen = 8

/// `nil` when `arg` is an ordinary reference. The scheme without a `sha256=` key gets its
/// own refusal rather than the URI parser's "unknown scheme": it is the shape a bare digest
/// pasted after the scheme takes.
nonisolated func digestPrefix(_ arg: String, scheme: String) throws(ArtifactRefError) -> String? {
    let trimmed = arg.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix(scheme) else { return nil }
    let body = trimmed.dropFirst(scheme.count)
    guard body.hasPrefix(digestKey) else { throw .digestKeyMissing(scheme: scheme) }
    return try validatedPrefix(String(body.dropFirst(digestKey.count)))
}

/// Digests are hex, so case carries nothing and an upper-case paste matches the lower-case
/// form the digest is rendered in.
nonisolated func validatedPrefix(_ raw: String) throws(ArtifactRefError) -> String {
    let prefix = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard prefix.count >= digestMinPrefixLen else { throw .tooShort(prefix) }
    guard prefix.allSatisfy({ $0.isASCII && $0.isHexDigit }) else { throw .notHex(prefix) }
    return prefix
}

/// Ambiguity is an error rather than a pick: two entries sharing a prefix are different
/// artifacts, and guessing would run the wrong one.
nonisolated func uniqueMatch<T>(
    _ prefix: String, _ candidates: [(digest: String, value: T)], kind: String,
    render: (T) -> String
) throws(ArtifactRefError) -> T {
    let matched = candidates.filter { $0.digest.hasPrefix(prefix) }.map(\.value)
    switch matched.count {
    case 1: return matched[0]
    case 0: throw .noMatch(kind: kind, prefix: prefix)
    default: throw .ambiguous(kind: kind, prefix: prefix, candidates: matched.map(render))
    }
}

/// Without the vault `custom_label`, which the server leaves out of
/// `model_descriptor_sha256` too, so a prefix from `headlessrun models` matches the
/// warehouse's rows (except a vault copy of a HuggingFace model, which the server records
/// under the HuggingFace descriptor).
nonisolated func modelDescriptorDigest(_ model: Model) throws -> String {
    try descriptorDigest(model.withoutAuthToken.withoutCustomLabel)
}

/// Answers with the stripped declaration (no token, no vault label), as the crate's
/// `resolve_model_arg` does.
nonisolated func resolveModel(digestPrefix prefix: String, installed: [Model]) throws -> Model {
    let candidates = try installed.map { model in
        (digest: try modelDescriptorDigest(model), value: model.withoutAuthToken.withoutCustomLabel)
    }
    return try uniqueMatch(prefix, candidates, kind: "model", render: \.description)
}

/// The wording is the crate's `anyhow` messages, with this client's listing verbs.
nonisolated enum ArtifactRefError: Error, Equatable, LocalizedError {
    case digestKeyMissing(scheme: String)
    case tooShort(String)
    case notHex(String)
    case noMatch(kind: String, prefix: String)
    case ambiguous(kind: String, prefix: String, candidates: [String])

    var errorDescription: String? {
        switch self {
        case let .digestKeyMissing(scheme):
            "`\(scheme)` takes a digest: `\(scheme)\(digestKey)<hex>` "
                + "(see the digests `headlessrun models` prints)"
        case let .tooShort(prefix):
            "digest `\(prefix)` is too short; give at least \(digestMinPrefixLen) hex chars"
        case let .notHex(prefix):
            "digest `\(prefix)` is not hex"
        case let .noMatch(kind, prefix):
            "no installed \(kind) has a descriptor digest starting `\(prefix)` "
                + "(`headlessrun \(kind)s` shows the installed set)"
        case let .ambiguous(kind, prefix, candidates):
            "digest `\(prefix)` is ambiguous across \(candidates.count) \(kind)s "
                + "(\(candidates.joined(separator: ", "))); use more characters"
        }
    }
}
