import Foundation

// HuggingFace coordinates: pipette-mgmt's
// `crates/pipette-plan-types/src/huggingface.rs`.

/// A HuggingFace org segment, e.g. `LiquidAI`.
nonisolated struct HfOrg: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._-]*/) != nil }
    static let rejection = ModelError.invalidOrg
}

/// A HuggingFace repo-name segment, e.g. `LFM2.5-350M-MLX-4bit`.
nonisolated struct HfRepoName: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._-]*/) != nil }
    static let rejection = ModelError.invalidRepoName
}

/// A subdirectory within a multi-model repo, e.g. `4bit` or `variants/4bit`.
nonisolated struct RepoSubpath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    /// `.` and `..` match the character class, so they are excluded by name, as in the
    /// crate's `is_relative_repo_subpath`. A subpath is server-authored and joined onto a
    /// cache path that is later removed, so a traversal would let a claim name a tree for
    /// deletion.
    static func validate(_ value: String) -> Bool {
        !value.isEmpty && value.split(separator: "/", omittingEmptySubsequences: false)
            .allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
                    && $0.wholeMatch(of: /[A-Za-z0-9._-]+/) != nil
            }
    }
    static let rejection = ModelError.invalidSubpath
    /// The model directory's name.
    var leaf: String { value.split(separator: "/").last.map(String.init) ?? value }
}

/// A tag, branch or commit. `/` is allowed so a ref like `refs/pr/1` validates.
nonisolated struct HfRevision: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._\/-]*/) != nil
    }
    static let rejection = ModelError.invalidRevision
}

/// An access token for a gated or private repo, carried in the plan.
///
/// Renders `<redacted>`, as the crate's `Display` and `Debug` do, so interpolating a token
/// cannot publish it. `value` is the one way to the secret.
nonisolated struct AuthToken: ValidatedString, CustomDebugStringConvertible {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { !value.isEmpty }
    /// Discards the input: the rejected value is the token.
    static let rejection: (String) -> ModelError = { _ in .invalidAuthToken }

    var description: String { "<redacted>" }
    var debugDescription: String { "AuthToken(<redacted>)" }
}

/// A HuggingFace repo coordinate.
nonisolated struct HfRepo: Hashable, Codable, Sendable, CustomStringConvertible {
    let org: HfOrg
    let repoName: HfRepoName
    /// `nil` resolves to the repo's mutable default branch. Part of the identity, so two
    /// cells pinning different revisions do not collide on one storage entry.
    var revision: HfRevision?
    /// Not part of the identity: see `==`.
    var authToken: AuthToken?

    /// `org/repo_name`, as HF clients take it; the revision goes separately.
    var description: String { "\(org)/\(repoName)" }

    /// `org/repo_name[@revision]`, the crate's `HfRepo::reference`: what a storage key is
    /// derived from.
    var reference: String {
        revision.map { "\(description)@\($0.value)" } ?? description
    }

    /// This coordinate without its credential: the crate's `without_auth_token` at the repo
    /// level. This type's `encode` writes `auth_token` when set, as the crate's does;
    /// `Model.encode` always drops it (`docs/pipette-ios/execution-alignment.md`).
    var withoutAuthToken: HfRepo {
        var copy = self
        copy.authToken = nil
        return copy
    }

    /// The token is excluded so a claim that ships one still matches the model already on
    /// disk rather than fetching the same weights under a second key. (The crate compares
    /// every field but strips the token before anything is keyed.)
    static func == (lhs: HfRepo, rhs: HfRepo) -> Bool {
        lhs.org == rhs.org && lhs.repoName == rhs.repoName && lhs.revision == rhs.revision
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(org)
        hasher.combine(repoName)
        hasher.combine(revision)
    }

    enum CodingKeys: String, CodingKey {
        case org
        case repoName = "repo_name"
        case revision
        case authToken = "auth_token"
    }

    /// Omits an absent revision or token rather than writing `null`, as the crate does.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(org, forKey: .org)
        try c.encode(repoName, forKey: .repoName)
        try c.encodeIfPresent(revision, forKey: .revision)
        try c.encodeIfPresent(authToken, forKey: .authToken)
    }

    /// The crate's `parse_org_repo`: split on the first `/`, so `a/b/c` is refused as a
    /// repo name.
    static func parseOrgRepo(_ orgRepo: String) throws -> HfRepo {
        guard let slash = orgRepo.firstIndex(of: "/") else {
            throw ModelError.repoMissingSeparator(orgRepo)
        }
        return HfRepo(org: try HfOrg(String(orgRepo[..<slash])),
                      repoName: try HfRepoName(String(orgRepo[orgRepo.index(after: slash)...])))
    }
}
