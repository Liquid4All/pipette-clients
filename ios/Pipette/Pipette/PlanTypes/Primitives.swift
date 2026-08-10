import Foundation

// The iOS counterpart of `crates/pipette-plan-types/src/primitives.rs`: the validated
// scalar types every coordinate is built from, plus the repo coordinate itself. Kept in
// one file for the same reason the crate does — they are one vocabulary, and a reader
// who knows the crate should find them where its module puts them.

// MARK: - Validated string newtypes
//
// Swift analog of the `nutype` wrappers in `pipette-plan-types`: a string newtype
// validates at construction *and* on `Codable` decode, so an invalid value is
// unrepresentable — rejected at the boundary, never caught by a later pass. Peer of
// this app's `StringNewtype`, but with a *throwing* constructor (validation) instead
// of the transparent unvalidated one. Extension members are `nonisolated` so they
// satisfy `nonisolated` conformers (required for `Sendable` under main-actor default
// isolation).

protocol ValidatedString: Codable, Hashable, Sendable, CustomStringConvertible {
    nonisolated var value: String { get }
    /// Trusted constructor — only called after `validate` has passed.
    nonisolated init(validated value: String)
    nonisolated static func validate(_ value: String) -> Bool
    /// The error thrown when `validate` fails (e.g. `ModelError.invalidOrg`).
    nonisolated static var rejection: (String) -> ModelError { get }
    /// Normalizes before `validate`, in nutype's sanitize-then-validate order. Identity
    /// unless the crate declares a `sanitize(with = …)` for the type.
    nonisolated static func sanitize(_ value: String) -> String
}

extension ValidatedString {
    nonisolated static func sanitize(_ value: String) -> String { value }

    /// The one public constructor: sanitizes, validates, or throws the type's rejection.
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

/// A HuggingFace org segment, e.g. `LiquidAI` — `^[A-Za-z0-9][A-Za-z0-9._-]*$`.
nonisolated struct HFOrg: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._-]*/) != nil }
    static let rejection = ModelError.invalidOrg
}

/// A HuggingFace repo-name segment, e.g. `LFM2.5-350M-MLX-4bit`.
/// A string the crate types `NonEmptyString` — the rule lives on the type so no
/// construction path can bypass it, rather than in whichever decoder remembered to check.
nonisolated struct NonEmptyString: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { !value.isEmpty }
    static let rejection = ModelError.emptyValue
}

nonisolated struct HFRepoName: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._-]*/) != nil }
    static let rejection = ModelError.invalidRepoName
}

/// A subdirectory within a multi-model repo, e.g. `4bit` or `variants/4bit`.
/// Slash-joined path segments of `[A-Za-z0-9._-]`; no leading/trailing/empty segment.
nonisolated struct RepoSubpath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    /// `.` and `..` match the segment character class, so they have to be excluded by
    /// name — as the crate's `is_relative_repo_subpath` does. A subpath is
    /// server-authored (a claim's `prefix`) and is joined onto a cache path that is
    /// later removed, so a traversal here would let a claim name a directory tree for
    /// deletion. Rejecting it at the type makes that unrepresentable rather than
    /// something each consumer has to remember.
    static func validate(_ value: String) -> Bool {
        !value.isEmpty && value.split(separator: "/", omittingEmptySubsequences: false)
            .allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
                    && $0.wholeMatch(of: /[A-Za-z0-9._-]+/) != nil
            }
    }
    static let rejection = ModelError.invalidSubpath
    /// The final path component — the model directory's name.
    var leaf: String { value.split(separator: "/").last.map(String.init) ?? value }
}

/// The crate's `strip_leading_current_dir`: a redundant `./` is dropped so a stored path
/// is normalized. Shared by both path types, as the crate shares the one sanitizer — and
/// load-bearing on `AbsolutePath`, where `./x` must fail *after* stripping rather than be
/// mistaken for a prefix worth honoring.
private nonisolated func strippingLeadingCurrentDir(_ raw: String) -> String {
    raw.hasPrefix("./") ? String(raw.dropFirst(2)) : raw
}

/// Both separators, as the crate's `split(['/', '\\'])` does. Splitting on `/` alone
/// would let a Windows-authored `a\..\b` past the traversal check.
private nonisolated func isPathSeparator(_ character: Character) -> Bool {
    character == "/" || character == "\\"
}

/// `C:` — an ASCII drive letter followed by a colon.
private nonisolated func hasDriveLetterPrefix(_ value: String) -> Bool {
    var characters = value.makeIterator()
    guard let first = characters.next(), first.isASCII, first.isLetter else { return false }
    return characters.next() == ":"
}

/// A portable, store-relative filesystem path — the crate's `RelativePath`. Never
/// absolute, so a stored coordinate stays valid across the container-UUID changes and
/// layout moves that invalidate an absolute one.
///
/// Mirrors `is_relative_fs_path`: rejects empty, `~`, a leading `/`, a Windows drive or
/// UNC prefix, and any empty / `.` / `..` segment. Wider than `RepoSubpath`, which
/// additionally constrains the character class — a stored path can contain whatever a
/// repo's filenames do.
nonisolated struct RelativePath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func sanitize(_ value: String) -> String { strippingLeadingCurrentDir(value) }
    static func validate(_ value: String) -> Bool {
        // A leading `/` covers the UNC `//server` form too.
        guard !value.isEmpty, !value.hasPrefix("~"), !value.hasPrefix("/"),
              !value.hasPrefix("\\\\"), !hasDriveLetterPrefix(value)
        else { return false }
        return value.split(omittingEmptySubsequences: false, whereSeparator: isPathSeparator)
            .allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static let rejection = ModelError.invalidRelativePath
}

/// An absolute host path — the crate's `AbsolutePath`. The bound form a source takes
/// after `bindUnder`; never a portable store coordinate.
///
/// Mirrors `is_absolute_path`: Unix `/…`, UNC, or a Windows drive, with no `~` and no
/// `.` / `..` segment.
nonisolated struct AbsolutePath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func sanitize(_ value: String) -> String { strippingLeadingCurrentDir(value) }
    static func validate(_ value: String) -> Bool {
        guard !value.isEmpty, !value.hasPrefix("~") else { return false }
        let unc = value.hasPrefix("\\\\") || value.hasPrefix("//")
        // `C:/…` or `C:\…` — the colon alone is not absolute.
        let drive = hasDriveLetterPrefix(value)
            && value.dropFirst(2).first.map(isPathSeparator) == true

        let body: Substring
        if unc {
            let rest = value.dropFirst(2)
            // UNC names a server and a share; an empty one is not a host path.
            guard rest.split(omittingEmptySubsequences: false, whereSeparator: isPathSeparator)
                .prefix(2).allSatisfy({ !$0.isEmpty })
            else { return false }
            body = rest
        } else if value.hasPrefix("/") {
            body = value.dropFirst()
        } else if drive {
            body = value.dropFirst(3)
        } else {
            return false
        }
        // A root with nothing after it is still absolute; anything after it carries the
        // same no-traversal, no-empty-segment rule.
        return body.isEmpty
            || body.split(omittingEmptySubsequences: false, whereSeparator: isPathSeparator)
                .allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static let rejection = ModelError.invalidAbsolutePath
}

/// A fetchable resource URL — the crate's `ResourceUrl`, `^(https?|file)://.+`.
nonisolated struct ResourceUrl: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /(https?|file):\/\/.+/) != nil
    }
    static let rejection = ModelError.invalidResourceUrl
}

/// A pin to a specific repo snapshot — a tag, branch or commit.
/// `^[A-Za-z0-9][A-Za-z0-9._/-]*$`, matching the crate; `/` is allowed so a ref like
/// `refs/pr/1` validates.
nonisolated struct HFRevision: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /[A-Za-z0-9][A-Za-z0-9._\/-]*/) != nil
    }
    static let rejection = ModelError.invalidRevision
}

/// A lowercase hex SHA-256 digest — `^[0-9a-f]{64}$`. Verifies a download; a local
/// file is identified by its path instead, so this is only ever set on a fetching arm.
nonisolated struct Sha256: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /[0-9a-f]{64}/) != nil
    }
    static let rejection = ModelError.invalidSha256
}

/// An access token for a gated or private repo, carried in the plan rather than read
/// from the environment.
///
/// `description` renders `<redacted>`, as the crate's hand-written `Display` and `Debug`
/// do — so interpolating a token, or a struct holding one, cannot publish it. `value` is
/// the one door to the secret, which makes auditing where it can escape a search for
/// `.value` on this type.
nonisolated struct AuthToken: ValidatedString, CustomDebugStringConvertible {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool { !value.isEmpty }
    /// Discards the input rather than threading it into the error — see
    /// `ModelError.invalidAuthToken`.
    static let rejection: (String) -> ModelError = { _ in .invalidAuthToken }

    var description: String { "<redacted>" }
    var debugDescription: String { "AuthToken(<redacted>)" }
}

// MARK: - Coordinates

/// A HuggingFace repo coordinate. The canonical `org/repo_name` parser is the single
/// authoritative split-and-validate path (mirrors `HfRepo::parse_slug`).
nonisolated struct HFRepo: Hashable, Codable, Sendable, CustomStringConvertible {
    let org: HFOrg
    /// `repoName`, not `name`, matching the crate's `HfRepo::repo_name` — on a repo,
    /// a bare `name` reads as "the name of the repo coordinate" rather than the
    /// second half of the slug.
    let repoName: HFRepoName
    /// Pin to a snapshot. `nil` resolves to the repo's default branch — latest and
    /// mutable — so a plan that wants reproducible weights sets it. Part of the repo's
    /// identity: it feeds `reference`, and therefore the storage key, so two cells
    /// pinning different revisions do not collide on one entry.
    var revision: HFRevision?
    /// Credential for a gated repo, carried in the plan. **Not** part of the repo's
    /// identity — see `==` and `hash(into:)` below.
    var authToken: AuthToken?

    /// Canonical `org/repo_name` slug — matches HuggingFace's URL/CLI convention.
    var description: String { "\(org)/\(repoName)" }

    /// The identity string, `org/repo_name[@revision]`, mirroring `HfRepo::reference`.
    /// This is what a storage key is derived from, which is why the revision belongs in
    /// it and the token does not.
    var reference: String {
        revision.map { "\(description)@\($0.value)" } ?? description
    }

    /// The same coordinate with the credential dropped, for anything that gets written
    /// down. `encode` emits `auth_token` — it has to, since that is how a claim's model
    /// arrives — so the boundary that must not keep a secret is the one that redacts.
    /// Mirrors the crate, which strips the token before persisting rather than teaching
    /// its encoder to lie.
    var withoutAuthToken: HFRepo {
        var copy = self
        copy.authToken = nil
        return copy
    }

    /// Identity is the coordinate, so the token is excluded from both. Two references to
    /// the same weights must compare equal whether or not one of them happens to carry a
    /// credential — otherwise a claim that ships a token would fail to match the model
    /// already on disk, and the same weights would be fetched twice under two keys.
    ///
    /// The crate reaches the same place from the other direction: it derives `Hash`/`Eq`
    /// over every field but strips the token before anything is persisted or keyed.
    static func == (lhs: HFRepo, rhs: HFRepo) -> Bool {
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

    /// Omits an absent revision or token rather than writing `null`, matching the
    /// crate's `skip_serializing_if = "Option::is_none"`.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(org, forKey: .org)
        try c.encode(repoName, forKey: .repoName)
        try c.encodeIfPresent(revision, forKey: .revision)
        try c.encodeIfPresent(authToken, forKey: .authToken)
    }

    static func parse(_ slug: String) throws -> HFRepo {
        guard let slash = slug.firstIndex(of: "/") else { throw ModelError.repoMissingSeparator(slug) }
        let name = String(slug[slug.index(after: slash)...])
        // HF ids are exactly two segments; a nested `/` is not an `org/repo_name`.
        guard !name.contains("/") else { throw ModelError.repoMissingSeparator(slug) }
        return HFRepo(org: try HFOrg(String(slug[..<slash])), repoName: try HFRepoName(name))
    }
}
