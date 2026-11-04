import Foundation

// Where an artifact is, and what it must hash to: pipette-mgmt's
// `crates/pipette-plan-types/src/location.rs`.

/// The crate's `ResourceUrl`.
nonisolated struct ResourceUrl: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /(https?|file):\/\/.+/) != nil
    }
    static let rejection = ModelError.invalidResourceUrl
}

/// The crate's `VaultPath`: `/vault/resources/{resource_id}/content` on the server.
///
/// A path, not a URL: a plan that named a host could send a signed fetch to any server.
/// `value` is always the canonical route, so a padded id reads back without its padding.
nonisolated struct VaultPath: ValidatedString {
    private static let prefix = "/vault/resources/"
    private static let suffix = "/content"

    let value: String
    init(validated value: String) { self.value = value }

    /// The route as the crate parses it: trimmed, then the id between the prefix and the
    /// suffix trimmed again by `VaultResourceId`.
    static func sanitize(_ value: String) -> String {
        guard let id = innerId(rustTrim(value)) else { return value }
        return prefix + VaultResourceId.sanitize(id) + suffix
    }

    static func validate(_ value: String) -> Bool {
        innerId(value).map(VaultResourceId.validate) ?? false
    }

    static let rejection = ModelError.invalidVaultPath

    var resourceId: VaultResourceId {
        VaultResourceId(validated: Self.innerId(value) ?? "")
    }

    private static func innerId(_ value: String) -> String? {
        guard value.hasPrefix(prefix), value.hasSuffix(suffix),
              value.count >= prefix.count + suffix.count
        else { return nil }
        return String(value.dropFirst(prefix.count).dropLast(suffix.count))
    }
}

/// The crate's `strip_leading_current_dir`. On `AbsolutePath`, `./x` must fail after
/// stripping rather than be mistaken for a prefix worth honoring.
private nonisolated func strippingLeadingCurrentDir(_ raw: String) -> String {
    raw.hasPrefix("./") ? String(raw.dropFirst(2)) : raw
}

/// Both separators, as the crate splits: `/` alone would let a Windows-authored
/// `a\..\b` past the traversal check.
private nonisolated func isPathSeparator(_ character: Character) -> Bool {
    character == "/" || character == "\\"
}

/// `C:`.
private nonisolated func hasDriveLetterPrefix(_ value: String) -> Bool {
    var characters = value.makeIterator()
    guard let first = characters.next(), first.isASCII, first.isLetter else { return false }
    return characters.next() == ":"
}

/// The crate's `RelativePath` (`is_relative_fs_path`). Never absolute, so a stored
/// coordinate survives the container-UUID changes that invalidate an absolute one. Wider
/// than `RepoSubpath`: a stored path can contain whatever a repo's filenames do.
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

/// The crate's `AbsolutePath` (`is_absolute_path`): the bound form a source takes after
/// `bindUnder`, never a portable store coordinate.
nonisolated struct AbsolutePath: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func sanitize(_ value: String) -> String { strippingLeadingCurrentDir(value) }
    static func validate(_ value: String) -> Bool {
        guard !value.isEmpty, !value.hasPrefix("~") else { return false }
        let unc = value.hasPrefix("\\\\") || value.hasPrefix("//")
        // The colon alone is not absolute.
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
        // A bare root is still absolute.
        return body.isEmpty
            || body.split(omittingEmptySubsequences: false, whereSeparator: isPathSeparator)
                .allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static let rejection = ModelError.invalidAbsolutePath
}

/// A lowercase hex SHA-256 digest. Only ever set on a fetching arm: a local file is
/// identified by its path.
nonisolated struct Sha256: ValidatedString {
    let value: String
    init(validated value: String) { self.value = value }
    static func validate(_ value: String) -> Bool {
        value.wholeMatch(of: /[0-9a-f]{64}/) != nil
    }
    static let rejection = ModelError.invalidSha256
}
