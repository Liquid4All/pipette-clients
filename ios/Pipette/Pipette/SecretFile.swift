import Foundation

/// Credential files: the crate's `secret_file.rs`.
nonisolated enum SecretFile {
    /// Owner-only (`0600`) and readable only after the first unlock, the accessibility the
    /// Keychain copy of the key uses. The atomic write replaces the file, so both attributes
    /// are applied on every call rather than once at creation.
    static func write(_ path: URL, _ contents: Data) throws {
        try contents.write(
            to: path, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: path.path)
    }
}
