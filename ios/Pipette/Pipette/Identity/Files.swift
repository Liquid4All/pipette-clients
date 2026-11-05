import Foundation

/// The file rules the three identity stores share.
nonisolated enum IdentityFiles {
    static func path(_ root: URL, _ name: String) -> URL {
        root.appendingPathComponent(name)
    }

    /// Nil only when the file is absent: one that does not read is an error.
    static func readJson<T: Decodable>(_ root: URL, _ name: String) throws -> T? {
        let data: Data
        do {
            data = try Data(contentsOf: path(root, name))
        } catch CocoaError.fileReadNoSuchFile {
            return nil
        }
        return try Coding.decoder.decode(T.self, from: data)
    }

    static func writeJson(_ root: URL, _ name: String, _ value: some Encodable) throws {
        try ensure(root)
        try Coding.encoder.encode(value).write(to: path(root, name), options: .atomic)
    }

    /// For the key file, which holds hex rather than JSON. Nil when absent.
    static func readText(_ root: URL, _ name: String) throws -> String? {
        do {
            return try String(contentsOf: path(root, name), encoding: .utf8)
        } catch CocoaError.fileReadNoSuchFile {
            return nil
        }
    }

    /// For the private key, the one file here that is a credential.
    static func writeSecret(_ root: URL, _ name: String, _ contents: Data) throws {
        try ensure(root)
        try SecretFile.write(path(root, name), contents)
    }

    /// Idempotent when the file is already gone.
    static func remove(_ root: URL, _ name: String) throws {
        do {
            try FileManager.default.removeItem(at: path(root, name))
        } catch CocoaError.fileNoSuchFile {
            return
        }
    }

    /// Create `identity/` and keep it out of backups (where the crate asserts `0700`).
    ///
    /// A restored backup would put the record and key on a device the server has never
    /// seen, so one `client_id` could sign from two places; Android excludes them too
    /// (`android/Pipette/app/src/main/res/xml/data_extraction_rules.xml`). `auth export` is
    /// the deliberate way to move an identity.
    ///
    /// The flag sits on the directory because the atomic writes replace each file, and a
    /// flag on a file would not survive its next write.
    private static func ensure(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if !LocalStorage.markExcludedFromBackup(root) {
            AppLog.storage.error("identity: the directory is not excluded from backups")
        }
    }
}
