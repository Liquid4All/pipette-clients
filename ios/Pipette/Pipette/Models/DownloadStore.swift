import Foundation

/// Disk-backed store for download records and resume-data blobs, in the workspace's
/// `downloadsDir` under Caches: the system may evict it, and the worst case is a restarted
/// download.
struct DownloadStore {
    let dir: URL

    /// Keys contain `/`. Percent-encoding is injective, unlike a `/` to `__` substitution
    /// that two keys can alias onto.
    private static func safeName(_ key: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "._-")
        return key.addingPercentEncoding(withAllowedCharacters: allowed)
            ?? key.replacingOccurrences(of: "/", with: "%2F")
    }

    /// Called before every write, since Caches can be evicted at any time. Reads need no
    /// directory: a missing one already means "nothing here".
    private func ensureDir() {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    private func recordPath(for key: String) -> URL {
        dir.appendingPathComponent(Self.safeName(key) + ".record")
    }

    private func resumePath(for key: String) -> URL {
        dir.appendingPathComponent(Self.safeName(key) + ".resumeData")
    }

    func saveRecord(_ record: DownloadRecord) {
        guard let data = try? PropertyListEncoder().encode(record) else { return }
        let key = LocalStorage.modelRelativePath(repo: record.repo, filename: record.filename)
        ensureDir()
        try? data.write(to: recordPath(for: key), options: .atomic)
    }

    func clearRecord(for key: String) {
        try? FileManager.default.removeItem(at: recordPath(for: key))
    }

    func allRecords() -> [DownloadRecord] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "record" }.compactMap { url in
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? PropertyListDecoder().decode(DownloadRecord.self, from: data)
        }
    }

    func saveResumeData(_ data: Data, for key: String) {
        ensureDir()
        try? data.write(to: resumePath(for: key), options: .atomic)
    }

    func loadResumeData(for key: String) -> Data? {
        try? Data(contentsOf: resumePath(for: key))
    }

    func hasResumeData(for key: String) -> Bool {
        FileManager.default.fileExists(atPath: resumePath(for: key).path)
    }

    func clearResumeData(for key: String) {
        try? FileManager.default.removeItem(at: resumePath(for: key))
    }
}
