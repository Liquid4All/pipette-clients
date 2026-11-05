import Foundation

// Freeze a directory pull before any payload byte moves: resolve the declared revision to
// one immutable commit, list every file, and record every length (PIP-591; the "frozen
// plan" of `docs/pipette-ios/model-store-design.md`). Two pulls of `main` can resolve to
// different commits, so a pull addressed by the declared string could resume the wrong
// tree. Costs one listing plus one HEAD per file; an unsized file refuses the transfer.

/// One file in a frozen plan.
nonisolated struct PlannedFile: Sendable, Equatable, Hashable, Codable {
    /// Repo-relative, as the listing gives it.
    let path: String
    let sizeBytes: Int64
    /// **Not a digest**: an opaque validator, the SHA-256 only for an LFS-backed file. Never
    /// report it as cryptographic verification.
    let etag: String?
    /// When absent, the frozen length is the file's only publication check.
    let sha256: Sha256?

    init(path: String, sizeBytes: Int64, etag: String?, sha256: Sha256? = nil) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.etag = etag
        self.sha256 = sha256
    }
}

/// A directory pull, frozen: one commit, every file, every length.
nonisolated struct ResolvedTransfer: Sendable, Equatable, Hashable, Codable {
    /// What the payload requests and the staging marker name, not the declared revision.
    let commit: String
    let files: [PlannedFile]

    var totalBytes: Int64 {
        files.reduce(0) { total, file in
            let (sum, overflow) = total.addingReportingOverflow(file.sizeBytes)
            return overflow ? .max : sum
        }
    }

    enum CodingKeys: String, CodingKey {
        case commit
        case files
    }
}

/// Why a pull could not be frozen, refused before any payload byte moves.
nonisolated enum TransferPlanningError: Error, Equatable, LocalizedError {
    /// "No such subtree": publishing nothing would make an entry that binds to an empty
    /// directory.
    case emptySelection(String)
    /// The files did not agree on one commit, or the server sent none.
    case unresolvedCommit(String)
    /// Admission cannot budget an unsized transfer.
    case unknownSize(path: String)
    case metadataUnavailable(reason: String)
    /// Absolute, or reaching upwards through `..`: it would write outside the entry.
    case unsafePath(String)
    /// Two paths that land on one file, including `Config.json` and `config.json` on a
    /// case-insensitive volume.
    case collidingPaths(String, String)

    var errorDescription: String? {
        switch self {
        case let .emptySelection(coordinate):
            return "`\(coordinate)` selects no files at that revision."
        case let .unresolvedCommit(coordinate):
            return "`\(coordinate)` could not be resolved to a single commit, so the "
                + "download would not be reproducible."
        case let .unknownSize(path):
            return "the server reported no size for `\(path)`, so the download cannot be "
                + "budgeted against the storage quota."
        case let .metadataUnavailable(reason):
            return "could not read the model's file listing: \(reason)"
        case let .unsafePath(path):
            return "`\(path)` does not stay inside the model's own directory, so the "
                + "download was refused."
        case let .collidingPaths(first, second):
            return "`\(first)` and `\(second)` would be stored as the same file, so the "
                + "download was refused."
        }
    }
}

/// The Hub reads a frozen plan needs, behind a seam for tests.
///
/// Not `HubApi.getFileMetadata(from:revision:matching:)`: it forwards the revision to the
/// HEADs but **not** to the listing, so a pinned revision is listed from `main`.
protocol HubMetadataSource: Sendable {
    func filenames(repo: HfRepo, revision: String, matching globs: [String]) async throws -> [String]
    func metadata(repo: HfRepo, revision: String, path: String) async throws -> HubFileMetadata
}

nonisolated struct HubFileMetadata: Sendable, Equatable {
    let sizeBytes: Int64?
    let commitHash: String?
    let etag: String?
    let sha256: Sha256?

    init(sizeBytes: Int64?, commitHash: String?, etag: String?, sha256: Sha256? = nil) {
        self.sizeBytes = sizeBytes
        self.commitHash = commitHash
        self.etag = etag
        self.sha256 = sha256
    }
}

nonisolated enum HubTransferPlanner {
    /// Refuse a listing whose paths would not lay out one file each inside the payload:
    /// every path stays inside the directory, and no two land on one file, compared
    /// case-insensitively as the iOS volume is. Runs before any metadata request.
    static func checkDestinations(of names: [String]) throws {
        var seen: [String: String] = [:]
        for path in names.sorted() {
            let parts = path.split(separator: "/", omittingEmptySubsequences: false)
            guard !path.hasPrefix("/"), !parts.contains(".."), !parts.contains(""),
                  !path.isEmpty
            else { throw TransferPlanningError.unsafePath(path) }
            let destination = path.lowercased()
            if let first = seen.updateValue(path, forKey: destination) {
                throw TransferPlanningError.collidingPaths(first, path)
            }
        }
    }

    /// Throws rather than returning a partial plan.
    static func resolve(_ ref: HfModelRef, globs: [String],
                        using source: any HubMetadataSource) async throws -> ResolvedTransfer {
        let repo = ref.repo
        let declared = ref.hubRevision
        let names: [String]
        do {
            names = try await source.filenames(repo: repo, revision: declared, matching: globs)
        } catch {
            throw TransferPlanningError.metadataUnavailable(reason: "\(error)")
        }
        guard !names.isEmpty else { throw TransferPlanningError.emptySelection(ref.description) }
        try checkDestinations(of: names)

        // Sorted so the plan is byte-identical whatever order the listing arrived in: it
        // is persisted, compared and logged.
        var files: [PlannedFile] = []
        var commits: Set<String> = []
        for path in names.sorted() {
            let metadata: HubFileMetadata
            do {
                metadata = try await source.metadata(repo: repo, revision: declared, path: path)
            } catch {
                throw TransferPlanningError.metadataUnavailable(reason: "\(error)")
            }
            guard let size = metadata.sizeBytes else {
                throw TransferPlanningError.unknownSize(path: path)
            }
            if let commit = metadata.commitHash { commits.insert(commit) }
            files.append(PlannedFile(path: path, sizeBytes: size, etag: metadata.etag,
                                     sha256: metadata.sha256))
        }
        // A branch that moved between two HEADs shows up as two commits; planning across
        // it would mix trees.
        guard commits.count == 1, let commit = commits.first else {
            throw TransferPlanningError.unresolvedCommit(ref.description)
        }
        let resolved = ResolvedTransfer(commit: commit, files: files)
        AppLog.storage.info(
            "resolved \(ref.description) at \(declared) to commit \(commit): "
                + "\(files.count) files, \(ByteFormat.fileSize(resolved.totalBytes))")
        return resolved
    }
}
