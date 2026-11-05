import Foundation

// Stage a fetch outside the entry, then publish it in one step: the crate's
// `install_dir_computing_manifest`, and section 3 of `docs/pipette-ios/model-store-design.md`.
//
//     models/.staging/<key>/<attempt-id>/{transfer.json, blobs/…}
//     models/<key>/{manifest.json, blobs/…}
//
// The manifest is the last thing written into staging: it is what makes a directory an
// entry, so a crash before it leaves an orphan the quota survey reads as garbage, never a
// half-built entry `find` would resolve. Staging lives on disk so `staged()` can find
// attempts that outlived the process.

/// Credential-free, so an abandoned staging directory cannot leak a token.
nonisolated struct TransferPlan: Codable, Equatable, Sendable {
    /// An attempt with any other version is not resumed but left for the sweep, as with
    /// `ModelManifest`. There is no migrator.
    static let planVersion = 1

    var planVersion: Int
    /// Format and prefix are read off this rather than stored twice.
    var declared: Model
    var startedAt: Date
    /// The frozen commit, file list, lengths and digests. Optional because staging opens
    /// before resolution, and the GGUF path has no listing to freeze.
    var resolved: ResolvedTransfer?

    init(declared: Model, startedAt: Date = Date(), resolved: ResolvedTransfer? = nil) {
        self.planVersion = Self.planVersion
        self.declared = declared.withoutAuthToken
        self.startedAt = startedAt
        self.resolved = resolved
    }

    enum CodingKeys: String, CodingKey {
        case planVersion = "plan_version"
        case declared
        case startedAt = "started_at"
        case resolved
    }

    static let fileName = "transfer.json"
}

/// Addressable on disk so whoever observes completion can finish it: the awaiting task,
/// a background-session delegate, or the next launch.
nonisolated struct StagedEntry: Sendable, Equatable {
    let key: ModelStorageKey
    /// So a retry never writes into a directory an earlier attempt is still filling.
    let attemptId: String
    /// `models/.staging/<key>/<attempt-id>`.
    let root: URL
    /// A repo subtree keeps its prefix here exactly as under the published entry.
    let blobs: URL
    let plan: TransferPlan
}

nonisolated enum StagingError: Error, Equatable, LocalizedError {
    case notStorable(String)
    case io(doing: String, source: String)
    /// Refused before publishing, so a half-transferred attempt never becomes an entry
    /// that `find` resolves and an engine fails to load.
    case incompletePayload(String)
    /// Absent or the wrong length against the frozen plan, which names the file and the
    /// gap: that is what tells a retry from a corrupt source.
    case truncatedFile(String, path: String, expectedBytes: Int64, actualBytes: Int64?)
    /// Right length, wrong content: distinct from a length failure so diagnostics can tell
    /// same-length corruption from an incomplete transfer.
    case digestMismatch(String, path: String, expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case let .notStorable(name): return "\(name) has no store entry to stage into."
        case let .io(doing, reason): return "\(doing): \(reason)"
        case let .incompletePayload(name):
            return "\(name) finished staging but its payload is incomplete, so it was "
                + "not published."
        case let .truncatedFile(name, path, expected, actual):
            if let actual {
                return "\(name) staged `\(path)` at \(ByteFormat.fileSize(actual)), but its "
                    + "source said \(ByteFormat.fileSize(expected)); it was not published."
            }
            return "\(name) is missing `\(path)`, which its source listed at "
                + "\(ByteFormat.fileSize(expected)); it was not published."
        case let .digestMismatch(name, path, expected, actual):
            return "\(name) staged `\(path)` with SHA-256 \(actual), but its source "
                + "said \(expected); it was not published."
        }
    }
}

/// `nonisolated`: staging moves multi-gigabyte directories and must run off the main actor.
nonisolated extension ModelArtifactStore {
    var stagingRoot: URL {
        modelsDir.appendingPathComponent(Entry.stagingDirName, isDirectory: true)
    }

    /// A fresh attempt id each time; an abandoned attempt costs only bytes the sweep reclaims.
    func stage(_ spec: Model) throws(StagingError) -> StagedEntry {
        let storable = spec.withoutAuthToken
        guard let key = ModelStorageKey.of(storable) else {
            throw .notStorable(storable.description)
        }
        let attemptId = UUID().uuidString
        let root = stagingRoot
            .appendingPathComponent(key.relativeDir, isDirectory: true)
            .appendingPathComponent(attemptId, isDirectory: true)
        let blobs = root.appendingPathComponent(Entry.blobsDirName, isDirectory: true)
        let plan = TransferPlan(declared: storable)
        do {
            try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
            try Self.writePlan(plan, at: root)
        } catch {
            throw .io(doing: "staging \(storable.description)", source: "\(error)")
        }
        AppLog.storage.info("staged \(storable.description) as attempt \(attemptId)")
        return StagedEntry(key: key, attemptId: attemptId, root: root, blobs: blobs, plan: plan)
    }

    /// Separate from `stage`, which runs before the transport has asked the Hub. Lets a
    /// later launch tell which commit an interrupted transfer's bytes belong to.
    @discardableResult
    func recordResolution(_ staged: StagedEntry,
                          _ resolved: ResolvedTransfer) -> StagedEntry {
        var plan = staged.plan
        plan.resolved = resolved
        let updated = StagedEntry(key: staged.key, attemptId: staged.attemptId,
                                  root: staged.root, blobs: staged.blobs, plan: plan)
        do {
            try Self.writePlan(plan, at: staged.root)
            AppLog.storage.info(
                "froze \(plan.declared.description) at commit \(resolved.commit): "
                    + "\(resolved.files.count) files, "
                    + "\(ByteFormat.fileSize(resolved.totalBytes))")
        } catch {
            // Best effort: the plan is a recovery aid, and the bytes publish without it.
            AppLog.storage.error(
                "could not record the frozen plan for \(plan.declared.description): \(error)")
        }
        return updated
    }

    /// Newest first. An attempt with no readable `transfer.json` is omitted: its
    /// coordinate is unknown, and the sweep treats it as garbage.
    func staged() -> [StagedEntry] {
        Self.children(of: stagingRoot)
            .flatMap { keyDir in
                Self.children(of: keyDir).compactMap { attempt -> StagedEntry? in
                    let key = ModelStorageKey(value: keyDir.lastPathComponent)
                    guard let plan = Self.readPlan(at: attempt) else { return nil }
                    return StagedEntry(
                        key: key, attemptId: attempt.lastPathComponent, root: attempt,
                        blobs: attempt.appendingPathComponent(Entry.blobsDirName,
                                                              isDirectory: true),
                        plan: plan)
                }
            }
            .sorted { $0.plan.startedAt > $1.plan.startedAt }
    }

    /// Verify the payload, write the manifest into staging last (so a write failure
    /// publishes nothing), then swap the directory into place.
    @discardableResult
    func publish(_ staged: StagedEntry) throws(StagingError) -> URL {
        let entry = modelsDir.appendingPathComponent(staged.key.relativeDir, isDirectory: true)
        let fm = FileManager.default
        // One transfer per key publishes once, so a missing root was swept or discarded,
        // not raced. Returning the existing entry would bind the caller to old bytes.
        guard fm.fileExists(atPath: staged.root.path) else {
            throw .io(doing: "publishing \(staged.plan.declared.description)",
                      source: "the staged attempt is gone")
        }
        let storable = staged.plan.declared
        guard let stored = storable.toStored(base: staged.key.blobsDir) else {
            throw .notStorable(storable.description)
        }
        // The same `toStored` rule against the attempt's `blobs`, so the staged layout is
        // never re-derived another way (the drift `stored` exists to prevent).
        guard let stagedForm = storable.toStored(base: Entry.blobsDirName) else {
            throw .notStorable(storable.description)
        }
        let now = Date()
        let probe = ModelManifest(fetchedAt: now, lastUsedAt: now, declared: storable,
                                  stored: stagedForm, blobsBytes: nil)
        guard let stagedPaths = Self.bound(probe, under: staged.root)?.boundPaths,
              Self.payloadIsPresent(at: stagedPaths.payload),
              stagedPaths.mmproj.map(Self.payloadIsPresent(at:)) ?? true
        else {
            throw .incompletePayload(storable.description)
        }
        if let resolved = staged.plan.resolved,
           let short = Self.firstShortfall(of: resolved, under: staged.blobs) {
            throw .truncatedFile(storable.description, path: short.path,
                                 expectedBytes: short.expected, actualBytes: short.actual)
        }
        if let resolved = staged.plan.resolved {
            do {
                if let mismatch = try Self.firstDigestMismatch(of: resolved,
                                                               under: staged.blobs) {
                    throw StagingError.digestMismatch(
                        storable.description, path: mismatch.path,
                        expected: mismatch.expected, actual: mismatch.actual)
                }
            } catch let error as StagingError {
                throw error
            } catch {
                throw .io(doing: "verifying \(storable.description)", source: "\(error)")
            }
        }
        let asPublished = ModelManifest(fetchedAt: now, lastUsedAt: now,
                                        declared: storable, stored: stored,
                                        blobsBytes: entrySizeBytes(at: staged.blobs),
                                        resolved: staged.plan.resolved)
        do {
            // Recovery state, not model data: remove it before the manifest makes this
            // directory publishable.
            let transfer = staged.root.appendingPathComponent(TransferPlan.fileName)
            if fm.fileExists(atPath: transfer.path) { try fm.removeItem(at: transfer) }
            try asPublished.write(to: ModelManifest.manifestURL(inEntryDir: staged.root))
            try replaceEntry(at: entry, with: staged.root)
        } catch let error as StagingError {
            throw error
        } catch {
            throw .io(doing: "publishing \(storable.description)", source: "\(error)")
        }
        AppLog.storage.info(
            "published \(storable.description) from attempt \(staged.attemptId) "
                + "(\(ByteFormat.fileSize(asPublished.blobsBytes ?? 0)))")
        discardStagingParentIfEmpty(for: staged)
        return entry
    }

    /// Best-effort: anything left behind is garbage the sweep reclaims.
    func discard(_ staged: StagedEntry) {
        try? FileManager.default.removeItem(at: staged.root)
        discardStagingParentIfEmpty(for: staged)
        AppLog.storage.info(
            "discarded the staged attempt \(staged.attemptId) for "
                + "\(staged.plan.declared.description)")
    }

    // MARK: - Internals

    /// `replaceItemAt` needs something to replace, so a first install is a plain move.
    /// Either way the store holds the old entry or the new one, never neither.
    private func replaceEntry(at entry: URL, with replacement: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: entry.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        guard fm.fileExists(atPath: entry.path) else {
            try fm.moveItem(at: replacement, to: entry)
            return
        }
        _ = try fm.replaceItemAt(entry, withItemAt: replacement,
                                 backupItemName: nil, options: [])
    }

    /// `payloadIsPresent` only says some bytes arrived; this catches lost files and a
    /// truncated sidecar. A planned path is repo-relative and sits at `path` under the
    /// attempt's `blobs/`, whole repo or subtree.
    private static func firstShortfall(
        of resolved: ResolvedTransfer, under blobs: URL
    ) -> (path: String, expected: Int64, actual: Int64?)? {
        resolved.files.lazy.compactMap { file -> (String, Int64, Int64?)? in
            let url = blobs.appendingPathComponent(file.path)
            let actual = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
            guard let actual else { return (file.path, file.sizeBytes, nil) }
            return Int64(actual) == file.sizeBytes
                ? nil
                : (file.path, file.sizeBytes, Int64(actual))
        }.first
    }

    /// A file without a source digest is not hashed: exact length is the fallback for
    /// sources that cannot attest to content.
    private static func firstDigestMismatch(
        of resolved: ResolvedTransfer, under blobs: URL
    ) throws -> (path: String, expected: String, actual: String)? {
        for file in resolved.files {
            guard let expected = file.sha256 else { continue }
            let url = blobs.appendingPathComponent(file.path)
            let actual = try ModelFetch.sha256Hex(ofFileAt: url)
            if actual != expected.value {
                return (file.path, expected.value, actual)
            }
        }
        return nil
    }

    /// A directory must hold at least one regular file: a whole-repo payload is the
    /// `blobs/` that `stage` creates, so existence alone would publish an empty entry.
    /// Whether it is loadable is the installer's per-format validator's question.
    private static func payloadIsPresent(at path: String) -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
            return false
        }
        guard isDirectory.boolValue else { return true }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        guard let walker = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { return false }
        return walker.contains {
            guard let url = $0 as? URL else { return false }
            return (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true
        }
    }

    private func discardStagingParentIfEmpty(for staged: StagedEntry) {
        let parent = staged.root.deletingLastPathComponent()
        let fm = FileManager.default
        let remaining = (try? fm.contentsOfDirectory(atPath: parent.path)) ?? []
        if remaining.isEmpty { try? fm.removeItem(at: parent) }
    }

    private static func writePlan(_ plan: TransferPlan, at root: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(plan)
            .write(to: root.appendingPathComponent(TransferPlan.fileName), options: .atomic)
    }

    private static func readPlan(at attempt: URL) -> TransferPlan? {
        guard let data = try? Data(contentsOf: attempt.appendingPathComponent(
            TransferPlan.fileName)) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let plan = try? decoder.decode(TransferPlan.self, from: data),
              plan.planVersion == TransferPlan.planVersion
        else { return nil }
        return plan
    }

    private static func children(of url: URL) -> [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil, options: [])) ?? []
    }
}
