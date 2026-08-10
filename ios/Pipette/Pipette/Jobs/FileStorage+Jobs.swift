import Foundation

extension FileStorage {
    func jobDir(jobId: JobId) -> URL {
        jobsDir.appendingPathComponent(jobId.value, isDirectory: true)
    }

    func saveJobManifest(_ manifest: JobManifest) {
        var manifest = manifest
        manifest.schemaVersion = JobManifestSchema.currentVersion
        let dir = jobDir(jobId: manifest.jobId)
        try? FileManager.default.createDirectory(
            at: dir.appendingPathComponent("cells", isDirectory: true),
            withIntermediateDirectories: true
        )
        do {
            let data = try Coding.encoder.encode(manifest)
            try data.write(to: dir.appendingPathComponent("manifest.json"), options: .atomic)
        } catch {
            // Loud on purpose: a dropped manifest write loses job state (a
            // newly adopted serverJobId, a status change) with no other trace.
            AppLog.storage.error("Failed to save job manifest \(manifest.jobId): \(error)")
        }
    }

    func loadJobManifest(jobId: JobId) -> JobManifest? {
        Self.decodeJobManifest(at: jobDir(jobId: jobId).appendingPathComponent("manifest.json"))
    }

    func loadAllJobManifests() -> [JobManifest] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: jobsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return dirs.compactMap { dir -> JobManifest? in
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else {
                return nil
            }
            return Self.decodeJobManifest(at: dir.appendingPathComponent("manifest.json"))
        }.sorted { $0.createdAt > $1.createdAt }
    }

    private static func decodeJobManifest(at path: URL) -> JobManifest? {
        guard let data = try? Data(contentsOf: path) else { return nil }
        do {
            return try JobManifestSchema.decode(data)
        } catch {
            // Loud on purpose: a silently skipped manifest reads as "the job vanished",
            // which is exactly how a botched schema change would otherwise present.
            // `discardUndecodableJobs()` is what eventually removes it.
            AppLog.storage.error("Failed to decode job manifest at \(path.path): \(error)")
            return nil
        }
    }

    /// Delete jobs whose manifest no longer decodes, and the results their cells produced.
    ///
    /// A manifest written before a wire rename names a type this build has no case for
    /// (`unknownModelType("hf_gguf_text")`). Such a job lists nowhere, runs nowhere, and
    /// logs a decode error on every sweep, so it is removed rather than left to accumulate.
    ///
    /// Results are keyed by cell under `results/`, not stored inside the job — removing the
    /// job tree alone would strand them, invisible to every listing and still counted
    /// against the quota. What failed is the *typed* decode, not the JSON, so the cell ids
    /// are still readable, which is enough to clean up after a manifest that no longer
    /// parses. Ids that aren't readable leave their payloads behind; the quota sweep is
    /// what reclaims those.
    ///
    /// Only a decode failure qualifies. An unreadable file is left alone — that can be a
    /// transient I/O error, and deleting a job over one would destroy history that is
    /// merely momentarily unavailable.
    @discardableResult
    func discardUndecodableJobs() -> [String] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: jobsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let discarded = dirs.compactMap { dir -> String? in
            guard (try? dir.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true,
                  let data = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
                  (try? JobManifestSchema.decode(data)) == nil
            else { return nil }
            strandedCellIds(in: data).forEach { results.delete($0) }
            guard (try? FileManager.default.removeItem(at: dir)) != nil else { return nil }
            return dir.lastPathComponent
        }

        if !discarded.isEmpty {
            AppLog.storage.info(
                "discarded \(discarded.count) undecodable job(s): \(discarded.joined(separator: ", "))")
        }
        return discarded
    }

    /// The cell ids of a manifest that failed its typed decode, read straight from the JSON.
    private func strandedCellIds(in data: Data) -> [CellId] {
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let cells = root?["cells"] as? [[String: Any]] ?? []
        return cells.compactMap { $0["cellId"] as? String }.map(CellId.init)
    }

    func recoverInterruptedJobs() {
        for manifest in loadAllJobManifests() {
            var recovered = manifest
            var changed = false
            // A leftover sentinel means the process died while that cell was
            // executing — a jetsam OOM kill never reaches an error path, so
            // this is the only place the crash becomes visible. Apply it
            // before the generic interrupted-state recovery below.
            if let sentinel = loadActiveCellSentinel(jobId: manifest.jobId) {
                changed = recovered.applyCrashEvidence(
                    sentinel: sentinel,
                    payloadIsFresh: crashPayloadIsFresh(sentinel, jobId: manifest.jobId)
                ) || changed
                clearActiveCellSentinel(jobId: manifest.jobId)
            }
            changed = recovered.recoverInterruptedRunState() || changed
            if changed {
                saveJobManifest(recovered)
            }
        }
    }

    // MARK: - Active-cell crash sentinel

    private func activeCellSentinelURL(jobId: JobId) -> URL {
        jobDir(jobId: jobId).appendingPathComponent("active-cell.json")
    }

    func saveActiveCellSentinel(_ sentinel: ActiveCellSentinel, jobId: JobId) {
        try? FileManager.default.createDirectory(
            at: jobDir(jobId: jobId),
            withIntermediateDirectories: true
        )
        do {
            let data = try Coding.encoder.encode(sentinel)
            try data.write(to: activeCellSentinelURL(jobId: jobId), options: .atomic)
        } catch {
            // Loud on purpose: without the sentinel on disk, a jetsam kill
            // mid-cell leaves no crash evidence for recovery to act on.
            AppLog.storage.error("Failed to save active-cell sentinel for \(jobId): \(error)")
        }
    }

    func loadActiveCellSentinel(jobId: JobId) -> ActiveCellSentinel? {
        guard let data = try? Data(contentsOf: activeCellSentinelURL(jobId: jobId)) else { return nil }
        return try? JSONDecoder().decode(ActiveCellSentinel.self, from: data)
    }

    func clearActiveCellSentinel(jobId: JobId) {
        try? FileManager.default.removeItem(at: activeCellSentinelURL(jobId: jobId))
    }

    /// True when the cell's payload was written during the sentinel's attempt
    /// (modified at/after `startedAt`): the benchmark finished and the kill
    /// landed between the payload write and the completed-status save, so
    /// recovery can keep the result instead of re-running the cell. An older
    /// payload is a previous attempt's and proves nothing about this one.
    func crashPayloadIsFresh(_ sentinel: ActiveCellSentinel, jobId: JobId) -> Bool {
        guard let url = results.payloadPath(of: sentinel.cellId),
              let values = try? url.resourceValues(forKeys: [.contentModificationDateKey]),
              let modified = values.contentModificationDate,
              let started = JobDateFormat.iso8601.date(from: sentinel.startedAt)
        else { return false }
        // 1s slack: ISO8601 round-trips drop sub-second precision.
        return modified >= started.addingTimeInterval(-1)
    }

    /// Remove the job and the results its cells produced.
    ///
    /// The results are deleted explicitly because they no longer live under the job:
    /// they sit in `results/<location>/<cellId>/`, keyed by cell, as the crate files
    /// them. Removing the job tree alone would leave them orphaned — invisible to every
    /// listing, and still counted against the storage quota.
    func deleteJob(jobId: JobId) {
        for cell in loadJobManifest(jobId: jobId)?.cells ?? [] {
            results.delete(cell.cellId)
        }
        try? FileManager.default.removeItem(at: jobDir(jobId: jobId))
    }
}
