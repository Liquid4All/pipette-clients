import Foundation

nonisolated extension PipetteWorkspace {
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
        filterJobManifests { _ in true }
    }

    /// Every job's list summary, newest first. Each full manifest is dropped once summarized.
    func loadJobSummaries() -> [JobSummary] {
        jobDirs().compactMap { dir in
            autoreleasepool {
                Self.decodeJobManifest(at: dir.appendingPathComponent("manifest.json")).map(JobSummary.init)
            }
        }.sorted { $0.createdAt > $1.createdAt }
    }

    /// The manifests `isIncluded` keeps, newest first. Each is decoded and judged in turn, so
    /// only the kept ones stay in memory.
    func filterJobManifests(_ isIncluded: (JobManifest) -> Bool) -> [JobManifest] {
        jobDirs().compactMap { dir in
            // Per job: decoding makes autoreleased Foundation objects that would otherwise
            // pile up for every manifest on the device until the run loop drains.
            autoreleasepool { () -> JobManifest? in
                guard let manifest = Self.decodeJobManifest(at: dir.appendingPathComponent("manifest.json")),
                      isIncluded(manifest)
                else { return nil }
                return manifest
            }
        }.sorted { $0.createdAt > $1.createdAt }
    }

    /// The manifests `include` picks from a peek at each one, newest first. Only those are
    /// decoded in full. One the peek cannot read is decoded anyway, so it never hides a job.
    func loadJobManifests(where include: (JobId, JobManifestPeek) -> Bool) -> [JobManifest] {
        jobDirs().compactMap { dir in
            autoreleasepool { () -> JobManifest? in
                let path = dir.appendingPathComponent("manifest.json")
                guard let data = try? Data(contentsOf: path) else { return nil }
                if let peek = try? JSONDecoder().decode(JobManifestPeek.self, from: data),
                   !include(JobId(dir.lastPathComponent), peek) {
                    return nil
                }
                return Self.decodeJobManifest(at: path)
            }
        }.sorted { $0.createdAt > $1.createdAt }
    }

    /// Every directory under `jobsDir`; each is named by its job's id.
    private func jobDirs() -> [URL] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: jobsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return dirs.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }
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
    /// A manifest written before a wire rename can name a type this build has no case for
    /// (`unknownModelType("hf_gguf_text")`); such a job lists and runs nowhere and logs on
    /// every read. Results live under `results/` keyed by cell, so they are deleted by the
    /// cell ids still readable from the raw JSON; unreadable ids are left to the quota sweep.
    ///
    /// Only a decode failure qualifies: an unreadable file may be a transient I/O error.
    ///
    /// A job holding a `remotePending` result is kept: an older build the farm installs
    /// reads a newer manifest as undecodable, and the newer build submits that result once
    /// it is back. An obsolete job holding one is kept too, since the two look the same.
    @discardableResult
    func discardUndecodableJobs() -> [String] {
        let discarded = jobDirs().compactMap { dir in
            autoreleasepool { () -> String? in
                guard let data = try? Data(contentsOf: dir.appendingPathComponent("manifest.json")),
                      (try? JobManifestSchema.decode(data)) == nil
                else { return nil }
                let cellIds = strandedCellIds(in: data)
                guard !cellIds.contains(where: { results.location(of: $0) == .remotePending }) else {
                    return nil
                }
                cellIds.forEach { results.delete($0) }
                guard (try? FileManager.default.removeItem(at: dir)) != nil else { return nil }
                return dir.lastPathComponent
            }
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
        // Only jobs a kill or an interrupted run left behind are decoded in full.
        let interrupted = loadJobManifests { jobId, peek in
            // A claimed job's results are for its claimer; with none left nothing reads it,
            // and keeping it grows the job list by one per claim.
            if peek.claimJobId != nil,
               peek.cells.allSatisfy({ results.location(of: $0.cellId) == nil }) {
                try? FileManager.default.removeItem(at: jobDir(jobId: jobId))
                return false
            }
            return FileManager.default.fileExists(atPath: activeCellSentinelURL(jobId: jobId).path)
                || peek.needsRunStateRecovery
        }
        for manifest in interrupted {
            var recovered = manifest
            var changed = false
            // A leftover sentinel is the only trace of a jetsam kill. Apply it before the
            // generic interrupted-state recovery.
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

    /// True when the cell's payload was written during the sentinel's attempt: the
    /// benchmark finished and the kill landed before the completed-status save, so recovery
    /// keeps the result instead of re-running the cell. An older payload is a previous
    /// attempt's and proves nothing.
    func crashPayloadIsFresh(_ sentinel: ActiveCellSentinel, jobId: JobId) -> Bool {
        guard let url = results.payloadPath(of: sentinel.cellId),
              let values = try? url.resourceValues(forKeys: [.contentModificationDateKey]),
              let modified = values.contentModificationDate,
              let started = JobDateFormat.iso8601.date(from: sentinel.startedAt)
        else { return false }
        // 1s slack: ISO8601 round-trips drop sub-second precision.
        return modified >= started.addingTimeInterval(-1)
    }

    /// Remove the job and the results its cells produced. Results sit in
    /// `results/<location>/<cellId>/`, outside the job tree, so removing the tree alone
    /// would orphan them, still counted against the storage quota.
    func deleteJob(jobId: JobId) {
        for cell in loadJobManifest(jobId: jobId)?.cells ?? [] {
            results.delete(cell.cellId)
        }
        try? FileManager.default.removeItem(at: jobDir(jobId: jobId))
    }
}
