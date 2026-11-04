import Foundation

// Whole-model admission: the gate every byte-adding path passes before a transfer
// starts. The CLI sweeps after publishing instead; `docs/storage-quota.md` says why
// mobile does not.
//
// One actor, one lock: `admit` surveys, evicts, remeasures and records a reservation
// under its isolation, so two concurrent callers cannot both be granted the same free
// bytes. The lock is not held across the transfer; the caller hands the grant back
// through `release`.
//
// A grant needs U + R + N <= Q (store usage, outstanding grants and this request within
// the quota) and R + N + H <= F (the same against free space, keeping headroom H back).
// R counts each live grant in full until its transfer ends, because partials land where
// U cannot see them (URLSession's temporary area, or a hub snapshot excluded from U while
// its grant stands). U includes the hub cache: `HubMLXModelDownloader` stages outside the
// store root.

/// Why admission refused a transfer. Each case carries its own numbers because each has
/// a different remedy: raise the quota, delete a pinned model, or free space outside the app.
nonisolated enum AdmissionError: Error, Equatable, LocalizedError {
    /// Refused rather than attempted: an unsized transfer is the one that overruns the
    /// quota, and no sweep can give bytes back to a volume that filled mid-write.
    case unknownSize(artifact: String)
    /// Even after evicting everything eligible. `protectedBytes` is what pins kept out of
    /// reach, which tells a person whether deleting something would help.
    case exceedsQuota(artifact: String, requiredBytes: Int64, availableBytes: Int64,
                      protectedBytes: Int64, quotaBytes: Int64)
    /// The quota allows it but the volume does not; eviction cannot fix this.
    case insufficientFreeSpace(artifact: String, requiredBytes: Int64, freeBytes: Int64,
                               headroomBytes: Int64)
    /// Proceeding would let a relaunched process promise the same bytes to another transfer.
    case reservationPersistenceFailed(artifact: String)

    var errorDescription: String? {
        switch self {
        case let .unknownSize(artifact):
            return "`\(artifact)` reports no download size, so it cannot be admitted; "
                + "the server must send a size for this model"
        case let .exceedsQuota(artifact, required, available, protected, quota):
            var sentence = "`\(artifact)` needs \(ByteFormat.fileSize(required)) but only "
                + "\(ByteFormat.fileSize(available)) of the "
                + "\(ByteFormat.fileSize(quota)) storage quota is free"
            if protected > 0 {
                sentence += "; \(ByteFormat.fileSize(protected)) is held by models a "
                    + "running or paused job needs"
            }
            return sentence + ". Raise the storage quota or delete a model."
        case let .insufficientFreeSpace(artifact, required, free, headroom):
            return "`\(artifact)` needs \(ByteFormat.fileSize(required)) but the device has "
                + "\(ByteFormat.fileSize(free)) free and keeps \(ByteFormat.fileSize(headroom)) "
                + "in reserve. Free up space on the device."
        case let .reservationPersistenceFailed(artifact):
            return "`\(artifact)` cannot start because its storage reservation could not be saved. "
                + "Try the download again."
        }
    }
}

nonisolated struct AdmissionRequest: Sendable, Equatable {
    /// Pinned for the life of the grant, so admission never evicts the partial it is
    /// about to resume.
    let key: ModelStorageKey
    /// `Model.description`, used to name the artifact in a refusal.
    let label: String
    /// Every byte the transfer will write, staging copies included.
    let declaredBytes: Int64?
}

/// A granted byte budget, handed back to `release` when the transfer ends, however it ends.
nonisolated struct Reservation: Sendable, Codable, Equatable, Identifiable {
    let id: UUID
    let key: ModelStorageKey
    let label: String
    let bytes: Int64
    let grantedAt: Date
}

/// A transfer the caller knows is still running, for `reconcile`.
nonisolated struct ActiveWriter: Sendable, Equatable {
    let key: ModelStorageKey
}

actor ModelAdmission {
    /// iOS starts evicting caches and refusing writes well before zero free bytes, so a
    /// granted transfer must not be what fills the device.
    static let headroomBytes: Int64 = 1 << 30

    private let ws: PipetteWorkspace
    /// Injected so a test can state the volume's answer; on a developer's disk every
    /// budget fits and the free-space check is never exercised.
    private let freeBytes: @Sendable (URL) -> Int64?
    /// Persisted, so a relaunch knows what an interrupted transfer was promised.
    private var grants: [UUID: Grant] = [:]
    /// Grants inherited from the ledger at init. Launch reconciliation may finish after
    /// this process has admitted new work; only these are eligible for that cleanup.
    private let restoredGrantIDs: Set<UUID>

    private struct Grant: Sendable, Codable, Equatable {
        let reservation: Reservation
    }

    init(ws: PipetteWorkspace,
         freeBytes: @escaping @Sendable (URL) -> Int64? = ModelAdmission.freeBytes(at:)) {
        self.ws = ws
        self.freeBytes = freeBytes
        let restored = Self.load(from: Self.ledgerURL(under: ws.root))
        grants = restored
        restoredGrantIDs = Set(restored.keys)
    }

    // MARK: - The gate

    /// Evicts here rather than after publishing (the CLI's way): a refusal costs nothing,
    /// whereas finding the overrun after the bytes land means the device is already full.
    func admit(_ request: AdmissionRequest, pins: SweepPins) throws(AdmissionError) -> Reservation {
        guard let need = request.declaredBytes else {
            throw .unknownSize(artifact: request.label)
        }
        var pins = pins
        pins.models.insert(request.key)

        let quota = ws.storageQuotaBytes
        let reserved = outstandingBytes
        // `survey` does not see the hub cache. A repo whose pull is still running is
        // excluded because its own grant already covers its snapshot.
        let hub = ws.hubBytes(excluding: pins.hubRepos)
        let ceiling = quota - reserved - need - hub

        var survey = ws.survey(pinning: pins)
        if ceiling >= 0, survey.usedBytes > ceiling {
            let sweep = plan(survey, quotaBytes: ceiling, pins: pins)
            if !sweep.evictions.isEmpty {
                AppLog.storage.info(
                    "admission evicting \(sweep.evictions.count) entries "
                        + "(\(ByteFormat.fileSize(sweep.freedBytes))) to admit \(request.label)")
                ws.reclaim(sweep, pinning: pins)
                survey = ws.survey(pinning: pins)
            }
        }

        guard ceiling >= 0, survey.usedBytes <= ceiling else {
            let available = max(0, quota - reserved - hub - survey.usedBytes)
            let error = AdmissionError.exceedsQuota(
                artifact: request.label, requiredBytes: need, availableBytes: available,
                protectedBytes: Self.protectedBytes(survey, pins: pins), quotaBytes: quota)
            AppLog.storage.warning("admission refused \(request.label): quota")
            throw error
        }

        // Measured after eviction so the freed bytes count. `nil` means the platform did
        // not answer, not that the device is full, so it does not refuse.
        if let free = freeBytes(ws.root) {
            guard reserved + need + Self.headroomBytes <= free else {
                let error = AdmissionError.insufficientFreeSpace(
                    artifact: request.label, requiredBytes: reserved + need,
                    freeBytes: free, headroomBytes: Self.headroomBytes)
                AppLog.storage.warning("admission refused \(request.label): free space")
                throw error
            }
        }

        let reservation = Reservation(id: UUID(), key: request.key, label: request.label,
                                      bytes: need, grantedAt: Date())
        grants[reservation.id] = Grant(reservation: reservation)
        do {
            try persist()
        } catch {
            grants.removeValue(forKey: reservation.id)
            AppLog.storage.error(
                "could not save the reservation for \(request.label): \(error.localizedDescription)")
            throw .reservationPersistenceFailed(artifact: request.label)
        }
        AppLog.storage.info(
            "admitted \(request.label) (\(ByteFormat.fileSize(need))); "
                + "outstanding \(ByteFormat.fileSize(outstandingBytes))")
        return reservation
    }

    /// Idempotent: releasing twice, or releasing a grant `reconcile` already dropped, is fine.
    func release(_ reservation: Reservation) {
        guard grants.removeValue(forKey: reservation.id) != nil else { return }
        persistBestEffort()
        AppLog.storage.info(
            "released the reservation for \(reservation.label); "
                + "outstanding \(ByteFormat.fileSize(outstandingBytes))")
    }

    /// Drop the grants of transfers that are no longer running. Grants outlive the process
    /// on purpose, but one whose transfer died with it would shrink the quota forever.
    ///
    /// `writers` must include any transfer the caller cannot rule out: only the caller sees
    /// URLSession tasks and on-disk partials. Over-keeping a grant costs a retry; dropping
    /// a live one corrupts the accounting.
    @discardableResult
    func reconcile(against writers: [ActiveWriter]) -> [Reservation] {
        reconcile(grantIDs: Set(grants.keys), against: writers)
    }

    /// Only grants inherited from the ledger. URLSession task discovery is asynchronous,
    /// so this process may already hold fresh grants that launch reconciliation must not touch.
    @discardableResult
    func reconcileRestored(against writers: [ActiveWriter]) -> [Reservation] {
        reconcile(grantIDs: restoredGrantIDs, against: writers)
    }

    private func reconcile(
        grantIDs: Set<UUID>, against writers: [ActiveWriter]
    ) -> [Reservation] {
        let alive = Set(writers.map(\.key))
        let abandoned = grants.filter {
            grantIDs.contains($0.key) && !alive.contains($0.value.reservation.key)
        }
        if !abandoned.isEmpty {
            for (id, grant) in abandoned {
                grants[id] = nil
                AppLog.storage.info(
                    "reclaimed the reservation for \(grant.reservation.label) "
                        + "(\(ByteFormat.fileSize(grant.reservation.bytes)) outstanding): "
                        + "no transfer is writing it")
            }
            persistBestEffort()
        }
        return grants.values
            .map(\.reservation)
            .filter { grantIDs.contains($0.id) }
            .sorted { $0.grantedAt < $1.grantedAt }
    }

    /// `R`.
    var outstandingBytes: Int64 {
        grants.values.reduce(0) { $0 + $1.reservation.bytes }
    }

    var reservations: [Reservation] {
        grants.values.map(\.reservation).sorted { $0.grantedAt < $1.grantedAt }
    }

    // MARK: - Measurement

    /// `ForImportantUsage` counts purgeable caches the system would reclaim for a write
    /// like a model download; plain `volumeAvailableCapacity` under-reports on a phone
    /// with full caches and would refuse downloads that would succeed.
    static func freeBytes(at url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }

    /// Garbage is never pinned, so only live model entries count.
    private static func protectedBytes(_ survey: StorageSurvey, pins: SweepPins) -> Int64 {
        survey.entries.reduce(0) { total, entry in
            guard case let .model(key, _, _) = entry.kind, pins.models.contains(key) else {
                return total
            }
            return total + entry.sizeBytes
        }
    }

    private static func directoryBytes(at url: URL) -> Int64 {
        entrySizeBytes(at: url)
    }

    // MARK: - The ledger

    private static func ledgerURL(under root: URL) -> URL {
        root.appendingPathComponent("state/admission.json", isDirectory: false)
    }

    /// Under `state/`, not `models/`: the survey treats unrecognized children of the store
    /// root as garbage, so a ledger there would be swept by the first admission that needed room.
    private var ledgerURL: URL { Self.ledgerURL(under: ws.root) }

    private static func load(from url: URL) -> [UUID: Grant] {
        guard let data = try? Data(contentsOf: url),
              let grants = try? JSONDecoder().decode([Grant].self, from: data)
        else { return [:] }
        // A duplicate id means a damaged ledger; keep the last rather than trap, since
        // crashing launch over bookkeeping would be unrecoverable.
        return Dictionary(grants.map { ($0.reservation.id, $0) }, uniquingKeysWith: { _, last in last })
    }

    /// Admission treats failure as a refusal: a transfer must not start unless the next
    /// process can see its reservation.
    private func persist() throws {
        let url = ledgerURL
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(
            grants.values.sorted { $0.reservation.grantedAt < $1.reservation.grantedAt })
            .write(to: url, options: .atomic)
    }

    /// A failed release write leaves a stale grant on disk, which is conservative: the
    /// next launch reconciles it instead of promising live bytes twice.
    private func persistBestEffort() {
        do {
            try persist()
        } catch {
            AppLog.storage.error(
                "could not write the admission ledger: \(error.localizedDescription)")
        }
    }
}
