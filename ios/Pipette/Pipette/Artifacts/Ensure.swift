import Foundation

// Find-or-fetch into the store, after `crates/pipette-artifacts/src/ensure.rs`: admit,
// fetch, publish, sweep (in `DownloadCoordinator.sweepAfterInstall`), return the bound
// model. A cache hit neither admits nor sweeps.
//
// Two divergences from the crate. Admission refuses before any byte moves unless the
// whole model fits the quota and free space (the crate sweeps after publishing; see
// `docs/storage-quota.md`). And the UI and a running job can want the same weights at
// once, so the byte-adding half goes through `TransferRegistry`: one transfer, one grant,
// any number of waiters.

/// The crate's `ensure_model`. Returns the bound model, so a caller gets launchable paths
/// rather than a promise that a transfer was queued.
///
/// `familyId` / `quant` are the catalog labels the transfer is displayed under. A `nil`
/// `declaredSizeBytes` is resolved from the vault records or the Hub on a miss, and a
/// model nobody can size is refused.
///
/// `stallTimeout` bounds time without progress, not the whole fetch, so a slow link that
/// keeps moving is never cut off.
@MainActor
func ensureModel(
    _ declared: Model,
    ws: PipetteWorkspace,
    coordinator: DownloadCoordinator,
    familyId: String? = nil,
    quant: String? = nil,
    declaredSizeBytes: Int64? = nil,
    jobId: JobId? = nil,
    stallTimeout: Duration = .seconds(600),
    progress: ProgressSink? = nil
) async throws -> Model {
    try await ws.models.ensure(declared) { declared in
        // Checked on the miss path only, so an installed arm still resolves. Without it,
        // the miss would surface later as an unsized admission or a declined start, not
        // as `notFetchableHere`.
        guard declared.isFetchableHere else {
            throw ModelStoreError.notFetchableHere(declared.artifactName)
        }
        guard let key = ModelStorageKey.of(declared) else {
            throw ModelStoreError.notStorable(declared.description)
        }
        // Runs once per transfer in a task the registry owns, so a caller whose own task
        // is cancelled only stops waiting and cannot hand back a budget still being spent.
        do {
            try await coordinator.transfers.join(key) {
                try await fetchAndWait(declared, ws: ws, coordinator: coordinator,
                                       familyId: familyId, quant: quant,
                                       declaredSizeBytes: declaredSizeBytes, jobId: jobId,
                                       stallTimeout: stallTimeout, progress: progress)
            }
        } catch is TransferCancelled {
            // Distinct from `CancellationError`: this means the weights are not coming,
            // that means only this caller left.
            throw ModelStoreError.fetchCancelled(declared.artifactName)
        }
    }
}

/// The byte-adding half, run once per transfer by `TransferRegistry`.
@MainActor
private func fetchAndWait(
    _ declared: Model,
    ws: PipetteWorkspace,
    coordinator: DownloadCoordinator,
    familyId: String?,
    quant: String?,
    declaredSizeBytes: Int64?,
    jobId: JobId?,
    stallTimeout: Duration,
    progress: ProgressSink?
) async throws {
    // Only new work is admitted: an entry that already holds a grant would be budgeted
    // twice, and resuming a paused download offline (no size readable) would be refused.
    // `DownloadCoordinator` owns the grant and releases it when the entry's last row
    // retires, so a paused transfer resumes on its grant and no competing download can
    // take that budget in between.
    //
    // Retire failed rows before reading `hasTransfer`, or this call would skip admission,
    // start nothing, and report the earlier failure again.
    let retired = await coordinator.retireFailed(for: declared)
    if !retired.isEmpty {
        AppLog.storage.info(
            "retired failed transfer(s) before a new fetch model=\(declared.artifactName) "
                + "count=\(retired.count)")
    }
    let entry = ModelStorageKey.of(declared)
    let admitted = entry.map { coordinator.hasReservation(for: $0) } ?? false
    if !admitted, !coordinator.hasTransfer(for: declared) {
        let reservation = try await admit(
            declared, ws: ws, coordinator: coordinator,
            declaredSizeBytes: declaredSizeBytes, jobId: jobId)
        guard let entry else {
            await coordinator.admission.release(reservation)
            throw ModelStoreError.fetch(
                model: declared.artifactName, source: "the model names no store entry")
        }
        coordinator.holdReservation(reservation, for: entry)
    }
    coordinator.startDownload(declared, familyId: familyId, quant: quant, jobId: jobId)
    // A declined start enqueues nothing, so read its reason now instead of at the deadline.
    guard coordinator.hasTransfer(for: declared) else {
        if let entry { coordinator.releaseReservation(for: entry) }
        throw ModelStoreError.fetch(
            model: declared.artifactName,
            source: coordinator.errorMessage ?? "the download did not start")
    }
    try await awaitFetch(of: declared, ws: ws, coordinator: coordinator,
                         familyId: familyId, quant: quant,
                         stallTimeout: stallTimeout, progress: progress)
}

/// The crate's `pre_fetch_size` plus admission. A vault model is sized from its records
/// (`VaultFetch.admissionBytes`), anything else from the Hub. A failed record read throws
/// the `VaultFetchError`, the same typed refusal the download would.
@MainActor
private func admit(
    _ declared: Model,
    ws: PipetteWorkspace,
    coordinator: DownloadCoordinator,
    declaredSizeBytes: Int64?,
    jobId: JobId?
) async throws -> Reservation {
    let size: Int64?
    if let declaredSizeBytes {
        size = declaredSizeBytes
    } else if let files = ModelFetch.vaultFiles(declared), let first = files.first {
        let access = try VaultAccess.of(coordinator.vaultSession, path: first.path)
        size = VaultFetch.admissionBytes(declared, records: try await VaultFetch.records(
            files, access: access, http: coordinator.vaultHttpClient, jobId: jobId,
            hooks: coordinator.retryHooks))
    } else {
        size = await ModelFetch.declaredSizeBytes(
            declared, http: coordinator.downloadHttpClient)
    }
    guard let key = ModelStorageKey.of(declared) else {
        throw ModelStoreError.notStorable(declared.description)
    }
    let request = AdmissionRequest(key: key, label: declared.description, declaredBytes: size)
    do throws(AdmissionError) {
        return try await coordinator.admission.admit(
            request, pins: coordinator.sweepPins(justInstalled: nil))
    } catch {
        throw ModelStoreError.admission(error)
    }
}

/// Wait until the store can resolve `declared`, the transfer fails, pauses or vanishes,
/// or no byte arrives for `stallTimeout`.
///
/// Polls the store, not a transfer row: a vision model is two transfers in one entry.
///
/// Pauses already in place when the wait starts predate the caller (interrupted before a
/// relaunch) and are resumed. A later pause belongs to a person (`ModelsView` offers
/// Pause) and is reported at once, unless the network raised it, which is lifted once more.
///
/// A fetch that inherited a pause gets one restart from scratch (`restartStalled`) if the
/// resume ends in a failed or vanished transfer: the resume data can name a partial the
/// server will not continue, and a CLI-driven phone has nobody to clear it by hand. A
/// transfer that failed on its own (404, gated repo, full disk) is reported, not retried.
///
/// `progress` fires only when the whole percent changes, so callers need not deduplicate.
@MainActor
private func awaitFetch(
    of declared: Model,
    ws: PipetteWorkspace,
    coordinator: DownloadCoordinator,
    familyId: String?,
    quant: String?,
    stallTimeout: Duration,
    progress: ProgressSink?
) async throws {
    var deadline = ContinuousClock.now.advanced(by: stallTimeout)
    // A directory pull reports a fraction and no bytes, so either one moving counts.
    var lastMoved: (bytes: Int64?, fraction: Double?) = (nil, nil)
    var lastPercent = -1
    func report(_ fraction: Double) {
        let percent = Int((fraction * 100).rounded())
        guard percent != lastPercent else { return }
        lastPercent = percent
        // Not derived from the fraction: that is an average across rows and bytes are a
        // sum, so the two disagree for multi-file models.
        let bytes = coordinator.transferred(for: declared)
        progress?(FetchProgress(artifact: declared.description, fraction: fraction,
                                doneBytes: bytes?.done,
                                totalBytes: bytes?.total))
    }

    // Before the loop, so only pauses that predate the wait are resumed; a person who
    // pauses mid-wait is not overruled. One attempt is enough: `resume(key:)` leaves the
    // row `.queued`, so a resume that takes is out of `.paused` before the first poll.
    let inheritedAPause = await coordinator.resumePaused(for: declared)
    // The network-pause resume applies however the transfer was found; the restart only
    // to an inherited pause.
    var mayResumeAgain = true
    var mayRestart = inheritedAPause

    /// False leaves the caller to report the stall: no inherited pause, already restarted,
    /// offline, or refused.
    func restartOnce() async -> Bool {
        guard mayRestart else { return false }
        mayRestart = false
        return await coordinator.restartStalled(for: declared, familyId: familyId, quant: quant)
    }

    while ContinuousClock.now < deadline {
        if ws.models.bound(declared) != nil {
            report(1)
            return
        }
        if let reason = coordinator.failureReason(for: declared) {
            // Typed, so the worker can tell whether every device would get the same refusal.
            if let refusal = coordinator.vaultFailure(for: declared) { throw refusal }
            if await restartOnce() { continue }
            throw ModelStoreError.fetch(model: declared.artifactName, source: reason)
        }
        if coordinator.isPaused(declared) {
            // A network pause is resumed, never restarted: its partial is worth keeping.
            if mayResumeAgain, coordinator.isPausedByNetwork(declared) {
                mayResumeAgain = false
                if await coordinator.resumePaused(for: declared) { continue }
            }
            throw ModelStoreError.fetchPaused(declared.artifactName)
        }
        // After the failure and pause arms, which need the row. A completion clears the
        // row only once the store resolves it, so a vanished row here stopped short.
        if !coordinator.hasTransfer(for: declared) {
            if await restartOnce() { continue }
            throw ModelStoreError.fetch(
                model: declared.artifactName, source: "the transfer stopped without completing")
        }
        let fraction = coordinator.fraction(for: declared)
        let moved = (bytes: coordinator.transferred(for: declared)?.done, fraction: fraction)
        if moved != lastMoved {
            deadline = ContinuousClock.now.advanced(by: stallTimeout)
        }
        lastMoved = moved
        if let fraction { report(fraction) }
        try await Task.sleep(for: .seconds(1))
    }
    throw ModelStoreError.fetchTimedOut(declared.artifactName)
}
