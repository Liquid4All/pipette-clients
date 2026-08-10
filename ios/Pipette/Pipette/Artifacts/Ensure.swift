import Foundation

// Find-or-fetch into the store — the crate's `pipette-artifacts/src/ensure.rs`.
//
// The order is the crate's: refuse an oversize artifact before any transfer starts, fetch,
// publish, sweep back under the cap, return the bound model. Peak disk is the quota plus
// this artifact, and a cache hit publishes nothing and therefore does not sweep.
//
// Where upstream calls `fetch_model` inline, this awaits `DownloadCoordinator`: the two
// pre-flight and post-publish steps already live there (`startDownload`'s declared-size
// refusal and `sweepAfterInstall`), so this supplies the wait and the store supplies
// find/touch/publish/bind.
//
// The `progress` callback led upstream rather than following it: a phone had to show a
// multi-gigabyte transfer moving long before `fetch_model` said anything. The CLI has since
// grown the same shape — a `ProgressSink` on `ArtifactsContext`, reporting one artifact at
// a time and summing its parts — so the two now agree on what a fetch reports, and
// `TransferFormat` keeps them agreeing on how it reads.

/// One report from a fetch in flight.
///
/// `fraction` is what every transfer can answer. The byte counts are nil for an MLX
/// directory pull, which reports an aggregate fraction and no bytes — so a caller renders
/// bytes and a rate when they are there and a percentage when they are not, rather than
/// inventing figures for the transfers that cannot supply them.
nonisolated struct FetchProgress: Sendable {
    let fraction: Double
    let doneBytes: Int64?
    let totalBytes: Int64?
}

/// The bound `Model` for `declared`, downloading it first if the store does not hold it —
/// the crate's `ensure_model`.
///
/// Returning the *bound* model is the contract: a caller gets launchable paths, not a
/// promise that a transfer was queued.
///
/// The coordinator is passed rather than defaulted to its singleton, so a caller's
/// dependency is visible and a test can supply its own.
///
/// `familyId` / `quant` are the catalog labels a transfer is displayed under, and
/// `declaredSizeBytes` is what the quota gate refuses on before any bytes move — the
/// crate's refuse-if-oversize step, which is why it belongs on this call and not only on
/// the coordinator.
@MainActor
func ensureModel(
    _ declared: Model,
    storage: Storage,
    coordinator: DownloadCoordinator,
    familyId: String? = nil,
    quant: String? = nil,
    declaredSizeBytes: Int64? = nil,
    timeout: Duration = .seconds(1800),
    progress: ((FetchProgress) -> Void)? = nil
) async throws -> Model {
    try await storage.modelStore.ensure(declared) { declared in
        // Refused on the miss path only, so an arm that is already installed still
        // resolves: `startDownload` declines a non-Hub coordinate without registering a
        // transfer, and waiting on one would burn the whole window to learn what the
        // type already says.
        guard declared.isFetchableHere else {
            throw ModelStoreError.notFetchableHere(declared.artifactName)
        }
        coordinator.startDownload(declared, familyId: familyId, quant: quant,
                                  declaredSizeBytes: declaredSizeBytes)
        // A declined start — oversize for the quota, or bytes already on disk that the
        // store could not resolve — enqueues nothing and leaves its reason behind. There
        // is no transfer to wait on, so read the refusal now instead of at the deadline.
        guard coordinator.hasTransfer(for: declared) else {
            throw ModelStoreError.fetchFailed(
                declared.artifactName,
                reason: coordinator.errorMessage ?? "the download did not start")
        }
        try await awaitFetch(of: declared, storage: storage, coordinator: coordinator,
                             timeout: timeout, progress: progress)
    }
}

/// Wait until the store can resolve `declared`, the transfer fails or pauses, or the
/// window closes.
///
/// Polls the *store*, not a transfer row: a vision model is two transfers in one entry, so
/// "both files have landed" is exactly what `bound` answers and no transfer alone does.
///
/// A user-paused transfer is reported immediately rather than waited out — nothing will
/// advance it, so holding the caller until the window expires would just delay the same
/// answer by half an hour.
///
/// `progress` fires only when the whole percent changes, so a caller can log or render
/// every call without deduplicating: the poll is per-second but bytes move continuously,
/// and no consumer displays finer than a percent.
@MainActor
private func awaitFetch(
    of declared: Model,
    storage: Storage,
    coordinator: DownloadCoordinator,
    timeout: Duration,
    progress: ((FetchProgress) -> Void)?
) async throws {
    let deadline = ContinuousClock.now.advanced(by: timeout)
    var lastPercent = -1
    func report(_ fraction: Double) {
        let percent = Int((fraction * 100).rounded())
        guard percent != lastPercent else { return }
        lastPercent = percent
        // Read alongside the fraction rather than derived from it: the fraction is an
        // average across a model's rows, and bytes are a sum, so recovering one from the
        // other would be wrong for exactly the multi-file models that need it.
        let bytes = coordinator.transferred(for: declared)
        progress?(FetchProgress(fraction: fraction,
                                doneBytes: bytes?.done,
                                totalBytes: bytes?.total))
    }

    while ContinuousClock.now < deadline {
        if storage.modelStore.bound(declared) != nil {
            report(1)
            return
        }
        if let reason = coordinator.failureReason(for: declared) {
            throw ModelStoreError.fetchFailed(declared.artifactName, reason: reason)
        }
        if coordinator.isPaused(declared) {
            throw ModelStoreError.fetchPaused(declared.artifactName)
        }
        if let fraction = coordinator.fraction(for: declared) { report(fraction) }
        try await Task.sleep(for: .seconds(1))
    }
    throw ModelStoreError.fetchTimedOut(declared.artifactName)
}
