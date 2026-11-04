import Foundation

// One transfer per model, however many callers want it, keeping the two meanings of
// cancellation apart (see `docs/pipette-ios/model-store-design.md`):
//
// - Stop waiting: a caller's own `Task` is cancelled. The transfer keeps running for the
//   other waiters, since another job may want the same weights a second later.
// - Cancel the download: `cancel(_:)`. The transfer stops and every waiter fails.
//
// Conflating them let a departing waiter release a byte budget the transfer was still
// spending. `@MainActor` rather than an actor: everything `join` serializes is already
// main-actor bound, and a second actor would add re-entrancy between steps.

nonisolated struct TransferCancelled: Error, Equatable {
    let key: ModelStorageKey
}

@MainActor
final class TransferRegistry {
    /// A class, so a waiter keeps referring to its transfer after `inFlight` moves on.
    private final class Transfer {
        /// Owned here rather than by any caller: that is the distinction this type draws.
        var task: Task<Void, Never>?
        var waiters: [UUID: CheckedContinuation<Void, any Error>] = [:]
        /// Waiters have failed but the transport has not confirmed it stopped, so the key
        /// stays occupied.
        var isCancelling = false
    }

    private var inFlight: [ModelStorageKey: Transfer] = [:]

    /// `start` runs exactly once per transfer, in a task this registry owns, so work
    /// done in it is correct however many callers join or leave. The first caller's
    /// `start` sets the stall window and progress sink; later callers see no progress
    /// of their own.
    func join(_ key: ModelStorageKey,
              start: @escaping @MainActor () async throws -> Void) async throws {
        // Before the lookup: `begin` admits a budget and starts a download, which a
        // caller already gone would never consume.
        try Task.checkCancellation()
        let transfer = inFlight[key] ?? begin(key, start)
        if transfer.isCancelling { throw TransferCancelled(key: key) }
        let waiter = UUID()
        try await withTaskCancellationHandler {
            let _: Void = try await withCheckedThrowingContinuation { continuation in
                // Cancelled before the handler was installed: it will not fire, and
                // nothing else would resume this.
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    transfer.waiters[waiter] = continuation
                }
            }
        } onCancel: {
            // The transfer is deliberately untouched.
            Task { @MainActor [weak self] in self?.stopWaiting(waiter, on: key) }
        }
    }

    /// The entry stays as a tombstone until the coordinator confirms the transport
    /// stopped, so a caller arriving meanwhile gets the same cancellation instead of
    /// starting a replacement the old callback could stop.
    func cancel(_ key: ModelStorageKey) {
        guard let transfer = inFlight[key], !transfer.isCancelling else { return }
        AppLog.storage.info("cancelling the transfer for \(key.value): "
            + "\(transfer.waiters.count) waiter(s) fail with it")
        transfer.isCancelling = true
        transfer.task?.cancel()
        answerWaiters(transfer, with: .failure(TransferCancelled(key: key)))
    }

    func finishCancellation(_ key: ModelStorageKey) {
        guard let transfer = inFlight[key], transfer.isCancelling else { return }
        inFlight[key] = nil
    }

    /// For tests, where the two cancellations are observable.
    func waiterCount(for key: ModelStorageKey) -> Int {
        inFlight[key]?.waiters.count ?? 0
    }

    /// Includes a transfer awaiting confirmed cancellation.
    func isRunning(_ key: ModelStorageKey) -> Bool {
        inFlight[key] != nil
    }

    // MARK: - Internals

    private func begin(_ key: ModelStorageKey,
                       _ start: @escaping @MainActor () async throws -> Void) -> Transfer {
        let transfer = Transfer()
        inFlight[key] = transfer
        // Unstructured: a child task would inherit the first caller's cancellation.
        transfer.task = Task { @MainActor [weak self] in
            let result: Result<Void, any Error>
            do {
                try await start()
                result = .success(())
            } catch {
                result = .failure(error)
            }
            self?.settle(key, transfer, with: result)
        }
        return transfer
    }

    /// Retires and answers in one main-actor step, so a `join` can never attach to a
    /// transfer that has already answered.
    private func settle(_ key: ModelStorageKey, _ transfer: Transfer,
                        with result: Result<Void, any Error>) {
        // `start` can return before the URLSession or Hub task stops; the coordinator
        // retires the tombstone through `finishCancellation` once it has.
        guard !transfer.isCancelling else { return }
        if inFlight[key] === transfer { inFlight[key] = nil }
        answerWaiters(transfer, with: result)
    }

    private func answerWaiters(_ transfer: Transfer, with result: Result<Void, any Error>) {
        let waiters = transfer.waiters
        transfer.waiters = [:]
        for continuation in waiters.values { continuation.resume(with: result) }
    }

    /// Idempotent, and harmless after the transfer settled or a later one took the key:
    /// the waiter's `UUID` is unique.
    private func stopWaiting(_ waiter: UUID, on key: ModelStorageKey) {
        guard let continuation = inFlight[key]?.waiters.removeValue(forKey: waiter) else {
            return
        }
        continuation.resume(throwing: CancellationError())
    }
}
