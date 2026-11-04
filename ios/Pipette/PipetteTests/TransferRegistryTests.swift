import Foundation
import Testing

@testable import Pipette

/// The two meanings of cancellation, which `docs/pipette-ios/model-store-design.md` calls
/// the most likely bug in this area. Every test here is a statement about which of the two
/// happened: a caller that stopped waiting, or a download that was cancelled.
@MainActor struct TransferRegistryTests {
    /// A transfer the test decides the end of. `run` blocks until `open()`, and a cancelled
    /// supervising task throws out of its sleep, which makes registry cancellation observable.
    private final class Gate {
        var isOpen = false
        private(set) var starts = 0
        private(set) var cancellations = 0
        private(set) var completions = 0

        /// The closure to hand `join`.
        func run() async throws {
            starts += 1
            while !isOpen {
                do {
                    try await Task.sleep(for: .milliseconds(2))
                } catch {
                    cancellations += 1
                    throw error
                }
            }
            completions += 1
        }

        func open() { isOpen = true }
    }

    private let key = ModelStorageKey(value: "org--m-MLX-4bit")

    /// Hold until `count` callers attach, so a test never asserts on a waiter before it
    /// reaches the registry. Bounded, so a regression fails rather than hangs.
    private func untilWaiters(_ count: Int, on registry: TransferRegistry) async throws {
        for _ in 0..<2000 where registry.waiterCount(for: key) != count {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(registry.waiterCount(for: key) == count)
    }

    // MARK: - One transfer, many callers

    /// Two callers wanting one model get one transfer. This is what makes the byte budget
    /// correct: `start` holds the `admit`, and running it twice would promise the same
    /// gigabytes twice.
    @Test func twoCallersShareOneTransfer() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        let first = Task { try await registry.join(key) { try await gate.run() } }
        let second = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(2, on: registry)

        gate.open()
        try await first.value
        try await second.value

        #expect(gate.starts == 1, "the second caller must join, not start a second transfer")
        #expect(gate.completions == 1)
        #expect(!registry.isRunning(key), "a settled transfer is retired")
    }

    /// A settled transfer is not reused: the next caller gets a fresh one, which is what
    /// makes a re-download after a failure possible.
    @Test func aLaterCallerStartsAFreshTransfer() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        gate.open()

        try await registry.join(key) { try await gate.run() }
        try await registry.join(key) { try await gate.run() }

        #expect(gate.starts == 2)
    }

    /// However a transfer ends, every waiter hears the same thing.
    @Test func aFailureReachesEveryWaiter() async throws {
        let registry = TransferRegistry()
        struct Boom: Error {}
        let first = Task { try await registry.join(key) { throw Boom() } }
        let second = Task { try await registry.join(key) { throw Boom() } }

        await #expect(throws: Boom.self) { try await first.value }
        await #expect(throws: Boom.self) { try await second.value }
    }

    // MARK: - Stop waiting

    /// **The distinction.** One caller cancelling its own task stops that caller waiting
    /// and nothing else: the transfer runs on and the other waiter is answered normally.
    @Test func aCallerThatStopsWaitingLeavesTheTransferRunning() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        let leaving = Task { try await registry.join(key) { try await gate.run() } }
        let staying = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(2, on: registry)

        leaving.cancel()
        await #expect(throws: CancellationError.self) { try await leaving.value }
        try await untilWaiters(1, on: registry)
        #expect(registry.isRunning(key), "the transfer is not the leaving caller's to stop")
        #expect(gate.cancellations == 0)

        gate.open()
        try await staying.value
        #expect(gate.completions == 1)
    }

    /// The last caller leaving is the same event as any other caller leaving: the bytes are
    /// already promised, and another caller may want them a second later.
    @Test func theLastCallerLeavingDoesNotStopTheTransfer() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        let only = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(1, on: registry)

        only.cancel()
        await #expect(throws: CancellationError.self) { try await only.value }
        try await untilWaiters(0, on: registry)

        #expect(registry.isRunning(key))
        #expect(gate.cancellations == 0)
        // Joining afterwards finds the transfer that is still running, rather than a
        // second one racing it for the same entry.
        let rejoining = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(1, on: registry)
        gate.open()
        try await rejoining.value
        #expect(gate.starts == 1)
    }

    /// A caller cancelled before it ever attaches is still a caller that stopped waiting,
    /// not a cancelled download. The window is real: `withTaskCancellationHandler` does not
    /// fire a handler installed after the cancellation.
    @Test func aCallerCancelledBeforeItAttachesStillLeavesTheTransferAlone() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        let running = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(1, on: registry)

        let late = Task { try await registry.join(key) { try await gate.run() } }
        late.cancel()
        await #expect(throws: CancellationError.self) { try await late.value }

        #expect(registry.isRunning(key))
        gate.open()
        try await running.value
        #expect(gate.starts == 1)
    }

    // MARK: - Cancel the download

    /// The other meaning: the transfer stops, and every waiter fails with it.
    @Test func cancellingTheDownloadFailsEveryWaiter() async throws {
        let registry = TransferRegistry()
        let gate = Gate()
        let first = Task { try await registry.join(key) { try await gate.run() } }
        let second = Task { try await registry.join(key) { try await gate.run() } }
        try await untilWaiters(2, on: registry)

        registry.cancel(key)

        await #expect(throws: TransferCancelled(key: key)) { try await first.value }
        await #expect(throws: TransferCancelled(key: key)) { try await second.value }
        #expect(registry.isRunning(key), "the key stays occupied until transport termination")
        for _ in 0..<2000 where gate.cancellations == 0 {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(gate.cancellations == 1)
        #expect(gate.completions == 0)
        registry.finishCancellation(key)
        #expect(!registry.isRunning(key))
    }

    /// Cancelling a key nothing is transferring is not an error: a person can tap Cancel on
    /// a row whose transfer has just finished.
    @Test func cancellingAnIdleKeyDoesNothing() {
        TransferRegistry().cancel(key)
    }

    /// A replacement cannot reuse the key until the cancelled transport terminates. Once
    /// the coordinator retires the tombstone, the next caller starts normally.
    @Test func aReplacementWaitsForCancellationRetirement() async throws {
        let registry = TransferRegistry()
        let cancelled = Gate()
        let fresh = Gate()
        let first = Task { try await registry.join(key) { try await cancelled.run() } }
        try await untilWaiters(1, on: registry)
        registry.cancel(key)
        await #expect(throws: TransferCancelled(key: key)) { try await first.value }

        await #expect(throws: TransferCancelled(key: key)) {
            try await registry.join(key) { try await fresh.run() }
        }
        #expect(fresh.starts == 0)

        for _ in 0..<2000 where cancelled.cancellations == 0 {
            try await Task.sleep(for: .milliseconds(1))
        }
        registry.finishCancellation(key)

        fresh.open()
        try await registry.join(key) { try await fresh.run() }

        #expect(fresh.completions == 1)
    }
}
