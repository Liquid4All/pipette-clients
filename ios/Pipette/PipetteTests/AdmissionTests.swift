import Foundation
import Testing

@testable import Pipette

/// Whole-model admission: the two inequalities, the eviction that runs between them, and
/// the reservation ledger that keeps two concurrent transfers from being promised the
/// same bytes. The scenario table is PIP-591 §8.
///
/// Free space is stated rather than read, so the quota half and the free-space half are
/// exercised independently: on a developer's disk the real reading passes every time and
/// the second inequality would never be tested at all.
@MainActor struct AdmissionTests {
    private static let epoch = Date(timeIntervalSince1970: 1_700_000_000)
    /// Comfortably past `headroomBytes`, for the tests whose subject is the quota.
    private static let ampleFreeSpace: Int64 = 64 << 30

    private func at(_ hoursLater: Double) -> Date {
        Self.epoch.addingTimeInterval(hoursLater * 3600)
    }

    private func admission(
        _ ws: PipetteWorkspace, free: Int64? = AdmissionTests.ampleFreeSpace
    ) -> ModelAdmission {
        ModelAdmission(ws: ws, freeBytes: { _ in free })
    }

    private func request(_ label: String, bytes: Int64?) throws -> AdmissionRequest {
        let spec = try ggufTextSpec("org/\(label)-GGUF", "\(label)-Q4_0.gguf")
        let key = try #require(ModelStorageKey.of(spec))
        return AdmissionRequest(key: key, label: label, declaredBytes: bytes)
    }

    private func entryNames(_ ws: PipetteWorkspace) -> Set<String> {
        Set((try? FileManager.default.contentsOfDirectory(atPath: ws.modelsDir.path)) ?? [])
    }

    // MARK: - The quota inequality

    /// An artifact bigger than the whole quota is refused without evicting anything:
    /// emptying the store cannot make room, so taking the other models would cost the
    /// user everything and still fail.
    @Test func aModelLargerThanTheQuotaIsRefusedAndEvictsNothing() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try ggufTextSpec("org/old-GGUF", "old-Q4_0.gguf"),
                         payloadBytes: 10_000, lastUsedAt: at(0))
        try ws.setStorageQuotaBytes(100_000)

        await #expect(throws: AdmissionError.self) {
            try await admission(ws).admit(
                try request("huge", bytes: 200_000), pins: SweepPins())
        }
        #expect(entryNames(ws).count == 1, "a hopeless request must not cost a model")
    }

    /// Room is made by evicting, not by refusing: the store is over the ceiling until the
    /// least-recently-used entry goes, and then the request fits.
    @Test func admissionEvictsLeastRecentlyUsedToMakeRoom() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try installEntry(ws, try ggufTextSpec("org/old-GGUF", "old-Q4_0.gguf"),
                         payloadBytes: 60_000, lastUsedAt: at(0))
        try installEntry(ws, try ggufTextSpec("org/new-GGUF", "new-Q4_0.gguf"),
                         payloadBytes: 10_000, lastUsedAt: at(5))
        try ws.setStorageQuotaBytes(100_000)

        let granted = try await admission(ws).admit(
            try request("next", bytes: 50_000), pins: SweepPins())

        #expect(granted.bytes == 50_000)
        let names = entryNames(ws)
        #expect(names.count == 1, "the least-recently-used entry should have gone")
        #expect(!names.contains(where: { $0.contains("old") }))
    }

    /// The refusal reports the pinned bytes separately: the number that tells someone
    /// finishing their job would free the space.
    @Test func aPinnedEntryIsReportedAsProtectedRatherThanEvicted() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let pinnedSpec = try ggufTextSpec("org/pinned-GGUF", "pinned-Q4_0.gguf")
        try installEntry(ws, pinnedSpec, payloadBytes: 80_000, lastUsedAt: at(0))
        try ws.setStorageQuotaBytes(100_000)
        let pins = SweepPins().withModel(pinnedSpec)

        do {
            _ = try await admission(ws).admit(try request("next", bytes: 50_000), pins: pins)
            Issue.record("a request that only fits by evicting a pinned entry must be refused")
        } catch let error as AdmissionError {
            guard case let .exceedsQuota(_, required, _, protected, quota) = error else {
                Issue.record("expected a quota refusal, got \(error)"); return
            }
            #expect(required == 50_000)
            #expect(protected >= 80_000, "the pinned payload must be reported as protected")
            #expect(quota == 100_000)
        }
        #expect(entryNames(ws).count == 1, "the pinned entry must survive")
    }

    /// Asserted as the invariant rather than a surviving-entry count: `entrySizeBytes`
    /// measures allocated blocks, so how many entries fit is the filesystem's business.
    @Test func aLoweredQuotaIsEnforcedOnTheNextAdmission() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        for (index, name) in ["a", "b", "c"].enumerated() {
            try installEntry(ws, try ggufTextSpec("org/\(name)-GGUF", "\(name)-Q4_0.gguf"),
                             payloadBytes: 30_000, lastUsedAt: at(Double(index)))
        }
        try ws.setStorageQuotaBytes(50_000)
        #expect(ws.survey().usedBytes > 50_000, "the store starts over the new quota")

        let granted = try await admission(ws).admit(
            try request("next", bytes: 20_000), pins: SweepPins())

        #expect(ws.survey().usedBytes + granted.bytes <= 50_000,
                "what is left, plus what was promised, must fit the lowered quota")
    }

    // MARK: - The free-space inequality

    /// The quota has room and the volume does not. Evicting store entries cannot fix
    /// this, so the refusal names free space rather than the quota.
    @Test func quotaRoomButNoFreeSpaceIsRefusedAsFreeSpace() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(8 << 30)
        // Room for the model, but not once the headroom is kept back.
        let free = ModelAdmission.headroomBytes + 1000

        do {
            _ = try await admission(ws, free: free).admit(
                try request("big", bytes: 5000), pins: SweepPins())
            Issue.record("a request that does not fit the volume must be refused")
        } catch let error as AdmissionError {
            guard case let .insufficientFreeSpace(_, _, reported, headroom) = error else {
                Issue.record("expected a free-space refusal, got \(error)"); return
            }
            #expect(reported == free)
            #expect(headroom == ModelAdmission.headroomBytes)
        }
    }

    /// The headroom is kept back, not merely preferred: a request that exactly consumes
    /// it is refused, and one byte less than that is admitted.
    @Test func theHeadroomIsTheBoundary() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(8 << 30)
        let free = ModelAdmission.headroomBytes + 10_000

        await #expect(throws: AdmissionError.self) {
            try await admission(ws, free: free).admit(
                try request("over", bytes: 10_001), pins: SweepPins())
        }
        _ = try await admission(ws, free: free).admit(
            try request("exact", bytes: 10_000), pins: SweepPins())
    }

    /// A volume that will not report its free space is not a reason to refuse: the quota
    /// check already passed, and `nil` means the platform did not answer rather than that
    /// the device is full.
    @Test func anUnreadableVolumeDoesNotRefuse() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)

        _ = try await admission(ws, free: nil).admit(
            try request("model", bytes: 1000), pins: SweepPins())
    }

    // MARK: - Reservations

    /// The case the whole actor exists for: two transfers that each fit on their own, but
    /// not together. Without a reservation both would survey the same empty store and
    /// both be admitted.
    @Test func twoRequestsThatEachFitButNotTogether() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let gate = admission(ws)

        _ = try await gate.admit(try request("first", bytes: 60_000), pins: SweepPins())
        await #expect(throws: AdmissionError.self) {
            try await gate.admit(try request("second", bytes: 60_000), pins: SweepPins())
        }
    }

    /// A released grant stops counting, however its transfer ended.
    @Test func releasingAGrantReturnsItsBudget() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let gate = admission(ws)

        let first = try await gate.admit(try request("first", bytes: 60_000), pins: SweepPins())
        await gate.release(first)
        #expect(await gate.outstandingBytes == 0)

        _ = try await gate.admit(try request("second", bytes: 60_000), pins: SweepPins())
    }

    /// A grant stands in full until its transfer ends, so the budget cannot be spent
    /// twice while the bytes are still moving.
    @Test func aGrantStaysOutstandingForTheWholeTransfer() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let gate = admission(ws)

        let grant = try await gate.admit(try request("first", bytes: 60_000), pins: SweepPins())
        #expect(await gate.outstandingBytes == 60_000)

        await #expect(throws: AdmissionError.self) {
            _ = try await gate.admit(try request("second", bytes: 60_000), pins: SweepPins())
        }

        await gate.release(grant)
        #expect(await gate.outstandingBytes == 0)
    }

    /// The snapshot behind a running pull is that transfer's partial, and its grant
    /// already covers it. Counting the tree again would charge one download twice and
    /// refuse a second that fits.
    @Test func aPinnedHubRepoIsNotChargedOnTopOfItsGrant() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let fm = FileManager.default
        let repoDir = ws.hubCacheDir
            .appendingPathComponent("models/meta/zephyr", isDirectory: true)
        try fm.createDirectory(at: repoDir, withIntermediateDirectories: true)
        try Data(count: 40_000).write(to: repoDir.appendingPathComponent("weights.safetensors"))

        #expect(ws.hubBytes(excluding: []) >= 40_000)
        #expect(ws.hubBytes(excluding: ["meta/zephyr"]) == 0)
    }

    /// Releasing twice, or releasing a grant `reconcile` already dropped, is not an error
    /// and does not hand the budget back twice.
    @Test func releaseIsIdempotent() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let gate = admission(ws)

        let grant = try await gate.admit(try request("first", bytes: 60_000), pins: SweepPins())
        await gate.release(grant)
        await gate.release(grant)
        #expect(await gate.outstandingBytes == 0)
    }

    // MARK: - Unknown size

    /// "Unknown size, try anyway" is exactly the transfer that overruns the quota.
    @Test func anUnsizedRequestIsRefused() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)

        do {
            _ = try await admission(ws).admit(
                try request("mystery", bytes: nil), pins: SweepPins())
            Issue.record("an unsized artifact must be refused")
        } catch let error as AdmissionError {
            guard case .unknownSize = error else {
                Issue.record("expected an unknown-size refusal, got \(error)"); return
            }
        }
    }

    // MARK: - The ledger

    /// A grant outlives the process that took it, so a relaunch mid-transfer does not
    /// promise the same bytes to a second download.
    @Test func grantsSurviveARelaunch() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)

        let first = admission(ws)
        let grant = try await first.admit(try request("first", bytes: 60_000), pins: SweepPins())

        let relaunched = admission(ws)
        #expect(await relaunched.outstandingBytes == 60_000)
        #expect(await relaunched.reservations.map(\.id) == [grant.id])
    }

    /// A transfer must not start when its reservation cannot survive a relaunch. The
    /// rejected grant is also removed from memory, so a retry is not charged for work
    /// that never began.
    @Test func failedLedgerPersistenceRefusesAndRollsBackTheGrant() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let state = ws.root.appendingPathComponent("state", isDirectory: false)
        try FileManager.default.createDirectory(
            at: ws.root, withIntermediateDirectories: true)
        try Data("not a directory".utf8).write(to: state)
        let gate = admission(ws)

        do {
            _ = try await gate.admit(
                try request("first", bytes: 60_000), pins: SweepPins())
            Issue.record("admission must fail when its reservation cannot be saved")
        } catch let error as AdmissionError {
            #expect(error == .reservationPersistenceFailed(artifact: "first"))
        }

        #expect(await gate.outstandingBytes == 0)
        #expect(await gate.reservations.isEmpty)

        try FileManager.default.removeItem(at: state)
        #expect(await admission(ws).outstandingBytes == 0)
    }

    /// Reconciliation gives back the grants of transfers that did not survive, and keeps
    /// the ones the caller says are still writing. Without the first half a killed
    /// download would shrink the quota forever.
    @Test func reconciliationDropsAbandonedGrantsAndKeepsLiveOnes() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)
        let gate = admission(ws)

        let live = try request("live", bytes: 10_000)
        let dead = try request("dead", bytes: 20_000)
        _ = try await gate.admit(live, pins: SweepPins())
        _ = try await gate.admit(dead, pins: SweepPins())
        #expect(await gate.outstandingBytes == 30_000)

        await gate.reconcile(against: [ActiveWriter(key: live.key)])

        #expect(await gate.outstandingBytes == 10_000)
        #expect(await gate.reservations.map(\.label) == ["live"])
    }

    /// Background-task discovery can finish after the relaunched process has admitted
    /// new work. Launch reconciliation owns only the ledger generation it loaded; a new
    /// grant must not be mistaken for an abandoned transfer from the previous process.
    @Test func restoredReconciliationDoesNotDropANewGrant() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1 << 30)

        let previous = admission(ws)
        let restored = try request("restored", bytes: 10_000)
        _ = try await previous.admit(restored, pins: SweepPins())

        let relaunched = admission(ws)
        let fresh = try request("fresh", bytes: 20_000)
        _ = try await relaunched.admit(fresh, pins: SweepPins())

        await relaunched.reconcileRestored(against: [ActiveWriter(key: restored.key)])

        #expect(await relaunched.outstandingBytes == 30_000)
        #expect(Set(await relaunched.reservations.map(\.label)) == ["restored", "fresh"])
    }

    /// The ledger lives outside the store root. Kept under `models/` it would be an
    /// unrecognized child, which the survey classifies as garbage, so the first
    /// admission that needed room would delete its own accounting.
    @Test func theLedgerIsNotSweptAsGarbage() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)
        let gate = admission(ws)
        _ = try await gate.admit(try request("first", bytes: 10_000), pins: SweepPins())

        ws.sweepToQuota(pinning: SweepPins())

        let relaunched = admission(ws)
        #expect(await relaunched.outstandingBytes == 10_000)
    }

    // MARK: - Bytes outside the store root

    /// The hub cache is outside the store root and `survey` does not walk it, but the
    /// bytes are real. Admission counts them, so an abandoned MLX snapshot cannot hide a
    /// quota overrun.
    @Test func theHubCacheCountsAgainstTheQuota() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(100_000)

        let stranded = ws.hubCacheDir
            .appendingPathComponent("models/org/repo", isDirectory: true)
        try FileManager.default.createDirectory(at: stranded, withIntermediateDirectories: true)
        try Data(repeating: 0x67, count: 80_000)
            .write(to: stranded.appendingPathComponent("blob.safetensors"))

        await #expect(throws: AdmissionError.self) {
            try await admission(ws).admit(
                try request("next", bytes: 50_000), pins: SweepPins())
        }
    }
}
