import Foundation
import Testing

@testable import Pipette

/// `ensureModel` — the find-or-fetch entry point every caller now goes through, so the
/// rules that decide *whether* to transfer are asserted here rather than at four call
/// sites.
///
/// Each case that expects a refusal passes a short `timeout`: the point is *which* error
/// comes back, and a regression that waits for the deadline should fail the test rather
/// than stall the suite for the full window.
///
/// `.serialized`: these share a temporary `dataRoot` and the coordinator's process-global
/// background session, so they must not race each other.
@Suite(.serialized) @MainActor struct EnsureModelTests {

    /// A hit is answered from the store: no transfer is registered, which is also what
    /// makes re-tapping Download on installed weights a no-op.
    @Test func aHitEnqueuesNoTransfer() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(storage, spec)
        let coordinator = DownloadCoordinator(storage: storage)

        let bound = try await ensureModel(spec, storage: storage, coordinator: coordinator,
                                          timeout: .seconds(2))

        #expect(!coordinator.hasTransfer(for: spec))
        #expect(bound.boundPaths?.payload == storage.modelStore.payloadPath(for: spec))
    }

    /// A coordinate this client cannot fetch is refused by name before anything is
    /// awaited. `startDownload` would decline it without registering a transfer, so
    /// waiting would only reach the same answer at the deadline.
    @Test func anUnfetchableArmIsRefusedByName() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let url = try ResourceUrl("https://example.com/weights.gguf")
        let spec = Model.ggufText(GgufText(source: .url(url: url, sha256: nil)))
        let coordinator = DownloadCoordinator(storage: storage)

        await #expect(throws: ModelStoreError.notFetchableHere(spec.artifactName)) {
            _ = try await ensureModel(spec, storage: storage, coordinator: coordinator,
                                      timeout: .seconds(2))
        }
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// An artifact larger than the whole quota is refused before the fetch — the crate's
    /// first `ensure_model` step. The coordinator declines it silently, so `ensureModel`
    /// has to surface that rather than wait on a transfer that was never created.
    @Test func anOversizeArtifactIsRefusedWithTheCoordinatorsReason() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.setStorageQuotaBytes(1_000_000)
        let spec = try ggufTextSpec("org/big-GGUF", "big-Q4_0.gguf")
        let coordinator = DownloadCoordinator(storage: storage)

        let failure = await #expect(throws: ModelStoreError.self) {
            _ = try await ensureModel(spec, storage: storage, coordinator: coordinator,
                                      declaredSizeBytes: 9_000_000, timeout: .seconds(2))
        }
        guard case let .fetchFailed(_, reason) = try #require(failure) else {
            Issue.record("expected a fetchFailed carrying the refusal"); return
        }
        #expect(reason.contains(ByteFormat.fileSize(9_000_000)))
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// Apple Foundation ships with the OS and has no store entry, so it is refused as
    /// not-storable rather than treated as something to download.
    @Test func appleFoundationIsNotEnsurable() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coordinator = DownloadCoordinator(storage: storage)

        await #expect(throws: ModelStoreError.notStorable(Model.appleFoundationText.artifactName)) {
            _ = try await ensureModel(.appleFoundationText, storage: storage,
                                      coordinator: coordinator, timeout: .seconds(2))
        }
    }
}
