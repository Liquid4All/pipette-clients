import Foundation
import Testing

@testable import Pipette

/// Regression coverage for the move/cancel TOCTOU (PIP-259): `captureProvenance`
/// reads `downloads[key]` *before* the move, so a `cancel(key:)` that races the
/// move (clearing the entry) can't strip the repo/source the installer needs to
/// record the moved file — which would orphan it from the next scan. (The record
/// half — writing the manifest from that provenance — is `GGUFModelInstallerTests`.)
///
/// Each test builds its own `DownloadCoordinator` on a temporary `FileStorage`.
/// `.serialized`: the file-download path lazily creates a background `URLSession`
/// keyed by a process-global identifier, so these must not race each other.
@Suite(.serialized) @MainActor struct DownloadCoordinatorProvenanceTests {
    private static let repo = "LiquidAI/LFM2.5-350M-GGUF"
    private static let filename = "LFM2.5-350M-Q4_0.gguf"
    private static let familyId = "lfm2.5-350m"

    /// Tier 1: with the in-memory `downloads[key]` entry present,
    /// `captureProvenance` returns its values directly — including the typed
    /// `source`, which key reconstruction (tier 3) can't recover.
    @Test func captureProvenanceReadsInMemoryEntryWhenPresent() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
        #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

        // Install a genuine in-flight entry carrying the typed source.
        let source = Model.ggufText(.init(source: .huggingFace(repo: try HFRepo.parse(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
        coordinator.startDownload(source, familyId: Self.familyId)
        defer { coordinator.cancel(key: key) }
        try #require(coordinator.downloads[key] != nil, "startDownload should populate the in-flight entry")

        let provenance = coordinator.captureProvenance(key: key)
        #expect(provenance.repo == Self.repo)
        #expect(provenance.filename == Self.filename)
        // The in-memory entry carries the typed source; key reconstruction would not.
        #expect(provenance.source == source)
    }

    /// Tier 3: no in-flight entry and no on-disk record → reconstruct
    /// `repo`/`filename` from the key, with a nil `source`.
    @Test func captureProvenanceReconstructsFromKeyWhenNoRecord() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
        #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

        let provenance = coordinator.captureProvenance(key: key)
        #expect(provenance.repo == Self.repo)
        #expect(provenance.filename == Self.filename)
        #expect(provenance.source == nil)
    }

    /// Igniting from a `Model` carries the typed definition into the in-flight
    /// entry, so `captureProvenance` recovers it (exact provenance, not strings).
    @Test func sourceDrivenIgnitionCarriesTheSource() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
        #expect(coordinator.downloads[key] == nil, "test process should start with no in-flight download")

        let source = Model.ggufText(.init(source: .huggingFace(repo: try HFRepo.parse(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
        coordinator.startDownload(source, familyId: Self.familyId)
        defer { coordinator.cancel(key: key) }

        let provenance = coordinator.captureProvenance(key: key)
        #expect(provenance.source == source)
        #expect(provenance.repo == Self.repo)
    }

    /// A sideload key (no repo bucket) reconstructs with a nil repo.
    @Test func captureProvenanceReconstructsSideloadKeyWithNilRepo() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: nil, filename: Self.filename)
        let provenance = coordinator.captureProvenance(key: key)
        #expect(provenance.repo == nil)
        #expect(provenance.filename == Self.filename)
    }

    /// The GGUF completion state transition: a successful install clears the
    /// in-flight entry and bumps `completedVersion` so views re-scan. (The relocate
    /// + manifest record itself is `GGUFModelInstallerTests`.)
    @Test func applyCompletionSuccessClearsEntryAndBumpsVersion() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
        let source = Model.ggufText(.init(source: .huggingFace(repo: try HFRepo.parse(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
        coordinator.startDownload(source, familyId: Self.familyId)
        try #require(coordinator.downloads[key] != nil)
        let versionBefore = coordinator.completedVersion

        coordinator.applyCompletion(key: key, result: .success(()))

        #expect(coordinator.downloads[key] == nil)
        #expect(coordinator.completedVersion == versionBefore + 1)
    }

    /// A failed completion keeps a dismissible `.failed` row rather than vanishing,
    /// matching the MLX path — visible, cancellable, detectable by the headless runner.
    @Test func applyCompletionFailureMarksRowFailed() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        let coordinator = DownloadCoordinator(storage: storage)
        let key = LocalStorage.modelRelativePath(repo: Self.repo, filename: Self.filename)
        let source = Model.ggufText(.init(source: .huggingFace(repo: try HFRepo.parse(Self.repo), path: try RepoSubpath(Self.filename), sha256: nil)))
        coordinator.startDownload(source, familyId: Self.familyId)
        defer { coordinator.cancel(key: key) }

        coordinator.applyCompletion(key: key, result: .failure(URLError(.timedOut)))

        let isFailed: Bool = if case .failed = coordinator.downloads[key]?.state { true } else { false }
        #expect(isFailed)
    }
}
