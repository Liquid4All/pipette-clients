import Foundation
import Testing

@testable import Pipette

/// `last_used_at` — the eviction order, refreshed best-effort whenever a model is
/// resolved for a run. Each test injects its own temporary `FileStorage`, so the suite
/// carries no shared global and runs in parallel.
struct ModelLastUsedTouchTests {
    private static let longAgo = Date(timeIntervalSince1970: 1_600_000_000)

    @Test func touchAdvancesLastUsedAndLeavesProvenanceAlone() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let fetchedAt = Self.longAgo
        let entry = try installEntry(storage, spec, lastUsedAt: Self.longAgo, fetchedAt: fetchedAt)

        storage.touchModelLastUsed(spec)

        let manifest = try #require(ModelManifest.forInstalledEntry(atDir: entry))
        #expect(manifest.lastUsedAt > Self.longAgo)
        #expect(manifest.declared == spec)
        #expect(manifest.fetchedAt == fetchedAt)
    }

    @Test func touchingAModelWithNoEntryIsANoOp() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/absent-GGUF", "absent-Q4_0.gguf")

        storage.touchModelLastUsed(spec)

        let entry = try #require(storage.modelStore.entryDir(for: spec))
        #expect(!FileManager.default.fileExists(atPath: entry.path))
    }

    /// An unreadable manifest is garbage the sweeper owns; a touch must never
    /// resurrect it by fabricating one.
    @Test func touchingAnUnreadableManifestCreatesNothing() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try #require(storage.modelStore.prepareEntryDir(for: spec))
        try Data("not json".utf8).write(to: ModelManifest.manifestURL(inEntryDir: entry))

        storage.touchModelLastUsed(spec)

        #expect(ModelManifest.forInstalledEntry(atDir: entry) == nil)
        let raw = try String(contentsOf: ModelManifest.manifestURL(inEntryDir: entry), encoding: .utf8)
        #expect(raw == "not json")
    }

    /// A write the filesystem refuses must not propagate: the model still resolves,
    /// it just keeps its old position in the eviction order.
    @Test func aFailedTouchWriteIsSwallowed() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let spec = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        let entry = try installEntry(storage, spec, lastUsedAt: Self.longAgo)
        let fm = FileManager.default
        try fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: entry.path)
        defer { try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: entry.path) }

        storage.touchModelLastUsed(spec)

        let manifest = try #require(ModelManifest.forInstalledEntry(atDir: entry))
        #expect(manifest.lastUsedAt == Self.longAgo)
    }
}
