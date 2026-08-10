import Foundation

actor BenchmarkSyncCoordinator {
    static let shared = BenchmarkSyncCoordinator()

    struct Key: Hashable {
        let serverURL: String
        let storageRoot: String
    }

    typealias SyncRunner = @Sendable (ServerURL, BenchmarkStore) async throws -> Int

    private struct InFlight {
        let id: Int
        let task: Task<Int, Error>
    }

    private struct StorageTail {
        let id: Int
        let task: Task<Void, Never>
    }

    private var nextTaskId = 0
    private var inFlight: [Key: InFlight] = [:]
    private var storageTails: [String: StorageTail] = [:]
    private let syncRunner: SyncRunner

    init(syncRunner: @escaping SyncRunner = { serverUrl, store in
        try await BenchmarkSync.sync(serverUrl: serverUrl, store: store)
    }) {
        self.syncRunner = syncRunner
    }

    /// Pull the catalog opportunistically, on a UI lifecycle event — a tab appearing, a
    /// login completing.
    ///
    /// **Not on a headless run.** The crate pulls the catalog from one place,
    /// `commands::sync`, so on the CLI a sync is something you ask for and never something
    /// that happens to you. A headless launch still builds the SwiftUI scene, so without
    /// this guard `MainTabView.onAppear` fired a network fetch and a disk write partway
    /// through a measured benchmark — at a moment decided by when the response landed.
    /// `headlessrun sync` remains the way to ask, and goes through `sync` below.
    nonisolated static func syncBestEffortInBackground(
        serverUrl: ServerURL,
        storage: Storage,
        reason: String,
        clearCacheFirst: Bool = false
    ) {
        guard !HeadlessRunner.isHeadless else { return }
        Task.detached {
            await shared.syncBestEffort(
                serverUrl: serverUrl,
                storage: storage,
                reason: reason,
                clearCacheFirst: clearCacheFirst
            )
        }
    }

    @discardableResult
    func sync(serverUrl: ServerURL, storage: Storage) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: storage.benchmarks,
            key: Self.key(serverUrl: serverUrl, storage: storage)
        )
    }

    @discardableResult
    func syncAfterClearingCache(serverUrl: ServerURL, storage: Storage) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: storage.benchmarks,
            key: Self.key(serverUrl: serverUrl, storage: storage),
            clearCache: { storage.benchmarks.clearRemote() },
            coalesceExisting: false
        )
    }

    @discardableResult
    func sync(serverUrl: ServerURL, store: BenchmarkStore, key: Key) async throws -> Int {
        try await sync(
            serverUrl: serverUrl,
            store: store,
            key: key,
            clearCache: nil,
            coalesceExisting: true
        )
    }

    @discardableResult
    private func sync(
        serverUrl: ServerURL,
        store: BenchmarkStore,
        key: Key,
        clearCache: (@Sendable () -> Void)?,
        coalesceExisting: Bool
    ) async throws -> Int {
        if coalesceExisting, let existing = inFlight[key] {
            return try await existing.task.value
        }

        let runner = syncRunner
        let priorStorageTail = storageTails[key.storageRoot]?.task
        let taskId = makeTaskId()
        let task = Task.detached {
            await priorStorageTail?.value
            clearCache?()
            return try await runner(serverUrl, store)
        }
        let storageTail = Task {
            _ = try? await task.value
        }
        inFlight[key] = InFlight(id: taskId, task: task)
        storageTails[key.storageRoot] = StorageTail(id: taskId, task: storageTail)

        do {
            let count = try await task.value
            finish(key: key, taskId: taskId)
            return count
        } catch {
            finish(key: key, taskId: taskId)
            throw error
        }
    }

    func syncBestEffort(
        serverUrl: ServerURL,
        storage: Storage,
        reason: String,
        clearCacheFirst: Bool = false
    ) async {
        do {
            if clearCacheFirst {
                _ = try await syncAfterClearingCache(serverUrl: serverUrl, storage: storage)
            } else {
                _ = try await sync(serverUrl: serverUrl, storage: storage)
            }
        } catch {
            AppLog.benchmarkSync.error("\(reason) sync failed: \(error)")
        }
    }

    private static func key(serverUrl: ServerURL, storage: Storage) -> Key {
        Key(
            serverURL: serverUrl.value,
            storageRoot: storage.dataRoot.standardizedFileURL.path
        )
    }

    private func makeTaskId() -> Int {
        nextTaskId += 1
        return nextTaskId
    }

    private func finish(key: Key, taskId: Int) {
        if inFlight[key]?.id == taskId {
            inFlight[key] = nil
        }
        if storageTails[key.storageRoot]?.id == taskId {
            storageTails[key.storageRoot] = nil
        }
    }
}
