import Foundation
import SwiftUI

/// Owns model downloads: GGUF files on a `URLSession`, directory and vault pulls as tasks.
///
/// The session is a background one with a stable identifier, so transfers outlive the app
/// and a relaunch reattaches to them. A one-shot headless run takes a foreground session
/// instead (`sessionConfiguration(foreground:)`).
///
/// Delegate callbacks arrive on `OperationQueue.main`, so they use
/// `MainActor.assumeIsolated` without a Task hop.
@MainActor
@Observable
final class DownloadCoordinator: NSObject {
    static let shared = DownloadCoordinator(ws: PipetteWorkspace.production)

    struct Download: Identifiable {
        /// Progress lives in the byte and fraction fields; `ModelsView` derives the status
        /// text from this.
        enum State: Equatable {
            case queued                  // waiting for a concurrency slot
            case connecting
            case downloading
            case pausing
            case paused
            case cancelling              // a tombstone until the transport has stopped
            case resuming
            case failed(reason: String)
        }
        /// A single-file (GGUF) transfer, a directory pull, or a vault pull: every file of a
        /// gguf, or a directory model's archive, from the registered server, in one row.
        enum Kind: Equatable { case file; case directory(prefix: String?); case vault }

        let filename: String
        /// The resolve URL of a `.file` transfer; `nil` for directory and vault pulls,
        /// which have no single URL to resume against.
        let url: URL?
        let repo: String?
        let familyId: String?
        var bytesDownloaded: Int64
        var totalBytes: Int64?
        var state: State
        var kind: Kind = .file
        /// A directory pull reports a fraction, not byte counts.
        var explicitFraction: Double?
        /// Carried to completion so the manifest is written verbatim.
        var source: Model?
        /// The planned digest `GGUFModelInstaller` checks before installing; nil when the
        /// source named none, which verifies nothing, as upstream.
        var sha256: Sha256?
        /// The catalog's authored quant; nil for a download not started from the catalog.
        var catalogQuant: String?
        /// A connectivity drop, not a person, paused this row, so `resumeAfterReconnect`
        /// may resume it. Set only by `applyNetworkInterruption`; `resume(key:)` clears it.
        /// In-memory only: the resume blob cannot say what paused it, so a relaunch restores
        /// this as false.
        var interruptedByNetwork: Bool = false
        /// So `ensureModel` can hand the worker a classifiable refusal.
        var vaultFailure: VaultFetchError?
        /// Counted from one; a `.vault` row counts across all its files. `resume(key:)`
        /// resets it.
        var attempt: Int = 1
        var vaultEpoch: VaultEpoch?
        /// The strong ETag of the response the saved resume data holds, which a resumed
        /// transfer must answer with. Nil means the resume cannot be checked, so
        /// `performResume` starts from zero.
        ///
        /// Checked here because the Hub's CDN answers a stale `If-Range` with `206` and
        /// URLSession appends every `206`, which would join two versions of a file.
        var resumeETag: String?

        /// Per transfer, not the storage key: a vision model is two transfers in one entry.
        var key: String {
            switch kind {
            case .file: return LocalStorage.modelRelativePath(repo: repo, filename: filename)
            // Matches `HfModelRef.key`.
            case let .directory(prefix): return [repo, prefix].compactMap { $0 }.joined(separator: "/")
            // `filename` is the entry's storage key, which names every file of the model.
            case .vault: return "vault/\(filename)"
            }
        }
        var id: String { key }
        var progress: Double? {
            if let explicitFraction { return explicitFraction }
            guard let total = totalBytes, total > 0 else { return nil }
            return Double(bytesDownloaded) / Double(total)
        }

        /// The filename parse is the last resort, for a record with no `source`; it cannot
        /// name an MLX quant.
        var quant: String? {
            catalogQuant ?? source?.quant ?? LocalStorage.parseQuant(from: filename)
        }
    }

    /// Orders a `.vault` row's progress reports: by the file's start offset, then by how
    /// often that file restarted from zero. Late reports from an earlier epoch are ignored.
    struct VaultEpoch: Comparable {
        let before: Int64
        let restarts: Int

        static func < (lhs: VaultEpoch, rhs: VaultEpoch) -> Bool {
            (lhs.before, lhs.restarts) < (rhs.before, rhs.restarts)
        }
    }

    private(set) var downloads: [String: Download] = [:]
    /// Bumped after each successful completion so views can rescan.
    private(set) var completedVersion: Int = 0
    var errorMessage: String?

    @ObservationIgnored let ws: PipetteWorkspace

    @ObservationIgnored let downloadStore: DownloadStore

    @ObservationIgnored var transport: ModelTransport
    /// Sizes a fetch before admission: the crate's `ArtifactsContext.download_http_client`.
    @ObservationIgnored let downloadHttpClient: HttpClient
    @ObservationIgnored let vaultHttpClient: HttpClient
    /// The registered identity a vault fetch signs as.
    @ObservationIgnored let vaultSession: () throws -> MgmtSession
    /// In-flight directory and vault pulls, keyed as `downloads`. Cancelling the task stops
    /// the pull.
    @ObservationIgnored private var mlxTasks: [String: Task<Void, Never>] = [:]

    /// URL-session tasks selected for cancellation, retained until every matching
    /// completion callback arrives. The row and reservation stay live for the same span.
    @ObservationIgnored private var cancellingURLTaskIDs: [String: Set<Int>] = [:]

    @ObservationIgnored private var lastProgressPublishedAt: [String: Date] = [:]

    private static let progressUpdateInterval: TimeInterval = 1.0

    /// Active transfers of every kind. 2 bounds bandwidth while still overlapping one
    /// transfer with the next; the rest wait in `.queued`.
    static let maxConcurrentDownloads = 2

    /// Downloads waiting for a slot, FIFO; `pumpQueue` runs each `start`.
    @ObservationIgnored private var pendingStarts: [(key: String, start: () -> Void)] = []

    /// From `handleEventsForBackgroundURLSession`; called once the session has replayed its
    /// events and no install is still running.
    @ObservationIgnored var backgroundCompletionHandler: (() -> Void)?

    nonisolated private static let sessionIdentifier = "ai.liquid.pipette.downloads"

    /// Whether `arguments` name a headless run that completes in one go, the only case a
    /// foreground session suits. `settings run` starts the resident worker, which can be
    /// suspended between claims; a deep link carries no `headlessrun`, and its user may
    /// background the app.
    nonisolated static func usesForegroundSession(_ arguments: [String]) -> Bool {
        guard let marker = arguments.firstIndex(of: "headlessrun") else { return false }
        let verbs = arguments[arguments.index(after: marker)...]
        return !(verbs.first == "settings" && verbs.dropFirst().first == "run")
    }

    /// `nsurlsessiond` throttles a background transfer hard on a phone warm from
    /// benchmarking (measured 373 KB/s at 73°C against 23 MB/s on the same link), and a
    /// one-shot cell gains nothing from surviving suspension.
    ///
    /// Accepted cost: a foreground session does not adopt transfers a previous launch left
    /// in `nsurlsessiond`, so the cell re-downloads.
    nonisolated static func sessionConfiguration(foreground: Bool) -> URLSessionConfiguration {
        let config = foreground
            ? URLSessionConfiguration.default
            : URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
        config.allowsCellularAccess = true
        config.waitsForConnectivity = true
        config.sessionSendsLaunchEvents = !foreground
        return config
    }

    @ObservationIgnored private lazy var session: URLSession = {
        URLSession(configuration: injectedSessionConfiguration
                       ?? Self.sessionConfiguration(foreground: foregroundSession),
                   delegate: self, delegateQueue: .main)
    }()

    @ObservationIgnored private let foregroundSession: Bool

    /// Test-only: a `URLProtocol` here holds a `.file` row in a chosen state. `nil` builds
    /// the real configuration.
    @ObservationIgnored private let injectedSessionConfiguration: URLSessionConfiguration?

    @ObservationIgnored private let reachability: NetworkReachable

    @ObservationIgnored let admission: ModelAdmission

    /// One per store entry, so a model two `ensureModel` callers want is fetched and
    /// admitted once.
    @ObservationIgnored let transfers = TransferRegistry()

    /// The grant each in-flight entry is spending. It belongs to the transfer, not the
    /// waiter: a waiter that gives up leaves bytes still landing, so the grant is held until
    /// the last row for the entry retires. Keyed by entry because a vision model's two
    /// files share one grant.
    @ObservationIgnored private var reservations: [ModelStorageKey: Reservation] = [:]

    /// The jitter, wait and log of `.file` and `.vault` retries, and of `Ensure`'s vault
    /// record reads.
    @ObservationIgnored let retryHooks: RetryHooks
    /// The wait of each `.file` transfer between two attempts; a pause or cancel cancels it.
    @ObservationIgnored private var retryTasks: [String: Task<Void, Never>] = [:]

    init(
        ws: PipetteWorkspace,
        transport: ModelTransport? = nil,
        downloadHttpClient: HttpClient = .shared,
        foregroundSession: Bool = DownloadCoordinator.usesForegroundSession(CommandLine.arguments),
        reachability: NetworkReachable = NetworkReachability.shared,
        sessionConfiguration: URLSessionConfiguration? = nil,
        vaultHttpClient: HttpClient = .shared,
        vaultSession: (() throws -> MgmtSession)? = nil,
        retryHooks: RetryHooks = .live
    ) {
        self.ws = ws
        self.transport = transport ?? HubModelTransport(downloadBase: ws.hubCacheDir)
        self.downloadHttpClient = downloadHttpClient
        self.vaultHttpClient = vaultHttpClient
        self.vaultSession = vaultSession ?? { try ws.registration.mgmtSession() }
        self.foregroundSession = foregroundSession
        self.reachability = reachability
        self.injectedSessionConfiguration = sessionConfiguration
        self.retryHooks = retryHooks
        self.downloadStore = DownloadStore(dir: ws.downloadsDir)
        self.admission = ModelAdmission(ws: ws)
        super.init()
        restoreState()
        // The handler runs off the main actor; the shared instance avoids capturing `self`
        // across that boundary.
        NetworkReachability.shared.onReconnect {
            Task { @MainActor in DownloadCoordinator.shared.resumeAfterReconnect() }
        }
    }

    // MARK: - Public API

    /// The single ignition point. A vault model goes to `startVaultDownload`, a directory
    /// model to `startDirectoryDownload`; a GGUF model's files come from
    /// `ModelFetch.planDownloads`, which makes every HTTP decision up front, as the crate's
    /// planner does. A vision model's two files are separate transfers into one entry.
    ///
    /// No quota check: that is `ensureModel`'s, as in the crate, and every ignition goes
    /// through it. `jobId` is for the vault log events.
    func startDownload(_ source: Model, familyId: String? = nil, quant: String? = nil,
                       jobId: JobId? = nil) {
        if let files = ModelFetch.vaultFiles(source) {
            startVaultDownload(source, files: files, familyId: familyId, quant: quant,
                               jobId: jobId)
            return
        }
        switch source {
        case .mlx, .onnxGenai:
            guard let ref = HfModelRef.of(source) else {
                // A path arm is installed bytes, never a pull; this client has no import.
                refuseUnfetchable(source, ModelFetchError.notFetchable(
                    "local/relative directory models are imported, not downloaded"))
                return
            }
            startDirectoryDownload(source, ref: ref, familyId: familyId, quant: quant)
            return
        case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText:
            break
        }
        let planned: [ModelFetch.Download]
        do {
            planned = try planDownloads(source)
        } catch {
            refuseUnfetchable(source, error)
            return
        }
        for download in planned {
            // Unreachable in practice. `return`, not `continue`: a vision model with only
            // one file enqueued never resolves, and `ensureModel` would wait out its window.
            guard let url = URL(string: download.url.value) else {
                refuseUnfetchable(source, ModelFetchError.invalidUrl(download.url.value))
                return
            }
            enqueueFileDownload(url: url, repo: source.repo?.description, familyId: familyId,
                                source: source, quant: quant, sha256: download.sha256)
        }
    }

    /// The file parts of the crate's `parts(declared, to_stored(declared, root))`. A source
    /// with no store form has no destination, so it is refused as `notStorable`.
    private func planDownloads(_ source: Model) throws -> [ModelFetch.Download] {
        guard let into = ws.models.fetchDestination(for: source) else {
            throw ModelStoreError.notStorable(source.artifactName)
        }
        return try ModelFetch.planDownloads(source, into: into)
    }

    /// One sentence for the UI; `reason` goes only to the log.
    private func refuseUnfetchable(_ source: Model, _ reason: Error) {
        errorMessage = "This model's source can't be downloaded on iOS."
        AppLog.storage.error(
            "startDownload: \(source.artifactName) cannot be planned: \(reason.localizedDescription)")
    }

    private func enqueueFileDownload(url: URL, repo: String?, familyId: String? = nil, source: Model? = nil,
                                     quant: String? = nil, sha256: Sha256? = nil) {
        let filename = url.lastPathComponent
        let key = LocalStorage.modelRelativePath(repo: repo, filename: filename)

        if downloads[key] != nil { return }
        // Refuse rather than transfer bytes with no store entry to land in.
        guard let spec = GGUFModelInstaller.entrySpec(repo: repo, filename: filename, source: source),
              let blobs = ws.models.blobsDir(for: spec) else {
            errorMessage = "\(filename) has no model coordinate to store it under."
            return
        }
        // The installer's own path, not the leaf, so a nested file already on disk is
        // found.
        let dest = GGUFModelInstaller.destination(for: spec, leaf: filename, under: blobs,
                                                  in: ws.models)
        if FileManager.default.fileExists(atPath: dest.path) {
            errorMessage = "\(filename) already exists."
            return
        }

        downloads[key] = Download(
            filename: filename,
            url: url,
            repo: repo,
            familyId: familyId,
            bytesDownloaded: 0,
            totalBytes: nil,
            state: .queued,
            source: source,
            sha256: sha256,
            catalogQuant: quant
        )
        // On enqueue, not when a slot frees: a queue that never drains should show as the
        // started-vs-completed gap.
        Analytics.capture(AnalyticsEvents.modelDownloadStarted, [AnalyticsEvents.modelId: filename])

        gate(key: key) { [weak self] in self?.beginFileDownload(key: key) }
    }

    /// The record is first saved here, when the transfer starts, so a still-queued
    /// download leaves nothing to reconcile on relaunch.
    private func beginFileDownload(key: String) {
        guard var dl = downloads[key], let url = dl.url else { return }

        var request = URLRequest(url: url)
        attachHfAuth(&request, claimToken: dl.source?.repo?.authToken, model: dl.source)

        let task = session.downloadTask(with: request)
        task.taskDescription = key
        task.resume()

        dl.state = .connecting
        downloads[key] = dl

        saveRecord(for: dl)
    }

    private func saveRecord(for dl: Download) {
        guard let url = dl.url else { return }
        downloadStore.saveRecord(DownloadRecord(
            filename: dl.filename,
            urlString: url.absoluteString,
            repo: dl.repo,
            familyId: dl.familyId,
            source: dl.source,
            sha256: dl.sha256,
            etag: dl.resumeETag
        ))
    }

    /// Start an MLX or ONNX Runtime GenAI pull, installed by `DirectoryModelInstaller`.
    ///
    /// An in-process `Task` around `HubApi`, so the row does not survive termination. A
    /// killed pull leaves the files it has so far in `hubCacheDir`, and starting again resumes
    /// from them until the next sweep reclaims them.
    func startDirectoryDownload(_ source: Model, ref: HfModelRef,
                                familyId: String? = nil, quant: String? = nil) {
        let key = ref.key
        if downloads[key] != nil { return }
        if let blobs = ws.models.blobsDir(for: source),
           FileManager.default.fileExists(atPath: blobs.path) {
            errorMessage = "\(ref.leaf) is already downloaded."
            return
        }
        downloads[key] = Download(
            filename: ref.leaf,
            url: nil,
            repo: ref.repo.description,
            familyId: familyId,
            bytesDownloaded: 0,
            totalBytes: nil,
            state: .queued,
            kind: .directory(prefix: ref.subpath?.value),
            explicitFraction: 0,
            source: source,
            catalogQuant: quant
        )
        // Pairs with the terminal events of finishDirectory/failDirectory, so a pull never shows
        // in the funnel as a completion with no start.
        Analytics.capture(AnalyticsEvents.modelDownloadStarted, [AnalyticsEvents.modelId: ref.leaf])

        gate(key: key) { [weak self] in self?.beginDirectoryDownload(source, ref: ref, key: key) }
    }

    private func beginDirectoryDownload(_ source: Model, ref: HfModelRef, key: String) {
        guard var dl = downloads[key] else { return }
        dl.state = .connecting
        downloads[key] = dl

        let credential = resolveHfToken(claimToken: ref.repo.authToken, model: dl.source)
        // HubApi builds its own requests, so only this line can say whether the pull is
        // authenticated.
        if let credential {
            AppLog.storage.info(
                "hf auth: directory pull of \(ref.key) authenticated source=\(credential.source.rawValue)")
        } else {
            AppLog.storage.info(
                "hf auth: no token available — pulling \(ref.key) anonymously")
        }
        let transport = self.transport
        mlxTasks[key] = Task { [weak self] in
            do {
                let outcome = try await transport.download(
                    source, ref: ref, token: credential?.token,
                    admit: { planned in
                        try await self?.reconcileGrant(for: source, planned: planned)
                    }
                ) { fraction in
                    Task { @MainActor [weak self] in self?.applyMLXProgress(key: key, fraction: fraction) }
                }
                await MainActor.run { [weak self] in
                    self?.finishDirectory(key: key, outcome: outcome, source: source)
                }
            } catch is CancellationError {
                // `cancel(key:)` owns the row.
            } catch DownloadError.cancelled {
                // `cancel(key:)` owns the row.
            } catch {
                await MainActor.run { [weak self] in self?.failDirectory(key: key, error: error) }
            }
        }
    }

    private func applyMLXProgress(key: String, fraction: Double) {
        guard var dl = downloads[key] else { return }
        guard dl.state != .cancelling else { return }
        // Debounced as the GGUF path; the final frame always publishes.
        let now = Date()
        let isComplete = fraction >= 1
        if let last = lastProgressPublishedAt[key], !isComplete,
           now.timeIntervalSince(last) < Self.progressUpdateInterval {
            return
        }
        lastProgressPublishedAt[key] = now
        dl.explicitFraction = min(max(fraction, 0), 1)
        dl.state = .downloading
        downloads[key] = dl
    }

    private func finishDirectory(key: String, outcome: TransferOutcome, source: Model) {
        let downloadedDir = outcome.directory
        if downloads[key]?.state == .cancelling {
            try? FileManager.default.removeItem(at: downloadedDir)
            finishCancellation(key: key, source: source)
            return
        }
        // A cancel that raced completion already removed the row.
        guard downloads[key] != nil else {
            try? FileManager.default.removeItem(at: downloadedDir)
            mlxTasks.removeValue(forKey: key)
            return
        }
        let installer = DirectoryModelInstaller(source: source, ws: ws,
                                                resolved: outcome.resolved)
        // Detached: ONNX validation walks the whole decoder graph and the publish moves
        // gigabytes. The attempt is pinned meanwhile so the sweep leaves it alone.
        let attempt = ModelStorageKey.of(source)
        if let attempt { liveStagingAttempts.insert(attempt.value) }
        pendingInstalls += 1
        Task.detached {
            let outcome = Result { try installer.install(from: downloadedDir) }
            await MainActor.run { [weak self] in
                guard let self else { return }
                if let attempt { self.liveStagingAttempts.remove(attempt.value) }
                self.pendingInstalls -= 1
                self.completeDirectoryInstall(key: key, source: source, outcome: outcome)
            }
        }
    }

    private func completeDirectoryInstall(
        key: String, source: Model, outcome: Result<Void, Error>
    ) {
        if case let .failure(error) = outcome {
            failDirectory(key: key, error: error)
            return
        }
        // A cancel during the install: the publish already happened, so undo it.
        guard downloads[key]?.state != .cancelling else {
            if let dir = ws.models.entryDir(for: source) {
                try? FileManager.default.removeItem(at: dir)
            }
            finishCancellation(key: key, source: source)
            return
        }
        guard downloads[key] != nil else {
            if let dir = ws.models.entryDir(for: source) {
                try? FileManager.default.removeItem(at: dir)
            }
            mlxTasks.removeValue(forKey: key)
            pumpQueue()
            return
        }
        // Before the row is cleared, so the download still pins itself through `downloads`.
        sweepAfterInstall(justInstalled: ModelStorageKey.of(source))
        let filename = downloads[key]?.filename
        // Carries size_bytes as the GGUF completion does, or an average-size query silently
        // covers GGUF only.
        let totalBytes = downloads[key]?.totalBytes
        downloads.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        mlxTasks.removeValue(forKey: key)
        releaseReservation(afterRetiring: key, source: source)
        completedVersion &+= 1
        Analytics.capture(AnalyticsEvents.modelDownloadCompleted, [
            AnalyticsEvents.modelId: filename,
            AnalyticsEvents.sizeBytes: totalBytes,
        ])
        pumpQueue()
    }

    private func failDirectory(key: String, error: Error) {
        mlxTasks.removeValue(forKey: key)
        if var dl = downloads[key] {
            dl.state = .failed(reason: "Failed: \(error.localizedDescription)")
            downloads[key] = dl
        }
        errorMessage = "MLX download failed: \(error.localizedDescription)"
        releaseReservation(afterRetiring: key, source: downloads[key]?.source)
        Analytics.capture(AnalyticsEvents.modelDownloadFailed, [
            AnalyticsEvents.modelId: downloads[key]?.filename,
            AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
        ])
        pumpQueue()
    }

    /// Pull a vault gguf's files one at a time, or a vault directory model's archive, as
    /// one in-process row.
    ///
    /// Not on the background `URLSession`: each signed request carries a single-use nonce
    /// and timestamp, which a system retry or relaunch would replay stale. A retry in this
    /// process continues with `Range`; the row is not persisted and cannot pause.
    ///
    /// Every record is read before a byte moves, so a refusal costs no download and the
    /// row knows its total. `jobId` is the job that started the row.
    func startVaultDownload(_ source: Model, files: [ModelFetch.VaultFile],
                            familyId: String? = nil, quant: String? = nil,
                            jobId: JobId? = nil) {
        guard let entry = ModelStorageKey.of(source) else {
            refuseUnfetchable(source, ModelStoreError.notStorable(source.artifactName))
            return
        }
        // For a vault arm, no store form means two files on one path, which the store
        // refuses as the crate does.
        guard ws.models.fetchDestination(for: source) != nil else {
            errorMessage = "\(source.artifactName) has two files that name one vault path, "
                + "so the store cannot hold them apart."
            AppLog.vault.error(
                "vault download refused: no store form model=\(source.artifactName) "
                    + "job_id=\(jobId?.value ?? "none")")
            return
        }
        let row = Download(filename: entry.value, url: nil, repo: nil, familyId: familyId,
                           bytesDownloaded: 0, totalBytes: nil, state: .queued, kind: .vault,
                           source: source, catalogQuant: quant)
        let key = row.key
        if downloads[key] != nil { return }
        if ws.models.bound(source) != nil {
            errorMessage = "\(source.artifactName) is already downloaded."
            return
        }
        downloads[key] = row
        Analytics.capture(AnalyticsEvents.modelDownloadStarted, [AnalyticsEvents.modelId: entry.value])
        gate(key: key) { [weak self] in
            self?.beginVaultDownload(source, files: files, key: key, jobId: jobId)
        }
    }

    private func beginVaultDownload(
        _ source: Model, files: [ModelFetch.VaultFile], key: String, jobId: JobId?
    ) {
        guard var dl = downloads[key], let first = files.first else { return }
        dl.state = .connecting
        downloads[key] = dl
        let http = vaultHttpClient
        let session = vaultSession
        let ws = self.ws
        mlxTasks[key] = Task { [weak self] in
            do {
                let access = try VaultAccess.of(session, path: first.path)
                guard let hooks = self?.retryHooks else { return }
                let records = try await VaultFetch.records(
                    files, access: access, http: http, jobId: jobId, hooks: hooks)
                let total = VaultFetch.sizeBytes(records)
                // The records give the first true size; a heavier pull is admitted again,
                // as the directory path's `admit` hook does.
                do {
                    try await self?.reconcileGrant(
                        for: source,
                        planned: VaultFetch.admissionBytes(source, records: records))
                } catch {
                    throw VaultFetchError.io(dest: key, source: error.localizedDescription)
                }
                self?.applyVaultProgress(key: key, done: 0, total: total)
                switch source {
                case .mlx, .onnxGenai:
                    if let record = records.first {
                        try await self?.fetchVaultArchive(
                            record, source: source, ws: ws, access: access,
                            http: http, key: key, jobId: jobId)
                    }
                case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText:
                    var done: Int64 = 0
                    for (file, record) in zip(files, records) {
                        try await self?.fetchVaultFile(
                            file, record: record, source: source, ws: ws,
                            access: access, http: http, key: key, jobId: jobId, before: done,
                            total: total)
                        done = ModelFetch.saturating(done, Int64(clamping: record.bytes))
                    }
                }
                self?.finishVault(key: key, source: source)
            } catch {
                // `cancel(key:)` owns the row; the transport reports the cancel as a
                // failure. A file installed before the cancel goes with the entry.
                guard !Task.isCancelled else {
                    if let dir = ws.models.entryDir(for: source) {
                        try? FileManager.default.removeItem(at: dir)
                    }
                    return
                }
                let failure = error as? VaultFetchError
                    ?? .io(dest: key, source: error.localizedDescription)
                self?.failVault(key: key, error: failure, jobId: jobId)
            }
        }
    }

    /// One vault file, streamed into its own scratch directory under `.staging` and then
    /// installed by `GGUFModelInstaller`. The digest is checked while streaming, so the
    /// installer gets none.
    ///
    /// Each retry signs new requests and continues the file only if the server confirms
    /// the object is the same; otherwise the file and progress restart from zero.
    private func fetchVaultFile(
        _ file: ModelFetch.VaultFile, record: VaultResource, source: Model, ws: PipetteWorkspace,
        access: VaultAccess, http: HttpClient, key: String, jobId: JobId?, before: Int64,
        total: Int64
    ) async throws {
        let leaf = file.path.storedFilename
        let scratch = ws.models.stagingRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let attempt = scratch.lastPathComponent
        liveStagingAttempts.insert(attempt)
        defer {
            liveStagingAttempts.remove(attempt)
            try? FileManager.default.removeItem(at: scratch)
        }
        var partial: VaultPartial?
        try await retrying(
            url: file.path.value,
            attempt: { n in
                beginVaultAttempt(key: key, retry: n > 1)
                try await VaultFetch.download(
                    record, access: access, http: http, to: scratch.appendingPathComponent(leaf),
                    partial: &partial, jobId: jobId
                ) { [weak self] progress in
                    Task { @MainActor [weak self] in
                        self?.applyVaultProgress(
                            key: key, done: before + progress.written, total: total,
                            epoch: VaultEpoch(before: before, restarts: progress.restarts))
                    }
                }
            },
            classify: VaultFetch.disposition(of:), hooks: retryHooks)
        let installer = GGUFModelInstaller(repo: nil, filename: leaf, source: source,
                                           sha256: nil, ws: ws)
        do {
            try installer.finish(adopted: scratch.appendingPathComponent(leaf))
        } catch {
            throw VaultFetchError.io(dest: leaf, source: error.localizedDescription)
        }
    }

    /// A vault directory model's archive, streamed into a scratch directory under
    /// `.staging`, digest-checked while streaming, then unpacked once and published by
    /// `DirectoryModelInstaller`. Retries as `fetchVaultFile` does.
    private func fetchVaultArchive(
        _ record: VaultResource, source: Model, ws: PipetteWorkspace, access: VaultAccess,
        http: HttpClient, key: String, jobId: JobId?
    ) async throws {
        let scratch = ws.models.stagingRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        // Pins the scratch directory and the entry's staged attempt against the sweep.
        let attempts = [scratch.lastPathComponent]
            + (ModelStorageKey.of(source).map { [$0.value] } ?? [])
        liveStagingAttempts.formUnion(attempts)
        defer {
            liveStagingAttempts.subtract(attempts)
            try? FileManager.default.removeItem(at: scratch)
        }
        let archive = scratch.appendingPathComponent(".vault-archive")
        let total = Int64(clamping: record.bytes)
        var partial: VaultPartial?
        try await retrying(
            url: "/vault/resources/\(record.resourceId.value)/content",
            attempt: { n in
                beginVaultAttempt(key: key, retry: n > 1)
                try await VaultFetch.download(
                    record, access: access, http: http, to: archive, partial: &partial,
                    jobId: jobId
                ) { [weak self] progress in
                    Task { @MainActor [weak self] in
                        self?.applyVaultProgress(
                            key: key, done: progress.written, total: total,
                            epoch: VaultEpoch(before: 0, restarts: progress.restarts))
                    }
                }
            },
            classify: VaultFetch.disposition(of:), hooks: retryHooks)
        try await Self.unpackAndInstall(
            record, archive: archive,
            into: scratch.appendingPathComponent("model", isDirectory: true),
            source: source, ws: ws, jobId: jobId)
    }

    /// Off the main actor, in the caller's task so a cancel stops the unpack too.
    ///
    /// A tree that fails validation is `.unusable`, permanent because the bytes are pinned
    /// and every device unpacks the same tree. A failed publish is this device's fault and
    /// may succeed on another attempt.
    @concurrent
    private nonisolated static func unpackAndInstall(
        _ record: VaultResource, archive: URL, into tree: URL, source: Model,
        ws: PipetteWorkspace, jobId: JobId?
    ) async throws(VaultFetchError) {
        // A refused tree can hold gigabytes, so it goes here rather than in the caller's
        // main-actor `defer`. A published tree is no longer at `tree`.
        defer { try? FileManager.default.removeItem(at: tree) }
        try VaultFetch.unpack(record, archive: archive, into: tree, jobId: jobId)
        let installer = DirectoryModelInstaller(source: source, ws: ws)
        do {
            try installer.validate(tree)
        } catch {
            throw .unusable(resourceId: record.resourceId, reason: error.localizedDescription)
        }
        do {
            try installer.publish(tree)
        } catch {
            throw .io(dest: tree.path, source: error.localizedDescription)
        }
    }

    /// A retry shows the row as `.connecting` until the first new byte, keeping its
    /// progress; a restart from zero moves progress back through the epoch.
    private func beginVaultAttempt(key: String, retry: Bool) {
        guard retry, var dl = downloads[key], dl.state != .cancelling else { return }
        dl.attempt += 1
        dl.state = .connecting
        downloads[key] = dl
        lastProgressPublishedAt.removeValue(forKey: key)
    }

    /// A report from an earlier `epoch` is late and ignored; the first of a later epoch may
    /// move progress back.
    private func applyVaultProgress(key: String, done: Int64, total: Int64,
                                    epoch: VaultEpoch? = nil) {
        guard var dl = downloads[key], dl.state != .cancelling else { return }
        var rewinds = false
        if let epoch {
            if let current = dl.vaultEpoch, epoch < current { return }
            rewinds = dl.vaultEpoch.map { epoch > $0 } ?? true
            dl.vaultEpoch = epoch
        }
        // Each chunk reports in its own `Task`, in no fixed order, so within an epoch a
        // lower count is stale.
        guard rewinds || done >= dl.bytesDownloaded else { return }
        // The first frame, the final frame and a restart always publish.
        let now = Date()
        let isEdge = rewinds || done == 0 || done >= total
        if let last = lastProgressPublishedAt[key], !isEdge,
           now.timeIntervalSince(last) < Self.progressUpdateInterval {
            return
        }
        lastProgressPublishedAt[key] = now
        dl.bytesDownloaded = done
        dl.totalBytes = total
        dl.state = .downloading
        downloads[key] = dl
    }

    private func finishVault(key: String, source: Model) {
        // A cancel after the last file installed: the entry is published, so remove it.
        guard downloads[key]?.state != .cancelling else {
            if let dir = ws.models.entryDir(for: source) {
                try? FileManager.default.removeItem(at: dir)
            }
            finishCancellation(key: key, source: source)
            return
        }
        sweepAfterInstall(justInstalled: ModelStorageKey.of(source))
        let totalBytes = downloads[key]?.totalBytes
        downloads.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        mlxTasks.removeValue(forKey: key)
        releaseReservation(afterRetiring: key, source: source)
        completedVersion &+= 1
        Analytics.capture(AnalyticsEvents.modelDownloadCompleted, [
            AnalyticsEvents.modelId: ModelStorageKey.of(source)?.value,
            AnalyticsEvents.sizeBytes: totalBytes,
        ])
        pumpQueue()
    }

    /// Keeps the `.failed` row, with its typed refusal, for a person to dismiss.
    private func failVault(key: String, error: VaultFetchError, jobId: JobId?) {
        mlxTasks.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        let reason = error.localizedDescription
        if var dl = downloads[key] {
            dl.state = .failed(reason: reason)
            dl.vaultFailure = error
            downloads[key] = dl
        }
        errorMessage = reason
        AppLog.vault.error(
            "vault download failed key=\(key) job_id=\(jobId?.value ?? "none") "
                + "permanent=\(error.isPermanent): \(reason)")
        releaseReservation(afterRetiring: key, source: downloads[key]?.source)
        Analytics.capture(AnalyticsEvents.modelDownloadFailed, [
            AnalyticsEvents.modelId: downloads[key]?.filename,
            AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
        ])
        pumpQueue()
    }

    func pause(key: String) {
        // Only `.file` rows can pause; another kind would show "Paused" while its Task
        // kept running.
        guard let dl = downloads[key], case .file = dl.kind else { return }
        // A `.paused` queued row would leave its start closure in `pendingStarts`, and
        // `pumpQueue` would later resurrect it.
        guard dl.state != .queued else { return }
        if var dl = downloads[key] {
            dl.state = .pausing
            downloads[key] = dl
        }
        // A row between attempts has only the wait to stop; its resume data is already
        // kept.
        retryTasks.removeValue(forKey: key)?.cancel()
        session.getAllTasks { [weak self] tasks in
            guard let self = self else { return }
            let match = tasks.first(where: { $0.taskDescription == key }) as? URLSessionDownloadTask
            guard let match else {
                Task { @MainActor [weak self] in
                    self?.markPaused(key: key)
                }
                return
            }
            let response = match.response
            match.cancel(byProducingResumeData: { [weak self] data in
                guard let self = self else { return }
                Task { @MainActor in
                    if let data = data {
                        self.keepResumeData(data, for: key, response: response)
                    }
                    self.markPaused(key: key)
                }
            })
        }
    }

    func resume(key: String) {
        // Only from `.paused`, which dedups re-entrant taps that would spawn a duplicate
        // task.
        guard let dl = downloads[key], case .file = dl.kind,
              case .paused = dl.state, dl.url != nil else { return }

        var queuedDl = dl
        queuedDl.state = .queued
        queuedDl.interruptedByNetwork = false
        queuedDl.attempt = 1
        downloads[key] = queuedDl
        gate(key: key) { [weak self] in self?.performResume(key: key) }
    }

    /// Resume the transfers a connectivity drop paused, never those a person paused.
    func resumeAfterReconnect() {
        let keys = downloads.compactMap { key, dl -> String? in
            guard dl.interruptedByNetwork, case .paused = dl.state else { return nil }
            return key
        }
        for key in keys { resume(key: key) }
    }

    /// Resume `source`'s paused transfers, for `ensureModel`: after a relaunch nothing else
    /// clears a pause, and its cause cannot be read back from disk, so this resumes because
    /// the caller needs the weights. Scoped to `source` so it never lifts a pause nobody
    /// asked about.
    ///
    /// Offline it resumes nothing: `performResume` spends the resume blob, and
    /// `waitsForConnectivity` would park the new task, losing the only record of the
    /// partial. It first waits for a reachability reading, since the monitor answers
    /// `false` until its first one lands right after launch.
    ///
    /// Returns whether it resumed anything, which `restartStalled(for:)` turns on.
    @discardableResult
    func resumePaused(for source: Model) async -> Bool {
        let keys = pausedKeys(for: source)
        // A caller with nothing to resume is not held up for the reading.
        if !keys.isEmpty { await awaitReachabilityReading() }
        guard reachability.isConnected else {
            if !keys.isEmpty {
                AppLog.storage.warning(
                    "offline: leaving \(keys.count) paused transfer(s) for "
                    + "\(source.artifactName) alone, resume data kept for a later run")
            }
            return false
        }
        for key in keys {
            AppLog.storage.info("resuming paused transfer key=\(key) — a fetch needs these weights")
            resume(key: key)
        }
        return !keys.isEmpty
    }

    /// Retire `source`'s failed rows and start them over, for a fetch whose resume did not
    /// take: a partial the server no longer range-serves fails in seconds, and a phone
    /// driven from the CLI has nobody to tap Resume.
    ///
    /// Only `.failed` rows go, as `markFailed` already spent their record and blob. A
    /// `.paused` row still holds a partial, and a moving file is left to finish;
    /// `startDownload` dedups on those, so only what was retired re-ignites.
    ///
    /// Offline it does nothing, as `resumePaused`. Returns whether `source` has a transfer
    /// to wait on afterwards.
    @discardableResult
    func restartStalled(
        for source: Model, familyId: String? = nil, quant: String? = nil
    ) async -> Bool {
        await awaitReachabilityReading()
        guard reachability.isConnected else {
            AppLog.storage.warning(
                "offline: not restarting \(source.artifactName), "
                + "its stalled transfer(s) keep what they have for a later run")
            return false
        }
        let stalled = await retireFailed(for: source)
        AppLog.storage.info(
            "restarting \(source.artifactName) from scratch after \(stalled.count) "
            + "stalled transfer(s) — a fetch needs these weights")
        startDownload(source, familyId: familyId, quant: quant)
        return hasTransfer(for: source)
    }

    /// Retire every failed row of `source` and return their keys. Holds until no old task
    /// carries one of the keys, so the next transfer cannot meet the old one's cancel.
    func retireFailed(for source: Model) async -> [String] {
        let failed = downloads.compactMap { key, dl -> String? in
            guard dl.source == source, case .failed = dl.state else { return nil }
            return key
        }
        for key in failed { cancel(key: key) }
        await awaitTaskCancellation(of: Set(failed))
        return failed
    }

    /// Cancel any live task still carrying one of `keys`, and wait for it.
    ///
    /// `cancel(key:)` cancels through `getAllTasks` and returns before that block runs, so
    /// a fresh task under the same `taskDescription` could be matched and killed. Here the
    /// block runs while the key names only the dead transfer; the double cancel is a no-op.
    private func awaitTaskCancellation(of keys: Set<String>) async {
        await withCheckedContinuation { continuation in
            session.getAllTasks { tasks in
                for task in tasks where keys.contains(task.taskDescription ?? "") {
                    task.cancel()
                }
                continuation.resume()
            }
        }
    }

    /// Hold until the path monitor has a reading, or `timeout` passes. `NWPathMonitor`
    /// answers within milliseconds unless launch work holds its queue off; a second is
    /// invisible against a model download. No reading by then counts as offline.
    private func awaitReachabilityReading(timeout: Duration = .seconds(1)) async {
        guard !reachability.hasReading else { return }
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while !reachability.hasReading, ContinuousClock.now < deadline {
            do { try await Task.sleep(for: .milliseconds(10)) } catch { return }
        }
        if !reachability.hasReading {
            AppLog.storage.warning(
                "reachability monitor gave no reading in \(timeout); treating as offline")
        }
    }

    /// Pause every in-flight `.file` transfer and wait up to `limit` for its resume data:
    /// a foreground session's transfer dies with the process. A background session's
    /// carries on in `nsurlsessiond`.
    func keepPartialsForNextLaunch(within limit: Duration = .seconds(5)) async {
        guard foregroundSession else { return }
        let keys = downloads.compactMap { key, dl -> String? in
            guard case .file = dl.kind,
                  [.connecting, .downloading, .resuming].contains(dl.state) else { return nil }
            return key
        }
        guard !keys.isEmpty else { return }
        for key in keys { pause(key: key) }
        let deadline = ContinuousClock.now.advanced(by: limit)
        while keys.contains(where: { downloads[$0]?.state == .pausing }),
              ContinuousClock.now < deadline {
            do { try await Task.sleep(for: .milliseconds(50)) } catch { break }
        }
        let kept = keys.filter { downloadStore.hasResumeData(for: $0) }
        AppLog.storage.info(
            "paused in-flight transfers before exit count=\(keys.count) "
                + "resumable=\(kept.count) keys=\(keys.joined(separator: ","))")
    }

    /// "Resume All": user-paused and network-paused alike.
    func resumeAll() {
        let keys = downloads.compactMap { key, dl -> String? in
            if case .paused = dl.state { return key }
            return nil
        }
        for key in keys { resume(key: key) }
    }

    /// Start a resumed or retried transfer. Uses the saved resume data only when a
    /// recorded ETag can check it, else starts from zero.
    private func performResume(key: String) {
        guard var dl = downloads[key], let url = dl.url else { return }

        let task: URLSessionDownloadTask
        if dl.resumeETag != nil, let resumeData = downloadStore.loadResumeData(for: key) {
            task = session.downloadTask(withResumeData: resumeData)
            downloadStore.clearResumeData(for: key)
        } else {
            if downloadStore.hasResumeData(for: key) {
                AppLog.storage.info(
                    "resume data dropped: no ETag to check the resume against, "
                    + "starting from zero key=\(key)")
                downloadStore.clearResumeData(for: key)
            }
            if dl.resumeETag != nil {
                dl.resumeETag = nil
                saveRecord(for: dl)
            }
            dl.bytesDownloaded = 0
            var request = URLRequest(url: url)
            attachHfAuth(&request, claimToken: dl.source?.repo?.authToken, model: dl.source)
            task = session.downloadTask(with: request)
        }
        task.taskDescription = key
        task.resume()

        dl.state = .resuming
        downloads[key] = dl
    }

    /// Cancel every download, for the sign-out reset (PIP-459): a transfer left running
    /// would land a model in the `models/` tree just cleared. Keys are snapshotted because
    /// `cancel(key:)` mutates `downloads`.
    func cancelAll() {
        for key in Set(downloads.values.compactMap { $0.source.flatMap(ModelStorageKey.of) }) {
            transfers.cancel(key)
        }
        for key in Array(downloads.keys) { cancel(key: key) }
    }

    /// Stop `source`'s bytes and fail everyone waiting on them. A caller that only stops
    /// waiting cancels its own task instead (see `TransferRegistry`).
    ///
    /// The registry is told first, so later callers see the tombstone rather than starting
    /// a replacement.
    func cancelTransfers(for source: Model) {
        if let key = ModelStorageKey.of(source) { transfers.cancel(key) }
        let keys = downloads.compactMap { key, dl in dl.source == source ? key : nil }
        for key in keys { cancel(key: key) }
    }

    func cancel(key: String) {
        guard var download = downloads[key], download.state != .cancelling else { return }
        let previousState = download.state
        download.state = .cancelling
        downloads[key] = download
        lastProgressPublishedAt.removeValue(forKey: key)
        // Queued, paused and failed rows have no transport running, so retire now.
        pendingStarts.removeAll { $0.key == key }
        switch previousState {
        case .queued, .paused, .failed:
            finishCancellation(key: key, source: download.source)
            return
        case .connecting, .downloading, .pausing, .resuming, .cancelling:
            break
        }

        // Retire the tombstone only once the Task returns. `finishDirectory` handles a
        // transport that returns a payload despite the cancel.
        if let mlxTask = mlxTasks[key] {
            mlxTask.cancel()
            Task { @MainActor [weak self] in
                await mlxTask.value
                self?.finishCancellation(key: key, source: download.source)
            }
            return
        }

        // The tombstone blocks key reuse until the delegate's completion retires it.
        session.getAllTasks { [weak self] tasks in
            Task { @MainActor [weak self] in
                guard let self, self.downloads[key]?.state == .cancelling else { return }
                let matches = tasks.filter { $0.taskDescription == key }
                guard !matches.isEmpty else {
                    self.finishCancellation(key: key, source: download.source)
                    return
                }
                self.cancellingURLTaskIDs[key] = Set(matches.map(\.taskIdentifier))
                for task in matches { task.cancel() }
            }
        }
    }

    private func finishCancellation(key: String, source: Model?) {
        guard downloads[key]?.state == .cancelling else { return }
        downloads.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        mlxTasks.removeValue(forKey: key)
        retryTasks.removeValue(forKey: key)?.cancel()
        cancellingURLTaskIDs.removeValue(forKey: key)
        downloadStore.clearRecord(for: key)
        downloadStore.clearResumeData(for: key)
        releaseReservation(afterRetiring: key, source: source)

        if let entry = source.flatMap(ModelStorageKey.of) {
            let entryStillPresent = downloads.values.contains {
                $0.source.flatMap(ModelStorageKey.of) == entry
            }
            if !entryStillPresent { transfers.finishCancellation(entry) }
        }
        pumpQueue()
    }

    private func noteCancelledURLTaskFinished(key: String, taskIdentifier: Int) {
        guard downloads[key]?.state == .cancelling else { return }
        guard var taskIDs = cancellingURLTaskIDs[key] else {
            finishCancellation(key: key, source: downloads[key]?.source)
            return
        }
        taskIDs.remove(taskIdentifier)
        guard taskIDs.isEmpty else {
            cancellingURLTaskIDs[key] = taskIDs
            return
        }
        finishCancellation(key: key, source: downloads[key]?.source)
    }

    // MARK: - Storage quota

    /// Re-admit the transfer at its planned size when that exceeds the grant, which was
    /// taken against a declared size. Throws rather than trimming: a package is one model.
    func reconcileGrant(for source: Model, planned: Int64) async throws {
        guard let key = ModelStorageKey.of(source),
              let held = reservations[key], planned > held.bytes
        else { return }
        AppLog.storage.warning(
            "planned size exceeds the grant for \(held.label); re-admitting")
        let request = AdmissionRequest(key: key, label: held.label, declaredBytes: planned)
        let pins = sweepPins(justInstalled: nil)
        // Released first: the old grant and the new are the same transfer.
        reservations.removeValue(forKey: key)
        await admission.release(held)
        do {
            reservations[key] = try await admission.admit(request, pins: pins)
        } catch {
            throw ModelStoreError.admission(error)
        }
    }

    /// Called once per entry by the `ensureModel` frame that admitted it, before the rows
    /// exist; released when the last row for `key` retires.
    func holdReservation(_ reservation: Reservation, for key: ModelStorageKey) {
        reservations[key] = reservation
    }

    func hasReservation(for key: ModelStorageKey) -> Bool {
        reservations[key] != nil
    }

    /// Give back `key`'s grant once no row can still write into it. A `.failed` row lingers
    /// for a person to dismiss but writes nothing.
    func releaseReservation(for key: ModelStorageKey) {
        guard let reservation = reservations[key] else { return }
        let stillWriting = downloads.values.contains { dl in
            guard dl.source.flatMap(ModelStorageKey.of) == key else { return false }
            if case .failed = dl.state { return false }
            return true
        }
        guard !stillWriting else { return }
        reservations.removeValue(forKey: key)
        let admission = self.admission
        Task { await admission.release(reservation) }
    }

    /// The ledger is keyed by entry. With no `source` (a completion redelivered after
    /// relaunch has no row), every entry is checked.
    private func releaseReservation(afterRetiring key: String, source: Model?) {
        if let entry = source.flatMap(ModelStorageKey.of) {
            releaseReservation(for: entry)
            return
        }
        for entry in reservations.keys { releaseReservation(for: entry) }
    }

    /// Release the grants of transfers that did not survive the last run, and adopt the
    /// rest. Every restored row and every `.staging` partial counts as alive: erring that
    /// way costs a retry, the other way promises the same bytes twice.
    private func reconcileReservations() async {
        let downloadWriters = Set(downloads.values
            .compactMap { $0.source.flatMap(ModelStorageKey.of) })
        let staging = stagingWriters()
        let retained = await admission.reconcileRestored(
            against: (Array(downloadWriters) + staging).map(ActiveWriter.init))

        // A completion can retire a row during the actor hop above, so adopt the grant and
        // release it through the ordinary no-writers check.
        for reservation in retained where downloadWriters.contains(reservation.key) {
            if reservations[reservation.key] == nil {
                reservations[reservation.key] = reservation
            }
            releaseReservation(for: reservation.key)
        }
    }

    /// Entry keys with a staging attempt on disk, which still hold their grant's bytes.
    private func stagingWriters() -> [ModelStorageKey] {
        let staging = ws.modelsDir.appendingPathComponent(
            Entry.stagingDirName, isDirectory: true)
        let children = (try? FileManager.default.contentsOfDirectory(
            at: staging, includingPropertiesForKeys: nil)) ?? []
        return children.map { ModelStorageKey(value: $0.lastPathComponent) }
    }

    /// The artifact lands first, then the store is swept back under the cap, so peak disk
    /// is the quota plus the newest artifact. Intended: a reservation would have to guess the
    /// final size.
    private func sweepAfterInstall(justInstalled: ModelStorageKey?) {
        ws.sweepToQuota(pinning: sweepPins(justInstalled: justInstalled))
    }

    /// After `startDownload`, tells acceptance from a decline, which leaves its reason in
    /// `errorMessage` and enqueues nothing.
    func hasTransfer(for source: Model) -> Bool {
        downloads.values.contains { $0.source == source }
    }

    private func pausedKeys(for source: Model) -> [String] {
        downloads.compactMap { key, dl in
            guard dl.source == source, case .paused = dl.state else { return nil }
            return key
        }
    }

    /// Whether every one of `source`'s rows is paused by the network, the pause a fetch may
    /// lift. A person's Pause is reported to the waiting fetch instead of being undone.
    func isPausedByNetwork(_ source: Model) -> Bool {
        let rows = downloads.values.filter { $0.source == source }
        return !rows.isEmpty && rows.allSatisfy { $0.state == .paused && $0.interruptedByNetwork }
    }

    func isPaused(_ source: Model) -> Bool {
        let rows = downloads.values.filter { $0.source == source }
        return !rows.isEmpty && rows.allSatisfy { $0.state == .paused }
    }

    /// A vision model's two rows average, so one file finishing shows as half.
    func fraction(for source: Model) -> Double? {
        let fractions = downloads.values.filter { $0.source == source }.compactMap(\.progress)
        guard !fractions.isEmpty else { return nil }
        return fractions.reduce(0, +) / Double(fractions.count)
    }

    /// Bytes moved for `source`, summed across its rows as the CLI sums an artifact's
    /// parts. Nil for a directory pull, which reports only a fraction. `total` is nil
    /// unless every row is sized: an understated total reads as a stall near the end.
    func transferred(for source: Model) -> (done: Int64, total: Int64?)? {
        let rows = downloads.values.filter { $0.source == source && $0.explicitFraction == nil }
        guard !rows.isEmpty else { return nil }
        let total = rows.reduce(Int64?.some(0)) { running, row in
            guard let running, let rowTotal = row.totalBytes else { return nil }
            return running + rowTotal
        }
        return (rows.reduce(0) { $0 + $1.bytesDownloaded }, total)
    }

    /// The reason any of `source`'s rows failed: either row of a vision model failing
    /// means its entry will not resolve.
    func failureReason(for source: Model) -> String? {
        downloads.values.lazy
            .filter { $0.source == source }
            .compactMap { if case let .failed(reason) = $0.state { reason } else { nil } }
            .first
    }

    /// What `ensureModel` throws in place of the text, so the worker can tell a refusal
    /// every device would meet from a failure worth another claim.
    func vaultFailure(for source: Model) -> VaultFetchError? {
        downloads.values.lazy
            .filter { $0.source == source }
            .compactMap(\.vaultFailure)
            .first
    }

    /// Entries a sweep must leave alone: the one just published, every row's entry, and
    /// every model a running or paused job needs. It reads job manifests, not `JobStore`,
    /// and decodes only running and paused ones: this runs before every fetch, and a device
    /// keeps all its finished jobs.
    func sweepPins(justInstalled: ModelStorageKey?) -> SweepPins {
        var pins = SweepPins.activeJobs(ws.loadJobManifests { _, peek in
            SweepPins.activeStatuses.contains(peek.status)
        })
        if let justInstalled { pins.models.insert(justInstalled) }
        for download in downloads.values {
            guard let source = download.source else { continue }
            pins = pins.withModel(source)
            // A directory pull's hub-cache tree is its partial, not a remnant.
            if case .directory = download.kind, let repo = download.repo {
                pins.hubRepos.insert(repo)
            }
        }
        pins.stagingAttempts = liveStagingAttempts
        return pins
    }

    /// `.staging` children still being written or hashed off the main actor; the sweep
    /// reads every other one as wreckage.
    @ObservationIgnored private var liveStagingAttempts: Set<String> = []

    // MARK: - Concurrency gate

    private var activeDownloadCount: Int {
        downloads.values.reduce(into: 0) { count, dl in
            switch dl.state {
            case .connecting, .downloading, .resuming, .pausing, .cancelling: count += 1
            case .queued, .paused, .failed: break
            }
        }
    }

    private func gate(key: String, start: @escaping () -> Void) {
        if activeDownloadCount < Self.maxConcurrentDownloads {
            start()
        } else {
            pendingStarts.append((key: key, start: start))
        }
    }

    private func pumpQueue() {
        while activeDownloadCount < Self.maxConcurrentDownloads, !pendingStarts.isEmpty {
            let next = pendingStarts.removeFirst()
            guard downloads[next.key] != nil else { continue }
            next.start()
        }
    }

    // MARK: - Restoration

    private func restoreState() {
        for record in downloadStore.allRecords() {
            let key = LocalStorage.modelRelativePath(repo: record.repo, filename: record.filename)
            guard let url = URL(string: record.urlString) else {
                downloadStore.clearRecord(for: key)
                continue
            }
            let isPaused = downloadStore.hasResumeData(for: key)
            downloads[key] = Download(
                filename: record.filename,
                url: url,
                repo: record.repo,
                familyId: record.familyId,
                bytesDownloaded: 0,
                totalBytes: nil,
                state: isPaused ? .paused : .connecting,
                source: record.source,
                sha256: record.sha256,
                // Only GGUF rows persist, and their quant re-derives from `source`.
                catalogQuant: nil,
                resumeETag: record.etag
            )
        }

        // `getAllTasks` is async: a download started before its callback has no live task
        // yet, so only restored rows are reconciled.
        let restoredKeys = Set(downloads.keys)

        // Touching the lazy session reattaches to background tasks from a previous launch.
        // Until it answers, every restored row is ambiguous.
        session.getAllTasks { [weak self] tasks in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                let liveNames = Set(tasks.compactMap { $0.taskDescription })
                // A restored `.connecting` row with no live task and no resume data is
                // stale; the system redelivers any pending completion via the delegate.
                // Only `.file` rows: other kinds never appear in `liveNames`.
                for (name, dl) in self.downloads {
                    guard case .connecting = dl.state,
                          restoredKeys.contains(name), case .file = dl.kind else { continue }
                    if !liveNames.contains(name)
                        && !self.downloadStore.hasResumeData(for: name) {
                        self.downloads.removeValue(forKey: name)
                        self.downloadStore.clearRecord(for: name)
                    }
                }
                await self.reconcileReservations()
            }
        }
    }

    // MARK: - Main-actor state mutations

    fileprivate func applyProgress(key: String, written: Int64, expected: Int64) {
        guard var dl = downloads[key] else { return }
        guard dl.state != .cancelling else { return }
        // At most one update per `progressUpdateInterval`; the first and final frames
        // always publish, and a dropped frame's bytes ride on the next one.
        let now = Date()
        let isComplete = expected > 0 && written >= expected
        if let last = lastProgressPublishedAt[key], !isComplete,
           now.timeIntervalSince(last) < Self.progressUpdateInterval {
            return
        }
        lastProgressPublishedAt[key] = now

        dl.bytesDownloaded = written
        if expected > 0 { dl.totalBytes = expected }
        dl.state = .downloading
        downloads[key] = dl
    }

    /// Provenance captured before the move, so a `cancel(key:)` that clears the row cannot
    /// cost the manifest its typed `source` or orphan the moved file.
    struct CompletionProvenance {
        let repo: String?
        let filename: String
        var source: Model?
        /// See `Download.sha256`. Nil for the key-reconstruction fallback.
        var sha256: Sha256?
    }

    /// From the row, else the on-disk record (a completion redelivered after relaunch has
    /// no row), else `repo`/`filename` parsed from the key with a nil `source`.
    func captureProvenance(key: String) -> CompletionProvenance {
        if let dl = downloads[key] {
            return CompletionProvenance(repo: dl.repo, filename: dl.filename,
                                        source: dl.source, sha256: dl.sha256)
        }
        if let record = downloadStore.allRecords().first(where: {
            LocalStorage.modelRelativePath(repo: $0.repo, filename: $0.filename) == key
        }) {
            return CompletionProvenance(repo: record.repo, filename: record.filename,
                                        source: record.source, sha256: record.sha256)
        }
        let filename = (key as NSString).lastPathComponent
        let repo = filename == key ? nil : (key as NSString).deletingLastPathComponent
        return CompletionProvenance(repo: repo, filename: filename)
    }

    /// Runs after the installer has published the artifact and its manifest.
    func applyCompletion(key: String, result: Result<Void, Error>) {
        lastProgressPublishedAt.removeValue(forKey: key)
        switch result {
        case .success:
            // Read before the clears below. Provenance, not the row: a completion redelivered
            // after relaunch has no row, and those are the longest downloads.
            let provenance = captureProvenance(key: key)
            let totalBytes = downloads[key]?.totalBytes
            let source = downloads[key]?.source
            if let dl = downloads[key], dl.attempt > 1, let url = dl.url {
                retryHooks.report(.recovered(url: url.absoluteString, attempts: dl.attempt))
            }
            downloads.removeValue(forKey: key)
            releaseReservation(afterRetiring: key, source: source)
            downloadStore.clearRecord(for: key)
            downloadStore.clearResumeData(for: key)
            completedVersion &+= 1
            Analytics.capture(AnalyticsEvents.modelDownloadCompleted, [
                AnalyticsEvents.modelId: provenance.filename,
                AnalyticsEvents.sizeBytes: totalBytes,
            ])
        case .failure(let error):
            markFailed(key: key, error: error)
        }
        pumpQueue()
    }

    /// Terminal: clears the record and resume data and keeps a dismissible `.failed` row
    /// the headless runner can detect. Callers own `pumpQueue`.
    private func markFailed(key: String, error: Error) {
        let reason = humanizedDownloadError(error)
        errorMessage = reason
        // Before the clears: a redelivered failure has no row, only the record.
        let provenance = captureProvenance(key: key)
        downloadStore.clearRecord(for: key)
        downloadStore.clearResumeData(for: key)
        if var dl = downloads[key] {
            dl.state = .failed(reason: reason)
            downloads[key] = dl
        }
        releaseReservation(afterRetiring: key, source: downloads[key]?.source)
        // Error kind only: the humanized reason embeds the source URL.
        Analytics.capture(AnalyticsEvents.modelDownloadFailed, [
            AnalyticsEvents.modelId: provenance.filename,
            AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
        ])
    }

    fileprivate func applyFailure(key: String, error: Error) {
        lastProgressPublishedAt.removeValue(forKey: key)
        markFailed(key: key, error: error)
        pumpQueue()
    }

    /// Keep a `.file` transfer resumable after a network error: save resume data, pause the
    /// row, keep its record, and raise no alert.
    ///
    /// Reached when the device goes offline mid-transfer, when a retry comes due offline,
    /// or when retries run out with a partial kept. `interruptedByNetwork` is set only
    /// offline, since online no reconnect will come to resume it.
    ///
    /// Logged because the row looks like one a person paused.
    fileprivate func applyNetworkInterruption(key: String, resumeData: Data?,
                                              response: URLResponse?, error: Error,
                                              interruptedByNetwork: Bool) {
        guard downloads[key]?.kind == .file else { return }
        if let resumeData { keepResumeData(resumeData, for: key, response: response) }
        guard var dl = downloads[key] else { return }
        lastProgressPublishedAt.removeValue(forKey: key)
        dl.state = .paused
        dl.interruptedByNetwork = interruptedByNetwork
        downloads[key] = dl
        AppLog.storage.warning(
            "transfer paused by a network error, kept resumable key=\(key) "
            + "code=\((error as NSError).code) reason=\(error.localizedDescription) "
            + "resumeData=\(downloadStore.hasResumeData(for: key)) attempt=\(dl.attempt) "
            + "interruptedByNetwork=\(interruptedByNetwork)")
        pumpQueue()
    }

    // MARK: - Retries

    /// Retry a transient `.file` failure while attempts remain; on the last attempt keep
    /// the partial if there is one; otherwise fail.
    ///
    /// During the wait the row keeps its slot as `.connecting`, so a waiting `ensureModel`
    /// sees no stall; a `Retry-After` can hold it up to `RetryPolicy.retryAfterLimit`.
    /// Offline when the wait ends, the row pauses as on a network drop.
    fileprivate func retryOrFail(key: String, error: Error, disposition: Disposition,
                                 resumeData: Data?, response: URLResponse?) {
        guard let dl = downloads[key], case .file = dl.kind, let url = dl.url else {
            applyFailure(key: key, error: error)
            return
        }
        let policy = RetryPolicy.download
        let next = policy.next(after: dl.attempt, disposition, jitter: retryHooks.jitter)
        if let event = RetryEvent(after: next, url: url.absoluteString, attempt: dl.attempt,
                                   policy: policy, error: error) {
            retryHooks.report(event)
        }
        switch next {
        case .giveUp:
            // `keepResumeData` drops a partial that joins two versions of the file.
            var keptPartial = false
            if case .transient = disposition, resumeData != nil {
                keepResumeData(resumeData, for: key, response: response)
                keptPartial = downloadStore.hasResumeData(for: key)
            }
            if keptPartial {
                applyNetworkInterruption(key: key, resumeData: nil, response: response,
                                         error: error,
                                         interruptedByNetwork: !reachability.isConnected)
            } else {
                applyFailure(key: key, error: error)
            }
        case .refuseRetryAfter:
            applyFailure(key: key, error: error)
        case let .retry(wait):
            keepResumeData(resumeData, for: key, response: response)
            guard var dl = downloads[key] else { return }
            dl.attempt += 1
            dl.state = .connecting
            downloads[key] = dl
            lastProgressPublishedAt.removeValue(forKey: key)
            let sleep = retryHooks.sleep
            retryTasks[key] = Task { [weak self] in
                // Paused or cancelled meanwhile; that path settled the row.
                do { try await sleep(wait) } catch { return }
                // A retry soon after launch can come due before the path monitor reports.
                await self?.awaitReachabilityReading()
                guard let self, !Task.isCancelled else { return }
                self.retryTasks.removeValue(forKey: key)
                guard self.downloads[key]?.state == .connecting else { return }
                // Offline, a new task would spend the resume data and park in
                // `waitsForConnectivity`; pausing keeps it for the reconnect.
                guard self.reachability.isConnected else {
                    self.applyNetworkInterruption(key: key, resumeData: nil, response: nil,
                                                  error: error, interruptedByNetwork: true)
                    return
                }
                self.performResume(key: key)
            }
        }
    }

    /// Keep `data` as resume data and record the ETag of `response`, whose bytes it holds.
    /// With no strong ETag the data is kept but `performResume` will start from zero.
    ///
    /// Discards the data when it is nil, or when `response` is a `206` to a resume with a
    /// different ETag, as its bytes already join two versions of the file.
    private func keepResumeData(_ data: Data?, for key: String, response: URLResponse?) {
        guard var dl = downloads[key] else { return }
        let changed = changedResumeETag(key: key, response: response) != nil
        if let data, !changed {
            downloadStore.saveResumeData(data, for: key)
            dl.resumeETag = Self.strongEtag(of: response)
        } else {
            if data != nil {
                AppLog.storage.info(
                    "resume data dropped: the resumed response has a different ETag, the next "
                    + "attempt starts from zero key=\(key)")
            }
            downloadStore.clearResumeData(for: key)
            dl.resumeETag = nil
        }
        downloads[key] = dl
        saveRecord(for: dl)
    }

    /// Both ETags when `response` is a `206` to a resume with a different ETag. Nil for a
    /// `200`, which URLSession writes from zero.
    private func changedResumeETag(
        key: String, response: URLResponse?
    ) -> (expected: String, received: String?)? {
        guard let expected = downloads[key]?.resumeETag,
              (response as? HTTPURLResponse)?.statusCode == 206 else { return nil }
        let received = Self.strongEtag(of: response)
        return received == expected ? nil : (expected, received)
    }

    /// The file changed on the server since the partial: start from zero.
    ///
    /// `stage` is `response` (caught at the resume's first response) or `finish` (for a
    /// resume whose `didResumeAtOffset` never reached the app, such as a background
    /// transfer completed while suspended).
    ///
    /// The restart counts as an attempt with no wait; with none left the row fails. A row
    /// no longer transferring (`.pausing`) gets no new task, and its next resume starts
    /// from zero.
    private func restartFromZero(key: String, expected: String, received: String?,
                                 at stage: String) {
        guard var dl = downloads[key] else { return }
        AppLog.storage.info(
            "resumed download has a different ETag, restarting from zero key=\(key) "
            + "expected=\(expected) received=\(received ?? "none") at=\(stage)")
        downloadStore.clearResumeData(for: key)
        dl.resumeETag = nil
        dl.bytesDownloaded = 0
        downloads[key] = dl
        saveRecord(for: dl)
        guard [.connecting, .downloading, .resuming].contains(dl.state) else {
            AppLog.storage.info(
                "restart from zero deferred: the row stopped transferring key=\(key) "
                + "state=\(String(describing: dl.state))")
            return
        }
        guard dl.attempt < RetryPolicy.download.attempts else {
            AppLog.storage.warning(
                "restart from zero refused: the attempts ran out key=\(key) "
                + "attempt=\(dl.attempt)")
            applyFailure(key: key, error: DownloadError.transport(
                "the model file changed on the server, and no attempts remain"))
            return
        }
        dl.attempt += 1
        downloads[key] = dl
        performResume(key: key)
    }

    /// A weak ETag (`W/"…"`) promises the same meaning, not the same bytes, so it cannot
    /// confirm a byte range.
    nonisolated static func strongEtag(of response: URLResponse?) -> String? {
        guard let etag = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "ETag"),
              !etag.isEmpty else { return nil }
        return strongEtagValue(etag)
    }

    /// Installs still holding their bytes. The background completion handler waits for
    /// them: calling it lets the system suspend the app mid-hash, losing the download.
    @ObservationIgnored private var pendingInstalls = 0
    @ObservationIgnored private var deferredBackgroundCompletion = false

    private func beginDeferredInstall(key: String, installer: GGUFModelInstaller, adopted: URL) {
        pendingInstalls += 1
        // Pinned while the hash runs, or the sweep reads the bytes as an orphan.
        let attempt = adopted.deletingLastPathComponent().lastPathComponent
        liveStagingAttempts.insert(attempt)
        // Detached: an inherited task would hash on the main actor. `finish` being
        // `nonisolated` does not help, as a synchronous call runs on its caller.
        Task.detached {
            let outcome = Result { try installer.finish(adopted: adopted) }
            await MainActor.run { [weak self] in
                guard let self else { return }
                self.liveStagingAttempts.remove(attempt)
                // `finish` cannot stop partway, so a cancel during the hash is undone here.
                guard self.downloads[key]?.state != .cancelling else {
                    if let spec = installer.entrySpec,
                       let dir = self.ws.models.entryDir(for: spec) {
                        try? FileManager.default.removeItem(at: dir)
                    }
                    self.pendingInstalls -= 1
                    self.finishCancellation(key: key, source: installer.entrySpec)
                    if self.pendingInstalls == 0, self.deferredBackgroundCompletion {
                        self.deferredBackgroundCompletion = false
                        self.callBackgroundCompletion()
                    }
                    return
                }
                if case .success = outcome {
                    self.sweepAfterInstall(
                        justInstalled: installer.entrySpec.flatMap(ModelStorageKey.of))
                }
                self.applyCompletion(key: key, result: outcome)
                self.pendingInstalls -= 1
                if self.pendingInstalls == 0, self.deferredBackgroundCompletion {
                    self.deferredBackgroundCompletion = false
                    self.callBackgroundCompletion()
                }
            }
        }
    }

    fileprivate func callBackgroundCompletion() {
        guard pendingInstalls == 0 else {
            deferredBackgroundCompletion = true
            return
        }
        backgroundCompletionHandler?()
        backgroundCompletionHandler = nil
    }

    private func markPaused(key: String) {
        guard var dl = downloads[key] else { return }
        if dl.state == .cancelling {
            downloadStore.clearResumeData(for: key)
            return
        }
        dl.state = .paused
        downloads[key] = dl
        pumpQueue()
    }
}

// MARK: - URLSessionDownloadDelegate

extension DownloadCoordinator: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                 didWriteData bytesWritten: Int64,
                                 totalBytesWritten: Int64,
                                 totalBytesExpectedToWrite: Int64) {
        guard let key = downloadTask.taskDescription else { return }
        MainActor.assumeIsolated {
            self.applyProgress(key: key,
                               written: totalBytesWritten,
                               expected: totalBytesExpectedToWrite)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                 didResumeAtOffset fileOffset: Int64,
                                 expectedTotalBytes: Int64) {
        guard let key = downloadTask.taskDescription else { return }
        MainActor.assumeIsolated {
            // A cancel or pause owns its task; the pause's `keepResumeData` drops a changed
            // `206` itself.
            if ![.cancelling, .pausing].contains(self.downloads[key]?.state),
               let change = self.changedResumeETag(key: key, response: downloadTask.response) {
                // Its later callbacks may already be queued; with no description they stop
                // at the guard, and `pause`/`cancel` match only the new task.
                downloadTask.taskDescription = nil
                downloadTask.cancel()
                self.restartFromZero(key: key, expected: change.expected,
                                     received: change.received, at: "response")
                return
            }
            self.applyProgress(key: key,
                               written: fileOffset,
                               expected: expectedTotalBytes)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                 didFinishDownloadingTo location: URL) {
        guard let key = downloadTask.taskDescription else { return }
        // `location` is deleted when this returns, so the adopt must run synchronously.
        MainActor.assumeIsolated {
            // Cancellation owns the task until `didCompleteWithError`, which follows even a
            // success.
            if self.downloads[key]?.state == .cancelling { return }
            let provenance = self.captureProvenance(key: key)
            let result: Result<Void, Error>
            let resumeETag = self.downloads[key]?.resumeETag
            if let http = downloadTask.response as? HTTPURLResponse,
               !(200...299).contains(http.statusCode) {
                // The URL travels with the error, as in the crate's `ModelFetchError::Http`:
                // otherwise a 404 on a pinned revision and on a mistyped filename read alike.
                var info: [String: Any] = [
                    NSLocalizedDescriptionKey:
                        "The download couldn't be completed (server returned status \(http.statusCode)). Please try again.",
                ]
                if let url = downloadTask.originalRequest?.url ?? http.url {
                    info[NSURLErrorFailingURLErrorKey] = url
                }
                var disposition = Disposition(
                    status: http.statusCode, retryAfter: retryAfter(of: http))
                // A refused resume (a `416`, or a `403` from an expired signed CDN URL)
                // refuses the partial, not the file; the next attempt starts from zero.
                if resumeETag != nil, disposition == .permanent {
                    disposition = .transient(retryAfter: nil)
                }
                self.retryOrFail(
                    key: key,
                    error: NSError(domain: "Pipette", code: http.statusCode, userInfo: info),
                    disposition: disposition, resumeData: nil, response: http)
                return
            } else if let change = self.changedResumeETag(key: key,
                                                          response: downloadTask.response) {
                // URLSession appended this `206` unchecked; returning discards the temp file.
                self.restartFromZero(key: key, expected: change.expected,
                                     received: change.received, at: "finish")
                return
            } else {
                let installer = GGUFModelInstaller(
                    repo: provenance.repo, filename: provenance.filename,
                    source: provenance.source, sha256: provenance.sha256, ws: self.ws
                )
                // `adopt` is a synchronous rename; `finish` hashes for seconds, so it runs
                // off the main actor.
                do {
                    let adopted = try installer.adopt(from: location)
                    self.beginDeferredInstall(key: key, installer: installer, adopted: adopted)
                    return
                } catch {
                    result = .failure(error)
                }
            }
            self.applyCompletion(key: key, result: result)
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                 didCompleteWithError error: Error?) {
        guard let key = task.taskDescription else { return }
        let cancelling = MainActor.assumeIsolated {
            guard self.downloads[key]?.state == .cancelling else { return false }
            self.noteCancelledURLTaskFinished(key: key, taskIdentifier: task.taskIdentifier)
            return true
        }
        if cancelling { return }
        guard let error else { return }
        // A pause or cancel surfaces as `URLError.cancelled`, already handled.
        let nsErr = error as NSError
        if nsErr.domain == NSURLErrorDomain && nsErr.code == NSURLErrorCancelled {
            return
        }
        // Offline keeps the download for the reconnect; a transient online failure retries;
        // the rest are terminal.
        let resumeData = nsErr.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let offline = isOfflineError(error)
        let disposition = (error as? URLError).map { Disposition(of: $0.code) } ?? .permanent
        MainActor.assumeIsolated {
            if offline {
                self.applyNetworkInterruption(key: key, resumeData: resumeData,
                                              response: task.response, error: error,
                                              interruptedByNetwork: true)
            } else {
                self.retryOrFail(key: key, error: error, disposition: disposition,
                                 resumeData: resumeData, response: task.response)
            }
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        MainActor.assumeIsolated {
            self.callBackgroundCompletion()
        }
    }
}
