import Foundation
import SwiftUI

/// The credential a HuggingFace fetch for `model` should present, most specific first:
/// the `auth_token` the plan shipped on the model coordinate, then the one stored for this
/// model when its claim arrived, then the user's own.
///
/// A `claimToken` wins over the Keychain. The CLI injects the plan's token into the fetch,
/// and a claim that carries a credential is naming the one that can read *that* repo;
/// falling back to whatever the device happens to have stored answers 401 for a gated repo
/// the plan could have opened. The middle tier covers a transfer whose in-memory claim is
/// gone (a relaunch). The last tier is the user's own token, which nothing writes yet — a
/// Settings field is the missing half, so today it is always absent on device.
///
/// Both transports resolve through here, so a repo that opens for one format can't 401 for
/// the other. Why iOS stores a token at all is in `docs/pipette-ios/execution-alignment.md`.
func resolveHfToken(claimToken: AuthToken?, model: Model?) -> HfCredential? {
    if let claimToken { return HfCredential(token: claimToken, source: .claim) }
    if let stored = model.flatMap({ KeychainHelper.loadHfToken(forModel: $0) }) {
        return HfCredential(token: stored, source: .stored)
    }
    return KeychainHelper.loadHfToken().map { HfCredential(token: $0, source: .user) }
}

/// A resolved credential and the tier that answered.
///
/// The tier is carried so the log line can name it. `source=claim` is the one that
/// matters operationally: it is the difference between a plan's token reaching the fetch
/// and the device quietly falling back to something else — the failure that made gated
/// MLX pulls 401 and public ones anonymous, and which no log said a word about.
struct HfCredential {
    let token: AuthToken
    let source: HfTokenSource
}

/// Where a credential came from, most specific first. `rawValue` is the log spelling.
enum HfTokenSource: String {
    /// The `auth_token` on the model coordinate this run was dispatched with.
    case claim
    /// Stored for this model when its claim arrived — a transfer that outlived its claim.
    case stored
    /// The device's own token, iOS's analogue of `PIPETTE_HF_TOKEN`.
    case user
}

/// Attach `Authorization: Bearer …` for huggingface.co requests. Other hosts are left
/// alone so the token isn't leaked to third parties.
///
/// Says which it did, on the record: an anonymous fetch of a public repo is
/// indistinguishable from an authenticated one until HF starts answering 429, and that is
/// exactly when someone needs to know which of the two this was. The token itself is never
/// logged — `AuthToken` renders `<redacted>`, and only the tier's name is interpolated.
func attachHfAuth(_ request: inout URLRequest, claimToken: AuthToken? = nil, model: Model? = nil) {
    guard let host = request.url?.host, host.hasSuffix("huggingface.co") else { return }
    let path = request.url?.path ?? "?"
    guard let credential = resolveHfToken(claimToken: claimToken, model: model) else {
        AppLog.storage.info("hf auth: no token available — requesting \(host)\(path) anonymously")
        return
    }
    request.setValue("Bearer \(credential.token.value)", forHTTPHeaderField: "Authorization")
    AppLog.storage.info(
        "hf auth: attached bearer token source=\(credential.source.rawValue) to \(host)\(path)")
}

/// Disk-backed store for download metadata and resume-data blobs.
///
/// Lives in `~/Library/Caches/Pipette/downloads/` — per iOS conventions,
/// re-downloadable content belongs in Caches. The system may evict files
/// here under storage pressure; worst case the user restarts a download.
enum DownloadStore {
    private static var dir: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let url = caches.appendingPathComponent("Pipette/downloads", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Download keys are `<repo>/<filename>`, which contain `/`.
    /// Percent-encode to a single, collision-free filename — an
    /// injective mapping, unlike a `/`→`__` substitution which two distinct
    /// keys can alias onto.
    private static func safeName(_ key: String) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "._-")
        return key.addingPercentEncoding(withAllowedCharacters: allowed)
            ?? key.replacingOccurrences(of: "/", with: "%2F")
    }

    private static func recordPath(for key: String) -> URL {
        dir.appendingPathComponent(safeName(key) + ".record")
    }

    private static func resumePath(for key: String) -> URL {
        dir.appendingPathComponent(safeName(key) + ".resumeData")
    }

    static func saveRecord(_ record: DownloadRecord) {
        guard let data = try? PropertyListEncoder().encode(record) else { return }
        let key = LocalStorage.modelRelativePath(repo: record.repo, filename: record.filename)
        try? data.write(to: recordPath(for: key))
    }

    static func clearRecord(for key: String) {
        try? FileManager.default.removeItem(at: recordPath(for: key))
    }

    static func allRecords() -> [DownloadRecord] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "record" }.compactMap { url in
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? PropertyListDecoder().decode(DownloadRecord.self, from: data)
        }
    }

    static func saveResumeData(_ data: Data, for key: String) {
        try? data.write(to: resumePath(for: key))
    }

    static func loadResumeData(for key: String) -> Data? {
        try? Data(contentsOf: resumePath(for: key))
    }

    static func hasResumeData(for key: String) -> Bool {
        FileManager.default.fileExists(atPath: resumePath(for: key).path)
    }

    static func clearResumeData(for key: String) {
        try? FileManager.default.removeItem(at: resumePath(for: key))
    }
}

/// Singleton that owns the `URLSession` used for model downloads.
///
/// The session is a background one, so downloads keep running when the app is
/// backgrounded or terminated; the system relaunches the app to deliver
/// completion events via the session identifier. The identifier is stable across
/// launches so reconnecting to an in-flight task is automatic —
/// `URLSession(configuration:delegate:queue:)` with a matching background config
/// attaches to the existing session. A one-shot headless run takes a foreground
/// session instead — see `sessionConfiguration(foreground:)`.
///
/// Delegate callbacks arrive on `OperationQueue.main` so we can use
/// `MainActor.assumeIsolated` inside them to mutate observed state without
/// an extra Task hop.
@MainActor
@Observable
final class DownloadCoordinator: NSObject {
    static let shared = DownloadCoordinator(storage: FileStorage.production)

    struct Download: Identifiable {
        /// The transfer's lifecycle state. Replaces the old `Phase` (3 cases) plus a
        /// free-form `status` String: the transitional statuses that `Phase` couldn't
        /// express are now first-class cases, and there is no parallel string to keep
        /// in sync. Numeric progress is orthogonal and stays in the byte/fraction
        /// fields below — the status *text* is derived from this state in `ModelsView`.
        enum State: Equatable {
            case queued                  // "Queued" — waiting for a free concurrency slot, not yet transferring
            case connecting              // "Connecting…" / "Reconnecting…"
            case downloading             // actively transferring; numeric progress stays in bytes/explicitFraction/progress
            case pausing                 // "Pausing…"
            case paused                  // "Paused"
            case resuming                // "Resuming…"
            case failed(reason: String)  // carries the error text
        }
        /// A single-file (GGUF) transfer vs an MLX repo *directory* pull.
        enum Kind: Equatable { case file; case directory(prefix: String?) }

        let filename: String
        /// The HF resolve URL for a `.file` (GGUF) transfer. `nil` for an MLX
        /// `.directory` pull — those go through HubApi and have no single URL to
        /// resume against, so there is no fabricated coordinate to carry here.
        let url: URL?
        let repo: String?
        let familyId: String?
        var bytesDownloaded: Int64
        var totalBytes: Int64?
        var state: State
        var kind: Kind = .file
        /// A directory pull only exposes an aggregate fraction (no byte counts),
        /// so its progress is carried here rather than derived from bytes.
        var explicitFraction: Double?
        /// The typed model definition this download was ignited from, carried
        /// through to completion so the manifest is written verbatim.
        var source: Model?
        /// The catalog's explicitly-authored quant, when the download was ignited from
        /// a `CatalogEntry`. Preferred over deriving from the artifact name so the row
        /// shows the declared quant even for a repo slug that doesn't encode it; nil for
        /// non-catalog downloads (headless get, sideload), which fall back to `source`.
        var catalogQuant: String?
        /// Set when a connectivity drop paused this transfer (vs a user pause), so
        /// `resumeAfterReconnect` can auto-resume it without touching downloads the
        /// user paused on purpose. Cleared when the download resumes. In-memory only
        /// (not persisted): auto-resume applies within a session; after an app
        /// restart a still-paused download recovers via Resume All instead.
        var interruptedByNetwork: Bool = false

        /// Stable per-transfer key — not the storage key: a vision model is two
        /// transfers in one entry. For a file: `<repo>/<filename>`, unique across
        /// repos for the same filename. For a directory: `<repo>[/<prefix>]`, unique
        /// per MLX model in a multi-model repo.
        var key: String {
            switch kind {
            case .file: return LocalStorage.modelRelativePath(repo: repo, filename: filename)
            // Matches `HFModelRef.key` (`<repo>` or `<repo>/<prefix>`), the canonical form.
            case let .directory(prefix): return [repo, prefix].compactMap { $0 }.joined(separator: "/")
            }
        }
        var id: String { key }
        var progress: Double? {
            if let explicitFraction { return explicitFraction }
            guard let total = totalBytes, total > 0 else { return nil }
            return Double(bytesDownloaded) / Double(total)
        }

        /// Quant label for the download row. Prefers the catalog's authored `quant`,
        /// then the typed `source`'s derivation, then a filename parse for a source-less
        /// (legacy/sideload) record. The filename parse alone can't name an MLX quant —
        /// an MLX download's `filename` is the repo/directory leaf, not a GGUF weight
        /// file — which is why the explicit/source reads are what keep the row from
        /// showing "unknown" for MLX.
        var quant: String? {
            catalogQuant ?? source?.quant ?? LocalStorage.parseQuant(from: filename)
        }
    }

    private(set) var downloads: [String: Download] = [:]
    /// Bumped after each successful completion so views can react with their
    /// own refresh logic (e.g., re-scanning the models directory).
    private(set) var completedVersion: Int = 0
    var errorMessage: String?

    // Below are internal machinery, not observed UI state — `@ObservationIgnored`
    // keeps them out of the observation graph (and `lazy` is incompatible with an
    // observed stored property).

    /// The device storage this coordinator writes downloads into. Injected so
    /// tests point it at a temporary root.
    @ObservationIgnored let storage: Storage

    /// MLX directory downloads route through `HubApi` behind this seam (GGUF stays
    /// on the `URLSession` path below). Injectable so tests can supply a fake.
    @ObservationIgnored var mlxDownloader: MLXModelDownloading
    /// In-flight MLX pulls, keyed the same as `downloads`. Cancelling the task is
    /// how `cancel(key:)` stops an MLX download (HubApi honors `Task.isCancelled`).
    @ObservationIgnored private var mlxTasks: [String: Task<Void, Never>] = [:]

    /// Last time a progress update was published, per download. Used to
    /// debounce `didWriteData` (which fires far more than once per second)
    /// down to at most one UI update per second per download.
    @ObservationIgnored private var lastProgressPublishedAt: [String: Date] = [:]

    /// Minimum gap between published progress updates for one download.
    private static let progressUpdateInterval: TimeInterval = 1.0

    /// Cap on simultaneously-active model transfers. Additional downloads wait
    /// in `.queued` and start as slots free, so adding many models no longer
    /// opens a connection per model at once. Counts GGUF (URLSession) and MLX
    /// (HubApi) transfers together. Tunable — 2 bounds bandwidth/connection use
    /// while still overlapping one transfer with the next. Internal (not private)
    /// so tests can assert against the configured cap.
    static let maxConcurrentDownloads = 2

    /// Downloads waiting for a slot, FIFO. Each `start` closure ignites the real
    /// transfer and is invoked by `pumpQueue` when a slot frees; the matching
    /// `downloads[key]` entry sits in `.queued` until then.
    @ObservationIgnored private var pendingStarts: [(key: String, start: () -> Void)] = []

    /// Set by the AppDelegate in `handleEventsForBackgroundURLSession`; called
    /// once the session finishes replaying its queued events to the delegate.
    @ObservationIgnored var backgroundCompletionHandler: (() -> Void)?

    nonisolated private static let sessionIdentifier = "ai.liquid.pipette.downloads"

    /// Whether `arguments` name a headless run that completes in one go, which is
    /// the only case a foreground session is right for.
    ///
    /// `settings run` is excluded: it starts the planner worker, which stays
    /// resident until killed and can be suspended between claims, so it needs a
    /// background session's ability to outlive suspension. Every other headless
    /// verb runs to completion with the app held active by `devicectl --console`.
    ///
    /// Deep links are excluded too, by carrying no `headlessrun`: a link-launched
    /// bench is driven by someone holding the phone, who may background it.
    nonisolated static func usesForegroundSession(_ arguments: [String]) -> Bool {
        guard let marker = arguments.firstIndex(of: "headlessrun") else { return false }
        let verbs = arguments[arguments.index(after: marker)...]
        return !(verbs.first == "settings" && verbs.dropFirst().first == "run")
    }

    /// A background session hands the transfer to `nsurlsessiond`, which iOS
    /// schedules discretionarily — it throttles hard on a phone warm from
    /// benchmarking (measured 373 KB/s at 73°C, against 23 MB/s available on the
    /// same link). A one-shot headless cell fails outright if the model does not
    /// arrive, so it gains nothing from surviving suspension and pays for it in
    /// throughput.
    ///
    /// Switching costs one thing knowingly: a foreground session does not adopt
    /// transfers a previous launch left running in `nsurlsessiond`, so those are
    /// orphaned rather than resumed. They finish or die on their own, and the
    /// cell re-downloads. Accepted because a cell owns the phone for its
    /// duration, so a transfer outliving one is already leftover state.
    nonisolated static func sessionConfiguration(foreground: Bool) -> URLSessionConfiguration {
        let config = foreground
            ? URLSessionConfiguration.default
            : URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
        config.allowsCellularAccess = true
        config.waitsForConnectivity = true
        // Background-only — asks the system to relaunch us to deliver completions.
        config.sessionSendsLaunchEvents = !foreground
        return config
    }

    @ObservationIgnored private lazy var session: URLSession = {
        URLSession(configuration: Self.sessionConfiguration(foreground: foregroundSession),
                   delegate: self, delegateQueue: .main)
    }()

    @ObservationIgnored private let foregroundSession: Bool

    init(
        storage: Storage,
        mlxDownloader: MLXModelDownloading? = nil,
        foregroundSession: Bool = DownloadCoordinator.usesForegroundSession(CommandLine.arguments)
    ) {
        self.storage = storage
        self.mlxDownloader = mlxDownloader ?? HubMLXModelDownloader(downloadBase: storage.hubCacheDir)
        self.foregroundSession = foregroundSession
        super.init()
        restoreState()
        // When connectivity returns, resume the transfers a network drop paused.
        // The handler runs off the main actor, so hop back; reference the shared
        // instance rather than capturing `self` across the concurrency boundary.
        NetworkReachability.shared.onReconnect {
            Task { @MainActor in DownloadCoordinator.shared.resumeAfterReconnect() }
        }
    }

    // MARK: - Public API

    /// The single ignition point: start a download straight from its typed model
    /// definition. Dispatches by variant — the transport (URLSession for GGUF,
    /// HubApi `Task` for MLX), the on-disk key, and provenance are all derived
    /// from `source`, so callers never build URLs or repo strings themselves.
    /// A vision model's two files are enqueued as separate transfers, both tagged
    /// with the same `source`, so both land in one store entry (see
    /// `GGUFModelInstaller`).
    ///
    /// `declaredSizeBytes` is the catalog's authored size, when the ignition has one:
    /// an artifact bigger than the whole quota is refused here, before any transfer
    /// starts. Ignitions with no declared size (headless, deep links) skip the check —
    /// the pre-flight is best-effort by construction.
    func startDownload(_ source: Model, familyId: String? = nil, quant: String? = nil,
                       declaredSizeBytes: Int64? = nil) {
        if let declaredSizeBytes, declaredSizeBytes > storage.storageQuotaBytes {
            let refusal = DownloadError.exceedsQuota(
                neededBytes: declaredSizeBytes, quotaBytes: storage.storageQuotaBytes)
            errorMessage = refusal.localizedDescription
            AppLog.storage.warning("Refusing download: \(refusal.localizedDescription)")
            return
        }
        switch source {
        case let .mlx(m):
            guard let ref = m.ref else { refuseUnfetchable(source); return }
            startMLXDownload(ref, familyId: familyId, quant: quant)
        // Only the HuggingFace arm names something this client can fetch. A store form
        // (`relativeFile` / `absoluteFile`) is already-installed bytes, and a `url` arm is
        // a body authored for the CLI — both decode, neither downloads here, which is the
        // same split upstream draws between reading a spec and being able to run it.
        case let .ggufText(m):
            guard case let .huggingFace(repo, path, _) = m.source else {
                refuseUnfetchable(source); return
            }
            enqueueGGUF(repo: repo, filename: path,
                        source: source, familyId: familyId, quant: quant)
        case let .ggufVision(m):
            guard case let .huggingFace(repo, model, _, mmproj, _) = m.source else {
                refuseUnfetchable(source); return
            }
            enqueueGGUF(repo: repo, filename: model,
                        source: source, familyId: familyId, quant: quant)
            enqueueGGUF(repo: repo, filename: mmproj,
                        source: source, familyId: familyId, quant: quant)
        case .appleFoundationText:
            // AFM ships with the OS — there is nothing to download.
            AppLog.storage.error("startDownload called for appleFoundation, which has no downloadable coordinate")
        }
    }

    /// A source this client cannot fetch: a store-relative or absolute path (already
    /// installed, by definition) or a `url` arm no iOS download path implements.
    private func refuseUnfetchable(_ source: Model) {
        errorMessage = "This model's source can't be downloaded on iOS."
        AppLog.storage.error("startDownload: \(source.artifactName) names no fetchable HuggingFace coordinate")
    }

    /// Build the HF resolve URL for a single GGUF file and hand it to the
    /// URLSession primitive, carrying the typed `source` for provenance.
    private func enqueueGGUF(repo: HFRepo, filename: RepoSubpath, source: Model?,
                             familyId: String?, quant: String? = nil) {
        // `main` only when the coordinate names no revision. Hardcoding it ignored a
        // plan's pin outright, and 404s a repo whose default branch is named something
        // else.
        let ref = repo.revision?.value ?? "main"
        guard let url = URL(
            string: "https://huggingface.co/\(repo.description)/resolve/\(ref)/\(filename.value)"
        ) else { return }
        enqueueFileDownload(url: url, repo: repo.description, familyId: familyId, source: source, quant: quant)
    }

    /// Low-level URLSession transport for a single file. Not the public ignition
    /// surface — callers use `startDownload(_:familyId:)`; this is the shared GGUF
    /// primitive it funnels into.
    private func enqueueFileDownload(url: URL, repo: String?, familyId: String? = nil, source: Model? = nil,
                                     quant: String? = nil) {
        let filename = url.lastPathComponent
        let key = LocalStorage.modelRelativePath(repo: repo, filename: filename)

        if downloads[key] != nil { return }
        // Without a coordinate there is no store entry to place the file in, so the
        // install would have nowhere to put it — refuse rather than transfer bytes
        // the scanner will never find.
        guard let spec = GGUFModelInstaller.entrySpec(repo: repo, filename: filename, source: source),
              let blobs = storage.modelStore.blobsDir(for: spec) else {
            errorMessage = "\(filename) has no model coordinate to store it under."
            return
        }
        // Dedup is scoped to this model's entry: a different repo's identically-named
        // GGUF keys to its own entry and downloads fine.
        if FileManager.default.fileExists(atPath: blobs.appendingPathComponent(filename).path) {
            errorMessage = "\(filename) already exists."
            return
        }

        // Register the row up front in `.queued`; the concurrency gate starts the
        // transfer now if a slot is free, else it waits until `pumpQueue` runs it.
        downloads[key] = Download(
            filename: filename,
            url: url,
            repo: repo,
            familyId: familyId,
            bytesDownloaded: 0,
            totalBytes: nil,
            state: .queued,
            source: source,
            catalogQuant: quant
        )
        // Past the dedup and existence guards above, so this is a genuinely new transfer and can't
        // double-count. Captured on enqueue rather than when the gate grants a slot: the queued
        // wait is part of the user's download, and a device that never drains its queue is exactly
        // what the started-vs-completed gap should show.
        Analytics.capture(AnalyticsEvents.modelDownloadStarted, [AnalyticsEvents.modelId: filename])

        gate(key: key) { [weak self] in self?.beginFileDownload(key: key) }
    }

    /// Ignite the URLSession transfer for a registered GGUF download once the
    /// concurrency gate grants it a slot. The resume record is persisted only
    /// here, so a still-queued download leaves nothing to reconcile on relaunch.
    private func beginFileDownload(key: String) {
        guard var dl = downloads[key], let url = dl.url else { return }

        var request = URLRequest(url: url)
        attachHfAuth(&request, claimToken: dl.source?.repo?.authToken, model: dl.source)

        let task = session.downloadTask(with: request)
        task.taskDescription = key
        task.resume()

        dl.state = .connecting
        downloads[key] = dl

        DownloadStore.saveRecord(DownloadRecord(
            filename: dl.filename,
            urlString: url.absoluteString,
            repo: dl.repo,
            familyId: dl.familyId,
            source: dl.source
        ))
    }

    /// Start (or no-op if already present) an MLX model download. Progress flows into
    /// the same `downloads` map the GGUF path uses; the finished model is installed
    /// via `MLXModelInstaller` (validate → relocate → record).
    ///
    /// Unlike GGUF, MLX pulls are an in-process `Task` around `HubApi` (not a
    /// background `URLSession`), so they aren't persisted via `DownloadStore` and the
    /// in-progress row doesn't survive app termination. This is deliberate: a killed
    /// pull leaves its partial files in `hubCacheDir`, and re-initiating the download
    /// resumes from that cache rather than restarting — so the model isn't lost, it
    /// just loses its live progress indicator until the user taps download again.
    /// That resume window closes at the next sweep, which reclaims hub snapshots no
    /// in-flight pull is holding (they are multi-GB and nothing else ever frees them).
    func startMLXDownload(_ ref: HFModelRef, familyId: String? = nil, quant: String? = nil) {
        let key = ref.key
        if downloads[key] != nil { return }
        if let blobs = storage.modelStore.blobsDir(for: .mlx(ref.asMlx())),
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
            source: .mlx(ref.asMlx()),
            catalogQuant: quant
        )
        // The MLX twin of the GGUF capture in `enqueueFileDownload`, and past the same dedup and
        // existence guards. Without it `finishMLX`/`failMLX` emit terminal events with no matching
        // start, so every MLX pull would show up in the funnel as a completion out of nowhere.
        Analytics.capture(AnalyticsEvents.modelDownloadStarted, [AnalyticsEvents.modelId: ref.leaf])

        gate(key: key) { [weak self] in self?.beginMLXDownload(ref, key: key) }
    }

    /// Launch the HubApi pull for a registered MLX download once the concurrency
    /// gate grants it a slot.
    private func beginMLXDownload(_ ref: HFModelRef, key: String) {
        guard var dl = downloads[key] else { return }
        dl.state = .connecting
        downloads[key] = dl

        // Resolved here, on the main actor, so the downloader stays a pure transport; the
        // coordinate's token is what a plan-dispatched pull must present.
        let credential = resolveHfToken(claimToken: ref.repo.authToken, model: dl.source)
        // The MLX twin of `attachHfAuth`'s line. HubApi builds its own requests, so this is
        // the only place that can say whether the pull carries a credential.
        if let credential {
            AppLog.storage.info(
                "hf auth: mlx pull of \(ref.key) authenticated source=\(credential.source.rawValue)")
        } else {
            AppLog.storage.info("hf auth: no token available — pulling mlx \(ref.key) anonymously")
        }
        let downloader = mlxDownloader
        mlxTasks[key] = Task { [weak self] in
            do {
                let dir = try await downloader.download(ref, token: credential?.token) { fraction in
                    Task { @MainActor [weak self] in self?.applyMLXProgress(key: key, fraction: fraction) }
                }
                await MainActor.run { [weak self] in
                    self?.finishMLX(key: key, downloadedDir: dir, ref: ref)
                }
            } catch is CancellationError {
                // `cancel(key:)` already cleared the UI entry.
            } catch DownloadError.cancelled {
                // Ditto — the downloader mapped task cancellation to a typed error.
            } catch {
                await MainActor.run { [weak self] in self?.failMLX(key: key, error: error) }
            }
        }
    }

    private func applyMLXProgress(key: String, fraction: Double) {
        guard var dl = downloads[key] else { return }
        // Debounce identically to the GGUF path; the final frame always publishes.
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

    /// Install the finished download (validate → relocate → record) and clear the
    /// in-flight entry. Validation/relocation live in `MLXModelInstaller`.
    private func finishMLX(key: String, downloadedDir: URL, ref: HFModelRef) {
        // A `cancel(key:)` that raced the download's completion already removed the
        // in-flight entry; don't install a model the user cancelled.
        guard downloads[key] != nil else {
            try? FileManager.default.removeItem(at: downloadedDir)
            mlxTasks.removeValue(forKey: key)
            return
        }
        do {
            try MLXModelInstaller(ref: ref, storage: storage).install(from: downloadedDir)
        } catch {
            failMLX(key: key, error: error)
            return
        }
        // Swept before the row is cleared, so the just-finished download still pins
        // itself through `downloads` as well as through `justInstalled`.
        sweepAfterInstall(justInstalled: ModelStorageKey.of(.mlx(ref.asMlx())))
        let filename = downloads[key]?.filename
        // Same properties as the GGUF completion in `applyCompletion`: without `size_bytes` here, any
        // average-download-size query silently becomes GGUF-only rather than visibly incomplete.
        let totalBytes = downloads[key]?.totalBytes
        downloads.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        mlxTasks.removeValue(forKey: key)
        completedVersion &+= 1
        Analytics.capture(AnalyticsEvents.modelDownloadCompleted, [
            AnalyticsEvents.modelId: filename,
            AnalyticsEvents.sizeBytes: totalBytes,
        ])
        pumpQueue()
    }

    private func failMLX(key: String, error: Error) {
        mlxTasks.removeValue(forKey: key)
        if var dl = downloads[key] {
            dl.state = .failed(reason: "Failed: \(error.localizedDescription)")
            downloads[key] = dl
        }
        errorMessage = "MLX download failed: \(error.localizedDescription)"
        Analytics.capture(AnalyticsEvents.modelDownloadFailed, [
            AnalyticsEvents.modelId: downloads[key]?.filename,
            AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
        ])
        pumpQueue()
    }

    func pause(key: String) {
        // MLX `.directory` pulls have no pause/resume — only cancel (which stops
        // the HubApi Task). Mirror the `.file` gating in `restoreState`: leaving
        // this open would flip a "Paused" UI while the MLX Task kept running.
        guard let dl = downloads[key], case .file = dl.kind else { return }
        // A queued download hasn't started transferring — there's nothing to
        // pause, and flipping it to `.paused` would strand its start closure in
        // `pendingStarts`, from which `pumpQueue` would later resurrect it.
        guard dl.state != .queued else { return }
        // `pause` UI state flips immediately; the resume-data callback writes
        // the blob when it arrives from the system.
        if var dl = downloads[key] {
            dl.state = .pausing
            downloads[key] = dl
        }
        session.getAllTasks { [weak self] tasks in
            guard let self = self else { return }
            let match = tasks.first(where: { $0.taskDescription == key }) as? URLSessionDownloadTask
            guard let match else {
                Task { @MainActor [weak self] in
                    self?.markPaused(key: key)
                }
                return
            }
            match.cancel(byProducingResumeData: { data in
                Task { @MainActor [weak self] in
                    guard let self = self else { return }
                    if let data = data {
                        DownloadStore.saveResumeData(data, for: key)
                    }
                    self.markPaused(key: key)
                }
            })
        }
    }

    func resume(key: String) {
        // Only `.file` (URLSession) downloads can resume; an MLX `.directory` pull
        // has no URLSession task and no URL to resume against. Resume is valid only
        // from `.paused` — guarding it dedups re-entrant taps and blocks resuming a
        // download that's already active (either would spawn a duplicate task).
        guard let dl = downloads[key], case .file = dl.kind,
              case .paused = dl.state, dl.url != nil else { return }

        // Route through the concurrency gate so resuming several paused downloads
        // can't push active transfers past the cap; a resume with no free slot
        // waits in `.queued` and is started later by `pumpQueue`.
        var queuedDl = dl
        queuedDl.state = .queued
        queuedDl.interruptedByNetwork = false
        downloads[key] = queuedDl
        gate(key: key) { [weak self] in self?.performResume(key: key) }
    }

    /// Resume the transfers a connectivity drop paused (see `applyNetworkInterruption`),
    /// leaving downloads the user paused on purpose (`interruptedByNetwork == false`)
    /// untouched. Wired to `NetworkReachability.onReconnect`.
    func resumeAfterReconnect() {
        let keys = downloads.compactMap { key, dl -> String? in
            guard dl.interruptedByNetwork, case .paused = dl.state else { return nil }
            return key
        }
        for key in keys { resume(key: key) }
    }

    /// Resume every paused download at once — the "Resume All" affordance —
    /// covering both user-paused and network-paused transfers.
    func resumeAll() {
        let keys = downloads.compactMap { key, dl -> String? in
            if case .paused = dl.state { return key }
            return nil
        }
        for key in keys { resume(key: key) }
    }

    /// Ignite the URLSession transfer for a resumed download once the concurrency
    /// gate grants it a slot — from saved resume data when present, else afresh.
    private func performResume(key: String) {
        guard var dl = downloads[key], let url = dl.url else { return }

        let task: URLSessionDownloadTask
        if let resumeData = DownloadStore.loadResumeData(for: key) {
            task = session.downloadTask(withResumeData: resumeData)
            DownloadStore.clearResumeData(for: key)
        } else {
            // Resume data missing — start over from zero with the original URL.
            var request = URLRequest(url: url)
            attachHfAuth(&request, claimToken: dl.source?.repo?.authToken, model: dl.source)
            task = session.downloadTask(with: request)
        }
        task.taskDescription = key
        task.resume()

        dl.state = .resuming
        downloads[key] = dl
    }

    /// Cancel every download, running or queued.
    ///
    /// For the sign-out reset (PIP-459), which deletes `models/`: a transfer left running would land a
    /// model into the tree that was just cleared, on a device whose registration is gone. The keys are
    /// snapshotted first because `cancel(key:)` mutates `downloads`.
    func cancelAll() {
        for key in Array(downloads.keys) { cancel(key: key) }
    }

    func cancel(key: String) {
        // Flip UI state immediately; the actual task cancel runs async.
        downloads.removeValue(forKey: key)
        lastProgressPublishedAt.removeValue(forKey: key)
        // Drop it from the wait queue if it never started, and fill the freed
        // slot from the queue once this cancel returns.
        pendingStarts.removeAll { $0.key == key }
        defer { pumpQueue() }
        DownloadStore.clearRecord(for: key)
        DownloadStore.clearResumeData(for: key)

        // MLX pull: cancelling the Task stops HubApi (it honors Task.isCancelled).
        if let mlxTask = mlxTasks.removeValue(forKey: key) {
            mlxTask.cancel()
            return
        }

        session.getAllTasks { tasks in
            tasks.first(where: { $0.taskDescription == key })?.cancel()
        }
    }

    // MARK: - Storage quota

    /// fetch → publish → sweep → return: the artifact lands first, then the store is
    /// swept back under the cap. Peak disk is the quota plus the newest artifact, and
    /// that is intended — a reservation would have to guess the final size.
    private func sweepAfterInstall(justInstalled: ModelStorageKey?) {
        storage.sweepToQuota(pinning: sweepPins(justInstalled: justInstalled))
    }

    /// Whether any transfer is registered for `source`. After `startDownload` this is how
    /// a caller tells acceptance from a silent decline — an oversize artifact, a
    /// coordinate this client cannot fetch, or bytes already on disk all leave their
    /// reason in `errorMessage` and enqueue nothing.
    func hasTransfer(for source: Model) -> Bool {
        downloads.values.contains { $0.source == source }
    }

    /// Whether every one of `source`'s transfers is paused, so nothing will advance it
    /// until the user resumes.
    func isPaused(_ source: Model) -> Bool {
        let rows = downloads.values.filter { $0.source == source }
        return !rows.isEmpty && rows.allSatisfy { $0.state == .paused }
    }

    /// Aggregate progress across `source`'s transfers, or nil when none report a
    /// fraction yet. A vision model's two rows average, so one file finishing shows as
    /// half rather than as done.
    func fraction(for source: Model) -> Double? {
        let fractions = downloads.values.filter { $0.source == source }.compactMap(\.progress)
        guard !fractions.isEmpty else { return nil }
        return fractions.reduce(0, +) / Double(fractions.count)
    }

    /// Bytes moved for `source`, summed across its rows so a vision model's two files
    /// rise once instead of restarting per file — the CLI sums an artifact's parts the
    /// same way.
    ///
    /// Nil when no row carries byte counts, which is every MLX directory pull: HubApi
    /// reports an aggregate fraction (`explicitFraction`) and no bytes, so those keep
    /// reporting a percentage alone.
    ///
    /// `total` is nil unless *every* row could be sized. A sum needs every term, and an
    /// understated total reads as a transfer that stalls just short of the end.
    func transferred(for source: Model) -> (done: Int64, total: Int64?)? {
        let rows = downloads.values.filter { $0.source == source && $0.explicitFraction == nil }
        guard !rows.isEmpty else { return nil }
        let total = rows.reduce(Int64?.some(0)) { running, row in
            guard let running, let rowTotal = row.totalBytes else { return nil }
            return running + rowTotal
        }
        return (rows.reduce(0) { $0 + $1.bytesDownloaded }, total)
    }

    /// The failure reason of any transfer belonging to `source`, if one failed.
    ///
    /// A vision model has two rows in one entry, so either failing means the entry will
    /// not resolve — which is what `ensureModel` is waiting to hear.
    func failureReason(for source: Model) -> String? {
        downloads.values.lazy
            .filter { $0.source == source }
            .compactMap { if case let .failed(reason) = $0.state { reason } else { nil } }
            .first
    }

    /// Entries a sweep must leave alone: the one just published, every in-flight
    /// download's entry (which covers a vision model's second transfer and every
    /// queued or paused row), and every model a running or paused job still needs.
    /// Job manifests are read rather than taking a `JobStore` dependency, so the
    /// coordinator doesn't gain a second store reference.
    func sweepPins(justInstalled: ModelStorageKey?) -> SweepPins {
        var pins = SweepPins.activeJobs(storage.loadAllJobManifests())
        if let justInstalled { pins.entries.insert(justInstalled) }
        for download in downloads.values {
            guard let source = download.source else { continue }
            if let key = ModelStorageKey.of(source) { pins.entries.insert(key) }
            // An MLX pull is still snapshotting into the hub cache; that tree is its
            // partial download, not a remnant.
            if case .directory = download.kind, let repo = download.repo {
                pins.hubRepos.insert(repo)
            }
        }
        return pins
    }

    // MARK: - Concurrency gate

    /// Number of downloads currently occupying a transfer slot (connecting or
    /// moving), excluding queued, paused, and failed entries. Counts GGUF
    /// (URLSession) and MLX (HubApi) transfers alike.
    private var activeDownloadCount: Int {
        downloads.values.reduce(into: 0) { count, dl in
            switch dl.state {
            case .connecting, .downloading, .resuming, .pausing: count += 1
            case .queued, .paused, .failed: break
            }
        }
    }

    /// Start `start` now if a slot is free; otherwise leave the download parked
    /// in `.queued` and enqueue it to run when a slot frees up.
    private func gate(key: String, start: @escaping () -> Void) {
        if activeDownloadCount < Self.maxConcurrentDownloads {
            start()
        } else {
            pendingStarts.append((key: key, start: start))
        }
    }

    /// Fill freed slots from the pending queue, FIFO. Skips entries the user
    /// cancelled while they were queued (their `downloads` entry is already gone).
    private func pumpQueue() {
        while activeDownloadCount < Self.maxConcurrentDownloads, !pendingStarts.isEmpty {
            let next = pendingStarts.removeFirst()
            guard downloads[next.key] != nil else { continue }
            next.start()
        }
    }

    // MARK: - Restoration

    private func restoreState() {
        for record in DownloadStore.allRecords() {
            let key = LocalStorage.modelRelativePath(repo: record.repo, filename: record.filename)
            guard let url = URL(string: record.urlString) else {
                DownloadStore.clearRecord(for: key)
                continue
            }
            let isPaused = DownloadStore.hasResumeData(for: key)
            downloads[key] = Download(
                filename: record.filename,
                url: url,
                repo: record.repo,
                familyId: record.familyId,
                bytesDownloaded: 0,
                totalBytes: nil,
                state: isPaused ? .paused : .connecting,
                source: record.source,
                // Only GGUF transfers are persisted, and their quant re-derives from
                // `source` (the weight filename), so no explicit quant need survive.
                catalogQuant: nil
            )
        }

        // Only entries restored from records above are candidates for
        // reconciliation. `getAllTasks` is async, so a download the user (or a
        // headless run) starts before its callback fires would otherwise be visible
        // to the loop below and wrongly deleted — it has no live task *yet*. Scope
        // the sweep to the restored set so a freshly-started download is never touched.
        let restoredKeys = Set(downloads.keys)

        // Force lazy session creation so we reconnect to any background tasks
        // still running from a previous launch, then reconcile.
        session.getAllTasks { [weak self] tasks in
            Task { @MainActor [weak self] in
                guard let self = self else { return }
                let liveNames = Set(tasks.compactMap { $0.taskDescription })
                // Entries that claim to be running but have no live task and no
                // resume-data → stale. Drop them; the system will redeliver
                // any pending completion events via the delegate.
                //
                // Only `.file` (URLSession-backed) downloads participate: an MLX
                // `.directory` pull is driven by a Swift `Task`/HubApi, not a
                // URLSession task, so it never appears in `liveNames` — reconciling
                // it here would wrongly delete an in-flight MLX download.
                // A restored, not-yet-reconnected `.file` download sits in
                // `.connecting` (restoreState maps non-paused records to it); those
                // are the entries this sweep reconciles. `.paused` (has resume data)
                // and MLX `.directory` pulls are left untouched.
                for (name, dl) in self.downloads {
                    guard case .connecting = dl.state,
                          restoredKeys.contains(name), case .file = dl.kind else { continue }
                    if !liveNames.contains(name)
                        && !DownloadStore.hasResumeData(for: name) {
                        self.downloads.removeValue(forKey: name)
                        DownloadStore.clearRecord(for: name)
                    }
                }
            }
        }
    }

    // MARK: - Main-actor state mutations

    fileprivate func applyProgress(key: String, written: Int64, expected: Int64) {
        guard var dl = downloads[key] else { return }
        // Debounce: publish at most one progress update per second per
        // download. The first frame (no prior timestamp) and the final frame
        // (written == total) always go through so the bar isn't a second stale
        // at start/finish. `didWriteData` fires continuously, so a dropped
        // frame's bytes are simply carried by the next accepted one.
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

    /// Provenance captured from `downloads[key]` *before* the async move so
    /// completion can write the model's sidecar manifest even if `cancel(key:)`
    /// raced the move and removed the in-memory entry. Re-reading `downloads[key]`
    /// after the move would lose the entry to that race, so the manifest would fall
    /// back to string reconstruction (losing the exact typed `source`) — or, for a
    /// bare key with no repo, be skipped entirely, orphaning the moved file.
    struct CompletionProvenance {
        let repo: String?
        let filename: String
        /// The typed definition the file was ignited from, when known. Lets
        /// completion write an exact manifest instead of reconstructing one.
        var source: Model?
    }

    /// Resolve download provenance for `key`. Prefers the in-memory
    /// `downloads[key]` entry: reading it here, before the move, is what closes
    /// the `cancel(key:)`-races-the-move window — it also carries the typed
    /// `source`, which the key-reconstruction fallback can't recover.
    ///
    /// The fallbacks cover a *different* case: a completion the system
    /// redelivers after relaunch, when no in-memory entry exists. They don't
    /// recover from a cancel race — `cancel(key:)` clears the on-disk record
    /// too. Falls back to the on-disk record; failing that, reconstructs
    /// `repo`/`filename` from the key itself (`<repo>/<filename>` or bare
    /// `<filename>` for sideloads), with a nil `source`.
    func captureProvenance(key: String) -> CompletionProvenance {
        if let dl = downloads[key] {
            return CompletionProvenance(repo: dl.repo, filename: dl.filename, source: dl.source)
        }
        if let record = DownloadStore.allRecords().first(where: {
            LocalStorage.modelRelativePath(repo: $0.repo, filename: $0.filename) == key
        }) {
            return CompletionProvenance(
                repo: record.repo, filename: record.filename, source: record.source)
        }
        let filename = (key as NSString).lastPathComponent
        let repo = filename == key ? nil : (key as NSString).deletingLastPathComponent
        return CompletionProvenance(repo: repo, filename: filename)
    }

    /// Clear the completed download's in-flight state and bump `completedVersion`
    /// (or record the failure). The artifact was already relocated and its manifest
    /// written by the `ModelInstaller` before this runs.
    func applyCompletion(key: String, result: Result<Void, Error>) {
        lastProgressPublishedAt.removeValue(forKey: key)
        switch result {
        case .success:
            // Both reads must precede the clears below: `downloads` and the `DownloadStore` record are
            // what `captureProvenance` reads, and `totalBytes` lives on the row about to be removed.
            // Provenance rather than `downloads[key]?.filename`: a completion the system redelivers
            // after relaunch has no in-memory row, and those are precisely the longest-running
            // downloads: reading the row directly would drop `model_id` on exactly the transfers
            // most worth attributing. `totalBytes` has no such fallback and is simply absent there.
            let provenance = captureProvenance(key: key)
            let totalBytes = downloads[key]?.totalBytes
            downloads.removeValue(forKey: key)
            DownloadStore.clearRecord(for: key)
            DownloadStore.clearResumeData(for: key)
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

    /// Record a terminal download error: surface the alert, clear the on-disk
    /// record + resume-data (terminal, so nothing restores on next launch), and
    /// keep the download as a dismissible `.failed` row — visible, cancellable,
    /// and detectable by the headless runner, matching the MLX path (`failMLX`)
    /// rather than silently vanishing. Callers own slot bookkeeping (`pumpQueue`).
    private func markFailed(key: String, error: Error) {
        let reason = humanizedDownloadError(error)
        errorMessage = reason
        // Before the clears below: a failure redelivered after relaunch has no in-memory row, and
        // the on-disk record `captureProvenance` falls back to is about to be deleted.
        let provenance = captureProvenance(key: key)
        DownloadStore.clearRecord(for: key)
        DownloadStore.clearResumeData(for: key)
        if var dl = downloads[key] {
            dl.state = .failed(reason: reason)
            downloads[key] = dl
        }
        // Error KIND only, never the humanized reason: it embeds the source URL.
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

    /// A connectivity error stopped an in-flight transfer. Unlike `applyFailure`,
    /// keep the download resumable: save any resume data, mark it `.paused` and
    /// flag it network-interrupted, and leave its record on disk — so it resumes
    /// automatically when connectivity returns (`resumeAfterReconnect`) or via the
    /// user's Resume All. No `errorMessage`: a transient blip shouldn't alarm.
    fileprivate func applyNetworkInterruption(key: String, resumeData: Data?) {
        guard var dl = downloads[key], case .file = dl.kind else { return }
        if let resumeData { DownloadStore.saveResumeData(resumeData, for: key) }
        lastProgressPublishedAt.removeValue(forKey: key)
        dl.state = .paused
        dl.interruptedByNetwork = true
        downloads[key] = dl
        pumpQueue()
    }

    fileprivate func callBackgroundCompletion() {
        backgroundCompletionHandler?()
        backgroundCompletionHandler = nil
    }

    private func markPaused(key: String) {
        guard var dl = downloads[key] else { return }
        dl.state = .paused
        downloads[key] = dl
        // Pausing frees a slot — let a queued download take it.
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
            self.applyProgress(key: key,
                               written: fileOffset,
                               expected: expectedTotalBytes)
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                 didFinishDownloadingTo location: URL) {
        guard let key = downloadTask.taskDescription else { return }
        // The temp file at `location` is deleted when this method returns, so the
        // install (relocate + record) must run synchronously here. The delegate
        // queue is `.main`, so we hop onto the main actor without a Task.
        MainActor.assumeIsolated {
            // Capture provenance *before* the move: a `cancel(key:)` racing the move
            // clears `downloads[key]`, so reading it after would lose `repo`/`source`
            // and orphan the moved file. The installer takes the captured provenance.
            let provenance = self.captureProvenance(key: key)
            let result: Result<Void, Error>
            if let http = downloadTask.response as? HTTPURLResponse,
               !(200...299).contains(http.statusCode) {
                // The failing URL travels with the error, as the crate's
                // `ModelFetchError::Http { url, .. }` carries it: a 404 on a pinned
                // revision and a 404 on a mistyped filename read identically without it,
                // and a headless run has no other channel to tell them apart.
                var info: [String: Any] = [
                    NSLocalizedDescriptionKey:
                        "The download couldn't be completed (server returned status \(http.statusCode)). Please try again.",
                ]
                if let url = downloadTask.originalRequest?.url ?? http.url {
                    info[NSURLErrorFailingURLErrorKey] = url
                }
                result = .failure(NSError(domain: "Pipette", code: http.statusCode, userInfo: info))
            } else {
                let installer = GGUFModelInstaller(
                    repo: provenance.repo, filename: provenance.filename,
                    source: provenance.source, storage: self.storage
                )
                result = Result { try installer.install(from: location) }
                if case .success = result {
                    self.sweepAfterInstall(
                        justInstalled: installer.entrySpec.flatMap(ModelStorageKey.of))
                }
            }
            self.applyCompletion(key: key, result: result)
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                 didCompleteWithError error: Error?) {
        guard let error = error, let key = task.taskDescription else { return }
        // User-initiated pause/cancel surfaces as URLError.cancelled — that
        // path already did its own cleanup (pause → resume data, cancel →
        // record removal), so ignore the error event here.
        let nsErr = error as NSError
        if nsErr.domain == NSURLErrorDomain && nsErr.code == NSURLErrorCancelled {
            return
        }
        // A connectivity error is transient — keep the download resumable so it
        // recovers on reconnect. Other errors (bad URL, gated auth, …) are
        // terminal and drop the download as before.
        let resumeData = nsErr.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let recoverable = isRecoverableNetworkError(error)
        MainActor.assumeIsolated {
            if recoverable {
                self.applyNetworkInterruption(key: key, resumeData: resumeData)
            } else {
                self.applyFailure(key: key, error: error)
            }
        }
    }

    nonisolated func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        MainActor.assumeIsolated {
            self.callBackgroundCompletion()
        }
    }
}

// MARK: - Error display

/// Single source of truth for how the app treats a `URLError.code`: whether the
/// failure is transient connectivity (keep the download resumable and auto-resume
/// on reconnect) and the user-facing message when we override the system one.
/// Keeps `isRecoverableNetworkError` and `humanizedDownloadError` from drifting
/// apart over the same enum.
nonisolated func downloadURLErrorInfo(_ code: URLError.Code) -> (recoverable: Bool, message: String?) {
    switch code {
    case .notConnectedToInternet, .networkConnectionLost:
        return (true, "No internet connection. Check your network and try again.")
    case .timedOut:
        return (true, "The download timed out. Check your connection and try again.")
    case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed:
        return (true, "Couldn't reach the download server. Please try again in a moment.")
    case .dataNotAllowed, .internationalRoamingOff:
        return (true, nil)
    case .userAuthenticationRequired:
        return (false, "This model is gated. Add a Hugging Face access token in Settings, then try again.")
    default:
        return (false, nil)
    }
}

/// Whether a URLSession failure is transient connectivity worth keeping the
/// download resumable for, rather than a terminal error no retry fixes.
nonisolated func isRecoverableNetworkError(_ error: Error) -> Bool {
    guard let urlError = error as? URLError else { return false }
    return downloadURLErrorInfo(urlError.code).recoverable
}

/// Turn a download failure into a short, human-readable sentence for the
/// "Download Error" alert. Maps common connectivity / HTTP-status cases to
/// plain language and falls back to the system description otherwise.
///
/// An HTTP failure also names the status and the URL it was returned for, as the crate's
/// `ModelFetchError::Http` does — the sentence alone cannot say *which* coordinate was
/// refused, and on a plan-dispatched run that is the whole diagnosis.
func humanizedDownloadError(_ error: Error) -> String {
    if let urlError = error as? URLError {
        return downloadURLErrorInfo(urlError.code).message ?? urlError.localizedDescription
    }

    let ns = error as NSError
    if ns.domain == "Pipette" {
        let sentence: String? = switch ns.code {
        case 401, 403:
            "Access to this model was denied. It may be gated. Add a Hugging Face token in Settings and try again."
        case 404:
            "This model file couldn't be found on the server. It may have moved or been removed."
        case 500...599:
            "The download server ran into a problem. Please try again in a moment."
        default:
            nil
        }
        if let sentence {
            guard let url = ns.userInfo[NSURLErrorFailingURLErrorKey] as? URL else {
                return sentence
            }
            return "\(sentence) (HTTP \(ns.code) for \(url.absoluteString))"
        }
    }

    return error.localizedDescription
}
