import Foundation

/// The device-storage seam: benchmark catalog, job manifests, per-cell result
/// artifacts, registration, and downloaded models. Injected as a dependency so
/// callers never reach a process-global — `FileStorage.production` backs the app
/// (Application Support + Caches), and tests pass their own `FileStorage(dataRoot:)`
/// or a mock. Root-independent metadata parsers live on the `LocalStorage`
/// namespace instead, since they need no storage.
///
/// Directory layout (under `FileStorage.dataRoot`):
///   identity/         — registration JSON, client settings
///   benchmarks/       — the generated `local/` and synced `remote/` catalog halves
///   results/          — one directory per result, under local/ or remote/{pending,synced}
///   state/evals/      — per-run eval checkpoints, so a killed run resumes
///   jobs/<jobId>/     — manifest plus per-cell payload/submission artifacts
///   models/<key>/     — one entry per model: its manifest + a `blobs/` payload,
///                       backup-excluded and capped by the storage quota
nonisolated protocol Storage: Sendable {
    // MARK: Roots
    var dataRoot: URL { get }
    var jobsDir: URL { get }
    var modelsDir: URL { get }
    var hubCacheDir: URL { get }

    // MARK: Eval checkpoints
    /// Per-sample resume state, as the CLI's `ws.eval_completions()`.
    var evalCompletions: EvalCompletionsStore { get }

    // MARK: Benchmark catalog
    /// The generated `local/` and synced `remote/` halves, as the CLI's `ws.benchmarks()`.
    var benchmarks: BenchmarkStore { get }

    // MARK: Identity
    /// The registration record and the signing key, as the CLI's `ws.identity()`.
    var identity: IdentityStore { get }

    // MARK: Lifecycle
    @discardableResult
    func cleanupLegacyEdgeEvalsStorage() -> [URL]
    /// One-shot move of the identity files from `metadata/` to `identity/`.
    @discardableResult
    func migrateIdentityDirectory() -> [String]
    func recoverInterruptedJobs()
    @MainActor func resetDeviceData() throws

    // MARK: Jobs
    func saveJobManifest(_ manifest: JobManifest)
    func loadJobManifest(jobId: JobId) -> JobManifest?
    func loadAllJobManifests() -> [JobManifest]
    func deleteJob(jobId: JobId)

    // MARK: Active-cell crash sentinel
    func saveActiveCellSentinel(_ sentinel: ActiveCellSentinel, jobId: JobId)
    func loadActiveCellSentinel(jobId: JobId) -> ActiveCellSentinel?
    func clearActiveCellSentinel(jobId: JobId)
    func crashPayloadIsFresh(_ sentinel: ActiveCellSentinel, jobId: JobId) -> Bool

    // MARK: Results
    /// Recorded results, keyed by cell id, as the CLI's `ws.results()`.
    var results: ResultsStore { get }

    // MARK: Models
    /// The artifact store rooted at `modelsDir`, which owns the entry layout. The
    /// per-spec accessors below are the protocol surface over it; new callers should
    /// prefer the store directly.
    var modelStore: ModelArtifactStore { get }
    func touchModelLastUsed(_ spec: Model)
    func resolveModelPath(_ storedPath: String?) -> String?
    @MainActor func availableModels() -> [DiscoveredModel]
    @MainActor func deleteModel(_ model: DiscoveredModel)

    // MARK: Storage quota
    var storageQuotaBytes: Int64 { get }
    /// The built-in limit, so the Settings ladder can offer "back to default" without
    /// a view naming the concrete store.
    var defaultStorageQuotaBytes: Int64 { get }
    func setStorageQuotaBytes(_ bytes: Int64) throws
    func storageUsageBytes() -> Int64
    /// What a sweep would reclaim right now, touching no disk — the crate's `quota::plan`.
    /// Separate from `sweepToQuota` so `storage gc dry-run=1` can report a plan it will
    /// not carry out.
    func sweepPlan(pinning pins: SweepPins) -> SweepPlan
    /// Carry out a plan already taken — the crate's `quota::apply_sweep`, plus the
    /// hub-cache phase that has no upstream analogue.
    @discardableResult
    func reclaim(_ plan: SweepPlan, pinning pins: SweepPins) -> SweepReport
    @discardableResult
    func sweepToQuota(pinning pins: SweepPins) -> SweepReport
}

extension Storage {
    /// Derived, so a conformer only has to supply `modelsDir`.
    nonisolated var modelStore: ModelArtifactStore { ModelArtifactStore(modelsDir: modelsDir) }

    /// Derived from `dataRoot` for the same reason — the identity store owns the
    /// `identity/` layout, so a conformer supplies no path for it either.
    nonisolated var identity: IdentityStore {
        IdentityStore(root: dataRoot.appendingPathComponent("identity", isDirectory: true))
    }

    /// Its own top-level directory, as the crate's `ws.benchmarks()` is.
    nonisolated var benchmarks: BenchmarkStore {
        BenchmarkStore(root: dataRoot.appendingPathComponent("benchmarks", isDirectory: true))
    }

    /// Results live beside the jobs that produced them rather than inside them, as the
    /// crate's `ws.results()` does — the location, not the job, is what tells you how far
    /// a result has travelled.
    nonisolated var results: ResultsStore {
        ResultsStore(root: dataRoot.appendingPathComponent("results", isDirectory: true))
    }

    /// Eval resume state, as the crate's `ws.eval_completions()` — under `state/`, where
    /// the crate keeps it too.
    nonisolated var evalCompletions: EvalCompletionsStore {
        EvalCompletionsStore(
            root: dataRoot.appendingPathComponent("state/evals", isDirectory: true))
    }
}

/// Production `Storage`: files under `dataRoot` (Application Support) plus a
/// `cacheRoot` (system Caches) for re-downloadable content. A value with explicit
/// roots — construct one with a temporary root in tests rather than mutating a global.
nonisolated struct FileStorage: Storage {
    /// Root data directory (`metadata/`, `jobs/`, `models/` live under it).
    let dataRoot: URL
    /// Root caches directory (re-downloadable content, e.g. the MLX hub cache).
    let cacheRoot: URL
    /// Where `identity` gets its signing key. Defaults to *none* so the Keychain — which
    /// is device-global and outlives any `dataRoot` — is reached only by the one store
    /// that names it. A store built for a test or a preview is therefore isolated by
    /// construction, as the crate's is: it keeps the key in a file under the workspace.
    var privateKeySource: @Sendable () -> PrivateKeyHex? = { nil }
    /// Where the registration record outlives an uninstall. Defaults off for the same
    /// reason as `privateKeySource`: the Keychain is device-global, so only the production
    /// store reaches it.
    var registrationMirror: RegistrationMirror = .disabled

    /// Own implementation rather than the protocol extension's: the key source is this
    /// type's business, so `Storage` need not carry it.
    nonisolated var identity: IdentityStore {
        IdentityStore(root: dataRoot.appendingPathComponent("identity", isDirectory: true),
                      privateKeySource: privateKeySource,
                      registrationMirror: registrationMirror)
    }

    private static let currentStorageDirectoryName = "Pipette"
    private static let legacyStorageDirectoryName = "EdgeEvals"

    /// The production storage: `Application Support/Pipette` for data and the
    /// system caches directory for re-downloadable content.
    /// The one store that reads the Keychain.
    static let production = FileStorage(
        dataRoot: applicationSupportRoot.appendingPathComponent(currentStorageDirectoryName, isDirectory: true),
        cacheRoot: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0],
        privateKeySource: { KeychainHelper.loadPrivateKey() },
        registrationMirror: .keychain
    )

    /// Create the directory (and any intermediate directories) if it doesn't
    /// exist, then return it. Collapses the create-then-return dance that every
    /// directory accessor repeated.
    static func ensureDirectory(_ url: URL) -> URL {
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static var applicationSupportRoot: URL {
        // `.applicationSupportDirectory` in `.userDomainMask` is guaranteed to
        // return a URL on iOS, so `.first!` can't trap here.
        FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
    }

    private var legacyDataRoot: URL? {
        let legacyRoot = dataRoot
            .deletingLastPathComponent()
            .appendingPathComponent(Self.legacyStorageDirectoryName, isDirectory: true)
        guard legacyRoot.standardizedFileURL != dataRoot.standardizedFileURL else {
            return nil
        }
        return legacyRoot
    }

    private var legacyCachesRoot: URL? {
        let legacyRoot = cacheRoot.appendingPathComponent(Self.legacyStorageDirectoryName, isDirectory: true)
        guard legacyRoot.standardizedFileURL != dataRoot.standardizedFileURL else {
            return nil
        }
        return legacyRoot
    }

    @discardableResult
    func cleanupLegacyEdgeEvalsStorage() -> [URL] {
        let fm = FileManager.default
        let roots = [legacyDataRoot, legacyCachesRoot].compactMap { $0 }
        var removed: [URL] = []

        for root in roots where fm.fileExists(atPath: root.path) {
            do {
                try fm.removeItem(at: root)
                removed.append(root)
            } catch {
                AppLog.storage.error("Failed to remove legacy storage at \(root.path): \(error.localizedDescription)")
            }
        }

        return removed
    }

    /// Where the identity files lived before they moved to `identity/`. Read by
    /// `migrateIdentityDirectory()` and nothing else — do not add callers.
    var legacyMetadataDir: URL {
        dataRoot.appendingPathComponent("metadata", isDirectory: true)
    }

    /// Move `metadata/{registration,settings}.json` into `identity/`, the name the CLI
    /// uses. One-shot and idempotent: a file already at the destination wins, and the
    /// old directory goes once it is empty.
    ///
    /// A move rather than a re-read at the old path because the alternative is carrying
    /// two locations forever — and losing the registration here would push a registered
    /// device back through setup with a fresh keypair, orphaning its submitted results.
    @discardableResult
    func migrateIdentityDirectory() -> [String] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: legacyMetadataDir.path) else { return [] }
        let destination = Self.ensureDirectory(
            dataRoot.appendingPathComponent("identity", isDirectory: true))

        var moved: [String] = []
        for name in ["registration.json", "settings.json"] {
            let from = legacyMetadataDir.appendingPathComponent(name)
            let to = destination.appendingPathComponent(name)
            guard fm.fileExists(atPath: from.path) else { continue }
            guard !fm.fileExists(atPath: to.path) else {
                try? fm.removeItem(at: from)
                continue
            }
            do {
                try fm.moveItem(at: from, to: to)
                moved.append(name)
            } catch {
                AppLog.storage.error("identity migration: \(name) did not move: \(error)")
            }
        }
        // Only when nothing is left — an unexpected file there is someone else's.
        if let rest = try? fm.contentsOfDirectory(atPath: legacyMetadataDir.path), rest.isEmpty {
            try? fm.removeItem(at: legacyMetadataDir)
        }
        if !moved.isEmpty {
            AppLog.storage.info("identity migration: moved \(moved.joined(separator: ", "))")
        }
        return moved
    }

    var jobsDir: URL {
        Self.ensureDirectory(dataRoot.appendingPathComponent("jobs", isDirectory: true))
    }

    @MainActor
    func resetDeviceData() throws {
        let fm = FileManager.default
        let removableDirectories = [
            dataRoot.appendingPathComponent("jobs", isDirectory: true),
            dataRoot.appendingPathComponent("results", isDirectory: true),
            dataRoot.appendingPathComponent("models", isDirectory: true),
            dataRoot.appendingPathComponent("benchmarks", isDirectory: true),
            legacyDataRoot,
            legacyCachesRoot
        ].compactMap { $0 }

        for directory in removableDirectories where fm.fileExists(atPath: directory.path) {
            try fm.removeItem(at: directory)
        }
    }

}

/// Root-independent storage utilities: model/benchmark metadata parsing, the
/// repo-namespaced path rule, and the shared preference. These need no data root,
/// so they stay a stateless namespace rather than `Storage` members.
nonisolated enum LocalStorage {
    static let defaultContributeResultsKey = "defaultContributeResults"
    static let lastReconciledCollectorKey = "lastReconciledCollector"
    /// When true, `PlannerWorker` claims jobs from the management server.
    static let plannerWorkerEnabledKey = "plannerWorkerEnabled"
    /// When true, the user turned analytics off in Settings. See ``analyticsOptOut``.
    static let analyticsOptOutKey = "analyticsOptOut"

    /// The collector our results were last reconciled against (see
    /// `SettingsView.resendResultsFromOtherCollectors`). Lets a refresh skip the
    /// whole-store scan when the configured collector hasn't changed since.
    /// Nil until the first successful reconcile.
    static var lastReconciledCollector: String? {
        get { UserDefaults.standard.string(forKey: lastReconciledCollectorKey) }
        set { UserDefaults.standard.set(newValue, forKey: lastReconciledCollectorKey) }
    }

    @discardableResult
    static func markExcludedFromBackup(_ url: URL) -> Bool {
        var mutableURL = url
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        do {
            try mutableURL.setResourceValues(resourceValues)
            return true
        } catch {
            return false
        }
    }

    /// The analytics opt-out, and the authority on it. `Analytics.start()` seeds
    /// `PostHogConfig.optOut` from this before `setup`, and the Settings toggle reads it back.
    ///
    /// Deliberately **not** delegated to the SDK's own persisted copy, even though PostHog has one
    /// and restores it in `setup`. On posthog-android 3.19.0 that restore was measured not to work
    /// at all: with the flag stored, `isOptOut()` read `false` right after `setup` and the next
    /// `app_launched` was captured and accepted by the ingest endpoint (see Android's
    /// `AnalyticsOptOutStore` for the full write-up). A value seeded into the config *before*
    /// `setup` does hold, on both platforms. This client has not been measured either way, so it
    /// takes the same defensive shape rather than trusting a mechanism already caught failing once.
    ///
    /// Opt-**out**: absent means collecting, matching the SDK's default and Android's store.
    static var analyticsOptOut: Bool {
        get { UserDefaults.standard.bool(forKey: analyticsOptOutKey) }
        set { UserDefaults.standard.set(newValue, forKey: analyticsOptOutKey) }
    }

    static var defaultContributeResults: Bool {
        get {
            // Opt-out: contribution is on by default (the payload is performance-only,
            // never personal/device data) and the user can turn it off in Settings or
            // per-job. Kept in sync with SettingsView's @AppStorage default.
            if UserDefaults.standard.object(forKey: defaultContributeResultsKey) == nil {
                return true
            }
            return UserDefaults.standard.bool(forKey: defaultContributeResultsKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: defaultContributeResultsKey)
        }
    }

    /// Planner claim-loop opt-in. Off by default — the device only pulls server
    /// jobs when the user enables it in Settings.
    static var plannerWorkerEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: plannerWorkerEnabledKey) }
        set { UserDefaults.standard.set(newValue, forKey: plannerWorkerEnabledKey) }
    }

    // MARK: - Per-download keys

    /// The stable per-transfer download key — `<hf_repo>/<filename>` when the repo is
    /// known, else just `<filename>`, so two repos publishing the same GGUF filename
    /// get distinct rows. Not a storage path: a model's on-disk location is its
    /// `ModelStorageKey` entry, and a vision model is two download rows in one entry.
    static func modelRelativePath(repo: String?, filename: String) -> String {
        if let repo, !repo.isEmpty { return "\(repo)/\(filename)" }
        return filename
    }

    // MARK: - Model / benchmark metadata helpers

    /// Q*, F*, BF*, IQ* tokens identify gguf quant/precision suffixes.
    static func isQuantToken(_ s: String) -> Bool {
        let u = s.uppercased()
        return u.hasPrefix("Q") || u.hasPrefix("F") || u.hasPrefix("BF") || u.hasPrefix("IQ")
    }

    /// Extract a quant token (e.g. `Q4_K_M`, `F16`, `BF16`, `IQ3_XXS`) from a
    /// gguf filename. Returns nil if no recognized quant token is found.
    static func parseQuant(from filename: String) -> String? {
        let stem = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        for part in stem.split(separator: "-").reversed() where isQuantToken(String(part)) {
            return String(part)
        }
        return nil
    }

    /// Strip `.gguf` and any trailing quant suffix, preserving original case:
    /// `LFM2.5-350M-Q4_0.gguf` → `LFM2.5-350M`.
    static func modelStem(from filename: String) -> String {
        let stem = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        let parts = stem.split(separator: "-")
        if let last = parts.last, isQuantToken(String(last)) {
            return parts.dropLast().joined(separator: "-")
        }
        return stem
    }

    /// A `<number><M|B>` token (e.g. `350M`, `1.5B`, `0.8B`) identifies a
    /// parameter-count tag. Case-insensitive on the unit; the numeric part may
    /// carry a decimal point. Rejects version-style tokens like `2.5` (no unit).
    static func isParamToken(_ s: String) -> Bool {
        guard let unit = s.last, "MmBb".contains(unit) else { return false }
        let num = s.dropLast()
        return num.contains(where: \.isNumber)
            && num.allSatisfy { $0.isNumber || $0 == "." }
    }

    /// Extract a parameter-size label (e.g. `350M`, `1.5B`) from a model's repo
    /// or filename. Prefers the HF repo (more canonical) then the filename,
    /// scanning `-`/`/`-delimited tokens. Returns the tag uppercased, or nil if
    /// none is found. This is the param count, not the on-disk byte size.
    static func parseParamSize(name: String, hfRepo: String?) -> String? {
        for source in [hfRepo, name].compactMap({ $0 }) {
            let stem = source.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
            for part in stem.split(whereSeparator: { $0 == "-" || $0 == "/" }) where isParamToken(String(part)) {
                return String(part).uppercased()
            }
        }
        return nil
    }

    /// Lowercased `modelStem`, additionally stripping a leading `mmproj-`.
    /// Used to pair base models with their mmproj files across quant variants.
    static func normalizedModelStem(_ filename: String) -> String {
        var base = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        if base.lowercased().hasPrefix("mmproj-") {
            base = String(base.dropFirst("mmproj-".count))
        }
        let parts = base.split(separator: "-")
        if let last = parts.last, isQuantToken(String(last)) {
            return parts.dropLast().joined(separator: "-").lowercased()
        }
        return base.lowercased()
    }

    /// Recover a stale `storedPath` against a specific `modelsRoot` (the
    /// absolute-path existence check is `FileStorage.resolveModelPath`'s).
    static func resolveModelPath(_ storedPath: String, modelsRoot: URL) -> String? {
        let fm = FileManager.default
        let comps = storedPath.split(separator: "/").map(String.init)
        guard let filename = comps.last, !filename.isEmpty else { return nil }
        // Preferred: reuse the stored `…/models/<entry…>/<file>` tail so the file
        // resolves into the same entry under the new modelsDir.
        if let modelsIdx = comps.lastIndex(of: "models"), modelsIdx + 1 < comps.count {
            let tail = comps[(modelsIdx + 1)...].joined(separator: "/")
            let candidate = modelsRoot.appendingPathComponent(tail).path
            if fm.fileExists(atPath: candidate) { return candidate }
        }
        // Fallback: locate the file by name under modelsRoot. If two repos hold
        // the same filename, refuse to guess — returning the other repo's bytes
        // would defeat the namespacing and mislabel the result's provenance.
        let matches = modelURLs(named: filename, under: modelsRoot)
        return matches.count == 1 ? matches[0].path : nil
    }

    /// All `*.gguf` named `filename` found anywhere under `root`.
    private static func modelURLs(named filename: String, under root: URL) -> [URL] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return [] }
        var matches: [URL] = []
        for case let url as URL in enumerator
        where url.pathExtension.lowercased() == "gguf" && url.lastPathComponent == filename {
            matches.append(url)
        }
        return matches
    }
}

// `DiscoveredModel` (the discovered-model presentation type) lives in
// Contracts/DiscoveredModel.swift.
