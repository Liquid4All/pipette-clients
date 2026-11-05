import Foundation
import Hub

/// Fetches one directory model (MLX or ONNX Runtime GenAI), hiding `HubApi` behind a seam
/// that tests fake. GGUF stays on the single-file `URLSession` path.
protocol ModelTransport: Sendable {
    /// Download the subtree `model` names. Reports a fraction `0...1` from any thread.
    /// Throws `DownloadError.cancelled` on cancellation and `.transport` once retries run
    /// out. A new call keeps the files a previous pull finished.
    ///
    /// Takes the model, not just the ref, because the root globs depend on the format.
    /// `token` is the caller-resolved credential; `nil` works only for a public repo.
    ///
    /// `admit` gets the plan's true total once, before any payload byte moves: the caller's
    /// reservation came from a declared size. Throwing from it abandons the pull.
    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome
}

/// Where the bytes are, plus the plan the transport froze, which only it knows and the
/// staged attempt must record.
nonisolated struct TransferOutcome: Sendable {
    let directory: URL
    /// `nil` for a transport that freezes nothing, such as a test stub.
    let resolved: ResolvedTransfer?
}

nonisolated struct HubModelTransport: ModelTransport {
    /// Stable across calls so a re-invoked pull skips files already fetched.
    let downloadBase: URL

    /// Names the revision a staged model directory was fetched at; see `resetStaging`.
    static let revisionMarker = ".pipette-revision"

    func download(_ model: Model, ref: HfModelRef, token: AuthToken?,
                  admit: @escaping @Sendable (Int64) async throws -> Void,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> TransferOutcome {
        // Offline mode off so HubApi's heuristic cannot short-circuit to a cache-only
        // result. `cache: nil`: the default `HubCache` keeps a second copy of every file
        // in the shared HF cache, which holds an evicted model's bytes until a sweep. It
        // buys no mid-file resume either: its partial blob only grows by a completed
        // range response.
        let hub = HubApi(downloadBase: downloadBase, cache: nil, hfToken: token?.value,
                         useBackgroundSession: false, useOfflineMode: false)
        // Everything below addresses the resolved commit, not the declared revision.
        let globs = ref.globs(for: ModelType.of(model))
        let resolved = try await HubTransferPlanner.resolve(ref, globs: globs,
                                                            using: HubApiMetadataSource(hub: hub))
        try await admit(resolved.totalBytes)
        // HubApi mirrors repo-relative paths, so a subpath model lives under
        // `<root>/<subpath>`.
        let root = hub.localRepoLocation(Hub.Repo(id: ref.repo.description))
        let modelDir = ref.subpath.map { root.appending(path: $0.value) } ?? root
        do {
            try Self.resetStaging(modelDir, revision: resolved.commit)
        } catch {
            throw DownloadError.io("\(error)")
        }
        do {
            try await Self.fetchEach(
                resolved.files, totalBytes: resolved.totalBytes,
                url: { ModelFetch.hfResolveUrl(repo: ref.repo, revision: resolved.commit,
                                                     path: $0.path) },
                progress: progress
            ) { file, fileProgress in
                // TODO: PIP-704. `snapshot` lists the repo again per file and per retry, and
                // a `429` there loses its `Retry-After`. Fetching from `resolved.files`
                // needs a per-file downloader; swift-transformers keeps `HubFileDownloader`
                // internal.
                _ = try await hub.snapshot(from: ref.repo.description, revision: resolved.commit,
                                           matching: [Self.exactGlob(file.path)]) { p in
                    fileProgress(p.fractionCompleted)
                }
            }
        } catch is CancellationError {
            throw DownloadError.cancelled
        } catch {
            throw DownloadError.transport("\(error)")
        }
        // Kept out of the installed payload; a pull that stops short leaves it for
        // `resetStaging`.
        try? FileManager.default.removeItem(at: modelDir.appending(path: Self.revisionMarker))
        // `DirectoryModelInstaller` validates an empty or partial result.
        return TransferOutcome(directory: modelDir, resolved: resolved)
    }

    /// Fetch `files` one at a time, each with its own retries, reporting the fraction of
    /// `totalBytes` done. A dropped connection restarts only the file in progress, from
    /// zero (PIP-704 adds a per-file `Range` resume).
    ///
    /// `HubApi.snapshot` can return early without throwing on cancellation, hence the
    /// explicit check after each file.
    static func fetchEach(
        _ files: [PlannedFile],
        totalBytes: Int64,
        url: (PlannedFile) -> String,
        policy: RetryPolicy = .download,
        hooks: RetryHooks = .live,
        progress: @escaping @Sendable (Double) -> Void,
        fetch: (PlannedFile, _ progress: @escaping @Sendable (Double) -> Void) async throws -> Void
    ) async throws {
        var done: Int64 = 0
        for file in files {
            let before = done
            try await retrying(
                policy, url: url(file),
                attempt: { _ in
                    try await fetch(file) { fraction in
                        let bytes = Double(before) + Double(file.sizeBytes) * min(max(fraction, 0), 1)
                        progress(totalBytes > 0 ? min(bytes / Double(totalBytes), 1) : 1)
                    }
                    try Task.checkCancellation()
                },
                classify: Disposition.init(ofHub:),
                hooks: hooks)
            done += file.sizeBytes
        }
    }

    /// A glob that matches `path` and nothing else under `fnmatch`. Escapes by Unicode
    /// scalar: a `Character` can join a `[` to a following combining mark, and `fnmatch`
    /// still reads that `[` as a pattern.
    static func exactGlob(_ path: String) -> String {
        path.unicodeScalars.reduce(into: "") { glob, scalar in
            if "\\*?[]".unicodeScalars.contains(scalar) { glob.append("\\") }
            glob.unicodeScalars.append(scalar)
        }
    }

    /// HubApi stages every revision of a repo at one path and resumes from leftovers, so
    /// drop the model directory unless it was staged at `revision`, the **resolved
    /// commit** (two pulls of `main` can resolve differently).
    static func resetStaging(_ modelDir: URL, revision: String) throws {
        let fm = FileManager.default
        let marker = modelDir.appending(path: revisionMarker)
        if fm.fileExists(atPath: modelDir.path),
           (try? String(contentsOf: marker, encoding: .utf8)) != revision {
            try fm.removeItem(at: modelDir)
        }
        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try Data(revision.utf8).write(to: marker)
    }
}

/// `HubApi` as the planner's metadata source; see `HubMetadataSource` for why it is two
/// calls.
nonisolated struct HubApiMetadataSource: HubMetadataSource {
    let hub: HubApi

    /// An LFS file's ETag is its SHA-256; a Git-backed file's is the shorter Git object id.
    /// Only the former parses as a digest.
    static func sha256Reference(fromHubETag etag: String?) -> Sha256? {
        guard let etag else { return nil }
        return try? Sha256(etag.lowercased())
    }

    func filenames(repo: HfRepo, revision: String,
                   matching globs: [String]) async throws -> [String] {
        try await retrying(
            url: "https://huggingface.co/api/models/\(repo.description)/revision/\(revision)",
            attempt: { _ in
                try await hub.getFilenames(from: Hub.Repo(id: repo.description),
                                           revision: revision, matching: globs)
            },
            classify: Disposition.init(ofHub:))
    }

    func metadata(repo: HfRepo, revision: String, path: String) async throws -> HubFileMetadata {
        // `ModelFetch`'s builder, not HubApi's (its `hostURL` is private): one spelling of
        // the resolve URL app-wide, so a HEAD and a GET cannot address different things.
        guard let url = URL(string: ModelFetch.hfResolveUrl(
            repo: repo, revision: revision, path: path)) else {
            throw TransferPlanningError.metadataUnavailable(
                reason: "could not build a resolve URL for \(path)")
        }
        let found = try await retrying(
            url: url.absoluteString,
            attempt: { _ in try await hub.getFileMetadata(url: url) },
            classify: Disposition.init(ofHub:))
        return HubFileMetadata(sizeBytes: found.size.map(Int64.init),
                               commitHash: found.commitHash, etag: found.etag,
                               sha256: Self.sha256Reference(fromHubETag: found.etag))
    }
}
