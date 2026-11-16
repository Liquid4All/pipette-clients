import Foundation
import Hub
import HuggingFace

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
    /// HubApi's own sessions take no `Host`, so a proxied pull runs on `http` (`ProxiedHub`).
    var proxy: CdnProxy? = CdnProxy.launch
    var http: HttpClient = .shared

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
        let proxied = proxy.map { ProxiedHub(proxy: $0, token: token, http: http) }
        // Everything below addresses the resolved commit, not the declared revision.
        let globs = ref.globs(for: ModelType.of(model))
        let source: any HubMetadataSource =
            if let proxied { proxied } else { HubApiMetadataSource(hub: hub) }
        let resolved = try await HubTransferPlanner.resolve(ref, globs: globs, using: source)
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
        var resumeData: [String: Data] = [:]
        do {
            try await Self.fetchEach(
                resolved.files, totalBytes: resolved.totalBytes,
                url: { ModelFetch.hfResolveUrl(repo: ref.repo, revision: resolved.commit,
                                                     path: $0.path) },
                progress: progress
            ) { file, fileProgress in
                if let proxied {
                    try await proxied.fetch(file, commit: resolved.commit, repo: ref.repo,
                                            into: root, hub: hub, resume: &resumeData[file.path],
                                            progress: fileProgress)
                    return
                }
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
            // A `URLError`'s `userInfo` holds its resume data, which holds the request and token.
            throw DownloadError.transport(error is URLError ? errorChain(error) : "\(error)")
        }
        // Kept out of the installed payload; a pull that stops short leaves it for
        // `resetStaging`.
        try? FileManager.default.removeItem(at: modelDir.appending(path: Self.revisionMarker))
        // `DirectoryModelInstaller` validates an empty or partial result.
        return TransferOutcome(directory: modelDir, resolved: resolved)
    }

    /// Fetch `files` one at a time, each with its own retries, reporting the fraction of
    /// `totalBytes` done. A dropped connection restarts only the file in progress: from
    /// zero through HubApi (PIP-704 adds a per-file `Range` resume), from where it stopped
    /// through the proxy.
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

/// A directory pull through `cdn-proxy=` on the app's own session: the planner's listing and
/// `HEAD`s, then each file, every one the request HubApi sends, routed with `Host` naming the
/// Hub. The proxy's resolve answers carry the Hub's `X-Repo-Commit`, `X-Linked-Etag` and
/// `X-Linked-Size`.
nonisolated struct ProxiedHub: HubMetadataSource {
    let proxy: CdnProxy
    /// The proxy's access check needs it.
    let token: AuthToken?
    let http: HttpClient

    func filenames(repo: HfRepo, revision: String,
                   matching globs: [String]) async throws -> [String] {
        // One path component, as HubApi encodes it: a revision may hold `/` (`refs/pr/1`).
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/revision")?
            .appending(component: revision) else {
            throw TransferPlanningError.metadataUnavailable(
                reason: "could not build a listing URL for \(repo)")
        }
        let (data, _) = try await retrying(
            url: url.absoluteString, attempt: { _ in try await send("GET", url) },
            classify: Disposition.init(ofHub:))
        // HubApi's selection: `fnmatch` with flags 0, so `*` crosses `/`.
        return try JSONDecoder().decode(ModelFetch.HfModelInfo.self, from: data).siblings
            .map(\.rfilename)
            .filter { name in globs.contains { fnmatch($0, name, 0) == 0 } }
    }

    func metadata(repo: HfRepo, revision: String, path: String) async throws -> HubFileMetadata {
        guard let url = URL(string: ModelFetch.hfResolveUrl(
            repo: repo, revision: revision, path: path)) else {
            throw TransferPlanningError.metadataUnavailable(
                reason: "could not build a resolve URL for \(path)")
        }
        let (_, found) = try await retrying(
            url: url.absoluteString, attempt: { _ in try await send("HEAD", url) },
            classify: Disposition.init(ofHub:))
        let etag = found.value(forHTTPHeaderField: "X-Linked-Etag")?
            .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        return HubFileMetadata(
            sizeBytes: found.value(forHTTPHeaderField: "X-Linked-Size").flatMap { Int64($0) },
            commitHash: found.value(forHTTPHeaderField: "X-Repo-Commit"), etag: etag,
            sha256: HubApiMetadataSource.sha256Reference(fromHubETag: etag))
    }

    /// One attempt at `file`, landing where `HubApi.snapshot` lands it, with its metadata, so
    /// a later pull through HubApi or the proxy takes it as fetched. A broken transfer leaves
    /// its resume data in `resume`, and the next attempt continues the file with `Range`.
    func fetch(_ file: PlannedFile, commit: String, repo: HfRepo, into root: URL, hub: HubApi,
               resume: inout Data?, progress: @escaping @Sendable (Double) -> Void) async throws {
        let destination = root.appending(path: file.path)
        // Where `snapshot` keeps it.
        let metadata = root.appending(path: ".cache/huggingface/download/\(file.path).metadata")
        if FileManager.default.fileExists(atPath: destination.path),
           try hub.readDownloadMetadata(metadataPath: metadata)?.commitHash == commit {
            return
        }
        guard let url = URL(string: ModelFetch.hfResolveUrl(
            repo: repo, revision: commit, path: file.path)) else {
            throw DownloadError.transport("could not build a resolve URL for \(file.path)")
        }
        let watch = TaskProgress(progress)
        let location: URL
        let response: URLResponse
        do {
            if let partial = resume {
                (location, response) = try await http.clientNoRedirects.download(
                    resumeFrom: partial, delegate: watch)
            } else {
                (location, response) = try await http.clientNoRedirects.download(
                    for: request("GET", url), delegate: watch)
            }
        } catch {
            resume = (error as? URLError)?.downloadTaskResumeData
            throw error
        }
        resume = nil
        defer { try? FileManager.default.removeItem(at: location) }
        _ = try check(response, "GET", url)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: location, to: destination)
        // HubApi records no file without one either, so a later pull fetches it again.
        if let etag = file.etag {
            try hub.writeDownloadMetadata(commitHash: commit, etag: etag, metadataPath: metadata)
        }
    }

    /// On the session that follows no redirect: the next hop would keep this `Host`.
    private func send(_ method: String, _ url: URL) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await http.clientNoRedirects.data(for: request(method, url))
        return (data, try check(response, method, url))
    }

    private func request(_ method: String, _ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let token { request.setValue("Bearer \(token.value)", forHTTPHeaderField: "Authorization") }
        return proxy.route(request)
    }

    /// Thrown as HubClient throws it, so `Disposition(ofHub:)` reads the status and
    /// `Retry-After`; its text is the status and the URL only.
    private func check(_ response: URLResponse, _ method: String,
                       _ url: URL) throws -> HTTPURLResponse {
        guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
        guard (200..<300).contains(http.statusCode) else {
            throw HTTPClientError.responseError(response: http, detail: "\(method) \(url.absoluteString)")
        }
        return http
    }
}

/// A transfer's fraction as its bytes land. `download(for:delegate:)` hands its delegate the
/// task, but on some SDKs not the byte counts, so this watches the task's own `progress`.
///
/// `@unchecked Sendable`: `observation` is set once, when the task is created, before it runs.
private nonisolated final class TaskProgress: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let report: @Sendable (Double) -> Void
    private var observation: NSKeyValueObservation?

    init(_ report: @escaping @Sendable (Double) -> Void) {
        self.report = report
    }

    func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
        observation = task.progress.observe(\.fractionCompleted) { [report] progress, _ in
            report(progress.fractionCompleted)
        }
    }
}
