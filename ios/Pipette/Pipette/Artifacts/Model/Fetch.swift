import CryptoKit
import Foundation

// The planning, install and sizing ends of a model fetch, after
// `crates/pipette-artifacts/src/model/fetch.rs`. The crate's `fetch_model` also moves the
// bytes; here `DownloadCoordinator` owns the transfer (background sessions, pause and
// resume), so this file keeps the pure planner, the installers and the size probes.

/// The completion step every finished download goes through: validate, relocate into
/// the store entry, publish the manifest. The work belongs off the main actor: a GGUF
/// install hashes the file, and a directory install validates and moves the whole package.
protocol ModelInstaller: Sendable {
    func install(from downloaded: URL) throws
}

/// A vision model's weights and projector are two transfers into the same entry, and
/// each writes the same manifest: whichever finishes first publishes the entry, so a
/// sweep never reclaims a half-downloaded VL model as manifest-less garbage.
@MainActor
struct GGUFModelInstaller: ModelInstaller {
    let repo: String?
    let filename: String
    let source: Model?
    /// Nil verifies nothing, as the crate's `None` does.
    var sha256: Sha256?
    let ws: PipetteWorkspace

    /// Prefers the `source` captured at ignition; a sideload or a record predating
    /// `source` is rebuilt as a single-file text model from repo and filename. Nil means
    /// there is no entry to place the file in, so the download must be refused.
    nonisolated static func entrySpec(repo: String?, filename: String, source: Model?) -> Model? {
        switch source {
        case .ggufText, .ggufVision, .ggufVisionSpeculative:
            return source
        case .mlx, .onnxGenai, .appleFoundationText:
            return nil  // not a single-file GGUF transfer
        case nil:
            guard let repoSlug = repo,
                  let parsedRepo = try? HfRepo.parseOrgRepo(repoSlug),
                  let parsedFilename = try? RepoSubpath(filename)
            else { return nil }
            return .ggufText(GgufText(source: .huggingFace(
                repo: parsedRepo, path: parsedFilename, sha256: nil)))
        }
    }

    nonisolated var entrySpec: Model? { Self.entrySpec(repo: repo, filename: filename, source: source) }

    /// Where the transfer named `leaf` belongs inside `spec`'s entry: the bound path whose
    /// last component is `leaf`, the same rule the manifest's `stored` records. A repo path
    /// may be nested (`quants/m.gguf`) while a transfer knows only the leaf, and installing
    /// under the leaf alone would leave the entry a husk that downloads again.
    ///
    /// Falls back to `blobs/<leaf>` when there is no bound form (a sideload or a spec
    /// rebuilt from strings). Two vision files sharing a leaf would be ambiguous, but they
    /// already collide on the transfer key and the crate's `to_stored` refuses them.
    ///
    /// Static so `DownloadCoordinator`'s pre-transfer existence check agrees with the install.
    nonisolated static func destination(for spec: Model, leaf: String, under blobs: URL,
                            in store: ModelArtifactStore) -> URL {
        let flat = blobs.appendingPathComponent(leaf)
        guard let paths = store.fetchDestination(for: spec)?.boundPaths else { return flat }
        let match = [paths.payload, paths.mmproj, paths.draft]
            .compactMap { $0 }
            .first { ($0 as NSString).lastPathComponent == leaf }
        return match.map { URL(fileURLWithPath: $0) } ?? flat
    }

    /// Must run synchronously inside the delegate callback: `URLSession` deletes its temp
    /// file when the callback returns. The expensive part waits for `finish(adopted:)`.
    ///
    /// Moves into `models/.staging`, which the quota sweep reclaims, so a process that
    /// dies between the two halves leaves an orphan the sweep removes, not a leak.
    nonisolated func adopt(from downloaded: URL) throws -> URL {
        guard let spec = entrySpec else {
            throw DownloadError.io("no model coordinate to store \(filename) under")
        }
        let scratch = ws.models.stagingRoot
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            let held = scratch.appendingPathComponent((filename as NSString).lastPathComponent)
            try FileManager.default.moveItem(at: downloaded, to: held)
            _ = spec
            return held
        } catch {
            throw DownloadError.io("\(error)")
        }
    }

    /// Verifies the digest at the last point before the bytes land, as the crate does.
    /// Hashing a multi-gigabyte file takes seconds, so the caller owes this a detached
    /// task: `nonisolated` permits that, but a synchronous call runs where it is called.
    ///
    /// Not staged-and-published like a directory model: a vision model's two files finish
    /// independently, so there is no single attempt to publish.
    nonisolated func finish(adopted: URL) throws {
        guard let spec = entrySpec,
              ws.models.prepareEntryDir(for: spec) != nil,
              let blobs = ws.models.blobsDir(for: spec)
        else { throw DownloadError.io("no model coordinate to store \(filename) under") }

        let fm = FileManager.default
        let dest = Self.destination(for: spec, leaf: filename, under: blobs,
                                    in: ws.models)
        defer { try? fm.removeItem(at: adopted.deletingLastPathComponent()) }
        try ModelFetch.verifySha256(of: adopted, against: sha256, landingAt: dest.path)
        do {
            try fm.createDirectory(at: dest.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
        } catch {
            throw DownloadError.io("\(error)")
        }
        try? fm.removeItem(at: dest)
        try fm.moveItem(at: adopted, to: dest)
        ws.models.publishManifest(for: spec)
    }

    /// Both halves in one call, for a caller with no main actor to keep free.
    nonisolated func install(from downloaded: URL) throws {
        try finish(adopted: try adopt(from: downloaded))
    }
}

/// Installs an MLX bundle or ONNX Runtime GenAI package, from a Hub pull or an unpacked
/// vault archive: validate, stage, publish. Only validation differs per format.
/// `nonisolated` so it runs off the main actor: ONNX validation walks the decoder graph
/// and the publish moves a multi-gigabyte directory.
nonisolated struct DirectoryModelInstaller: ModelInstaller, Sendable {
    let source: Model
    let ws: PipetteWorkspace
    /// Recorded in the staged attempt so a later launch knows which commit the bytes
    /// belong to.
    var resolved: ResolvedTransfer?

    /// An ONNX package is held to its own `genai_config.json`, including the external-data
    /// sidecar a truncated transfer loses.
    nonisolated func validate(_ downloaded: URL) throws {
        switch source {
        case .mlx:
            try MlxModelLayout.validate(downloaded, name: source.description)
        case .onnxGenai:
            try OnnxGenaiModels.validatePackage(at: downloaded.path)
        case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText:
            throw DownloadError.io("\(source.artifactName) is not a directory model")
        }
    }

    nonisolated func install(from downloaded: URL) throws {
        do {
            try validate(downloaded)
        } catch {
            // Drop it so the next attempt re-fetches rather than resuming garbage.
            try? FileManager.default.removeItem(at: downloaded)
            throw error
        }
        try publish(downloaded)
    }

    /// The tree lands at `blobs/` (whole repo or vault archive) or `blobs/<prefix>` (repo
    /// subtree), as the crate lays it out. Goes through `stage` / `publish`, so a failure
    /// part-way leaves the old entry in place.
    nonisolated func publish(_ downloaded: URL) throws {
        let spec = source
        let store = ws.models
        var staged = try store.stage(spec)
        if let resolved { staged = store.recordResolution(staged, resolved) }
        do {
            // Bound against the attempt's own `blobs/`, so a subtree keeps its prefix.
            guard let stagedForm = spec.withoutAuthToken.toStored(base: Entry.blobsDirName),
                  let bundle = stagedForm.underRoot(staged.root)?.boundPaths?.payload
            else { throw DownloadError.io("no model coordinate to store \(spec) under") }
            let dest = URL(fileURLWithPath: bundle, isDirectory: true)
            let fm = FileManager.default
            do {
                try fm.createDirectory(at: dest.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                // For a whole-repo model the bundle is the empty `blobs/` that `stage`
                // made, so it must go before the move. It is the attempt's own scratch.
                try? fm.removeItem(at: dest)
                try fm.moveItem(at: downloaded, to: dest)
            } catch {
                throw DownloadError.io("\(error)")
            }
            // The store holds either the previous entry or this one, never neither.
            try store.publish(staged)
        } catch {
            store.discard(staged)
            throw error
        }
    }
}

// MARK: - Planning

/// The crate's `ModelFetchError`, narrowed to what the pure planner and the digest check
/// can say; transfer failures arrive from `URLSession` as `URLError`.
nonisolated enum ModelFetchError: Error, Equatable, LocalizedError {
    /// The message is a complete explanation, as the crate's is.
    case notFetchable(String)
    case invalidUrl(String)
    case sha256Mismatch(dest: String, expected: String, actual: String)

    var errorDescription: String? {
        switch self {
        case let .notFetchable(why): why
        case let .invalidUrl(url): "invalid download URL: \(url)"
        case let .sha256Mismatch(dest, expected, actual):
            "sha256 mismatch for \(dest): expected \(expected), got \(actual)"
        }
    }
}

extension ModelFetch {
    /// The crate's `Download`. Nested because `DownloadCoordinator.Download` already owns
    /// the bare name for a transfer row.
    nonisolated struct Download: Hashable, Sendable {
        let url: ResourceUrl
        let auth: AuthToken?
        let sha256: Sha256?
        /// Not read by the transfer: `URLSession` picks its own temporary path, and
        /// `GGUFModelInstaller` re-derives this from the store at install time with the
        /// same rule, so the digest error names the same path. Kept because the crate's
        /// record carries it, and for callers that plan without enqueueing.
        let dest: AbsolutePath
    }

    /// The file parts of the crate's `parts`, landing at the `absolute*` paths in `into`
    /// (the store's bound form). Local files and Apple Foundation are refused rather than
    /// planned: here a path is already-installed bytes and Apple Foundation is never fetched.
    ///
    /// Pure (no network or filesystem), so every HTTP decision is testable without a server.
    nonisolated static func planDownloads(_ declared: Model, into: Model) throws -> [Download] {
        switch (declared, into) {
        case let (.ggufText(d), .ggufText(i)):
            return try ggufTextDownloads(declared: d.source, into: i.source)
        case let (.ggufVision(d), .ggufVision(i)):
            return try ggufVisionDownloads(declared: d.source, into: i.source)
        case let (.ggufVisionSpeculative(d), .ggufVisionSpeculative(i)):
            return try ggufVisionSpeculativeDownloads(declared: d.source, into: i.source)
        // The crate plans a directory as a snapshot part; here the HubApi pull lists and
        // fetches it itself.
        case (.mlx, _), (.onnxGenai, _):
            throw ModelFetchError.notFetchable(
                "directory snapshots need a network repo listing; planned by the fetcher, not here")
        case (.appleFoundationText, _):
            throw ModelFetchError.notFetchable("the Apple Foundation model has no fetchable files")
        default:
            throw ModelFetchError.notFetchable("declared and localized model kinds disagree")
        }
    }

    /// The crate's `gguf_text_downloads`.
    private nonisolated static func ggufTextDownloads(
        declared: GgufTextSource, into: GgufTextSource
    ) throws -> [Download] {
        let dest: AbsolutePath
        switch into {
        case let .absoluteFile(path):
            dest = path
        case .relativeFile:
            throw ModelFetchError.notFetchable("gguf-text fetch dest must be abs Local, got Relative")
        case .huggingFace, .url, .vaultUrl:
            throw ModelFetchError.notFetchable("bound gguf-text is not an Absolute source")
        }
        switch declared {
        case let .huggingFace(repo, path, sha256):
            return [Download(url: try hfFileUrl(repo: repo, path: path),
                             auth: repo.authToken, sha256: sha256, dest: dest)]
        case let .url(url, sha256):
            return [Download(url: url, auth: nil, sha256: sha256, dest: dest)]
        case .vaultUrl:
            throw ModelFetchError.notFetchable(Self.vaultIsNotPlanned)
        case .absoluteFile, .relativeFile:
            throw ModelFetchError.notFetchable("local/relative gguf-text is imported, not downloaded")
        }
    }

    /// The crate's `gguf_vision_downloads`: weights then projector.
    private nonisolated static func ggufVisionDownloads(
        declared: GgufVisionSource, into: GgufVisionSource
    ) throws -> [Download] {
        guard case let .absoluteFiles(modelDest, mmprojDest) = into else {
            throw ModelFetchError.notFetchable("gguf-vision fetch dest must be abs Local")
        }
        switch declared {
        case let .huggingFace(repo, model, modelSha256, mmproj, mmprojSha256):
            return [
                Download(url: try hfFileUrl(repo: repo, path: model),
                         auth: repo.authToken, sha256: modelSha256, dest: modelDest),
                Download(url: try hfFileUrl(repo: repo, path: mmproj),
                         auth: repo.authToken, sha256: mmprojSha256, dest: mmprojDest),
            ]
        case let .url(model, modelSha256, mmproj, mmprojSha256):
            return [
                Download(url: model, auth: nil, sha256: modelSha256, dest: modelDest),
                Download(url: mmproj, auth: nil, sha256: mmprojSha256, dest: mmprojDest),
            ]
        case .vaultUrl:
            throw ModelFetchError.notFetchable(Self.vaultIsNotPlanned)
        case .absoluteFiles, .relativeFiles:
            throw ModelFetchError.notFetchable("local/relative gguf-vision is imported, not downloaded")
        }
    }

    /// The vision pair (via `ggufVisionDownloads`, so the two cannot drift) plus a separate
    /// drafter. A merged drafter is the weights, already in the pair.
    private nonisolated static func ggufVisionSpeculativeDownloads(
        declared: GgufVisionSpeculativeSource, into: GgufVisionSpeculativeSource
    ) throws -> [Download] {
        var downloads = try ggufVisionDownloads(declared: declared.vision, into: into.vision)
        guard case let .absoluteFiles(_, _, draftDest) = into else {
            throw ModelFetchError.notFetchable(
                "gguf-vision-speculative fetch dest must be abs Local")
        }
        switch (declared, draftDest) {
        case let (.huggingFace(repo, _, _, _, _, draft?, draftSha256), dest?):
            downloads.append(Download(url: try hfFileUrl(repo: repo, path: draft),
                                      auth: repo.authToken, sha256: draftSha256, dest: dest))
        case let (.url(_, _, _, _, draft?, draftSha256), dest?):
            downloads.append(Download(url: draft, auth: nil, sha256: draftSha256, dest: dest))
        // Separate and merged drafters are different measurements, so a mismatch between
        // spec and binding is refused rather than read as the other.
        case (.huggingFace(_, _, _, _, _, .some, _), nil),
             (.url(_, _, _, _, .some, _), nil):
            throw ModelFetchError.notFetchable(
                "the model names a separate drafter but bound no path for it")
        case (.huggingFace(_, _, _, _, _, nil, _), .some),
             (.url(_, _, _, _, nil, _), .some):
            throw ModelFetchError.notFetchable(
                "a drafter path was bound for a model that names no separate drafter")
        case (.huggingFace(_, _, _, _, _, nil, _), nil), (.url(_, _, _, _, nil, _), nil):
            break
        case (.vaultUrl, _):
            throw ModelFetchError.notFetchable(Self.vaultIsNotPlanned)
        case (.absoluteFiles, _), (.relativeFiles, _):
            throw ModelFetchError.notFetchable(
                "local/relative gguf-vision-speculative is imported, not downloaded")
        }
        return downloads
    }

    /// Vault files go out as signed requests (see `vaultFiles`), not URL plans.
    private nonisolated static let vaultIsNotPlanned =
        "a vault file is fetched with a signed request, not planned as a URL download"

    nonisolated struct VaultFile: Hashable, Sendable {
        let path: VaultPath
        let pin: Sha256?
    }

    /// Weights, projector, then a separate drafter for a `vault_url` gguf; the one archive
    /// for a `vault_archive` directory model; `nil` for every other arm.
    nonisolated static func vaultFiles(_ declared: Model) -> [VaultFile]? {
        switch declared {
        case let .ggufText(m):
            guard case let .vaultUrl(url, sha256, _, _) = m.source else { return nil }
            return [VaultFile(path: url, pin: sha256)]
        case let .ggufVision(m):
            guard case let .vaultUrl(model, modelSha, mmproj, mmprojSha, _, _) = m.source
            else { return nil }
            return [VaultFile(path: model, pin: modelSha), VaultFile(path: mmproj, pin: mmprojSha)]
        case let .ggufVisionSpeculative(m):
            guard case let .vaultUrl(model, modelSha, mmproj, mmprojSha, draft, draftSha, _, _)
                    = m.source
            else { return nil }
            return [VaultFile(path: model, pin: modelSha), VaultFile(path: mmproj, pin: mmprojSha)]
                + (draft.map { [VaultFile(path: $0, pin: draftSha)] } ?? [])
        case let .mlx(m):
            guard case let .vaultArchive(url, sha256, _, _) = m.source else { return nil }
            return [VaultFile(path: url, pin: sha256)]
        case let .onnxGenai(m):
            guard case let .vaultArchive(url, sha256, _, _) = m.source else { return nil }
            return [VaultFile(path: url, pin: sha256)]
        case .appleFoundationText:
            return nil
        }
    }

    /// The crate's `hf_file_url`. The size probe uses the same `hfResolveUrl`, so the
    /// bytes sized are the bytes fetched.
    private nonisolated static func hfFileUrl(repo: HfRepo, path: RepoSubpath) throws -> ResourceUrl {
        let raw = hfResolveUrl(repo: repo, path: path.value)
        guard let url = URL(string: raw), let resource = try? ResourceUrl(url.absoluteString)
        else { throw ModelFetchError.invalidUrl(raw) }
        return resource
    }

    // MARK: Digest

    /// Read in chunks so a multi-GB weight file never buffers in memory.
    nonisolated static func sha256Hex(ofFileAt file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// A nil `expected` verifies nothing, as in the crate.
    nonisolated static func verifySha256(of file: URL, against expected: Sha256?,
                                         landingAt dest: String) throws {
        guard let expected else { return }
        let actual = try sha256Hex(ofFileAt: file)
        guard actual == expected.value else {
            throw ModelFetchError.sha256Mismatch(dest: dest, expected: expected.value, actual: actual)
        }
    }
}

// MARK: - Sizing

/// The namespace for the crate's `model/fetch.rs` names: the planner and digest check
/// (the extension above) and, here, sizing for admission, asked before a fetch starts.
enum ModelFetch {
    /// The crate's `hf_resolve_url`; an unpinned repo resolves against `main`.
    nonisolated static func hfResolveUrl(repo: HfRepo, path: String) -> String {
        hfResolveUrl(repo: repo, revision: repo.revision?.value ?? "main", path: path)
    }

    /// For a caller holding a resolved commit. One spelling of the resolve URL, so what a
    /// planner HEADs and what a transfer fetches cannot diverge.
    nonisolated static func hfResolveUrl(repo: HfRepo, revision: String,
                                         path: String) -> String {
        "https://huggingface.co/\(repo.description)/resolve/\(revision)/\(path)"
    }

    /// The crate's `declared_size_bytes`: `Content-Length` for single files, the Hub's
    /// blob listing for a directory. `nil` means nobody could say, which admission refuses.
    static func declaredSizeBytes(_ declared: Model, http: HttpClient) async -> Int64? {
        switch declared {
        case .ggufText, .ggufVision, .ggufVisionSpeculative:
            return await remoteFilesSizeBytes(declared, http: http)
        case let .mlx(m):
            guard let ref = m.ref else { return nil }
            return await hfDirSizeBytes(ref, of: declared, http: http)
        case let .onnxGenai(m):
            guard let ref = m.ref else { return nil }
            return await hfDirSizeBytes(ref, of: declared, http: http)
        case .appleFoundationText:
            return 0  // ships with the OS: nothing lands in the store
        }
    }

    /// One unsized file leaves the whole unsized: a partial total could pass an artifact
    /// that does not fit.
    private static func remoteFilesSizeBytes(_ declared: Model, http: HttpClient) async -> Int64? {
        let repo: HfRepo
        let paths: [RepoSubpath]
        switch declared {
        case let .ggufText(m):
            guard case let .huggingFace(r, path, _) = m.source else { return nil }
            (repo, paths) = (r, [path])
        case let .ggufVision(m):
            guard case let .huggingFace(r, weights, _, projector, _) = m.source else { return nil }
            (repo, paths) = (r, [weights, projector])
        case let .ggufVisionSpeculative(m):
            // A merged drafter is inside the weights already counted.
            guard case let .huggingFace(r, weights, _, projector, _, draft, _) = m.source
            else { return nil }
            (repo, paths) = (r, [weights, projector] + (draft.map { [$0] } ?? []))
        case .mlx, .onnxGenai, .appleFoundationText:
            return nil
        }
        var total: Int64 = 0
        for path in paths {
            guard let url = URL(string: hfResolveUrl(repo: repo, path: path.value)),
                  let bytes = await contentLength(url, of: declared, http: http)
            else { return nil }
            total = saturating(total, bytes)
        }
        return total
    }

    /// The crate's `content_length`. `http.client` follows the resolve URL's redirect to
    /// the CDN, so this is the blob's length, not the redirect's.
    private static func contentLength(_ url: URL, of declared: Model, http: HttpClient) async -> Int64? {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        attachHfAuth(&request, claimToken: declared.repo?.authToken, model: declared)
        guard let (_, response) = await send(request, sizing: "\(url)", http: http) else {
            return nil
        }
        return response.value(forHTTPHeaderField: "Content-Length").flatMap { Int64($0) }
    }

    /// The crate's `hf_dir_size_bytes`, but filtered by the same globs HubApi matches
    /// (`fnmatch`, flags 0) rather than by prefix: an MLX root pull fetches only the MLX
    /// file set, and counting a repo's GGUF quants would refuse bytes that never land.
    private static func hfDirSizeBytes(_ ref: HfModelRef, of declared: Model,
                                       http: HttpClient) async -> Int64? {
        guard let info = await repoInfo(ref, of: declared, http: http) else { return nil }
        let globs = ref.globs(for: ModelType.of(declared))
        let fetched = info.siblings.filter { sibling in
            globs.contains { fnmatch($0, sibling.rfilename, 0) == 0 }
        }
        // An empty match is "no such subtree", not a zero-byte one.
        guard !fetched.isEmpty else { return nil }
        var total: Int64 = 0
        for sibling in fetched {
            // A listing that omits one file's size cannot bound the pull.
            guard let size = sibling.size else { return nil }
            total = saturating(total, size)
        }
        return total
    }

    /// The crate's `repo_info` with blobs: `GET /api/models/{repo}/revision/{rev}?blobs=true`.
    private static func repoInfo(_ ref: HfModelRef, of declared: Model,
                                 http: HttpClient) async -> HfModelInfo? {
        // One path component, as HubApi encodes it: a revision may hold `/` (`refs/pr/1`).
        let revision = (ref.repo.revision?.value ?? "main")
            .addingPercentEncoding(withAllowedCharacters: pathComponentAllowed) ?? "main"
        guard let url = URL(string: "https://huggingface.co/api/models/\(ref.repo)"
                                + "/revision/\(revision)?blobs=true")
        else { return nil }
        var request = URLRequest(url: url)
        attachHfAuth(&request, claimToken: ref.repo.authToken, model: declared)
        guard let (data, _) = await send(request, sizing: ref.key, http: http) else { return nil }
        do {
            return try JSONDecoder().decode(HfModelInfo.self, from: data)
        } catch {
            warnUnsized(ref.key, reason: "unreadable listing: \(error)")
            return nil
        }
    }

    /// Logs why and returns nil rather than throwing. Unlike the crate, where a `nil` size
    /// only skips the pre-flight, admission here refuses it.
    private static func send(_ request: URLRequest, sizing what: String,
                             http: HttpClient) async -> (Data, HTTPURLResponse)? {
        let reason: String
        do {
            let (data, response) = try await http.client.data(for: request)
            if let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) {
                return (data, http)
            }
            reason = "HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)"
        } catch {
            reason = error.localizedDescription
        }
        warnUnsized(what, reason: reason)
        return nil
    }

    /// The figures are the server's, so an overflow must not trap; `.max` fails any quota.
    nonisolated static func saturating(_ a: Int64, _ b: Int64) -> Int64 {
        let (sum, overflow) = a.addingReportingOverflow(b)
        return overflow ? .max : sum
    }

    private static func warnUnsized(_ what: String, reason: String) {
        AppLog.storage.warning(
            "sizing \(what) failed (\(reason)); the storage quota cannot be checked before the fetch")
    }

    private static let pathComponentAllowed = CharacterSet.urlPathAllowed
        .subtracting(CharacterSet(charactersIn: "/"))

    private nonisolated struct HfModelInfo: Decodable {
        let siblings: [HfSibling]
    }

    /// `size` is absent without `blobs=true`, and on a file the Hub has not sized.
    private nonisolated struct HfSibling: Decodable {
        let rfilename: String
        let size: Int64?
    }
}
