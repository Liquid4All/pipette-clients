import Foundation
import Hub

/// Fetches one MLX model directory, hiding swift-transformers' `HubApi` behind a
/// seam so the coordinator/UI depend on an interface (and tests inject a fake, no
/// network). GGUF stays on the single-file `URLSession` path; only MLX directory
/// pulls route here.
protocol MLXModelDownloading: Sendable {
    /// Download `ref` and return the on-disk model directory. Reports an aggregate
    /// fraction `0...1` (any thread). Throws `DownloadError.cancelled` on task
    /// cancellation and `.transport` on a Hub failure; resumes a partial pull from
    /// cache rather than restarting.
    ///
    /// `token` is the caller-resolved HF credential (see `resolveHfToken`); `nil` fetches
    /// anonymously, correct only for a public repo.
    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL
}

nonisolated struct HubMLXModelDownloader: MLXModelDownloading {
    /// Stable base for HubApi's cache + snapshot output; kept put across calls so a
    /// re-invoked pull skips already-fetched files. The caller relocates the finished
    /// model out of here into its store entry.
    let downloadBase: URL

    func download(_ ref: HFModelRef, token: AuthToken?,
                  progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        // Force online: a user-initiated download shouldn't be short-circuited to a
        // cache-only result by HubApi's offline-mode heuristic. Foreground session —
        // the download runs while the app is active (the user is watching).
        // `.value` only here, at the boundary that needs the raw string.
        let hub = HubApi(downloadBase: downloadBase, hfToken: token?.value,
                         useBackgroundSession: false, useOfflineMode: false)
        let root: URL
        do {
            root = try await hub.snapshot(from: ref.repo.description, matching: ref.globs) { p in
                progress(min(max(p.fractionCompleted, 0), 1))
            }
        } catch is CancellationError {
            throw DownloadError.cancelled
        } catch {
            throw DownloadError.transport("\(error)")
        }
        // An empty/partial result is caught by `MLXModelInstaller`'s validation gate.
        // HubApi mirrors each file at its repo-relative path, so a subpath model's
        // `config.json`/weights live under `<root>/<subpath>`.
        return ref.subpath.map { root.appending(path: $0.value) } ?? root
    }
}
