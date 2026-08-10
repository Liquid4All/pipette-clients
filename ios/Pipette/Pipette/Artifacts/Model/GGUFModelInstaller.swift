import Foundation

/// Installs a finished single-file (GGUF) download: relocate the weight file into
/// `blobs/` inside the model's store entry, then write the entry manifest that makes
/// the on-disk copy self-describing (the next disk scan rebuilds the model from it).
///
/// A vision model's weights and projector are two transfers that key to the *same*
/// entry, so both land in one `blobs/` and each writes the same manifest — whichever
/// finishes first publishes the entry, which is what keeps a sweep from reclaiming a
/// half-downloaded VL model as manifest-less garbage.
@MainActor
struct GGUFModelInstaller: ModelInstaller {
    let repo: String?
    let filename: String
    /// The typed definition the file was ignited from, when known.
    let source: Model?
    let storage: Storage

    /// The coordinate this file's entry is keyed on. Prefers the exact `source`
    /// captured at ignition; a sideload or a record predating `source` reconstructs a
    /// single-file text model from the repo + filename strings. Nil means there is no
    /// entry to place the file in, so the download must be refused rather than
    /// dropped somewhere the scanner will never find it.
    static func entrySpec(repo: String?, filename: String, source: Model?) -> Model? {
        switch source {
        case .ggufText, .ggufVision:
            return source
        case .mlx, .appleFoundationText:
            return nil  // not a single-file GGUF transfer
        case nil:
            guard let repoSlug = repo,
                  let parsedRepo = try? HFRepo.parse(repoSlug),
                  let parsedFilename = try? RepoSubpath(filename)
            else { return nil }
            return .ggufText(GgufText(source: .huggingFace(
                repo: parsedRepo, path: parsedFilename, sha256: nil)))
        }
    }

    var entrySpec: Model? { Self.entrySpec(repo: repo, filename: filename, source: source) }

    func install(from downloaded: URL) throws {
        guard let spec = entrySpec,
              storage.modelStore.prepareEntryDir(for: spec) != nil,
              let blobs = storage.modelStore.blobsDir(for: spec)
        else { throw DownloadError.io("no model coordinate to store \(filename) under") }

        let fm = FileManager.default
        let dest = blobs.appendingPathComponent(filename)
        do {
            // The parent, not `blobs`: a repo-relative path may name a subdirectory
            // (`quants/m.gguf`), and the crate stores it nested — `to_stored` joins the
            // whole path under the entry and takes the leaf only for local sources.
            try fm.createDirectory(at: dest.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
        } catch {
            throw DownloadError.io("\(error)")
        }
        try? fm.removeItem(at: dest)
        try fm.moveItem(at: downloaded, to: dest)
        storage.modelStore.publishManifest(for: spec)
    }
}
