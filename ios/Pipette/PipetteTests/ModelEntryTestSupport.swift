import Foundation
@testable import Pipette

// Fixtures for the model-store entry layout (`models/<key>/{manifest.json,
// blobs/…}`), shared by the layout, quota, and touch suites.

nonisolated func ggufTextSpec(_ repo: String, _ filename: String) throws -> Model {
    .ggufText(GgufText(source: .huggingFace(repo: try HFRepo.parse(repo), path: try RepoSubpath(filename), sha256: nil)))
}

/// Non-throwing `ggufTextSpec` for fixtures in positions that cannot `try` — a stored
/// property or a default argument. The coordinates here are literals, so the validating
/// initializers would only ever succeed; this states that rather than forcing it.
nonisolated func ggufTextFixture(_ repoSlug: String, _ filename: String) -> Model {
    let parts = repoSlug.split(separator: "/", maxSplits: 1)
    let org = HFOrg(validated: String(parts[0]))
    let name = HFRepoName(validated: String(parts.count > 1 ? parts[1] : parts[0]))
    return .ggufText(GgufText(source: .huggingFace(
        repo: HFRepo(org: org, repoName: name),
        path: RepoSubpath(validated: filename), sha256: nil)))
}

nonisolated func ggufVisionSpec(_ repo: String, _ filename: String, _ mmproj: String) throws -> Model {
    .ggufVision(GgufVision(source: .huggingFace(repo: try HFRepo.parse(repo), model: try RepoSubpath(filename), modelSha256: nil, mmproj: try RepoSubpath(mmproj), mmprojSha256: nil)))
}

nonisolated func mlxSpec(_ repo: String, subpath: String? = nil) throws -> Model {
    .mlx(try HFModelRef.parse(repo: repo, subpath: subpath).asMlx())
}

/// Files a complete MLX bundle must carry for `MLXModelLayout` and discovery.
nonisolated let mlxBundleFiles = ["config.json", "model.safetensors", "tokenizer.json"]

/// Write a store entry the way the installers do: the manifest at the entry root plus
/// the payload the spec names under `blobs/`. Returns the entry directory.
@discardableResult
nonisolated func installEntry(
    _ storage: FileStorage,
    _ spec: Model,
    payloadBytes: Int = 1,
    lastUsedAt: Date = Date(),
    fetchedAt: Date? = Date()
) throws -> URL {
    let fm = FileManager.default
    guard let entry = storage.modelStore.prepareEntryDir(for: spec),
          let blobs = storage.modelStore.blobsDir(for: spec) else {
        throw ModelError.unknownModelType("spec has no storage key")
    }
    try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
    let payload = Data(repeating: 0x67, count: payloadBytes)
    // A repo-relative path may name a subdirectory, and the store nests it.
    func writePayload(_ relative: String) throws {
        let dest = blobs.appendingPathComponent(relative)
        try fm.createDirectory(at: dest.deletingLastPathComponent(),
                               withIntermediateDirectories: true)
        try payload.write(to: dest)
    }
    // Write where `toStored` says the files go, so a fixture entry and a real install
    // agree by construction rather than by two copies of the layout rule.
    let stored = spec.toStored(base: Entry.blobsDirName)
    switch stored {
    case let .ggufText(m):
        guard case let .relativeFile(path) = m.source else { break }
        try writePayload(String(path.value.dropFirst(Entry.blobsDirName.count + 1)))
    case let .ggufVision(m):
        guard case let .relativeFiles(model, mmproj) = m.source else { break }
        try writePayload(String(model.value.dropFirst(Entry.blobsDirName.count + 1)))
        try writePayload(String(mmproj.value.dropFirst(Entry.blobsDirName.count + 1)))
    case .mlx:
        for name in mlxBundleFiles { try payload.write(to: blobs.appendingPathComponent(name)) }
    case .appleFoundationText, nil:
        break
    }
    // `stored` and `blobsBytes` as a real publish records them; `fetchedAt` / `lastUsedAt`
    // stay callable so the quota suite can age an entry.
    ModelManifest(declared: spec, fetchedAt: fetchedAt, lastUsedAt: lastUsedAt,
                  blobsBytes: DiskUsage.bytes(at: blobs), stored: stored)
        .writeQuietly(to: ModelManifest.manifestURL(inEntryDir: entry))
    return entry
}
