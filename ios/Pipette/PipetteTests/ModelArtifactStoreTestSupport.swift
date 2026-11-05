import Foundation
@testable import Pipette

// Fixtures for the model-store entry layout (`models/<key>/{manifest.json,
// blobs/…}`), shared by the store, layout and quota suites.

nonisolated func ggufTextSpec(_ repo: String, _ filename: String) throws -> Model {
    .ggufText(GgufText(source: .huggingFace(repo: try HfRepo.parseOrgRepo(repo), path: try RepoSubpath(filename), sha256: nil)))
}

/// Non-throwing `ggufTextSpec` for positions that cannot `try` (a stored property, a
/// default argument). The coordinates are literals, so validation could only succeed.
nonisolated func ggufTextFixture(_ repoSlug: String, _ filename: String) -> Model {
    let parts = repoSlug.split(separator: "/", maxSplits: 1)
    let org = HfOrg(validated: String(parts[0]))
    let name = HfRepoName(validated: String(parts.count > 1 ? parts[1] : parts[0]))
    return .ggufText(GgufText(source: .huggingFace(
        repo: HfRepo(org: org, repoName: name),
        path: RepoSubpath(validated: filename), sha256: nil)))
}

nonisolated func ggufVisionSpec(_ repo: String, _ filename: String, _ mmproj: String) throws -> Model {
    .ggufVision(GgufVision(source: .huggingFace(repo: try HfRepo.parseOrgRepo(repo), model: try RepoSubpath(filename), modelSha256: nil, mmproj: try RepoSubpath(mmproj), mmprojSha256: nil)))
}

/// A drafter-carrying vision spec. `draft` absent is the merged arrangement, a different
/// measurement rather than a missing file.
nonisolated func ggufVisionSpeculativeSpec(
    _ repo: String, _ filename: String, _ mmproj: String, draft: String? = nil
) throws -> Model {
    .ggufVisionSpeculative(GgufVisionSpeculative(source: .huggingFace(
        repo: try HfRepo.parseOrgRepo(repo),
        model: try RepoSubpath(filename), modelSha256: nil,
        mmproj: try RepoSubpath(mmproj), mmprojSha256: nil,
        draft: try draft.map { try RepoSubpath($0) }, draftSha256: nil)))
}

nonisolated func mlxSpec(_ repo: String, subpath: String? = nil) throws -> Model {
    .mlx(try HfModelRef.parse(repo: repo, subpath: subpath).asMlx())
}

nonisolated func onnxGenaiSpec(_ repo: String, subpath: String? = nil) throws -> Model {
    let ref = try HfModelRef.parse(repo: repo, subpath: subpath)
    return .onnxGenai(OnnxGenai(source: .huggingFace(repo: ref.repo, prefix: ref.subpath)))
}

/// Files a complete MLX bundle must carry for `MlxModelLayout` and discovery.
nonisolated let mlxBundleFiles = ["config.json", "model.safetensors", "tokenizer.json"]

/// The shape a complete ONNX Runtime GenAI package has on disk. Only the config is
/// load-bearing for the store, which keys and binds on it; `OnnxGenaiModels` is what holds
/// the package to what the config names.
nonisolated let onnxGenaiPackageFiles = ["genai_config.json", "model.onnx", "tokenizer.json"]

/// A manifest for `spec` as a publish records it: `stored` is the spec's store form under
/// its key's `blobsDir`, and `lastUsedAt` defaults to `fetchedAt`, as a publish seeds it.
nonisolated func manifestFixture(
    _ spec: Model, fetchedAt: Date = Date(), lastUsedAt: Date? = nil, blobsBytes: Int64? = nil
) throws -> ModelManifest {
    guard let key = ModelStorageKey.of(spec), let stored = spec.toStored(base: key.blobsDir) else {
        throw ModelError.unknownModelType("spec has no store form")
    }
    return ModelManifest(fetchedAt: fetchedAt, lastUsedAt: lastUsedAt ?? fetchedAt,
                         declared: spec, stored: stored, blobsBytes: blobsBytes)
}

/// Write a store entry the way the installers do: the manifest at the entry root plus
/// the payload the spec names under `blobs/`. Returns the entry directory.
@discardableResult
nonisolated func installEntry(
    _ ws: PipetteWorkspace,
    _ spec: Model,
    payloadBytes: Int = 1,
    lastUsedAt: Date = Date(),
    fetchedAt: Date = Date()
) throws -> URL {
    let fm = FileManager.default
    guard let key = ModelStorageKey.of(spec),
          let entry = ws.models.prepareEntryDir(for: spec),
          let paths = ws.models.fetchDestination(for: spec)?.boundPaths else {
        throw ModelError.unknownModelType("spec has no storage key")
    }
    let payload = Data(repeating: 0x67, count: payloadBytes)
    // Write where `stored` binds, so a fixture entry and a real install agree by
    // construction rather than by two copies of the layout rule.
    if case .onnxGenai = spec {
        let package = URL(fileURLWithPath: paths.payload, isDirectory: true)
        try fm.createDirectory(at: package, withIntermediateDirectories: true)
        for name in onnxGenaiPackageFiles {
            try payload.write(to: package.appendingPathComponent(name))
        }
    } else if case .mlx = spec {
        let bundle = URL(fileURLWithPath: paths.payload, isDirectory: true)
        try fm.createDirectory(at: bundle, withIntermediateDirectories: true)
        for name in mlxBundleFiles { try payload.write(to: bundle.appendingPathComponent(name)) }
    } else {
        for path in [paths.payload] + (paths.mmproj.map { [$0] } ?? []) {
            let url = URL(fileURLWithPath: path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try payload.write(to: url)
        }
    }
    // `stored` and `blobsBytes` as a real publish records them; `fetchedAt` / `lastUsedAt`
    // stay callable so the quota suite can age an entry.
    let blobs = ws.modelsDir.appendingPathComponent(key.blobsDir, isDirectory: true)
    try manifestFixture(spec, fetchedAt: fetchedAt, lastUsedAt: lastUsedAt,
                        blobsBytes: entrySizeBytes(at: blobs))
        .writeQuietly(to: ModelManifest.manifestURL(inEntryDir: entry))
    return entry
}

/// Put a single-file `declared`'s bytes where the store binds them and publish its
/// manifest, as the coordinator's installers do once a transfer lands: a test's fetch
/// for `ModelArtifactStore.ensure`.
nonisolated func landAndPublish(_ store: ModelArtifactStore, _ declared: Model) throws {
    guard let paths = store.fetchDestination(for: declared)?.boundPaths else {
        throw ModelError.unknownModelType("spec has no fetch target")
    }
    for path in [paths.payload] + (paths.mmproj.map { [$0] } ?? []) {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("gguf".utf8).write(to: url)
    }
    store.publishManifest(for: declared)
}

/// Rewrite an entry's manifest JSON in place, for a test that has to stage what no encode
/// produces: a tampered `stored`, a stray token, another version.
nonisolated func editManifest(_ entry: URL, _ edit: (inout [String: Any]) -> Void) throws {
    let url = ModelManifest.manifestURL(inEntryDir: entry)
    guard var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
        throw ModelError.unknownModelType("manifest is not a JSON object")
    }
    edit(&json)
    try JSONSerialization.data(withJSONObject: json).write(to: url)
}
