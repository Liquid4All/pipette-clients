import Foundation
import Testing

@testable import Pipette

/// Canonical (key-sorted) `model_descriptor` / `runtime_descriptor` wire values for an MLX
/// cell, shared so hand-built payload fixtures don't drift across suites.
enum SubmissionFixtures {
    static let mlxModelDescriptor =
        #"{"org":"LiquidAI","repo_name":"LFM2.5-8B-MLX-4bit","source":"huggingface","type":"mlx"}"#
    static let mlxRuntimeDescriptor =
        #"{"flavor":"ios-arm64","packages":{"mlx_swift":{"repository_url":"github.com/ml-explore/mlx-swift","repository_version":"0.31.6"}},"type":"mlx_ios_pipette"}"#
}

/// A `PipetteWorkspace` under a fresh temporary directory, so suites share no global and
/// run in parallel. Pair with `removeWorkspace` in a `defer`.
nonisolated func makeTemporaryWorkspace() -> PipetteWorkspace {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("PipetteTests-\(UUID().uuidString)", isDirectory: true)
    return PipetteWorkspace(
        root: root.appendingPathComponent("data", isDirectory: true),
        cacheRoot: root.appendingPathComponent("cache", isDirectory: true)
    )
}

/// Remove the temporary tree (both `root` and `cacheRoot` live under it).
nonisolated func removeWorkspace(_ ws: PipetteWorkspace) {
    try? FileManager.default.removeItem(at: ws.root.deletingLastPathComponent())
}

/// A minimal submittable payload for the cell, by default `remotePending`: the result a
/// sweep picks up.
nonisolated func writePayload(
    ws: PipetteWorkspace,
    cellId: CellId,
    benchmarkId: String,
    at location: BenchmarkResultLocation = .remotePending
) throws {
    // Both descriptors: the submit path refuses a payload without them
    // (`requireDescriptors`).
    let data = try JSONSerialization.data(withJSONObject: [
        "cell_id": cellId.value,
        "benchmark_id": benchmarkId,
        "model_descriptor": #"{"type":"gguf_text"}"#,
        "runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#
    ])
    try ws.results.saveResult(
        location, cellId, payload: data, extras: Data("{}".utf8))
}

func cell(
    id: CellId,
    benchmarkId: String,
    source: Model = ggufTextFixture("test/model-GGUF", "model.gguf"),
    benchmarkSource: BenchmarkSource? = nil
) -> JobCell {
    JobCell(
        cellId: id,
        benchmarkId: benchmarkId,
        benchmarkType: BenchmarkType(rawValue: benchmarkId),
        runStatus: .completed,
        serverJobId: nil,
        errorMessage: nil,
        source: source,
        benchmarkSource: benchmarkSource
    )
}

/// A fixed phone identity. Literal rather than `detectDeviceInfo()`, so wire-shape
/// assertions don't change with the host the tests run on.
func deviceInfoFixture(
    formFactor: DeviceFormFactor = .phone,
    osBuild: String? = "22F76",
    chipModel: String = "Apple A19 Pro"
) -> DeviceInfo {
    DeviceInfo(
        deviceName: "iPhone 17 Pro",
        deviceFormFactor: formFactor,
        deviceOsName: "iOS",
        deviceOsVersion: "26.4",
        deviceOsBuild: osBuild,
        deviceOsSecurityPatch: nil,
        deviceCpuArch: "arm64",
        deviceOsFamily: "iOS",
        deviceChipModel: chipModel,
        deviceRamBytes: 8_589_934_592,
        deviceGpuModel: nil,
        deviceGpuVramBytes: nil,
        deviceNpuModel: nil,
        deviceNpuVramBytes: nil
    )
}

func powerStateFixture(
    batteryLevel: Int32? = 82,
    powerState: DevicePowerState? = .notCharging,
    powerSaveMode: Bool = false
) -> PowerReading {
    PowerReading(batteryLevel: batteryLevel, powerState: powerState, powerSaveMode: powerSaveMode)
}

/// The collector every fixture registers against. Throws, so a fixture URL that
/// ``ServerUrl`` stops accepting fails the test that uses it.
func registrationData() throws -> IdentityRegistration {
    IdentityRegistration(
        clientId: try ClientId("client-1"),
        status: "approved",
        serverUrl: try ServerUrl("https://collector.example.com"),
        organization: "Example",
        contactEmail: "user@example.com",
        registeredAt: "2026-05-28T16:41:00Z",
        clerkUserId: "user_1",
        clerkSessionId: "session_1",
        clerkPrimaryEmail: "user@example.com",
        clerkLinkedAt: "2026-05-28T16:42:00Z"
    )
}

/// A `POST /benchmarks/batch` reply decoded through the client's own type, so a fixture
/// cannot describe a response the real decoder would reject.
func batchResponse(_ results: [[String: Any]]) throws -> BatchSubmitResponse {
    let data = try JSONSerialization.data(withJSONObject: ["results": results])
    return try Coding.decoder.decode(BatchSubmitResponse.self, from: data)
}

/// A real 32-byte seed, because ``PrivateKey`` refuses anything that cannot sign.
let testPrivateKeyHex = String(repeating: "a3f8", count: 16)

/// The identity a fixture signs as.
func testIdentity(clientId: String = "client-1") throws -> ClientIdentity {
    ClientIdentity(clientId: try ClientId(clientId), privateKeyHex: try PrivateKey(testPrivateKeyHex))
}

/// A `RunRequest` for a payload test: the cell identity `buildSubmissionPayloadFromRun`
/// reads off the request, so a test states the model and runtime once instead of unpacking
/// them.
nonisolated func payloadRequest(
    model: DeclaredBound<Model>,
    benchmarkId: String = "decode_throughput_512_100",
    benchmark: BenchmarkDefinition? = nil
) -> RunRequest {
    RunRequest(
        runtime: .alreadyBound(Runtime.thisBuild(for: model.declared)),
        model: model,
        runtimeFlags: nil,
        benchmarkFlags: nil,
        benchmark: benchmark ?? .decodeThroughput(DecodeThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: benchmarkId),
            parameterPrefillTokens: 512,
            parameterDecodeTokens: 100)))
}

/// The GGUF-text model the payload tests run, as the declared/bound pair a prepared request
/// carries: the plan coordinate plus the same spec rewritten to its `absoluteFile` arm.
nonisolated func ggufTextResolved(
    _ repo: String = "LiquidAI/LFM2-350M-GGUF",
    _ file: String = "LFM2-350M-Q4_K_M.gguf",
    path: String = "/tmp/LFM2-350M-Q4_K_M.gguf"
) throws -> DeclaredBound<Model> {
    DeclaredBound(declared: try ggufTextSpec(repo, file),
                  bound: .ggufText(.init(source: .absoluteFile(path: try AbsolutePath(path)))))
}

/// A `BenchmarkStore` under a fresh temporary directory, as the crate's `temp_store`.
/// Pair with `removeStore`.
nonisolated func makeTemporaryBenchmarkStore() -> BenchmarkStore {
    BenchmarkStore(root: FileManager.default.temporaryDirectory
        .appendingPathComponent("PipetteBenchmarks-\(UUID().uuidString)", isDirectory: true))
}

nonisolated func removeStore(_ store: BenchmarkStore) {
    try? FileManager.default.removeItem(at: store.root)
}

/// Caching on the remote half as a pull or a claim does, like the crate's `cache_remote`.
nonisolated extension BenchmarkStore {
    func cacheRemote(json: String, etag: String? = nil) throws {
        try putRemote(ConvertedRemote(
            Coding.decoder.decode(RemoteBenchmark.self, from: Data(json.utf8))),
            etag: etag.map { try EntityTag($0) })
    }

    func cacheRemote(_ definition: BenchmarkDefinition, etag: String? = nil) throws {
        try putRemote(ConvertedRemote(
            Coding.decoder.decode(RemoteBenchmark.self, from: Coding.encoder.encode(definition))),
            etag: etag.map { try EntityTag($0) })
    }

    func remoteDetailPath(_ id: String) -> URL {
        root.appendingPathComponent("remote/detail/\(id).json")
    }

    func remoteEtagPath(_ id: String) -> URL {
        root.appendingPathComponent("remote/detail/\(id).etag")
    }

    /// Where a pull keeps the list and the sync state.
    func statePath(_ name: String) -> URL {
        root.appendingPathComponent("remote/state/\(name)")
    }

    /// The bookkeeping a build before `remote/state/` keeps, at its paths: the list rows it
    /// converted, and its sync state with ETags.
    func writeOlderBuildBookkeeping(
        rows: [String], server: String, listEtag: String, benchmarkEtags: [String: String]
    ) throws {
        let remote = root.appendingPathComponent("remote", isDirectory: true)
        try FileManager.default.createDirectory(at: remote, withIntermediateDirectories: true)
        try Data("[\(rows.joined(separator: ","))]".utf8)
            .write(to: remote.appendingPathComponent("index.json"))
        let state: [String: Any] = [
            "pulled_at": "2026-10-18T00:00:00Z", "benchmark_count": rows.count,
            "server_url": server, "benchmarks_etag": listEtag, "benchmark_etags": benchmarkEtags,
        ]
        try JSONSerialization.data(withJSONObject: state)
            .write(to: remote.appendingPathComponent("sync.json"))
    }

    /// What a build before `parameter_num_images` cached for ``img80Id``: no image count, so
    /// it reads as one image.
    @discardableResult
    func writeLegacyImg80() throws -> URL {
        let path = root.appendingPathComponent("remote/\(img80Id).json")
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"""
            {
              "benchmark_id" : "vl_throughput_512x512_text1024_decode64_img80",
              "benchmark_type" : "vl_throughput",
              "parameter_decode_tokens" : 64,
              "parameter_image_height" : 512,
              "parameter_image_width" : 512,
              "parameter_text_tokens" : 1024
            }
            """#.utf8).write(to: path)
        return path
    }
}

nonisolated let img80Id = "vl_throughput_512x512_text1024_decode64_img80"

nonisolated let img80Wire = #"{"benchmark_id":"vl_throughput_512x512_text1024_decode64_img80","benchmark_type":"vl_throughput","parameter_image_width":512,"parameter_image_height":512,"parameter_text_tokens":1024,"parameter_decode_tokens":64,"parameter_num_images":80}"#

/// Compares two documents by value, whatever their key order or number spelling.
nonisolated func jsonDocument(_ data: Data) throws -> NSDictionary {
    try #require(JSONSerialization.jsonObject(with: data) as? NSDictionary)
}
