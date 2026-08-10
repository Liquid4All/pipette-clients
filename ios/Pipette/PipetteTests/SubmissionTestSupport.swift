import Foundation
@testable import Pipette

// Shared fixtures for the result-submission suites
// (ResultSubmissionServiceTests, ResultUploaderTests, UnsubmittedResultCountTests).

/// Canonical (key-sorted) plan-types descriptor strings for an MLX cell — the
/// opaque `model_descriptor` / `runtime_descriptor` wire values. Shared so the
/// hand-built payload fixtures don't drift across suites.
enum SubmissionFixtures {
    static let mlxModelDescriptor =
        #"{"org":"LiquidAI","repo_name":"LFM2.5-8B-MLX-4bit","source":"huggingface","type":"mlx"}"#
    static let mlxRuntimeDescriptor =
        #"{"flavor":"ios-arm64","packages":{"mlx_swift":{"repository_url":"github.com/ml-explore/mlx-swift","repository_version":"0.31.6"}},"type":"mlx_ios_pipette"}"#
}

/// A `FileStorage` rooted at a fresh temporary directory. Each test builds its
/// own and injects it into the unit under test, so suites carry no shared global
/// and run in parallel. Pair with `removeStorage` in a `defer`.
nonisolated func makeTemporaryStorage() -> FileStorage {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("PipetteTests-\(UUID().uuidString)", isDirectory: true)
    return FileStorage(
        dataRoot: root.appendingPathComponent("data", isDirectory: true),
        cacheRoot: root.appendingPathComponent("cache", isDirectory: true)
    )
}

/// Remove the temporary tree (both `dataRoot` and `cacheRoot` live under it).
nonisolated func removeStorage(_ storage: FileStorage) {
    try? FileManager.default.removeItem(at: storage.dataRoot.deletingLastPathComponent())
}

/// Write a minimal submittable payload for the cell, at `location`.
///
/// Defaults to `remotePending` — a result the sweep should pick up, which is what almost
/// every submission test is about.
nonisolated func writePayload(
    storage: Storage,
    cellId: CellId,
    benchmarkId: String,
    at location: BenchmarkResultLocation = .remotePending
) throws {
    // Both descriptors: the submit path refuses a payload without them
    // (`ResultSubmissionService.descriptorRefusal`).
    let data = try JSONSerialization.data(withJSONObject: [
        "cell_id": cellId.value,
        "benchmark_id": benchmarkId,
        "model_descriptor": #"{"type":"gguf_text"}"#,
        "runtime_descriptor": #"{"type":"llamacpp_ios_pipette"}"#
    ])
    try storage.results.saveResult(
        location, cellId, payload: data, extras: Data("{}".utf8))
}

func cell(
    id: CellId,
    benchmarkId: String,
    source: Model = ggufTextFixture("test/model-GGUF", "model.gguf")
) -> JobCell {
    JobCell(
        cellId: id,
        benchmarkId: benchmarkId,
        benchmarkType: BenchmarkType(rawValue: benchmarkId),
        runStatus: .completed,
        serverJobId: nil,
        errorMessage: nil,
        source: source
    )
}

/// A fixed phone identity. Literal rather than `DeviceProbe.detectDeviceInfo()`, so
/// wire-shape assertions don't change with the host the tests run on.
func deviceInfoFixture(
    formFactor: DeviceFormFactor = .phone,
    osBuild: String? = "22F76"
) -> DeviceInfo {
    DeviceInfo(
        deviceName: "iPhone 17 Pro",
        deviceFormFactor: formFactor,
        deviceOsName: "iOS",
        deviceOsVersion: "26.4",
        deviceOsBuild: osBuild,
        deviceOsSecurityPatch: nil,
        deviceChipModel: "Apple A19 Pro",
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
) -> PowerState {
    PowerState(batteryLevel: batteryLevel, powerState: powerState, powerSaveMode: powerSaveMode)
}

func registrationData() -> IdentityRegistration {
    IdentityRegistration(
        clientId: ClientID("client-1"),
        status: "approved",
        serverUrl: ServerURL("https://collector.example.com"),
        organization: "Example",
        contactEmail: "user@example.com",
        registeredAt: "2026-05-28T16:41:00Z",
        clerkUserId: "user_1",
        clerkSessionId: "session_1",
        clerkPrimaryEmail: "user@example.com",
        clerkLinkedAt: "2026-05-28T16:42:00Z"
    )
}

func batchResponse(_ results: [[String: Any]]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: ["results": results])
    return String(decoding: data, as: UTF8.self)
}

/// A `RunRequest` for a payload test: the cell identity `PayloadBuilder` reads off the
/// request, so a test states the model and runtime once instead of unpacking them.
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
        benchmark: benchmark ?? .decodeThroughput(benchmarkId: benchmarkId,
                                                  prefillTokens: 512, decodeTokens: 100))
}

/// The GGUF-text model those tests run, as the declared/bound pair a prepared request
/// carries: the plan coordinate plus the same spec rewritten to its `absoluteFile` arm.
nonisolated func ggufTextResolved(
    _ repo: String = "LiquidAI/LFM2-350M-GGUF",
    _ file: String = "LFM2-350M-Q4_K_M.gguf",
    path: String = "/tmp/LFM2-350M-Q4_K_M.gguf"
) throws -> DeclaredBound<Model> {
    DeclaredBound(declared: try ggufTextSpec(repo, file),
                  bound: .ggufText(.init(source: .absoluteFile(path: try AbsolutePath(path)))))
}

/// A `BenchmarkStore` rooted at a fresh temporary directory — the crate's `temp_store`.
/// The store is concrete, so tests point one at scratch space rather than substituting a
/// fake. Pair with `removeStore`.
nonisolated func makeTemporaryBenchmarkStore() -> BenchmarkStore {
    BenchmarkStore(root: FileManager.default.temporaryDirectory
        .appendingPathComponent("PipetteBenchmarks-\(UUID().uuidString)", isDirectory: true))
}

nonisolated func removeStore(_ store: BenchmarkStore) {
    try? FileManager.default.removeItem(at: store.root)
}
