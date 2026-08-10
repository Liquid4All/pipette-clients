import Foundation
import Testing

@testable import Pipette

/// `RunCell.prepare` — the one place a `RunRequest` is assembled, so what it fixes about a
/// cell is asserted here rather than at each run site.
///
/// `.serialized`: these share a temporary `dataRoot` and the coordinator's process-global
/// background session.
@Suite(.serialized) @MainActor struct RunCellTests {

    /// The cell a plan would send, so a test names it once rather than restating the axes
    /// at every `prepare`.
    private func spec(_ model: Model, _ benchmark: BenchmarkDefinition,
                      numberGpuLayers: UInt32? = 99, ctxSize: UInt32? = 4096,
                      nUbatch: UInt32? = 512, threads: UInt32? = nil,
                      swaFull: Bool? = nil) -> ClientRunSpec {
        ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: model,
            runtime: Runtime.thisBuild(for: model),
            runtimeFlags: RuntimeFlagRef(
                benchmarkType: benchmark.type,
                runtimeType: RuntimeType.of(Runtime.thisBuild(for: model)),
                modelType: ModelType.of(model),
                numberGpuLayers: numberGpuLayers, ctxSize: ctxSize, nUbatch: nUbatch,
                threads: threads, swaFull: swaFull))
    }

    /// An installed model resolves to its store paths, and both runtime halves are this
    /// build — a cell declares no runtime this client can be, so there is no second value
    /// to carry.
    @Test func anInstalledModelBindsToItsStorePathsAndThisBuild() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_0.gguf")
        try installEntry(storage, model, payloadBytes: 4096)
        let benchmark = BenchmarkDefinition.prefillThroughput(
            benchmarkId: "prefill_throughput_512", prefillTokens: 512)

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark), benchmark: benchmark,
            storage: storage, coordinator: DownloadCoordinator(storage: storage))

        // The declared half is the plan coordinate; the bound half is the same spec with
        // its source rewritten to the store's absolute path.
        #expect(request.model.declared == model)
        #expect(request.model.bound.boundPaths?.payload
                == storage.modelStore.payloadPath(for: model))
        #expect(request.runtime.declared == request.runtime.bound)
        #expect(request.runtime.bound == Runtime.thisBuild(for: request.model.declared))
    }

    /// The flags are the cell's own: keyed on all three axes, with the llama.cpp settings
    /// the spec asked for. `runtimeFlagsRef` would refuse anything else at run time.
    @Test func theFlagsAreKeyedOnTheCellsAxes() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let model = try ggufTextSpec("org/b-GGUF", "b-Q4_0.gguf")
        try installEntry(storage, model)
        let benchmark = BenchmarkDefinition.decodeThroughput(
            benchmarkId: "decode_throughput_512_100", prefillTokens: 512, decodeTokens: 100)

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark), benchmark: benchmark,
            storage: storage, coordinator: DownloadCoordinator(storage: storage))

        let flags = try #require(request.runtimeFlags)
        #expect(flags.axes == (.decodeThroughput, .llamacppIosPipette, .ggufText))
        let flat = RuntimeFlagRef(flags)
        #expect(flat.numberGpuLayers == 99)
        #expect(flat.ctxSize == 4096)
        #expect(flat.nUbatch == 512)
    }

    /// Apple Foundation ships with the OS: nothing to ensure, no path, and no flags
    /// variant upstream — so the request carries none rather than an unset one.
    @Test func appleFoundationSkipsTheStoreEntirely() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let coordinator = DownloadCoordinator(storage: storage)
        let benchmark = BenchmarkDefinition.prefillThroughput(
            benchmarkId: "prefill_throughput_512", prefillTokens: 512)

        let request = try await RunCell.prepare(
            // No AFM flags variant exists upstream, so the entry resolves to nothing.
            spec: spec(.appleFoundationText, benchmark,
                       numberGpuLayers: nil, ctxSize: nil, nUbatch: nil),
            benchmark: benchmark,
            storage: storage, coordinator: coordinator)

        // Nothing to bind, so the two halves are equal — as they are for the runtime.
        #expect(request.model.declared == .appleFoundationText)
        #expect(request.model.bound == .appleFoundationText)
        #expect(request.runtime.bound == .appleFoundation(
            privateThermal: Runtime.privateThermalBuild))
        #expect(request.runtimeFlags == nil)
        #expect(!coordinator.hasTransfer(for: .appleFoundationText))
    }

    /// The authored entry reaches the request unchanged — including a knob no job setting
    /// spells. This is the carry `prepare` exists to do: before it, flags were rebuilt from
    /// an argument per knob, and one with no argument beside it could not survive.
    @Test func theAuthoredEntryReachesTheRequest() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let model = try ggufTextSpec("org/c-GGUF", "c-Q4_0.gguf")
        try installEntry(storage, model)
        let benchmark = BenchmarkDefinition.maxMemoryUsage(
            benchmarkId: "max_memory_usage_512", prefillTokens: 512)

        let request = try await RunCell.prepare(
            spec: spec(model, benchmark, ctxSize: 2048, threads: 3, swaFull: false),
            benchmark: benchmark,
            storage: storage, coordinator: DownloadCoordinator(storage: storage))

        let flat = RuntimeFlagRef(try #require(request.runtimeFlags))
        #expect(flat.ctxSize == 2048)
        #expect(flat.threads == 3)
        #expect(flat.swaFull == false)
    }

    /// An entry naming a knob this cell does not carry fails the cell. Degrading it to
    /// "no flags" would run on engine defaults and report those as what the cell asked
    /// for — a result that looks valid and answers a different question.
    @Test func anEntryNamingAForeignKnobFailsTheCell() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let model = try mlxSpec("org/m-MLX-4bit")
        try installEntry(storage, model)
        let benchmark = BenchmarkDefinition.decodeThroughput(
            benchmarkId: "decode_throughput_512_100", prefillTokens: 512, decodeTokens: 100)
        // `ctx_size` is a llama knob; an MLX cell's variant carries the prefill chunk alone.
        let foreign = ClientRunSpec(
            benchmark: benchmark.benchmarkId, model: model,
            runtime: Runtime.thisBuild(for: model),
            runtimeFlags: RuntimeFlagRef(
                benchmarkType: benchmark.type, runtimeType: .mlxIosPipette, modelType: .mlx,
                ctxSize: 4096))

        await #expect(throws: RuntimeFlagResolveError.self) {
            _ = try await RunCell.prepare(
                spec: foreign, benchmark: benchmark,
                storage: storage, coordinator: DownloadCoordinator(storage: storage))
        }
    }

    /// Substituting a cell's load settings must not cost it the groups it named. The
    /// executor stands in for a pre-spec cell's `runtime_flags`; a rebuild there would drop
    /// the readiness gate from the record while the gate still held the run.
    @Test func replacingTheLoadSettingsKeepsTheOtherGroups() throws {
        let model = try ggufTextSpec("org/d-GGUF", "d-Q4_0.gguf")
        let gate = BenchmarkFlagRef(
            benchmarkType: .prefillThroughput, runtimeType: .llamacppIosPipette,
            modelType: .ggufText,
            readiness: ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true))
        let original = ClientRunSpec(
            benchmark: "prefill_throughput_512", model: model,
            runtime: Runtime.thisBuild(for: model), benchmarkFlags: gate)

        let replaced = original.replacingRuntimeFlags(
            RuntimeFlagRef(benchmarkType: .prefillThroughput, runtimeType: .llamacppIosPipette,
                           modelType: .ggufText, numberGpuLayers: 99))

        #expect(replaced.runtimeFlags?.numberGpuLayers == 99)
        #expect(replaced.benchmarkFlags?.readiness?.maxWaitSecs == 1800)
        #expect(replaced.benchmarkFlags?.readiness?.skipThermal == true)
        #expect(replaced.model == original.model)
    }

    /// A coordinate this client cannot fetch fails the cell at assembly, before anything
    /// loads — the ensure's refusal, surfaced by name.
    @Test func anUnfetchableModelIsRefusedBeforeAnythingLoads() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let url = try ResourceUrl("https://example.com/weights.gguf")
        let model = Model.ggufText(GgufText(source: .url(url: url, sha256: nil)))
        let benchmark = BenchmarkDefinition.prefillThroughput(
            benchmarkId: "prefill_throughput_512", prefillTokens: 512)

        await #expect(throws: ModelStoreError.notFetchableHere(model.artifactName)) {
            _ = try await RunCell.prepare(
                spec: spec(model, benchmark), benchmark: benchmark,
                storage: storage, coordinator: DownloadCoordinator(storage: storage))
        }
    }
}
