import Foundation
import Testing

@testable import Pipette

/// The in-process ONNX Runtime GenAI engine: what it refuses to load, how it composes its
/// intervals, and, against a real package, that it loads and generates.
@MainActor struct OnnxGenaiRuntimeTests {
    // MARK: - Package validation

    private func makePackage(_ body: (URL) throws -> Void) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("onnx-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try body(dir)
        return dir
    }

    private func writeConfig(_ dir: URL, decoder: String) throws {
        let config: [String: Any] = ["model": ["decoder": ["filename": decoder]]]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("genai_config.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer_config.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
    }

    private func protobufVarint(_ value: Int) -> Data {
        var value = UInt64(value)
        var bytes = Data()
        repeat {
            var byte = UInt8(value & 0x7F)
            value >>= 7
            if value != 0 { byte |= 0x80 }
            bytes.append(byte)
        } while value != 0
        return bytes
    }

    private func protobufBytesField(_ number: Int, _ payload: Data) -> Data {
        var bytes = protobufVarint((number << 3) | 2)
        bytes.append(protobufVarint(payload.count))
        bytes.append(payload)
        return bytes
    }

    private func protobufVarintField(_ number: Int, _ value: Int) -> Data {
        var bytes = protobufVarint(number << 3)
        bytes.append(protobufVarint(value))
        return bytes
    }

    /// A minimal ModelProto → GraphProto → TensorProto carrying one external-data
    /// `location`. The name is arbitrary: ONNX does not prescribe a filename suffix.
    private func graphReferencing(_ sidecar: String) -> Data {
        var entry = protobufBytesField(1, Data("location".utf8))
        entry.append(protobufBytesField(2, Data(sidecar.utf8)))
        var tensor = protobufBytesField(13, entry)
        tensor.append(protobufVarintField(14, 1))
        let graph = protobufBytesField(5, tensor)
        return protobufBytesField(7, graph)
    }

    private func selfContainedGraph() -> Data { protobufBytesField(7, Data()) }

    /// The memory cell's counter adds only file-backed growth past the bracket's start:
    /// pages that were already resident, or that the OS evicted since, add nothing.
    @Test(arguments: [
        (2_000, 500, 60, 2_440),
        (2_000, 60, 60, 2_000),
        (2_000, 40, 60, 2_000),
    ] as [(UInt64, UInt64, UInt64, UInt64)])
    func theMemoryCounterAddsMappedGrowthOnly(footprint: UInt64, fileBacked: UInt64,
                                              base: UInt64, expected: UInt64) {
        #expect(OnnxGenaiRuntime.withMappedWeights(footprint: footprint, fileBacked: fileBacked,
                                                   base: base) == expected)
    }

    /// A directory with real `.onnx` graphs and no `genai_config.json` is a plain ONNX
    /// export, not an ONNX Runtime GenAI package. Every `LiquidAI/*-ONNX` repo is exactly
    /// this, so the refusal has to name the distinction rather than fail at load.
    @Test func aPlainOnnxExportIsRefusedByName() throws {
        let dir = try makePackage { dir in
            try Data("graph".utf8).write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        let spec = try onnxGenaiSpec("org/plain-onnx")
        let request = try boundRequest(spec, dir.path)
        #expect(throws: BoundModelDirError.self) {
            try ResolvedOnnxGenaiModel(request.model.bound)
        }
    }

    /// A config naming a graph that is not on disk is an incomplete package: the load
    /// would succeed far enough to be confusing.
    @Test func aMissingDecoderGraphIsRefused() throws {
        let dir = try makePackage { try writeConfig($0, decoder: "model.onnx") }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package with no decoder graph must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("model.onnx"))
        }
    }

    @Test(arguments: [
        "../outside.onnx",
        "/tmp/outside.onnx",
        "graphs/../model.onnx",
        "graphs//model.onnx",
        "graphs\\model.onnx",
    ])
    func aDecoderPathMustBeNormalizedAndRelative(_ decoder: String) throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: decoder)
            try Data("graph".utf8).write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("an unsafe decoder path must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("normalized relative path"))
            #expect(detail.contains(decoder))
        }
    }

    @Test func aDecoderSymlinkCannotEscapeThePackage() throws {
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("onnx-outside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        try Data("graph".utf8).write(to: outside.appendingPathComponent("model.onnx"))

        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "graphs/model.onnx")
            try FileManager.default.createSymbolicLink(
                at: dir.appendingPathComponent("graphs"), withDestinationURL: outside)
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a decoder symlink outside the package must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("symbolic link"))
        }
    }

    /// The sidecar is the multi-gigabyte half of a large package, and the half a truncated
    /// transfer loses. A graph referencing one that is absent is refused before the engine
    /// loads it and fails at the first inference.
    @Test func aMissingExternalDataSidecarIsRefused() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            // A graph whose bytes reference a sidecar, as a real >2GB export does.
            try graphReferencing("model.onnx_data")
                .write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package missing its external-data file must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("model.onnx_data"))
        }
    }

    /// ONNX external-data locations are arbitrary POSIX paths, not an extension
    /// convention. A missing `weights.bin` must be caught just like `model.onnx_data`.
    @Test func aMissingArbitrarilyNamedExternalDataFileIsRefused() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing("weights.bin")
                .write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package missing arbitrarily named external data must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("weights.bin"))
        }
    }

    @Test func aCompletePackagePasses() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing("model.onnx_data")
                .write(to: dir.appendingPathComponent("model.onnx"))
            try Data("weights".utf8).write(to: dir.appendingPathComponent("model.onnx_data"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        try OnnxGenaiModels.validatePackage(at: dir.path)
    }

    @Test func aCompletePackageWithArbitrarilyNamedExternalDataPasses() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing("weights.bin")
                .write(to: dir.appendingPathComponent("model.onnx"))
            try Data("weights".utf8).write(to: dir.appendingPathComponent("weights.bin"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        try OnnxGenaiModels.validatePackage(at: dir.path)
    }

    @Test func aNestedGraphResolvesItsSidecarBesideIt() throws {
        let dir = try makePackage { dir in
            let graphs = dir.appendingPathComponent("graphs", isDirectory: true)
            try FileManager.default.createDirectory(at: graphs, withIntermediateDirectories: true)
            try writeConfig(dir, decoder: "graphs/model.onnx")
            try graphReferencing("model.onnx_data")
                .write(to: graphs.appendingPathComponent("model.onnx"))
            try Data("weights".utf8).write(to: graphs.appendingPathComponent("model.onnx_data"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        try OnnxGenaiModels.validatePackage(at: dir.path)
    }

    @Test func anExternalDataPathCannotTraverseOutOfThePackage() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing("../outside.onnx_data")
                .write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("an external-data traversal must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("../outside.onnx_data"))
            #expect(detail.contains("normalized relative path"))
        }
    }

    /// A self-contained graph references no sidecar, which is not a fault.
    @Test func aSelfContainedGraphNeedsNoSidecar() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try selfContainedGraph().write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        try OnnxGenaiModels.validatePackage(at: dir.path)
    }

    @Test func aMissingTokenizerVocabularyIsRefused() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try FileManager.default.removeItem(at: dir.appendingPathComponent("tokenizer.json"))
            try selfContainedGraph().write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package missing its tokenizer vocabulary must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("tokenizer.json"))
        }
    }

    @Test func aMissingTokenizerConfigIsRefused() throws {
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try FileManager.default.removeItem(
                at: dir.appendingPathComponent("tokenizer_config.json"))
            try selfContainedGraph().write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package missing its tokenizer config must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("tokenizer_config.json"))
        }
    }

    /// The catalog model's long external-data name survives protobuf decoding exactly.
    @Test func aLongExternalDataLocationIsReadCorrectly() throws {
        let sidecar = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx.data"
        #expect(sidecar.utf8.count == 53, "53 is ASCII `5`, which is the whole problem")
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing(sidecar).write(to: dir.appendingPathComponent("model.onnx"))
            try Data("weights".utf8).write(to: dir.appendingPathComponent(sidecar))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        try OnnxGenaiModels.validatePackage(at: dir.path)
    }

    @Test func aMissingLongExternalDataLocationIsNamedExactly() throws {
        let sidecar = "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx.data"
        let dir = try makePackage { dir in
            try writeConfig(dir, decoder: "model.onnx")
            try graphReferencing(sidecar).write(to: dir.appendingPathComponent("model.onnx"))
        }
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try OnnxGenaiModels.validatePackage(at: dir.path)
            Issue.record("a package missing its external-data file must be refused")
        } catch let error as RuntimeError {
            guard case let .engine(detail) = error else {
                Issue.record("expected an engine error, got \(error)"); return
            }
            #expect(detail.contains("`\(sidecar)`"))
        }
    }

    // MARK: - Interval composition

    /// End to end is composed from the two measured intervals, never a wall clock: a
    /// 1-token generation is its time to first token, and each further token adds one
    /// mean interval.
    @Test(arguments: [
        (generated: 1, ttft: 100.0, tpot: Double?.none, expected: 100.0),
        (generated: 2, ttft: 100.0, tpot: Double?.some(10), expected: 110.0),
        (generated: 5, ttft: 100.0, tpot: Double?.some(10), expected: 140.0),
    ])
    func endToEndIsReconstructedFromTheIntervals(
        generated: Int, ttft: Double, tpot: Double?, expected: Double
    ) {
        let timing = OnnxGenaiRuntime.GenerationTiming(
            promptTokens: 512, generatedTokens: generated, ttftMs: ttft, tpotMs: tpot)
        #expect(timing.endToEndMs == expected)
    }

    // MARK: - Support matrix

    /// `max_memory_usage` is supported here and not on Apple Foundation, because this
    /// engine runs in-process where `ProcessMemory` can see it.
    @Test func theSupportMatrixIsTheOneTheDocStates() {
        for benchmark in [BenchmarkType.prefillThroughput, .decodeThroughput,
                          .endToEndLatency, .maxMemoryUsage] {
            #expect(isBenchmarkSupported(benchmark, on: .onnxGenai), "\(benchmark)")
        }
        for benchmark in [BenchmarkType.eval, .vlThroughput, .vlMaxMemory] {
            #expect(!isBenchmarkSupported(benchmark, on: .onnxGenai), "\(benchmark)")
        }
    }

    // MARK: - Against a real package

    /// The engine's own tiny test package, from inside the test bundle. Not `#filePath`,
    /// which names the *build machine's* filesystem: `ios/copy-onnx-test-fixture.sh` copies
    /// it in from the submodule at build time, so a device run finds it too.
    ///
    /// `nil` when the submodule is not checked out. The tests below are gated on it with
    /// `.enabled(if:)` rather than returning early, because a test that quietly returns
    /// reports as a pass.
    nonisolated private static func fixtureModel() -> String? {
        guard let resources = Bundle(for: BundleMarker.self).resourceURL else { return nil }
        let model = resources.appendingPathComponent("OnnxGenaiFixture", isDirectory: true)
        return FileManager.default.fileExists(
            atPath: model.appendingPathComponent("genai_config.json").path) ? model.path : nil
    }

    /// Locates the test bundle. `Bundle.module` needs a SwiftPM resource declaration, which
    /// an Xcode test target does not have.
    nonisolated private final class BundleMarker {}

    /// Exercise the structural parser against the engine's real ONNX protobuf and
    /// tokenizer layout, not only the deliberately minimal protobuf fixtures above.
    @Test(.enabled(if: fixtureModel() != nil)) func aRealPackagePassesStructuralValidation() throws {
        try OnnxGenaiModels.validatePackage(at: try #require(Self.fixtureModel()))
    }

    /// **The engine end to end, in the app.** Loads a real package through `OGAConfig` /
    /// `OGAModel` / `OGATokenizer` and generates, which is what proves the static archive,
    /// the bridging header and the Objective-C surface all line up.
    ///
    /// The prompt is hand-picked ids rather than `promptIds`: this fixture pairs a
    /// 50257-entry GPT-2 vocab with a 1000-row embedding, so anything its tokenizer
    /// produces is out of range. Real packages do not have that problem, and the prompt
    /// path is exercised by `theTokenizerEncodesThroughTheBridge` below.
    @Test(.enabled(if: fixtureModel() != nil)) func theEngineLoadsAndGeneratesAgainstARealPackage() async throws {
        let model = try #require(Self.fixtureModel())

        let timing = try await OnnxGenaiRuntime.withFreshModel(path: model) { loaded in
            try OnnxGenaiRuntime.generate(loaded, promptIds: [1, 2, 3, 4], decodeTokens: 8)
        }

        #expect(timing.promptTokens == 4)
        #expect(timing.generatedTokens == 8)
        #expect(timing.ttftMs > 0)
        #expect((timing.tpotMs ?? 0) > 0)
        // Composed, not timed: it must be at least the time to the first token.
        #expect(timing.endToEndMs >= timing.ttftMs)
    }

    /// The tokenizer reaches Swift through the same bridge, and the ids it returns survive
    /// the pointer borrow `encode` copies out of.
    @Test(.enabled(if: fixtureModel() != nil)) func theTokenizerEncodesThroughTheBridge() async throws {
        let model = try #require(Self.fixtureModel())

        let ids = try await OnnxGenaiRuntime.withFreshModel(path: model) { loaded in
            try OnnxGenaiRuntime.encode("The capital of France is", with: loaded.tokenizer)
        }

        #expect(!ids.isEmpty)
    }

    /// `min_length` pins the generation to the length the cell asked for, which is this
    /// engine's `ignore_eos`: without it a short generation stops at the first EOS and the
    /// rep reports a decode it did not run.
    @Test(.enabled(if: fixtureModel() != nil)) func theGenerationRunsTheRequestedLengthRatherThanStoppingAtEos() async throws {
        let model = try #require(Self.fixtureModel())

        for decode in [2, 16] {
            let timing = try await OnnxGenaiRuntime.withFreshModel(path: model) { loaded in
                try OnnxGenaiRuntime.generate(loaded, promptIds: [1, 2, 3, 4],
                                              decodeTokens: decode)
            }
            #expect(timing.generatedTokens == decode)
        }
    }

    /// A zero-length generation measures nothing and is refused rather than reported as an
    /// instant one.
    @Test(.enabled(if: fixtureModel() != nil)) func aZeroTokenGenerationIsRefused() async throws {
        let model = try #require(Self.fixtureModel())

        await #expect(throws: RuntimeError.self) {
            try await OnnxGenaiRuntime.withFreshModel(path: model) { loaded in
                try OnnxGenaiRuntime.generate(loaded, promptIds: [1, 2], decodeTokens: 0)
            }
        }
    }
}
