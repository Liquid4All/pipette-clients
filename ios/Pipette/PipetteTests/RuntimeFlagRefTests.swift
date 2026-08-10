import Foundation
import Testing

@testable import Pipette

/// `RuntimeFlagRef` decoding and its resolution to an iOS variant.
///
/// Two refusal mechanisms, and the distinction is the point of these tests: a knob no
/// iOS variant can carry is not *declared*, so it is refused while decoding as an
/// unknown field; a knob that is declared but wrong for this cell's variant decodes
/// fine and is refused by `resolve()`. Before this suite existed, neither path had a
/// test, which is how three declared-but-never-refused knobs went unnoticed.
struct RuntimeFlagRefTests {

    private func group(
        benchmark: String = "prefill_throughput",
        runtime: String = "llamacpp_ios_pipette",
        model: String = "gguf_text",
        _ knobs: [String: Any] = [:]
    ) throws -> Data {
        var group: [String: Any] = [
            "benchmark_type": benchmark, "runtime_type": runtime, "model_type": model,
        ]
        for (key, value) in knobs { group[key] = value }
        return try JSONSerialization.data(withJSONObject: group)
    }

    private func decode(_ data: Data) throws -> RuntimeFlagRef {
        try JSONDecoder().decode(RuntimeFlagRef.self, from: data)
    }

    /// The refusals a ref can raise, with the axes taken off the ref itself — so an
    /// expectation stays about the refusal rather than restating the triple.
    private func noVariant(_ ref: RuntimeFlagRef) -> RuntimeFlagResolveError {
        .noSuchCombination(benchmarkType: ref.benchmarkType, runtimeType: ref.runtimeType,
                           modelType: ref.modelType)
    }

    private func knobRefused(_ knob: String, _ ref: RuntimeFlagRef) -> RuntimeFlagResolveError {
        .knobNotAllowed(knob: knob, benchmarkType: ref.benchmarkType,
                        runtimeType: ref.runtimeType, modelType: ref.modelType)
    }

    // MARK: - The subset resolves

    @Test func anIosLlamaCellCarriesEveryDeclaredKnob() throws {
        let ref = try decode(try group([
            "number_gpu_layers": 8, "ctx_size": 4096, "n_ubatch": 512, "threads": 6,
            "swa_full": false,
        ]))

        guard case let .prefillLlamacppIosPipetteGgufText(gpuLayers, ctxSize, nUbatch, threads,
                                                          swaFull) = try ref.resolve() else {
            Issue.record("expected the prefill llamacpp gguf-text variant")
            return
        }
        #expect(gpuLayers == 8)
        #expect(ctxSize == 4096)
        #expect(nUbatch == 512)
        #expect(threads == 6)
        #expect(swaFull == false)
    }

    /// Every knob is optional, so a group naming only its axes is legal.
    @Test func aGroupWithNoKnobsResolves() throws {
        guard case let .prefillLlamacppIosPipetteGgufText(gpuLayers, ctxSize, nUbatch, threads,
                                                          swaFull) =
            try decode(try group()).resolve()
        else {
            Issue.record("expected the prefill llamacpp gguf-text variant")
            return
        }
        #expect(gpuLayers == nil && ctxSize == nil && nUbatch == nil && threads == nil
                && swaFull == nil)
    }

    @Test func anIosMlxCellCarriesThePrefillChunkOnly() throws {
        let ref = try decode(try group(
            runtime: "mlx_ios_pipette", model: "mlx", ["n_ubatch": 256]))

        guard case let .prefillMlxIosPipetteMlx(nUbatch) = try ref.resolve() else {
            Issue.record("expected the prefill mlx variant")
            return
        }
        #expect(nUbatch == 256)
    }

    // MARK: - Refused while decoding: not declared at all

    /// A server-family knob is absent from the wire form, so it never reaches
    /// `resolve()` — `rejectUnknownFields` refuses it, naming the field.
    @Test(arguments: [
        ("dtype", "float16" as Any),
        ("tensor_parallel_size", 2 as Any),
        ("gpus", "all" as Any),
        ("shm_size", "1g" as Any),
        ("ipc", "host" as Any),
        ("mmap", true as Any),
        ("flash_attention", "on" as Any),
        ("no_cache", true as Any),
        ("envs", ["A=1"] as Any),
        ("raw", ["--verbose"] as Any),
    ])
    func aKnobNoIosVariantCarriesIsRefusedAsUnknown(_ name: String, _ value: Any) throws {
        let data = try group([name: value])

        #expect(throws: UnknownFlagField.self) { _ = try decode(data) }
    }

    // MARK: - Refused while resolving: declared, wrong variant

    /// An MLX variant carries the prefill chunk only, so every other llama setting decodes
    /// and is then refused by name.
    @Test(arguments: ["number_gpu_layers", "ctx_size", "threads"])
    func aLlamaKnobOnAnMlxCellIsRefusedByName(_ knob: String) throws {
        let ref = try decode(try group(
            runtime: "mlx_ios_pipette", model: "mlx", [knob: 8]))

        #expect(throws: knobRefused(knob, ref)) {
            _ = try ref.resolve()
        }
    }

    /// `swa_full` is refused the same way, in its own test because it is the one boolean
    /// among them — the sweep above feeds every knob a number.
    @Test func theSwaPolicyOnAnMlxCellIsRefusedByName() throws {
        let ref = try decode(try group(
            runtime: "mlx_ios_pipette", model: "mlx", ["swa_full": true]))

        #expect(throws: knobRefused("swa_full", ref)) {
            _ = try ref.resolve()
        }
    }

    // MARK: - No variant at all

    /// plan-types defines no `RuntimeFlags` variant for Apple Foundation on any
    /// benchmark, so a group naming it is not a knob problem — the triple has no
    /// variant to resolve to.
    @Test func anAppleFoundationCellHasNoVariant() throws {
        let ref = try decode(try group(
            runtime: "apple_foundation", model: "apple_foundation_text"))

        #expect(throws: noVariant(ref)) {
            _ = try ref.resolve()
        }
    }

    /// `vl_throughput` is gguf-vision only; the same benchmark against gguf-text has
    /// no variant.
    @Test func vlThroughputResolvesOnlyForVision() throws {
        let vision = try decode(try group(benchmark: "vl_throughput", model: "gguf_vision"))
        #expect((try? vision.resolve()) != nil)

        let text = try decode(try group(benchmark: "vl_throughput", model: "gguf_text"))
        #expect(throws: noVariant(text)) {
            _ = try text.resolve()
        }
    }

    /// A desktop runtime is a spelling this client must recognize in order to refuse
    /// it as a cell it cannot run, rather than as an unreadable payload.
    @Test func aDesktopRuntimeDecodesAndThenHasNoVariant() throws {
        let ref = try decode(try group(runtime: "docker_vllm", model: "torch"))

        #expect(throws: noVariant(ref)) {
            _ = try ref.resolve()
        }
    }
}
