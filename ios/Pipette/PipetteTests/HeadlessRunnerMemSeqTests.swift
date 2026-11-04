import Testing

@testable import Pipette

/// `diag memseq` checks cross-cell isolation, so each of its cells has to be the one a job
/// runs: a pinned `ctx_size` sizes llama past the shipped window, and any llama knob on an
/// MLX or ONNX cell gets it refused.
@Suite struct HeadlessRunnerMemSeqTests {

    /// No context: the engine sizes it from the benchmark, as it does for a job cell.
    @Test func aLlamaCellCarriesTheBatchAndLeavesTheContextToTheEngine() throws {
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_K_M.gguf")
        let benchmark = HeadlessRunner.memSeqBenchmark(prefill: 512)
        let spec = HeadlessRunner.memSeqSpec(model: model, batch: 256, benchmark: benchmark)

        #expect(spec.benchmark.value == "max_memory_usage_512")
        #expect(spec.runtime == Runtime.thisBuild(for: model))
        #expect(spec.runtimeFlags == RuntimeFlagRef(
            benchmarkType: .maxMemoryUsage, runtimeType: .llamacppIosPipette,
            modelType: .ggufText, nUbatch: 256))
        #expect(LlamaRuntimeFlags.ctxSize(for: benchmark) == 513)
        _ = try spec.runtimeFlags?.resolve()
    }

    @Test func aLongerPrefillNamesItsBenchmarkAndSizesTheContext() throws {
        let model = try ggufTextSpec("org/a-GGUF", "a-Q4_K_M.gguf")
        let benchmark = HeadlessRunner.memSeqBenchmark(prefill: 4000)
        let spec = HeadlessRunner.memSeqSpec(model: model, batch: 512, benchmark: benchmark)

        #expect(spec.benchmark.value == "max_memory_usage_4000")
        #expect(LlamaRuntimeFlags.ctxSize(for: benchmark) == 4001)
    }

    /// The prefill chunk alone, which is the one knob an MLX variant declares.
    @Test func anMLXCellCarriesThePrefillChunkAndResolves() throws {
        let model = try mlxSpec("org/b-MLX-4bit")
        let spec = HeadlessRunner.memSeqSpec(
            model: model, batch: 64, benchmark: HeadlessRunner.memSeqBenchmark(prefill: 512))

        #expect(spec.runtimeFlags == RuntimeFlagRef(
            benchmarkType: .maxMemoryUsage, runtimeType: .mlxIosPipette,
            modelType: .mlx, nUbatch: 64))
        _ = try spec.runtimeFlags?.resolve()
    }

    /// Plan-types declares no ONNX variant, so any entry would be refused at `prepare`.
    @Test func anOnnxCellCarriesNoFlags() throws {
        let model = try onnxGenaiSpec("org/c-onnx")
        let spec = HeadlessRunner.memSeqSpec(
            model: model, batch: 512, benchmark: HeadlessRunner.memSeqBenchmark(prefill: 512))

        #expect(spec.runtime == Runtime.thisBuild(for: model))
        #expect(spec.runtimeFlags == nil)
    }
}
