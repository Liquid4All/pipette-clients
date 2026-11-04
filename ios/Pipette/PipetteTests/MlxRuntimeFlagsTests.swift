import Foundation
import Testing

@testable import Pipette

/// `MlxRuntimeFlags`: the MLX engine's overlay onto a cell's flags.
struct MlxRuntimeFlagsTests {
    private func request(flags: RuntimeFlags?) throws -> RunRequest {
        let model = try mlxSpec("org/m-MLX-4bit")
        return RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)),
            model: .alreadyBound(model),
            runtimeFlags: flags,
            benchmarkFlags: nil,
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))
    }

    /// MLX's one setting is filled as the llama overlay fills its knobs, and the llama knobs
    /// stay absent: an MLX variant carries none, so reporting one would look authored.
    @Test func theOverlayReportsOnlyThePrefillChunk() throws {
        let ran = try MlxRuntimeFlags.forRun(try request(flags: nil))

        #expect(ran == .decodeThroughputMlxIosPipetteMlx(nUbatch: MlxRuntime.defaultPrefillChunk))
        #expect(ran.numberGpuLayers == nil)
        #expect(ran.ctxSize == nil)
    }

    /// An authored chunk survives the overlay, as it does on the llama side.
    @Test func theOverlayKeepsAnAuthoredChunk() throws {
        let ran = try MlxRuntimeFlags.forRun(try request(flags: .decodeThroughputMlxIosPipetteMlx(nUbatch: 128)))

        #expect(ran.nUbatch == 128)
    }
}
