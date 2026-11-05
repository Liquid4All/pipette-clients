import Foundation
import Testing

@testable import Pipette

/// `MlxExecute`: the per-kind dispatch in front of the MLX engine. The kernels themselves
/// need weights and a GPU, so what is testable here is the order of the steps around them.
struct MlxExecuteTests {
    /// A bundle `ResolvedMlxModel` accepts; nothing here loads it.
    private func bundle() throws -> URL {
        let dir = try makeFixtureDirectory()
        _ = try writeFixtureFile(dir, "config.json")
        return dir
    }

    /// The gate clears before the model loads, as the crate's timing cells gate before
    /// `start_server`: a cell that waited for a cool device and then loaded weights for
    /// twenty seconds has warmed it up again before its first rep. Asserted through a gate
    /// that refuses, since a load would otherwise fail on the placeholder config.
    @Test func theGateClearsBeforeTheModelLoads() async throws {
        let dir = try bundle()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try mlxSpec("org/m-MLX-4bit")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
            model: DeclaredBound(declared: declared,
                                 bound: try boundModel(declared, dir.path, nil)),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .decodeThroughput(DecodeThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
                parameterPrefillTokens: 512,
                parameterDecodeTokens: 100)))

        await #expect(throws: RuntimeError.readiness("hot")) {
            try await MlxExecute.run(
                request, runContext: NoCorpus(root: dir),
                readinessGate: { throw RuntimeError.readiness("hot") }, observer: noObserver())
        }
    }

    /// Eval does not gate: accuracy is thermal-invariant, so the crate's eval cell takes no
    /// readiness wait at all. A refusing gate must not reach it, and the cell fails on its
    /// missing samples instead.
    @Test func anEvalCellDoesNotGate() async throws {
        let dir = try bundle()
        defer { try? FileManager.default.removeItem(at: dir) }
        let declared = try mlxSpec("org/m-MLX-4bit")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
            model: DeclaredBound(declared: declared,
                                 bound: try boundModel(declared, dir.path, nil)),
            runtimeFlags: nil, benchmarkFlags: nil,
            benchmark: .eval(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "ifbench"),
                parameterEvalId: EvalId("ifbench"),
                parameterDatasetName: "ifbench",
                parameterMaxTokens: 64,
                parameterMcqChoices: nil,
                samples: nil)))

        await #expect(throws: EvalMeasurementError.missingSamples) {
            try await MlxExecute.run(
                request, runContext: NoCorpus(root: dir),
                readinessGate: { throw RuntimeError.readiness("hot") }, observer: noObserver())
        }
    }
}
