import Foundation
import Testing

@testable import Pipette

/// `EvalCell` and `Sampling`, the crate's `eval_cell.rs`. What `Sampling` sends is each engine's
/// own request, so those tests live in `EngineSamplingTests`.
@Suite struct EvalCellTests {
    private func benchmark(mcqChoices: [String]? = [],
                           samples: [EvalSample]? = []) -> EvalBenchmark {
        EvalBenchmark(benchmarkId: BenchmarkId(validated: "eval_cell"),
                      parameterEvalId: EvalId("math_500"), parameterDatasetName: "d",
                      parameterMaxTokens: 8, parameterMcqChoices: mcqChoices, samples: samples)
    }

    /// An empty answer set is no answer set.
    @Test func theCellReadsItsKnobs() throws {
        let cell = try EvalCell(benchmark())
        #expect(cell.mcqChoices == nil)
        #expect(cell.maxTokens == 8)
        #expect(try EvalCell(benchmark(mcqChoices: ["A", "B"])).mcqChoices == ["A", "B"])
    }

    /// The cell reads its sampling off the benchmark, so an engine that holds a cell holds
    /// what the catalog said.
    @Test func theCellCarriesTheBenchmarksSampling() throws {
        var pinned = benchmark()
        pinned.parameterTemperature = try SamplingTemperature(1.0)
        pinned.parameterSeed = try SamplingSeed(7)
        let cell = try EvalCell(pinned)
        #expect(cell.sampling.temperature.value == 1.0)
        #expect(cell.sampling.seedFor("s0#3") == 10)
    }

    /// A body whose samples never arrived fails the cell before anything loads; an empty list
    /// is left alone, as upstream leaves it.
    @Test func aCellWithoutSamplesIsRefused() throws {
        #expect(throws: EvalMeasurementError.missingSamples) {
            _ = try EvalCell(benchmark(samples: nil))
        }
        _ = try EvalCell(benchmark(samples: []))
    }
}
