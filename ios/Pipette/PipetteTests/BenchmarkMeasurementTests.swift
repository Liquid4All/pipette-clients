import Testing
import Foundation
@testable import Pipette

/// `BenchmarkMeasurement` outside its typed entries, the crate's `measurement/mod.rs`.
@Suite struct BenchmarkMeasurementTests {
    private func endToEnd(batchSize: UInt32) -> BenchmarkDefinition {
        .endToEndLatency(EndToEndLatencyBenchmark(
            benchmarkId: BenchmarkId(validated: "end_to_end_latency_512_256_bs\(batchSize)"),
            parameterPrefillTokens: 512, parameterDecodeTokens: 256,
            parameterBatchSize: BatchSize(validated: batchSize)))
    }

    /// A batch of one passes, as does a kind with no batch; more is refused naming the id.
    @Test func oneInFlightIsRequiredByName() throws {
        let engine = "this engine runs one sequence"
        try BenchmarkMeasurement.requireOneInFlight(endToEnd(batchSize: 1), engine: engine)
        let error = #expect(throws: RuntimeError.self) {
            try BenchmarkMeasurement.requireOneInFlight(endToEnd(batchSize: 4), engine: engine)
        }
        let message = "\(try #require(error))"
        #expect(message.contains("end_to_end_latency_512_256_bs4"))
        #expect(message.contains("4 requests in flight"))
        #expect(message.contains(engine))
        try BenchmarkMeasurement.requireOneInFlight(
            .maxMemoryUsage(MaxMemoryUsageBenchmark(
                benchmarkId: BenchmarkId(validated: "max_memory_usage_512"),
                parameterPrefillTokens: 512)),
            engine: engine)
    }

    /// The rep's value is the mean over its prompts, and the reference batch size is that
    /// single reading unchanged.
    @Test(arguments: [([12.5], 12.5), ([10.0, 20, 30, 40], 25.0)] as [([Double], Double)])
    func theRepValueIsThePerRequestMean(_ samples: [Double], _ mean: Double) throws {
        #expect(abs(try BenchmarkMeasurement.perRequestMean(samples) - mean) < 1e-9)
    }

    @Test func aRepWithNoReadingsHasNoMean() {
        #expect(throws: MeasurementError.noReadings) {
            _ = try BenchmarkMeasurement.perRequestMean([])
        }
    }

    // MARK: - The reduction

    /// What a reader is shown: every sample, in the order it was collected, next to the
    /// mean it produced. The exact string the crate pins, over the same readings.
    @Test func reportsTheSamplesBehindTheMean() {
        let samples: [Double] = [2, 4, 4, 4, 5]
        let line = BenchmarkMeasurement.reductionLine(
            label: "prefill_throughput", samples: samples,
            stats: BenchmarkMeasurement.meanStddev(samples))
        #expect(line == "prefill_throughput: samples [2.000, 4.000, 4.000, 4.000, 5.000] ms "
            + "-> mean 3.800 ms stddev 1.095 ms")
    }

    /// A cell that collected nothing must not read as one that measured zero.
    @Test func reportsAnEmptySeriesAsEmpty() {
        let line = BenchmarkMeasurement.reductionLine(
            label: "decode_throughput", samples: [], stats: BenchmarkMeasurement.meanStddev([]))
        #expect(line.contains("samples [] ms"), "\(line)")
    }

    /// Bessel-corrected, as the crate's `reduces_samples` cases pin it.
    @Test(arguments: [
        ([Double](), 0.0, 0.0),
        ([42.0], 42.0, 0.0),
        ([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 5.0, 2.138),
    ])
    func reducesSamples(_ samples: [Double], _ mean: Double, _ stddev: Double) {
        let stats = BenchmarkMeasurement.meanStddev(samples)
        #expect(abs(stats.meanMs - mean) < 1e-9, "mean was \(stats.meanMs)")
        #expect(abs(stats.stddevMs - stddev) < 0.01, "stddev was \(stats.stddevMs)")
    }
}
