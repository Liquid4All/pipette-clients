import Foundation

// What an eval cell asks of an engine: the Swift counterpart of
// `crates/pipette-ops/src/eval_cell.rs`. The sample loop itself is `BenchmarkMeasurement.eval`.

/// What an eval benchmark asked for: the crate's `EvalCell`, with the knobs an engine here
/// reads. No `enable_thinking` or `reasoning_effort`, which a claim refuses, and no
/// `workers`: every engine here decodes one sequence.
nonisolated struct EvalCell: Sendable {
    let maxTokens: UInt32
    /// The answer set of a multiple-choice eval, when the benchmark has one.
    let mcqChoices: [String]?
    /// What the requests sample at, benchmark and eval-id policy resolved.
    let sampling: Sampling

    /// The cell `benchmark` describes: the crate's `EvalCell::new`. Checked before anything
    /// loads, so a body whose samples never arrived costs no model load.
    init(_ benchmark: EvalBenchmark) throws {
        guard benchmark.samples != nil else { throw EvalMeasurementError.missingSamples }
        maxTokens = benchmark.parameterMaxTokens.value.magnitude
        // Constraining generation to a choice set with no choices would answer nothing.
        mcqChoices = benchmark.parameterMcqChoices.flatMap { $0.isEmpty ? nil : $0 }
        sampling = try Sampling.fromBenchmark(benchmark)
    }
}

/// The sampling one eval cell runs under: the crate's `eval_cell::Sampling`. The temperature
/// is always resolved: the entry's pin, else the eval id's policy. A cut the entry left
/// unpinned stays `nil`, so the engine's own default still decides.
nonisolated struct Sampling: Sendable {
    let temperature: SamplingTemperature
    let topP: TopP?
    let topK: TopK?
    let minP: MinP?
    /// Held for `seedFor`: a sample draws with its own seed, never the entry's base one.
    private let benchmark: EvalBenchmark

    static func fromBenchmark(_ benchmark: EvalBenchmark) throws -> Sampling {
        Sampling(temperature: try benchmark.samplingTemperature(), topP: benchmark.parameterTopP,
                 topK: benchmark.parameterTopK, minP: benchmark.parameterMinP,
                 benchmark: benchmark)
    }

    func seedFor(_ sampleId: String) -> UInt32? { benchmark.sampleSeed(sampleId) }
}

/// Whether an eval's sampling temperature draws a sample or decodes greedily.
///
/// Zero is greedy on every engine, and ignores the cuts and the seed, as `llama-server`
/// and the crate's MLX server treat it.
nonisolated func drawsSample(atTemperature temperature: Double) -> Bool { temperature > 0 }
