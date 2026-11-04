import FoundationModels
import Testing
@testable import Pipette

/// The pure AFM logic (eval's chat-message to session mapping, the timing benchmarks'
/// cap-reached threshold, eval sampling), which runs without the on-device model.
struct AFMRuntimeTests {
    // MARK: - splitMessages

    struct SplitCase {
        let name: String
        let messages: [[String: String]]
        let instructions: String
        let prompt: String
    }

    @Test(arguments: [
        SplitCase(name: "single user verbatim",
                  messages: [["role": "user", "content": "What is 2+2?"]],
                  instructions: "", prompt: "What is 2+2?"),
        SplitCase(name: "system + user",
                  messages: [["role": "system", "content": "Be terse."],
                             ["role": "user", "content": "Hi"]],
                  instructions: "Be terse.", prompt: "Hi"),
        SplitCase(name: "two system turns join with blank line",
                  messages: [["role": "system", "content": "A"],
                             ["role": "system", "content": "B"],
                             ["role": "user", "content": "Q"]],
                  instructions: "A\n\nB", prompt: "Q"),
        SplitCase(name: "multi-turn folds role-labeled",
                  messages: [["role": "user", "content": "Q1"],
                             ["role": "assistant", "content": "A1"],
                             ["role": "user", "content": "Q2"]],
                  instructions: "", prompt: "user: Q1\nassistant: A1\nuser: Q2"),
        SplitCase(name: "missing role defaults to user",
                  messages: [["content": "no role"]],
                  instructions: "", prompt: "no role"),
        SplitCase(name: "empty",
                  messages: [],
                  instructions: "", prompt: ""),
    ])
    func splitMessages(_ c: SplitCase) {
        let (instructions, prompt) = AFMRuntime.splitMessages(c.messages)
        #expect(instructions == c.instructions, "instructions [\(c.name)]")
        #expect(prompt == c.prompt, "prompt [\(c.name)]")
    }

    // MARK: - capFloor

    /// The cap-reached threshold is cap − max(2, cap/20): ~5% drift for large caps,
    /// a fixed 2-token floor for small ones.
    @Test(arguments: [
        (cap: 100, floor: 95),   // 5% = 5
        (cap: 200, floor: 190),  // 5% = 10
        (cap: 40, floor: 38),    // 5% = 2 (== the min)
        (cap: 10, floor: 8),     // 5% = 0 → clamped to 2
        (cap: 1, floor: -1),     // degenerate, but must not trap
    ])
    func capFloor(_ c: (cap: Int, floor: Int)) {
        #expect(AFMRuntime.capFloor(c.cap) == c.floor, "capFloor(\(c.cap))")
    }

    // MARK: - eval sampling

    private func sampling(temperature: Double = 1, topP: Double? = nil, topK: Int32? = nil,
                          minP: Double? = nil, seed: Int64? = nil) throws -> Sampling {
        try Sampling.fromBenchmark(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_math_500"),
            parameterEvalId: .known(.math500), parameterDatasetName: "2026.06.1",
            parameterMaxTokens: 8, parameterTemperature: try SamplingTemperature(temperature),
            parameterTopP: try topP.map { try TopP($0) },
            parameterTopK: try topK.map { try TopK($0) },
            parameterMinP: try minP.map { try MinP($0) },
            parameterSeed: try seed.map { try SamplingSeed($0) }))
    }

    /// The framework takes one cut and no min-p, so a pin it cannot apply fails the cell;
    /// greedy applies no cut, so nothing is refused there.
    @Test func aPinTheFrameworkCannotApplyIsRefused() throws {
        #expect(throws: RuntimeError.self) {
            try AFMRuntime.requireHonoured(try sampling(minP: 0.05))
        }
        #expect(throws: RuntimeError.self) {
            try AFMRuntime.requireHonoured(try sampling(topP: 0.95, topK: 64))
        }
        try AFMRuntime.requireHonoured(try sampling(temperature: 0, topP: 0.95, topK: 64,
                                                    minP: 0.05))
        try AFMRuntime.requireHonoured(try sampling(topP: 0.95, topK: 0, minP: 0))
    }

    /// Each honoured pin lands on the mode, the sample's seed with it.
    @Test func theHonouredPinsBecomeTheSamplingMode() throws {
        let cases: [(Sampling, UInt32?, GenerationOptions.SamplingMode?)] = [
            (try sampling(topK: 64), 7, .random(top: 64, seed: 7)),
            (try sampling(topP: 0.95), 7, .random(probabilityThreshold: 0.95, seed: 7)),
            (try sampling(), 7, .random(probabilityThreshold: 1, seed: 7)),
            (try sampling(topK: 0), nil, .random(probabilityThreshold: 1, seed: nil)),
            (try sampling(topP: 1), nil, .random(probabilityThreshold: 1, seed: nil)),
            (try sampling(), nil, nil),
            (try sampling(temperature: 0, topK: 64), 7, .greedy),
        ]
        for (sampling, seed, mode) in cases {
            let options = AFMRuntime.evalOptions(sampling, seed: seed, maxTokens: 8)
            #if compiler(>=6.4)
            #expect(options.samplingMode == mode, "\(sampling)")
            #else
            #expect(options.sampling == mode, "\(sampling)")
            #endif
        }
    }
}
