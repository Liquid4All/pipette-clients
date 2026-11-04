import Foundation
import MLXLMCommon
import Testing
import llama

@testable import Pipette

/// Each engine's sampler for every sample the contract's `sampling.json` records, held to
/// what the engine the crate sends the same request to builds from it. Apple Foundation
/// has no such engine; `AFMRuntimeTests` holds what it refuses.
struct EngineSamplingTests {

    /// llama.cpp: the chain `llama-server` builds for a `/completion` request from
    /// `common_params_sampling`, read off the header this app's llama.cpp is built from.
    @Test(.enabled(if: isSimulator)) func llamaDrawsThroughTheServersChain() throws {
        let server = try ServerSampling.vendored()
        // Every other stage the server chains is off at its default, which leaves these.
        let applied = ["TOP_K", "TOP_P", "MIN_P", "TEMPERATURE"]
        #expect(server.order.filter(applied.contains) == applied)
        for stage in server.order where !applied.contains(stage) {
            #expect(try server.isOff(stage), "llama-server chains \(stage) on by default")
        }
        // Mirostat would replace the chain, and a dynamic range makes its `temp_ext` more
        // than the temperature.
        #expect(try server.number("mirostat") == 0)
        #expect(try server.number("dynatemp_range") == 0)
        for draw in try Self.draws() {
            let expected: [LlamaCpp.SamplerStage] = draw.temperature > 0 ? [
                .topK(draw.topK ?? server.topK),
                .topP(draw.topP.map(Float.init) ?? server.topP, minKeep: server.minKeep),
                .minP(draw.minP.map(Float.init) ?? server.minP, minKeep: server.minKeep),
                .temperature(Float(draw.temperature)),
                .dist(seed: draw.seed ?? LLAMA_DEFAULT_SEED),
            ] : [.greedy]
            #expect(LlamaCpp.evalChain(try draw.sampling(), seed: draw.seed) == expected, "\(draw)")
        }
    }

    /// MLX: what `mlx_lm.server` reads off the same request. A cut the request leaves out
    /// is its `--top-p 1.0`, `--top-k 0` and `--min-p 0.0` default, each of which disables
    /// it (mlx-lm 0.31.3, the version the crate's catalog pins).
    @Test(.enabled(if: isSimulator)) func mlxDrawsWithTheServersParameters() throws {
        for draw in try Self.draws() {
            let parameters = MlxRuntime.evalParameters(try draw.sampling(), seed: draw.seed)
            #expect(parameters.temperature == Float(draw.temperature), "\(draw)")
            #expect(parameters.topP == draw.topP.map(Float.init) ?? 1, "\(draw)")
            #expect(parameters.topK == draw.topK.map(Int.init) ?? 0, "\(draw)")
            #expect(parameters.minP == draw.minP.map(Float.init) ?? 0, "\(draw)")
            #expect(parameters.seed == draw.seed.map(UInt64.init), "\(draw)")
        }
    }

    /// One sample `sampling.json` records: what its request carries.
    struct Draw: CustomStringConvertible {
        let description: String
        let temperature: Double
        let topP: Double?
        let topK: Int32?
        let minP: Double?
        let seed: UInt32?

        /// This client's `Sampling` for the same pins. The temperature is always recorded,
        /// so no eval id's policy decides it.
        func sampling() throws -> Sampling {
            try Sampling.fromBenchmark(EvalBenchmark(
                benchmarkId: BenchmarkId(validated: "recorded"), parameterEvalId: EvalId("recorded"),
                parameterDatasetName: "recorded", parameterMaxTokens: 1,
                parameterTemperature: try SamplingTemperature(temperature),
                parameterTopP: try topP.map { try TopP($0) },
                parameterTopK: try topK.map { try TopK($0) },
                parameterMinP: try minP.map { try MinP($0) }))
        }
    }

    static func draws() throws -> [Draw] {
        let recorded = try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: root.appending(path: "crates/pipette-contract/fixtures/sampling.json")))
            as? [String: [String: [String: NSNumber]]])
        var draws: [Draw] = []
        for (file, samples) in recorded.sorted(by: { $0.key < $1.key }) {
            for (id, drawn) in samples.sorted(by: { $0.key < $1.key }) {
                let temperature = try #require(drawn["temperature"], "\(file) \(id) has none")
                draws.append(Draw(description: "\(file) \(id)",
                                  temperature: temperature.doubleValue,
                                  topP: drawn["top_p"]?.doubleValue, topK: drawn["top_k"]?.int32Value,
                                  minP: drawn["min_p"]?.doubleValue, seed: drawn["seed"]?.uint32Value))
            }
        }
        try #require(!draws.isEmpty, "sampling.json records no sample")
        return draws
    }

    /// The repo root, from the source tree, which only a simulator can see.
    static let root = URL(filePath: #filePath)
        .deletingLastPathComponent()   // PipetteTests/
        .deletingLastPathComponent()   // ios/Pipette/
        .deletingLastPathComponent()   // ios/
        .deletingLastPathComponent()

    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }
}

/// `common_params_sampling` in the vendored `common/common.h`: what `llama-server` samples a
/// request with where the request says nothing.
private struct ServerSampling {
    let topK: Int32
    let topP: Float
    let minP: Float
    let minKeep: Int
    /// The `COMMON_SAMPLER_TYPE_*` stages it chains, in order, without the prefix.
    let order: [String]
    private let fields: String

    static func vendored() throws -> ServerSampling {
        let header = try String(
            contentsOf: EngineSamplingTests.root.appending(path: "vendor/llama.cpp/common/common.h"),
            encoding: .utf8)
        let start = try #require(header.range(of: "struct common_params_sampling {"))
        let end = try #require(header[start.upperBound...].range(of: "\n};"))
        let fields = String(header[start.upperBound ..< end.lowerBound])
        let samplers = try #require(fields.firstMatch(of: #/samplers\s*=\s*\{([^}]*)\}/#)).1
        return ServerSampling(
            topK: try #require(Int32(try Self.value("top_k", in: fields))),
            topP: try #require(Float(try Self.value("top_p", in: fields))),
            minP: try #require(Float(try Self.value("min_p", in: fields))),
            minKeep: try #require(Int(try Self.value("min_keep", in: fields))),
            order: samplers.matches(of: #/COMMON_SAMPLER_TYPE_([A-Z_]+)/#).map { String($0.1) },
            fields: fields)
    }

    /// The number `name` defaults to.
    func number(_ name: String) throws -> Double {
        let value = try Self.value(name, in: fields)
        return try #require(Double(value), "common_params_sampling's \(name) is \(value)")
    }

    /// Whether `stage` does nothing at the server's defaults, as each field's comment says.
    func isOff(_ stage: String) throws -> Bool {
        switch stage {
        case "PENALTIES":
            try number("penalty_repeat") == 1 && number("penalty_freq") == 0
                && number("penalty_present") == 0
        case "DRY": try number("dry_multiplier") == 0
        case "TOP_N_SIGMA": try number("top_n_sigma") == -1
        case "TYPICAL_P": try number("typ_p") == 1
        case "XTC": try number("xtc_probability") == 0
        default: false
        }
    }

    /// The literal `name` is initialized to, without a float's `f`.
    private static func value(_ name: String, in fields: String) throws -> String {
        let pattern = try Regex("\\b\(name)\\s*=\\s*(-?[0-9.]+)f?\\s*;")
        let match = try #require(fields.firstMatch(of: pattern), "common_params_sampling has no \(name)")
        return try #require(match.output[1].substring).description
    }
}
