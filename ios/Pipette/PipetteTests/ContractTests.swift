import Foundation
import Testing

@testable import Pipette

/// The contract fixtures `pipette-contract` also reads: every example round-trips through
/// the Swift mirror, unless it holds a variant `ios/contract-refusals.json` refuses; and
/// what the crate derives from the examples, this client derives too.
struct ContractTests {

    @Test(.enabled(if: isSimulator)) func everyExampleRoundTripsOrHoldsARefusedVariant() throws {
        let examples = try Self.examples()
        try #require(!examples.isEmpty, "variants.json names no examples")
        for (file, refusals) in examples.sorted(by: { $0.key < $1.key }) {
            let roundTrip = try Self.roundTrip(file)
            let json = try Data(contentsOf: Self.fixtures.appending(path: file))
            try Self.check(file, json, refusals: refusals, roundTrip)
        }
    }

    /// `params.json`: the parameters a submission declares for each benchmark example. An
    /// example holding a refused variant (such as `vlm_eval`) has no Swift mirror and is
    /// skipped.
    @Test(.enabled(if: isSimulator)) func everyBenchmarkDeclaresTheCratesParams() throws {
        let recorded: [String: [String: Any]] = try Self.derived("params.json")
        try #require(!recorded.isEmpty, "params.json records no benchmark")
        let refused = try Self.examples().filter { !$0.value.isEmpty }
        for (file, expected) in recorded.sorted(by: { $0.key < $1.key })
        where refused[file] == nil {
            let definition = try JSONDecoder().decode(
                BenchmarkDefinition.self, from: Data(contentsOf: Self.fixtures.appending(path: file)))
            let params = BenchmarkParams(definition)
            var declared = try Self.object(params)
            declared["benchmark_type"] = params.benchmarkType.rawValue
            #expect(declared as NSDictionary == expected as NSDictionary, "\(file)")
        }
    }

    /// `runtime-build-flags.json`: the build flag the server's catalog reads for each iOS
    /// app runtime example. This app reports the flag, so a different digest here withholds
    /// every catalog offer of the build.
    @Test(.enabled(if: isSimulator)) func everyRuntimeGivesTheCratesBuildFlag() throws {
        let recorded: [String: String] = try Self.derived("runtime-build-flags.json")
        try #require(!recorded.isEmpty, "runtime-build-flags.json records no runtime")
        for (file, expected) in recorded.sorted(by: { $0.key < $1.key }) {
            let runtime = try JSONDecoder().decode(
                Runtime.self, from: Data(contentsOf: Self.fixtures.appending(path: file)))
            #expect(try runtimeBuildFlag(runtime) == expected, "\(file)")
        }
    }

    /// `sampling.json`: what each sample of each eval example draws under, as the crate's
    /// engines write it into a request.
    @Test(.enabled(if: isSimulator)) func everyEvalSampleDrawsUnderTheCratesSampling() throws {
        let recorded: [String: [String: [String: Any]]] = try Self.derived("sampling.json")
        try #require(!recorded.isEmpty, "sampling.json records no eval")
        for (file, samples) in recorded.sorted(by: { $0.key < $1.key }) {
            let definition = try JSONDecoder().decode(
                BenchmarkDefinition.self, from: Data(contentsOf: Self.fixtures.appending(path: file)))
            guard case let .eval(benchmark) = definition else {
                Issue.record("\(file) is no eval")
                continue
            }
            let sampling = try Sampling.fromBenchmark(benchmark)
            for (id, expected) in samples.sorted(by: { $0.key < $1.key }) {
                var drawn: [String: Any] = ["temperature": sampling.temperature.value]
                drawn["top_p"] = sampling.topP?.value
                drawn["top_k"] = sampling.topK?.value
                drawn["min_p"] = sampling.minP?.value
                drawn["seed"] = sampling.seedFor(id)
                #expect(drawn as NSDictionary == expected as NSDictionary, "\(file) \(id)")
            }
        }
    }

    /// `defaults.json`: each field an example may leave out reads as the crate reads it, and
    /// every other field is required here too, except those `ios/contract-differences.json`
    /// lists, which must still read differently.
    @Test(.enabled(if: isSimulator)) func everyOmittedFieldReadsAsTheCrateReadsIt() throws {
        let recorded: [String: [String: Any]] = try Self.derived("defaults.json")
        var differences = try JSONDecoder().decode(
            [String: [String: String]].self,
            from: Data(contentsOf: Self.root.appending(path: "ios/contract-differences.json")))
        for (file, refusals) in try Self.examples().sorted(by: { $0.key < $1.key })
        where refusals.isEmpty {
            guard let roundTrip = try Self.decoded(file) else { continue }
            let authored = try Self.json(file)
            let omittable = recorded[file] ?? [:]
            for pointer in Field.pointers(in: authored) {
                let difference = differences[file]?.removeValue(forKey: pointer)
                let without = try Field.setting(pointer, to: nil, in: authored)
                let read = Self.read(without, with: roundTrip)
                let expected = try Self.omitted(omittable[pointer], at: pointer, from: without,
                                                in: file)
                let same = (try? read.get()) as? NSDictionary == expected as? NSDictionary
                if let difference {
                    #expect(!same, """
                        \(file): \(pointer) reads as the crate reads it now, so drop it from \
                        contract-differences.json (\(difference))
                        """)
                } else {
                    #expect(same, """
                        \(file): without \(pointer), reads as \(read), where the crate \
                        reads \(String(describing: expected))
                        """)
                }
            }
        }
        let unread = differences.filter { !$0.value.isEmpty }
        #expect(unread.isEmpty, "contract-differences.json names fields this test never reads: \(unread)")
    }

    /// `bounds.json`: the values at and past each bounded field are accepted and refused
    /// as the crate accepts and refuses them.
    @Test(.enabled(if: isSimulator)) func everyBoundIsWhereTheCrateHasIt() throws {
        let recorded: [String: [String: [String: [Any]]]] = try Self.derived("bounds.json")
        try #require(!recorded.isEmpty, "bounds.json tries no field")
        let examples = try Self.examples()
        for (file, fields) in recorded.sorted(by: { $0.key < $1.key }) {
            let refusals = try #require(examples[file], "bounds.json: \(file) is no example")
            guard refusals.isEmpty, let roundTrip = try Self.decoded(file) else { continue }
            let authored = try Self.json(file)
            for (pointer, bounds) in fields.sorted(by: { $0.key < $1.key }) {
                for value in bounds["accepted"] ?? [] {
                    let tried = try Field.setting(pointer, to: value, in: authored)
                    let read = Self.read(tried, with: roundTrip)
                    #expect((try? read.get()) as? NSDictionary == tried as? NSDictionary,
                            "\(file): \(pointer) = \(value) reads as \(read)")
                }
                for value in bounds["refused"] ?? [] {
                    let tried = try Field.setting(pointer, to: value, in: authored)
                    let read = Self.read(tried, with: roundTrip)
                    #expect((try? read.get()) == nil,
                            "\(file): \(pointer) = \(value) is refused by the crate, but reads as \(read)")
                }
            }
        }
    }

    /// The repo root, from the source tree, which only a simulator can see.
    private static let root = URL(filePath: #filePath)
        .deletingLastPathComponent()   // PipetteTests/
        .deletingLastPathComponent()   // ios/Pipette/
        .deletingLastPathComponent()   // ios/
        .deletingLastPathComponent()

    private static let fixtures = root.appending(path: "crates/pipette-contract/fixtures")

    /// Every example, with the refusals of the variants it holds.
    private static func examples() throws -> [String: [String]] {
        let variants = try JSONDecoder().decode(
            [String: [String: [String]]].self,
            from: Data(contentsOf: fixtures.appending(path: "variants.json")))
        let refusals = try JSONDecoder().decode(
            [String: [String: String]].self,
            from: Data(contentsOf: root.appending(path: "ios/contract-refusals.json")))
        var examples: [String: [String]] = [:]
        for (enumName, held) in variants {
            for (variant, files) in held {
                let refusal = refusals[enumName]?[variant].map { "\(enumName).\(variant): \($0)" }
                for file in files {
                    examples[file, default: []] += refusal.map { [$0] } ?? []
                }
            }
        }
        return examples
    }

    /// How `file`'s section is read and written back, as this client reads and writes it.
    private static func roundTrip(_ file: String) throws -> (Data) throws -> Data {
        switch file.prefix(while: { $0 != "/" }) {
        case "model": roundTrip(Model.self)
        case "runtime": roundTrip(Runtime.self)
        // As a claim reads it: the flat ref, resolved to typed flags and back.
        case "model_flags": {
            let flat = try JSONDecoder().decode(ModelFlagRef.self, from: $0)
            return try JSONEncoder().encode(ModelFlagRef(flat.resolve()))
        }
        case "runtime_flags": {
            let flat = try JSONDecoder().decode(RuntimeFlagRef.self, from: $0)
            return try JSONEncoder().encode(RuntimeFlagRef(flat.resolve()))
        }
        case "benchmark_flags": {
            let flat = try JSONDecoder().decode(BenchmarkFlagRef.self, from: $0)
            return try JSONEncoder().encode(BenchmarkFlagRef(flat.resolve()))
        }
        case "benchmark": roundTrip(BenchmarkDefinition.self)
        // As a submission writes it: the untagged fields beside the kind.
        case "benchmark_params": {
            let params = try JSONDecoder().decode(TaggedParams.self, from: $0).params
            var declared = try object(params)
            declared["benchmark_type"] = params.benchmarkType.rawValue
            return try JSONSerialization.data(withJSONObject: declared)
        }
        case "client_run_spec": roundTrip(ClientRunSpec.self)
        case "device": roundTrip(DeviceInfo.self)
        case "eval_completion": roundTrip(BenchmarkEvalCompletion.self)
        case "failure_code": roundTrip(FailureCode.self)
        default: throw UnknownSection(file: file)
        }
    }

    /// `roundTrip`, for a section this client reads. It only writes `benchmark_params` and
    /// `device`, so which of their fields a reader requires is not this client's to say.
    private static func decoded(_ file: String) throws -> ((Data) throws -> Data)? {
        switch file.prefix(while: { $0 != "/" }) {
        case "benchmark_params", "device": nil
        default: try roundTrip(file)
        }
    }

    /// `value` read and written back, or why it was refused.
    private static func read(_ value: Any, with roundTrip: (Data) throws -> Data)
        -> Result<Any, any Error> {
        Result { try json(roundTrip(JSONSerialization.data(withJSONObject: value))) }
    }

    /// What `without` reads as by its `defaults.json` entry; `nil`, a refusal, when the
    /// field is required and so has none.
    private static func omitted(_ entry: Any?, at pointer: String, from without: Any,
                                in file: String) throws -> Any? {
        guard let entry else { return nil }
        if entry as? String == "absent" { return without }
        guard let entry = entry as? [String: Any], entry.count == 1,
              let value = entry["default"] else {
            throw MalformedDefault(file: file, pointer: pointer, entry: "\(entry)")
        }
        return try Field.setting(pointer, to: value, in: without)
    }

    private static func roundTrip<T: Codable>(_: T.Type) -> (Data) throws -> Data {
        { try JSONEncoder().encode(JSONDecoder().decode(T.self, from: $0)) }
    }

    private static func derived<T>(_ name: String) throws -> T {
        try #require(try JSONSerialization.jsonObject(
            with: Data(contentsOf: fixtures.appending(path: name))) as? T, "\(name)")
    }

    private static func json(_ file: String) throws -> Any {
        try json(Data(contentsOf: fixtures.appending(path: file)))
    }

    private static func json(_ data: Data) throws -> Any {
        try JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed)
    }

    private static func object(_ value: some Encodable) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(value)) as? [String: Any])
    }

    private static func check(_ file: String, _ json: Data, refusals: [String],
                              _ roundTrip: (Data) throws -> Data) throws {
        if !refusals.isEmpty {
            if (try? roundTrip(json)) != nil {
                Issue.record("\(file) is refused (\(refusals.joined(separator: "; "))) but decodes")
            }
            return
        }
        let encoded: Data
        do {
            encoded = try roundTrip(json)
        } catch {
            Issue.record("\(file) does not decode: \(error). Mirror its variants, or refuse one.")
            return
        }
        // As objects, not dictionaries: a `failure_code` example is a bare string.
        let got = try Self.json(encoded) as? NSObject
        let want = try Self.json(json) as? NSObject
        #expect(got == want, "\(file) re-encodes differently: \(String(decoding: encoded, as: UTF8.self))")
    }

    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }
}

/// An example in a section this test has no reader for: a section the crate added.
private struct UnknownSection: Error, CustomStringConvertible {
    let file: String
    var description: String { "\(file) is in a section this test does not decode" }
}

/// A `defaults.json` entry that is neither `"absent"` nor `{"default": …}`.
private struct MalformedDefault: Error, CustomStringConvertible {
    let file: String
    let pointer: String
    let entry: String
    var description: String {
        "defaults.json: \(file) \(pointer) is \(entry), neither \"absent\" nor a default"
    }
}

/// A field of an example by its JSON pointer, as `pipette-contract`'s `fields` names it:
/// reached through objects only, never into an array.
private enum Field {
    struct Missing: Error, CustomStringConvertible {
        let pointer: String
        var description: String { "\(pointer) is in no object" }
    }

    static func pointers(in value: Any, at: String = "") -> [String] {
        guard let object = value as? [String: Any] else { return [] }
        return object.sorted(by: { $0.key < $1.key }).flatMap { key, field -> [String] in
            let pointer = at + "/" + key.replacingOccurrences(of: "~", with: "~0")
                .replacingOccurrences(of: "/", with: "~1")
            return [pointer] + pointers(in: field, at: pointer)
        }
    }

    /// `value` with `pointer`'s field set to `field`, or without it when `field` is `nil`.
    static func setting(_ pointer: String, to field: Any?, in value: Any) throws -> Any {
        let keys = pointer.split(separator: "/", omittingEmptySubsequences: false).dropFirst()
            .map { $0.replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~") }
        return try set(keys[...], field, in: value, pointer: pointer)
    }

    private static func set(_ keys: ArraySlice<String>, _ field: Any?, in value: Any,
                            pointer: String) throws -> Any {
        guard var object = value as? [String: Any], let key = keys.first else {
            throw Missing(pointer: pointer)
        }
        if keys.count == 1 {
            object[key] = field
        } else {
            guard let inner = object[key] else { throw Missing(pointer: pointer) }
            object[key] = try set(keys.dropFirst(), field, in: inner, pointer: pointer)
        }
        return object
    }
}

/// The crate's tagged `BenchmarkParams`, read as the server reads a declared pair. This
/// client only writes one, so the reader lives with the test.
private struct TaggedParams: Decodable {
    let params: BenchmarkParams

    private enum Key: String, CodingKey {
        case benchmarkType = "benchmark_type"
        case prefillTokens = "prefill_tokens"
        case decodeTokens = "decode_tokens"
        case batchSize = "batch_size"
        case maxTokens = "max_tokens"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case textTokens = "text_tokens"
        case draftLength = "draft_length"
        case evalId = "eval_id"
        case datasetName = "dataset_name"
        case maxContext = "max_context"
        case numImages = "num_images"
        case audioDurationMs = "audio_duration_ms"
        case audioSampleRateHz = "audio_sample_rate_hz"
        case audioChannels = "audio_channels"
        case mcqChoices = "mcq_choices"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case seed
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        func count(_ key: Key) throws -> NonNegativeCount {
            try c.decode(NonNegativeCount.self, forKey: key)
        }
        func positive(_ key: Key) throws -> PositiveCount { try c.decode(PositiveCount.self, forKey: key) }
        func batch(_ key: Key) throws -> BatchSize { try c.decode(BatchSize.self, forKey: key) }
        params = switch try c.decode(BenchmarkType.self, forKey: .benchmarkType) {
        case .prefillThroughput:
            .prefillThroughput(prefillTokens: try count(.prefillTokens),
                               batchSize: try batch(.batchSize))
        case .decodeThroughput:
            .decodeThroughput(prefillTokens: try count(.prefillTokens),
                              decodeTokens: try positive(.decodeTokens),
                              batchSize: try batch(.batchSize))
        case .endToEndLatency:
            .endToEndLatency(prefillTokens: try count(.prefillTokens),
                             decodeTokens: try positive(.decodeTokens),
                             batchSize: try batch(.batchSize))
        case .maxMemoryUsage:
            .maxMemoryUsage(prefillTokens: try count(.prefillTokens))
        case .eval:
            .eval(evalId: try c.decode(EvalId.self, forKey: .evalId),
                  datasetName: try c.decode(String.self, forKey: .datasetName),
                  maxTokens: try positive(.maxTokens),
                  maxContext: try c.decodeIfPresent(PositiveCount.self, forKey: .maxContext),
                  mcqChoices: try c.decodeIfPresent([String].self, forKey: .mcqChoices),
                  temperature: try c.decode(SamplingTemperature.self, forKey: .temperature),
                  topP: try c.decodeIfPresent(TopP.self, forKey: .topP),
                  topK: try c.decodeIfPresent(TopK.self, forKey: .topK),
                  minP: try c.decodeIfPresent(MinP.self, forKey: .minP),
                  seed: try c.decodeIfPresent(SamplingSeed.self, forKey: .seed))
        case .vlThroughput:
            .vlThroughput(imageWidth: try positive(.imageWidth),
                          imageHeight: try positive(.imageHeight),
                          textTokens: try count(.textTokens),
                          decodeTokens: try positive(.decodeTokens),
                          numImages: try count(.numImages))
        case .vlMaxMemory:
            .vlMaxMemory(imageWidth: try positive(.imageWidth),
                         imageHeight: try positive(.imageHeight),
                         textTokens: try count(.textTokens),
                         decodeTokens: try c.decodeIfPresent(PositiveCount.self,
                                                             forKey: .decodeTokens),
                         numImages: try count(.numImages))
        case .vlSpecDecode:
            .vlSpecDecode(draftLength: try positive(.draftLength),
                          maxTokens: try positive(.maxTokens), batchSize: try batch(.batchSize),
                          temperature: try c.decode(SamplingTemperature.self,
                                                    forKey: .temperature))
        case .asrThroughput:
            .asrThroughput(audioDurationMs: try c.decode(AudioDurationMillis.self, forKey: .audioDurationMs),
                           audioSampleRateHz: try c.decode(AudioSampleRate.self, forKey: .audioSampleRateHz),
                           audioChannels: try c.decode(AudioChannels.self, forKey: .audioChannels),
                           textTokens: try count(.textTokens),
                           decodeTokens: try c.decode(AsrDecodeTokens.self, forKey: .decodeTokens))
        case .asrMaxMemory:
            .asrMaxMemory(audioDurationMs: try c.decode(AudioDurationMillis.self, forKey: .audioDurationMs),
                          audioSampleRateHz: try c.decode(AudioSampleRate.self, forKey: .audioSampleRateHz),
                          audioChannels: try c.decode(AudioChannels.self, forKey: .audioChannels),
                          textTokens: try count(.textTokens),
                          decodeTokens: try c.decode(AsrDecodeTokens.self, forKey: .decodeTokens))
        }
    }
}
