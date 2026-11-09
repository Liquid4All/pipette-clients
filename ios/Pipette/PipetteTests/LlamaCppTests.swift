import Foundation
import Testing

import llama
@testable import Pipette

/// Only the pure mappings; the ops that need a loaded model are exercised on device.
struct LlamaCppTests {
    /// The Rust and Kotlin mirrors pin the same words: three spellings of one value drift
    /// silently, and the failure is a run that loads one mode and reports another.
    @Test func everyLoadModeSpellsItsPlanWord() throws {
        #expect(LlamacppLoadMode.allCases.map(\.rawValue)
                == ["none", "mmap", "mlock", "mmap_mlock", "direct_io"])
        #expect(LlamacppLoadMode.allCases.count == 5)
    }

    /// The engine sets this on `llama_model_params`, so a swapped pair would load one mode
    /// and report another with every other test still green.
    @Test func everyLoadModeMapsToItsLlamaDiscriminant() throws {
        #expect(LlamacppLoadMode.read.llamaValue == LLAMA_LOAD_MODE_NONE)
        #expect(LlamacppLoadMode.mmap.llamaValue == LLAMA_LOAD_MODE_MMAP)
        #expect(LlamacppLoadMode.mlock.llamaValue == LLAMA_LOAD_MODE_MLOCK)
        #expect(LlamacppLoadMode.mmapMlock.llamaValue == LLAMA_LOAD_MODE_MMAP_MLOCK)
        #expect(LlamacppLoadMode.directIo.llamaValue == LLAMA_LOAD_MODE_DIRECT_IO)
    }

    // MARK: - The thread count a cell runs with

    /// Matches llama.cpp's own default (`common.cpp`'s `common_cpu_get_num_physical_cores`,
    /// `hw.perflevel0.physicalcpu`), so a cell schedules its work the way the stock tools would.
    @Test(arguments: [(cores: 2, expected: UInt32(2)), (cores: 6, expected: UInt32(6)),
                      (cores: 12, expected: UInt32(12))])
    func theThreadCountIsThePerformanceCoreCount(_ c: (cores: Int, expected: UInt32)) {
        #expect(LlamaCpp.threadCount(performanceCores: c.cores, activeProcessors: 10) == c.expected)
    }

    /// The simulator reports no performance level; the llama.swiftui fallback stands in
    /// rather than a count of zero.
    @Test(arguments: [(processors: 10, expected: UInt32(8)), (processors: 6, expected: UInt32(4)),
                      (processors: 2, expected: UInt32(1)), (processors: 1, expected: UInt32(1))])
    func anUnreadableSysctlFallsBackToTheOlderCap(_ c: (processors: Int, expected: UInt32)) {
        #expect(LlamaCpp.threadCount(performanceCores: nil,
                                     activeProcessors: c.processors) == c.expected)
        // A kernel that answers zero is no answer either.
        #expect(LlamaCpp.threadCount(performanceCores: 0,
                                     activeProcessors: c.processors) == c.expected)
    }

    // MARK: - The eval sampler

    /// Same grammar the crate writes and pins.
    @Test func theMcqGrammarAdmitsExactlyTheChoices() {
        #expect(LlamaCpp.mcqGrammar(["A"]) == #"root ::= "A""#)
        #expect(LlamaCpp.mcqGrammar(["A", "B", "C"]) == #"root ::= "A" | "B" | "C""#)
    }

    /// A choice holding a quote or a backslash stays one literal, so it cannot end the
    /// rule early and admit tokens the cell never offered.
    @Test func aQuotedChoiceStaysOneLiteral() {
        #expect(LlamaCpp.mcqGrammar([#"say "hi""#]) == #"root ::= "say \"hi\"""#)
        #expect(LlamaCpp.mcqGrammar([#"a\b"#]) == #"root ::= "a\\b""#)
    }

    /// `llama-server` samples greedily at or below zero; the other engines share the rule.
    @Test(arguments: [(temperature: 0.0, draws: false), (-1.0, false),
                      (0.6, true), (1.0, true)])
    func theTemperatureDecidesWhetherASampleIsDrawn(_ c: (temperature: Double, draws: Bool)) {
        #expect(drawsSample(atTemperature: c.temperature) == c.draws)
    }

    // MARK: - Classifying a failure

    /// `tensor not allocated` is the assert the crate's llama.cpp eval keys on; without it a
    /// model that did not fit fails with the generic engine error and no actionable hint.
    @Test(arguments: ["ggml_backend_metal_buffer_type_alloc_buffer: failed to allocate buffer",
                      "posix_memalign failed", "MTLCompiler: out of memory",
                      "ggml-backend.cpp:348: GGML_ASSERT(tensor->data != NULL && \"tensor not allocated\") failed"])
    func anAllocationFailureIsClassifiedAsOutOfMemory(_ log: String) {
        #expect(llamaLogIsOOM(log))
    }

    @Test(arguments: ["llama_model_load: error loading model: tensor 'output.weight' not found",
                      "unknown model architecture", ""])
    func anotherFailureIsNotClassifiedAsOutOfMemory(_ log: String) {
        #expect(!llamaLogIsOOM(log))
        #expect(!llamaLogIsUnreadableFile(log))
    }

    /// The crate's `missing_file` and `truncated_file` lines, so a bad file reads as the CLI
    /// files it.
    @Test(arguments: [
        "gguf_init_from_file: failed to open GGUF file '/m/q.gguf' (No such file or directory)",
        "gguf_init_from_file_impl: failed to read magic",
        "llama_model_load: error loading model: tensor 'output.weight' data is not within "
            + "the file bounds, model is corrupted or incomplete",
    ])
    func aFileLlamaCannotReadIsClassified(_ log: String) {
        #expect(llamaLogIsUnreadableFile(log))
        #expect(!llamaLogIsOOM(log))
    }

    /// `llama_decode` answering `1` means it could not find a KV slot for the batch: the
    /// context filled, which llama.h's own advice is to *increase*. Never `.outOfMemory`,
    /// whose hint (`formatJobError`) offers a smaller context and would send a user the
    /// opposite way. Worded as `decode_error_message` in android/native/jni.cpp.
    @Test func aFullKvCacheIsAnEngineFailureNotOutOfMemory() {
        guard case let .engine(detail) = LlamaCpp.decodeFailure(1, phase: "prefill") else {
            Issue.record("rc=1 should classify as an engine failure"); return
        }
        #expect(detail.contains("prefill"))
        #expect(detail.contains("KV cache is full"))
        #expect(detail.contains("exceeds the allocated context size"))
        #expect(detail.contains("rc=1"))
    }

    @Test(arguments: [Int32(2), -1])
    func anotherDecodeCodeIsAnEngineFailure(_ rc: Int32) {
        guard case let .engine(detail) = LlamaCpp.decodeFailure(rc, phase: "decode") else {
            Issue.record("rc=\(rc) should classify as an engine failure"); return
        }
        #expect(detail.contains("rc=\(rc)"))
    }

    /// llama.h's fatal codes, below -1, on which an eval ends its cell rather than failing
    /// one sample.
    @Test(arguments: [Int32(-2), -3])
    func aFatalDecodeCodeIsABackendFailure(_ rc: Int32) {
        guard case let .backend(detail) = LlamaCpp.decodeFailure(rc, phase: "decode") else {
            Issue.record("rc=\(rc) should classify as a backend failure"); return
        }
        #expect(detail.contains("rc=\(rc)"))
    }

    // MARK: - The generate loop

    enum GenerationCall: Equatable { case sample(llama_token), decode(llama_token) }
    enum Boom: Error { case decode }

    /// Hands out 100, 101, ... from `sample` and records every call; decode number
    /// `failOnDecode` (1-based) throws `Boom.decode`. Each sample is stamped after `pause`,
    /// each decode before it, so a test can bracket the last-sample instant.
    final class GenerationRecorder {
        var calls: [GenerationCall] = []
        var failOnDecode: Int?
        var pause: TimeInterval = 0
        var sampledAt: [ContinuousClock.Instant] = []
        var decodeStartedAt: [ContinuousClock.Instant] = []
        var sampled: [llama_token] {
            calls.compactMap { call in
                guard case let .sample(t) = call else { return nil }
                return t
            }
        }
        var decoded: [llama_token] {
            calls.compactMap { call in
                guard case let .decode(t) = call else { return nil }
                return t
            }
        }

        func run(count: Int, end: GenerationEnd) throws -> LlamaGeneration {
            try LlamaCpp.generationSteps(count: count, end: end, sample: {
                let token = llama_token(100 + self.sampled.count)
                self.calls.append(.sample(token))
                Thread.sleep(forTimeInterval: self.pause)
                self.sampledAt.append(.now)
                return token
            }, decode: { token in
                self.calls.append(.decode(token))
                self.decodeStartedAt.append(.now)
                Thread.sleep(forTimeInterval: self.pause)
                if self.decoded.count == self.failOnDecode { throw Boom.decode }
            })
        }
    }

    @Test(arguments: GenerationEnd.allCases, [1, 2, 3, 256])
    func eachDecodeFeedsBackTheTokenJustSampled(_ end: GenerationEnd, _ count: Int) throws {
        let recorder = GenerationRecorder()
        let generation = try recorder.run(count: count, end: end)
        let all = (0 ..< count).flatMap { i -> [GenerationCall] in
            let t = llama_token(100 + i)
            return [.sample(t), .decode(t)]
        }
        #expect(recorder.calls == (end == .afterLastSample ? Array(all.dropLast()) : all))
        #expect(generation.tokens == count)
    }

    @Test(arguments: GenerationEnd.allCases, [0, -1])
    func aCountBelowOneIsRefusedBeforeAnyCall(_ end: GenerationEnd, _ count: Int) {
        let recorder = GenerationRecorder()
        #expect(throws: RuntimeError.engine("a generation of \(count) tokens measures nothing")) {
            _ = try recorder.run(count: count, end: end)
        }
        #expect(recorder.calls.isEmpty)
    }

    @Test(arguments: GenerationEnd.allCases)
    func theLastSampleInstantFallsBeforeItsDecode(_ end: GenerationEnd) throws {
        let recorder = GenerationRecorder()
        recorder.pause = 0.002
        let generation = try recorder.run(count: 3, end: end)
        #expect(generation.lastSampledAt >= (try #require(recorder.sampledAt.last)))
        if end == .afterLastDecode {
            #expect(generation.lastSampledAt <= (try #require(recorder.decodeStartedAt.last)))
        }
    }

    @Test(arguments: GenerationEnd.allCases)
    func theFirstSampleInstantFallsBeforeTheFirstDecode(_ end: GenerationEnd) throws {
        let recorder = GenerationRecorder()
        recorder.pause = 0.002
        let generation = try recorder.run(count: 3, end: end)
        let first = try #require(generation.firstSampledAt)
        #expect(first >= (try #require(recorder.sampledAt.first)))
        #expect(first <= (try #require(recorder.decodeStartedAt.first)))
    }

    @Test(arguments: GenerationEnd.allCases)
    func aFailedDecodeEndsTheGeneration(_ end: GenerationEnd) {
        let recorder = GenerationRecorder()
        recorder.failOnDecode = 2
        #expect(throws: Boom.decode) {
            _ = try recorder.run(count: 5, end: end)
        }
        #expect(recorder.sampled.count == 2)
        #expect(recorder.decoded.count == 2)
    }

    /// The one failure where the ends differ: a generation that stops after the last sample
    /// never runs the decode that fails.
    @Test func aFailingLastDecodeFailsOnlyTheGenerationThatRunsIt() throws {
        let decodesLast = GenerationRecorder()
        decodesLast.failOnDecode = 3
        #expect(throws: Boom.decode) {
            _ = try decodesLast.run(count: 3, end: .afterLastDecode)
        }
        #expect(decodesLast.sampled.count == 3)
        #expect(decodesLast.decoded.count == 3)
        let stopsAtSample = GenerationRecorder()
        stopsAtSample.failOnDecode = 3
        #expect(try stopsAtSample.run(count: 3, end: .afterLastSample).tokens == 3)
        #expect(stopsAtSample.decoded.count == 2)
    }
}
