import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// Live `asr_throughput` on real models, to compare with pipette-cli on llama-server. Runs
/// only when `PIPETTE_ASR_PARITY` names a job list of host paths (xcodebuild passes
/// `TEST_RUNNER_PIPETTE_ASR_PARITY`). Each job writes its own result file, so a refusal is
/// data rather than a failure. `PIPETTE_ASR_PARITY_PIN` names the llama.cpp build to run
/// on, for example `llamacpp-prismml`. The test app must carry that build (`PIPETTE_LLAMA_PINS`).
/// If the variable is unset, the suite runs on main. Run the suite alone
/// (`-only-testing:PipetteTests/AsrLiveParityTests`), because a process keeps the first
/// build that any test loads.
@Suite struct AsrLiveParityTests {
    struct Jobs: Decodable { let jobs: [Job] }

    struct Job: Codable {
        let model: String
        let mmproj: String
        let audioDurationMs: Int32
        let textTokens: Int
        let sampleRateHz: Int32
        let channels: UInt16
        let decodeTokens: Int32
        let output: String
    }

    struct Outcome: Encodable {
        let job: Job
        let llamaCommit = LlamaPins.current.build.commit
        var promptTokens: UInt32?
        var audioTokens: UInt32?
        var audioTokensSource: String?
        var contextSizing: String?
        var beyondLimit: Bool?
        var ctxSize: UInt32?
        /// From a second load through the same `asrPrompt`: the run does not expose it.
        var renderedPrompt: String?
        var clientExtras: String?
        /// The clip `WavTone` writes, to hold to the CLI's `wav::tone`.
        var clipBytes: Int?
        var clipSha256: String?
        var error: String?
        var promptError: String?
    }

    nonisolated static let jobList = ProcessInfo.processInfo.environment["PIPETTE_ASR_PARITY"]
    nonisolated static let pin = ProcessInfo.processInfo.environment["PIPETTE_ASR_PARITY_PIN"]

    @Test(.enabled(if: jobList != nil)) func everyJobRunsThroughTheLiveEngine() async throws {
        if let name = Self.pin {
            try LlamaPins.load(try #require(LlamaPins.carried.first { $0.name == name }))
        }
        let list = try JSONDecoder.snake.decode(
            Jobs.self, from: Data(contentsOf: URL(filePath: try #require(Self.jobList))))
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        for job in list.jobs {
            let outcome = await Self.run(job)
            try encoder.encode(outcome).write(to: URL(filePath: job.output))
        }
    }

    static func run(_ job: Job) async -> Outcome {
        var outcome = Outcome(job: job)
        let request: RunRequest
        let benchmark: AsrThroughputBenchmark
        do {
            benchmark = AsrThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "asr_throughput_\(job.audioDurationMs)ms_"
                    + "text\(job.textTokens)_decode\(job.decodeTokens)"),
                parameterAudioDurationMs: try AudioDurationMillis(job.audioDurationMs),
                parameterAudioSampleRateHz: try AudioSampleRate(job.sampleRateHz),
                parameterAudioChannels: try AudioChannels(job.channels),
                parameterTextTokens: try NonNegativeCount(job.textTokens),
                parameterDecodeTokens: try AsrDecodeTokens(job.decodeTokens))
            let declared = try ggufVisionSpec(
                "parity/asr-GGUF", URL(filePath: job.model).lastPathComponent,
                URL(filePath: job.mmproj).lastPathComponent)
            request = RunRequest(
                runtime: .alreadyBound(Runtime.thisBuild(for: declared)),
                model: DeclaredBound(declared: declared,
                                     bound: try boundModel(declared, job.model, job.mmproj)),
                runtimeFlags: nil, benchmarkFlags: nil, benchmark: .asrThroughput(benchmark))
        } catch {
            outcome.error = "\(error)"
            return outcome
        }

        let tone = WavTone.tone(durationMs: UInt32(job.audioDurationMs),
                                sampleRateHz: UInt32(job.sampleRateHz), channels: job.channels)
        outcome.clipBytes = tone.count
        outcome.clipSha256 = SHA256.hash(data: tone).map { String(format: "%02x", $0) }.joined()

        do {
            let response = try await LlamaExecute.run(
                request, runContext: NoCorpus(root: FileManager.default.temporaryDirectory),
                readinessGate: {}, observer: noObserver())
            if case let .asrThroughput(prompt, audio, _, _, _, _, _, _) = response.resultData {
                (outcome.promptTokens, outcome.audioTokens) = (prompt, audio)
            }
            outcome.ctxSize = response.runtimeFlags?.ctxSize
            outcome.clientExtras = response.clientExtras
            if let extras = response.clientExtras,
               let object = try JSONSerialization.jsonObject(with: Data(extras.utf8)) as? [String: Any] {
                outcome.audioTokensSource = object["asr_audio_tokens_source"] as? String
                outcome.contextSizing = object["asr_context_sizing"] as? String
                outcome.beyondLimit = object["asr_beyond_model_audio_limit"] as? Bool
            }
        } catch {
            outcome.error = "\(error.localizedDescription) [\(error)]"
        }

        do {
            let model = try ResolvedGgufVisionModel(request.model.bound)
            let cell = try AsrCell(benchmark, recipe: AsrRecipe.forMmproj(model.mmproj))
            // Rendering needs no context sized for the clip: it only tokenizes text.
            outcome.renderedPrompt = try await LlamaExecute.withAsrClip(cell) { clip in
                try await LlamaExecute.withCalibratedAudio(
                    request, model, cell, flags: try LlamaRuntimeFlags.forRun(request),
                    clipPath: clip, calibrate: false
                ) { aud, _ in try await LlamaExecute.asrPrompt(aud, cell).prompt }
            }.0
        } catch {
            outcome.promptError = "\(error.localizedDescription) [\(error)]"
        }
        return outcome
    }
}

private extension JSONDecoder {
    static var snake: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }
}
