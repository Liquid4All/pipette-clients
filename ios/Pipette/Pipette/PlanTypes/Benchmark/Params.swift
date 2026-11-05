import Foundation

// The parameters a benchmark is identified by, without its id: pipette-mgmt's
// `crates/pipette-plan-types/src/benchmark/params.rs`.

/// Every `parameter_*` field of a definition, without the prefix. A submission declares
/// these beside the id and the server refuses a pair that does not match its catalog, so
/// it never files a run against a stale or edited definition under that id.
///
/// An eval's temperature is the one it runs at: the entry's, else the eval-id rule's. Its
/// other sampling knobs are `nil` when unpinned, left to the client.
nonisolated enum BenchmarkParams: Equatable, Sendable {
    case prefillThroughput(prefillTokens: NonNegativeCount, batchSize: BatchSize)
    case decodeThroughput(prefillTokens: NonNegativeCount, decodeTokens: PositiveCount,
                          batchSize: BatchSize)
    case endToEndLatency(prefillTokens: NonNegativeCount, decodeTokens: PositiveCount,
                         batchSize: BatchSize)
    case maxMemoryUsage(prefillTokens: NonNegativeCount)
    case eval(evalId: EvalId, datasetName: String, maxTokens: PositiveCount,
              maxContext: PositiveCount?, mcqChoices: [String]?, temperature: SamplingTemperature,
              topP: TopP?, topK: TopK?, minP: MinP?, seed: SamplingSeed?)
    case vlThroughput(imageWidth: PositiveCount, imageHeight: PositiveCount,
                      textTokens: NonNegativeCount, decodeTokens: PositiveCount,
                      numImages: NonNegativeCount)
    case vlMaxMemory(imageWidth: PositiveCount, imageHeight: PositiveCount,
                     textTokens: NonNegativeCount, decodeTokens: PositiveCount?,
                     numImages: NonNegativeCount)
    case vlSpecDecode(draftLength: PositiveCount, maxTokens: PositiveCount, batchSize: BatchSize,
                      temperature: SamplingTemperature)
    case asrThroughput(audioDurationMs: AudioDurationMillis, audioSampleRateHz: AudioSampleRate,
                       audioChannels: AudioChannels, textTokens: NonNegativeCount,
                       decodeTokens: AsrDecodeTokens)
    case asrMaxMemory(audioDurationMs: AudioDurationMillis, audioSampleRateHz: AudioSampleRate,
                      audioChannels: AudioChannels, textTokens: NonNegativeCount,
                      decodeTokens: AsrDecodeTokens)

    /// The crate's `From<&BenchmarkDefinition>`.
    init(_ definition: BenchmarkDefinition) {
        switch definition {
        case let .prefillThroughput(b):
            self = .prefillThroughput(prefillTokens: b.parameterPrefillTokens,
                                      batchSize: b.parameterBatchSize)
        case let .decodeThroughput(b):
            self = .decodeThroughput(prefillTokens: b.parameterPrefillTokens,
                                     decodeTokens: b.parameterDecodeTokens,
                                     batchSize: b.parameterBatchSize)
        case let .endToEndLatency(b):
            self = .endToEndLatency(prefillTokens: b.parameterPrefillTokens,
                                    decodeTokens: b.parameterDecodeTokens,
                                    batchSize: b.parameterBatchSize)
        case let .maxMemoryUsage(b):
            self = .maxMemoryUsage(prefillTokens: b.parameterPrefillTokens)
        case let .eval(b):
            // Every eval-id rule's temperature is in range (`EvalIdTests` checks it).
            self = .eval(evalId: b.parameterEvalId, datasetName: b.parameterDatasetName,
                         maxTokens: b.parameterMaxTokens, maxContext: b.parameterMaxContext,
                         mcqChoices: b.parameterMcqChoices,
                         temperature: (try? b.samplingTemperature()) ?? .greedy,
                         topP: b.parameterTopP, topK: b.parameterTopK, minP: b.parameterMinP,
                         seed: b.parameterSeed)
        case let .vlThroughput(b):
            self = .vlThroughput(imageWidth: b.parameterImageWidth,
                                 imageHeight: b.parameterImageHeight,
                                 textTokens: b.parameterTextTokens,
                                 decodeTokens: b.parameterDecodeTokens,
                                 numImages: b.parameterNumImages)
        case let .vlMaxMemory(b):
            self = .vlMaxMemory(imageWidth: b.parameterImageWidth,
                                imageHeight: b.parameterImageHeight,
                                textTokens: b.parameterTextTokens,
                                decodeTokens: b.parameterDecodeTokens,
                                numImages: b.parameterNumImages)
        case let .vlSpecDecode(b):
            self = .vlSpecDecode(draftLength: b.parameterDraftLength,
                                 maxTokens: b.parameterMaxTokens,
                                 batchSize: b.parameterBatchSize,
                                 temperature: b.parameterTemperature)
        case let .asrThroughput(b):
            self = .asrThroughput(audioDurationMs: b.parameterAudioDurationMs,
                                  audioSampleRateHz: b.parameterAudioSampleRateHz,
                                  audioChannels: b.parameterAudioChannels,
                                  textTokens: b.parameterTextTokens,
                                  decodeTokens: b.parameterDecodeTokens)
        case let .asrMaxMemory(b):
            self = .asrMaxMemory(audioDurationMs: b.parameterAudioDurationMs,
                                 audioSampleRateHz: b.parameterAudioSampleRateHz,
                                 audioChannels: b.parameterAudioChannels,
                                 textTokens: b.parameterTextTokens,
                                 decodeTokens: b.parameterDecodeTokens)
        }
    }

    var benchmarkType: BenchmarkType {
        switch self {
        case .prefillThroughput: .prefillThroughput
        case .decodeThroughput: .decodeThroughput
        case .endToEndLatency: .endToEndLatency
        case .maxMemoryUsage: .maxMemoryUsage
        case .eval: .eval
        case .vlThroughput: .vlThroughput
        case .vlMaxMemory: .vlMaxMemory
        case .vlSpecDecode: .vlSpecDecode
        case .asrThroughput: .asrThroughput
        case .asrMaxMemory: .asrMaxMemory
        }
    }
}

/// The untagged `benchmark_params` object: the tag rides as the submission's own
/// `benchmark_type`, the crate's `declared_benchmark_params` wire form.
nonisolated extension BenchmarkParams: Encodable {
    private enum Key: String, CodingKey {
        case prefillTokens = "prefill_tokens"
        case decodeTokens = "decode_tokens"
        case batchSize = "batch_size"
        case evalId = "eval_id"
        case datasetName = "dataset_name"
        case maxTokens = "max_tokens"
        case maxContext = "max_context"
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case textTokens = "text_tokens"
        case numImages = "num_images"
        case draftLength = "draft_length"
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

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: Key.self)
        switch self {
        case let .prefillThroughput(prefill, batchSize):
            try c.encode(prefill, forKey: .prefillTokens)
            try c.encode(batchSize, forKey: .batchSize)
        case let .decodeThroughput(prefill, decode, batchSize),
             let .endToEndLatency(prefill, decode, batchSize):
            try c.encode(prefill, forKey: .prefillTokens)
            try c.encode(decode, forKey: .decodeTokens)
            try c.encode(batchSize, forKey: .batchSize)
        case let .maxMemoryUsage(prefill):
            try c.encode(prefill, forKey: .prefillTokens)
        case let .eval(evalId, datasetName, maxTokens, maxContext, mcqChoices, temperature,
                       topP, topK, minP, seed):
            try c.encode(evalId, forKey: .evalId)
            try c.encode(datasetName, forKey: .datasetName)
            try c.encode(maxTokens, forKey: .maxTokens)
            try c.encodeIfPresent(maxContext, forKey: .maxContext)
            try c.encodeIfPresent(mcqChoices, forKey: .mcqChoices)
            try c.encode(temperature, forKey: .temperature)
            try c.encodeIfPresent(topP, forKey: .topP)
            try c.encodeIfPresent(topK, forKey: .topK)
            try c.encodeIfPresent(minP, forKey: .minP)
            try c.encodeIfPresent(seed, forKey: .seed)
        case let .vlThroughput(width, height, text, decode, numImages):
            try c.encode(width, forKey: .imageWidth)
            try c.encode(height, forKey: .imageHeight)
            try c.encode(text, forKey: .textTokens)
            try c.encode(decode, forKey: .decodeTokens)
            try c.encode(numImages, forKey: .numImages)
        case let .vlMaxMemory(width, height, text, decode, numImages):
            try c.encode(width, forKey: .imageWidth)
            try c.encode(height, forKey: .imageHeight)
            try c.encode(text, forKey: .textTokens)
            try c.encodeIfPresent(decode, forKey: .decodeTokens)
            try c.encode(numImages, forKey: .numImages)
        case let .vlSpecDecode(draftLength, maxTokens, batchSize, temperature):
            try c.encode(draftLength, forKey: .draftLength)
            try c.encode(maxTokens, forKey: .maxTokens)
            try c.encode(batchSize, forKey: .batchSize)
            try c.encode(temperature, forKey: .temperature)
        case let .asrThroughput(durationMs, sampleRate, channels, text, decode),
             let .asrMaxMemory(durationMs, sampleRate, channels, text, decode):
            try c.encode(durationMs, forKey: .audioDurationMs)
            try c.encode(sampleRate, forKey: .audioSampleRateHz)
            try c.encode(channels, forKey: .audioChannels)
            try c.encode(text, forKey: .textTokens)
            try c.encode(decode, forKey: .decodeTokens)
        }
    }
}
