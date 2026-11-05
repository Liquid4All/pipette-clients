import Foundation
import Testing
import llama

@testable import Pipette

/// `AsrRecipe` against the crate's `asr.rs` tests, its templates against the crate's sources.
@Suite struct AsrRecipeTests {

    /// Gemma 4's model card prompt spelled out, so the recipe is checked against the card.
    private static let googleAsrPrompt = "Transcribe the following speech segment in English "
        + "into English text.\n\nFollow these specific instructions for formatting the "
        + "answer:\n* Only output the transcription, with no newlines.\n* When transcribing "
        + "numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 "
        + "instead of three."

    @Test(arguments: [
        ("lfm2a", false, "lfm2-audio", "Perform ASR.", ""),
        ("lfm3a", false, "lfm3-audio", "Perform ASR in English.", ""),
        ("qwen2.5o", true, "qwen2.5-omni", "You are a speech recognition model.",
         "Transcribe the English audio into text without any punctuation marks."),
        ("qwen2.5oa", false, "qwen2.5-omni", "You are a speech recognition model.",
         "Transcribe the English audio into text without any punctuation marks."),
        ("voxtral", false, "voxtral", nil, ""),
        ("gemma4a", true, "gemma4", nil, googleAsrPrompt),
        ("gemma4ua", true, "gemma4", nil, googleAsrPrompt),
        ("qwen3a", false, "qwen3-asr", nil, ""),
        ("qwen3a", true, "qwen3-omni", nil, "Transcribe the English audio into text."),
        ("minicpmoa", false, "minicpm-o", nil,
         "Please listen to the audio carefully and transcribe it with high precision.\n"),
    ] as [(String, Bool, String, String?, String)])
    func theProjectorPicksTheRecipe(_ projector: String, _ withVision: Bool, _ name: String,
                                    _ system: String?, _ instruction: String) throws {
        let recipe = try AsrRecipe.forEncoder(projector, withVision: withVision)
        #expect(recipe.kind.rawValue == name)
        #expect(recipe.system == system)
        #expect(recipe.instruction == instruction)
    }

    @Test(arguments: ["ultravox", "qwen2a", ""])
    func anEncoderWithoutARecipeIsRefused(_ projector: String) {
        #expect(throws: AsrRefusalError.noRecipe(name: projector)) {
            try AsrRecipe.forEncoder(projector, withVision: false)
        }
    }

    /// The crate's `audio_encoder_name` precedence, and an image-only projector refused.
    @Test func theMetadataNamesTheEncoder() throws {
        #expect(throws: AsrRefusalError.noAudioEncoder) {
            try AsrRecipe.of(AsrProjectorMetadata(hasVisionEncoder: true, projectorType: "gemma3"))
        }
        let split = AsrProjectorMetadata(hasAudioEncoder: true, hasVisionEncoder: true,
                                         audioProjectorType: "qwen3a", projectorType: "qwen3vl")
        #expect(try AsrRecipe.of(split).kind == .qwen3Omni)
        let legacy = AsrProjectorMetadata(hasAudioEncoder: true, projectorType: "voxtral")
        #expect(try AsrRecipe.of(legacy).kind == .voxtral)
        #expect(try AsrRecipe.of(AsrProjectorMetadata(openbmbMinicpmoAudio: true)).kind == .minicpmO)
    }

    /// The crate's `check_duration` cases: only `gemma4a`, only off 16 kHz, only on an edge.
    @Test(arguments: [
        ("gemma4a", UInt32(2_500), UInt32(16_000), true),
        ("gemma4a", 30_000, 16_000, true),
        ("gemma4a", 120_000, 16_000, true),
        ("gemma4a", 30_000, 48_000, false),
        ("gemma4a", 60_000, 48_000, false),
        ("gemma4a", 30_001, 48_000, true),
        ("gemma4ua", 30_000, 48_000, true),
        ("voxtral", 30_000, 48_000, true),
    ])
    func aClipOnAWindowEdgeIsRefused(_ projector: String, _ durationMs: UInt32,
                                     _ rate: UInt32, _ accepted: Bool) throws {
        let recipe = try AsrRecipe.forEncoder(projector, withVision: false)
        if accepted {
            try recipe.checkDuration(durationMs, sampleRateHz: rate)
        } else {
            #expect(throws: AsrRefusalError.clipOnWindowEdge(
                recipe: "gemma4", durationMs: durationMs, sampleRateHz: rate, windowMs: 30_000)) {
                try recipe.checkDuration(durationMs, sampleRateHz: rate)
            }
        }
    }

    @Test func beyondTheLimitIsGemmasPast30Seconds() {
        #expect(!AsrRecipe.gemma4.beyondDocumentedLimit(30_000))
        #expect(AsrRecipe.gemma4.beyondDocumentedLimit(30_001))
        #expect(AsrRecipe.gemma4Unified.beyondDocumentedLimit(119_900))
        #expect(!AsrRecipe.voxtral.beyondDocumentedLimit(600_000))
    }

    @Test func theUserTextJoinsTheTextBeforeTheAsk() {
        #expect(AsrRecipe.qwen25Omni.userText("") == AsrRecipe.qwen25Omni.instruction)
        #expect(AsrRecipe.lfm2Audio.userText("text") == "text")
        #expect(AsrRecipe.qwen25Omni.userText("text")
            == "text\n" + AsrRecipe.qwen25Omni.instruction)
        #expect(AsrRecipe.lfm2Audio.userText("") == "")
    }

    /// No newline around the marker, as llama-server flattens a turn holding media.
    @Test func theMarkerGoesOnTheRecipesSide() {
        #expect(AsrRecipe.gemma4.userContent("ask", marker: "<m>") == "ask<m>")
        #expect(AsrRecipe.qwen3Omni.userContent("ask", marker: "<m>") == "<m>ask")
        #expect(AsrRecipe.qwen3Omni.userContent("ask", marker: nil) == "ask")
    }

    /// `text_chars`: bytes, not characters.
    @Test func theTextBytesAreTheSystemAndInstruction() {
        #expect(AsrRecipe.lfm2Audio.textBytes == 12)
        #expect(AsrRecipe.voxtral.textBytes == 0)
        #expect(AsrRecipe.qwen25Omni.textBytes == UInt32(35 + 69))
    }

    /// JSON text, as llama-server's `item.value().dump()` holds them.
    @Test func theKwargsAreTheServersJsonText() {
        #expect(AsrRecipe.voxtral.templateKwargs == [
            .init(key: "enable_thinking", json: "false"),
            .init(key: "preserve_reasoning", json: "true"),
            .init(key: "transcription", json: "true"),
            .init(key: "language", json: "\"en\""),
        ])
        #expect(AsrRecipe.minicpmO.kwargs == [.init(key: "use_tts_template", json: "true")])
        #expect(AsrRecipe.qwen3Asr.kwargs == [.init(key: "language", json: "\"English\"")])
    }

    /// The CLI's `CtxInputs::ctx_size`, with no floor; the upper bound before the clip is counted.
    @Test func theContextIsSizedAsTheCli() throws {
        let benchmark = BenchmarkDefinition.asrThroughput(AsrThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "asr_throughput_15000ms_text1024_decode256"),
            parameterAudioDurationMs: try AudioDurationMillis(15_000),
            parameterAudioSampleRateHz: try AudioSampleRate(16_000),
            parameterAudioChannels: try AudioChannels(1),
            parameterTextTokens: 1024, parameterDecodeTokens: try AsrDecodeTokens(256)))
        #expect(LlamaRuntimeFlags.ctxSize(for: benchmark, perImageTokens: 188,
                                          recipeTextBytes: 12) == 188 + 1024 + 12 + 256 + 64)
        #expect(LlamaRuntimeFlags.ctxSize(for: benchmark) == 750 + 1024 + 256 + 64)
    }

    // MARK: - Templates

    private static let crate = URL(filePath: #filePath)
        .deletingLastPathComponent()   // PipetteTests/
        .deletingLastPathComponent()   // ios/Pipette/
        .deletingLastPathComponent()   // ios/
        .deletingLastPathComponent()
        .appending(path: "crates/pipette-llamacpp/src")

    /// A `concat!` of string literals in the crate's source, unescaped.
    private func rustConcat(_ name: String) throws -> String {
        let source = try String(contentsOf: Self.crate.appending(path: "asr.rs"), encoding: .utf8)
        let start = try #require(source.range(of: "const \(name): &str = concat!(\n"))
        let end = try #require(source.range(of: "\n);", range: start.upperBound ..< source.endIndex))
        return try source[start.upperBound ..< end.lowerBound]
            .split(separator: "\n")
            .map { line in
                let literal = line.trimmingCharacters(in: .whitespaces)
                let body = try #require(literal.hasPrefix("\"") && literal.hasSuffix("\",")
                    ? literal.dropFirst().dropLast(2) : nil)
                return unescape(String(body))
            }
            .joined()
    }

    private func unescape(_ literal: String) -> String {
        var out = ""
        var escaped = false
        for c in literal {
            if escaped {
                out.append(c == "n" ? "\n" : c)
                escaped = false
            } else if c == "\\" {
                escaped = true
            } else {
                out.append(c)
            }
        }
        return out
    }

    /// The repo is on disk only beside a simulator.
    private static var isSimulator: Bool {
        #if targetEnvironment(simulator)
        true
        #else
        false
        #endif
    }

    /// The copies cannot drift from what the CLI serves.
    @Test(.enabled(if: isSimulator)) func theTemplatesAreTheCratesByteForByte() throws {
        #expect(try AsrTemplates.lfm3Asr == rustConcat("LFM3_ASR_TEMPLATE"))
        #expect(try AsrTemplates.qwen3Asr == rustConcat("QWEN3_ASR_TEMPLATE"))
        let voxtral = try Data(contentsOf: Self.crate.appending(path: "asr_templates/voxtral.jinja"))
        #expect(Data(AsrTemplates.voxtral.utf8) == voxtral)
    }

    /// The crate's render tests. Qwen3-ASR's prefill shows a recipe kwarg reaching the template.
    @Test func theRecipesRenderTheirNativePrompts() throws {
        #expect(try AsrRecipe.lfm3Audio.render(nil, userContent: "<__media__>")
            == "<|im_start|>system\nPerform ASR in English.<|im_end|>\n"
            + "<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n")
        #expect(try AsrRecipe.qwen3Asr.render(nil, userContent: "<__media__>")
            == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<__media__><|im_end|>\n"
            + "<|im_start|>assistant\nlanguage English<asr_text>")
        #expect(try AsrRecipe.qwen3Asr.render(nil, userContent: "")
            == "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|im_end|>\n"
            + "<|im_start|>assistant\nlanguage English<asr_text>")
        #expect(try AsrRecipe.voxtral.render(nil, userContent: "<__media__>")
            == "[INST]<__media__>[/INST]lang:en[TRANSCRIBE]")
    }

    /// The kwarg wins over the template's default.
    @Test func thinkingIsOff() throws {
        let template = "{% if enable_thinking %}think{% else %}plain{% endif %}"
            + "{% for m in messages %}{{ m.content }}{% endfor %}"
        let rendered = try template.withCString {
            try LlamaCpp.jinjaChatTemplate($0, model: nil, [(role: "user", content: "x")],
                                           enableThinking: false,
                                           kwargs: AsrRecipe.lfm2Audio.templateKwargs
                                               .map { (key: $0.key, json: $0.json) })
        }
        #expect(rendered == "plainx")
    }

    // MARK: - The projector's header

    private func writeProjector(_ build: (OpaquePointer) -> Void) throws -> String {
        let path = FileManager.default.temporaryDirectory
            .appending(path: "mmproj-\(UUID().uuidString).gguf").path
        let ctx = try #require(gguf_init_empty())
        defer { gguf_free(ctx) }
        build(ctx)
        #expect(gguf_write_to_file(ctx, path, true))
        return path
    }

    @Test func readsTheProjectorsAudioKeys() throws {
        let path = try writeProjector { ctx in
            gguf_set_val_bool(ctx, "clip.has_audio_encoder", true)
            gguf_set_val_bool(ctx, "clip.has_vision_encoder", true)
            gguf_set_val_str(ctx, "clip.audio.projector_type", "gemma4a")
            gguf_set_val_str(ctx, "clip.projector_type", "gemma4v")
        }
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try AsrProjectorMetadata.read(path) == AsrProjectorMetadata(
            hasAudioEncoder: true, hasVisionEncoder: true, audioProjectorType: "gemma4a",
            projectorType: "gemma4v"))
        #expect(try AsrRecipe.forMmproj(path) == .gemma4)
    }

    /// As the crate's reader; gguf would abort on the wrong getter.
    @Test func aKeyOfAnotherTypeIsAbsent() throws {
        let path = try writeProjector { ctx in
            gguf_set_val_str(ctx, "clip.has_audio_encoder", "true")
            gguf_set_val_u32(ctx, "clip.projector_type", 7)
        }
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try AsrProjectorMetadata.read(path) == AsrProjectorMetadata())
        #expect(throws: AsrRefusalError.noAudioEncoder) { try AsrRecipe.forMmproj(path) }
    }

    /// No `clip.*` key: architecture, pool-step key and projector tensor must all match.
    @Test(arguments: [true, false])
    func recognizesOpenbmbsWhisperFile(_ withTensor: Bool) throws {
        let tensors = try #require(ggml_init(ggml_init_params(mem_size: 1 << 16, mem_buffer: nil,
                                                              no_alloc: true)))
        defer { ggml_free(tensors) }
        let path = try writeProjector { ctx in
            gguf_set_val_str(ctx, "general.architecture", "whisper")
            gguf_set_val_u32(ctx, "audio_pool_step", 2)
            let tensor = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 4)
            ggml_set_name(tensor, withTensor ? "audio_projector.linear2.weight" : "other.weight")
            gguf_add_tensor(ctx, tensor)
        }
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(try AsrProjectorMetadata.read(path).openbmbMinicpmoAudio == withTensor)
        if withTensor { #expect(try AsrRecipe.forMmproj(path) == .minicpmO) }
    }

    @Test func aMissingProjectorFailsTheRead() {
        #expect(throws: RuntimeError.self) {
            try AsrProjectorMetadata.read("/nonexistent/mmproj.gguf")
        }
    }
}
