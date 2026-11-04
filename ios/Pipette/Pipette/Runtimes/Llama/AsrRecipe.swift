import Foundation
import llama

// The crate's `pipette-llamacpp/src/asr.rs` recipes and `gguf_meta.rs` projector reader: a
// chat model transcribes only when asked as it was trained, so the projector's encoder picks
// the ask. The CLI's raw-flag refusals have no counterpart: `llamacpp_ios_pipette` has none.

/// The crate's `AsrRefusalError`. All but the last two refuse before anything loads.
nonisolated enum AsrRefusalError: Error, Equatable, LocalizedError {
    case noAudioEncoder
    case noRecipe(name: String)
    case clipOnWindowEdge(recipe: String, durationMs: UInt32, sampleRateHz: UInt32,
                          windowMs: UInt32)
    case textUnconfirmed(textTokens: UInt32, withText: Int, withoutText: Int)
    case countUnconfirmed(calibrated: Int, markers: UInt32, measured: Int)

    var errorDescription: String? {
        switch self {
        case .noAudioEncoder:
            "the projector holds no audio encoder (`clip.has_audio_encoder` is not set): an "
                + "ASR cell needs a model whose projector takes audio"
        case let .noRecipe(name):
            "no ASR recipe for the audio encoder \"\(name)\": a cell times the prompt the "
                + "model's public usage asks it with, and none has been checked for this "
                + "encoder; add a recipe (crates/pipette-llamacpp/src/asr.rs and "
                + "AsrRecipe.swift) rather than time an unchecked one"
        case let .clipOnWindowEdge(recipe, durationMs, sampleRateHz, windowMs):
            "\(recipe): the clip is \(durationMs) ms at \(sampleRateHz) Hz, a whole number of "
                + "the encoder's \(windowMs) ms windows; resampled to 16 kHz it can come out a "
                + "sample or two past the last window, which the encoder cannot encode, so the "
                + "cell is refused rather than left to the resampler's rounding. Send it at "
                + "16 kHz, or at a length that is not a multiple of \(windowMs) ms"
        case let .textUnconfirmed(textTokens, withText, withoutText):
            "the engine tokenized \(withText) prompt tokens with the cell's text and "
                + "\(withoutText) without it, where the text is \(textTokens) tokens: the row "
                + "would not be the text the entry names"
        case let .countUnconfirmed(calibrated, markers, measured):
            "the engine tokenized \(measured) more tokens with the clip than without it, "
                + "where the counted \(calibrated) audio tokens and the recipe's \(markers) "
                + "audio-marker tokens make \(calibrated + Int(markers)): the count is not the "
                + "prompt's, so the context was not sized exactly"
        }
    }
}

/// The crate's `ProjectorMetadata`.
nonisolated struct AsrProjectorMetadata: Equatable, Sendable {
    var hasAudioEncoder = false
    /// Tells Qwen3-Omni (both towers) from Qwen3-ASR (audio only), which share `qwen3a`.
    var hasVisionEncoder = false
    var audioProjectorType: String?
    var projectorType: String?
    /// OpenBMB's whisper-architecture MiniCPM-o audio GGUF, which carries no `clip.*` key.
    var openbmbMinicpmoAudio = false

    var holdsAudioEncoder: Bool { hasAudioEncoder || openbmbMinicpmoAudio }

    var audioEncoderName: String? {
        audioProjectorType ?? projectorType ?? (openbmbMinicpmoAudio ? "minicpmoa" : nil)
    }

    /// `no_alloc`, so never the weights. A key of another type reads as absent, as in the
    /// crate: `gguf_get_val_*` would abort on it.
    static func read(_ path: String) throws -> Self {
        let params = gguf_init_params(no_alloc: true, ctx: nil)
        guard let ctx = gguf_init_from_file(path, params) else {
            throw RuntimeError.engine("reading the metadata of \(path) failed")
        }
        defer { gguf_free(ctx) }
        func key(_ name: String, _ type: gguf_type) -> Int64? {
            let id = gguf_find_key(ctx, name)
            return id >= 0 && gguf_get_kv_type(ctx, id) == type ? id : nil
        }
        func bool(_ name: String) -> Bool { key(name, GGUF_TYPE_BOOL).map { gguf_get_val_bool(ctx, $0) } ?? false }
        func string(_ name: String) -> String? {
            key(name, GGUF_TYPE_STRING).map { String(cString: gguf_get_val_str(ctx, $0)) }
        }
        var found = Self()
        found.hasAudioEncoder = bool("clip.has_audio_encoder")
        found.hasVisionEncoder = bool("clip.has_vision_encoder")
        found.audioProjectorType = string("clip.audio.projector_type")
        found.projectorType = string("clip.projector_type")
        // As Pipette's ASR build of llama.cpp recognizes the file.
        found.openbmbMinicpmoAudio = string("general.architecture") == "whisper"
            && gguf_find_key(ctx, "audio_pool_step") >= 0
            && gguf_find_tensor(ctx, "audio_projector.linear2.weight") >= 0
        return found
    }
}

/// How a family of audio models is asked to transcribe: the crate's `Recipe`.
nonisolated struct AsrRecipe: Equatable, Sendable {
    enum Kind: String, Sendable {
        case voxtral
        case lfm2Audio = "lfm2-audio"
        case lfm3Audio = "lfm3-audio"
        case qwen25Omni = "qwen2.5-omni"
        case gemma4
        case minicpmO = "minicpm-o"
        case qwen3Asr = "qwen3-asr"
        case qwen3Omni = "qwen3-omni"
    }

    enum Order: Sendable {
        case instructionThenClip
        case clipThenInstruction
    }

    /// A `--chat-template-kwargs` entry as the server holds it: the value is JSON text.
    struct Kwarg: Equatable, Sendable {
        let key: String
        let json: String
    }

    let kind: Kind
    /// For a model whose ASR mode is entered through a system turn.
    let system: String?
    /// Empty sends the clip alone.
    let instruction: String
    let order: Order
    /// `--chat-template`; `nil` renders the GGUF's own.
    let chatTemplate: String?
    let kwargs: [Kwarg]
    let documentedMaxAudioMs: UInt32?
    let encoderWindowMs: UInt32?
    /// Prompt tokens a clip adds beside its audio tokens: the encoder's begin/end markers.
    let audioMarkers: UInt32

    /// The crate's `AudioEncoder::from_name` and `Recipe::for_encoder`.
    static func forEncoder(_ name: String, withVision: Bool) throws -> Self {
        switch name {
        case "voxtral": .voxtral
        case "lfm2a": .lfm2Audio
        case "lfm3a": .lfm3Audio
        case "qwen2.5o", "qwen2.5oa": .qwen25Omni
        case "gemma4a": .gemma4
        case "gemma4ua": .gemma4Unified
        case "qwen3a" where withVision: .qwen3Omni
        case "qwen3a": .qwen3Asr
        case "minicpmoa": .minicpmO
        default: throw AsrRefusalError.noRecipe(name: name)
        }
    }

    static func of(_ metadata: AsrProjectorMetadata) throws -> Self {
        guard metadata.holdsAudioEncoder else { throw AsrRefusalError.noAudioEncoder }
        return try forEncoder(metadata.audioEncoderName ?? "",
                              withVision: metadata.hasVisionEncoder)
    }

    static func forMmproj(_ path: String) throws -> Self {
        let recipe = try of(AsrProjectorMetadata.read(path))
        AppLog.jobRun.info("asr recipe \(recipe.kind.rawValue) for \(path)")
        return recipe
    }

    /// The crate's `check_duration`: off 16 kHz, whole encoder windows can resample past the edge.
    func checkDuration(_ durationMs: UInt32, sampleRateHz: UInt32) throws {
        if let windowMs = encoderWindowMs, durationMs % windowMs == 0,
           sampleRateHz != WavTone.sampleRate {
            throw AsrRefusalError.clipOnWindowEdge(recipe: kind.rawValue, durationMs: durationMs,
                                                   sampleRateHz: sampleRateHz,
                                                   windowMs: windowMs)
        }
    }

    /// The crate's `beyond_documented_limit`: the row still runs, flagged.
    func beyondDocumentedLimit(_ durationMs: UInt32) -> Bool {
        guard let max = documentedMaxAudioMs, durationMs > max else { return false }
        AppLog.jobRun.warning(
            "\(kind.rawValue): the \(durationMs) ms clip is longer than the model's documented "
                + "\(max) ms audio limit; its speed and memory are measured, and the row is flagged")
        return true
    }

    /// Bytes of system and instruction text: an upper bound on their tokens.
    var textBytes: UInt32 { UInt32((system?.utf8.count ?? 0) + instruction.utf8.count) }

    /// The crate's `user_text`: the cell's text goes before the ask, on its side of the clip.
    func userText(_ text: String) -> String {
        switch (text.isEmpty, instruction.isEmpty) {
        case (true, _): instruction
        case (false, true): text
        case (false, false): "\(text)\n\(instruction)"
        }
    }

    /// As llama-server flattens a turn holding media: no newline around the marker.
    func userContent(_ userText: String, marker: String?) -> String {
        guard let marker else { return userText }
        return switch order {
        case .instructionThenClip: userText + marker
        case .clipThenInstruction: marker + userText
        }
    }

    /// `--reasoning off`'s `enable_thinking`, the `preserve_reasoning` default the main and
    /// lfm3 servers set (prismml's sets it only on request; no ASR turn reads it), then the
    /// recipe's own.
    var templateKwargs: [Kwarg] {
        [Kwarg(key: "enable_thinking", json: "false"),
         Kwarg(key: "preserve_reasoning", json: "true")] + kwargs
    }

    /// Always through Jinja, as llama-server renders every request. A `nil` `model` (tests)
    /// renders BOS/EOS empty and needs `chatTemplate`.
    func render(_ model: OpaquePointer?, userContent: String) throws -> String {
        let messages = (system.map { [(role: "system", content: $0)] } ?? [])
            + [(role: "user", content: userContent)]
        let render = { (tmpl: UnsafePointer<CChar>?) in
            try LlamaCpp.jinjaChatTemplate(tmpl, model: model, messages, enableThinking: false,
                                           kwargs: templateKwargs.map { (key: $0.key, json: $0.json) })
        }
        if let chatTemplate { return try chatTemplate.withCString(render) }
        return try render(model.flatMap { llama_model_chat_template($0, nil) })
    }
}

// MARK: - The recipes, as the crate spells them

nonisolated extension AsrRecipe {
    static let voxtral = Self(
        kind: .voxtral, system: nil, instruction: "", order: .clipThenInstruction,
        chatTemplate: AsrTemplates.voxtral,
        kwargs: [Kwarg(key: "transcription", json: "true"), Kwarg(key: "language", json: "\"en\"")],
        documentedMaxAudioMs: nil, encoderWindowMs: nil, audioMarkers: 1)

    static let lfm2Audio = Self(
        kind: .lfm2Audio, system: "Perform ASR.", instruction: "", order: .instructionThenClip,
        chatTemplate: nil, kwargs: [], documentedMaxAudioMs: nil, encoderWindowMs: nil,
        audioMarkers: 0)

    static let lfm3Audio = Self(
        kind: .lfm3Audio, system: "Perform ASR in English.", instruction: "",
        order: .instructionThenClip, chatTemplate: AsrTemplates.lfm3Asr, kwargs: [],
        documentedMaxAudioMs: nil, encoderWindowMs: nil, audioMarkers: 0)

    static let qwen25Omni = Self(
        kind: .qwen25Omni, system: "You are a speech recognition model.",
        instruction: "Transcribe the English audio into text without any punctuation marks.",
        order: .clipThenInstruction, chatTemplate: nil, kwargs: [], documentedMaxAudioMs: nil,
        encoderWindowMs: nil, audioMarkers: 2)

    /// Google's ASR prompt for a known language (`google/gemma-4-E2B-it`, "Audio"); 30 s is
    /// the documented limit, and E2B's and E4B's encoder cuts 30 s windows.
    static let gemma4 = Self(
        kind: .gemma4, system: nil,
        instruction: "Transcribe the following speech segment in English into English text."
            + "\n\nFollow these specific instructions for formatting the answer:\n"
            + "* Only output the transcription, with no newlines.\n"
            + "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point "
            + "seven, and write 3 instead of three.",
        order: .instructionThenClip, chatTemplate: nil, kwargs: [],
        documentedMaxAudioMs: 30_000, encoderWindowMs: 30_000, audioMarkers: 2)

    /// Gemma 4 12B's encoder takes the clip whole, so no length is a window edge.
    static let gemma4Unified = Self(
        kind: .gemma4, system: gemma4.system, instruction: gemma4.instruction,
        order: gemma4.order, chatTemplate: nil, kwargs: [],
        documentedMaxAudioMs: gemma4.documentedMaxAudioMs, encoderWindowMs: nil,
        audioMarkers: gemma4.audioMarkers)

    static let minicpmO = Self(
        kind: .minicpmO, system: nil,
        instruction: "Please listen to the audio carefully and transcribe it with high precision.\n",
        order: .instructionThenClip, chatTemplate: nil,
        kwargs: [Kwarg(key: "use_tts_template", json: "true")],
        documentedMaxAudioMs: nil, encoderWindowMs: nil, audioMarkers: 2)

    static let qwen3Asr = Self(
        kind: .qwen3Asr, system: nil, instruction: "", order: .instructionThenClip,
        chatTemplate: AsrTemplates.qwen3Asr,
        kwargs: [Kwarg(key: "language", json: "\"English\"")],
        documentedMaxAudioMs: nil, encoderWindowMs: nil, audioMarkers: 2)

    static let qwen3Omni = Self(
        kind: .qwen3Omni, system: nil, instruction: "Transcribe the English audio into text.",
        order: .clipThenInstruction, chatTemplate: nil, kwargs: [], documentedMaxAudioMs: nil,
        encoderWindowMs: nil, audioMarkers: 2)
}

/// The templates the CLI serves with `--chat-template`, byte for byte (`AsrRecipeTests`
/// holds them to the crate's sources).
nonisolated enum AsrTemplates {
    /// `LFM3_ASR_TEMPLATE`: the GGUF's own opens a `<think>` block.
    static let lfm3Asr = [
        #"{{- bos_token -}}"#,
        #"{%- for message in messages -%}"#,
        #"{{- '<|im_start|>' + message['role'] + '\n' -}}"#,
        #"{%- if message['content'] is string -%}{{- message['content'] -}}"#,
        #"{%- else -%}{%- for part in message['content'] -%}"#,
        #"{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}"#,
        #"{%- endfor -%}{%- endif -%}"#,
        #"{{- '<|im_end|>\n' -}}"#,
        #"{%- endfor -%}"#,
        #"{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"#,
    ].joined()

    /// `QWEN3_ASR_TEMPLATE`: the processor's always-present system turn and the
    /// `language English<asr_text>` prefill.
    static let qwen3Asr = [
        #"{{- '<|im_start|>system\n' -}}"#,
        #"{%- for message in messages -%}{%- if message['role'] == 'system' -%}"#,
        #"{%- if message['content'] is string -%}{{- message['content'] -}}"#,
        #"{%- else -%}{%- for part in message['content'] -%}"#,
        #"{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}"#,
        #"{%- endfor -%}{%- endif -%}"#,
        #"{%- endif -%}{%- endfor -%}"#,
        #"{{- '<|im_end|>\n<|im_start|>user\n' -}}"#,
        #"{%- for message in messages -%}{%- if message['role'] == 'user' -%}"#,
        #"{%- if message['content'] is string -%}{{- message['content'] -}}"#,
        #"{%- else -%}{%- for part in message['content'] -%}"#,
        #"{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}"#,
        #"{%- endfor -%}{%- endif -%}"#,
        #"{%- endif -%}{%- endfor -%}"#,
        #"{{- '<|im_end|>\n' -}}"#,
        #"{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}"#,
        #"{%- if language is defined and language -%}"#,
        #"{{- 'language ' + language + '<asr_text>' -}}"#,
        #"{%- endif -%}{%- endif -%}"#,
    ].joined()

    /// `asr_templates/voxtral.jinja`, its trailing newline included: the published GGUFs
    /// carry a text-only Devstral template.
    static let voxtral = #"""
{{- bos_token -}}
{%- for message in messages -%}
    {%- if message.role == 'user' -%}
        {{- '[INST]' -}}
        {%- if message.content is string -%}
            {{- message.content -}}
        {%- else -%}
            {%- for part in message.content -%}
                {{- part.text -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '[/INST]' -}}
        {%- if transcription -%}
            {%- if language -%}{{- 'lang:' + language -}}{%- endif -%}
            {{- '[TRANSCRIBE]' -}}
        {%- endif -%}
    {%- elif message.role == 'assistant' -%}
        {{- message.content + eos_token -}}
    {%- else -%}
        {{- raise_exception('Voxtral audio evaluation requires user/assistant turns.') -}}
    {%- endif -%}
{%- endfor -%}

"""#
}
