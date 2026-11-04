import Foundation
import llama

// The mtmd C calls the vision cells need, as `LlamaCpp` is for text, and the Swift
// counterpart of the Android shim's `ee_mtmd_*`. mtmd's headers are in the `llama` module
// (`ios/build-llama.sh`), so `import llama` brings them. Where the crate counts one frame with
// `llama-mtmd-cli`, this counts it with the projector the cell has already loaded.

/// An explicit `free()`, as `LlamaModel` has, so hundreds of MB of projector don't wait on
/// ARC. It borrows the `llama_model` it was built from, so it must be freed before the model.
nonisolated final class MtmdContext {
    let ctx: OpaquePointer
    private var freed = false

    init(ctx: OpaquePointer) { self.ctx = ctx }

    func free() {
        guard !freed else { return }
        freed = true
        mtmd_free(ctx)
    }

    deinit { free() }
}

/// The encoder a cell's projector must hold.
nonisolated enum MtmdMedia: String, Sendable {
    case vision
    case audio
}

nonisolated enum LlamaMtmd {
    /// Headroom for the chat template's scaffolding around the prompt.
    static let ctxMargin: UInt32 = 64

    /// Each frame brings its own boundary markers, so a flat margin tips short on a
    /// multi-frame cell.
    static let ctxMarginPerImage: UInt32 = 8

    /// Every frame is a preprocessed image tensor held until the rep ends.
    static let maxNumImages: UInt32 = 4096

    nonisolated struct CtxInputs: Equatable, Sendable {
        let width: UInt32
        let height: UInt32
        let textTokens: UInt32
        let decodeTokens: UInt32
        /// At least one.
        let numImages: UInt32

        init(width: UInt32, height: UInt32, textTokens: UInt32, decodeTokens: UInt32,
             numImages: UInt32) throws {
            guard numImages <= LlamaMtmd.maxNumImages else {
                throw RuntimeError.engine(
                    "parameter_num_images \(numImages) exceeds the supported maximum "
                        + "\(LlamaMtmd.maxNumImages)")
            }
            self.width = width
            self.height = height
            self.textTokens = textTokens
            self.decodeTokens = decodeTokens
            self.numImages = max(numImages, 1)
        }

        /// `fallbackFloor` applies to the estimate alone: `vl_throughput` passes 8192, and
        /// `vl_max_memory` none, since padding its window would inflate the peak it reports.
        func ctxSize(calibrated: UInt32?, fallbackFloor: UInt32) -> UInt32 {
            let perImage = calibrated ?? fallbackPerImage(width: width, height: height)
            let floor = calibrated == nil ? fallbackFloor : 0
            return max(floor, perImage
                .multipliedSaturating(numImages)
                .addingSaturating(textTokens)
                .addingSaturating(decodeTokens)
                .addingSaturating(ctxMargin)
                .addingSaturating(ctxMarginPerImage.multipliedSaturating(numImages)))
        }
    }

    /// What the crate's `mtmd::calibrate` counts for one gray frame, marker-free.
    static func perImageTokens(_ mtmd: MtmdContext, width: UInt32, height: UInt32) throws -> Int {
        let chunks = try tokenizeGray(mtmd, text: defaultMediaMarker, width: width,
                                      height: height, images: 1)
        defer { chunks.free() }
        return chunks.imageTokens
    }

    /// The patch-14 grid bounds the tiling encoders; the floor covers the families that emit
    /// a flat count (gemma-4 at 256, InternVL's single sub-448px tile), without which a 128²
    /// cell would estimate 81 against ~256 real and overflow.
    static func fallbackPerImage(width: UInt32, height: UInt32) -> UInt32 {
        let patchDivisor: UInt32 = 14
        // gemma-4's flat count, and InternVL's single tile.
        let encoderFloor: UInt32 = 256
        return max(encoderFloor,
                   (width / patchDivisor).multipliedSaturating(height / patchDivisor))
    }

    /// Read from the library rather than restated: a mismatched marker tokenizes to pure
    /// text and would silently measure a text cell under a vision name.
    static var defaultMediaMarker: String { String(cString: mtmd_default_marker()) }

    /// mtmd's own warm-up is off because the cell runs its own at the real shape, and its
    /// timing printout for the reason the context sets `no_perf`: the cell measures its
    /// own numbers. Failures are `ModelLoadFailure`s, as the weights' are, and the log
    /// capture continues the model's so an encoder OOM is classified the same way.
    static func load(model: OpaquePointer, mmprojPath: String, useGpu: Bool,
                     threads: UInt32, media: MtmdMedia = .vision) throws -> MtmdContext {
        LlamaCpp.beginLoadCapture(reset: false)
        defer { LlamaCpp.endLoadCapture() }
        guard let ctx = pipette_llama_mtmd_init(mmprojPath, model, useGpu, Int32(threads)) else {
            throw LlamaCpp.loadFailure("mmproj load failed: \(mmprojPath)")
        }
        let mtmd = MtmdContext(ctx: ctx)
        // A projector without the cell's encoder loads fine, then tokenizes every marker to
        // nothing. An audio encoder must also name its rate, which the clip resamples to.
        let supported = switch media {
        case .vision: mtmd_support_vision(ctx)
        case .audio: mtmd_support_audio(ctx) && mtmd_get_audio_sample_rate(ctx) > 0
        }
        guard supported else {
            mtmd.free()
            throw ModelLoadFailure(
                cause: .engine("projector has no \(media.rawValue) encoder: \(mmprojPath)"))
        }
        return mtmd
    }

    /// `text` carries one media marker per image. Solid gray because content doesn't
    /// affect the token count or encode cost (`docs/methodology/vl-throughput.md`, "Token
    /// Control"). The returned chunks must be freed.
    static func tokenizeGray(_ mtmd: MtmdContext, text: String, width: UInt32, height: UInt32,
                             images: Int) throws -> VisionChunks {
        guard width > 0, height > 0 else {
            throw RuntimeError.engine("image dimensions \(width)x\(height) are not positive")
        }
        guard images > 0 else {
            throw RuntimeError.engine("vl cell asked for \(images) images")
        }
        // A caller bug; refused by name ahead of mtmd's marker-count rc=1.
        guard !text.isEmpty else {
            throw RuntimeError.engine("vl prompt is empty (it must carry a media marker)")
        }
        // The dimensions are unrestricted plan fields, so the size is computed wide and
        // refused before the allocation can overflow.
        let pixels = UInt64(width) * UInt64(height)
        guard pixels <= Self.maxPixels else {
            throw RuntimeError.engine(
                "image dimensions \(width)x\(height) out of range (max \(Self.maxPixels) pixels)")
        }
        var bitmaps: [OpaquePointer?] = []
        // Dead once `mtmd_tokenize` has preprocessed them, and at high frame counts the
        // larger allocation.
        defer { bitmaps.forEach { mtmd_bitmap_free($0) } }
        let gray = [UInt8](repeating: 128, count: Int(pixels) * 3)
        for _ in 0 ..< images {
            // `mtmd_bitmap_init` copies, so one buffer feeds every frame; separate frames let
            // a video encoder merge them itself.
            let bitmap = gray.withUnsafeBufferPointer {
                mtmd_bitmap_init(width, height, $0.baseAddress)
            }
            guard let bitmap else { throw RuntimeError.engine("mtmd_bitmap_init failed") }
            bitmaps.append(bitmap)
        }

        return try tokenize(mtmd, text: text, bitmaps: &bitmaps, addSpecial: true, media: .vision)
    }

    /// The corpus counterpart of `tokenizeGray`, where the image content is the workload.
    ///
    /// A legacy render is tokenized without `addSpecial` only so prompts already measured
    /// don't move; for a vocabulary that adds a BOS this leaves out the one the server adds,
    /// a known gap. The returned chunks must be freed.
    static func tokenizeFiles(_ mtmd: MtmdContext, text: String, paths: [String],
                              addSpecial: Bool) throws -> VisionChunks {
        guard !text.isEmpty else {
            throw RuntimeError.engine("vl prompt is empty (it must carry a media marker)")
        }
        var bitmaps: [OpaquePointer?] = []
        defer { bitmaps.forEach { mtmd_bitmap_free($0) } }
        for path in paths {
            // No placeholder: an unreadable corpus image fails the sample.
            guard let bitmap = pipette_llama_mtmd_bitmap_from_file(mtmd.ctx, path) else {
                throw RuntimeError.engine("failed to read the corpus image \(path)")
            }
            bitmaps.append(bitmap)
        }

        return try tokenize(mtmd, text: text, bitmaps: &bitmaps, addSpecial: addSpecial,
                            media: .vision)
    }

    /// Decodes the WAV through mtmd's own helper, as llama-server decodes an `input_audio`
    /// part. The returned chunks must be freed.
    static func tokenizeClip(_ mtmd: MtmdContext, text: String, clipPath: String) throws -> VisionChunks {
        guard !text.isEmpty else {
            throw RuntimeError.engine("asr prompt is empty (it must carry a media marker)")
        }
        // No placeholder: the server hashes and decodes the clip it sends.
        guard let bitmap = pipette_llama_mtmd_bitmap_from_file(mtmd.ctx, clipPath) else {
            throw RuntimeError.engine("failed to read the clip \(clipPath)")
        }
        var bitmaps: [OpaquePointer?] = [bitmap]
        defer { mtmd_bitmap_free(bitmap) }
        return try tokenize(mtmd, text: text, bitmaps: &bitmaps, addSpecial: true, media: .audio)
    }

    /// As llama-server counts a clip-free prompt's `prompt_n`.
    static func promptTokens(_ mtmd: MtmdContext, text: String) throws -> Int {
        var bitmaps: [OpaquePointer?] = []
        let chunks = try tokenize(mtmd, text: text, bitmaps: &bitmaps, addSpecial: true,
                                  media: .audio)
        defer { chunks.free() }
        return chunks.promptTokens
    }

    /// What the crate's `asr::calibrate` counts for the cell's clip, marker-free.
    static func clipAudioTokens(_ mtmd: MtmdContext, clipPath: String) throws -> Int {
        let chunks = try tokenizeClip(mtmd, text: defaultMediaMarker, clipPath: clipPath)
        defer { chunks.free() }
        return chunks.audioTokens
    }

    private static func tokenize(_ mtmd: MtmdContext, text: String,
                                 bitmaps: inout [OpaquePointer?], addSpecial: Bool,
                                 media: MtmdMedia) throws -> VisionChunks {
        guard let chunks = mtmd_input_chunks_init() else {
            throw RuntimeError.engine("mtmd_input_chunks_init failed")
        }
        let rc = text.withCString { cText -> Int32 in
            var input = mtmd_input_text()
            // `mtmd_tokenize` reads exactly `text_len` bytes and never falls back to `strlen`.
            input.text = cText
            input.text_len = text.utf8.count
            input.add_special = addSpecial
            input.parse_special = true
            return pipette_llama_mtmd_tokenize(mtmd.ctx, chunks, &input, &bitmaps, bitmaps.count)
        }
        guard rc == 0 else {
            mtmd_input_chunks_free(chunks)
            throw RuntimeError.engine(Self.tokenizeFailure(rc, media))
        }
        return VisionChunks(promptTokens: mtmd_helper_get_n_tokens(chunks),
                            imageTokens: Self.mediaTokens(chunks, MTMD_INPUT_CHUNK_TYPE_IMAGE),
                            audioTokens: Self.mediaTokens(chunks, MTMD_INPUT_CHUNK_TYPE_AUDIO),
                            handle: chunks,
                            release: { mtmd_input_chunks_free(chunks) })
    }

    /// What upstream's server hands its speculator (`server_tokens::get_text_tokens`).
    /// Shorter than `n_past`, which is correct: the speculator takes the position separately.
    static func textTokens(_ chunks: VisionChunks) -> [Int32] {
        guard let handle = chunks.handle else { return [] }
        var tokens: [Int32] = []
        for index in 0 ..< mtmd_input_chunks_size(handle) {
            guard let chunk = mtmd_input_chunks_get(handle, index),
                  mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT
            else { continue }
            var count = 0
            guard let ids = mtmd_input_chunk_get_tokens_text(chunk, &count) else { continue }
            tokens.append(contentsOf: UnsafeBufferPointer(start: ids, count: count))
        }
        return tokens
    }

    /// Returns the position the prefill left, which the corpus turns hand the speculator.
    static func evalChunks(_ mtmd: MtmdContext, _ m: LlamaModel, _ chunks: VisionChunks,
                           from nPast: Int32) throws -> Int32 {
        guard let handle = chunks.handle else {
            throw RuntimeError.engine("\(Self.media(chunks)) chunks have no engine handle")
        }
        var newPast: llama_pos = nPast
        let nBatch = Int32(llama_n_batch(m.ctx))
        let rc = mtmd_helper_eval_chunks(mtmd.ctx, m.ctx, handle,
                                         nPast, /* seq_id */ 0, nBatch,
                                         /* logits_last */ true, &newPast)
        guard rc == 0 else { throw LlamaCpp.decodeFailure(rc, phase: "\(Self.media(chunks)) prefill") }
        llama_synchronize(m.ctx)
        return newPast
    }

    /// From position 0: the caller clears the cache first.
    static func evalChunks(_ mtmd: MtmdContext, _ m: LlamaModel, _ chunks: VisionChunks) throws {
        guard let handle = chunks.handle else {
            throw RuntimeError.engine("\(Self.media(chunks)) chunks have no engine handle")
        }
        var newPast: llama_pos = 0
        // `llama_n_batch`, not `n_ctx`: `LlamaCpp.load` caps `n_batch` below the context,
        // and the helper's sub-batched `llama_decode` calls fail for anything above it.
        let nBatch = Int32(llama_n_batch(m.ctx))
        let rc = mtmd_helper_eval_chunks(mtmd.ctx, m.ctx, handle,
                                         /* n_past */ 0, /* seq_id */ 0, nBatch,
                                         /* logits_last */ true, &newPast)
        guard rc == 0 else { throw LlamaCpp.decodeFailure(rc, phase: "\(Self.media(chunks)) prefill") }
        // So the caller's wall clock measures compute, not submission.
        llama_synchronize(m.ctx)
    }

    /// Summed per media chunk rather than against a text-only baseline, so the count is
    /// exact and marker-free (`docs/methodology/vl-image-tokens.md`).
    private static func mediaTokens(_ chunks: OpaquePointer, _ type: mtmd_input_chunk_type) -> Int {
        var total = 0
        for index in 0 ..< mtmd_input_chunks_size(chunks) {
            guard let chunk = mtmd_input_chunks_get(chunks, index),
                  mtmd_input_chunk_get_type(chunk) == type
            else { continue }
            total += mtmd_input_chunk_get_n_tokens(chunk)
        }
        return total
    }

    /// ASR and VL share the prefill, so its errors name the media the chunks carry.
    private static func media(_ chunks: VisionChunks) -> MtmdMedia {
        chunks.audioTokens > 0 ? .audio : .vision
    }

    private static func tokenizeFailure(_ rc: Int32, _ media: MtmdMedia) -> String {
        let noun = media == .vision ? "image" : "audio"
        return switch rc {
        case 1: "mtmd_tokenize failed: the marker count does not match the \(noun) count (rc=1)"
        case 2: "mtmd_tokenize failed: \(noun) preprocessing error (rc=2)"
        default: "mtmd_tokenize failed (rc=\(rc))"
        }
    }

    /// `pipette_ops::png::gray`'s bound, far past any real encoder input. It is not the
    /// device's: a frame near it is ~192 MB of RGB, enough to exhaust a phone.
    private static let maxPixels: UInt64 = 64 * 1024 * 1024
}
