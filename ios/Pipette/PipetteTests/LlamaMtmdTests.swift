import Foundation
import Testing

@testable import Pipette

/// Sizing a vision cell's context: the first half of `LlamaMtmd`, as
/// `crates/pipette-llamacpp/src/mtmd.rs` tests it.
struct LlamaMtmdTests {
    /// The 256² workload the crate's sizing tests share.
    private func inputs(textTokens: UInt32 = 0, decodeTokens: UInt32,
                        numImages: UInt32) throws -> LlamaMtmd.CtxInputs {
        try LlamaMtmd.CtxInputs(width: 256, height: 256, textTokens: textTokens,
                                decodeTokens: decodeTokens, numImages: numImages)
    }

    /// Frames scale the image content *and* the per-image margin: a flat margin tips short
    /// on a multi-frame cell, whose template repeats its boundary markers per frame.
    @Test func ctxMarginScalesWithFrameCount() throws {
        let at = { (numImages: UInt32) in
            try self.inputs(decodeTokens: 64, numImages: numImages)
                .ctxSize(calibrated: 64, fallbackFloor: 0)
        }
        #expect(try at(1) == 64 + 64 + LlamaMtmd.ctxMargin + LlamaMtmd.ctxMarginPerImage)
        let f80 = try at(80)
        #expect(f80 == 64 * 80 + 64 + LlamaMtmd.ctxMargin + LlamaMtmd.ctxMarginPerImage * 80)
        // The live failure shape: measured ~330 scaffolding tokens at 80 frames must fit
        // inside the margin the formula reserves.
        #expect(f80 - (64 * 80 + 64) >= 330, "margin too small: \(f80)")
    }

    /// As the crate reads it (`num_images.max(1)`).
    @Test func aZeroFrameCountReadsAsOne() throws {
        #expect(try inputs(decodeTokens: 0, numImages: 0)
            == inputs(decodeTokens: 0, numImages: 1))
    }

    /// One frame's token estimate, since an undersized estimate is a vision cell that
    /// overflows its context and measures nothing. The patch-14 grid rules above the floor;
    /// the floor covers the encoders that emit a flat count regardless of input size, and
    /// the sub-patch dimensions the grid would map to zero.
    @Test(arguments: [
        (UInt32(128), UInt32(128), UInt32(256)),   // grid would say 81: floored
        (UInt32(8), UInt32(8), UInt32(256)),       // sub-patch: floored
        (UInt32(256), UInt32(256), UInt32(324)),   // 18 × 18
        (UInt32(512), UInt32(512), UInt32(1296)),  // 36 × 36
    ])
    func fallbackPerImageFloorsFlatEncoders(_ width: UInt32, _ height: UInt32,
                                            _ perImage: UInt32) {
        #expect(LlamaMtmd.fallbackPerImage(width: width, height: height) == perImage)
    }

    /// One policy for both runners: the count × frames + text + decode + the same margins
    /// either way; the floor applies to the estimate alone.
    @Test(arguments: [
        // The estimate: 324 × 5 + 10 + 64 + 64 + 8 × 5.
        (UInt32?.none, UInt32(0), UInt32(1798)),
        (UInt32?.none, UInt32(8192), UInt32(8192)),
        // The count, unfloored: 64 × 5 + 10 + 64 + 64 + 8 × 5.
        (UInt32?.some(64), UInt32(8192), UInt32(498)),
    ] as [(UInt32?, UInt32, UInt32)])
    func ctxSizeAppliesSharedMargins(_ calibrated: UInt32?, _ floor: UInt32,
                                     _ expected: UInt32) throws {
        #expect(try inputs(textTokens: 10, decodeTokens: 64, numImages: 5)
            .ctxSize(calibrated: calibrated, fallbackFloor: floor) == expected)
    }
}
