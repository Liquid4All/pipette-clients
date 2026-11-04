import Foundation
import Testing

@testable import Pipette

/// Unit tests for `LlamaExecute.vlMaxMemory`'s workload, driven against the fake
/// `VisionInference` witness `VlThroughputTests` uses, so no model and no projector load.
/// The peak itself is `ProcessMemory`'s, tested there and on a device.
@Suite struct VlMaxMemoryTests {
    private typealias FakeVision = VlThroughputTests.FakeVision

    private func run(_ vis: VisionInference, width: PositiveCount = 224,
                     height: PositiveCount = 224, text: NonNegativeCount = 0,
                     decode: PositiveCount? = nil,
                     images: NonNegativeCount = 1) async throws {
        try await LlamaExecute.vlMaxMemory(
            VlMaxMemoryBenchmark(
                benchmarkId: BenchmarkId(validated: "vl_max_memory_\(width)x\(height)"),
                parameterImageWidth: width, parameterImageHeight: height,
                parameterTextTokens: text, parameterDecodeTokens: decode,
                parameterNumImages: images),
            vis)
    }

    /// No warm-up or repeat, as the crate's single request: a second pass in the same
    /// process would start from the first one's buffers. Without a decode count it decodes
    /// one token, as the text cell's `-pg <N>,1` does.
    @Test func runsOnePrefillAndOneDecodeStep() async throws {
        let fake = FakeVision()
        try await run(fake.make())
        #expect(fake.tokenizeChunksCount == 1)
        #expect(fake.evalCount == 1)
        #expect(fake.generateCount == 1)
        #expect(fake.lastGenerateCount == 1)
        #expect(fake.generateEnds == [.afterLastDecode])
    }

    @Test func decodesTheCellsDecodeCount() async throws {
        let fake = FakeVision()
        try await run(fake.make(), decode: 256)
        #expect(fake.evalCount == 1)
        #expect(fake.generateCount == 1)
        #expect(fake.lastGenerateCount == 256)
        #expect(fake.generateEnds == [.afterLastDecode])
    }

    /// The same message `vl_throughput` sends: the text, then one marker per frame, inside
    /// one templated user turn.
    @Test(arguments: [NonNegativeCount(1), 5])
    func sendsVlThroughputsMessage(_ images: NonNegativeCount) async throws {
        let fake = FakeVision()
        try await run(fake.make(), images: images)
        let markers = String(repeating: "<__media__>", count: Int(images.value))
        #expect(fake.prompts == ["<|user|>\(markers)<|assistant|>"])
    }

    @Test func forwardsTheCellsShapeToTheEngine() async throws {
        let fake = FakeVision()
        try await run(fake.make(), width: 512, height: 384, text: 12, images: 3)
        #expect(fake.lastWidth == 512)
        #expect(fake.lastHeight == 384)
        #expect(fake.lastImages == 3)
        #expect(fake.lastAddSpecial == true)
    }

    /// As in the crate, the peak is read whatever the one decode produced.
    @Test func anUncountedDecodeStillSucceeds() async throws {
        let fake = FakeVision()
        fake.generateReturns = 0
        try await run(fake.make())
        #expect(fake.generateCount == 1)
    }

    /// Without an image chunk the cell would have loaded the projector and prefilled text,
    /// so it fails before the prefill rather than report a text peak as a vision one.
    @Test func aPromptWithNoImageChunkFailsBeforeThePrefill() async {
        let fake = FakeVision()
        fake.imageTokens = 0
        await #expect(throws: RuntimeError.engine(
            "vl_max_memory prompt tokenized to 1300 tokens with no image chunk among them; "
                + "the media marker did not expand")) {
            try await run(fake.make())
        }
        #expect(fake.evalCount == 0)
        #expect(fake.freedChunks == 1)
    }

    @Test func aFailedPrefillReleasesItsChunks() async {
        let fake = FakeVision()
        fake.failEvalOnCall = 0
        await #expect(throws: FakeVision.Boom.engine) { try await run(fake.make()) }
        #expect(fake.freedChunks == 1)
    }

    /// The frame count is bounded as `vl_throughput` bounds it, before anything is built.
    @Test func anAbsurdFrameCountIsRefusedBeforeAnythingRuns() async {
        let fake = FakeVision()
        let over = LlamaMtmd.maxNumImages + 1
        await #expect(throws: RuntimeError.engine(
            "parameter_num_images \(over) exceeds the supported maximum "
                + "\(LlamaMtmd.maxNumImages)")) {
            try await run(fake.make(), images: NonNegativeCount(validated: Int32(over)))
        }
        #expect(fake.tokenizeChunksCount == 0)
    }
}
