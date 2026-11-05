import Testing

@testable import Pipette

/// `MlxRuntime.cacheFillRanges`: which prompt windows `prefillAsMlxLm` feeds the cache with
/// and which token it holds back for the logits, as `mlx_lm`'s prefill strides them. The
/// forwards need weights and a GPU.
struct MlxRuntimeGenerateTests {
    /// One token, smaller than a window, equal to one, one over, and many: `generate_step`
    /// strides the first `P - 1` tokens from zero, so a prompt of exactly one window leaves
    /// its last token to the sampled forward and one token over fills the whole window.
    @Test(arguments: [
        (1, 512, []),
        (2, 512, [0 ..< 1]),
        (100, 512, [0 ..< 99]),
        (512, 512, [0 ..< 511]),
        (513, 512, [0 ..< 512]),
        (1025, 512, [0 ..< 512, 512 ..< 1024]),
        (4096, 512, [0 ..< 512, 512 ..< 1024, 1024 ..< 1536, 1536 ..< 2048, 2048 ..< 2560,
                     2560 ..< 3072, 3072 ..< 3584, 3584 ..< 4095]),
        (4096, 0, [0 ..< 4095]),
    ] as [(Int, Int, [Range<Int>])])
    func thePromptFillsInWindowsUpToItsLastToken(count: Int, chunk: Int,
                                                 expected: [Range<Int>]) {
        #expect(MlxRuntime.cacheFillRanges(count: count, chunk: chunk) == expected)
    }

    /// Whatever the shape, the windows are contiguous from zero, none wider than the chunk,
    /// and end one short of the prompt, so the held-back token is always the last.
    @Test(arguments: [1, 2, 63, 64, 65, 127, 128, 129, 1000])
    func theWindowsCoverAllButTheLastToken(count: Int) {
        let ranges = MlxRuntime.cacheFillRanges(count: count, chunk: 64)
        #expect((ranges.first?.lowerBound ?? 0) == 0)
        #expect((ranges.last?.upperBound ?? 0) == count - 1)
        #expect(zip(ranges, ranges.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound })
        #expect(ranges.allSatisfy { !$0.isEmpty && $0.count <= 64 })
    }
}
