package ai.liquid.pipette.llama

import org.junit.Assert.assertEquals
import org.junit.Test

class LlamaCppTest {

  @Test
  fun memoryPeakDeltaSubtractsBaseline() {
    assertEquals(350L, memoryPeakDelta(100L, 450L, 300L))
  }

  @Test
  fun memoryPeakDeltaUsesFinalSample() {
    assertEquals(350L, memoryPeakDelta(100L, 300L, 450L))
  }

  @Test
  fun memoryPeakDeltaSaturatesWhenCounterDropsBelowBaseline() {
    assertEquals(0L, memoryPeakDelta(450L, 300L, 200L))
  }

  @Test
  fun aStopTokenClassifiesAsEos() {
    assertEquals(BenchmarkEvalCompletionStopReason.EOS, classifyStop(RawStop.StopToken))
  }

  @Test
  fun exhaustingTheBudgetClassifiesAsTruncated() {
    assertEquals(BenchmarkEvalCompletionStopReason.TRUNCATED, classifyStop(RawStop.CapReached))
  }
}
