package ai.liquid.pipette.llama

import java.nio.charset.StandardCharsets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RuneAccumulatorTest {

  @Test
  fun accumulateAsciiEmitsImmediately() {
    val acc = RuneAccumulator()
    val chunk1 = "Hello, ".toByteArray(StandardCharsets.UTF_8)
    val chunk2 = "world!".toByteArray(StandardCharsets.UTF_8)

    assertEquals("Hello, ", acc.accumulate(chunk1))
    assertEquals("world!", acc.accumulate(chunk2))
    assertEquals("", acc.flush())
  }

  @Test
  fun accumulateTwoByteCharacterSplit() {
    val acc = RuneAccumulator()
    val fullBytes = "é".toByteArray(StandardCharsets.UTF_8) // 0xC3, 0xA9
    assertEquals(2, fullBytes.size)

    val chunk1 = byteArrayOf(fullBytes[0])
    val chunk2 = byteArrayOf(fullBytes[1])

    assertEquals("", acc.accumulate(chunk1))
    assertEquals("é", acc.accumulate(chunk2))
    assertEquals("", acc.flush())
  }

  @Test
  fun accumulateThreeByteCharacterSplit() {
    val acc = RuneAccumulator()
    val fullBytes = "€".toByteArray(StandardCharsets.UTF_8) // 0xE2, 0x82, 0xAC
    assertEquals(3, fullBytes.size)

    val chunk1 = byteArrayOf(fullBytes[0])
    val chunk2 = byteArrayOf(fullBytes[1])
    val chunk3 = byteArrayOf(fullBytes[2])

    assertEquals("", acc.accumulate(chunk1))
    assertEquals("", acc.accumulate(chunk2))
    assertEquals("€", acc.accumulate(chunk3))
    assertEquals("", acc.flush())
  }

  @Test
  fun accumulateFourByteEmojiSplit() {
    val acc = RuneAccumulator()
    val fullBytes = "🚀".toByteArray(StandardCharsets.UTF_8) // 4 bytes
    assertEquals(4, fullBytes.size)

    val chunk1 = byteArrayOf(fullBytes[0], fullBytes[1])
    val chunk2 = byteArrayOf(fullBytes[2], fullBytes[3])

    assertEquals("", acc.accumulate(chunk1))
    assertEquals("🚀", acc.accumulate(chunk2))
    assertEquals("", acc.flush())
  }

  @Test
  fun accumulateMixedStringWithSplitInMiddle() {
    val acc = RuneAccumulator()
    val euroBytes = "€".toByteArray(StandardCharsets.UTF_8)
    val prefix = "Price: ".toByteArray(StandardCharsets.UTF_8)
    val suffix = "100".toByteArray(StandardCharsets.UTF_8)

    val chunk1 = prefix + byteArrayOf(euroBytes[0])
    val chunk2 = byteArrayOf(euroBytes[1], euroBytes[2]) + suffix

    assertEquals("Price: ", acc.accumulate(chunk1))
    assertEquals("€100", acc.accumulate(chunk2))
    assertEquals("", acc.flush())
  }

  @Test
  fun flushDrainsIncompleteTrailingBytes() {
    val acc = RuneAccumulator()
    val incomplete = byteArrayOf(0xE2.toByte(), 0x82.toByte()) // 2 of 3 bytes for €

    assertEquals("", acc.accumulate(incomplete))
    val flushed = acc.flush()
    assertTrue(flushed.isNotEmpty())

    // Idempotent: subsequent flush returns empty string
    assertEquals("", acc.flush())
  }

  @Test
  fun completeUtf8LengthCalculations() {
    assertEquals(0, RuneAccumulator.completeUtf8Length(byteArrayOf()))
    assertEquals(5, RuneAccumulator.completeUtf8Length("hello".toByteArray(StandardCharsets.UTF_8)))

    val euro = "€".toByteArray(StandardCharsets.UTF_8)
    assertEquals(3, RuneAccumulator.completeUtf8Length(euro))
    assertEquals(0, RuneAccumulator.completeUtf8Length(byteArrayOf(euro[0])))
    assertEquals(0, RuneAccumulator.completeUtf8Length(byteArrayOf(euro[0], euro[1])))

    val rocket = "🚀".toByteArray(StandardCharsets.UTF_8)
    assertEquals(4, RuneAccumulator.completeUtf8Length(rocket))
    assertEquals(0, RuneAccumulator.completeUtf8Length(byteArrayOf(rocket[0], rocket[1])))
    assertEquals(0, RuneAccumulator.completeUtf8Length(byteArrayOf(rocket[0], rocket[1], rocket[2])))
  }
}
