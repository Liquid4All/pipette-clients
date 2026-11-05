package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Test

class ByteFormatTest {
  @Test
  fun `file sizes use decimal units`() {
    assertEquals("149.1 MB", ByteFormat.fileSize(149_100_000L))
    assertEquals("1.28 GB", ByteFormat.fileSize(1_280_000_000L))
  }

  @Test
  fun `file sizes omit insignificant decimal places`() {
    assertEquals("538 MB", ByteFormat.fileSize(538_000_000L))
    assertEquals("1.9 GB", ByteFormat.fileSize(1_900_000_000L))
  }
}
