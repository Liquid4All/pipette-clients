package ai.liquid.pipette

import java.io.ByteArrayOutputStream
import java.io.InputStream

internal object HttpResponseBody {
  class TooLarge(limit: Int) : IllegalStateException("Response exceeded $limit bytes")

  fun read(input: InputStream, limit: Int): ByteArray {
    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
    val output = ByteArrayOutputStream()
    while (true) {
      val remaining = limit - output.size()
      val count = input.read(buffer, 0, minOf(buffer.size, remaining + 1))
      if (count == -1) return output.toByteArray()
      if (count > remaining) throw TooLarge(limit)
      output.write(buffer, 0, count)
    }
  }
}
