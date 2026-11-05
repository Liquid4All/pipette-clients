package ai.liquid.pipette.llama

import java.nio.charset.StandardCharsets

/**
 * Accumulates UTF-8 byte sequences across token pieces during detokenization.
 *
 * Tokenizers often emit sub-character byte fragments (e.g. byte-pair encodings splitting a 3-byte or 4-byte UTF-8 code point across consecutive
 * tokens). Decoding incomplete fragments immediately produces Unicode replacement characters (U+FFFD). RuneAccumulator buffers incomplete suffix
 * sequences, emitting only complete UTF-8 characters and retaining uncompleted byte sequences until subsequent tokens provide the remaining bytes.
 *
 * Calling [flush] drains any remaining bytes upon stream completion and is idempotent.
 */
class RuneAccumulator {
  private var pending = ByteArray(0)

  /**
   * Accumulates the provided UTF-8 [bytes] and returns any complete characters formed. Incomplete trailing code-point bytes remain buffered until the
   * next call or [flush].
   */
  fun accumulate(bytes: ByteArray): String {
    if (bytes.isEmpty()) return ""

    val combined =
      if (pending.isEmpty()) {
        bytes
      } else {
        val merged = ByteArray(pending.size + bytes.size)
        System.arraycopy(pending, 0, merged, 0, pending.size)
        System.arraycopy(bytes, 0, merged, pending.size, bytes.size)
        merged
      }

    val completeLen = completeUtf8Length(combined)
    if (completeLen == 0) {
      pending = combined
      return ""
    }

    val text = String(combined, 0, completeLen, StandardCharsets.UTF_8)
    val remaining = combined.size - completeLen
    if (remaining > 0) {
      val leftover = ByteArray(remaining)
      System.arraycopy(combined, completeLen, leftover, 0, remaining)
      pending = leftover
    } else {
      pending = ByteArray(0)
    }

    return text
  }

  /**
   * Terminal drain: decodes any buffered residue (using replacement characters if incomplete) and clears the buffer. Calling [flush] on an empty
   * accumulator returns an empty string.
   */
  fun flush(): String {
    if (pending.isEmpty()) return ""
    val text = String(pending, StandardCharsets.UTF_8)
    pending = ByteArray(0)
    return text
  }

  companion object {
    /**
     * Determines the length of the longest prefix of [bytes] consisting solely of complete UTF-8 sequences. Any incomplete sequence at the end is
     * excluded from the returned length.
     */
    internal fun completeUtf8Length(bytes: ByteArray): Int {
      val len = bytes.size
      if (len == 0) return 0

      // Look at the trailing 1 to 4 bytes to find if an incomplete multi-byte sequence starts there.
      var foundLeader = false
      var leaderIndex = -1
      val maxLookback = minOf(4, len)

      for (k in 0 until maxLookback) {
        val b = bytes[len - 1 - k].toInt() and 0xFF
        if ((b and 0x80) == 0) {
          // Single-byte ASCII.
          // If we haven't seen a multi-byte sequence before this, all bytes up to len are complete.
          break
        }
        if ((b and 0xC0) == 0xC0) {
          // Found a multi-byte leader byte (11xxxxxx)
          foundLeader = true
          leaderIndex = len - 1 - k
          break
        }
        // Continuation byte (10xxxxxx) -> keep looking back for the leader
      }

      if (!foundLeader) {
        // No incomplete multi-byte leader was found in the tail window.
        // All trailing bytes are either ASCII or standalone/continuation bytes.
        return len
      }

      val leader = bytes[leaderIndex].toInt() and 0xFF
      val expectedLength =
        when {
          (leader and 0xE0) == 0xC0 -> 2
          (leader and 0xF0) == 0xE0 -> 3
          (leader and 0xF8) == 0xF0 -> 4
          else -> 1 // Invalid leader byte; treated as complete so decoder handles it
        }

      val actualLength = len - leaderIndex
      return if (actualLength >= expectedLength) {
        len
      } else {
        leaderIndex
      }
    }
  }
}
