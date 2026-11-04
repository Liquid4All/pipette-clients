package ai.liquid.pipette.ops

/** Shared natural-language passage used by every benchmark client to build synthetic prompts of an exact token count. */
object PromptSeed {
  /** The seed passage, loaded from the classpath resource prompt_seed.txt. */
  val text: String by lazy {
    val stream =
      PromptSeed::class.java.classLoader?.getResourceAsStream("prompt_seed.txt") ?: error("prompt_seed.txt resource not found on classpath")
    stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
  }

  private const val TOKENIZE_BUDGET = 64
  private const val PREFIX_SCAN_WINDOW_CHARS = 32

  private val TAIL_EXTENSIONS = arrayOf(" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".")

  /** Build a string that tokenizes to exactly [targetTokens] under the caller's [count] callback. */
  fun buildPromptText(targetTokens: Int, count: (String) -> Int): String = buildPromptTextFrom(text, targetTokens, count)

  /** Internal build function accepting a custom seed string. */
  fun buildPromptTextFrom(seed: String, targetTokens: Int, count: (String) -> Int): String {
    if (targetTokens == 0) return ""
    if (seed.isEmpty()) error("seed text is empty")
    val target = targetTokens
    val budget = Budget(TOKENIZE_BUDGET)

    val seedCount = budget.count(count, seed)
    if (seedCount == 0) error("seed text tokenizes to zero tokens")

    val charsPerToken = seed.length.toDouble() / seedCount.toDouble()
    val initialChars = (target.toDouble() * charsPerToken * 1.20).toInt()
    val repeats = (initialChars / seed.length) + 2
    val pool = StringBuilder(seed.repeat(repeats))

    var fullCount = budget.count(count, pool.toString())
    while (fullCount < target) {
      pool.append(seed)
      fullCount = budget.count(count, pool.toString())
    }
    if (fullCount == target) {
      return pool.toString()
    }

    val poolStr = pool.toString()
    if (target <= PREFIX_SCAN_WINDOW_CHARS) {
      when (val scan = scanPrefixWindow(poolStr, 0, target, budget, count)) {
        is PrefixScan.Done -> return scan.text
        is PrefixScan.Under -> Unit
      }
    }

    var lo = 0
    var hi = poolStr.length
    while (hi - lo > 1) {
      val mid = clampCharBoundary(poolStr, lo + (hi - lo) / 2)
      if (mid == lo || mid == hi) break
      val countAtMid = budget.count(count, poolStr.substring(0, mid))
      when {
        countAtMid < target -> lo = mid
        countAtMid == target -> return poolStr.substring(0, mid)
        else -> hi = mid
      }
    }

    val under =
      when (val scan = scanPrefixWindow(poolStr, lo, target, budget, count)) {
        is PrefixScan.Done -> return scan.text
        is PrefixScan.Under -> scan
      }

    var curText = under.text
    var curCount = under.count

    while (curCount < target) {
      val step =
        tailStep(curText, curCount, target, budget, count)
          ?: error("tail-grow stuck at $curCount tokens, target $target: no extension advanced without overshooting")
      when (step) {
        is TailStep.Done -> return step.text
        is TailStep.Progress -> {
          curText = step.text
          curCount = step.count
        }
      }
    }

    error("builder ended at $curCount tokens, target was $target: bisection invariant broke")
  }

  private fun clampCharBoundary(s: String, pos: Int): Int {
    if (pos >= s.length) return s.length
    var p = pos
    while (p > 0 && Character.isLowSurrogate(s[p])) {
      p--
    }
    return p
  }

  private sealed interface PrefixScan {
    data class Done(val text: String) : PrefixScan

    data class Under(val text: String, val count: Int) : PrefixScan
  }

  private fun scanPrefixWindow(pool: String, start: Int, target: Int, budget: Budget, count: (String) -> Int): PrefixScan {
    var pos = start.coerceAtMost(pool.length)
    var bestText: String? = null
    var bestCount = -1
    var steps = 0
    while (true) {
      val text = pool.substring(0, pos)
      val tokenCount = budget.count(count, text)
      if (tokenCount == target) {
        return PrefixScan.Done(text)
      }
      if (tokenCount < target) {
        val shouldReplace = bestText == null || tokenCount > bestCount || (tokenCount == bestCount && text.length > bestText.length)
        if (shouldReplace) {
          bestText = text
          bestCount = tokenCount
        }
      }
      if (steps >= PREFIX_SCAN_WINDOW_CHARS || pos >= pool.length) {
        break
      }
      pos = nextCharBoundary(pool, pos)
      steps++
    }

    val best = bestText ?: error("no prefix below target near bisection boundary")
    return PrefixScan.Under(best, bestCount)
  }

  private fun nextCharBoundary(s: String, pos: Int): Int {
    var next = (pos + 1).coerceAtMost(s.length)
    while (next < s.length && Character.isLowSurrogate(s[next])) {
      next++
    }
    return next
  }

  private sealed interface TailStep {
    data class Done(val text: String) : TailStep

    data class Progress(val text: String, val count: Int) : TailStep
  }

  private fun tailStep(text: String, current: Int, target: Int, budget: Budget, count: (String) -> Int): TailStep? {
    for (ext in TAIL_EXTENSIONS) {
      val candidate = text + ext
      val candidateCount = budget.count(count, candidate)
      if (candidateCount == target) {
        return TailStep.Done(candidate)
      } else if (candidateCount in (current + 1) until target) {
        return TailStep.Progress(candidate, candidateCount)
      }
    }
    return null
  }

  /** Repeat [seed] until the output reaches exactly [targetLen] tokens. */
  fun repeatTokenSequence(seed: IntArray, targetLen: Int): IntArray {
    if (targetLen == 0) return IntArray(0)
    require(seed.isNotEmpty()) { "seed token list must not be empty" }
    val result = IntArray(targetLen)
    for (i in 0 until targetLen) {
      result[i] = seed[i % seed.size]
    }
    return result
  }

  fun repeatTokenSequence(seed: List<Int>, targetLen: Int): List<Int> {
    if (targetLen == 0) return emptyList()
    require(seed.isNotEmpty()) { "seed token list must not be empty" }
    val result = ArrayList<Int>(targetLen)
    for (i in 0 until targetLen) {
      result.add(seed[i % seed.size])
    }
    return result
  }

  private class Budget(private var remaining: Int) {
    fun count(fn: (String) -> Int, text: String): Int {
      if (remaining == 0) {
        error("prompt builder exceeded tokenize budget of $TOKENIZE_BUDGET calls")
      }
      remaining--
      return fn(text)
    }
  }
}
