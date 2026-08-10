package ai.liquid.pipette.service

import android.content.Context
import java.io.File

/**
 * Moves a benchmark result JSON across the AIDL boundary without tripping `TransactionTooLargeException`. Small payloads ride inline in the parcel;
 * anything over [BenchmarkResult.INLINE_LIMIT_BYTES] spills to a file in the shared app cache (both processes run as the same UID, so the path
 * resolves on either side). The proxy [resolve]s a result back to JSON and deletes the spill file, so the scratch dir self-cleans on the happy path.
 */
object ResultSpill {
  private fun dir(context: Context): File = File(context.cacheDir, "benchmark-results").apply { mkdirs() }

  /** Wrap [json] as a successful result, spilling to a file when it's large. */
  fun packageResult(context: Context, json: String): BenchmarkResult {
    val bytes = json.toByteArray(Charsets.UTF_8)
    if (bytes.size <= BenchmarkResult.INLINE_LIMIT_BYTES) {
      return BenchmarkResult(ok = true, inlineJson = json)
    }
    val file = File(dir(context), "result-${System.nanoTime()}.json")
    file.writeText(json)
    return BenchmarkResult(ok = true, referencePath = file.absolutePath)
  }

  /** Read a successful result back to its JSON, deleting any spill file. */
  fun resolve(result: BenchmarkResult): String {
    result.inlineJson?.let {
      return it
    }
    val path = result.referencePath ?: error("Result carried no payload")
    val file = File(path)
    return try {
      file.readText()
    } finally {
      file.delete()
    }
  }
}
