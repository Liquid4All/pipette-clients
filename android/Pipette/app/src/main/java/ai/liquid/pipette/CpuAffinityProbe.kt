package ai.liquid.pipette

import java.io.File
import org.json.JSONObject

/**
 * A point-in-time view of the calling process's CPU scheduling constraints, used to diagnose OEM cpuset demotion. On Samsung (One UI) a process that
 * is not the visible `top-app` — even a foreground service, as our `:benchmark` process is — is placed in a restrictive cpuset cgroup (`/moderate`,
 * `/background`) whose `cpuset.cpus` excludes the prime cores, throttling inference; on Pixel/AOSP a foreground-service process lands in
 * `/foreground` with the big cores present. Capturing this per run (in both the `:benchmark` and main processes) confirms and quantifies the
 * demotion.
 *
 * All fields are best-effort: a value is null when the underlying `/proc` or `/sys` entry can't be read.
 */
data class CpuAffinitySnapshot(
  /** `/proc/self/cpuset` — the cpuset cgroup path, e.g. `/top-app`, `/foreground`, `/moderate`, `/background`. The primary demotion signal. */
  val cpusetPath: String?,
  /** `Cpus_allowed_list` from `/proc/self/status` — the CPUs the process may run on after the cgroup intersect, e.g. `0-3` (little cores only). */
  val allowedCpus: String?,
  /** Number of CPUs in [allowedCpus]. */
  val allowedCount: Int,
  /** Total online CPUs on the device. */
  val totalCpus: Int,
  /** `Runtime.availableProcessors()` — cross-check (reflects the affinity-limited count on Android). */
  val availableProcessors: Int,
  /** CPUs in the highest `scaling_max_freq` tier (the prime/big cores), e.g. `6-7`; null when per-core frequencies are unreadable. */
  val topTierCpus: String?,
  /** True when at least one top-tier (highest-frequency) CPU is NOT in [allowedCpus] — i.e. inference is barred from the fastest cores. */
  val excludesTopTier: Boolean,
) {
  /** One-line logcat summary. */
  fun summary(): String =
    "cpuset=${cpusetPath ?: "?"} allowed=${allowedCpus ?: "?"} ($allowedCount/$totalCpus) " +
      "topTier=${topTierCpus ?: "?"} excludesTopTier=$excludesTopTier availProc=$availableProcessors"

  fun toJson(): String =
    JSONObject()
      .apply {
        put("cpuset_path", cpusetPath ?: JSONObject.NULL)
        put("allowed_cpus", allowedCpus ?: JSONObject.NULL)
        put("allowed_count", allowedCount)
        put("total_cpus", totalCpus)
        put("available_processors", availableProcessors)
        put("top_tier_cpus", topTierCpus ?: JSONObject.NULL)
        put("excludes_top_tier", excludesTopTier)
      }
      .toString()

  companion object {
    fun fromJson(json: String): CpuAffinitySnapshot? =
      runCatching {
          val o = JSONObject(json)
          CpuAffinitySnapshot(
            cpusetPath = o.optStringOrNull("cpuset_path"),
            allowedCpus = o.optStringOrNull("allowed_cpus"),
            allowedCount = o.optInt("allowed_count"),
            totalCpus = o.optInt("total_cpus"),
            availableProcessors = o.optInt("available_processors"),
            topTierCpus = o.optStringOrNull("top_tier_cpus"),
            excludesTopTier = o.optBoolean("excludes_top_tier"),
          )
        }
        .getOrNull()

    private fun JSONObject.optStringOrNull(key: String): String? = if (isNull(key)) null else optString(key).ifBlank { null }
  }
}

/**
 * Reads the calling process's cpuset / CPU-affinity state. Pure parsing is split out ([buildSnapshot], [parseCpuList]) so it is unit-testable off a
 * device; [snapshot] wires the `/proc` and `/sys` reads to it. Never throws — unreadable entries degrade to null.
 */
object CpuAffinityProbe {
  fun snapshot(): CpuAffinitySnapshot {
    val cpusetPath = readFile("/proc/self/cpuset")?.trim()?.ifBlank { null }
    val allowedList = readProcStatusField("Cpus_allowed_list")
    return buildSnapshot(
      cpusetPath = cpusetPath,
      allowedList = allowedList,
      availableProcessors = Runtime.getRuntime().availableProcessors(),
      maxFreqByCpu = readMaxFreqByCpu(),
    )
  }

  /** Pure snapshot assembly from already-read raw values. */
  fun buildSnapshot(cpusetPath: String?, allowedList: String?, availableProcessors: Int, maxFreqByCpu: Map<Int, Long>): CpuAffinitySnapshot {
    val allowed = allowedList?.let { parseCpuList(it) }.orEmpty().toSortedSet()
    // Highest CPU index + 1, not the entry count — so a mid-range core whose
    // scaling_max_freq is unreadable doesn't undercount the total (which would
    // make the allowed/total summary misleading).
    val totalCpus = maxFreqByCpu.keys.maxOrNull()?.plus(1) ?: 0
    val topTier =
      if (maxFreqByCpu.isEmpty()) {
        emptySet()
      } else {
        val maxFreq = maxFreqByCpu.values.max()
        maxFreqByCpu.filterValues { it == maxFreq }.keys.toSortedSet()
      }
    val excludesTopTier = topTier.isNotEmpty() && allowed.isNotEmpty() && topTier.any { it !in allowed }
    return CpuAffinitySnapshot(
      cpusetPath = cpusetPath,
      allowedCpus = allowedList?.trim()?.ifBlank { null },
      allowedCount = allowed.size,
      totalCpus = totalCpus,
      availableProcessors = availableProcessors,
      topTierCpus = if (topTier.isEmpty()) null else formatCpuList(topTier),
      excludesTopTier = excludesTopTier,
    )
  }

  /** Parse a Linux CPU-list ("0-3,5,7-8") into individual indices. Malformed segments are skipped. */
  fun parseCpuList(list: String): List<Int> =
    list.split(',').flatMap { segment ->
      val part = segment.trim()
      when {
        part.isEmpty() -> emptyList()
        part.contains('-') -> {
          val (lo, hi) = part.split('-', limit = 2).map { it.trim().toIntOrNull() ?: return@flatMap emptyList() }
          if (lo <= hi) (lo..hi).toList() else emptyList()
        }
        else -> part.toIntOrNull()?.let { listOf(it) } ?: emptyList()
      }
    }

  /** Inverse of [parseCpuList]: render a sorted set of indices back into compact range form ("6-7", "0,2,4"). */
  fun formatCpuList(cpus: Set<Int>): String {
    val sorted = cpus.toSortedSet().toList()
    if (sorted.isEmpty()) return ""
    val ranges = mutableListOf<String>()
    var start = sorted.first()
    var prev = start
    for (i in 1 until sorted.size) {
      val c = sorted[i]
      if (c == prev + 1) {
        prev = c
      } else {
        ranges += if (start == prev) "$start" else "$start-$prev"
        start = c
        prev = c
      }
    }
    ranges += if (start == prev) "$start" else "$start-$prev"
    return ranges.joinToString(",")
  }

  private fun readMaxFreqByCpu(): Map<Int, Long> {
    val cpuDir = File("/sys/devices/system/cpu")
    val cpus = cpuDir.listFiles { f -> f.name.matches(Regex("cpu\\d+")) }.orEmpty()
    return cpus
      .mapNotNull { dir ->
        val index = dir.name.removePrefix("cpu").toIntOrNull() ?: return@mapNotNull null
        val freq = readFile("${dir.path}/cpufreq/scaling_max_freq")?.trim()?.toLongOrNull() ?: return@mapNotNull null
        index to freq
      }
      .toMap()
  }

  private fun readProcStatusField(field: String): String? =
    readFile("/proc/self/status")?.lineSequence()?.firstOrNull { it.startsWith("$field:") }?.substringAfter(':')?.trim()?.ifBlank { null }

  private fun readFile(path: String): String? = runCatching { File(path).readText() }.getOrNull()
}
