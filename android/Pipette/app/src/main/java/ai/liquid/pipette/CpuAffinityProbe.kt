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
  /**
   * How many CPUs the `scaling_max_freq` sweep actually read, against [totalCpus] possible ones.
   *
   * The coverage behind [topTierCpus] and [excludesTopTier], and the reason it is worth carrying rather than inferring. An offline core has no
   * `cpufreq` directory, so it drops out of the sweep silently, and the tier is then computed as the maximum of whatever survived. If the cores that
   * dropped out were the fast ones, [topTierCpus] names a slower tier that the process is perfectly free to run on, and [excludesTopTier] reads false
   * for want of data rather than because anything is well.
   *
   * A consumer that treats `excludesTopTier == false` as evidence of health has to check this first. `CpusetTracker`'s never-attained arm does,
   * because that comparison is its only corroboration; `LOST_CPUS` does not need to, because it still has the core-count clause to fall back on.
   */
  val sweptCpus: Int,
) {
  /** Whether the frequency sweep saw every CPU, so [topTierCpus] names the real prime tier rather than the best of what happened to be online. */
  val sweepComplete: Boolean
    get() = totalCpus > 0 && sweptCpus >= totalCpus

  /** One-line logcat summary. Coverage is printed only when it is partial, since the common case is complete and would be noise on every line. */
  fun summary(): String =
    "cpuset=${cpusetPath ?: "?"} allowed=${allowedCpus ?: "?"} ($allowedCount/$totalCpus) " +
      "topTier=${topTierCpus ?: "?"} excludesTopTier=$excludesTopTier availProc=$availableProcessors" +
      if (sweepComplete) "" else " sweep=$sweptCpus/$totalCpus"

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
        put("swept_cpus", sweptCpus)
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
            // Defaults to the CPU count, i.e. a complete sweep. Both ends of this hop ship in the same APK, so a payload without the key can only
            // come from a build that did not track coverage, and reading that as partial would disarm a guard rather than arm one.
            sweptCpus = o.optInt("swept_cpus", o.optInt("total_cpus")),
          )
        }
        .getOrNull()

    private fun JSONObject.optStringOrNull(key: String): String? = if (isNull(key)) null else optString(key).ifBlank { null }
  }
}

/**
 * One `scaling_max_freq` sweep: what it read, and how many CPUs it was trying to read.
 *
 * The two are carried together rather than the map alone because the map cannot report its own gaps. An offline core has no `cpufreq` directory and
 * simply does not appear, so a caller holding only [byCpu] cannot tell a four-core device from an eight-core one whose big cluster was parked when
 * the sweep ran. [possibleCpus] comes from the `cpuN` directory listing, which is present for every CPU the kernel knows about, online or not.
 */
data class CpuFreqSweep(val byCpu: Map<Int, Long>, val possibleCpus: Int)

/**
 * Reads the calling process's cpuset / CPU-affinity state. Pure parsing is split out ([buildSnapshot], [parseCpuList]) so it is unit-testable off a
 * device; [snapshot] wires the `/proc` and `/sys` reads to it, and [sample] the `/proc` ones only. Never throws; unreadable entries degrade to null.
 */
object CpuAffinityProbe {
  fun snapshot(): CpuAffinitySnapshot = sample(freqSweep())

  /**
   * The device's per-CPU `scaling_max_freq` sweep: a directory listing plus one open+read per core, with the count of cores it owed a reading for.
   *
   * Split out because it is effectively static (which cores are the prime tier does not change with cpuset membership), so a caller that samples
   * repeatedly reads it once and feeds it back to [sample] rather than paying for it every poll. `PipetteBenchmarkService` does exactly that, holding
   * one sweep for the life of the process and handing it to every `CpusetWatcher` sample. Measured in-process on a Pixel 10 Pro Fold (8 cores,
   * `CpusetWatcherCostTest`): this costs ~1119 µs, against ~1515 µs for a full [snapshot] and ~430 µs for a [sample]. Those are three separate
   * 500-iteration means rather than a decomposition of one, so they do not subtract exactly; the ratio that matters is ~3.5x per poll.
   *
   * Note the cost is dominated by the JVM work, not the syscalls: the same eight reads issued straight from a shell take ~152 µs. What is expensive
   * here is the `listFiles` over `/sys/devices/system/cpu`, which enumerates far more than the `cpuN` entries, plus a `Regex` match per entry. If
   * this ever needs to be cheaper, that is the part to attack, not the reads.
   *
   * Reusing the sweep is also the more correct choice than re-reading it. Thermal throttling can lower `scaling_max_freq` mid-run, and the question a
   * demotion check asks is "did this process lose the cores it started on", not "did the definition of a fast core move underneath it".
   */
  fun freqSweep(): CpuFreqSweep = readFreqSweep()

  /** A snapshot reusing an already-read [freqSweep]. See [freqSweep] for why that is worth threading through. */
  fun sample(sweep: CpuFreqSweep): CpuAffinitySnapshot =
    buildSnapshot(
      cpusetPath = readFile("/proc/self/cpuset")?.trim()?.ifBlank { null },
      allowedList = readProcStatusField("Cpus_allowed_list"),
      availableProcessors = Runtime.getRuntime().availableProcessors(),
      sweep = sweep,
    )

  /** Pure snapshot assembly from already-read raw values. */
  fun buildSnapshot(cpusetPath: String?, allowedList: String?, availableProcessors: Int, sweep: CpuFreqSweep): CpuAffinitySnapshot {
    val allowed = allowedList?.let { parseCpuList(it) }.orEmpty().toSortedSet()
    val maxFreqByCpu = sweep.byCpu
    // The larger of the directory listing and the highest index read. The listing is the authority, since it covers offline cores that the sweep
    // cannot see; the highest index is the floor, because a core we have a frequency for demonstrably exists whatever the listing said. Using the
    // highest index alone (as this did) undercounts whenever the *top* core is the one that dropped out, which is the case that matters most: it
    // makes an incomplete sweep look complete.
    val totalCpus = maxOf(sweep.possibleCpus, maxFreqByCpu.keys.maxOrNull()?.plus(1) ?: 0)
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
      sweptCpus = maxFreqByCpu.size,
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

  private fun readFreqSweep(): CpuFreqSweep {
    val cpuDir = File("/sys/devices/system/cpu")
    val cpus = cpuDir.listFiles { f -> f.name.matches(Regex("cpu\\d+")) }.orEmpty()
    val byCpu =
      cpus
        .mapNotNull { dir ->
          val index = dir.name.removePrefix("cpu").toIntOrNull() ?: return@mapNotNull null
          val freq = readFile("${dir.path}/cpufreq/scaling_max_freq")?.trim()?.toLongOrNull() ?: return@mapNotNull null
          index to freq
        }
        .toMap()
    // Counted from the directory listing rather than from the map, so a core that exists but could not be read is still counted as one we owed a
    // reading for. That difference is the whole signal.
    return CpuFreqSweep(byCpu = byCpu, possibleCpus = cpus.count { it.name.removePrefix("cpu").toIntOrNull() != null })
  }

  private fun readProcStatusField(field: String): String? =
    readFile("/proc/self/status")?.lineSequence()?.firstOrNull { it.startsWith("$field:") }?.substringAfter(':')?.trim()?.ifBlank { null }

  private fun readFile(path: String): String? = runCatching { File(path).readText() }.getOrNull()
}
