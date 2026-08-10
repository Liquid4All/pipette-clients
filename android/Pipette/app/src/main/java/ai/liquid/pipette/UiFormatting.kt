package ai.liquid.pipette

import kotlin.math.roundToInt

private const val PERCENT = 100
private const val MILLIS_PER_SECOND = 1_000
private const val SECONDS_PER_MINUTE = 60

/** Naive English pluralization used in status/summary strings. */
fun plural(word: String, count: Int): String = if (count == 1) word else "${word}s"

/**
 * Thermal-severity accent kind, kept free of any Android/Compose Color so both the Compose screens (via `accentColor`) and the classic-Views
 * `BenchmarkActivity` (which maps it to an `R.color`) share one classifier. Compose-free on purpose so `:benchmark` can use it without loading
 * Compose.
 */
enum class AccentKind {
  NOMINAL,
  SERIOUS,
  CRITICAL,
  MUTED,
}

/** Map a human thermal description to a severity accent (shared by Settings, pocket mode, and BenchmarkActivity). */
fun thermalAccentKind(description: String): AccentKind {
  val desc = description.lowercase()
  return when {
    listOf("critical", "severe", "shutdown", "emergency").any { desc.contains(it) } -> AccentKind.CRITICAL
    listOf("serious", "hot", "throttl", "moderate").any { desc.contains(it) } -> AccentKind.SERIOUS
    else -> AccentKind.NOMINAL
  }
}

/** Short thermal label (e.g. for the pocket-mode line). Shared by screens + the Activity. */
fun thermalLabel(provider: AndroidThermalStatusProvider): String = AndroidThermalStatusProvider.statusLabel(provider.currentStatus())

/**
 * Thermal readout combining the OS thermal-state word with the granular thermal-headroom percentage (0% cool → 100% at the throttle threshold; higher
 * = hotter), e.g. `"Moderate · 72%"`. The percentage can read **above 100%** when the SoC is throttling past the threshold — that's kept as-is since
 * it's meaningful for a benchmark tool. Android has no in-app °C sensor, so headroom is the closest device-temperature signal. Falls back to the
 * state word alone when headroom is unavailable (`NaN`), so it never shows a blank or a bare number.
 */
fun thermalHeadroomLabel(provider: AndroidThermalStatusProvider): String {
  val word = AndroidThermalStatusProvider.displayStatusLabel(provider.currentStatus())
  val headroom = provider.cachedHeadroom()
  if (headroom.isNaN()) return word
  // Float × Int is defined in Kotlin (→ Float). coerceAtLeast(0) guards the nonsensical negative side;
  // the upper side is intentionally uncapped (see kdoc) so a throttling device reads e.g. "115%".
  val pct = (headroom * PERCENT).roundToInt().coerceAtLeast(0)
  return "$word · $pct%"
}

/** Verbose thermal label (e.g. for Settings). */
fun thermalDescription(provider: AndroidThermalStatusProvider): String = AndroidThermalStatusProvider.displayStatusLabel(provider.currentStatus())

/**
 * Caption for the live cooldown timer at a render instant, e.g. `"Cooling 0:20 / 3:00 max"`. Shared by the Compose pocket (JobLiveActivity) and the
 * classic-Views BenchmarkActivity — pure Kotlin (no Compose), so both processes can call it. [sinceMillis] is when cooling began; [nowMillis] is the
 * render clock.
 */
fun coolingCaption(sinceMillis: Long, nowMillis: Long): String {
  val max = (Readiness.COOLDOWN_MAX_MILLIS / MILLIS_PER_SECOND).toInt()
  // Clamp: the gate polls on an interval, so it can sit a poll past its own deadline before
  // returning, and the once-a-second ticker would otherwise render an elapsed above `max`.
  val elapsed = ((nowMillis - sinceMillis).coerceAtLeast(0L) / MILLIS_PER_SECOND).toInt().coerceAtMost(max)
  return "Cooling ${clock(elapsed)} / ${clock(max)} max"
}

private fun clock(seconds: Int): String = "%d:%02d".format(seconds / SECONDS_PER_MINUTE, seconds % SECONDS_PER_MINUTE)
