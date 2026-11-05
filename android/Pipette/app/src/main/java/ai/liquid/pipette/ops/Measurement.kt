package ai.liquid.pipette.ops

import java.util.Locale
import kotlin.math.pow
import kotlin.math.sqrt

const val REPS = 5

/** Called around each measured rep, in field order. */
class MeasurementObserver(val onBeforeRepStart: () -> Unit, val onAfterRepEnd: () -> Unit)

fun positiveFinite(metric: String, value: Double): Double {
  if (!value.isFinite() || value <= 0.0) {
    error("invalid $metric: $value")
  }
  return value
}

fun expectTokens(metric: String, actual: Int, expected: Int) {
  if (actual != expected) {
    error("engine returned $metric $actual, expected $expected")
  }
}

data class Rep<T>(val value: T, val elapsedNanos: Long) {
  fun elapsedMs(): Double = elapsedNanos.toDouble() / 1_000_000.0
}

data class Stats(val meanMs: Double, val stddevMs: Double)

class Measurement<T>(val label: String, val reps: List<Rep<T>>, val stats: Stats) {
  fun stats(): Stats = stats

  fun metric(name: String, sampleFn: (Rep<T>) -> Double): Stats {
    val samples = reps.map(sampleFn)
    val s = meanStddev(samples)
    val line = reductionLine("$label $name", samples, s)
    logMeasurement(line)
    return s
  }

  fun first(): T? = reps.firstOrNull()?.value
}

fun meanStddev(samples: List<Double>): Stats {
  val n = samples.size.toDouble()
  if (n == 0.0) return Stats(0.0, 0.0)
  val mean = samples.sum() / n
  if (n <= 1.0) return Stats(mean, 0.0)
  val variance = samples.sumOf { (it - mean).pow(2) } / (n - 1.0)
  return Stats(mean, sqrt(variance))
}

fun reductionLine(label: String, samples: List<Double>, stats: Stats): String {
  val values = samples.joinToString(", ") { String.format(Locale.US, "%.3f", it) }
  return String.format(Locale.US, "%s: samples [%s] ms -> mean %.3f ms stddev %.3f ms", label, values, stats.meanMs, stats.stddevMs)
}

internal var logMeasurement: (String) -> Unit = { line ->
  try {
    android.util.Log.i("pipette-native", line)
  } catch (_: RuntimeException) {
    println("[pipette-native] $line")
  }
}

fun <T> runMeasurement(
  label: String,
  gate: () -> Unit,
  observer: MeasurementObserver,
  prepare: (Int) -> Unit,
  work: (Int) -> T,
  sample: (Int, Rep<T>) -> Double,
): Measurement<T> {
  val reps = mutableListOf<Rep<T>>()
  val samples = mutableListOf<Double>()

  for (idx in 0 until REPS) {
    gate()
    observer.onBeforeRepStart()
    logMeasurement("$label: measurement run ${idx + 1}/$REPS")
    prepare(idx)

    val startedAt = System.nanoTime()
    val value = work(idx)
    val elapsed = System.nanoTime() - startedAt
    observer.onAfterRepEnd()

    val rep = Rep(value, elapsed)
    val s = sample(idx, rep)
    logMeasurement(String.format(Locale.US, "%s: measurement run %d/%d sampled %.3f ms", label, idx + 1, REPS, s))
    reps.add(rep)
    samples.add(s)
  }

  val stats = meanStddev(samples)
  logMeasurement(reductionLine(label, samples, stats))
  return Measurement(label, reps, stats)
}
