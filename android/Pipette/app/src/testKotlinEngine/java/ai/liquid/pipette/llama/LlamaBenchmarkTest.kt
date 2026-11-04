package ai.liquid.pipette.llama

import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.ReadinessOutcome
import ai.liquid.pipette.thermal.AndroidThermalStatus
import java.util.concurrent.CancellationException
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class LlamaBenchmarkTest {

  @Test
  fun parseBenchmarkDefExtractsAllFields() {
    val json =
      """
      {
        "benchmark_id": "prefill_throughput_512",
        "benchmark_type": "prefill_throughput",
        "parameter_prefill_tokens": 512,
        "parameter_decode_tokens": 128,
        "parameter_max_tokens": 256,
        "parameter_mcq_choices": ["A", "B", "C"],
        "parameter_image_width": 224,
        "parameter_image_height": 224,
        "parameter_text_tokens": 64,
        "parameter_num_images": 1
      }
      """
        .trimIndent()

    val def = LlamaBenchmark.parseBenchmarkDef(json)
    assertEquals("prefill_throughput_512", def.benchmarkId)
    assertEquals("prefill_throughput", def.benchmarkType)
    assertEquals(512, def.parameterPrefillTokens)
    assertEquals(128, def.parameterDecodeTokens)
    assertEquals(256, def.parameterMaxTokens)
    assertEquals(listOf("A", "B", "C"), def.parameterMcqChoices)
    assertEquals(224, def.parameterImageWidth)
    assertEquals(224, def.parameterImageHeight)
    assertEquals(64, def.parameterTextTokens)
    assertEquals(1, def.parameterNumImages)
  }

  @Test
  fun requiredCtxSizeComputesBoundsPerType() {
    val prefillDef = LlamaBenchmark.BenchmarkDef(benchmarkId = "prefill_512", benchmarkType = "prefill_throughput", parameterPrefillTokens = 512)
    assertEquals(512, LlamaBenchmark.requiredCtxSize(prefillDef))

    val decodeDef =
      LlamaBenchmark.BenchmarkDef(
        benchmarkId = "decode_512_128",
        benchmarkType = "decode_throughput",
        parameterPrefillTokens = 512,
        parameterDecodeTokens = 128,
      )
    assertEquals(640, LlamaBenchmark.requiredCtxSize(decodeDef))

    val e2eDef =
      LlamaBenchmark.BenchmarkDef(
        benchmarkId = "e2e_512_128",
        benchmarkType = "end_to_end_latency",
        parameterPrefillTokens = 512,
        parameterDecodeTokens = 128,
      )
    assertEquals(640, LlamaBenchmark.requiredCtxSize(e2eDef))

    val maxMemDef = LlamaBenchmark.BenchmarkDef(benchmarkId = "max_mem_512", benchmarkType = "max_memory_usage", parameterPrefillTokens = 512)
    assertEquals(513, LlamaBenchmark.requiredCtxSize(maxMemDef))

    val vlDef =
      LlamaBenchmark.BenchmarkDef(
        benchmarkId = "vl_throughput",
        benchmarkType = "vl_throughput",
        parameterImageWidth = 224,
        parameterImageHeight = 224,
        parameterTextTokens = 50,
        parameterDecodeTokens = 32,
      )
    // 224 / 14 = 16 patches per dimension. 16 * 16 = 256 image tokens + 50 text + 32 decode = 338.
    assertEquals(338, LlamaBenchmark.requiredCtxSize(vlDef))

    val evalDef = LlamaBenchmark.BenchmarkDef(benchmarkId = "eval", benchmarkType = "eval", parameterMaxTokens = 128)
    assertEquals(LlamaBenchmark.EVAL_PROMPT_BUDGET + 128, LlamaBenchmark.requiredCtxSize(evalDef))
  }

  @Test
  fun checkCtxSizeThrowsWhenContextTooSmall() {
    val def =
      LlamaBenchmark.BenchmarkDef(
        benchmarkId = "decode_512_128",
        benchmarkType = "decode_throughput",
        parameterPrefillTokens = 512,
        parameterDecodeTokens = 128,
      )

    // Passes when contextSize >= 640.
    LlamaBenchmark.checkCtxSize(def, 1024)
    LlamaBenchmark.checkCtxSize(def, 640)

    // Fails when contextSize < 640.
    val ex = assertThrows(IllegalStateException::class.java) { LlamaBenchmark.checkCtxSize(def, 512) }
    assertTrue(ex.message!!.contains("context_size 512 is too small for benchmark 'decode_512_128'"))
    assertTrue(ex.message!!.contains("needs at least 640"))
  }

  @Test
  fun readinessGateHandlesNullAndReadyCleanly() {
    // Null callback is no-op.
    LlamaBenchmark.readinessGate(null)

    // Ready (null return) is no-op.
    val readyCooldown = BenchmarkCooldownCallback { null }
    LlamaBenchmark.readinessGate(readyCooldown)
  }

  @Test
  fun readinessGateThrowsCancellationOnCancelPrefix() {
    val cancelCooldown = BenchmarkCooldownCallback { "${ReadinessOutcome.CANCELLED_PREFIX}user tapped cancel" }
    assertThrows(CancellationException::class.java) { LlamaBenchmark.readinessGate(cancelCooldown) }
  }

  @Test
  fun readinessGateThrowsOnTimeoutPrefix() {
    val timeoutCooldown = BenchmarkCooldownCallback { "${ReadinessOutcome.TIMED_OUT_PREFIX}headroom 0.98 status moderate" }
    val ex = assertThrows(IllegalStateException::class.java) { LlamaBenchmark.readinessGate(timeoutCooldown) }
    assertTrue(ex.message!!.contains("device failed to cool within the time budget: last seen headroom 0.98 status moderate"))
  }

  @Test
  fun sampleThermalParsesHeadroomAndStatus() {
    val validCallback =
      object : BenchmarkThermalCallback {
        override fun sampleHeadroom(): Float = 0.45f

        override fun sampleStatus(): Int = 0 // NONE
      }

    val reading = LlamaBenchmark.sampleThermal(validCallback)
    assertNotNull(reading)
    assertEquals(0.45f, reading!!.androidThermalHeadroom)
    assertEquals(AndroidThermalStatus.NONE, reading.androidThermalStatus)

    val nanCallback =
      object : BenchmarkThermalCallback {
        override fun sampleHeadroom(): Float = Float.NaN

        override fun sampleStatus(): Int = 99 // out of bounds
      }

    val nanReading = LlamaBenchmark.sampleThermal(nanCallback)
    assertNotNull(nanReading)
    assertNull(nanReading!!.androidThermalHeadroom)
    assertNull(nanReading.androidThermalStatus)
  }

  @Test
  fun mergeThermalPopulatesJsonWhenComplete() {
    val result = JSONObject()
    val before =
      listOf(LlamaBenchmark.ThermalReading(0.3f, AndroidThermalStatus.NONE), LlamaBenchmark.ThermalReading(0.35f, AndroidThermalStatus.NONE))
    val after =
      listOf(LlamaBenchmark.ThermalReading(0.32f, AndroidThermalStatus.NONE), LlamaBenchmark.ThermalReading(0.38f, AndroidThermalStatus.NONE))

    LlamaBenchmark.mergeThermal(result, before, after)

    assertTrue(result.has("device_android_thermal_headroom_before"))
    assertTrue(result.has("device_android_thermal_headroom_after"))
    assertTrue(result.has("device_android_thermal_status_before"))
    assertTrue(result.has("device_android_thermal_status_after"))

    val hrBefore = result.getJSONArray("device_android_thermal_headroom_before")
    assertEquals(2, hrBefore.length())
    assertEquals(0.3, hrBefore.getDouble(0), 1e-4)

    val stAfter = result.getJSONArray("device_android_thermal_status_after")
    assertEquals("none", stAfter.getString(0))
  }

  @Test
  fun mergeThermalOmitsIncompleteTelemetry() {
    val result = JSONObject()
    val beforeWithNullHeadroom =
      listOf(LlamaBenchmark.ThermalReading(0.3f, AndroidThermalStatus.NONE), LlamaBenchmark.ThermalReading(null, AndroidThermalStatus.NONE))
    val after =
      listOf(LlamaBenchmark.ThermalReading(0.32f, AndroidThermalStatus.NONE), LlamaBenchmark.ThermalReading(0.38f, AndroidThermalStatus.NONE))

    LlamaBenchmark.mergeThermal(result, beforeWithNullHeadroom, after)

    // before headroom had a null sample, so all-or-nothing scalar series omits it.
    assertTrue(!result.has("device_android_thermal_headroom_before"))
    // after headroom was complete.
    assertTrue(result.has("device_android_thermal_headroom_after"))
    // status was complete on both.
    assertTrue(result.has("device_android_thermal_status_before"))
    assertTrue(result.has("device_android_thermal_status_after"))
  }

  @Test
  fun reportMeasurementThrowsOnCancellation() {
    val cancellingProgress = BenchmarkProgressCallback { _, _, _ -> false }
    assertThrows(CancellationException::class.java) { LlamaBenchmark.reportMeasurement(cancellingProgress, 0, 5) }

    val continuingProgress = BenchmarkProgressCallback { _, _, _ -> true }
    LlamaBenchmark.reportMeasurement(continuingProgress, 0, 5)
  }
}
