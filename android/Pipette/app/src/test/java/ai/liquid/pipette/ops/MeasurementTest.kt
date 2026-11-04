package ai.liquid.pipette.ops

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class MeasurementTest {

  @Test
  fun testStatsCalculation() {
    val empty = meanStddev(emptyList())
    assertEquals(0.0, empty.meanMs, 0.001)
    assertEquals(0.0, empty.stddevMs, 0.001)

    val single = meanStddev(listOf(42.0))
    assertEquals(42.0, single.meanMs, 0.001)
    assertEquals(0.0, single.stddevMs, 0.001)

    // Samples: [10.0, 20.0, 30.0] -> mean = 20.0, var = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = 200 / 2 = 100 -> stddev = 10.0
    val stats = meanStddev(listOf(10.0, 20.0, 30.0))
    assertEquals(20.0, stats.meanMs, 0.001)
    assertEquals(10.0, stats.stddevMs, 0.001)
  }

  @Test
  fun testRunMeasurementExecutionOrder() {
    val events = mutableListOf<String>()
    val gate = {
      events.add("gate")
      Unit
    }
    val observer = MeasurementObserver(onBeforeRepStart = { events.add("observer_start") }, onAfterRepEnd = { events.add("observer_finish") })

    val m =
      runMeasurement(
        label = "test_cell",
        gate = gate,
        observer = observer,
        prepare = { idx -> events.add("prepare_$idx") },
        work = { idx ->
          events.add("work_$idx")
          idx * 10
        },
        sample = { idx, rep ->
          events.add("sample_$idx")
          rep.value.toDouble()
        },
      )

    assertEquals(REPS, m.reps.size)
    assertEquals(0, m.first())
    // 5 iterations * 6 events each = 30 events in fixed sequence
    assertEquals(30, events.size)
    for (i in 0 until REPS) {
      val base = i * 6
      assertEquals("gate", events[base])
      assertEquals("observer_start", events[base + 1])
      assertEquals("prepare_$i", events[base + 2])
      assertEquals("work_$i", events[base + 3])
      assertEquals("observer_finish", events[base + 4])
      assertEquals("sample_$i", events[base + 5])
    }
  }

  @Test
  fun testPositiveFiniteValidation() {
    assertEquals(123.45, positiveFinite("tok/s", 123.45), 0.001)

    try {
      positiveFinite("tok/s", 0.0)
      fail("0.0 must fail")
    } catch (e: IllegalStateException) {
      assertTrue(e.message!!.contains("invalid tok/s"))
    }

    try {
      positiveFinite("tok/s", -1.0)
      fail("-1.0 must fail")
    } catch (e: IllegalStateException) {
      assertTrue(e.message!!.contains("invalid tok/s"))
    }

    try {
      positiveFinite("tok/s", Double.NaN)
      fail("NaN must fail")
    } catch (e: IllegalStateException) {
      assertTrue(e.message!!.contains("invalid tok/s"))
    }
  }

  @Test
  fun testExpectTokensValidation() {
    expectTokens("tokens", 512, 512)

    try {
      expectTokens("tokens", 511, 512)
      fail("token mismatch must throw")
    } catch (e: IllegalStateException) {
      assertTrue(e.message!!.contains("engine returned tokens 511, expected 512"))
    }
  }

  @Test
  fun testThermalSeriesPairing() {
    val series = ThermalSeries()
    series.start(ThermalReading(thermalStatus = "none", thermalHeadroom = 0.2f))
    series.finish(ThermalReading(thermalStatus = "light", thermalHeadroom = 0.3f))

    val pairs = series.take()
    assertEquals(1, pairs.size)
    assertEquals("none", pairs[0].first.thermalStatus)
    assertEquals("light", pairs[0].second.thermalStatus)

    // Second start without finish must throw
    series.start(ThermalReading())
    try {
      series.start(ThermalReading())
      fail("double start must throw")
    } catch (e: IllegalStateException) {
      assertTrue(e.message!!.contains("previous one was still open"))
    }
  }
}
