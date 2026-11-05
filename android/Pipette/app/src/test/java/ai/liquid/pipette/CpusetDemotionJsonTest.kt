package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.BACKGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The binder hop out of `:benchmark`.
 *
 * [CpusetDemotion.fromJson] swallows every failure to null and `RemoteBenchmarkEngine` reads null as "no demotion", so a renamed field or an unparsed
 * [CpusetDemotion.Reason] would disarm the guard with nothing failing anywhere. This is the only thing standing between that and a silent regression.
 *
 * Runs on plain JUnit: the module puts a real `org.json` on the test classpath (`libs.json.java`), so the serialization under test is the same code
 * that runs on a device rather than a stub.
 */
class CpusetDemotionJsonTest {
  @Test
  fun everyReasonSurvivesTheRoundTrip() {
    CpusetDemotion.Reason.entries.forEach { reason ->
      val original =
        CpusetDemotion(
          baseline = snapshot(TOP_APP, allowed = "0-7", allowedCount = 8),
          observed = snapshot(BACKGROUND, allowed = "0-2", allowedCount = 3, excludesTopTier = true),
          reason = reason,
        )

      assertEquals("Reason.$reason must survive the hop", original, CpusetDemotion.fromJson(original.toJson()))
    }
  }

  @Test
  fun aMalformedPayloadDecodesToNullRatherThanThrowing() {
    // Null is read as "no demotion", so this path fails open by design; it must at least not throw
    // across binder, where the exception would surface as a run failure instead.
    assertNull(CpusetDemotion.fromJson("not json"))
    assertNull(CpusetDemotion.fromJson("""{"reason":"NOT_A_REASON","baseline":"{}","observed":"{}"}"""))
  }
}
