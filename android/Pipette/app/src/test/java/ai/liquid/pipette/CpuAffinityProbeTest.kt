package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CpuAffinityProbeTest {
  @Test
  fun parseCpuList_handlesRangesSinglesAndGaps() {
    assertEquals(listOf(0, 1, 2, 3, 5, 7, 8), CpuAffinityProbe.parseCpuList("0-3,5,7-8"))
    assertEquals(listOf(4), CpuAffinityProbe.parseCpuList("4"))
    assertEquals(emptyList<Int>(), CpuAffinityProbe.parseCpuList(""))
  }

  @Test
  fun parseCpuList_skipsMalformedSegments() {
    assertEquals(listOf(0, 1, 5), CpuAffinityProbe.parseCpuList("0-1,foo,5"))
    assertEquals(emptyList<Int>(), CpuAffinityProbe.parseCpuList("9-3")) // inverted range dropped
  }

  @Test
  fun formatCpuList_isInverseOfParse() {
    assertEquals("0-3,5,7-8", CpuAffinityProbe.formatCpuList(setOf(8, 0, 1, 2, 3, 5, 7)))
    assertEquals("6-7", CpuAffinityProbe.formatCpuList(setOf(6, 7)))
    assertEquals("", CpuAffinityProbe.formatCpuList(emptySet()))
  }

  // Samsung-style demotion: an 8-core SoC (0-3 little @1.8G, 4-6 big @2.6G, 7
  // prime @3.0G), :benchmark process placed in /moderate and barred from the
  // prime (and one big) core.
  private val samsungMaxFreq =
    mapOf(0 to 1_800_000L, 1 to 1_800_000L, 2 to 1_800_000L, 3 to 1_800_000L, 4 to 2_600_000L, 5 to 2_600_000L, 6 to 2_600_000L, 7 to 3_000_000L)

  @Test
  fun buildSnapshot_flagsSamsungDemotion() {
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/moderate",
        allowedList = "0-5",
        availableProcessors = 6,
        sweep = CpuFreqSweep(samsungMaxFreq, possibleCpus = 8),
      )
    assertEquals("/moderate", snap.cpusetPath)
    assertEquals("0-5", snap.allowedCpus)
    assertEquals(6, snap.allowedCount)
    assertEquals(8, snap.totalCpus)
    assertEquals("7", snap.topTierCpus)
    assertTrue("prime core 7 is excluded", snap.excludesTopTier)
  }

  // Pixel/AOSP: a 9-core Tensor-style SoC (0-3 little, 4-7 mid, 8 prime); a
  // foreground-service process lands in /foreground with every core allowed.
  private val pixelMaxFreq =
    mapOf(
      0 to 1_700_000L,
      1 to 1_700_000L,
      2 to 1_700_000L,
      3 to 1_700_000L,
      4 to 2_400_000L,
      5 to 2_400_000L,
      6 to 2_400_000L,
      7 to 2_400_000L,
      8 to 3_100_000L,
    )

  @Test
  fun buildSnapshot_pixelKeepsPrimeCore() {
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/foreground",
        allowedList = "0-8",
        availableProcessors = 9,
        sweep = CpuFreqSweep(pixelMaxFreq, possibleCpus = 9),
      )
    assertEquals("/foreground", snap.cpusetPath)
    assertEquals(9, snap.allowedCount)
    assertEquals("8", snap.topTierCpus)
    assertFalse("all cores present, prime included", snap.excludesTopTier)
  }

  @Test
  fun buildSnapshot_missingFrequenciesLeaveTopTierNull() {
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = null,
        allowedList = "0-3",
        availableProcessors = 4,
        sweep = CpuFreqSweep(emptyMap(), possibleCpus = 4),
      )
    assertNull(snap.topTierCpus)
    assertFalse(snap.excludesTopTier)
    // The four cpuN directories exist even though not one of their frequencies could be read, so the device has four CPUs and the sweep covered none
    // of them. Derived from the map alone this used to report zero CPUs, which understated the SoC and made an empty sweep indistinguishable from a
    // device with no cores at all.
    assertEquals(4, snap.totalCpus)
    assertEquals(0, snap.sweptCpus)
    assertFalse(snap.sweepComplete)
  }

  @Test
  fun buildSnapshot_totalCpusUsesMaxIndexNotEntryCount() {
    // Core 3's scaling_max_freq is unreadable (gap in the map). totalCpus must
    // still be 8 (max index 7 + 1), not 7 (entry count) — else the
    // allowed/total summary undercounts the SoC.
    val gapped = samsungMaxFreq.filterKeys { it != 3 }
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/moderate",
        allowedList = "0-5",
        availableProcessors = 6,
        sweep = CpuFreqSweep(gapped, possibleCpus = 8),
      )
    assertEquals(8, snap.totalCpus)
    assertEquals(7, snap.sweptCpus)
    assertFalse("seven of eight cores read is not a complete sweep", snap.sweepComplete)
  }

  /**
   * The case a max-index total cannot see: the *prime* core is the one that dropped out.
   *
   * An offline core has no `cpufreq` directory, so cpu7 vanishing takes the top tier with it. Derived from the surviving indices alone, totalCpus
   * would be 7 and the sweep would look complete at 7 of 7, leaving `topTierCpus` naming the 2.6 GHz cluster that the process is perfectly free to
   * run on. The directory listing is what keeps the denominator honest.
   */
  @Test
  fun buildSnapshot_aMissingPrimeCoreStillCountsAgainstTheSweep() {
    val primeParked = samsungMaxFreq.filterKeys { it != 7 }
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/foreground",
        allowedList = "0-6",
        availableProcessors = 7,
        sweep = CpuFreqSweep(primeParked, possibleCpus = 8),
      )

    assertEquals("the listing knows about the parked core even though the sweep could not read it", 8, snap.totalCpus)
    assertEquals(7, snap.sweptCpus)
    assertFalse(snap.sweepComplete)
    // The trap this exists for: the surviving maximum looks like a prime tier, and it is fully allowed.
    assertEquals("4-6", snap.topTierCpus)
    assertFalse("which reads as healthy, and is why sweepComplete has to be consulted", snap.excludesTopTier)
  }

  @Test
  fun buildSnapshot_aCompleteSweepSaysSo() {
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/moderate",
        allowedList = "0-5",
        availableProcessors = 6,
        sweep = CpuFreqSweep(samsungMaxFreq, possibleCpus = 8),
      )
    assertEquals(8, snap.sweptCpus)
    assertTrue(snap.sweepComplete)
  }

  @Test
  fun snapshot_jsonRoundTrips() {
    val snap =
      CpuAffinityProbe.buildSnapshot(
        cpusetPath = "/moderate",
        allowedList = "0-5",
        availableProcessors = 6,
        sweep = CpuFreqSweep(samsungMaxFreq, possibleCpus = 8),
      )
    assertEquals(snap, CpuAffinitySnapshot.fromJson(snap.toJson()))
  }

  @Test
  fun fromJson_returnsNullOnGarbage() {
    assertNull(CpuAffinitySnapshot.fromJson("not json"))
  }
}
