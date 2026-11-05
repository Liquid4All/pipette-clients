package ai.liquid.pipette.fakes

import ai.liquid.pipette.CpuAffinitySnapshot

/**
 * One [CpuAffinitySnapshot] builder for every test that needs one, rather than a copy per file.
 *
 * Worth centralizing rather than repeating: `excludesTopTier` is only meaningful relative to which cores a fixture calls the prime tier, so two
 * builders disagreeing about that would have two suites testing subtly different things while appearing to test one.
 *
 * The defaults describe a healthy 8-core device with its prime tier present, so a test only states the field it is actually exercising.
 */
object CpuAffinityFixtures {
  const val TOP_APP = "/top-app"
  const val FOREGROUND = "/foreground"
  const val BACKGROUND = "/background"

  /** The prime-core tier every fixture agrees on. Matches the Pixel 10 Pro Fold, where cpu2-4 are the 3.05 GHz cores. */
  const val TOP_TIER = "2-4"

  fun snapshot(
    cpuset: String?,
    allowed: String? = "0-7",
    allowedCount: Int = 8,
    excludesTopTier: Boolean = false,
    topTierCpus: String? = TOP_TIER,
    // Defaults to a complete sweep, matching the other defaults describing a healthy device. A test that wants the partial case says so, and gets a
    // snapshot whose prime-tier fields are present but not to be trusted.
    sweptCpus: Int = 8,
  ): CpuAffinitySnapshot =
    CpuAffinitySnapshot(
      cpusetPath = cpuset,
      allowedCpus = allowed,
      allowedCount = allowedCount,
      totalCpus = 8,
      availableProcessors = allowedCount,
      topTierCpus = topTierCpus,
      excludesTopTier = excludesTopTier,
      sweptCpus = sweptCpus,
    )

  /** A snapshot with nothing readable, standing in for a `/proc` read that failed. */
  fun unreadable(): CpuAffinitySnapshot = snapshot(cpuset = null, allowed = null, allowedCount = 0, topTierCpus = null)

  /**
   * A process denied the prime tier: `/foreground` on five cores with [TOP_TIER] excluded. The shape of the screen-off case behind
   * `CpusetDemotion.Reason.NEVER_TOP_APP`, in this object's own core layout rather than the one that device reported.
   *
   * Here rather than per file precisely because of what this object exists for. The exclusion is not decoration: it is the corroborating evidence the
   * never-attained check requires before it will judge a sample at all, so a copy of this that drifted on `excludesTopTier` would leave one suite
   * silently testing nothing while still passing.
   *
   * [cpuset] is a parameter so the unreadable-path case can reuse the same core layout, which is the point of that case: readable cores, unreadable
   * group.
   */
  fun demoted(cpuset: String? = FOREGROUND): CpuAffinitySnapshot =
    snapshot(cpuset = cpuset, allowed = "0-1,5-7", allowedCount = 5, excludesTopTier = true)
}
