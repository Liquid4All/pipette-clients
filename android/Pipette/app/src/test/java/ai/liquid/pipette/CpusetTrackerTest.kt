package ai.liquid.pipette

import ai.liquid.pipette.fakes.CpuAffinityFixtures.FOREGROUND
import ai.liquid.pipette.fakes.CpuAffinityFixtures.TOP_APP
import ai.liquid.pipette.fakes.CpuAffinityFixtures.demoted
import ai.liquid.pipette.fakes.CpuAffinityFixtures.snapshot
import ai.liquid.pipette.fakes.CpuAffinityFixtures.unreadable
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Exercises [CpusetTracker]'s policy off-device. Everything here is the pure decision core: no polling, no `/proc`, no threads. Samples are handed in
 * one at a time exactly as [CpusetWatcher] would hand them over.
 */
class CpusetTrackerTest {
  @Test
  fun firstSampleOnlyEstablishesTheBaseline() {
    val tracker = CpusetTracker(confirmations = 1)
    assertNull(tracker.offer(snapshot(TOP_APP)))
  }

  @Test
  fun leavingTopAppFiresOnlyAfterTheConfirmationCount() {
    val tracker = CpusetTracker(confirmations = 2)
    tracker.offer(snapshot(TOP_APP))
    // One demoted sample is a blip, not a verdict: a cell that has been running
    // for minutes must not be discarded on a single reading.
    assertNull(tracker.offer(snapshot(FOREGROUND)))
    val demotion = tracker.offer(snapshot(FOREGROUND))
    assertNotNull(demotion)
    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, demotion?.reason)
    assertEquals(TOP_APP, demotion?.baseline?.cpusetPath)
    assertEquals(FOREGROUND, demotion?.observed?.cpusetPath)
  }

  @Test
  fun aRecoveryBetweenTwoDemotedSamplesClearsTheStreak() {
    val tracker = CpusetTracker(confirmations = 2)
    tracker.offer(snapshot(TOP_APP))
    tracker.offer(snapshot(FOREGROUND))
    // Back on top: the run was never actually descheduled for a meaningful span.
    assertNull(tracker.offer(snapshot(TOP_APP)))
    // So the next demoted sample starts a fresh streak rather than completing the old one.
    assertNull(tracker.offer(snapshot(FOREGROUND)))
    assertNotNull(tracker.offer(snapshot(FOREGROUND)))
  }

  /**
   * The reason the baseline is the best state seen and not the first one: `JobRunner.launchPocketOnce` starts the pocket Activity just before the
   * model load and the `top-app` promotion lands asynchronously, so the first sample of a run is routinely taken while the process is still in
   * `/foreground`. Pinning the baseline there would compare a later drop back to `/foreground` as equal and never fire.
   */
  @Test
  fun aLatePromotionBecomesTheBaselineSoALaterDropStillFires() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(FOREGROUND))
    assertNull(tracker.offer(snapshot(TOP_APP)))
    val demotion = tracker.offer(snapshot(FOREGROUND))
    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, demotion?.reason)
  }

  /**
   * The fallback signal, for kernels whose cpuset path never says `top-app` (or says nothing useful): the process may simply run on fewer cores than
   * it started with.
   */
  @Test
  fun aShrunkenCpuSetFiresWithoutAnyTopAppSignal() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(cpuset = null, allowed = "0-7", allowedCount = 8))
    val demotion = tracker.offer(snapshot(cpuset = null, allowed = "0-3", allowedCount = 4, excludesTopTier = true))
    assertEquals(CpusetDemotion.Reason.LOST_CPUS, demotion?.reason)
  }

  /** Same core count, different cores: the prime tier is gone and the count alone would miss it. */
  @Test
  fun losingThePrimeTierFiresEvenWhenTheCoreCountIsUnchanged() {
    val tracker = CpusetTracker(confirmations = 1)
    // Both hold four cores; only the second is missing the prime tier (2-4), so the count alone cannot tell them apart.
    tracker.offer(snapshot(cpuset = FOREGROUND, allowed = "2-5", allowedCount = 4, excludesTopTier = false))
    val demotion = tracker.offer(snapshot(cpuset = FOREGROUND, allowed = "0-1,5-6", allowedCount = 4, excludesTopTier = true))
    assertEquals(CpusetDemotion.Reason.LOST_CPUS, demotion?.reason)
  }

  /**
   * There is no device-independent ordering of `/foreground` against `/moderate`, so a move between them on an unchanged core set is not ranked.
   * Guessing would produce confident nonsense on exactly the OEM devices this guard exists for.
   */
  @Test
  fun movingBetweenTwoNonTopAppGroupsWithTheSameCoresDoesNotFire() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(FOREGROUND))
    assertNull(tracker.offer(snapshot("/moderate")))
  }

  @Test
  fun firesAtMostOnce() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(TOP_APP))
    assertNotNull(tracker.offer(snapshot(FOREGROUND)))
    // The watcher stops its thread on the first fire; this guarantees the policy
    // agrees, so a caller that keeps polling cannot fail a second cell on one event.
    assertNull(tracker.offer(snapshot(FOREGROUND)))
    assertNull(tracker.offer(snapshot(TOP_APP)))
  }

  @Test
  fun describeNamesWhatWasLost() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(TOP_APP, allowed = "0-7", allowedCount = 8, excludesTopTier = false))
    val text = tracker.offer(snapshot(FOREGROUND, allowed = "0-3", allowedCount = 4, excludesTopTier = true))?.describe().orEmpty()
    // This string is what lands in the cell's errorMessage, so it has to say more
    // than "something happened": it is the whole explanation a user ever sees.
    assertTrue(text, text.contains(TOP_APP) && text.contains(FOREGROUND))
    assertTrue(text, text.contains("0-7") && text.contains("0-3"))
    assertTrue(text, text.contains("prime cores"))
  }

  /**
   * An unreadable `/proc` entry is an absence of evidence, not evidence of demotion. Every field of [CpuAffinitySnapshot] degrades to null or zero
   * rather than throwing, so a null `cpusetPath` compares unequal to "/top-app" and an unparseable `Cpus_allowed_list` yields zero CPUs. Scored
   * naively, two transient read failures would fail a cell that was running perfectly.
   */
  @Test
  fun anUnreadableSampleIsNeverScoredAsADemotion() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(TOP_APP))
    assertNull(tracker.offer(unreadable()))
    assertNull(tracker.offer(unreadable()))
  }

  /** A bad read in the middle of a real demotion must not reset the streak either, or a flaky probe could hold off detection indefinitely. */
  @Test
  fun anUnreadableSampleDoesNotClearAnInFlightStreak() {
    val tracker = CpusetTracker(confirmations = 2)
    tracker.offer(snapshot(TOP_APP))
    assertNull(tracker.offer(snapshot(FOREGROUND)))
    assertNull(tracker.offer(unreadable()))
    assertNotNull(tracker.offer(snapshot(FOREGROUND)))
  }

  /** A partial read is judged on the half that is present, not the half that is missing. */
  @Test
  fun aPartialSampleIsJudgedOnlyOnWhatItCouldRead() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.offer(snapshot(TOP_APP, allowed = "0-7", allowedCount = 8))
    // cpuset readable and unchanged, Cpus_allowed_list unreadable: nothing to conclude.
    assertNull(tracker.offer(snapshot(TOP_APP, allowed = "", allowedCount = 0)))
    // cpuset unreadable, but the CPU set visibly shrank: that half is enough.
    assertEquals(CpusetDemotion.Reason.LOST_CPUS, tracker.offer(snapshot(cpuset = null, allowed = "0-3", allowedCount = 4))?.reason)
  }

  /**
   * The baseline has to outlive a cell. A process still demoted when the next cell starts would otherwise adopt the demotion as its own normal, see
   * no transition, and measure on in silence. `CellDemotionWatch` carries it by seeding each cell's tracker from the last one's [bestSeen].
   */
  @Test
  fun aSeededBaselineFiresOnAPersistentDemotionFromTheFirstSample() {
    val firstCell = CpusetTracker(confirmations = 1)
    firstCell.offer(snapshot(TOP_APP, allowed = "0-7", allowedCount = 8))
    assertNotNull(firstCell.offer(snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7)))

    // What CellDemotionWatch.stop/beginCell do between cells.
    val nextCell = CpusetTracker(confirmations = 1, seed = firstCell.bestSeen)

    // Still demoted, and still recognized: the first sample of the new cell is compared against the
    // carried `/top-app` baseline rather than becoming the baseline itself.
    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, nextCell.offer(snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7))?.reason)
  }

  /** A fresh tracker per cell is what clears the fired latch, so a cell after a demotion is guarded again. */
  @Test
  fun aNewCellsTrackerCanFireAgain() {
    val firstCell = CpusetTracker(confirmations = 1)
    firstCell.offer(snapshot(TOP_APP))
    assertNotNull(firstCell.offer(snapshot(FOREGROUND)))
    assertNull("a tracker fires at most once", firstCell.offer(snapshot(FOREGROUND)))

    val nextCell = CpusetTracker(confirmations = 1, seed = firstCell.bestSeen)
    assertNotNull("the next cell must be guarded again", nextCell.offer(snapshot(FOREGROUND)))
  }

  /** [CpusetTracker.bestSeen] tracks promotions, not the first sample; it is the baseline carried into the next cell. */
  @Test
  fun bestSeenReportsThePromotedState() {
    val tracker = CpusetTracker(confirmations = 2)
    tracker.offer(snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7))
    tracker.offer(snapshot(TOP_APP, allowed = "0-7", allowedCount = 8))
    assertEquals(TOP_APP, tracker.bestSeen?.cpusetPath)
  }

  /**
   * A baseline adopted from a partial sample carries `allowedCount` 0 and `excludesTopTier` false by default. Scoring a later complete sample against
   * those defaults would report a demotion that never happened.
   */
  @Test
  fun aPartialBaselineDoesNotManufactureALostPrimeTier() {
    val tracker = CpusetTracker(confirmations = 1)
    // cpuset readable, /proc/self/status not: usable enough to keep, not enough to compare cores against.
    tracker.offer(snapshot(TOP_APP, allowed = "", allowedCount = 0))
    assertNull(tracker.offer(snapshot(TOP_APP, allowed = "0-3", allowedCount = 4, excludesTopTier = true)))
  }

  /**
   * [CpusetTracker.observe] is the cell-start probe's path in. It must record how good things can be without ever scoring: with a seeded baseline it
   * could otherwise spend the fired latch on a demotion nobody is listening for yet, before the watcher is even armed.
   */
  @Test
  fun observeSeedsTheBaselineWithoutScoringOrFiring() {
    val tracker = CpusetTracker(confirmations = 1, seed = snapshot(TOP_APP, allowed = "0-7", allowedCount = 8))

    // Already demoted at cell start. Not a verdict: observe never returns one and must not consume the latch.
    tracker.observe(snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7))

    // Proof the latch is intact: the watcher's first real sample still fires.
    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, tracker.offer(snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7))?.reason)
  }

  /**
   * A non-`top-app` sample must never displace a `top-app` baseline, however many cores it claims. If it did, `demotionReason`'s LEFT_TOP_APP clause
   * could never fire again and only the weaker LOST_CPUS fallback would remain. `offer` is protected by ordering; `observe` relies on `isBetter`.
   */
  @Test
  fun observeNeverDisplacesATopAppBaselineWithASlowerGroup() {
    val tracker = CpusetTracker(confirmations = 1, seed = snapshot(TOP_APP, allowed = "0-3", allowedCount = 4))

    // More cores, but not top-app. Tempting to call "better"; it is not.
    tracker.observe(snapshot(FOREGROUND, allowed = "0-7", allowedCount = 8))

    assertEquals(TOP_APP, tracker.bestSeen?.cpusetPath)
    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, tracker.offer(snapshot(FOREGROUND, allowed = "0-7", allowedCount = 8))?.reason)
  }

  @Test
  fun observeIgnoresAnUnreadableSample() {
    val tracker = CpusetTracker(confirmations = 1)
    tracker.observe(unreadable())
    assertNull("an unreadable probe must not become the baseline", tracker.bestSeen)
  }

  @Test
  fun observePromotesWhenTheSampleIsGenuinelyBetter() {
    val tracker = CpusetTracker(confirmations = 1, seed = snapshot(FOREGROUND, allowed = "0-6", allowedCount = 7))
    tracker.observe(snapshot(TOP_APP, allowed = "0-7", allowedCount = 8))
    assertEquals(TOP_APP, tracker.bestSeen?.cpusetPath)
  }

  // ── never attaining top-app at all ──────────────────────────────────────

  /**
   * The Pixel repro, and the blind spot this arm exists for. Every sample is `/foreground` with the prime tier denied, so each one compares equal to
   * a `/foreground` baseline and the two transition reasons stay silent while the whole cell is measured without the prime cores.
   */
  @Test
  fun aRunThatNeverReachesTopAppIsRefused() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull("the grace has to allow for an asynchronous promotion", tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    val demotion = tracker.offer(demoted())

    assertNotNull(demotion)
    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, demotion?.reason)
    assertEquals("there was no transition, so both sides are the same observation", FOREGROUND, demotion?.baseline?.cpusetPath)
    assertEquals(FOREGROUND, demotion?.observed?.cpusetPath)
  }

  /**
   * The false positive that judging on the cpuset name alone would create.
   *
   * A process in the root cpuset reports `/`, which is neither null nor `/top-app` and yet says nothing whatever about scheduling. Firing on the name
   * would fail every cell on every device that does not spell its groups the way we expect, which is the exact population `LOST_CPUS` exists to
   * serve. The prime cores are all allowed here, so there is no evidence of anything.
   */
  @Test
  fun aNonTopAppPathThatKeepsThePrimeCoresIsNotRefused() {
    val tracker = CpusetTracker(topAppGrace = 2)

    repeat(6) { assertNull("an uninformative cpuset name is not evidence of demotion", tracker.offer(snapshot("/"))) }
    assertEquals(0, tracker.neverTopAppSeen)
  }

  /** The promotion lands asynchronously after the pocket Activity starts, so a healthy cell spends its first samples outside `top-app`. */
  @Test
  fun aSlowPromotionInsideTheGraceIsNotRefused() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    assertNull("arriving on the last sample of the grace is still arriving", tracker.offer(snapshot(TOP_APP)))
    // Asserted on the streak itself rather than by feeding a fourth demoted sample. That sample would prove nothing here: reaching `top-app` makes it
    // the baseline, so a later drop is routed to `LEFT_TOP_APP` and never reaches this arm at all (`aRunThatReachedTopAppReportsTheTransitionInstead`
    // pins that). Zero is what separates spent from merely paused.
    assertEquals("and the streak is spent, not merely paused", 0, tracker.neverTopAppSeen)
  }

  /**
   * A null `cpusetPath` means the kernel did not say where the process is, not that it is somewhere bad.
   *
   * This is the *usable* unreadable case: the cpuset path could not be read but the CPU list could, so the sample survives `offer`'s usability guard
   * and reaches this arm. The all-unreadable sample is covered by `anUnreadableSampleIsNeverScoredAsADemotion`, which never gets this far.
   */
  @Test
  fun anUnreadableCpusetIsNeverJudgedAsNeverAttaining() {
    val tracker = CpusetTracker(topAppGrace = 2)

    repeat(6) { assertNull("absence of evidence is not evidence of demotion", tracker.offer(demoted(cpuset = null))) }
    assertEquals(0, tracker.neverTopAppSeen)
  }

  /**
   * Held, not cleared, across a sample whose cpuset could not be read, exactly as `offer` holds `streak` across a sample it cannot use.
   *
   * Clearing there would let one flaky `/proc/self/cpuset` read per grace window disarm the guard indefinitely on a device that is genuinely never
   * granting `top-app`.
   */
  @Test
  fun anUnreadableCpusetDoesNotClearAnInFlightNeverAttainedStreak() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    assertNull("the bad read neither confirms nor clears", tracker.offer(demoted(cpuset = null)))

    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, tracker.offer(demoted())?.reason)
  }

  /**
   * The other half of the evidence, and the subtler way to lose it.
   *
   * `excludesTopTier` is derived from the allowed-CPU list, and `CpuAffinityProbe.buildSnapshot` reports it as false when that list is unreadable. So
   * a failed `/proc/self/status` read arrives looking exactly like good news: cpuset readable, prime cores apparently fine. Cleared rather than held,
   * one such read per grace window would disarm the guard indefinitely on a device that never grants `top-app`.
   */
  @Test
  fun anUnreadableCpuListDoesNotClearAnInFlightNeverAttainedStreak() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    assertNull(
      "a missing allowed-CPU list is not evidence the prime cores were granted",
      tracker.offer(snapshot(FOREGROUND, allowed = null, allowedCount = 0)),
    )

    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, tracker.offer(demoted())?.reason)
  }

  /**
   * The third way to lose the evidence, and the one with no other arm to fall back on.
   *
   * `excludesTopTier` needs the `/sys` frequency sweep as well as the allowed-CPU list, and a sweep that read nothing leaves `topTierCpus` null and
   * `excludesTopTier` false, which is indistinguishable from "the prime cores were granted" unless it is held rather than cleared. Unlike
   * [CpusetDemotion.Reason.LOST_CPUS] this arm has no core-count clause to fall back on: the prime-tier comparison is its only corroboration.
   */
  @Test
  fun anUnreadableFrequencySweepDoesNotClearAnInFlightNeverAttainedStreak() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    assertNull(
      "a missing prime tier is not evidence the prime cores were granted",
      tracker.offer(snapshot(FOREGROUND, allowed = "0-1,5-7", allowedCount = 5, topTierCpus = null)),
    )

    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, tracker.offer(demoted())?.reason)
  }

  /**
   * A partial frequency sweep: the one form of missing evidence that does not look missing.
   *
   * Every field is populated here, a cpuset and an allowed-CPU list and a prime tier. But the tier was computed from the cores that were online, and
   * the ones that dropped out were the fast ones, so it names a slower cluster this process is free to run on and `excludesTopTier` reads false.
   * Judged on that, the guard clears its streak and stays silent on a dozing device, which is precisely the population it was written for.
   */
  @Test
  fun aPartialFrequencySweepDoesNotClearAnInFlightNeverAttainedStreak() {
    val tracker = CpusetTracker(topAppGrace = 3)

    assertNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
    // Same allowed set as `demoted()`, so nothing here is a core-count change; the only thing that moved is which cores the sweep managed to read.
    // The surviving tier `5-6` sits inside the allowed set, so `excludesTopTier` reads false, which is exactly the false reassurance under test.
    assertNull(
      "a tier computed from whichever cores were online is not evidence the prime cores were granted",
      tracker.offer(snapshot(FOREGROUND, allowed = "0-1,5-7", allowedCount = 5, topTierCpus = "5-6", sweptCpus = 7)),
    )

    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, tracker.offer(demoted())?.reason)
  }

  /**
   * Once a run has been to `top-app`, a fall away from it is `LEFT_TOP_APP`'s to report, with the transition it actually saw.
   *
   * A structural claim rather than a behavioural one, and it would survive deleting the never-attained arm entirely: `demotionReason` classifies
   * these samples before `neverTopApp` is consulted, which is exactly the precedence being recorded. Kept for that, and because the grace is set low
   * enough here that a reordering handing these samples to the other arm would show up as a changed reason.
   */
  @Test
  fun aRunThatReachedTopAppReportsTheTransitionInstead() {
    val tracker = CpusetTracker(confirmations = 2, topAppGrace = 2)
    tracker.offer(snapshot(TOP_APP))

    assertNull(tracker.offer(snapshot(FOREGROUND)))
    val demotion = tracker.offer(snapshot(FOREGROUND))

    assertEquals(CpusetDemotion.Reason.LEFT_TOP_APP, demotion?.reason)
    assertEquals("the baseline is the good state it actually had", TOP_APP, demotion?.baseline?.cpusetPath)
  }

  /**
   * The condition is a property of the run, not of one cell.
   *
   * A job of cells shorter than the grace would otherwise reset the count on every cell and never reach a verdict, though the device denied `top-app`
   * throughout. `CellDemotionWatch` carries the streak the same way it carries the baseline.
   */
  @Test
  fun theStreakCarriesIntoTheNextCell() {
    val first = CpusetTracker(topAppGrace = 4)
    repeat(3) { assertNull(first.offer(demoted())) }
    assertEquals(3, first.neverTopAppSeen)

    val next = CpusetTracker(topAppGrace = 4, neverTopAppSeed = first.neverTopAppSeen)

    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, next.offer(demoted())?.reason)
  }

  /**
   * A carried streak is clamped into the grace's range, which bounds the number without changing any verdict.
   *
   * Deliberately not asserted as "the next cell gets grace": it does not. A clamped seed of `grace - 1` and an unclamped one both fire on the next
   * judgeable sample, which is what the second assertion pins. The clamp keeps the carried value from growing without limit across a long job while
   * meaning nothing more than "at the grace".
   */
  @Test
  fun aCarriedStreakIsClampedIntoTheGraceRange() {
    val next = CpusetTracker(topAppGrace = 3, neverTopAppSeed = 99)

    assertEquals("clamped to one below the grace, however large the carried value", 2, next.neverTopAppSeen)
    assertEquals(CpusetDemotion.Reason.NEVER_TOP_APP, next.offer(demoted())?.reason)
  }

  /** A carried streak must not outlive the condition: one good sample in the next cell says the device is granting `top-app` after all. */
  @Test
  fun aCarriedStreakIsClearedByReachingTopApp() {
    val next = CpusetTracker(topAppGrace = 2, neverTopAppSeed = 99)

    assertNull(next.offer(snapshot(TOP_APP)))
    assertEquals(0, next.neverTopAppSeen)
  }

  /** Fires once and never again, like the transition arms, so one cell cannot report two verdicts. */
  @Test
  fun neverAttainingFiresOnlyOnce() {
    val tracker = CpusetTracker(topAppGrace = 1)

    assertNotNull(tracker.offer(demoted()))
    assertNull(tracker.offer(demoted()))
  }

  /** A grace of zero would fire on the first sample that is not judgeable, `top-app` included, which inverts the whole reason. */
  @Test(expected = IllegalArgumentException::class)
  fun aGraceBelowOneIsRefused() {
    CpusetTracker(topAppGrace = 0)
  }

  /** The message has no transition to render, so it must not print an arrow between two identical states. */
  @Test
  fun theRefusalReadsAsAStateRatherThanAMove() {
    val demotion = CpusetTracker(topAppGrace = 1).offer(demoted())!!
    val text = demotion.describe()

    assertTrue(text, text.contains("never scheduled this benchmark as the foreground app"))
    assertTrue("names the prime cores it never got: $text", text.contains("never reaching the prime cores (2-4)"))
    assertTrue("tells an unattended caller what to change: $text", text.contains("kept awake"))
    assertFalse("no arrow between two identical states: $text", text.contains("->"))
  }
}
