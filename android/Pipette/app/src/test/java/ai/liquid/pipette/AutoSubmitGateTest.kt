package ai.liquid.pipette

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the rule deciding whether a finished cell's result is uploaded mid-run.
 *
 * The gate runs after every cell of a benchmark job, so getting it wrong is expensive in both directions: too permissive and an offline device spends
 * the run's wall-clock on uploads that cannot succeed, between measurements that are supposed to be undisturbed; too strict and results sit on disk
 * until the job ends, which is exactly the stranding this per-cell upload exists to prevent. Mirrors the iOS client's `shouldAutoSubmit`.
 */
class AutoSubmitGateTest {
  private fun manifest(contribute: Boolean?): JobManifest =
    JobManifest(nGpuLayers = 99, contextSize = 4096, cells = mutableListOf(), status = JobStatus.RUNNING, contributeResults = contribute)

  private val registration =
    RegistrationData(
      clientId = "client-1",
      status = "approved",
      serverUrl = "https://mgmt.example.com",
      organization = "org",
      contactEmail = "someone@example.com",
      registeredAt = "2026-07-29T00:00:00Z",
    )

  @Test
  fun uploadsWhenContributingAndOnlineAndRegistered() {
    assertTrue(shouldAutoSubmit(manifest(contribute = true), online = true, registration = registration))
  }

  /** The user did not opt this job into contributing; its results are theirs alone and must never leave the device. */
  @Test
  fun neverUploadsAJobNotMarkedForContribution() {
    assertFalse(shouldAutoSubmit(manifest(contribute = false), online = true, registration = registration))
    assertFalse(shouldAutoSubmit(manifest(contribute = null), online = true, registration = registration))
  }

  /**
   * Offline is a skip, not a failure. The payload is already on disk, so the end-of-run sweep or the next launch sends it, whereas trying here would
   * stall the run behind a connect timeout between two timed cells.
   */
  @Test
  fun skipsWhileOffline() {
    assertFalse(shouldAutoSubmit(manifest(contribute = true), online = false, registration = registration))
  }

  /** No registration means no client id to submit under, and nothing to sign the request with. */
  @Test
  fun skipsWithoutARegistration() {
    assertFalse(shouldAutoSubmit(manifest(contribute = true), online = true, registration = null))
  }

  /**
   * The end-of-run sweep is deliberately NOT gated on reachability, unlike the per-cell path.
   *
   * That probe exists to keep a connect timeout out of the gap between two measurements; once the run is over there is no measurement left to
   * protect. It also reads false on a firewalled or LAN-only network, the deployments most likely to run an on-prem management server, and this sweep
   * ran unconditionally before the per-cell upload existed, so gating it would be a silent regression for exactly those devices.
   */
  @Test
  fun sweepIgnoresReachabilityAndOnlyNeedsSomethingUnsent() {
    val completedUnsent =
      manifest(contribute = true).apply {
        status = JobStatus.COMPLETED
        cells.add(JobCell(benchmarkId = "b", benchmarkType = "t", modelPath = "/m/a.gguf", modelName = "a.gguf", runStatus = CellRunStatus.COMPLETED))
      }
    assertTrue(shouldSweepAtRunEnd(completedUnsent, registration))

    // Everything already acknowledged: nothing to do, so no "Submitting results..." flash.
    completedUnsent.cells.single().serverJobId = "server-1"
    assertFalse(shouldSweepAtRunEnd(completedUnsent, registration))
  }

  /** A run that was cancelled or died isn't swept; its results go out on the next launch instead. */
  @Test
  fun sweepSkipsAJobThatDidNotComplete() {
    val paused =
      manifest(contribute = true).apply {
        status = JobStatus.PAUSED
        cells.add(JobCell(benchmarkId = "b", benchmarkType = "t", modelPath = "/m/a.gguf", modelName = "a.gguf", runStatus = CellRunStatus.COMPLETED))
      }
    assertFalse(shouldSweepAtRunEnd(paused, registration))
  }
}
