package ai.liquid.pipette.fakes

import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.ResultSubmissionOutcome
import ai.liquid.pipette.ResultSubmitter

/**
 * Records submission calls so tests can assert whether/when a job was submitted.
 *
 * [failFirst] makes the leading N calls fail the way a management-server outage does (errors reported, no `serverJobId` stamped), so the end-of-run
 * sweep, which only has work to do when an earlier upload failed, is reachable from a runner-level test. Without it every call succeeds and the sweep
 * is permanently unexercised.
 */
class FakeResultSubmitter(private val failFirst: Int = 0) : ResultSubmitter {
  data class Submission(val jobId: String, val clientId: String)

  val submissions = mutableListOf<Submission>()

  override fun submit(manifest: JobManifest, registration: RegistrationData): ResultSubmissionOutcome {
    submissions += Submission(manifest.jobId, registration.clientId)
    if (submissions.size <= failFirst) {
      return ResultSubmissionOutcome(manifest, 0, listOf("simulated submit failure"))
    }
    val accepted = manifest.cells.filter { it.runStatus == CellRunStatus.COMPLETED && it.serverJobId == null }
    // Stamp the server id the real [ai.liquid.pipette.ResultSubmissionService] writes on a successful
    // submit. Without it this double reports every cell as perpetually unsent, so callers that skip a
    // submit with nothing to do (the end-of-run sweep after the per-cell uploads) look broken here
    // while behaving correctly in production.
    accepted.forEach { it.serverJobId = "server-${it.cellId}" }
    return ResultSubmissionOutcome(manifest, accepted.size, emptyList())
  }
}
