package ai.liquid.pipette.headless

import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStatus
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.toHex

/**
 * The manifests and cells the headless tests build, in one place.
 *
 * Four test classes in this package need the same shape and had grown their own copies, differing only in which statuses the cells carried. Those
 * statuses are the interesting part of most of these tests, so they are parameters here rather than a reason to keep four fixtures in step.
 */
object HeadlessTestFixtures {

  /** A cell with the fields every headless assertion reads and nothing else. [status] defaults to PENDING, as [JobCell] itself does. */
  fun cell(cellId: String, status: CellRunStatus = CellRunStatus.PENDING): JobCell =
    JobCell(
      cellId = cellId,
      benchmarkId = "prefill_throughput_100",
      benchmarkType = "prefill_throughput",
      modelPath = "m.gguf",
      modelName = "org/repo",
      runStatus = status,
    )

  /**
   * A manifest holding one [cell] per entry in [cells], named `cell-0` upward and carrying that entry's status.
   *
   * [status] is the job's own, which is not derived from the cells: a job can be RUNNING with every cell still PENDING, and `verb=jobs` prints both.
   */
  fun manifest(jobId: String, cells: List<CellRunStatus> = listOf(CellRunStatus.PENDING), status: JobStatus = JobStatus.COMPLETED): JobManifest =
    JobManifest(
      jobId = jobId,
      createdAt = "2026-08-12T00:00:00Z",
      nGpuLayers = 99,
      contextSize = 4096,
      cells = cells.mapIndexed { i, cellStatus -> cell("cell-$i", cellStatus) }.toMutableList(),
      status = status,
    )

  /** A registration record with every field populated, so a test that asserts a line can tell a dropped field from an empty one. */
  fun registration(clientId: String = "ev1_a3f8c2d1", status: String = "approved"): RegistrationData =
    RegistrationData(
      clientId = clientId,
      status = status,
      serverUrl = "https://pipette.liquid.ai",
      organization = "Liquid AI",
      contactEmail = "ops@liquid.ai",
      registeredAt = "2026-08-12T00:00:00Z",
    )

  /**
   * A real Ed25519 seed as hex, derived from [fill] so the value is stable across runs.
   *
   * Real rather than arbitrary hex because the identity verbs derive a public key from it, so a stub would exercise the refusal path instead of the
   * one under test. Deterministic because a test asserting a whole line needs to know what the line will say.
   */
  fun privateKeyHex(fill: Int = 7): PrivateKeyHex = PrivateKeyHex(ByteArray(32) { (it + fill).toByte() }.toHex())
}
