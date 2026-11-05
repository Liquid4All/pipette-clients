package ai.liquid.pipette.fakes

import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobRunner
import ai.liquid.pipette.RuntimeFlagRef
import java.util.concurrent.Executor

/**
 * The cell-loop scaffolding shared by [ai.liquid.pipette.JobRunnerTest] and [ai.liquid.pipette.JobRunnerDemotionTest], rather than a copy per file.
 *
 * The two suites have to agree on what a job under test looks like: they assert on the same manifest, over the same fakes, and a second `runner`
 * builder that wired a different executor would make one of them quietly test a different thing.
 */
object JobRunnerFixtures {
  /** Same-thread executor, so `resume` blocks until the job finishes and the assertions read the final manifest state. */
  val directExecutor = Executor { it.run() }

  const val DECODE_ID = "decode_throughput_512_100"
  const val DECODE_TYPE = "decode_throughput"

  fun cell(modelPath: String, benchmarkId: String, benchmarkType: String?, runtimeFlags: RuntimeFlagRef? = null): JobCell =
    JobCell(
      benchmarkId = benchmarkId,
      benchmarkType = benchmarkType,
      modelPath = modelPath,
      modelName = modelPath.substringAfterLast('/'),
      runStatus = CellRunStatus.PENDING,
      runtimeFlags = runtimeFlags,
    )

  fun runner(
    store: FakeJobStore,
    engine: FakeBenchmarkEngine,
    readiness: FakeReadinessGate = FakeReadinessGate(),
    submitter: FakeResultSubmitter = FakeResultSubmitter(),
  ): JobRunner = JobRunner(storage = store, engine = engine, submissionService = submitter, readiness = readiness, executor = directExecutor) {}
}
