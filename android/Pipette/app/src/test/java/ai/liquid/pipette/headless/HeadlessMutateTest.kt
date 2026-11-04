package ai.liquid.pipette.headless

import ai.liquid.pipette.IdentityBundle
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.ModelUri
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.RegistrationData
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What a mutating verb decides to touch, and what it refuses.
 *
 * The whole reason resolution is a pure function over [HeadlessInspect.Facts] is so the destructive half can be pinned here rather than only on a
 * device. A delete is the one operation where "it did nothing" and "it did the wrong thing" are both silent on the happy path, so every case below
 * asserts the *target*, not just that something was returned.
 */
class HeadlessMutateTest {

  private fun model(name: String, repo: String? = "LiquidAI/LFM2.5-230M-GGUF") =
    ModelFile(name = name, path = "/models/$name", sizeBytes = 149_080_928L, hfRepo = repo, onDisk = true)

  private fun facts(
    models: List<ModelFile> = emptyList(),
    manifests: List<JobManifest> = emptyList(),
    results: List<HeadlessInspect.Result> = emptyList(),
    running: Boolean = false,
    privateKeyHex: PrivateKeyHex? = null,
    privateKeyUnreadable: Boolean = false,
    registration: RegistrationData? = null,
  ) =
    HeadlessInspect.Facts(
      buildVersion = "dev-gabc1234-debug",
      packageName = "ai.liquid.pipette.debug",
      registered = true,
      jobRunning = running,
      models = models,
      manifests = manifests,
      benchmarks = emptyList(),
      selectableIds = emptySet(),
      modelsDirPath = "/sdcard/models",
      freeBytes = 1_000L,
      runtime = HeadlessInspect.Runtime(id = "llama_cpp", available = true, build = "b4567"),
      results = results,
      registration = registration,
      privateKeyHex = privateKeyHex,
      privateKeyUnreadable = privateKeyUnreadable,
    )

  private fun result(cellId: String, jobId: String) =
    HeadlessInspect.Result(
      jobId = jobId,
      cellId = cellId,
      benchmarkId = "prefill_throughput_100",
      benchmarkType = "prefill_throughput",
      modelName = "LiquidAI/LFM2.5-230M-GGUF",
      quant = "q4_0",
      serverJobId = null,
      metricValue = 305.331,
      metricUnit = "tok/s",
      createdAt = "2026-08-12T00:00:00Z",
    )

  @Suppress("LongParameterList") // one optional per selector across every mutating verb; the alternative is a builder per verb
  private fun query(
    verb: HeadlessVerb,
    jobId: String? = null,
    resultId: String? = null,
    name: String? = null,
    repo: String? = null,
    model: String? = null,
    bundle: String? = null,
    force: Boolean = false,
  ) =
    HeadlessCommand.Query(
      verb = verb,
      jobId = jobId,
      resultId = resultId,
      name = name,
      repo = repo,
      modelUri = model?.let { (ModelUri.parse(it) as ModelUri.Result.Ok).model },
      // As the parser sets it: the raw text, which is what a refusal quotes back.
      modelSpelling = model,
      bundle = bundle,
      force = force,
    )

  /** A real, restorable bundle, so an import test that expects to proceed is not passing on a refusal for some other reason. */
  private fun token(privateKeyHex: PrivateKeyHex = HeadlessTestFixtures.privateKeyHex()) =
    IdentityBundle(privateKeyHex = privateKeyHex, registration = HeadlessTestFixtures.registration(), userToken = "hf_user").encode()

  @Suppress("LongParameterList") // one per branch of the import decision; each is a separate refusal below
  private fun resolveImport(
    bundle: String,
    storedKey: PrivateKeyHex? = null,
    force: Boolean = false,
    keyUnreadable: Boolean = false,
    registered: Boolean = true,
  ) =
    HeadlessMutate.resolve(
      query(HeadlessVerb.AUTH_IMPORT, bundle = bundle, force = force),
      facts(
        privateKeyHex = storedKey,
        privateKeyUnreadable = keyUnreadable,
        registration = if (registered) HeadlessTestFixtures.registration() else null,
      ),
    )

  @Test
  fun jobRmTargetsTheNamedJob() {
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.JOB_RM, jobId = "job-a"),
        facts(manifests = listOf(HeadlessTestFixtures.manifest("job-a"), HeadlessTestFixtures.manifest("job-b"))),
      )
    assertEquals(HeadlessMutate.Outcome.DeleteJob("job-a"), outcome)
  }

  @Test
  fun jobRmRefusesAnIdThatIsNotHere() {
    val outcome =
      HeadlessMutate.resolve(query(HeadlessVerb.JOB_RM, jobId = "nope"), facts(manifests = listOf(HeadlessTestFixtures.manifest("job-a"))))
    assertEquals(HeadlessMutate.Outcome.Refused("no job with id 'nope'"), outcome)
  }

  /**
   * `JobRunner` writes payloads into the job directory as cells finish, so removing it mid-run leaves the runner writing to a path that is gone. The
   * Jobs screen has the same hole and does not check; this surface is the unattended one and does.
   */
  @Test
  fun jobRmRefusesWhileAJobIsRunning() {
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.JOB_RM, jobId = "job-a"),
        facts(manifests = listOf(HeadlessTestFixtures.manifest("job-a")), running = true),
      )
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused && outcome.reason.contains("a job is running"))
  }

  /** Export reads the job and writes elsewhere, so a running job is no reason to refuse it. */
  @Test
  fun jobExportIsAllowedWhileAJobIsRunning() {
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.JOB_EXPORT, jobId = "job-a"),
        facts(manifests = listOf(HeadlessTestFixtures.manifest("job-a")), running = true),
      )
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.ExportJob && outcome.manifest.jobId == "job-a")
  }

  /** A cell id is unique across jobs, so resolving one has to hand back the job that owns it: `clearCellArtifacts` takes both. */
  @Test
  fun resultsDeleteCarriesTheOwningJob() {
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.RESULTS_DELETE, resultId = "cell-x"),
        facts(results = listOf(result("cell-w", "job-a"), result("cell-x", "job-b"))),
      )
    assertEquals(HeadlessMutate.Outcome.DeleteResult(jobId = "job-b", cellId = "cell-x"), outcome)
  }

  @Test
  fun resultsDeleteRefusesACellWithNoStoredResult() {
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.RESULTS_DELETE, resultId = "cell-x"), facts(results = emptyList()))
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused && outcome.reason.contains("cell-x"))
  }

  @Test
  fun modelsRmMatchesTheFileNameExactly() {
    val world = facts(models = listOf(model("LFM2.5-230M-Q4_0.gguf"), model("LFM2.5-230M-Q4_K_M.gguf")))
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_RM, name = "LFM2.5-230M-Q4_0.gguf"), world)
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels && outcome.files.map { it.name } == listOf("LFM2.5-230M-Q4_0.gguf"))
  }

  /**
   * Exact, not substring, and this is the case that separates them.
   *
   * `run`'s `match=` is a substring because a loose handle there costs a wrong measurement. The same looseness here would delete every quant of the
   * family when the caller named one.
   */
  @Test
  fun modelsRmDoesNotMatchOnASubstring() {
    val world = facts(models = listOf(model("LFM2.5-230M-Q4_0.gguf"), model("LFM2.5-230M-Q4_K_M.gguf")))
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_RM, name = "LFM2.5-230M"), world)
    assertTrue("a partial name must not delete the family: $outcome", outcome is HeadlessMutate.Outcome.Refused)
  }

  /** `repo=` alone names every quant of that repo, which is the reclaim-a-family case. */
  @Test
  fun modelsRmByRepoTakesEveryQuantOfIt() {
    val world =
      facts(
        models =
          listOf(
            model("LFM2.5-230M-Q4_0.gguf", repo = "LiquidAI/LFM2.5-230M-GGUF"),
            model("LFM2.5-230M-Q4_K_M.gguf", repo = "LiquidAI/LFM2.5-230M-GGUF"),
            model("LFM2-700M-Q4_0.gguf", repo = "LiquidAI/LFM2-700M-GGUF"),
          )
      )
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_RM, repo = "LiquidAI/LFM2.5-230M-GGUF"), world)
    // Which two, not just how many: a filter that picked the wrong pair would satisfy a count.
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels)
    assertEquals(listOf("LFM2.5-230M-Q4_0.gguf", "LFM2.5-230M-Q4_K_M.gguf"), (outcome as HeadlessMutate.Outcome.DeleteModels).files.map { it.name })
  }

  /**
   * A vision coordinate takes its projector only when nothing else on device pairs with it.
   *
   * Android stores by coordinate, `models/<repo>/<filename>`, so one `mmproj-model-f16.gguf` serves every quant of that repo. Deleting the Q4_0 and
   * its projector would leave the Q8_0 on disk and unable to run, which is the failure this guards. iOS cannot hit it: `ModelArtifactStore` keeps
   * each spec's files as its own entry.
   */
  @Test
  fun aSharedProjectorSurvivesWhileASiblingQuantNeedsIt() {
    val world =
      facts(
        models =
          listOf(
            model("LFM2.5-VL-Q4_0.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"),
            model("LFM2.5-VL-Q8_0.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"),
            model("mmproj-model-f16.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"),
          )
      )
    val coordinate = "gguf-vision://repo=LiquidAI/LFM2.5-VL-GGUF&model=LFM2.5-VL-Q4_0.gguf&mmproj=mmproj-model-f16.gguf"

    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_DELETE, model = coordinate), world)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels)
    assertEquals(listOf("LFM2.5-VL-Q4_0.gguf"), (outcome as HeadlessMutate.Outcome.DeleteModels).files.map { it.name })
  }

  /**
   * A projector is kept for a VL model in another repo, too.
   *
   * `JobRunner.isVlCompatible` pairs on the same repo *or* the same normalized stem, so a repo-only test would call this projector unneeded and
   * delete it out from under the model that was running with it.
   */
  @Test
  fun aProjectorSharedAcrossReposIsKeptForTheModelInTheOtherOne() {
    val world =
      facts(
        models =
          listOf(
            model("LFM2.5-VL-Q4_0.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"),
            model("mmproj-LFM2.5-VL-f16.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"),
            model("LFM2.5-VL-Q8_0.gguf", repo = "someone/LFM2.5-VL-GGUF-mirror"),
          )
      )
    val coordinate = "gguf-vision://repo=LiquidAI/LFM2.5-VL-GGUF&model=LFM2.5-VL-Q4_0.gguf&mmproj=mmproj-LFM2.5-VL-f16.gguf"

    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_DELETE, model = coordinate), world)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels)
    assertEquals(listOf("LFM2.5-VL-Q4_0.gguf"), (outcome as HeadlessMutate.Outcome.DeleteModels).files.map { it.name })
  }

  /** With the last quant of the repo going, the projector goes too: leaving it would be an orphan holding the space the caller asked to reclaim. */
  @Test
  fun theProjectorGoesWithTheLastQuantThatNeededIt() {
    val world =
      facts(
        models =
          listOf(model("LFM2.5-VL-Q4_0.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"), model("mmproj-model-f16.gguf", repo = "LiquidAI/LFM2.5-VL-GGUF"))
      )
    val coordinate = "gguf-vision://repo=LiquidAI/LFM2.5-VL-GGUF&model=LFM2.5-VL-Q4_0.gguf&mmproj=mmproj-model-f16.gguf"

    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_DELETE, model = coordinate), world)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels)
    assertEquals(listOf("LFM2.5-VL-Q4_0.gguf", "mmproj-model-f16.gguf"), (outcome as HeadlessMutate.Outcome.DeleteModels).files.map { it.name })
  }

  /** Both selectors narrow rather than widen, so a name that exists under a different repo is not matched. */
  @Test
  fun modelsRmWithBothSelectorsRequiresBoth() {
    val world = facts(models = listOf(model("LFM2.5-230M-Q4_0.gguf", repo = "LiquidAI/LFM2.5-230M-GGUF")))
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.MODELS_RM, name = "LFM2.5-230M-Q4_0.gguf", repo = "someone/else"), world)
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
  }

  @Test
  fun modelsDeleteResolvesACoordinateToItsBytes() {
    val world = facts(models = listOf(model("LFM2.5-230M-Q4_0.gguf", repo = "LiquidAI/LFM2.5-230M-GGUF")))
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.MODELS_DELETE, model = "gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf"),
        world,
      )
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.DeleteModels && outcome.files.single().name == "LFM2.5-230M-Q4_0.gguf")
  }

  /**
   * A coordinate that names nothing on disk is a refusal, not a delete of zero files.
   *
   * `resolveCoordinate` always hands back a pair, synthesizing the half that is absent, so the check that matters is `onDisk` rather than
   * nullability. The caller asked to reclaim space; if none would be reclaimed they should hear so.
   */
  @Test
  fun modelsDeleteRefusesACoordinateWithNoBytesHere() {
    val outcome =
      HeadlessMutate.resolve(
        query(HeadlessVerb.MODELS_DELETE, model = "gguf-text://repo=LiquidAI/LFM2-700M-GGUF&path=LFM2-700M-Q4_0.gguf"),
        facts(models = listOf(model("LFM2.5-230M-Q4_0.gguf"))),
      )
    // The coordinate is quoted back as written, not as parsed: interpolating the `Model` prints a data-class dump a caller cannot copy into a
    // command,
    // and its spaces and parens break the `key=value` line format this surface is read with.
    val reason = (outcome as HeadlessMutate.Outcome.Refused).reason
    assertEquals("no bytes on device for model='gguf-text://repo=LiquidAI/LFM2-700M-GGUF&path=LFM2-700M-Q4_0.gguf'", reason)
  }

  /**
   * No delete happens while a job runs, whichever delete asked.
   *
   * `job-rm` refused this from the start; the other three did not, and each fails silently in its own way. `JobRunner` loads a model file per cell,
   * so deleting weights it has not reached yet fails every later cell of a run that was going to succeed, and `clearCellArtifacts` races the payload
   * write of the cell in flight. Both report success and surface later, in a run nobody was watching.
   */
  @Test
  fun noDeleteHappensWhileAJobRuns() {
    val world =
      facts(
        manifests = listOf(HeadlessTestFixtures.manifest("job-1")),
        models = listOf(model("LFM2.5-230M-Q4_0.gguf")),
        results = listOf(result("cell-0", "job-1")),
        running = true,
      )
    listOf(
        query(HeadlessVerb.RESULTS_DELETE, resultId = "cell-0"),
        query(HeadlessVerb.MODELS_RM, name = "LFM2.5-230M-Q4_0.gguf"),
        query(HeadlessVerb.MODELS_DELETE, model = "gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf"),
      )
      .forEach { asked ->
        val outcome = HeadlessMutate.resolve(asked, world)
        // On the reason, not just on Refused: a resolver that stopped finding its target would refuse for "not found" and satisfy the weaker
        // assertion, so it could not tell a working guard from a broken lookup.
        val guarded = outcome is HeadlessMutate.Outcome.Refused && outcome.reason.contains("a job is running")
        assertTrue("${asked.verb.spelling} was not stopped by the running job: $outcome", guarded)
      }
  }

  /**
   * `job-export` is outside the guard on purpose: a run in flight makes its CSV incomplete, not wrong.
   *
   * Refusing it would deny a caller the results they already have, which is the opposite of what the guard is for.
   */
  @Test
  fun exportStillWorksWhileAJobRuns() {
    val world = facts(manifests = listOf(HeadlessTestFixtures.manifest("job-1")), running = true)
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.JOB_EXPORT, jobId = "job-1"), world)
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.ExportJob)
  }

  /**
   * A target that does not exist is reported as absent even while a job runs.
   *
   * The ordering `deleteJob` argues for, applied to the guard: "no such result" is actionable, where "a job is running" would send the caller off to
   * wait for something that was never going to help.
   */
  @Test
  fun anUnknownTargetBeatsTheRunningJobRefusal() {
    val world = facts(manifests = listOf(HeadlessTestFixtures.manifest("job-1")), results = listOf(result("cell-0", "job-1")), running = true)
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.RESULTS_DELETE, resultId = "no-such-cell"), world)
    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused && outcome.reason.contains("no-such-cell"))
  }

  /**
   * Every MUTATE verb resolves to something, which is the guarantee [HeadlessMutate.resolve]'s `else` arm costs.
   *
   * A `when` over the five is not exhaustive over the enum, so a sixth mutating verb would compile and fall to the `else`, refusing at runtime with
   * "is not a mutating verb" for a verb that plainly is. This walks the enum instead, so the omission fails here rather than on a device. The
   * arguments are deliberately the ones each verb needs; a verb reaching the `else` is what is being tested, not whether the target exists.
   */
  @Test
  fun mutatingVerbsAllResolve() {
    val world =
      facts(
        manifests = listOf(HeadlessTestFixtures.manifest("job-1")),
        models = listOf(model("LFM2.5-230M-Q4_0.gguf")),
        results = listOf(result("cell-0", "job-1")),
      )
    HeadlessVerb.entries
      .filter { it.mutates }
      .forEach { verb ->
        val outcome =
          HeadlessMutate.resolve(
            query(
              verb,
              jobId = "job-1",
              resultId = "cell-0",
              name = "LFM2.5-230M-Q4_0.gguf",
              model = "gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf",
              bundle = token(),
            ),
            world,
          )
        val unhandled = outcome is HeadlessMutate.Outcome.Refused && outcome.reason.endsWith("is not a mutating verb")
        assertFalse("${verb.spelling} fell through to the else arm: $outcome", unhandled)
      }
  }

  // ── auth-import ─────────────────────────────────────────────────────────

  /** A device holding no key restores unforced, which is the flow the whole verb pair exists for: hardware that has just been replaced. */
  @Test
  fun importOntoADeviceWithNoKeyProceedsUnforced() {
    val outcome = resolveImport(token())

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.ImportIdentity)
    val decided = outcome as HeadlessMutate.Outcome.ImportIdentity
    assertEquals(HeadlessTestFixtures.privateKeyHex(), decided.bundle.privateKeyHex)
    assertEquals(HeadlessTestFixtures.registration(), decided.bundle.registration)
    assertFalse("nothing was replaced", decided.replacesAnotherKey)
  }

  /**
   * A different key is refused without `force=1`, and the refusal says what an overwrite would cost.
   *
   * The one refusal on this surface that guards data the verb never touches: the results already submitted under the key it would replace.
   */
  @Test
  fun importRefusesOverADifferentKeyUnlessForced() {
    val outcome = resolveImport(token(), storedKey = HeadlessTestFixtures.privateKeyHex(fill = 99))

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    val reason = (outcome as HeadlessMutate.Outcome.Refused).reason
    assertTrue(reason, reason.contains("orphan"))
    assertTrue("offers the override, got: $reason", reason.contains("force=1"))
  }

  /**
   * A key the Keystore cannot open is refused, rather than read as an empty slot and overwritten.
   *
   * `Secrets.exportPrivateKeyHex` answers null for an empty slot and for an outage alike. Treating the outage as "no key here" turns the conflict
   * refusal off exactly when it matters: `seal` can succeed while `unseal` fails, so the import would destroy a key the next read would have
   * returned.
   */
  @Test
  fun importRefusesOverAKeyItCannotRead() {
    val outcome = resolveImport(token(), keyUnreadable = true)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    val reason = (outcome as HeadlessMutate.Outcome.Refused).reason
    assertTrue("names the Keystore, got: $reason", reason.contains("Keystore"))
    assertTrue("sends the caller to a retry, got: $reason", reason.contains("Try again"))
    assertTrue("names both identities, got: $reason", reason.endsWith(" client_id=ev1_a3f8c2d1 bundle_client_id=ev1_a3f8c2d1"))
  }

  /**
   * Every refusal names the identity it protected and the one it declined to restore.
   *
   * `HeadlessRunActivity` mirrors an `auth-import` refusal into the `pipette-auth` stream, which is the standing record of what happened to this
   * device's identity. A refusal recorded there without a client id says something was declined and nothing about what.
   */
  @Test
  fun everyImportRefusalNamesBothIdentities() {
    val other = HeadlessTestFixtures.privateKeyHex(fill = 99)
    val refusals =
      listOf(
        resolveImport(token(), storedKey = other),
        resolveImport(token(), keyUnreadable = true),
        resolveImport(token(), force = true, keyUnreadable = true),
      )

    for (outcome in refusals) {
      assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
      val reason = (outcome as HeadlessMutate.Outcome.Refused).reason
      assertTrue("the protected identity, got: $reason", reason.contains(" client_id=ev1_a3f8c2d1"))
      assertTrue("the offered identity, got: $reason", reason.contains(" bundle_client_id=ev1_a3f8c2d1"))
    }
  }

  /** A device with no registration has no id to name, and an empty `client_id=` would send a reader looking for one. */
  @Test
  fun aRefusalOnAnUnregisteredDeviceNamesOnlyTheBundle() {
    val outcome = resolveImport(token(), keyUnreadable = true, registered = false)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    val reason = (outcome as HeadlessMutate.Outcome.Refused).reason
    assertFalse("no device id to name, got: $reason", reason.contains(" client_id="))
    assertTrue("the bundle is still named, got: $reason", reason.contains(" bundle_client_id=ev1_a3f8c2d1"))
  }

  /**
   * `force=1` does not lift that refusal, because forcing cannot express a decision the operator can make.
   *
   * An unreadable slot reads exactly like an empty one, so the caller cannot tell a fresh device from the identity they would destroy. The outage
   * passes and the key returns, which makes this worth retrying rather than overriding.
   */
  @Test
  fun forceDoesNotOverrideAKeyThatCannotBeRead() {
    val outcome = resolveImport(token(), force = true, keyUnreadable = true)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    assertTrue("$outcome", (outcome as HeadlessMutate.Outcome.Refused).reason.contains("force=1 does not override this"))
  }

  /**
   * A malformed bundle is still refused first, so an outage does not mask a typo.
   *
   * The ordering the KDoc lists: a token that cannot decode never reaches the device at all, or the caller retries the outage forever over a bundle
   * that was never going to import.
   */
  @Test
  fun aMalformedBundleIsRefusedBeforeTheUnreadableKey() {
    val outcome = resolveImport("not-a-bundle", keyUnreadable = true)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    assertFalse("$outcome", (outcome as HeadlessMutate.Outcome.Refused).reason.contains("Keystore"))
  }

  /** `force=1` is the operator saying they meant it, and the decision carries the fact that another identity went. */
  @Test
  fun forcedImportProceedsAndRecordsThatItReplacedAKey() {
    val outcome = resolveImport(token(), storedKey = HeadlessTestFixtures.privateKeyHex(fill = 99), force = true)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.ImportIdentity)
    assertTrue("the replacement is what force=1 authorised", (outcome as HeadlessMutate.Outcome.ImportIdentity).replacesAnotherKey)
  }

  /**
   * The key the bundle itself carries is no conflict, so a rerun after a half-finished import completes the restore.
   *
   * Compared case-insensitively, because the value is hex and nothing guarantees which spelling a device stored. An exact match would refuse the
   * rerun this test describes and leave the operator with an identity they can neither finish nor clear.
   */
  @Test
  fun importProceedsUnforcedOverTheKeyTheBundleItselfCarries() {
    val stored = PrivateKeyHex(HeadlessTestFixtures.privateKeyHex().value.uppercase())

    val outcome = resolveImport(token(), storedKey = stored)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.ImportIdentity)
    assertFalse("the same key is not a replacement", (outcome as HeadlessMutate.Outcome.ImportIdentity).replacesAnotherKey)
  }

  /**
   * A malformed bundle is refused before the device is consulted, so a mistyped token cannot start a decision.
   *
   * Asserted over a key that *would* conflict: a bundle that failed to decode has to be refused for that reason and not for the conflict, or the
   * caller reaches for `force=1` and gets the same refusal again.
   */
  @Test
  fun importRefusesAMalformedBundleBeforeItLooksAtTheDevice() {
    val outcome = resolveImport("not a bundle!", storedKey = HeadlessTestFixtures.privateKeyHex(fill = 99), force = true)

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    assertTrue((outcome as HeadlessMutate.Outcome.Refused).reason, outcome.reason.contains("not base64url"))
  }

  /** No identity swap while a job runs: the submissions still in flight would report under the restored client id. */
  @Test
  fun importIsRefusedWhileAJobRuns() {
    val outcome = HeadlessMutate.resolve(query(HeadlessVerb.AUTH_IMPORT, bundle = token()), facts(running = true))

    assertTrue("$outcome", outcome is HeadlessMutate.Outcome.Refused)
    assertTrue((outcome as HeadlessMutate.Outcome.Refused).reason, outcome.reason.contains("a job is running"))
  }
}
