package ai.liquid.pipette.headless

import ai.liquid.pipette.CompletedResultsCsvExporter
import ai.liquid.pipette.LocalStorage
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.Secrets
import java.io.File

/**
 * Carries out what [HeadlessMutate] decided, and returns the lines that say what happened.
 *
 * The fourth part of the split this surface is built on: [HeadlessFacts] gathers, [HeadlessInspect] words an answer, [HeadlessMutate] decides which
 * thing to touch, and this does it. Separated from [HeadlessRunActivity] so the destructive half is reachable from a JVM test, which matters more
 * here than anywhere else on the surface: a delete's failure modes are all quiet, and the only two device passes that exercised these verbs each
 * found something the pure tests could not have.
 *
 * It returns lines rather than logging them, so a test can assert what a caller would have read. The activity logs them and sets its result from
 * [Report.exit].
 */
object HeadlessAct {

  /** Where `job-export` writes, beside the model store under `getExternalFilesDir`, so `adb pull` reaches it and uninstall reclaims it. */
  private const val EXPORTS_DIR = "exports"

  /** What happened, in the order it should be logged, and the exit code that goes with it. */
  data class Report(val lines: List<String>, val exit: Int)

  /**
   * Perform [outcome] against [storage] and [secrets].
   *
   * [secrets] is the other half of the device's identity, and only `auth-import` touches it. Passed alongside [storage] rather than reached for
   * because the two writes an import makes have to land in one order, and this is the only object that knows the order.
   *
   * [externalFilesDir] is `getExternalFilesDir(null)`, passed in rather than reached for, because it is the only Android service an act needs and
   * because it is nullable in a way that matters: see [exportJob].
   *
   * The whole body is caught, not just the arms that call [attempt], so a throw from any future arm is still attributed to the verb that caused it.
   * `HeadlessRunActivity.onCreate` catches too and is what guarantees the terminal sentinel; this catch is about the message, which would otherwise
   * be the activity's generic one and would not name the act that failed.
   */
  fun perform(outcome: HeadlessMutate.Outcome, storage: LocalStorage, secrets: Secrets, verb: String, externalFilesDir: File?): Report =
    runCatching { dispatch(outcome, storage, secrets, verb, externalFilesDir) }
      .getOrElse { Report(listOf("FAILED $verb: ${reason(it)}"), HeadlessRunActivity.EXIT_FAILURE) }

  private fun dispatch(outcome: HeadlessMutate.Outcome, storage: LocalStorage, secrets: Secrets, verb: String, externalFilesDir: File?): Report =
    when (outcome) {
      is HeadlessMutate.Outcome.Refused -> Report(listOf("REFUSED ${outcome.reason}"), HeadlessRunActivity.EXIT_FAILURE)
      is HeadlessMutate.Outcome.DeleteJob ->
        attempt(verb) {
          // `deleteRecursively` reports a partial delete by returning false and `LocalStorage` discards it, so the directory itself is the only
          // witness. Its prior existence is read first because "gone" and "was never here" look identical afterwards, and only one of them is a
          // delete: `loadAllJobManifests` takes the id from the manifest's contents, so a manifest whose id disagrees with its directory name would
          // otherwise report DELETED for a directory nothing touched.
          val dir = storage.jobDir(outcome.jobId)
          val existed = dir.exists()
          storage.deleteJob(outcome.jobId)
          check(existed) { "no job directory at ${dir.absolutePath}" }
          check(!dir.exists()) { "job directory is still on disk" }
          // `job=` and not the `job id=` that `verb=jobs` prints. Within these verbs the id is spelled one way, matching the `EXPORTED job=` line
          // below, so a consumer greping the mutating verbs for a job id needs one pattern rather than two.
          "DELETED job=${outcome.jobId}"
        }
      is HeadlessMutate.Outcome.DeleteResult ->
        attempt(verb) {
          storage.clearCellArtifacts(outcome.jobId, outcome.cellId)
          // All three artifacts, not just the payload. `clearCellArtifacts` removes the payload, the submission record and the metrics, and discards
          // every `File.delete` Boolean. Checking the payload alone would report a clear that partly happened, and since the payload is what
          // `HeadlessFacts.isReportable` keys on, the cell would stop listing while its siblings stayed on disk with no verb able to reach them.
          val left =
            listOf(
                storage.cellPayloadFile(outcome.jobId, outcome.cellId),
                storage.cellSubmissionFile(outcome.jobId, outcome.cellId),
                storage.cellMetricsFile(outcome.jobId, outcome.cellId),
              )
              .filter { it.exists() }
          // Names what did go as well as what did not, for the reason [deleteModels] reports per file: the payload is deleted first, and once it is
          // gone the cell stops listing under `verb=results`, so a bare failure would leave the caller thinking nothing happened to a result they can
          // no longer see.
          check(left.isEmpty()) {
            val names = left.joinToString(",") { it.name }
            // Which sentence is true depends on whether the payload was the survivor, and the two send an operator to different places: with the
            // payload gone the result has already stopped listing and the leftovers are unreachable, while with the payload still there nothing has
            // visibly changed at all.
            if (left.any { it.name == storage.cellPayloadFile(outcome.jobId, outcome.cellId).name }) "could not clear $names; the result still lists"
            else "cleared the payload but not $names; the result no longer lists and those bytes remain"
          }
          "DELETED result cell=${outcome.cellId} job=${outcome.jobId}"
        }
      is HeadlessMutate.Outcome.DeleteModels -> deleteModels(storage, outcome.files, verb)
      is HeadlessMutate.Outcome.ExportJob -> exportJob(storage, outcome, verb, externalFilesDir)
      is HeadlessMutate.Outcome.ImportIdentity -> importIdentity(storage, secrets, outcome)
    }

  /**
   * Write the restored identity, in the one order that survives an interruption.
   *
   * The signing key goes first, through [Secrets.replacePrivateKeyHex], which stages and promotes under one lock. That call seals the promoted value
   * before it clears anything, which keeps a momentarily unreachable Keystore from destroying every copy of a key at once. It also compares the
   * promoted slot against [HeadlessMutate.Outcome.ImportIdentity.observedKey] first, so an import a concurrent registration overtook writes nothing
   * rather than promoting one identity's key under another's client id. The registration follows, because a registration with no key reports the
   * device as registered while every signed request fails, and an interruption there is the worse of the two half-states. The token goes last,
   * because it is the only part a later `Settings` visit can replace by hand.
   *
   * Each write is checked and a failure throws, which [perform]'s catch turns into `FAILED auth-import: …`. A rerun of the same bundle finishes
   * whatever a failure left half-done: the key it finds is the one this attempt wrote, so [HeadlessMutate] reads it as no conflict and needs no
   * `force=1`.
   *
   * **A forced import never clears the existing Hugging Face token**, even one that replaces another identity. A bundle carrying `user_token`
   * overwrites it; a bundle without one leaves it alone. The token authenticates against huggingface.co rather than against the management server, so
   * it is the operator's to manage and not an attribute of the client id.
   *
   * iOS holds the same position, and it takes reading two files to see it. Its import clears the *per-model* tokens on a replacing import
   * (`AuthCommands.swift`, `replacesAnotherKey ? clearModelTokens() : 0`), and `deleteAllModelHfTokens` filters Keychain accounts on the `hf_token:`
   * prefix (`KeychainHelper.swift`), which the bare `hf_token` account the user token lives in does not match. `deleteHfToken()` is reached only from
   * a user-initiated sign-out in Settings. Even `auth reset force=1` leaves the user token, after a comment weighing whether "forget this device's
   * identity" should take every credential.
   *
   * This client has one token slot rather than one per model, and it is the counterpart of the account iOS preserves. There is nothing here that
   * corresponds to the tokens iOS clears; a bundle's per-model tokens are counted and reported as `model_tokens_discarded=` below.
   *
   * PIP-547 asked for the token to be cleared, from a reading of the model-token clear that does not carry to this slot. The ticket records the
   * correction.
   *
   * The exit code turns on whether a token the bundle carried actually stored, mirroring iOS's `missing == 0`. The identity itself is the half worth
   * having and it is reported as landed either way, but a token that did not store means a gated download 401s later, which is worth an exit code
   * rather than a line the operator has to compare against the export's.
   */
  private fun importIdentity(storage: LocalStorage, secrets: Secrets, outcome: HeadlessMutate.Outcome.ImportIdentity): Report {
    val bundle = outcome.bundle
    // The two races read as retryable and the two Keystore faults do not, so each says what the operator should do next rather than sharing one
    // "could not store the signing key" that sends a retry at a fault and a support ticket at a lost race.
    when (secrets.replacePrivateKeyHex(bundle.privateKeyHex.value, outcome.observedKey?.value, overStagedKey = outcome.forced)) {
      Secrets.KeyReplacement.REPLACED -> Unit
      Secrets.KeyReplacement.KEY_CHANGED ->
        error(
          "this device replaced its signing key while the import decided; the import wrote nothing. Rerun it" +
            identities(storage, bundle.registration.clientId)
        )
      Secrets.KeyReplacement.KEY_STAGED ->
        error(
          "a registration is claiming this device; the import wrote nothing. Let it finish and rerun, or pass force=1 if none is running" +
            identities(storage, bundle.registration.clientId)
        )
      Secrets.KeyReplacement.NOT_STAGED -> error("could not stage the signing key; the import wrote nothing")
      Secrets.KeyReplacement.NOT_PROMOTED -> error("could not store the signing key; the import wrote nothing else")
    }
    storage.saveRegistration(bundle.registration)
    val restored = if (bundle.userToken?.let { secrets.saveHfToken(it) } == true) 1 else 0
    val missing = bundle.tokenCount - restored
    val line = buildString {
      append("IMPORTED identity client_id=${bundle.registration.clientId} server_url=${bundle.registration.serverUrl}")
      append(" signing_key=ok registration=ok replaced_key=${outcome.replacesAnotherKey} hf_tokens=$restored")
      // Both omitted at zero, on the same grounds as `model`'s absent `repo`: `hf_tokens_failed=0` invites a reader to look for a failure there was
      // none of, and every successful import would carry it.
      if (missing != 0) append(" hf_tokens_failed=$missing")
      // Only an iOS bundle can carry these, and this client has one token slot rather than one per model. Reported because a credential that was
      // silently dropped means a gated download 401s on the restored device with nothing in the log to explain it.
      if (bundle.modelTokenCount != 0) append(" model_tokens_discarded=${bundle.modelTokenCount}")
    }
    return Report(listOf(line), if (missing == 0) HeadlessRunActivity.EXIT_OK else HeadlessRunActivity.EXIT_FAILURE)
  }

  /**
   * The two identities an `auth-import` failure is about, appended to its message.
   *
   * The same pair, in the same spelling, that [HeadlessMutate] appends to a refusal: `pipette-auth` keeps both kinds of declined import, and either
   * one is worth nothing to a later reader if it names no client. `client_id` keeps meaning "this device" as it does everywhere on this surface, so
   * the bundle's takes a key of its own.
   *
   * The device's id is read here rather than carried on the outcome because only these two branches want it, and both have already decided the import
   * writes nothing. Dropped when the device holds no registration, which is the ordinary state for the flow this verb is for.
   */
  private fun identities(storage: LocalStorage, bundleClientId: String): String = buildString {
    storage.loadRegistration()?.clientId?.let { append(" client_id=$it") }
    append(" bundle_client_id=$bundleClientId")
  }

  /**
   * Delete each file on its own, then report what actually went.
   *
   * Not one `runCatching` around the loop, which is what this was. [LocalStorage.deleteModel] throws on a path outside the model root, so it can fail
   * on one file while the ones before it are already gone, and a single catch would report total failure for a partial delete. "Failed" that means
   * "some of your models are deleted" is the worst reading available on an unattended surface.
   *
   * Deduplicated by path first. A vision coordinate names its weights and its projector separately and nothing stops a caller writing the same file
   * for both, which `ModelCatalog.resolveCoordinate` faithfully resolves to the same file twice; without this the reclaimed `bytes=` would be double
   * what the disk gave back.
   *
   * Files are named `repo/file` rather than by file name alone. The store buckets by repo, so two repos can hold the same file name, and a bare name
   * would leave a caller unable to tell from the log which one went.
   */
  private fun deleteModels(storage: LocalStorage, files: List<ModelFile>, verb: String): Report {
    // [HeadlessMutate] refuses both empty selections, so this is a latent hole rather than a live one. Stated anyway because the alternative is the
    // one outcome on this surface that prints nothing at all: a caller would read a bare successful sentinel and have no line saying what happened.
    if (files.isEmpty()) return Report(listOf("FAILED $verb: resolved to no files"), HeadlessRunActivity.EXIT_FAILURE)
    val attempted = files.distinctBy { it.path }.map { model -> model to whyNotDeleted(storage, model) }
    val gone = attempted.filter { it.second == null }.map { it.first }
    val deleted =
      if (gone.isEmpty()) emptyList()
      else listOf("DELETED models count=${gone.size} bytes=${gone.sumOf { it.sizeBytes }} files=${gone.joinToString(",", transform = ::label)}")
    val failed = attempted.filter { it.second != null }
    val failure =
      failed
        .takeIf { it.isNotEmpty() }
        ?.let { left ->
          "FAILED $verb: ${left.size} of ${attempted.size} not deleted: " + left.joinToString(", ") { (model, why) -> "${label(model)} ($why)" }
        }
    return Report(deleted + listOfNotNull(failure), if (failed.isEmpty()) HeadlessRunActivity.EXIT_OK else HeadlessRunActivity.EXIT_FAILURE)
  }

  /**
   * Delete one file, and say why it did not go if it did not. Null means it did.
   *
   * Existence is read on both sides of the delete, and the two readings catch different lies. Before, because `deleteModel` no-ops on a file that is
   * not there and returns normally, so a stale entry from `ModelStore`'s cached scan would otherwise be counted as bytes reclaimed when nothing was.
   * After, because `deleteModel` discards `File.delete`'s Boolean, so a file it could not remove also returns normally.
   *
   * The throwable's own message is carried through rather than dropped, since the failure line is the only record a caller on this surface gets, and
   * "outside app model storage" and a permission error want different responses.
   */
  private fun whyNotDeleted(storage: LocalStorage, model: ModelFile): String? {
    val file = File(model.path)
    val existed = file.exists()
    // Handed to the store even when the file is already gone, because `deleteModel` is what invalidates `ModelStore`'s cached scan and this activity
    // can run in a process that holds one. Without the call, the next `verb=models` there would keep listing the file and `storage-status` would keep
    // counting bytes that are not on disk.
    val thrown = runCatching { storage.deleteModel(model) }.exceptionOrNull()
    return when {
      !existed -> "already absent"
      // Asked before the throwable, because the delete can succeed and then throw on the way out: `deleteModel` prunes empty parent directories
      // afterwards, and `canonicalPath` in that walk is an IO call. The disk is the authority on whether the bytes went.
      !file.exists() -> null
      thrown != null -> reason(thrown)
      else -> "still on disk after delete"
    }
  }

  /**
   * Write the job's CSV and say where it went.
   *
   * [externalFilesDir] is null while external storage is unmounted, and `File(null, name)` is a path relative to the process working directory rather
   * than an error, so without this the export would land somewhere unreachable and report success.
   */
  private fun exportJob(storage: LocalStorage, outcome: HeadlessMutate.Outcome.ExportJob, verb: String, externalFilesDir: File?): Report =
    attempt(verb) {
      val dir = externalFilesDir ?: error("external storage is unavailable")
      val exportsDir = File(dir, EXPORTS_DIR).apply { mkdirs() }
      val target = File(exportsDir, CompletedResultsCsvExporter.filename(outcome.manifest))
      target.writeText(CompletedResultsCsvExporter.csv(storage, outcome.manifest))
      val payloadsDir = File(exportsDir, outcome.manifest.jobId).apply { mkdirs() }
      outcome.manifest.cells.forEach { cell ->
        val payload = storage.cellPayloadFile(outcome.manifest.jobId, cell.cellId)
        if (payload.exists()) {
          payload.copyTo(File(payloadsDir, "${cell.benchmarkId}_${cell.cellId}.json"), overwrite = true)
        }
      }
      "EXPORTED job=${outcome.manifest.jobId} path=${target.absolutePath} bytes=${target.length()}"
    }

  /**
   * Run one act that produces a single line, reporting a throw as a failure.
   *
   * Where an act that succeeds gets its line, and the exit code that goes with it. Every act here touches the filesystem, where a permission or a
   * race is always reachable, so the failure arm is the normal path rather than a formality.
   */
  private inline fun attempt(verb: String, act: () -> String): Report =
    runCatching(act)
      .fold(
        onSuccess = { Report(listOf(it), HeadlessRunActivity.EXIT_OK) },
        onFailure = { Report(listOf("FAILED $verb: ${reason(it)}"), HeadlessRunActivity.EXIT_FAILURE) },
      )

  /** A throwable as one line: its message, or its type when it carries none, which is the shape every other refusal on this surface already has. */
  internal fun reason(error: Throwable): String = error.message ?: error.javaClass.simpleName

  private fun label(model: ModelFile): String = model.hfRepo?.let { "$it/${model.name}" } ?: model.name
}
