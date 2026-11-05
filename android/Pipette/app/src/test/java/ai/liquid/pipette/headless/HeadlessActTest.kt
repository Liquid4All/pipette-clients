package ai.liquid.pipette.headless

import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.FakeAead
import ai.liquid.pipette.IdentityBundle
import ai.liquid.pipette.LocalStorage
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.Secrets
import androidx.test.core.app.ApplicationProvider
import com.google.crypto.tink.Aead
import java.io.File
import java.security.GeneralSecurityException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * What [HeadlessAct] actually does to disk, and what it says about it.
 *
 * [HeadlessMutateTest] pins which thing a verb targets; this pins what happens when the delete is carried out, which is the half that was previously
 * reachable only from a device. Both device passes on this surface found something the pure tests could not, and the reason is structural: a delete's
 * failure modes are all quiet. `deleteModel` discards `File.delete`'s Boolean, `deleteRecursively` reports a partial delete by returning false, and
 * `LocalStorage` drops that too, so nothing throws when a delete does not happen.
 *
 * Robolectric rather than a pure unit test because [LocalStorage] needs a `Context` and every assertion here is a filesystem question.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class HeadlessActTest {

  private val storage by lazy { LocalStorage(ApplicationProvider.getApplicationContext()) }

  /**
   * A real [Secrets] over a fake AEAD, because Robolectric has no working Keystore and `auth-import` writes through the sealing path.
   *
   * [FakeAead] models the one property this depends on: a write that lands can be read back. What it deliberately does not model is confidentiality,
   * so nothing here may assert that a stored value is unreadable.
   */
  private val secrets by lazy { Secrets(ApplicationProvider.getApplicationContext(), FakeAead()) }

  private fun perform(outcome: HeadlessMutate.Outcome, verb: String, externalFilesDir: File? = null) =
    HeadlessAct.perform(outcome, storage, secrets, verb, externalFilesDir)

  private fun onDiskModel(repo: String, name: String, bytes: Int = 4): ModelFile {
    val file = File(File(storage.modelsDir, repo).apply { mkdirs() }, name)
    file.writeBytes(ByteArray(bytes))
    return ModelFile(name = name, path = file.absolutePath, sizeBytes = bytes.toLong(), hfRepo = repo, onDisk = true)
  }

  /** The happy path, stated in full: the bytes go, and the line names what went. */
  @Test
  fun deletingModelsRemovesThemAndReportsTheirSummedSize() {
    val first = onDiskModel("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_0.gguf", bytes = 10)
    val second = onDiskModel("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_K_M.gguf", bytes = 32)

    val report = perform(HeadlessMutate.Outcome.DeleteModels(listOf(first, second)), "models-rm")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(
      listOf(
        "DELETED models count=2 bytes=42 files=LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf," + "LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_K_M.gguf"
      ),
      report.lines,
    )
    assertFalse(File(first.path).exists())
    assertFalse(File(second.path).exists())
  }

  /**
   * A file that cannot be deleted does not take the successful ones down with it, and the failure says why.
   *
   * `deleteModel` throws on a path outside the model root, which is the reachable version of "one of these fails". With one `runCatching` around the
   * loop, the file that was already gone would be reported as a failure too, and "FAILED" that means "some of your models are deleted" is the worst
   * reading available on a surface nobody is watching. The throwable's own message is carried because this line is the only record the caller gets.
   */
  @Test
  fun aFileThatCannotBeDeletedIsReportedWithoutDisowningTheOnesThatWere() {
    val real = onDiskModel("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_0.gguf", bytes = 7)
    val strayFile = File.createTempFile("outside-the-model-root", ".gguf")
    val outsideTheRoot = ModelFile(name = strayFile.name, path = strayFile.absolutePath, sizeBytes = 1L, hfRepo = null, onDisk = true)

    val report = perform(HeadlessMutate.Outcome.DeleteModels(listOf(real, outsideTheRoot)), "models-rm")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertEquals("DELETED models count=1 bytes=7 files=LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf", report.lines.first())
    assertTrue(report.lines.last(), report.lines.last().startsWith("FAILED models-rm: 1 of 2 not deleted: ${strayFile.name} ("))
    assertTrue("carries the cause, got: ${report.lines.last()}", report.lines.last().contains("outside app model storage"))
    assertFalse("the file it could delete is gone", File(real.path).exists())
    assertTrue("the file it refused to touch is untouched", strayFile.exists())
    strayFile.delete()
  }

  /**
   * A file that was already gone is not counted as bytes reclaimed.
   *
   * `deleteModel` no-ops on a file that is not there and returns normally, so checking only after the delete would call this a success. It is
   * reachable rather than theoretical: `ModelStore` caches its scan for the life of the process, so a listing can name a file that something else has
   * since removed, and reporting `bytes=` for it is the exact lie the per-file verification exists to prevent.
   */
  @Test
  fun aFileThatWasAlreadyGoneIsNotReportedAsReclaimed() {
    val stale = onDiskModel("LiquidAI/LFM2.5-230M-GGUF", "LFM2.5-230M-Q4_0.gguf", bytes = 9)
    File(stale.path).delete()

    val report = perform(HeadlessMutate.Outcome.DeleteModels(listOf(stale)), "models-delete")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertEquals(listOf("FAILED models-delete: 1 of 1 not deleted: LiquidAI/LFM2.5-230M-GGUF/LFM2.5-230M-Q4_0.gguf (already absent)"), report.lines)
  }

  /**
   * One file named twice is deleted once and counted once.
   *
   * A vision coordinate names weights and projector separately, and nothing stops a caller writing the same file for both, which
   * `ModelCatalog.resolveCoordinate` resolves to the same file twice. Without the dedup the reclaimed `bytes=` would be double what the disk gave
   * back, and the second delete would then report the file it had just removed as already absent.
   */
  @Test
  fun theSameFileNamedTwiceIsCountedOnce() {
    val model = onDiskModel("LiquidAI/LFM2.5-VL-GGUF", "LFM2.5-VL-Q4_0.gguf", bytes = 16)

    val report = perform(HeadlessMutate.Outcome.DeleteModels(listOf(model, model.copy())), "models-delete")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(listOf("DELETED models count=1 bytes=16 files=LiquidAI/LFM2.5-VL-GGUF/LFM2.5-VL-Q4_0.gguf"), report.lines)
  }

  /** Deleting a job takes the directory with it, and the line is not printed unless it did. */
  @Test
  fun deletingAJobRemovesItsDirectory() {
    val dir = storage.jobDir("job-1").apply { mkdirs() }
    File(dir, "manifest.json").writeText("{}")

    val report = perform(HeadlessMutate.Outcome.DeleteJob("job-1"), "job-rm")

    assertEquals(listOf("DELETED job=job-1"), report.lines)
    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertFalse(dir.exists())
  }

  /**
   * A job whose directory is not there does not report DELETED.
   *
   * Resolution found a manifest, so something named this job, but `loadAllJobManifests` reads the id out of the manifest's contents rather than from
   * the directory name. If the two disagree, `deleteJob` removes nothing and a check that only looked afterwards would pass on an empty result.
   */
  @Test
  fun aJobWithNoDirectoryIsAFailureRatherThanASilentSuccess() {
    val report = perform(HeadlessMutate.Outcome.DeleteJob("never-written"), "job-rm")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertTrue(report.lines.single(), report.lines.single().startsWith("FAILED job-rm: no job directory at "))
  }

  /**
   * Deleting a result clears every artifact the cell has, not only the payload.
   *
   * The payload is what makes it stop being a result (see `HeadlessFacts.isReportable`), which is exactly why the others have to go with it: once the
   * cell stops listing, nothing on this surface can reach a submission record left beside it.
   */
  @Test
  fun deletingAResultClearsEveryArtifact() {
    storage.ensureCellArtifactsDir("job-1", "cell-0")
    storage.cellPayloadFile("job-1", "cell-0").writeText("{}")
    storage.cellSubmissionFile("job-1", "cell-0").writeText("{}")

    val report = perform(HeadlessMutate.Outcome.DeleteResult("job-1", "cell-0"), "results-delete")

    assertEquals(listOf("DELETED result cell=cell-0 job=job-1"), report.lines)
    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(null, storage.submittablePayloadDir("job-1", "cell-0"))
    assertFalse("the submission record goes too", storage.cellSubmissionFile("job-1", "cell-0").exists())
  }

  /**
   * A metrics file goes with the rest.
   *
   * Nothing in this client writes `metrics.json` today, which is exactly why it is worth pinning: a file with no writer is the one a future change to
   * `clearCellArtifacts` would forget, and the check that would have caught it is the one this test executes.
   */
  @Test
  fun deletingAResultClearsTheMetricsFileToo() {
    storage.ensureCellArtifactsDir("job-3", "cell-0")
    storage.cellPayloadFile("job-3", "cell-0").writeText("{}")
    storage.cellMetricsFile("job-3", "cell-0").writeText("{}")

    val report = perform(HeadlessMutate.Outcome.DeleteResult("job-3", "cell-0"), "results-delete")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertFalse(storage.cellMetricsFile("job-3", "cell-0").exists())
  }

  /** A cell that only ever had a payload is the normal case, and the all-artifacts check must not fail on the files it never had. */
  @Test
  fun deletingAResultWithNoSubmissionRecordStillSucceeds() {
    storage.ensureCellArtifactsDir("job-2", "cell-0")
    storage.cellPayloadFile("job-2", "cell-0").writeText("{}")

    val report = perform(HeadlessMutate.Outcome.DeleteResult("job-2", "cell-0"), "results-delete")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(listOf("DELETED result cell=cell-0 job=job-2"), report.lines)
  }

  /**
   * With external storage unmounted there is nowhere to export to, and that is said rather than worked around.
   *
   * `getExternalFilesDir` returns null there, and `File(null, name)` is a path relative to the process working directory rather than an error, so
   * without the guard the CSV would be written somewhere the caller cannot reach and reported as a success.
   */
  @Test
  fun exportingWithNoExternalStorageFailsWithAReason() {
    val manifest = HeadlessTestFixtures.manifest("job-1", cells = listOf(CellRunStatus.COMPLETED))

    val report = perform(HeadlessMutate.Outcome.ExportJob(manifest), "job-export")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertEquals(listOf("FAILED job-export: external storage is unavailable"), report.lines)
  }

  /** Export writes the CSV where it says it did, and the byte count is the file's. */
  @Test
  fun exportingWritesTheCsvAndNamesThePath() {
    val manifest = HeadlessTestFixtures.manifest("job-1", cells = listOf(CellRunStatus.COMPLETED))
    val exportsRoot = File(storage.modelsDir.parentFile, "external").apply { mkdirs() }

    val report = perform(HeadlessMutate.Outcome.ExportJob(manifest), "job-export", exportsRoot)

    val line = report.lines.single()
    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertTrue(line, line.startsWith("EXPORTED job=job-1 path=${exportsRoot.absolutePath}/exports/"))
    val written = File(line.substringAfter("path=").substringBefore(" bytes="))
    assertTrue("the path it printed is the file it wrote", written.exists())
    assertEquals("bytes= is the file's own length", "bytes=${written.length()}", line.substringAfterLast(" "))
  }

  /** A refusal is carried through unchanged: [HeadlessMutate] words it, and nothing here re-decides it. */
  @Test
  fun aRefusalIsPassedThroughAsIs() {
    val report = perform(HeadlessMutate.Outcome.Refused("no job with id 'nope'"), "job-rm")

    assertEquals(listOf("REFUSED no job with id 'nope'"), report.lines)
    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
  }

  private fun bundle(privateKeyHex: PrivateKeyHex = HeadlessTestFixtures.privateKeyHex(), userToken: String? = "hf_user", modelTokens: Int = 0) =
    IdentityBundle(
      privateKeyHex = privateKeyHex,
      registration = HeadlessTestFixtures.registration(),
      userToken = userToken,
      modelTokenCount = modelTokens,
    )

  /** The whole restore: the key can sign, the registration is on disk, and the token is readable. */
  @Test
  fun importingWritesTheKeyTheRegistrationAndTheToken() {
    val key = HeadlessTestFixtures.privateKeyHex()

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(key), observedKey = null), "auth-import")

    assertEquals(report.lines.toString(), HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(
      listOf(
        "IMPORTED identity client_id=ev1_a3f8c2d1 server_url=https://pipette.liquid.ai " +
          "signing_key=ok registration=ok replaced_key=false hf_tokens=1"
      ),
      report.lines,
    )
    assertEquals("the key is the promoted one, so signing reaches it", key.value, secrets.exportPrivateKeyHex())
    assertEquals(HeadlessTestFixtures.registration(), storage.loadRegistration())
    assertEquals("hf_user", secrets.loadHfToken())
  }

  /**
   * A refused import writes nothing, which is the property the whole decide-then-act split exists for.
   *
   * Asserted against the store rather than against the report, because a refusal that still wrote would report itself correctly and be wrong.
   */
  @Test
  fun aRefusedImportLeavesTheDeviceAsItWas() {
    val report = perform(HeadlessMutate.Outcome.Refused("refusing over the signing key this device already holds"), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertEquals(null, secrets.exportPrivateKeyHex())
    assertEquals(null, storage.loadRegistration())
    assertEquals(null, secrets.loadHfToken())
  }

  /**
   * A rerun of the same bundle after a partial import finishes the restore.
   *
   * The reachable half-state: the key landed and the registration did not, which is what an interrupted process or a failed write leaves. The rerun
   * has to be idempotent over the key it already wrote, or the operator is left with an identity they cannot complete and cannot clear.
   */
  @Test
  fun rerunningTheSameBundleCompletesAPartialImport() {
    val key = HeadlessTestFixtures.privateKeyHex()
    assertEquals(Secrets.KeyReplacement.REPLACED, secrets.replacePrivateKeyHex(key.value, expected = null))
    assertEquals("the half-state this test is about", null, storage.loadRegistration())

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(key), observedKey = key), "auth-import")

    assertEquals(report.lines.toString(), HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(key.value, secrets.exportPrivateKeyHex())
    assertEquals(HeadlessTestFixtures.registration(), storage.loadRegistration())
  }

  /** A bundle carrying no token restores the identity and reports no token, rather than reporting one that failed. */
  @Test
  fun importingABundleWithNoTokenReportsNoneRatherThanAFailure() {
    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(userToken = null), observedKey = null), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertTrue(report.lines.single(), report.lines.single().endsWith("hf_tokens=0"))
    assertEquals(null, secrets.loadHfToken())
  }

  /**
   * The token the device already holds survives an import that carries none.
   *
   * The Hugging Face token authenticates against huggingface.co rather than against the management server, so it belongs to the operator and not to
   * the `client_id` being replaced. iOS keeps its own user token through both an import and an `auth reset force=1`; what it clears on a replacing
   * import is the per-model tokens, which this client has no counterpart for. See [HeadlessAct.importIdentity] for the accounts that separate them.
   */
  @Test
  fun anImportCarryingNoTokenLeavesTheStoredOneAlone() {
    val replaced = HeadlessTestFixtures.privateKeyHex(fill = 77)
    assertEquals(Secrets.KeyReplacement.REPLACED, secrets.replacePrivateKeyHex(replaced.value, expected = null))
    assertTrue(secrets.saveHfToken("hf_kept"))

    perform(HeadlessMutate.Outcome.ImportIdentity(bundle(userToken = null), observedKey = replaced), "auth-import")

    assertEquals("hf_kept", secrets.loadHfToken())
  }

  /**
   * A token that did not store fails the import, though the identity itself lands and is reported as landed.
   *
   * iOS returns `missing == 0` for the same reason: a token that is not there means a gated download 401s later, and an exit code is what an
   * unattended caller reads. The identity is the half worth having, so it stays.
   */
  @Test
  fun aTokenThatCannotBeStoredFailsTheImportButKeepsTheIdentity() {
    val tokenHostile = Secrets(ApplicationProvider.getApplicationContext(), TokenHostileAead())
    val outcome = HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = null)

    val report = HeadlessAct.perform(outcome, storage, tokenHostile, "auth-import", null)

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertTrue(report.lines.single(), report.lines.single().endsWith("hf_tokens=0 hf_tokens_failed=1"))
    assertEquals("the identity is the half worth having", HeadlessTestFixtures.registration(), storage.loadRegistration())
    assertEquals(HeadlessTestFixtures.privateKeyHex().value, tokenHostile.exportPrivateKeyHex())
  }

  /**
   * A Keystore that cannot seal at all fails before the registration is touched.
   *
   * The write order is what this asserts. A registration on disk with no key reports the device as registered while every signed request fails, and
   * that is the worse of the two half-states to be interrupted into, so the key goes first.
   */
  @Test
  fun aKeyThatCannotBeSealedFailsBeforeTheRegistrationIsWritten() {
    val sealFails = Secrets(ApplicationProvider.getApplicationContext(), FakeAead(failEncrypt = true))
    val outcome = HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = null)

    val report = HeadlessAct.perform(outcome, storage, sealFails, "auth-import", null)

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertTrue(report.lines.single(), report.lines.single().startsWith("FAILED auth-import: could not stage the signing key"))
    assertEquals(null, storage.loadRegistration())
  }

  /**
   * A Keystore that seals every slot except the Hugging Face token's.
   *
   * The slot name is the AEAD's associated data, which is what makes a per-slot failure expressible at all. It is the only arrangement in which the
   * import's own token accounting is reachable: [FakeAead] with `failEncrypt` fails the key first and never gets there.
   */
  private class TokenHostileAead : Aead {
    private val delegate = FakeAead()

    override fun encrypt(plaintext: ByteArray, associatedData: ByteArray): ByteArray {
      if (String(associatedData, Charsets.UTF_8) == "hf_token") throw GeneralSecurityException("hf_token slot unavailable")
      return delegate.encrypt(plaintext, associatedData)
    }

    override fun decrypt(ciphertext: ByteArray, associatedData: ByteArray): ByteArray = delegate.decrypt(ciphertext, associatedData)
  }

  /** Per-model tokens only an iOS bundle can carry are reported as dropped, so a 401 on the restored device has a line explaining it. */
  @Test
  fun perModelTokensThisClientCannotHoldAreReportedAsDiscarded() {
    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(modelTokens = 2), observedKey = null), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertTrue(report.lines.single(), report.lines.single().endsWith("hf_tokens=1 model_tokens_discarded=2"))
  }

  /** A forced import says so, because replacing another identity is the one thing this verb does that costs data it never touched. */
  @Test
  fun aForcedImportRecordsThatItReplacedAnotherKey() {
    val replaced = HeadlessTestFixtures.privateKeyHex(fill = 99)
    assertEquals(Secrets.KeyReplacement.REPLACED, secrets.replacePrivateKeyHex(replaced.value, expected = null))

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = replaced), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_OK, report.exit)
    assertTrue(report.lines.single(), report.lines.single().contains(" replaced_key=true "))
    assertEquals(HeadlessTestFixtures.privateKeyHex().value, secrets.exportPrivateKeyHex())
  }

  /**
   * An import that a registration overtook writes nothing and says which one it lost.
   *
   * `HeadlessFacts` reads the promoted key before the decision, and `RegistrationService.register` promotes on its own dispatcher, so the two can
   * interleave. The registration's key survives, and the registration keeps the client id the server issued for it.
   */
  @Test
  fun anImportThatARegistrationOvertookWritesNothing() {
    val registered = HeadlessTestFixtures.privateKeyHex(fill = 71)
    assertEquals(Secrets.KeyReplacement.REPLACED, secrets.replacePrivateKeyHex(registered.value, expected = null))

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = null), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertTrue(report.lines.single(), report.lines.single().startsWith("FAILED auth-import: this device replaced its signing key"))
    assertEquals("the registration's key survives", registered.value, secrets.exportPrivateKeyHex())
    assertEquals(null, storage.loadRegistration())
  }

  /**
   * An import running against a registration that is waiting on the server writes nothing.
   *
   * The staged key is one the server may already hold. Promoting over it would leave that client record unsignable, and the registration would then
   * fail on a pending slot the import emptied.
   */
  @Test
  fun anImportWritesNothingWhileARegistrationWaitsOnTheServer() {
    val registering = secrets.generatePendingSigningKeyPair()

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = null), "auth-import")

    assertEquals(HeadlessRunActivity.EXIT_FAILURE, report.exit)
    assertTrue(report.lines.single(), report.lines.single().startsWith("FAILED auth-import: a registration is claiming this device"))
    // The device holds no registration here, so only the offered identity can be named. `pipette-auth` keeps this line.
    assertTrue("names the offered identity, got: ${report.lines.single()}", report.lines.single().endsWith(" bundle_client_id=ev1_a3f8c2d1"))
    assertEquals(null, storage.loadRegistration())
    assertTrue("the registration can still complete", secrets.promotePendingPrivateKey())
    assertEquals(registering, secrets.exportPrivateKeyHex()?.let { IdentityBundle.publicKeyHexOf(PrivateKeyHex(it)) })
  }

  /**
   * `force=1` imports over a staged key, which is the only way past a registration that died before it promoted.
   *
   * A killed registration leaves its staged key behind, and nothing on this surface clears that slot, so the refusal above would otherwise stand for
   * the life of the install. The promoted-key comparison is untouched by `force=1` reaching this far: that one guards an identity that exists.
   */
  @Test
  fun aForcedImportProceedsOverAStrandedStagedKey() {
    secrets.generatePendingSigningKeyPair()

    val report = perform(HeadlessMutate.Outcome.ImportIdentity(bundle(), observedKey = null, forced = true), "auth-import")

    assertEquals(report.lines.toString(), HeadlessRunActivity.EXIT_OK, report.exit)
    assertEquals(HeadlessTestFixtures.privateKeyHex().value, secrets.exportPrivateKeyHex())
    assertEquals(HeadlessTestFixtures.registration(), storage.loadRegistration())
  }
}
