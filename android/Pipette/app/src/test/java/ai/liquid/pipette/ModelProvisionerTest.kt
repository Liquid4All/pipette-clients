package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [ModelProvisioner]'s four arms, plus the two ways a wait ends without a file.
 *
 * Every case is driven through the persisted record state rather than through a callback, because that is what the real thing reads: the provisioner
 * deliberately does not register `DownloadRegistry` callbacks (it would steal the Models screen's only slot), so a test that fired callbacks would be
 * exercising a mechanism production does not use.
 */
class ModelProvisionerTest {
  private val repo = HfRepo.parseSlug("LiquidAI/LFM2-1.2B-GGUF")
  private val weights = GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf")
  private val projector = GgufFilename.parse("mmproj-lfm2-f16.gguf")
  private val textRef = Model.GgufText(HfGgufText(repo, weights))
  private val visionRef = Model.GgufVision(HfGgufVision(repo, weights, projector))
  private val weightsKey = "LiquidAI/LFM2-1.2B-GGUF/lfm2-1.2b-q4_k_m.gguf"
  private val projectorKey = "LiquidAI/LFM2-1.2B-GGUF/mmproj-lfm2-f16.gguf"

  private class Downloads : ModelProvisioner.Downloads {
    val resumed = mutableListOf<String>()
    val enqueued = mutableListOf<String>()

    override fun resume(key: String) {
      resumed += key
    }

    override fun enqueue(url: String, repo: String?) {
      enqueued += url
    }
  }

  private fun record(key: String, state: String) =
    DownloadStore.Record(
      key = key,
      filename = key.substringAfterLast('/'),
      urlString = "https://example.invalid/$key",
      repo = key.substringBeforeLast('/'),
      familyId = null,
      displayName = null,
      destPath = "/models/$key",
      partialPath = "/models/$key.part",
      state = state,
    )

  /**
   * Builds a provisioner whose world changes as the poll advances: [onDisk] is consulted fresh each call, so a test can make the file appear on the
   * Nth look the way a finishing transfer does.
   */
  private fun provisioner(
    downloads: Downloads,
    onDisk: () -> Set<String>,
    records: () -> List<DownloadStore.Record>,
    isOnline: () -> Boolean = { true },
    deadlineMillis: Long = 10_000L,
  ): ModelProvisioner {
    var clock = 0L
    return ModelProvisioner(
      resolvePath = { key -> if (key in onDisk()) "/models/$key" else null },
      downloads = downloads,
      records = records,
      isOnline = isOnline,
      deadlineMillis = deadlineMillis,
      pollMillis = 1_000L,
      // No real sleeping: the poll advances the clock the provisioner reads, so a deadline case finishes instantly instead of in ten seconds.
      sleep = { clock += it },
      now = { clock },
    )
  }

  @Test
  fun `a model already on disk is returned without touching the download layer`() {
    val downloads = Downloads()
    val result = provisioner(downloads, onDisk = { setOf(weightsKey) }, records = { emptyList() }).ensureWeights(textRef)

    assertEquals("/models/$weightsKey", result.getOrNull())
    assertTrue("nothing to fetch: ${downloads.enqueued}", downloads.enqueued.isEmpty())
    assertTrue(downloads.resumed.isEmpty())
  }

  @Test
  fun `an absent model is enqueued and waited for`() {
    val downloads = Downloads()
    var looks = 0
    // Absent on the first two looks, present on the third: the transfer finishing while the provisioner polls.
    val onDisk = { if (looks++ >= 2) setOf(weightsKey) else emptySet() }
    // No row until the enqueue, then a running one, which is what `enqueueDownload` does: it persists the record before the worker starts. Without
    // modelling that, the first poll would see no record and no file and correctly call it a transfer that ended without producing one.
    val records = { if (downloads.enqueued.isEmpty()) emptyList() else listOf(record(weightsKey, DownloadWorker.STATE_RUNNING)) }
    val result = provisioner(downloads, onDisk = onDisk, records = records).ensureWeights(textRef)

    assertEquals("/models/$weightsKey", result.getOrNull())
    assertEquals(1, downloads.enqueued.size)
    // Through the coordinator's parser, so the repo and filename land in the resolve URL.
    assertTrue(downloads.enqueued.single(), downloads.enqueued.single().contains("LiquidAI/LFM2-1.2B-GGUF/resolve/main/"))
  }

  /** The PIP-490 case iOS had to fix separately: a pause left by a process death is not terminal, and nothing else on a headless device clears it. */
  @Test
  fun `a paused download is resumed rather than reported as failed`() {
    val downloads = Downloads()
    var looks = 0
    val onDisk = { if (looks++ >= 1) setOf(weightsKey) else emptySet() }
    val result = provisioner(downloads, onDisk = onDisk, records = { listOf(record(weightsKey, DownloadWorker.STATE_PAUSED)) }).ensureWeights(textRef)

    assertTrue("${result.exceptionOrNull()}", result.isSuccess)
    assertEquals(listOf(weightsKey), downloads.resumed)
    assertTrue("a paused row must be resumed, not re-enqueued", downloads.enqueued.isEmpty())
  }

  @Test
  fun `a download already in flight is waited on, not re-enqueued`() {
    val downloads = Downloads()
    var looks = 0
    val onDisk = { if (looks++ >= 1) setOf(weightsKey) else emptySet() }
    val result =
      provisioner(downloads, onDisk = onDisk, records = { listOf(record(weightsKey, DownloadWorker.STATE_RUNNING)) }).ensureWeights(textRef)

    assertTrue(result.isSuccess)
    assertTrue("enqueueDownload rejects a duplicate, so this must not be attempted", downloads.enqueued.isEmpty())
    assertTrue(downloads.resumed.isEmpty())
  }

  @Test
  fun `a failed transfer reports the failure`() {
    val downloads = Downloads()
    var polls = 0
    // Resumed once (the stale-row retry), then stays failed: a genuine repeat failure.
    val records = {
      polls++
      listOf(record(weightsKey, DownloadWorker.STATE_FAILED))
    }
    val result = provisioner(downloads, onDisk = { emptySet() }, records = records).ensureWeights(textRef)

    assertTrue(result.isFailure)
    assertTrue("${result.exceptionOrNull()?.message}", result.exceptionOrNull()?.message?.contains("failed") == true)
  }

  /** A wait on a network the device does not have has no end, so it reports at once instead of holding the run to the deadline. */
  @Test
  fun `waiting for a network while offline reports immediately`() {
    val downloads = Downloads()
    val result =
      provisioner(
          downloads,
          onDisk = { emptySet() },
          records = { listOf(record(weightsKey, DownloadWorker.STATE_WAITING_NETWORK)) },
          isOnline = { false },
          deadlineMillis = 1_000_000L,
        )
        .ensureWeights(textRef)

    assertTrue(result.isFailure)
    assertTrue("${result.exceptionOrNull()?.message}", result.exceptionOrNull()?.message?.contains("network") == true)
  }

  @Test
  fun `a transfer that never progresses gives up at the deadline`() {
    val downloads = Downloads()
    val result =
      provisioner(downloads, onDisk = { emptySet() }, records = { listOf(record(weightsKey, DownloadWorker.STATE_RUNNING)) }, deadlineMillis = 5_000L)
        .ensureWeights(textRef)

    assertTrue(result.isFailure)
    assertTrue("${result.exceptionOrNull()?.message}", result.exceptionOrNull()?.message?.contains("did not finish") == true)
  }

  @Test
  fun `a text model has no projector to ensure`() {
    assertNull(provisioner(Downloads(), onDisk = { emptySet() }, records = { emptyList() }).ensureMmproj(textRef))
  }

  /** Half a vision pair present: only the missing half is fetched, so recovering a projector does not re-transfer gigabytes of weights. */
  @Test
  fun `a vision pair fetches only the half that is missing`() {
    val downloads = Downloads()
    var looks = 0
    val onDisk = { if (looks++ >= 1) setOf(weightsKey, projectorKey) else setOf(weightsKey) }
    val p = provisioner(downloads, onDisk = onDisk, records = { listOf(record(projectorKey, DownloadWorker.STATE_RUNNING)) })

    assertEquals("/models/$projectorKey", p.ensureMmproj(visionRef)?.getOrNull())
    assertEquals("/models/$weightsKey", p.ensureWeights(visionRef).getOrNull())
    assertTrue("the projector was already in flight and the weights were present", downloads.enqueued.isEmpty())
  }

  @Test
  fun `status is reported while a transfer is waited on`() {
    val seen = mutableListOf<String>()
    var looks = 0
    val onDisk = { if (looks++ >= 2) setOf(weightsKey) else emptySet() }
    provisioner(Downloads(), onDisk = onDisk, records = { listOf(record(weightsKey, DownloadWorker.STATE_RUNNING)) }).ensureWeights(textRef) {
      seen += it
    }

    assertTrue("$seen", seen.any { it.contains("lfm2-1.2b-q4_k_m.gguf") })
  }
}
