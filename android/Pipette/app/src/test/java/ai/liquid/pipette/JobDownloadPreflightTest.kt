package ai.liquid.pipette

import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_ID
import ai.liquid.pipette.fakes.JobRunnerFixtures.DECODE_TYPE
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class JobDownloadPreflightTest {
  @get:Rule val temp = TemporaryFolder()

  private fun model(name: String, bytes: Long, installed: Boolean = false): ModelFile {
    val repo = HfRepo.parseSlug("LiquidAI/Test")
    val path = LocalStorage.modelRelativePath(repo.reference, name)
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse(name)))
    return ModelFile(name, path, bytes, repo.toString(), modelRef = ref, onDisk = installed)
  }

  @Test
  fun `only missing files used by the job count once`() {
    val missing = model("missing.gguf", 0L)
    val installed = model("installed.gguf", 80L, installed = true)
    val unused = model("unused.gguf", 80L)
    val cells =
      listOf(missing, missing, installed).map {
        JobCell(benchmarkId = DECODE_ID, benchmarkType = DECODE_TYPE, modelPath = it.path, modelName = it.name, modelRef = it.modelRef)
      }

    assertEquals(listOf(missing), pendingJobDownloads(cells, listOf(missing, installed, unused, missing)))
  }

  @Test
  fun `the batch must fit even when its files fit individually`() {
    val root =
      object : File(temp.root.path) {
        override fun getUsableSpace(): Long = 100L
      }
    val known = model("known.gguf", 60L)
    val unknown = model("unknown.gguf", 0L)

    val failure =
      assertThrows(IllegalStateException::class.java) {
        checkJobDownloadSpace(root, listOf(known, unknown)) {
          assertEquals(unknown, it)
          60L
        }
      }
    assertTrue(failure.message.orEmpty().contains("Not enough storage"))

    val partial = File(root, "${unknown.storageKey}.part")
    partial.parentFile?.mkdirs()
    partial.writeBytes(ByteArray(20))
    checkJobDownloadSpace(root, listOf(known, unknown)) { 60L }
  }

  @Test
  fun `no size from the host blocks the batch`() {
    val failure = assertThrows(IllegalStateException::class.java) { checkJobDownloadSpace(temp.root, listOf(model("unknown.gguf", 0L))) { null } }
    assertTrue(failure.message.orEmpty().contains("unknown.gguf"))
  }
}
