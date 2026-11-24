package ai.liquid.pipette

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * [DownloadStore.Record] keeps the stored key `familyId` for [DownloadStore.Record.modelId], so saved records still load. Robolectric supplies
 * [JSONObject].
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class DownloadStoreRecordTest {
  @Test
  fun `a record stores its model id under familyId and reads it back`() {
    val saved =
      JSONObject()
        .put("key", "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf")
        .put("filename", "LFM2.5-350M-Q4_0.gguf")
        .put("urlString", "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_0.gguf")
        .put("familyId", "lfm2.5-350m")
        .put("destPath", "/models/LFM2.5-350M-Q4_0.gguf")
        .put("partialPath", "/models/LFM2.5-350M-Q4_0.gguf.part")

    val record = DownloadStore.Record.fromJson(saved)

    assertEquals("lfm2.5-350m", record.modelId)
    assertEquals("lfm2.5-350m", record.toJson().getString("familyId"))
  }
}
