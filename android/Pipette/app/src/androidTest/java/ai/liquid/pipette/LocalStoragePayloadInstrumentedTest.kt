package ai.liquid.pipette

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device counterpart to the Robolectric `LocalStoragePayloadTest`: drives the real [LocalStorage.writePayload] on the emulator/handset — real
 * [android.content.Context], real internal filesystem, real [DeviceInfo] values — and reads the written `payload.json` back. Confirms PIP-387's
 * `model_descriptor` / `runtime_descriptor` / `runtime_flags` and PIP-432's `benchmark_flags` land in the actual submission artifact and
 * `mmproj_quant` is gone, on a genuine Android runtime.
 */
@RunWith(AndroidJUnit4::class)
class LocalStoragePayloadInstrumentedTest {
  @Test
  fun writePayloadEmitsRefsOnDevice() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val storage = LocalStorage(context)
    val jobId = "pip387-instr-${System.nanoTime()}"
    val cellId = "cell-1"

    storage.writePayload(
      resultJson = JSONObject().put("decode_time_ms", 123.0).toString(),
      cellId = cellId,
      jobId = jobId,
      modelName = "LiquidAI/LFM2.5-350M-GGUF",
      modelPath = "/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf",
      mmprojPath = null,
      nGpuLayers = 99,
      contextSize = 4096,
      runtimeVersion = "b8683",
      runtimeCpuBackend = "dotprod,neon",
      benchmarkType = BenchmarkType.DECODE_THROUGHPUT,
      readinessPolicy = ReadinessPolicy(maxWaitSecs = 180L, skipThermal = false),
    )

    val payload = JSONObject(File(storage.ensureCellArtifactsDir(jobId, cellId), "payload.json").readText())

    assertEquals(
      """{"org":"LiquidAI","path":"LFM2.5-350M-Q4_0.gguf","repo_name":"LFM2.5-350M-GGUF","source":"huggingface","type":"gguf_text"}""",
      payload.getString("model_descriptor"),
    )
    assertEquals(
      """{"flavor":"android-arm64-v8","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b8683","type":"llamacpp_apk_pipette"}""",
      payload.getString("runtime_descriptor"),
    )
    assertEquals("-ngl 99 -c 4096", payload.getString("runtime_flags"))
    // `SubmissionRef.canonical` renders every scalar itself rather than through
    // JSONObject.toString(), so these bytes are platform-independent by design;
    // this asserts the field lands in the real artifact, not that the framework
    // formats it the same way.
    assertEquals("""{"readiness":{"max_wait_secs":180,"skip_thermal":false}}""", payload.getString("benchmark_flags"))
    assertFalse("mmproj_quant must not be emitted", payload.has("mmproj_quant"))

    // Real on-device metadata rode along (Robolectric can't vouch for these values).
    assertTrue("device_name should be populated on a real device", payload.getString("device_name").isNotBlank())
    assertEquals("Android", payload.getString("device_os_name"))

    // Tidy up the synthetic job artifacts.
    File(storage.ensureCellArtifactsDir(jobId, cellId), "payload.json").parentFile?.deleteRecursively()
  }
}
