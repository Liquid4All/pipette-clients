package ai.liquid.pipette

import androidx.test.core.app.ApplicationProvider
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Drives the real [LocalStorage.writePayload] on Robolectric (a genuine Android [android.content.Context] against a sandboxed app filesystem +
 * shadowed system services) and reads the written `payload.json` back: the end-to-end check that PIP-387's `model_descriptor` / `runtime_descriptor`
 * / `runtime_flags` and PIP-432's `benchmark_flags` land in the submission payload and `mmproj_quant` is gone. [SubmissionRefTest] pins the spec
 * serialization itself; this pins that `writePayload` wires it in.
 */
// A stock Application, not the manifest PipetteApp: LocalStorage needs only a Context, while PipetteApp.onCreate wires up WorkManager/Clerk/etc.
// that have no place in this unit test.
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class LocalStoragePayloadTest {
  private val storage by lazy { LocalStorage(ApplicationProvider.getApplicationContext()) }

  private fun writeAndRead(
    modelName: String,
    modelPath: String,
    mmprojPath: String?,
    benchmarkType: BenchmarkType = BenchmarkType.DECODE_THROUGHPUT,
    readinessPolicy: ReadinessPolicy = ReadinessPolicy(maxWaitSecs = 180L, skipThermal = false),
  ): JSONObject {
    val jobId = "job-1"
    val cellId = "cell-1"
    storage.writePayload(
      resultJson = JSONObject().put("decode_time_ms", 250.0).toString(),
      cellId = cellId,
      jobId = jobId,
      modelName = modelName,
      modelPath = modelPath,
      mmprojPath = mmprojPath,
      nGpuLayers = 99,
      contextSize = 4096,
      runtimeVersion = "b8683",
      runtimeCpuBackend = "dotprod,neon",
      benchmarkType = benchmarkType,
      readinessPolicy = readinessPolicy,
    )
    return JSONObject(File(storage.ensureCellArtifactsDir(jobId, cellId), "payload.json").readText())
  }

  @Test
  fun textModelPayloadCarriesRefsAndDropsMmprojQuant() {
    val payload = writeAndRead("LiquidAI/LFM2.5-350M-GGUF", "/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf", null)

    // Scalar grouping fields stay.
    assertEquals("LiquidAI/LFM2.5-350M-GGUF", payload.getString("model_name"))
    assertEquals("Q4_0", payload.getString("model_quant"))
    assertEquals("b8683", payload.getString("runtime_version"))
    assertEquals("llamacpp_apk_pipette", payload.getString("runtime_name"))

    // New lossless descriptors, as JSON strings.
    assertEquals(
      """{"org":"LiquidAI","path":"LFM2.5-350M-Q4_0.gguf","repo_name":"LFM2.5-350M-GGUF","source":"huggingface","type":"gguf_text"}""",
      payload.getString("model_descriptor"),
    )
    assertEquals(
      """{"flavor":"android-arm64-v8","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b8683","type":"llamacpp_apk_pipette"}""",
      payload.getString("runtime_descriptor"),
    )
    assertEquals("-ngl 99 -c 4096", payload.getString("runtime_flags"))

    // What the harness ran under, as distinct from the load knobs above: the readiness policy that gated this cell. The only record of it the server
    // ever sees, since readiness is decided entirely client-side.
    assertEquals("""{"readiness":{"max_wait_secs":180,"skip_thermal":false}}""", payload.getString("benchmark_flags"))

    // The app build that produced the run — distinct from runtime_version
    // above, which names the engine it drove.
    assertEquals(BuildConfig.VERSION_NAME, payload.getString("client_version"))

    // Retired server-side.
    assertFalse("mmproj_quant must not be emitted", payload.has("mmproj_quant"))

    // The result JSON and real device metadata still ride along.
    assertEquals(250.0, payload.getDouble("decode_time_ms"), 0.0)
    assertEquals("Android", payload.getString("device_os_name"))
    assertTrue(payload.has("device_name"))
  }

  @Test
  fun visionModelPayloadEncodesMmprojInsideModelRef() {
    val payload =
      writeAndRead(
        "LiquidAI/LFM2.5-VL-450M-GGUF",
        "/models/LiquidAI/LFM2.5-VL-450M-GGUF/LFM2.5-VL-450M-Q4_0.gguf",
        "/models/LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-f16.gguf",
      )

    assertEquals(
      """{"mmproj":"mmproj-f16.gguf","model":"LFM2.5-VL-450M-Q4_0.gguf",""" +
        """"org":"LiquidAI","repo_name":"LFM2.5-VL-450M-GGUF","source":"huggingface","type":"gguf_vision"}""",
      payload.getString("model_descriptor"),
    )
    assertFalse("mmproj_quant must not be emitted", payload.has("mmproj_quant"))
  }

  @Test
  fun importedFileWithoutHfSlugElidesModelRef() {
    val payload = writeAndRead("local-model.gguf", "/models/local-model.gguf", null)

    assertFalse("model_descriptor elided when no HF coordinate", payload.has("model_descriptor"))
    // runtime_descriptor is always known on-device, so it's still present.
    assertTrue(payload.has("runtime_descriptor"))
  }

  @Test
  fun payloadElidesModelFlagsWhenTheModelCarriesNone() {
    // Absent, not `enable_thinking=false`: nothing on Android builds a model with a flag today, so there is genuinely nothing to report. Pins that
    // the payload elides rather than emitting a default, which would be a value the warehouse could group on.
    val payload = writeAndRead("LiquidAI/LFM2.5-350M-GGUF", "/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf", null)

    assertFalse("model_flags must be elided when the model carries none", payload.has("model_flags"))
  }

  @Test
  fun waivedThermalGateIsRecordedInBenchmarkFlags() {
    // The point of PIP-434 pairing with this field: a run with the thermal criterion waived is not
    // comparable to a gated one, so the submission has to say which it was. Without this the only
    // evidence would be a thermal reading outside the band, inferred from the very numbers a reader
    // is trying to interpret.
    val payload =
      writeAndRead(
        "LiquidAI/LFM2.5-350M-GGUF",
        "/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf",
        null,
        readinessPolicy = ReadinessPolicy(maxWaitSecs = 180L, skipThermal = true),
      )

    assertEquals("""{"readiness":{"max_wait_secs":180,"skip_thermal":true}}""", payload.getString("benchmark_flags"))
  }

  @Test
  fun maxMemoryCellElidesBenchmarkFlags() {
    // The flag schema has no variant for max_memory_usage, so the field is omitted rather than describing Android's between-cell gate in a shape a
    // consumer parsing this column with `BenchmarkFlags` can't read. [SubmissionRefTest] pins the rule; this pins that writePayload honors it.
    val payload =
      writeAndRead(
        "LiquidAI/LFM2.5-350M-GGUF",
        "/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf",
        null,
        benchmarkType = BenchmarkType.MAX_MEMORY_USAGE,
      )

    assertFalse("benchmark_flags must be elided for a cell the schema models none for", payload.has("benchmark_flags"))
  }
}
