package ai.liquid.pipette.onnx

import ai.liquid.pipette.EngineActor
import ai.liquid.pipette.HfOnnxGenai
import ai.liquid.pipette.HfRepo
import ai.liquid.pipette.HfRevision
import ai.liquid.pipette.HuggingFaceHttp
import ai.liquid.pipette.LlamacppLoadMode
import ai.liquid.pipette.Model
import ai.liquid.pipette.ModelStore
import ai.liquid.pipette.OnnxGenaiPackage
import ai.liquid.pipette.OnnxPackageFetcher
import ai.liquid.pipette.RepoSubpath
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The path a headless `model=onnx-genai://…` run takes, end to end on a device and against HuggingFace itself: [OnnxPackageFetcher] puts the package
 * in a models directory, [ModelStore] lists it, and [EngineActor] runs a cell on the path the store names.
 *
 * Off unless asked for, since it downloads the package: `-e onnxFetch true`. The catalog's Phi-3 row by default (2.73 GB); `-e onnxFetchRepo
 * org/name` and `-e onnxFetchPrefix a/b` name another, `-e onnxFetchRev <ref>` pins it (which also checks that the package on disk records that
 * snapshot and does not pass for another), and `-e onnxPrefillTokens` sets the cell's prompt. It fetches into its own directory under the app's
 * external files, and a second run finds the package there and fetches nothing.
 */
@RunWith(AndroidJUnit4::class)
class OnnxPackageFetchInstrumentedTest {
  private val tag = "OnnxGenaiInstrumented"
  private val arguments = InstrumentationRegistry.getArguments()

  @Test
  fun testAPackageIsFetchedListedAndRun() {
    assumeTrue("pass -e onnxFetch true to download the package", arguments.getString("onnxFetch") == "true")
    assumeTrue("this build has no ONNX Runtime GenAI engine", OnnxGenai.isBuilt)
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelsDir = File(context.getExternalFilesDir(null), "onnx-fetch-test-models")
    val repo = HfRepo.parseSlug(arguments.getString("onnxFetchRepo") ?: PHI3_REPO)
    val prefix = (arguments.getString("onnxFetchPrefix") ?: PHI3_PREFIX).takeIf { it.isNotEmpty() }?.let { RepoSubpath.parse(it) }
    val revision = arguments.getString("onnxFetchRev")?.let { HfRevision.parse(it) }
    val prefill = arguments.getString("onnxPrefillTokens")?.toInt() ?: PREFILL
    val ref = HfOnnxGenai(repo, prefix, revision)

    var lastStatus = ""
    val started = System.nanoTime()
    val path =
      OnnxPackageFetcher(modelsDir, HuggingFaceHttp { null })
        .ensure(ref) { status ->
          if (status != lastStatus) Log.i(tag, status)
          lastStatus = status
        }
        .getOrThrow()
    Log.i(tag, "fetched $path in ${(System.nanoTime() - started) / NANOS_PER_SECOND} s")

    assertEquals(File(modelsDir, ref.relativeDir).absolutePath, path)
    OnnxGenaiPackage.validate(File(path))
    assertTrue("nothing is left staged", File(path).parentFile?.list().orEmpty().none { it.endsWith(OnnxPackageFetcher.STAGING_SUFFIX) })

    if (revision != null) {
      // The package says which snapshot it is, and is not taken for another: a full commit that is not this one is refused without a download.
      val recorded = File(path, OnnxPackageFetcher.REVISION_FILE).readText()
      Log.i(tag, "recorded snapshot $recorded for $revision")
      assertTrue(recorded.matches(Regex("[0-9a-f]{40}")))
      val other = ref.copy(revision = HfRevision.parse(OTHER_COMMIT))
      val refusal = OnnxPackageFetcher(modelsDir, HuggingFaceHttp { null }).ensure(other) {}.exceptionOrNull()?.message.orEmpty()
      Log.i(tag, "another snapshot: $refusal")
      assertTrue(refusal, refusal.contains("pinned to $OTHER_COMMIT"))
    }

    // Among whatever earlier runs left in this directory.
    val listed = ModelStore(modelsDir).availableModels().single { (it.modelRef as? Model.OnnxGenai)?.model?.relativeDir == ref.relativeDir }
    Log.i(tag, "listed ${listed.name}: ${listed.sizeBytes} bytes at ${listed.path}")
    // The listed coordinate is the one fetched, with the snapshot the package records.
    assertEquals(path, listed.path)

    // Through the actor, as a job reaches the engine: the path alone tells it which engine loads the model.
    val benchmark =
      JSONObject()
        .put("benchmark_id", "prefill_throughput_$prefill")
        .put("benchmark_type", "prefill_throughput")
        .put("parameter_prefill_tokens", prefill)
    val actor = EngineActor()
    try {
      val result = JSONObject(actor.runFreshSync(benchmark.toString(), listed.path, 0, CONTEXT, 0, LlamacppLoadMode.READ, null, null, null, null))
      Log.i(tag, "through EngineActor: $result")
      assertTrue(result.getDouble("prefill_time_ms") > 0.0)
    } finally {
      actor.destroy()
    }
  }

  private companion object {
    const val PHI3_REPO = "microsoft/Phi-3-mini-4k-instruct-onnx"
    const val PHI3_PREFIX = "cpu_and_mobile/cpu-int4-rtn-block-32"
    const val PREFILL = 64
    const val OTHER_COMMIT = "1111111111111111111111111111111111111111"
    const val CONTEXT = 4096
    const val NANOS_PER_SECOND = 1_000_000_000L
  }
}
