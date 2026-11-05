package ai.liquid.pipette

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The manifest round-trip for [Model], and the arms that must degrade rather than throw.
 *
 * This is a *stored* format, so the decode side is the load-bearing half: a job file written by this build has to be readable by the next one, and a
 * file written by a build that spelled something differently has to be readable too, minus the part that no longer parses.
 */
class ModelRefManifestJsonTest {
  private val repo = HfRepo.parseSlug("LiquidAI/LFM2-1.2B-GGUF")

  @Test
  fun `a text model round-trips`() {
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf")))

    assertEquals(ref, Model.fromJsonOrNull(ref.toJson()))
  }

  @Test
  fun `a vision model round-trips with both filenames`() {
    val ref = Model.GgufVision(HfGgufVision(repo, GgufFilename.parse("lfm2-vl-1.6b-q4_k_m.gguf"), GgufFilename.parse("mmproj-lfm2-vl-f16.gguf")))

    assertEquals(ref, Model.fromJsonOrNull(ref.toJson()))
  }

  /** Without this the re-fetch of a gated repo would go out unauthenticated and 401, and the cell would report the weights as missing. */
  @Test
  fun `requiresAuth survives the round-trip`() {
    val gated = HfRepo.parseSlug("LiquidAI/Private-GGUF").copy(requiresAuth = true)
    val ref = Model.GgufText(HfGgufText(gated, GgufFilename.parse("private-q4_k_m.gguf")))

    val decoded = Model.fromJsonOrNull(ref.toJson())

    assertEquals(ref, decoded)
    assertEquals(true, decoded?.requiresAuth)
  }

  @Test
  fun `model flags survive the round-trip`() {
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf"), ModelFlags(enableThinking = true)))

    assertEquals(ref, Model.fromJsonOrNull(ref.toJson()))
  }

  @Test
  fun `a reasoning effort survives the round-trip`() {
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf"), ModelFlags(reasoningEffort = ReasoningEffort.MEDIUM)))

    assertEquals(ref, Model.fromJsonOrNull(ref.toJson()))
    assertEquals("medium", ref.toJson().getString("reasoning_effort"))
  }

  /** An effort this build cannot spell drops out rather than failing the whole entry, as any other part that no longer parses does. */
  @Test
  fun `an unknown reasoning effort decodes as absent`() {
    val json =
      JSONObject()
        .put("type", "hf_gguf_text")
        .put("repo", "LiquidAI/LFM2-1.2B-GGUF")
        .put("filename", "lfm2-1.2b-q4_k_m.gguf")
        .put("reasoning_effort", "none")

    val decoded = Model.fromJsonOrNull(json)

    assertNotNull(decoded)
    assertNull(decoded?.modelFlags?.reasoningEffort)
  }

  /** An unset flag stays unset rather than decoding as `false`, which would report a value the author never asked for. */
  @Test
  fun `an absent thinking flag decodes as absent, not as false`() {
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf")))

    assertNull(Model.fromJsonOrNull(ref.toJson())?.modelFlags?.enableThinking)
  }

  @Test
  fun `an absent object decodes to null`() {
    assertNull(Model.fromJsonOrNull(null))
  }

  @Test
  fun `an unknown variant decodes to null rather than throwing`() {
    val json = JSONObject().put("type", "hf_mlx").put("repo", "LiquidAI/LFM2-1.2B").put("filename", "model.safetensors")

    assertNull(Model.fromJsonOrNull(json))
  }

  @Test
  fun `a malformed repo decodes to null rather than throwing`() {
    val json = JSONObject().put("type", "hf_gguf_text").put("repo", "no-slash-here").put("filename", "lfm2-1.2b-q4_k_m.gguf")

    assertNull(Model.fromJsonOrNull(json))
  }

  @Test
  fun `a vision entry missing its projector decodes to null`() {
    val json = JSONObject().put("type", "hf_gguf_vision").put("repo", "LiquidAI/LFM2-VL-GGUF").put("filename", "lfm2-vl-q4_k_m.gguf")

    assertNull(Model.fromJsonOrNull(json))
  }

  /** A cell that predates the field, and a locally imported file that has no coordinate: both decode to a cell that simply cannot re-fetch itself. */
  @Test
  fun `a cell with no reference round-trips as one`() {
    val cell = JobCell(benchmarkId = "decode_throughput_512_100", benchmarkType = "decode_throughput", modelPath = "/models/a.gguf", modelName = "a")

    val decoded = JobCell.fromJson(cell.toJson())

    assertNull(decoded.modelRef)
    // Omitted rather than written as null, so a manifest from before this field round-trips unchanged.
    assertEquals(false, cell.toJson().has("modelRef"))
  }

  @Test
  fun `a cell carries its reference through a manifest round-trip`() {
    val ref = Model.GgufText(HfGgufText(repo, GgufFilename.parse("lfm2-1.2b-q4_k_m.gguf")))
    val cell =
      JobCell(
        benchmarkId = "decode_throughput_512_100",
        benchmarkType = "decode_throughput",
        modelPath = "LiquidAI/LFM2-1.2B-GGUF/lfm2-1.2b-q4_k_m.gguf",
        modelName = "LiquidAI/LFM2-1.2B-GGUF",
        modelRef = ref,
      )

    assertEquals(ref, JobCell.fromJson(cell.toJson()).modelRef)
  }
}
