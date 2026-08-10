package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the canonical wire shape of the `model_descriptor` / `runtime_descriptor` / `runtime_flags` / `benchmark_flags` submission specs, so the
 * Kotlin encoder can't drift from the reshaped plan-types schema (PIP-340). For the three descriptors it also pins parity with iOS
 * `SubmissionRef.swift`, which emits the same sorted-key form; `benchmark_flags` has no iOS encoder yet (PIP-429), so nothing cross-platform is
 * guarded there. The exact-string assertions double as canonical-form (sorted keys, no whitespace) guards.
 */
class SubmissionRefTest {
  /**
   * The descriptor for a coordinate, or null when it can't be formed. `SubmissionRef` no longer exposes this pairing: production reads the typed
   * model for its flags too, so building the string is now the caller's step.
   */
  private fun modelDescriptorOrNull(modelName: String, modelFilename: String, mmprojFilename: String?): String? =
    SubmissionRef.typedModelOrNull(modelName, modelFilename, mmprojFilename)?.let { SubmissionRef.model(it) }

  @Test
  fun textModelDescriptorIsCanonicalTaggedShape() {
    val ref = modelDescriptorOrNull("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.gguf", null)
    assertEquals(
      """{"org":"LiquidAI","path":"LFM2.5-350M-Q4_0.gguf","repo_name":"LFM2.5-350M-GGUF","source":"huggingface","type":"gguf_text"}""",
      ref,
    )
  }

  @Test
  fun visionModelDescriptorNamesModelAndMmproj() {
    val ref = modelDescriptorOrNull("LiquidAI/LFM2.5-VL-450M-GGUF", "LFM2.5-VL-450M-Q4_0.gguf", "mmproj-f16.gguf")
    assertEquals(
      """{"mmproj":"mmproj-f16.gguf","model":"LFM2.5-VL-450M-Q4_0.gguf",""" +
        """"org":"LiquidAI","repo_name":"LFM2.5-VL-450M-GGUF","source":"huggingface","type":"gguf_vision"}""",
      ref,
    )
  }

  @Test
  fun modelDescriptorElidedWhenModelNameIsNotAnHfSlug() {
    // Imported single file: modelName is a bare filename (no `org/repo`), so there's no lossless coordinate to send.
    assertNull(modelDescriptorOrNull("some-local-model.gguf", "some-local-model.gguf", null))
  }

  @Test
  fun modelDescriptorElidedWhenFilenameIsNotGguf() {
    assertNull(modelDescriptorOrNull("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_0.bin", null))
  }

  @Test
  fun visionModelDescriptorElidedWhenMmprojIsNotGguf() {
    assertNull(modelDescriptorOrNull("LiquidAI/LFM2.5-VL-450M-GGUF", "LFM2.5-VL-450M-Q4_0.gguf", "mmproj-f16.bin"))
  }

  @Test
  fun runtimeDescriptorIsLlamacppApkPipetteWithRepoAndFlavor() {
    assertEquals(
      """{"flavor":"android-arm64-v8","repository_url":"github.com/ggml-org/llama.cpp",""" +
        """"repository_version":"b8683","type":"llamacpp_apk_pipette"}""",
      SubmissionRef.runtime("b8683"),
    )
  }

  @Test
  fun runtimeFlagsMatchIosCliString() {
    // Byte-identical to iOS PayloadBuilder's `-ngl <n> -c <ctx>`; n_ubatch is intentionally not recorded (iOS omits it too).
    assertEquals("-ngl 99 -c 4096", SubmissionRef.runtimeFlags(nGpuLayers = 99, contextSize = 4096))
  }

  @Test
  fun modelDescriptorNeverCarriesModelFlags() {
    // The reshaped plan-types `Model` no longer embeds generation flags — they're a separate per-cell concept — so `model_descriptor` must not carry
    // them even when the (internal) typed model does. Mirrors iOS, whose `ModelRef` encoder emits only the coordinate.
    val repo = HfRepo.parseSlug("LiquidAI/Qwen-GGUF")
    val filename = GgufFilename.parse("qwen-Q4_0.gguf")
    val withFlags = SubmissionRef.model(Model.GgufText(HfGgufText(repo, filename, ModelFlags(enableThinking = true))))
    assertTrue("model_flags must never appear in the descriptor", !withFlags.contains("model_flags"))
    assertEquals("""{"org":"LiquidAI","path":"qwen-Q4_0.gguf","repo_name":"Qwen-GGUF","source":"huggingface","type":"gguf_text"}""", withFlags)
  }

  @Test
  fun benchmarkFlagsRecordTheReadinessPolicyThatGatedTheCell() {
    // The shape `BenchmarkFlags::submission_value` produces for a timing variant: the axis keys stripped, leaving the resolved readiness block. Units
    // and spelling match the plan's `readiness.max_wait_secs` / `readiness.skip_thermal` so a reader compares values rather than converting them.
    assertEquals(
      """{"readiness":{"max_wait_secs":180,"skip_thermal":false}}""",
      SubmissionRef.benchmarkFlagsOrNull(BenchmarkType.PREFILL_THROUGHPUT, ReadinessPolicy(maxWaitSecs = 180L, skipThermal = false)),
    )
  }

  @Test
  fun benchmarkFlagsReportAWaivedThermalGate() {
    // The whole point of the field: a waived gate is otherwise invisible in a result. No app surface sets this yet (PIP-434), but the encoder must
    // carry it the moment one does, rather than the waiver landing silently.
    assertEquals(
      """{"readiness":{"max_wait_secs":180,"skip_thermal":true}}""",
      SubmissionRef.benchmarkFlagsOrNull(BenchmarkType.DECODE_THROUGHPUT, ReadinessPolicy(maxWaitSecs = 180L, skipThermal = true)),
    )
  }

  @Test
  fun modelFlagsAreReportedOnEvalOnly() {
    val repo = HfRepo.parseSlug("LiquidAI/Qwen-GGUF")
    val filename = GgufFilename.parse("qwen-Q4_0.gguf")
    val thinking = Model.GgufText(HfGgufText(repo, filename, ModelFlags(enableThinking = true)))

    // `enable_thinking` changes what the model generates, which moves an eval score and nothing else. Reporting it on a throughput row would split
    // warehouse joins on a value that had no effect on the number being joined, so every non-eval arm reports nothing even when the flag is set.
    assertEquals("enable_thinking=true", thinking.modelFlags.submissionString(BenchmarkType.EVAL))
    assertNull(thinking.modelFlags.submissionString(BenchmarkType.PREFILL_THROUGHPUT))
    assertNull(thinking.modelFlags.submissionString(BenchmarkType.DECODE_THROUGHPUT))
    assertNull(thinking.modelFlags.submissionString(BenchmarkType.END_TO_END_LATENCY))
    assertNull(thinking.modelFlags.submissionString(BenchmarkType.MAX_MEMORY_USAGE))
    assertNull(thinking.modelFlags.submissionString(BenchmarkType.VL_THROUGHPUT))
  }

  @Test
  fun modelFlagsAreAbsentForADefaultModelEvenOnEval() {
    // The state every Android submission is in today: `typedModelOrNull` builds with default flags, so there is nothing to report even on the one
    // benchmark type that would report it. This is what makes the field absent rather than `enable_thinking=false`.
    val default = SubmissionRef.typedModelOrNull("LiquidAI/Qwen-GGUF", "qwen-Q4_0.gguf", null)
    assertNotNull(default)
    assertNull(default!!.modelFlags.submissionString(BenchmarkType.EVAL))
  }

  @Test
  fun benchmarkFlagsAreElidedForCellsTheSchemaModelsNoReadinessFor() {
    val policy = ReadinessPolicy(maxWaitSecs = 180L, skipThermal = false)
    // max_memory_usage has no flag variant at all, and eval's variant has no readiness field. Android's between-cell gate does run before both,
    // unlike the CLI's, so the omission means "nothing to report in this schema" rather than "no wait happened".
    assertNull(SubmissionRef.benchmarkFlagsOrNull(BenchmarkType.MAX_MEMORY_USAGE, policy))
    assertNull(SubmissionRef.benchmarkFlagsOrNull(BenchmarkType.EVAL, policy))
  }
}
