package ai.liquid.pipette

import org.json.JSONObject
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
  fun runtimeFlagsAreTheResolvedKnobsAsCanonicalJson() {
    // The flat wire form of plan-types `RuntimeFlagRef`, keys sorted. This is what iOS submits via `RuntimeFlags.submissionValue()`, and what the
    // payload field is documented to carry. Replaces a `-ngl <n> -c <ctx>` summary that named two knobs and silently dropped the rest.
    assertEquals(
      """{"ctx_size":4096,"load_mode":"none","n_ubatch":512,"number_gpu_layers":99}""",
      SubmissionRef.runtimeFlags(RuntimeFlagRef(numberGpuLayers = 99, ctxSize = 4096, nUbatch = 512, loadMode = LlamacppLoadMode.READ)),
    )
  }

  /**
   * A defaulted knob is reported exactly like an authored one. This is the property that makes the record reproducible: nothing downstream can tell
   * which client version chose what, so every value has to be present rather than implied.
   */
  @Test
  fun resolvedFlagsFillEveryKnobTheEngineAnswered() {
    val resolved = RuntimeFlagRef().resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 512)
    assertEquals(99, resolved.numberGpuLayers)
    assertEquals(4096, resolved.ctxSize)
    assertEquals(512, resolved.nUbatch)
    assertEquals(LlamacppLoadMode.READ, resolved.loadMode)
  }

  /** A cell that states a mode keeps it; only the knobs it left unset are answered by the engine. */
  @Test
  fun anAuthoredLoadModeSurvivesResolution() {
    val resolved = RuntimeFlagRef(loadMode = LlamacppLoadMode.MMAP).resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 512)
    assertEquals(LlamacppLoadMode.MMAP, resolved.loadMode)
    assertEquals("""{"ctx_size":4096,"load_mode":"mmap","n_ubatch":512,"number_gpu_layers":99}""", SubmissionRef.runtimeFlags(resolved))
  }

  /**
   * The two spellings of a load mode have to agree: the plan word the record carries, and llama's discriminant the engine loads with. `fromCValue` is
   * the reverse the `:benchmark` service applies to the int it receives over binder. A drift here loads one mode and reports another.
   */
  @Test
  fun loadModeSpellingsRoundTrip() {
    LlamacppLoadMode.entries.forEach { mode ->
      assertEquals(mode, LlamacppLoadMode.fromWireOrNull(mode.wire))
      assertEquals(mode, LlamacppLoadMode.fromCValue(mode.cValue))
    }
    // The wire words plan-types serializes, and the discriminants llama.h fixes.
    assertEquals(listOf("none", "mmap", "mlock", "mmap_mlock", "direct_io"), LlamacppLoadMode.entries.map { it.wire })
    assertEquals(listOf(0, 1, 2, 3, 4), LlamacppLoadMode.entries.map { it.cValue })
    // An unknown mode is refused rather than substituted: running a different load than the cell asked for is the one failure that would go
    // unnoticed.
    assertEquals(null, LlamacppLoadMode.fromWireOrNull("mmap+mlock"))
  }

  /** A flag set this engine can act on in full refuses nothing; the cases that do refuse are the three below. */
  @Test
  fun aFullySupportedFlagSetRefusesNothing() {
    assertEquals(emptyList<String>(), RuntimeFlagRef(numberGpuLayers = 99, ctxSize = 4096, loadMode = LlamacppLoadMode.MMAP).unsupportedKnobs())
  }

  /**
   * A load mode this build does not know is kept verbatim, refused by name, and re-emitted on save. Substituting the default would run a different
   * load than the cell asked for; dropping the word would destroy the authored job file on the next write.
   */
  @Test
  fun anUnknownLoadModeIsRefusedRatherThanSubstituted() {
    val parsed = RuntimeFlagRef.fromJsonOrNull(JSONObject().put("ctx_size", 4096).put("load_mode", "wishful"))
    assertNotNull(parsed)
    assertNull(parsed!!.loadMode)
    assertEquals("wishful", parsed.unknownLoadMode)
    assertEquals(listOf("load_mode=wishful"), parsed.unsupportedKnobs())
    // The other authored knobs survive, and a save round-trips the word this build could not read.
    assertEquals(4096, parsed.ctxSize)
    assertEquals("wishful", parsed.toJson().getString("load_mode"))
  }

  /**
   * The micro-batch is reported as the engine will apply it, not as the cell asked: the shim clamps it to the context and maps a non-positive request
   * to llama.cpp's 512. A cell on a 100-token context that recorded 512 would describe a prefill chunk the run never used.
   */
  @Test
  fun theReportedMicroBatchIsTheOneTheEngineWillUse() {
    assertEquals(100, RuntimeFlagRef().resolve(nGpuLayers = 99, contextSize = 100, nUbatch = 512).nUbatch)
    assertEquals(512, RuntimeFlagRef().resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 512).nUbatch)
    assertEquals(512, RuntimeFlagRef().resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 0).nUbatch)
    // A non-positive context is the shim's fallback case, which this side cannot name: report the request rather than a clamped zero.
    assertEquals(512, RuntimeFlagRef().resolve(nGpuLayers = 99, contextSize = 0, nUbatch = 512).nUbatch)
    // A cell's own request is clamped the same way.
    assertEquals(64, RuntimeFlagRef(nUbatch = 64).resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 512).nUbatch)
  }

  /**
   * A knob this runtime has no home for is refused by name rather than dropped. `mmap` matters most: it is the spelling the CLI cells still take, so
   * it is what a hand-authored job file copies from a desktop plan, and dropping it would load `none` while the author asked to map.
   */
  @Test
  fun aKnobThisRuntimeDoesNotDeclareIsRefusedRatherThanDropped() {
    // Every knob plan-types declares for another llama family but not for this one, including the two only the iOS engine applies.
    listOf<Pair<String, Any>>("mmap" to true, "flash_attention" to "on", "no_cache" to true, "threads" to 6, "swa_full" to true, "raw" to "x")
      .forEach { (key, value) ->
        val parsed = RuntimeFlagRef.fromJsonOrNull(JSONObject().put(key, value).put("ctx_size", 4096))
        assertNotNull(parsed)
        assertEquals(listOf(key), parsed!!.unsupportedKnobs())
        // And it survives the save the runner performs right after failing the cell: dropping it would leave the rerun with nothing to refuse, so the
        // cell would quietly load the default the author never asked for.
        assertEquals(value, parsed.toJson().get(key))
        assertEquals(listOf(key), RuntimeFlagRef.fromJsonOrNull(parsed.toJson())!!.unsupportedKnobs())
      }
  }

  /** A context the shim would replace with llama's own default is refused rather than reported as if it had been used. */
  @Test
  fun aNonPositiveContextIsRefused() {
    assertEquals(listOf("ctx_size=0"), RuntimeFlagRef(ctxSize = 0).unsupportedKnobs())
    assertEquals(emptyList<String>(), RuntimeFlagRef(ctxSize = 1).unsupportedKnobs())
  }

  /** Resolving before checking still cannot substitute a mode this build could not read. */
  @Test
  fun resolvingAnUnknownLoadModeLeavesItUnanswered() {
    val parsed = RuntimeFlagRef.fromJsonOrNull(JSONObject().put("load_mode", "wishful"))!!
    assertNull(parsed.resolve(nGpuLayers = 99, contextSize = 4096, nUbatch = 512).loadMode)
  }

  /** An explicit JSON null is "unset", not `false`: the app's own writers emit null for an absent optional. */
  @Test
  fun anExplicitNullKnobDecodesAsUnset() {
    val parsed = RuntimeFlagRef.fromJsonOrNull(JSONObject().put("swa_full", JSONObject.NULL).put("ctx_size", JSONObject.NULL))
    assertNotNull(parsed)
    assertNull(parsed!!.ctxSize)
    // An explicit null is not an authored knob, so nothing is refused for it either.
    assertEquals(emptyList<String>(), parsed.unsupportedKnobs())
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
  fun modelFlagsAreReportedOnTheEvalCellsOnly() {
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
  fun aVisionEvalReportsTheThinkingFlagAlone() {
    // plan-types refuses `reasoning_effort` on a vision cell, so a vlm_eval row carries `enable_thinking` and nothing else.
    val flags = ModelFlags(enableThinking = true, reasoningEffort = ReasoningEffort.LOW)
    assertEquals("enable_thinking=true", flags.submissionString(BenchmarkType.VLM_EVAL))
    assertNull(ModelFlags(reasoningEffort = ReasoningEffort.LOW).submissionString(BenchmarkType.VLM_EVAL))
  }

  @Test
  fun modelFlagsSpellTheReasoningEffortAfterTheThinkingFlag() {
    val repo = HfRepo.parseSlug("LiquidAI/gpt-oss-GGUF")
    val filename = GgufFilename.parse("gpt-oss-Q4_0.gguf")
    // Rows at two efforts must hash apart, so the effort is in the string, after the thinking flag as plan-types orders them, and alone when
    // no thinking flag is set.
    val both = Model.GgufText(HfGgufText(repo, filename, ModelFlags(enableThinking = true, reasoningEffort = ReasoningEffort.HIGH)))
    val effortAlone = Model.GgufText(HfGgufText(repo, filename, ModelFlags(reasoningEffort = ReasoningEffort.LOW)))

    assertEquals("enable_thinking=true,reasoning_effort=high", both.modelFlags.submissionString(BenchmarkType.EVAL))
    assertEquals("reasoning_effort=low", effortAlone.modelFlags.submissionString(BenchmarkType.EVAL))
    assertNull(both.modelFlags.submissionString(BenchmarkType.PREFILL_THROUGHPUT))
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
