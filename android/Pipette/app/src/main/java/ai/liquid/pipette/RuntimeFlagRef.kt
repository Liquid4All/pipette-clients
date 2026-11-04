package ai.liquid.pipette

import org.json.JSONObject

/**
 * The load knobs one cell runs with: the Kotlin mirror of plan-types `RuntimeFlagRef`, narrowed to the `llamacpp_apk_pipette × gguf_*` variants this
 * app can actually run.
 *
 * Flat, like the wire form it mirrors: upstream keeps a closed enum of `(benchmark, runtime, model)` variants and flattens it to this bag of optional
 * knobs at the serde boundary, so a client that only ever runs one runtime needs the bag alone. The axis keys are deliberately absent, because they
 * identify the cell, and plan-types' `submission_value` drops them too.
 *
 * Every field is optional and means "the cell did not say": [resolve] is what turns absence into this engine's answer, and the resolved value is what
 * both the load and the submitted record use. That pairing is the point: a submitted `runtime_flags` that omitted a defaulted knob would describe a
 * run nobody could reproduce.
 *
 * A knob this engine cannot honor is refused rather than dropped, via [unsupportedKnobs]. Silently ignoring one would report a setting that never
 * reached the load, which is the same failure as omitting a defaulted one, only harder to notice.
 */
data class RuntimeFlagRef(
  val numberGpuLayers: Int? = null,
  val ctxSize: Int? = null,
  val nUbatch: Int? = null,
  val loadMode: LlamacppLoadMode? = null,
  /**
   * The `load_mode` word from a stored cell that this build does not recognize, kept verbatim so [toJson] cannot destroy it on the next save and so
   * [unsupportedKnobs] can name it. Null whenever [loadMode] parsed.
   */
  val unknownLoadMode: String? = null,
  /**
   * Knobs this runtime has no home for, as authored: key to value, kept from the parsed JSON so [unsupportedKnobs] can name them and [toJson] can
   * write back what the cell actually said. plan-types refuses these for `llamacpp_apk_pipette` at parse time; a hand-authored job file never went
   * through that check, which is the only way one reaches this class.
   */
  val refused: Map<String, Any> = emptyMap(),
) {
  /**
   * The knobs this cell authored that the engine cannot honor, by wire name. Empty for every cell the in-app UI composes, since it authors none.
   *
   * The caller fails the cell on a non-empty list. Two kinds land here. [refused] holds knobs plan-types would have rejected at parse, which a
   * hand-authored job file never went through. `ctx_size <= 0` and an unreadable [unknownLoadMode] are knobs this variant does declare, whose value
   * this engine still cannot act on, so only a check here can catch them.
   */
  fun unsupportedKnobs(): List<String> = buildList {
    // A context the shim would reject and replace with llama's own default. Refused rather
    // than mirrored: guessing that default here is how the record starts describing a load
    // that did not happen.
    if (ctxSize != null && ctxSize <= 0) add("ctx_size=$ctxSize")
    if (unknownLoadMode != null) add("load_mode=$unknownLoadMode")
    // Knobs plan-types refuses for this runtime outright. `mmap` above all: it is the
    // spelling the CLI cells still take, so it is the one a hand-authored job file copies
    // from a desktop plan, and dropping it silently would load `none` while the author asked
    // for a mapped read.
    addAll(refused.keys)
  }

  /**
   * This cell's knobs with the engine's answer filled in wherever it stated none. [contextSize] and [nUbatch] come from the job the cell belongs to
   * (Android sizes context per benchmark at the job layer, unlike iOS, which derives it in the engine).
   *
   * Only the knobs the engine actually applies are answered. Anything in [refused] is carried through untouched by `copy`, because a cell holding one
   * is failed before this runs; answering it would invent a value for a load that never happens.
   */
  fun resolve(nGpuLayers: Int, contextSize: Int, nUbatch: Int): RuntimeFlagRef {
    val resolvedCtx = ctxSize ?: contextSize
    // What the shim will actually use: `<= 0` means llama.cpp's 512 default, and the
    // micro-batch is then clamped to the context (`ee_llama_model_load`). Recording the
    // requested value instead would claim a 512-token prefill chunk on a 100-token context.
    val requestedUbatch = (this.nUbatch ?: nUbatch).let { if (it > 0) it else DEFAULT_UBATCH }
    return copy(
      numberGpuLayers = numberGpuLayers ?: nGpuLayers,
      ctxSize = resolvedCtx,
      // Clamped only against a context the shim will keep: for a non-positive one it falls
      // back to llama's own default, which this side cannot name, so reporting the request
      // is closer to the truth than reporting zero.
      nUbatch = if (resolvedCtx > 0) minOf(requestedUbatch, resolvedCtx) else requestedUbatch,
      // A mode this build could not read is left unanswered rather than defaulted, so a caller that resolves before checking [unsupportedKnobs]
      // still cannot load one thing and report another. The authored word survives in [unknownLoadMode] either way.
      loadMode = loadMode ?: if (unknownLoadMode != null) null else LlamacppLoadMode.DEFAULT,
    )
  }

  /**
   * Persisted / submitted form: the set knobs only. Key order is `org.json`'s own and therefore unspecified, which is why
   * [SubmissionRef.runtimeFlags] canonicalizes (sorted keys, no whitespace) before anything reaches the wire. Matches plan-types
   * `RuntimeFlags::submission_value`.
   */
  fun toJson(): JSONObject =
    JSONObject().apply {
      numberGpuLayers?.let { put("number_gpu_layers", it) }
      ctxSize?.let { put("ctx_size", it) }
      nUbatch?.let { put("n_ubatch", it) }
      // The parsed mode, or the unrecognized word the cell authored: a save must not quietly rewrite a job file into one this build likes better.
      (loadMode?.wire ?: unknownLoadMode)?.let { put("load_mode", it) }
      // Same rule for a key this runtime does not declare. The runner saves the manifest
      // right after failing the cell, so dropping these here would mean the rerun no longer
      // sees them, stops refusing, and loads the default the author never asked for.
      refused.forEach { (key, value) -> put(key, value) }
    }

  companion object {
    /** llama.cpp's own default prefill micro-batch, as `ee_llama_model_load` applies it for a non-positive request. */
    private const val DEFAULT_UBATCH = 512

    /**
     * The llama.cpp knobs plan-types declares for other llama families but not for this one: the CLI cells' argv-only settings and their `raw` escape
     * hatch, plus the two load-path knobs only the iOS engine applies. A cell authoring any of them is refused rather than run without it.
     */
    private val REFUSED_KEYS = listOf("mmap", "flash_attention", "no_cache", "threads", "swa_full", "raw")

    /**
     * Parse a stored / authored flag set, or null when there is nothing to parse. Unknown keys are ignored rather than refused, because a job file
     * written by a newer build can still run here, and the knobs this build understands are exactly the ones it can honor.
     *
     * A `load_mode` this build does not know is kept as [unknownLoadMode] rather than dropped or substituted, so the cell is refused by name at run
     * time and the authored word survives a load/save cycle. See [LlamacppLoadMode.fromWireOrNull].
     */
    fun fromJsonOrNull(json: JSONObject?): RuntimeFlagRef? {
      if (json == null) return null
      val wire = json.optNullableString("load_mode")
      val loadMode = wire?.let { LlamacppLoadMode.fromWireOrNull(it) }
      return RuntimeFlagRef(
        refused = REFUSED_KEYS.filter { json.has(it) && !json.isNull(it) }.associateWith { json.get(it) },
        numberGpuLayers = json.optNullableInt("number_gpu_layers"),
        ctxSize = json.optNullableInt("ctx_size"),
        nUbatch = json.optNullableInt("n_ubatch"),
        loadMode = loadMode,
        unknownLoadMode = if (wire != null && loadMode == null) wire else null,
      )
    }
  }
}
