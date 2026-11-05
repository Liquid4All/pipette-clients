package ai.liquid.pipette

/**
 * How llama.cpp gets a model's weights into memory. The Kotlin mirror of plan-types `LlamacppLoadMode`, which mirrors upstream's `llama_load_mode`.
 *
 * Three spellings have to agree for a cell to survive the round trip, so each is stated here rather than derived:
 * - [wire] is the plan / submission word (`snake_case`, as the plan-types enum serializes).
 * - [cValue] is llama's own enum discriminant, which is what crosses JNI and reaches `llama_model_params.load_mode`. Passing the number rather than
 *   the word keeps the shim from having to parse anything, and keeps this file the single place the mapping is written down.
 * - The C name each entry documents is the upstream constant it must equal.
 *
 * [READ] is the benchmark default and what every Android run used before this setting existed: weights in anonymous RAM, so a `max_memory_usage`
 * figure is comparable with the desktop CLI's and a model too large for the device fails instead of quietly streaming from storage.
 */
// The `cValue`s are llama's own `llama_load_mode` discriminants, so naming each one would
// only restate the entry it sits on; the C constant each case documents is the reference.
@Suppress("MagicNumber")
enum class LlamacppLoadMode(val wire: String, val cValue: Int) {
  /** `LLAMA_LOAD_MODE_NONE`: read the weights into anonymous memory. Named as plan-types names it; the wire word stays `none`. */
  READ("none", 0),

  /** `LLAMA_LOAD_MODE_MMAP`: memory-map the file. llama.cpp's own default, but not this app's. */
  MMAP("mmap", 1),

  /** `LLAMA_LOAD_MODE_MLOCK`: read in, then lock the pages against eviction. */
  MLOCK("mlock", 2),

  /** `LLAMA_LOAD_MODE_MMAP_MLOCK`: map the file and lock the mapping. */
  MMAP_MLOCK("mmap_mlock", 3),

  /** `LLAMA_LOAD_MODE_DIRECT_IO`: bypass the page cache where the platform supports it. */
  DIRECT_IO("direct_io", 4);

  companion object {
    /** What a cell that named no load mode runs with. Stated once, so the engine and the reported flags cannot disagree about it. */
    val DEFAULT = READ

    /**
     * The mode [wire] names, or null when it names none of them. Null rather than a throw or a silent fallback: the caller is parsing a cell it did
     * not author, and a mode this build does not know is a cell it must refuse rather than run under a substituted setting.
     */
    fun fromWireOrNull(wire: String?): LlamacppLoadMode? = entries.firstOrNull { it.wire == wire }

    /**
     * The mode llama's discriminant [cValue] names, falling back to [DEFAULT] for a value no entry claims.
     *
     * Used on the far side of the `:benchmark` binder hop, where the int came from this same enum in the same APK, so an unknown value means a bug
     * rather than untrusted input, and matching the native shim's own clamp keeps a cell running (and reporting) instead of failing at the boundary.
     */
    fun fromCValue(cValue: Int): LlamacppLoadMode = entries.firstOrNull { it.cValue == cValue } ?: DEFAULT
  }
}
