package ai.liquid.pipette

/**
 * Maps a model family to its vendor brand logo (mirrors the iOS client's `ModelBrand`). Detection keys off the model name/repo text rather than the
 * HF org, because redistributors (e.g. `unsloth/gemma-...`) repackage other vendors' models — the org would mislabel them. Returns a drawable
 * resource id, or null when there's no logo for the vendor (caller falls back to a neutral glyph).
 */
object ModelBrand {
  fun logoRes(name: String, hfRepo: String?): Int? {
    val hay = "$name ${hfRepo ?: ""}".lowercase()
    return when {
      hay.contains("lfm") || hay.contains("liquid") -> R.drawable.brand_liquid
      hay.contains("gemma") -> R.drawable.brand_google
      hay.contains("granite") || hay.contains("ibm") -> R.drawable.brand_ibm
      hay.contains("qwen") -> R.drawable.brand_qwen
      hay.contains("llama") || hay.contains("meta") -> R.drawable.brand_meta
      hay.contains("mistral") || hay.contains("ministral") || hay.contains("mixtral") -> R.drawable.brand_mistral
      else -> null
    }
  }
}
