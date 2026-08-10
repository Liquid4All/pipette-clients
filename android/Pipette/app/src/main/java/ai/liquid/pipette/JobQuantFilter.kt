package ai.liquid.pipette

enum class JobQuantFilter(val label: String) {
  ALL("All quants"),
  Q4_0("q4_0"),
  Q4_K_M("q4_km"),
  Q5_K_M("q5_km");

  fun matches(quant: String?): Boolean {
    val normalized = quant?.uppercase() ?: return false
    return when (this) {
      ALL -> true
      Q4_0 -> normalized == "Q4_0"
      Q4_K_M -> normalized == "Q4_K_M"
      Q5_K_M -> normalized == "Q5_K_M"
    }
  }
}
