package ai.liquid.pipette.compose.settings

import android.content.Context
import org.json.JSONArray

/** A bundled open-source component (vendored llama.cpp + statically linked Rust crates), loaded from assets/ThirdPartyLicenses.json. */
data class Acknowledgement(val name: String, val license: String, val text: String)

/** Loads the third-party license list (same JSON the iOS client bundles). Cached after first read. */
object Acknowledgements {
  @Volatile private var cached: List<Acknowledgement>? = null

  fun all(context: Context): List<Acknowledgement> {
    cached?.let {
      return it
    }
    val loaded =
      runCatching {
          val json = context.assets.open("ThirdPartyLicenses.json").bufferedReader().use { it.readText() }
          val arr = JSONArray(json)
          (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Acknowledgement(o.getString("name"), o.optString("license"), o.optString("text"))
          }
        }
        .getOrDefault(emptyList())
    cached = loaded
    return loaded
  }
}
