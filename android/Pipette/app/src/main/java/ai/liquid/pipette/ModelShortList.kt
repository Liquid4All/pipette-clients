package ai.liquid.pipette

import android.util.Log
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject

/**
 * The models the server offers this device: the catalog short list (`GET /clients/me/descriptor-catalog`), or the copy of it that the last refresh
 * cached. [ModelTemplateCatalog.current] uses [ModelTemplateCatalog.fallback] when this list is empty.
 *
 * A list belongs to the registration that fetched it ([Owner]). It can contain entries private to that registration's organization. [LocalStorage]
 * calls [clear] when it deletes the registration or saves a different identity.
 */
object ModelShortList {
  private const val TAG = "ModelShortList"
  private val LOGO_DIGEST = Regex("[0-9a-f]{64}")

  /** The registration a list was fetched for. */
  data class Owner(val serverUrl: String, val clientId: String, val organization: String) {
    companion object {
      fun of(registration: RegistrationData): Owner = Owner(registration.serverUrl, registration.clientId, registration.organization)
    }
  }

  private val lock = Any()
  private var issued = 0L
  private var published = 0L
  private val _offered = MutableStateFlow<List<PresetModel>>(emptyList())

  /** The current offered rows. Empty when no list belongs to the current registration. */
  val changes: StateFlow<List<PresetModel>> = _offered.asStateFlow()

  val offered: List<PresetModel>
    get() = _offered.value

  /** Install the cached list in [cacheFile] if [registration] fetched it. Call once at startup, before any screen reads the list. */
  fun loadCache(cacheFile: File, registration: RegistrationData?, ramBytes: Long) {
    val rows =
      runCatching {
          val cache = JSONObject(cacheFile.readText())
          val owner = ownerFromJson(cache.getJSONObject("owner"))
          if (registration == null || owner != Owner.of(registration)) null else parse(cache.getString("body"), ramBytes)
        }
        .getOrNull()
    synchronized(lock) {
      published = issued
      _offered.value = rows.orEmpty()
    }
    if (rows != null) Log.i(TAG, "model list loaded from cache: offered=${rows.size}")
  }

  /**
   * Fetches the list for [registration], publishes it, and saves a copy in [cacheFile]. Returns the number of offered rows. Returns null if [clear]
   * runs or a newer response publishes before this one. Network and parse failures propagate and keep the current list.
   *
   * Checks [currentRegistration] before publication and discards a response if the registration identity changed.
   */
  fun refresh(
    cacheFile: File,
    registration: RegistrationData,
    currentRegistration: () -> RegistrationData?,
    ramBytes: Long,
    fetch: (RegistrationData) -> String,
  ): Int? {
    val owner = Owner.of(registration)
    val request = synchronized(lock) { ++issued }
    val body = fetch(registration)
    val rows = parse(body, ramBytes)
    val cache = JSONObject().put("owner", ownerToJson(owner)).put("body", body)
    synchronized(lock) {
      if (request <= published || currentRegistration()?.let { Owner.of(it) } != owner) return null
      published = request
      _offered.value = rows
      // The rows publish even if the write fails: the fetch succeeded, and only the next launch loses the copy.
      runCatching { writeAtomically(cacheFile, cache.toString()) }.onFailure { Log.w(TAG, "model list cache write failed", it) }
    }
    Log.i(TAG, "model list refreshed: offered=${rows.size}")
    return rows.size
  }

  /** Clears the list, deletes [cacheFile], and discards responses to previous refresh requests. */
  fun clear(cacheFile: File) {
    synchronized(lock) {
      published = issued
      _offered.value = emptyList()
      cacheFile.delete()
    }
    Log.i(TAG, "model list cache cleared")
  }

  /**
   * Parses one row per Hugging Face `gguf_text` member. Skips members whose known size exceeds [ramBytes] and members with unsupported coordinates.
   * Logs the number of excluded members. Throws if the body is not a short list.
   *
   * An entry can offer one quant from more than one member: another repo, or another revision of the file. The catalog then labels the members
   * `<quant>@<variant>`, and each such row gets its [PresetModel.variant]. A member with the plain label keeps no variant.
   */
  internal fun parse(body: String, ramBytes: Long): List<PresetModel> {
    val items = JSONObject(body).getJSONArray("items")
    val models = (0 until items.length()).mapNotNull { items.optJSONObject(it) }.filter { it.optString("descriptor_kind") == "model" }
    val members =
      models.flatMap { item ->
        val array = item.optJSONArray("members") ?: return@flatMap emptyList()
        val family = (item.opt("family") as? String)?.trim()?.takeIf { it.isNotEmpty() }
        // The digest goes into a request path, so a value of any other form leaves the bundled brand logo.
        val logo = (item.opt("family_logo_sha256") as? String)?.takeIf { LOGO_DIGEST.matches(it) }
        (0 until array.length()).map { Entry(item.optString("name"), family, logo) to array.optJSONObject(it) }
      }
    val supported = members.mapNotNull { (entry, member) -> member?.let { presetOrNull(entry, it) } }
    val fitting = supported.filter { (it.row.sizeBytes ?: 0L) <= ramBytes }
    val repeated = fitting.groupingBy { it.row.name to it.label.substringBefore('@') }.eachCount().filterValues { it > 1 }.keys
    val rows =
      fitting.map { (row, label) ->
        if ((row.name to label.substringBefore('@')) in repeated) row.copy(variant = label.substringAfter('@', "").ifEmpty { null }) else row
      }
    if (members.size > rows.size) {
      Log.w(TAG, "model list members skipped: unsupported=${members.size - supported.size} larger_than_ram=${supported.size - fitting.size}")
    }
    return rows
  }

  private data class Entry(val name: String, val family: String?, val familyLogoDigest: String?)

  /** A parsed row and its member's quant label from the catalog, such as `q4_k_m@bartowski`. */
  private data class Parsed(val row: PresetModel, val label: String)

  @Suppress("ReturnCount") // one guard per field that can fail to parse; folding them would hide which one did
  private fun presetOrNull(entry: Entry, member: JSONObject): Parsed? {
    val descriptor = member.optJSONObject("descriptor") ?: return null
    if (descriptor.optString("type") != "gguf_text" || descriptor.optString("source") != "huggingface") return null
    val org = HfOrg.parseOrNull(descriptor.optString("org")) ?: return null
    val repoName = HfRepoName.parseOrNull(descriptor.optString("repo_name")) ?: return null
    val revision = if (descriptor.has("revision")) HfRevision.parseOrNull(descriptor.optString("revision")) ?: return null else null
    val file = GgufFilename.parseOrNull(descriptor.optString("path")) ?: return null
    val identifier = "${HfRepo(org, repoName, revision = revision).reference}/$file"
    val sizeBytes = if (member.isNull("size_bytes")) null else member.optLong("size_bytes").takeIf { it > 0 }
    val label = member.optJSONObject("member_key")?.optString("quantization").orEmpty()
    val row =
      PresetModel(
        id = member.optString("selector").ifBlank { identifier },
        name = entry.name,
        identifier = identifier,
        sizeBytes = sizeBytes,
        quantLabel = label.substringBefore('@').uppercase().ifEmpty { null },
        family = entry.family ?: ModelFamily.defaultFamily(entry.name),
        familyLogoDigest = entry.familyLogoDigest,
      )
    return Parsed(row, label)
  }

  private fun ownerToJson(owner: Owner): JSONObject =
    JSONObject().put("server_url", owner.serverUrl).put("client_id", owner.clientId).put("organization", owner.organization)

  private fun ownerFromJson(json: JSONObject): Owner =
    Owner(json.getString("server_url"), json.getString("client_id"), json.getString("organization"))

  private fun writeAtomically(file: File, text: String) {
    file.parentFile?.mkdirs()
    val temp = File(file.parentFile, "${file.name}.tmp")
    temp.writeText(text)
    check(temp.renameTo(file)) { "could not move ${temp.name} into place" }
  }
}
