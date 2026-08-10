package ai.liquid.pipette

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Persistent record of outstanding downloads — the single source of truth for "what the user wants downloaded," shared by [DownloadCoordinator]
 * (writes on enqueue/pause/resume, clears on cancel) and [DownloadWorker] (marks failed, clears on success). Each record carries its own lifecycle
 * [Record.state] (queued/paused/failed) so a relaunch can rebuild the exact UI state and re-enqueue only what should still be running — there is no
 * separate paused-key side channel. Records survive process death so a resumed worker and the UI agree on what's outstanding; a completed download
 * clears its record so it is never re-queued.
 */
object DownloadStore {
  private const val PREFS = "pipette_downloads"
  private const val KEY_RECORDS = "records_json"

  data class Record(
    val key: String,
    val filename: String,
    val urlString: String,
    val repo: String?,
    val familyId: String?,
    val displayName: String?,
    val destPath: String,
    val partialPath: String,
    val state: String = DownloadWorker.STATE_QUEUED,
  ) {
    /** A record we can actually act on: it needs the identity + the URL and paths a resume depends on. */
    fun isValid(): Boolean = key.isNotBlank() && urlString.isNotBlank() && destPath.isNotBlank() && partialPath.isNotBlank()

    fun toJson(): JSONObject =
      JSONObject()
        .put("key", key)
        .put("filename", filename)
        .put("urlString", urlString)
        .putOptString("repo", repo)
        .putOptString("familyId", familyId)
        .putOptString("displayName", displayName)
        .put("destPath", destPath)
        .put("partialPath", partialPath)
        .put("state", state)

    companion object {
      fun fromJson(json: JSONObject): Record =
        Record(
          key = json.getString("key"),
          filename = json.getString("filename"),
          urlString = json.optString("urlString"),
          repo = json.optNullableString("repo"),
          familyId = json.optNullableString("familyId"),
          displayName = json.optNullableString("displayName"),
          destPath = json.getString("destPath"),
          partialPath = json.getString("partialPath"),
          state = json.optString("state").ifBlank { DownloadWorker.STATE_QUEUED },
        )
    }
  }

  private fun prefs(context: Context) = context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

  @Synchronized
  fun records(context: Context): List<Record> {
    val raw = prefs(context).getString(KEY_RECORDS, "[]") ?: "[]"
    val array = runCatching { JSONArray(raw) }.getOrNull() ?: return emptyList()
    // Parse each record independently so one malformed/partial entry is skipped instead of dropping every outstanding download.
    return (0 until array.length()).mapNotNull { index ->
      runCatching { Record.fromJson(array.getJSONObject(index)) }.getOrNull()?.takeIf { it.isValid() }
    }
  }

  @Synchronized fun saveRecord(context: Context, record: Record) = writeRecords(context, records(context).filterNot { it.key == record.key } + record)

  /** Move a record to a new lifecycle [state] (no-op if it's gone — e.g. cancelled or completed in the meantime). */
  @Synchronized
  fun updateState(context: Context, key: String, state: String) {
    val records = records(context)
    val record = records.firstOrNull { it.key == key } ?: return
    writeRecords(context, records.filterNot { it.key == key } + record.copy(state = state))
  }

  @Synchronized fun clearRecord(context: Context, key: String) = writeRecords(context, records(context).filterNot { it.key == key })

  private fun writeRecords(context: Context, records: List<Record>) {
    val array = JSONArray()
    records.forEach { array.put(it.toJson()) }
    prefs(context).edit().putString(KEY_RECORDS, array.toString()).apply()
  }
}
