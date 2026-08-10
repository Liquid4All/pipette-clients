package ai.liquid.pipette.fakes

import ai.liquid.pipette.BenchmarkStore

/** In-memory [BenchmarkStore] for driving [ai.liquid.pipette.BenchmarkSync] without a filesystem. */
class InMemoryBenchmarkStore : BenchmarkStore {
  var index: String? = null
  val details = mutableMapOf<String, String>()
  var syncState: String? = null

  override fun readIndex(): String? = index

  override fun writeIndex(json: String) {
    index = json
  }

  override fun readDetail(id: String): String? = details[id]

  override fun writeDetail(id: String, json: String) {
    details[id] = json
  }

  override fun pruneDetails(keeping: Collection<String>) {
    val keep = keeping.toSet()
    details.keys.retainAll(keep)
  }

  override fun readSyncState(): String? = syncState

  override fun writeSyncState(json: String) {
    syncState = json
  }
}
