package ai.liquid.pipette

import java.io.File

/**
 * On-device persistence for the server-synced benchmark catalog, mirroring iOS's `BenchmarkStore` protocol. Kept as an interface so [BenchmarkSync]
 * is testable against an in-memory fake without touching the filesystem.
 *
 * Three pieces, matching the iOS on-disk layout:
 * - **index** — the kept list entries (definitions only), server-shaped JSON array. The sole source the picker reads.
 * - **detail/<id>** — the per-id benchmark detail (carries eval `samples`). One file per benchmark.
 * - **sync-state** — the ETag bookkeeping ([BenchmarkSync.SyncState]) that drives the next conditional pull.
 */
interface BenchmarkStore {
  fun readIndex(): String?

  fun writeIndex(json: String)

  fun readDetail(id: String): String?

  fun writeDetail(id: String, json: String)

  /** Delete detail files for ids no longer in the catalog (the index self-prunes; the details don't). */
  fun pruneDetails(keeping: Collection<String>)

  fun readSyncState(): String?

  fun writeSyncState(json: String)
}

/**
 * File-backed [BenchmarkStore] rooted at [dir] (in production, `filesDir/Pipette/benchmarks`). Layout:
 * ```
 * <dir>/index.json          — kept list entries
 * <dir>/detail/<id>.json    — per-id detail
 * <dir>/sync.json           — sync state
 * ```
 */
class FileBenchmarkStore(private val dir: File) : BenchmarkStore {
  private val indexFile: File
    get() = File(dir, "index.json")

  private val detailDir: File
    get() = File(dir, "detail")

  private val syncStateFile: File
    get() = File(dir, "sync.json")

  override fun readIndex(): String? = indexFile.takeIf { it.exists() }?.readText()

  override fun writeIndex(json: String) {
    dir.mkdirs()
    indexFile.writeText(json)
  }

  override fun readDetail(id: String): String? = detailFile(id)?.takeIf { it.exists() }?.readText()

  override fun writeDetail(id: String, json: String) {
    val file = detailFile(id) ?: return
    detailDir.mkdirs()
    file.writeText(json)
  }

  override fun pruneDetails(keeping: Collection<String>) {
    val keep = keeping.mapNotNull { detailFile(it)?.name }.toSet()
    detailDir.listFiles { file -> file.isFile && file.name.endsWith(".json") }?.forEach { file -> if (file.name !in keep) file.delete() }
  }

  override fun readSyncState(): String? = syncStateFile.takeIf { it.exists() }?.readText()

  override fun writeSyncState(json: String) {
    dir.mkdirs()
    syncStateFile.writeText(json)
  }

  /**
   * The detail file for [id], or null if the id can't be a safe filename (contains a path separator or `..`). Benchmark ids are alphanumeric with
   * `_`/`x` in practice, but guard so a hostile/odd server id can't escape [detailDir].
   */
  private fun detailFile(id: String): File? {
    val unsafe = id.isBlank() || id in setOf(".", "..") || id.any { it == '/' || it == '\\' }
    return if (unsafe) null else File(detailDir, "$id.json")
  }
}
