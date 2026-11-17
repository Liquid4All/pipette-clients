package ai.liquid.pipette

import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

/**
 * Puts an ONNX Runtime GenAI package on disk from the coordinate that names it, as [ModelProvisioner] puts a GGUF file there.
 *
 * A package is a directory, and it is only a model once every file is present, so it is fetched the way the iOS store fetches one
 * (`docs/pipette-ios/onnx-genai-runtime.md`, Device validation):
 * 1. the repo is frozen to one commit before any byte moves (the coordinate's own revision, or `main` as it is then), so the files cannot come from
 *    two revisions,
 * 2. the files are staged in a sibling directory named for that commit, each resumed from where a previous attempt stopped,
 * 3. each is checked against the size, and for an LFS file the SHA-256, the listing declared,
 * 4. the staged directory is validated as a package ([OnnxGenaiPackage.validate]),
 * 5. and only then renamed into place, so the models directory never holds a package that is half there. The commit goes with it, in [REVISION_FILE],
 *    which is how a later run pinned to a revision knows whether this is that snapshot.
 *
 * Cancelled through the status callback: the caller throws from it, and the transfer reports through it every few megabytes, so a cancelled run stops
 * within one report and keeps what it staged for the next attempt.
 *
 * Blocking, and outside WorkManager, unlike a GGUF download. [JobRunner] calls it from the thread that already blocks for the length of a cell,
 * inside the foreground hold a run keeps, so the transfer needs no worker of its own; the cost is that the Models tab shows nothing of it and a
 * process death resumes only when the cell is run again. The app's download layer names a transfer by one `.gguf` file, and teaching it directories
 * is the larger change this stands in for.
 */
@Suppress("TooManyFunctions") // one per step of the fetch above, and one per kind of staging directory it reclaims
class OnnxPackageFetcher(
  private val modelsDir: File,
  private val http: Http = HuggingFaceHttp { null },
  private val freeBytes: (File) -> Long = { it.usableSpace },
  /** Called once a package has been renamed into place: the models directory now holds something a cached listing of it does not. */
  private val onPublished: () -> Unit = {},
  private val now: () -> Long = { System.currentTimeMillis() },
) : ModelProvisioner.Packages {
  /** What the fetcher asks of the network, so its suite needs none. */
  interface Http {
    /** The body of a GET, as text. */
    fun get(url: String): String

    /** Append the body of a GET to [dest] from byte [offset] on, reporting the bytes written so far. Returns the file's length afterwards. */
    fun download(url: String, dest: File, offset: Long, onBytes: (Long) -> Unit): Long
  }

  /** One file of the listing: its path under the package, its size, and its SHA-256 when HuggingFace stores it in LFS. */
  data class Entry(val path: String, val size: Long, val sha256: String?)

  override fun ensure(ref: HfOnnxGenai, onStatus: (String) -> Unit): Result<String> = runCatching {
    val dest = File(modelsDir, ref.relativeDir)
    if (OnnxGenaiPackage.isPackage(dest.path)) {
      // Published whole, or put there by hand: either way it is what the cell will load, so it is checked rather than replaced.
      OnnxGenaiPackage.validate(dest)
      ref.revision?.let { requireSnapshot(ref, dest, it) }
      return@runCatching dest.absolutePath
    }
    check(!dest.exists()) { "`${dest.path}` exists and is not an ONNX Runtime GenAI package; remove it to fetch ${ref.name}" }

    onStatus("Resolving ${ref.name}")
    val commit = resolveCommit(ref.repo, ref.revision)
    val entries = list(ref, commit)
    val staging = File(dest.parentFile, ".${dest.name}.${commit.take(COMMIT_PREFIX)}$STAGING_SUFFIX")
    discardOtherStagings(dest, staging)
    discardAbandonedStagings(staging)
    staging.mkdirs()

    val missing = entries.filter { entry -> File(staging, entry.path).let { !it.isFile || it.length() != entry.size } }
    val needed = missing.sumOf { it.size - partial(staging, it).length() }
    val free = freeBytes(staging)
    check(needed <= free) { "${ref.name} needs ${needed / BYTES_PER_MB} MB more and this device has ${free / BYTES_PER_MB} MB free" }

    val total = entries.sumOf { it.size }
    var done = total - needed
    for (entry in missing) {
      val before = partial(staging, entry).length()
      fetch(ref, commit, staging, entry) { written -> onStatus("Downloading ${ref.name} ${percent(done + written - before, total)}%") }
      done += entry.size - before
    }

    try {
      OnnxGenaiPackage.validate(staging)
    } catch (invalid: OnnxGenaiPackage.Invalid) {
      // Every file arrived as listed and the whole is still not a package, so no retry can finish it: kept, it would be bytes nothing lists
      // and nothing deletes, skipped over and refused again each time.
      staging.deleteRecursively()
      throw invalid
    }
    File(staging, REVISION_FILE).writeText(commit)
    check(staging.renameTo(dest)) { "could not move ${ref.name} into place at `${dest.path}`" }
    onPublished()
    dest.absolutePath
  }

  /**
   * The commit to fetch: the one [revision] names, or the one `main` points at now. Every later request names it, so a push to the repo mid-transfer
   * cannot mix two revisions. A full commit SHA is its own answer and asks HuggingFace nothing.
   */
  private fun resolveCommit(repo: HfRepo, revision: HfRevision?): String {
    val named = revision?.value ?: "main"
    if (COMMIT.matches(named)) return named
    val sha = JSONObject(http.get("$API/models/$repo/revision/$named")).optString("sha")
    check(COMMIT.matches(sha)) { "HuggingFace named no commit for $repo at $named" }
    return sha
  }

  /**
   * A package on disk is the snapshot a pinned coordinate asks for only if its record says so. One with no record was put there by hand, and one with
   * another commit's is other weights: to run either as the pinned snapshot would file a result under bytes the caller did not name.
   */
  private fun requireSnapshot(ref: HfOnnxGenai, dest: File, revision: HfRevision) {
    val recorded = File(dest, REVISION_FILE).takeIf { it.isFile }?.readText()?.trim()
    checkNotNull(recorded) {
      "`${dest.path}` has no record of the snapshot it was fetched at, and ${ref.name} is pinned to $revision; remove it to fetch that snapshot"
    }
    val wanted = resolveCommit(ref.repo, revision)
    check(recorded == wanted) { "`${dest.path}` holds ${ref.name} at $recorded, and it is pinned to $wanted; remove it to fetch that snapshot" }
  }

  /** The files under the package's directory at [commit], with their paths made relative to it. */
  private fun list(ref: HfOnnxGenai, commit: String): List<Entry> {
    val root = ref.prefix?.let { "/$it" }.orEmpty()
    val listing = JSONArray(http.get("$API/models/${ref.repo}/tree/$commit$root?recursive=true"))
    // One page is all this reads. A package has a dozen files; a listing that fills the page is a whole repo, not a package.
    check(listing.length() < PAGE) { "${ref.relativeDir} lists $PAGE or more files, which is not one package; name its directory with `prefix`" }
    val entries =
      (0 until listing.length())
        .map { listing.getJSONObject(it) }
        .filter { it.optString("type") == "file" }
        .map { file ->
          val path = file.getString("path").removePrefix(root.removePrefix("/")).removePrefix("/")
          // A listed path is server-authored and joined onto the staging directory, so it has to stay inside it.
          check(RepoSubpath.parseOrNull(path) != null) { "${ref.relativeDir} lists `$path`, which is not a path inside the package" }
          val lfs = file.optJSONObject("lfs")
          Entry(path, file.getLong("size"), lfs?.optString("oid")?.takeIf { SHA256.matches(it) })
        }
    check(entries.any { it.path == OnnxGenaiPackage.CONFIG_FILE }) {
      "${ref.relativeDir} has no ${OnnxGenaiPackage.CONFIG_FILE}, so it is not an ONNX Runtime GenAI package; " +
        "a transformers.js export has `.onnx` graphs and no such file"
    }
    return entries
  }

  private fun partial(staging: File, entry: Entry): File = File(staging, "${entry.path}$PART_SUFFIX")

  /** One file: resumed from its `.part`, checked against the listing, then given its name. */
  private fun fetch(ref: HfOnnxGenai, commit: String, staging: File, entry: Entry, onBytes: (Long) -> Unit) {
    val target = File(staging, entry.path)
    val part = partial(staging, entry)
    part.parentFile?.mkdirs()
    // A `.part` past the declared size is from another listing: start over rather than append to it.
    if (part.length() > entry.size) part.delete()
    val root = ref.prefix?.let { "$it/" }.orEmpty()
    // One already at its size was complete when the previous attempt stopped before checking it, so it is checked, not fetched again.
    val length =
      if (part.isFile && part.length() == entry.size) {
        entry.size
      } else {
        http.download("$HOST/${ref.repo}/resolve/$commit/$root${entry.path}", part, part.length(), onBytes)
      }
    if (length != entry.size) {
      throw IOException("${entry.path} is $length bytes and the listing declares ${entry.size}")
    }
    entry.sha256?.let { expected ->
      val actual = sha256(part)
      if (actual != expected) {
        part.delete()
        throw IOException("${entry.path} does not match the SHA-256 the listing declares")
      }
    }
    target.delete()
    check(part.renameTo(target)) { "could not finish ${entry.path}" }
  }

  /** A staging directory of another commit is a transfer nothing will resume. */
  private fun discardOtherStagings(dest: File, current: File) {
    stagingsOf(dest).filter { it != current }.forEach { it.deleteRecursively() }
  }

  /**
   * A staging directory nothing has written to for [ABANDONED_AFTER_MILLIS] belongs to a run that was never repeated. The model list does not show it
   * and no delete names it, so the next fetch of anything is where it is reclaimed.
   */
  private fun discardAbandonedStagings(current: File) {
    val cutoff = now() - ABANDONED_AFTER_MILLIS
    modelsDir
      .walkTopDown()
      .onEnter { !it.name.endsWith(STAGING_SUFFIX) }
      .flatMap { dir -> dir.listFiles { file -> file.isDirectory && file.name.endsWith(STAGING_SUFFIX) }.orEmpty().asSequence() }
      .filter { it != current && lastWritten(it) < cutoff }
      .toList()
      .forEach { it.deleteRecursively() }
  }

  private fun lastWritten(staging: File): Long = staging.walkTopDown().maxOf { it.lastModified() }

  private fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
      val buffer = ByteArray(BUFFER)
      while (true) {
        val read = input.read(buffer)
        if (read < 0) break
        digest.update(buffer, 0, read)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }

  private fun percent(done: Long, total: Long): Long = if (total <= 0) 0 else (done * PERCENT / total).coerceIn(0, PERCENT)

  companion object {
    /** What a staging directory's name ends with. [ModelStore] skips such a directory: it can hold a `genai_config.json` and is not a model. */
    const val STAGING_SUFFIX = ".staging"

    /** How long an untouched staging directory is kept for the run that started it to come back: a week. */
    const val ABANDONED_AFTER_MILLIS = 7L * 24 * 60 * 60 * 1000

    /** The staging directories of the package at [dest], of whatever commit. */
    private fun stagingsOf(dest: File): List<File> =
      dest.parentFile
        ?.listFiles { file -> file.isDirectory && file.name.startsWith(".${dest.name}.") && file.name.endsWith(STAGING_SUFFIX) }
        .orEmpty()
        .toList()

    /** Remove whatever a fetch of the package at [dest] left staged. For a delete of the package, which is otherwise the only thing they outlive. */
    fun discardStagings(dest: File) = stagingsOf(dest).forEach { it.deleteRecursively() }

    /** Inside a published package: the commit it was fetched at. Not part of the package, which names every file it needs. */
    const val REVISION_FILE = ".pipette-revision"

    private const val HOST = "https://huggingface.co"
    private const val API = "$HOST/api"
    private const val PART_SUFFIX = ".part"
    private const val PAGE = 1000
    private const val COMMIT_PREFIX = 12
    private const val BUFFER = 1 shl 20
    private const val BYTES_PER_MB = 1_000_000L
    private const val PERCENT = 100L
    private val COMMIT = Regex("^[0-9a-f]{40}$")
    private val SHA256 = Regex("^[0-9a-f]{64}$")
  }
}

/** The fetcher's network: plain `HttpURLConnection` through [HuggingFaceConnection], the connection [DownloadWorker] opens too. */
class HuggingFaceHttp(private val token: () -> String?) : OnnxPackageFetcher.Http {
  override fun get(url: String): String {
    val connection = open(url, 0)
    try {
      if (connection.responseCode != HttpURLConnection.HTTP_OK) throw IOException("HTTP ${connection.responseCode} for $url")
      return connection.inputStream.bufferedReader().use { it.readText() }
    } finally {
      connection.disconnect()
    }
  }

  override fun download(url: String, dest: File, offset: Long, onBytes: (Long) -> Unit): Long {
    val connection = open(url, offset)
    try {
      // 206 continues the `.part`; 200 means the server ignored the range, so the file starts over.
      val resumed = connection.responseCode == HttpURLConnection.HTTP_PARTIAL
      if (!resumed && connection.responseCode != HttpURLConnection.HTTP_OK) throw IOException("HTTP ${connection.responseCode} for $url")
      RandomAccessFile(dest, "rw").use { file ->
        val start = if (resumed) offset else 0L
        file.setLength(start)
        file.seek(start)
        connection.inputStream.use { input -> copy(input, file, start, onBytes) }
      }
      return dest.length()
    } finally {
      connection.disconnect()
    }
  }

  private fun copy(input: InputStream, file: RandomAccessFile, start: Long, onBytes: (Long) -> Unit) {
    val buffer = ByteArray(BUFFER)
    var written = start
    var reported = start
    while (true) {
      val read = input.read(buffer)
      if (read < 0) break
      file.write(buffer, 0, read)
      written += read
      if (written - reported >= REPORT_EVERY) {
        reported = written
        onBytes(written)
      }
    }
  }

  private fun open(urlString: String, offset: Long): HttpURLConnection = HuggingFaceConnection.open(urlString, offset, token)

  private companion object {
    const val BUFFER = 1 shl 16
    const val REPORT_EVERY = 8L * 1024 * 1024
  }
}
