package ai.liquid.pipette

import java.io.File
import java.io.IOException
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * That a package reaches the models directory whole or not at all: one commit, staged, checked against the listing, validated, then renamed.
 *
 * Against a network that is a map of URLs, so the cases a real one produces least on demand (a transfer cut short, a repo pushed to mid-fetch, a file
 * that arrives altered) are each a line of setup.
 */
class OnnxPackageFetcherTest {
  @get:Rule val temp = TemporaryFolder()

  private val repo = HfRepo.parseSlug("microsoft/Phi-3-mini-4k-instruct-onnx")
  private val ref = HfOnnxGenai(repo, RepoSubpath.parse("cpu_and_mobile/cpu-int4"))
  private val commit = "5f5f794c1c23c9d5ee142af85df02a6cc52d6945"

  /** The package's files: a whole one by [OnnxGenaiPackage.validate]'s lights, with the weights in LFS as HuggingFace keeps them. */
  private val files =
    linkedMapOf(
      OnnxGenaiPackage.CONFIG_FILE to """{"model": {"decoder": {"filename": "model.onnx"}}}""".toByteArray(),
      "model.onnx" to OnnxGraphBytes.graphNaming("model.onnx.data"),
      "tokenizer_config.json" to "{}".toByteArray(),
      "tokenizer.json" to "{}".toByteArray(),
      // Last, as the largest: what a cut transfer leaves unfinished.
      "model.onnx.data" to ByteArray(WEIGHTS) { (it % 251).toByte() },
    )

  private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  /** HuggingFace as two endpoints and a file host. Counts what it served, and can cut a download short or serve other bytes. */
  private inner class Hub(var commit: String = this@OnnxPackageFetcherTest.commit) : OnnxPackageFetcher.Http {
    val served = mutableMapOf<String, MutableList<Long>>()
    var cutAfter: Pair<String, Int>? = null
    var altered: String? = null
    var listed: Map<String, ByteArray> = files
    var extraListing: List<JSONObject> = emptyList()

    /** What `main` resolves to; any other name resolves to [commit], the one whose files this hub serves. */
    var mainCommit: String? = null
    val revisionsAsked = mutableListOf<String>()
    val revisionLookups: Int
      get() = revisionsAsked.size

    override fun get(url: String): String =
      when {
        "/revision/" in url -> {
          val named = url.substringAfter("/revision/")
          revisionsAsked += named
          JSONObject().put("sha", if (named == "main") mainCommit ?: commit else commit).toString()
        }
        "/tree/$commit/" in url -> {
          val listing = JSONArray()
          listing.put(JSONObject().put("type", "directory").put("path", "${ref.prefix}/sub").put("size", 0))
          listed.forEach { (name, bytes) ->
            val entry = JSONObject().put("type", "file").put("path", "${ref.prefix}/$name").put("size", bytes.size)
            if (name.endsWith(".data")) entry.put("lfs", JSONObject().put("oid", sha256(bytes)).put("size", bytes.size))
            listing.put(entry)
          }
          extraListing.forEach { listing.put(it) }
          listing.toString()
        }
        else -> throw IOException("HTTP 404 for $url")
      }

    override fun download(url: String, dest: File, offset: Long, onBytes: (Long) -> Unit): Long {
      val name = url.substringAfter("/resolve/$commit/${ref.prefix}/")
      val bytes = (if (name == altered) ByteArray(files.getValue(name).size) else files[name]) ?: throw IOException("HTTP 404 for $url")
      served.getOrPut(name) { mutableListOf() } += offset
      val stop = cutAfter?.takeIf { it.first == name }?.second ?: bytes.size
      dest.parentFile?.mkdirs()
      java.io.FileOutputStream(dest, offset > 0).use { it.write(bytes, offset.toInt(), stop - offset.toInt()) }
      onBytes(dest.length())
      if (stop < bytes.size) throw IOException("connection reset")
      return dest.length()
    }
  }

  private var published = 0
  private var clock = 0L

  private fun fetcher(hub: Hub, freeBytes: Long = Long.MAX_VALUE) =
    OnnxPackageFetcher(temp.root, hub, freeBytes = { freeBytes }, onPublished = { published += 1 }, now = { clock })

  private val dest: File
    get() = File(temp.root, ref.relativeDir)

  private fun stagings(): List<File> = dest.parentFile?.listFiles { file -> file.name.endsWith(OnnxPackageFetcher.STAGING_SUFFIX) }.orEmpty().toList()

  @Test
  fun `a package is fetched from one commit and published whole`() {
    val hub = Hub()
    val statuses = mutableListOf<String>()

    val path = fetcher(hub).ensure(ref) { statuses += it }.getOrThrow()

    assertEquals(dest.absolutePath, path)
    files.forEach { (name, bytes) -> assertTrue(name, File(dest, name).readBytes().contentEquals(bytes)) }
    OnnxGenaiPackage.validate(dest)
    assertTrue("nothing is left staged", stagings().isEmpty())
    assertEquals(files.keys, hub.served.keys)
    assertEquals("Resolving cpu-int4", statuses.first())
    assertTrue(statuses.last(), statuses.last().startsWith("Downloading cpu-int4 100%"))
  }

  /** Every cell after the first that names the same package: checked, and nothing fetched. */
  @Test
  fun `a package already on disk is checked and returned without the network`() {
    val hub = Hub()
    fetcher(hub).ensure(ref) {}.getOrThrow()
    hub.served.clear()
    hub.commit = "0000000000000000000000000000000000000000"

    assertEquals(dest.absolutePath, fetcher(hub).ensure(ref) {}.getOrThrow())
    assertTrue(hub.served.isEmpty())
  }

  /** One put there by hand and missing its weights is refused with the reason, not fetched over: the bytes are someone's. */
  @Test
  fun `a broken package on disk is refused rather than replaced`() {
    val hub = Hub()
    fetcher(hub).ensure(ref) {}.getOrThrow()
    File(dest, "model.onnx.data").delete()

    val reason = fetcher(hub).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("incomplete"))
  }

  /**
   * A transfer cut short leaves nothing in the models directory, and the next attempt continues the file from where it stopped instead of from its
   * start: the weights are the multi-gigabyte half.
   */
  @Test
  fun `an interrupted fetch publishes nothing and resumes`() {
    val hub = Hub()
    hub.cutAfter = "model.onnx.data" to WEIGHTS / 2

    val first = fetcher(hub).ensure(ref) {}

    assertTrue(first.isFailure)
    assertFalse("nothing is published", dest.exists())
    assertEquals(1, stagings().size)

    hub.cutAfter = null
    hub.served.clear()
    fetcher(hub).ensure(ref) {}.getOrThrow()

    assertEquals("only the unfinished file is asked for again", setOf("model.onnx.data"), hub.served.keys)
    assertEquals("and from where it stopped", listOf((WEIGHTS / 2).toLong()), hub.served["model.onnx.data"])
    assertTrue(File(dest, "model.onnx.data").readBytes().contentEquals(files.getValue("model.onnx.data")))
  }

  /** An LFS file is checked against the SHA-256 its listing declares, so bytes altered on the way are refused and not kept. */
  @Test
  fun `a file that does not match its digest is refused`() {
    val hub = Hub()
    hub.altered = "model.onnx.data"

    val reason = fetcher(hub).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("SHA-256"))
    assertFalse(dest.exists())
    assertFalse("the altered bytes are not kept to resume from", File(stagings().single(), "model.onnx.data.part").exists())
  }

  /** A repo pushed to between two attempts: the old commit's staging is a transfer nothing will resume, so it goes. */
  @Test
  fun `a staging directory of another commit is discarded`() {
    val hub = Hub()
    hub.cutAfter = "model.onnx.data" to WEIGHTS / 2
    fetcher(hub).ensure(ref) {}
    val stale = stagings().single()

    hub.cutAfter = null
    hub.commit = "1111111111111111111111111111111111111111"
    fetcher(hub).ensure(ref) {}.getOrThrow()

    assertFalse(stale.exists())
    assertTrue(stagings().isEmpty())
  }

  /** A pinned coordinate is fetched at its own commit whatever `main` has become, and asks HuggingFace nothing to learn a full SHA. */
  @Test
  fun `a pinned package is fetched at its revision`() {
    val hub = Hub()
    val pinned = ref.copy(revision = HfRevision.parse(commit))
    hub.mainCommit = "1111111111111111111111111111111111111111"

    fetcher(hub).ensure(pinned) {}.getOrThrow()

    assertEquals(commit, File(dest, OnnxPackageFetcher.REVISION_FILE).readText())
    assertEquals("no revision lookup for a full commit", 0, hub.revisionLookups)
    OnnxGenaiPackage.validate(dest)
  }

  /**
   * A package on disk satisfies a pin only if it was fetched at that commit. One from another, or one put there by hand with no record, is refused
   * with what to do about it: run as the pinned snapshot, it would file a result under bytes the caller did not name.
   */
  @Test
  fun `a package on disk from another snapshot does not satisfy a pin`() {
    val hub = Hub()
    fetcher(hub).ensure(ref) {}.getOrThrow()

    assertEquals(dest.absolutePath, fetcher(hub).ensure(ref.copy(revision = HfRevision.parse(commit))) {}.getOrThrow())

    val other = fetcher(hub).ensure(ref.copy(revision = HfRevision.parse("1111111111111111111111111111111111111111"))) {}
    assertTrue(other.exceptionOrNull()?.message.orEmpty(), other.exceptionOrNull()?.message.orEmpty().contains("pinned to 1111111111"))

    File(dest, OnnxPackageFetcher.REVISION_FILE).delete()
    val unrecorded = fetcher(hub).ensure(ref.copy(revision = HfRevision.parse(commit))) {}
    assertTrue(unrecorded.exceptionOrNull()?.message.orEmpty(), unrecorded.exceptionOrNull()?.message.orEmpty().contains("no record"))

    assertEquals("an unpinned coordinate takes what is there", dest.absolutePath, fetcher(hub).ensure(ref) {}.getOrThrow())
  }

  /** A tag or a branch is resolved to its commit, once, and that commit is what is fetched and recorded. */
  @Test
  fun `a named revision is resolved to its commit`() {
    val hub = Hub()
    hub.mainCommit = "1111111111111111111111111111111111111111"

    fetcher(hub).ensure(ref.copy(revision = HfRevision.parse("v1.0"))) {}.getOrThrow()

    assertEquals(listOf("v1.0"), hub.revisionsAsked)
    assertEquals(commit, File(dest, OnnxPackageFetcher.REVISION_FILE).readText())
  }

  /** The transformers.js case again, caught at the listing: nothing is downloaded to learn it. */
  @Test
  fun `a directory with no genai config is refused before any download`() {
    val hub = Hub()
    hub.listed = files - OnnxGenaiPackage.CONFIG_FILE

    val reason = fetcher(hub).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("transformers.js"))
    assertTrue(hub.served.isEmpty())
  }

  /** A listed path is server-authored and joined onto the staging directory. */
  @Test
  fun `a listed path that leaves the package is refused`() {
    val hub = Hub()
    hub.extraListing = listOf(JSONObject().put("type", "file").put("path", "${ref.prefix}/../../escape").put("size", 1))

    val reason = fetcher(hub).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("not a path inside the package"))
    assertTrue(hub.served.isEmpty())
  }

  @Test
  fun `a device without room is told before any download`() {
    val hub = Hub()

    val reason = fetcher(hub, freeBytes = 1_000L).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("MB free"))
    assertTrue(hub.served.isEmpty())
  }

  private companion object {
    const val WEIGHTS = 4096
  }

  /** The models directory changed behind whatever cached a listing of it, and the fetcher says so once, when the package is in place. */
  @Test
  fun `publishing a package is announced once`() {
    val hub = Hub()

    fetcher(hub).ensure(ref) {}.getOrThrow()
    assertEquals(1, published)

    fetcher(hub).ensure(ref) {}.getOrThrow()
    assertEquals("a package already there is not published again", 1, published)
  }

  /**
   * The status callback is the one hook a blocking transfer gives a cancel: the caller throws from it. The fetch stops, publishes nothing, and keeps
   * what it staged, so the run that resumes continues the file.
   */
  @Test
  fun `a cancel thrown from the status callback stops the fetch and keeps the staging`() {
    val hub = Hub()

    val result =
      fetcher(hub).ensure(ref) { status -> if (status.startsWith("Downloading")) throw java.util.concurrent.CancellationException("cancelled") }

    assertTrue(result.exceptionOrNull() is java.util.concurrent.CancellationException)
    assertFalse(dest.exists())
    assertEquals(0, published)
    assertEquals(1, stagings().size)

    hub.served.clear()
    fetcher(hub).ensure(ref) {}.getOrThrow()
    assertFalse("the file the cancel interrupted is not fetched again from its start", hub.served.containsKey(OnnxGenaiPackage.CONFIG_FILE))
  }

  /**
   * Every file arrived as listed and the whole is still not a package: no retry can finish it, so the staging goes. Kept, it would be bytes the model
   * list never shows and no delete names, refused again on every attempt.
   */
  @Test
  fun `a staged package that fails validation is discarded`() {
    val hub = Hub()
    hub.listed = files - "tokenizer.json"

    val reason = fetcher(hub).ensure(ref) {}.exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("tokenizer vocabulary"))
    assertFalse(dest.exists())
    assertTrue("nothing is left staged", stagings().isEmpty())
  }

  /** A staging directory nothing has written to for a week belongs to a run nobody repeated, and the next fetch of anything reclaims it. */
  @Test
  fun `an abandoned staging directory of another package is reclaimed`() {
    val abandoned = File(temp.root, "other/repo/.int4.0123456789ab${OnnxPackageFetcher.STAGING_SUFFIX}").apply { mkdirs() }
    File(abandoned, "model.onnx.data.part").writeText("half")
    val recent = File(temp.root, "other/repo/.fp16.0123456789ab${OnnxPackageFetcher.STAGING_SUFFIX}").apply { mkdirs() }
    File(recent, "model.onnx.data.part").writeText("half")
    val old = System.currentTimeMillis() - 2 * OnnxPackageFetcher.ABANDONED_AFTER_MILLIS
    abandoned.walkTopDown().forEach { it.setLastModified(old) }
    clock = System.currentTimeMillis()

    fetcher(Hub()).ensure(ref) {}.getOrThrow()

    assertFalse(abandoned.exists())
    assertTrue("one still within its week is another run's to resume", recent.exists())
  }

  /** Deleting a package takes what a fetch of it left staged, which nothing else would. */
  @Test
  fun `the stagings of a package are discarded with it`() {
    val hub = Hub()
    hub.cutAfter = "model.onnx.data" to WEIGHTS / 2
    fetcher(hub).ensure(ref) {}
    assertEquals(1, stagings().size)

    OnnxPackageFetcher.discardStagings(dest)

    assertTrue(stagings().isEmpty())
  }
}
