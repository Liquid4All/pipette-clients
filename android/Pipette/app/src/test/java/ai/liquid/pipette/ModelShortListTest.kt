package ai.liquid.pipette

import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * [ModelShortList]: which catalog members become rows, and when an answer may publish. Robolectric supplies [JSONObject] and `Log`.
 *
 * [ModelShortList] is process-wide, so each test starts and ends with a [ModelShortList.clear].
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class ModelShortListTest {
  @get:Rule val temp = TemporaryFolder()

  private val cacheFile by lazy { File(temp.root, "catalog/short-list.json") }
  private val registration = registration("client-1")

  private fun registration(clientId: String) =
    RegistrationData(
      clientId = clientId,
      status = "approved",
      serverUrl = "https://pipette.example",
      organization = "example",
      contactEmail = "user@example.com",
      registeredAt = "2026-11-19T00:00:00Z",
    )

  private fun gguf(path: String, revision: String? = "abc123", type: String = "gguf_text") =
    JSONObject().put("type", type).put("source", "huggingface").put("org", "LiquidAI").put("repo_name", "LFM2.5-350M-GGUF").put("path", path).apply {
      revision?.let { put("revision", it) }
    }

  private fun member(descriptor: JSONObject?, sizeBytes: Long? = 229_312_224L, label: String = "q4_k_m") =
    JSONObject()
      .put("selector", "lfm2.5-350m::gguf::$label")
      .put("member_key", JSONObject().put("kind", "model").put("format", "gguf").put("quantization", label))
      .put("size_bytes", sizeBytes ?: JSONObject.NULL)
      .apply { descriptor?.let { put("descriptor", it) } }

  private fun shortList(
    vararg members: JSONObject,
    kind: String = "model",
    name: String = "lfm2.5-350m",
    family: Any? = null,
    logo: Any? = null,
  ): String =
    JSONObject()
      .put(
        "items",
        JSONArray()
          .put(
            JSONObject().put("name", name).put("descriptor_kind", kind).put("members", JSONArray(members.toList())).apply {
              family?.let { put("family", it) }
              logo?.let { put("family_logo_sha256", it) }
            }
          ),
      )
      .toString()

  private fun refresh(body: String, current: () -> RegistrationData? = { registration }): Int? =
    ModelShortList.refresh(cacheFile, registration, current, ramBytes = RAM) { body }

  @Before
  @After
  fun reset() {
    ModelShortList.clear(cacheFile)
  }

  @Test
  fun `a pinned text member becomes a row that downloads the pin`() {
    val row = ModelShortList.parse(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf"))), RAM).single()

    assertEquals("lfm2.5-350m", row.name)
    assertEquals("lfm2.5-350m::gguf::q4_k_m", row.id)
    assertEquals(229_312_224L, row.sizeBytes)
    assertEquals("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/abc123/LFM2.5-350M-Q4_K_M.gguf", row.parsed.url)
    assertEquals("LiquidAI/LFM2.5-350M-GGUF@abc123/LFM2.5-350M-Q4_K_M.gguf", row.downloadKey)
    assertEquals("Q4_K_M", row.quant)
  }

  /** An entry from an older server has no family, so the row takes the text of its name before the first hyphen. */
  @Test
  fun `a row takes the family of its entry, or else the start of its name`() {
    val cases =
      listOf(
        Triple("the catalog family", "LFM", "LFM"),
        Triple("a blank family", "  ", "lfm2.5"),
        Triple("no family", null, "lfm2.5"),
        Triple("an explicit null family", JSONObject.NULL, "lfm2.5"),
        Triple("a numeric family", 42, "lfm2.5"),
      )
    cases.forEach { (case, family, expected) ->
      val row = ModelShortList.parse(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")), family = family), RAM).single()
      assertEquals(case, expected, row.family)
    }
  }

  @Test
  fun `a row keeps the logo digest of its family and drops any other value`() {
    val digest = "c2e7d21098f9131392a4dc3789f3b8de9561b38e3ad627389872d24a5ca4fa02"
    val cases =
      listOf(
        Triple("a digest", digest, digest),
        Triple("an upper-case digest", digest.uppercase(), null),
        Triple("a short digest", digest.drop(1), null),
        Triple("a path", "../$digest", null),
        Triple("no logo", null, null),
        Triple("an explicit null logo", JSONObject.NULL, null),
      )
    cases.forEach { (case, logo, expected) ->
      val row = ModelShortList.parse(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")), logo = logo), RAM).single()
      assertEquals(case, expected, row.familyLogoDigest)
    }
  }

  /** Each member decodes on its own, so a member this app cannot run costs that member and nothing else. */
  @Test
  fun `members this app cannot download or load are dropped one at a time`() {
    val good = member(gguf("LFM2.5-350M-Q4_K_M.gguf"))
    val cases =
      listOf(
        "a vision member" to member(gguf("LFM2.5-350M-Q4_K_M.gguf", type = "gguf_vision")),
        "an MLX member" to member(gguf("LFM2.5-350M-Q4_K_M.gguf", type = "mlx")),
        "a revision with a slash" to member(gguf("LFM2.5-350M-Q4_K_M.gguf", revision = "refs/pr/3")),
        "a path in a subdirectory" to member(gguf("sub/LFM2.5-350M-Q4_K_M.gguf")),
        "no descriptor" to member(null),
        "a file larger than the RAM" to member(gguf("LFM2.5-350M-F16.gguf"), sizeBytes = RAM + 1),
      )
    cases.forEach { (case, bad) ->
      assertEquals(case, listOf(good.getString("selector")), ModelShortList.parse(shortList(bad, good), RAM).map { it.id })
    }
  }

  @Test
  fun `a quant the entry offers more than once gets a variant per member`() {
    val rows =
      ModelShortList.parse(
        shortList(
          member(gguf("LFM2.5-350M-Q4_0.gguf"), label = "q4_0"),
          member(gguf("LFM2.5-350M-Q4_K_M.gguf", revision = "aaa111"), label = "q4_k_m@bartowski"),
          member(gguf("LFM2.5-350M-Q4_K_M.gguf", revision = "bbb222"), label = "q4_k_m@xhtoken"),
          member(gguf("LFM2.5-350M-Q8_0.gguf", revision = "ccc333"), label = "q8_0"),
          member(gguf("LFM2.5-350M-Q8_0.gguf", revision = "ddd444"), label = "q8_0@20260922"),
        ),
        RAM,
      )

    assertEquals(listOf(null, "bartowski", "xhtoken", null, "20260922"), rows.map { it.variant })
    assertEquals("the display name stays the entry name", setOf("lfm2.5-350m"), rows.map { it.name }.toSet())
  }

  /** A repeat that the RAM filter removes leaves one member, which then needs no variant to tell it apart. */
  @Test
  fun `a quant left once after the RAM filter gets no variant`() {
    val rows =
      ModelShortList.parse(
        shortList(
          member(gguf("LFM2.5-350M-Q4_K_M.gguf", revision = "aaa111"), label = "q4_k_m@bartowski"),
          member(gguf("LFM2.5-350M-Q4_K_M.gguf", revision = "bbb222"), sizeBytes = RAM + 1, label = "q4_k_m@xhtoken"),
        ),
        RAM,
      )

    assertNull(rows.single().variant)
  }

  /** The file name does not always carry the quant in the form the app parses, so the catalog's label decides. */
  @Test
  fun `a row takes its quant from the catalog label`() {
    val cases = listOf("Spark-X2.6-1.7B.Q4_K_M.gguf" to "q4_k_m@mradermacher", "Ternary-Bonsai-2-4B-PQ2_0.gguf" to "pq2_0")
    cases.forEach { (path, label) ->
      val row = ModelShortList.parse(shortList(member(gguf(path), label = label)), RAM).single()
      assertEquals(path, label.substringBefore('@').uppercase(), row.quant)
    }
  }

  @Test
  fun `a member with no recorded size has an unknown size`() {
    val row = ModelShortList.parse(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf"), sizeBytes = null)), RAM).single()

    assertNull(row.sizeBytes)
    assertEquals("", row.sizeLabel)
  }

  @Test
  fun `runtime entries are not rows`() {
    assertTrue(ModelShortList.parse(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")), kind = "runtime"), RAM).isEmpty())
  }

  @Test
  fun `a refresh publishes the list and the cache restores it for the same registration only`() {
    assertEquals(1, refresh(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")))))
    assertEquals(1, ModelTemplateCatalog.current.size)

    ModelShortList.loadCache(cacheFile, registration, RAM)
    assertEquals(1, ModelShortList.offered.size)

    ModelShortList.loadCache(cacheFile, registration("client-2"), RAM)
    assertTrue(ModelShortList.offered.isEmpty())
    assertEquals(ModelTemplateCatalog.fallback, ModelTemplateCatalog.current)
  }

  /** The server offers no model, so the app shows the fallback, and the next launch does too rather than an older cached list. */
  @Test
  fun `an answer with no models replaces the cache`() {
    refresh(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf"))))
    assertEquals(0, refresh(shortList()))

    ModelShortList.loadCache(cacheFile, registration, RAM)
    assertEquals(ModelTemplateCatalog.fallback, ModelTemplateCatalog.current)
  }

  @Test
  fun `offline fallback hides rows larger than device RAM`() {
    val ramBytes = 2_000_000_000L
    try {
      ModelTemplateCatalog.limitFallbackTo(ramBytes)

      val offered = ModelTemplateCatalog.current
      assertTrue(offered.isNotEmpty())
      assertTrue(offered.size < ModelTemplateCatalog.fallback.size)
      assertTrue(offered.all { (it.sizeBytes ?: 0L) <= ramBytes })
    } finally {
      ModelTemplateCatalog.limitFallbackTo(RAM)
    }
  }

  @Test
  fun `an answer for a registration that has since changed does not publish`() {
    assertNull(refresh(shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")))) { registration("client-2") })
    assertTrue(ModelShortList.offered.isEmpty())
    assertFalse(cacheFile.exists())
  }

  @Test
  fun `an answer that a clear overtook does not publish`() {
    val result =
      ModelShortList.refresh(cacheFile, registration, { registration }, RAM) {
        ModelShortList.clear(cacheFile)
        shortList(member(gguf("LFM2.5-350M-Q4_K_M.gguf")))
      }

    assertNull(result)
    assertTrue(ModelShortList.offered.isEmpty())
  }

  /** Two catalog entries can name one file at one revision. One download, sized by the larger figure so the warning does not undercount. */
  @Test
  fun `selected rows for one file download once at the larger size`() {
    val one = PresetModel("a", "A", "LiquidAI/LFM2.5-350M-GGUF@abc123/LFM2.5-350M-Q4_K_M.gguf", 100L)
    val alias = one.copy(id = "b", name = "B", sizeBytes = 200L)
    val other = one.copy(id = "c", identifier = "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf")

    assertEquals(listOf("b", "c"), ModelTemplateCatalog.downloads(listOf(one, alias, other)).map { it.id })
  }

  private companion object {
    const val RAM = 12_000_000_000L
  }
}
