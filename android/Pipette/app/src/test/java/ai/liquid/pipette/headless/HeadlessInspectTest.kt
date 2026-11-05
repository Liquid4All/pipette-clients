package ai.liquid.pipette.headless

import ai.liquid.pipette.BenchmarkDefinition
import ai.liquid.pipette.BenchmarkId
import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.IdentityBundle
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStatus
import ai.liquid.pipette.ManagementClient
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.RegistrationData
import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The read-only verbs' output format.
 *
 * Asserted here rather than by reading logcat because the format *is* the interface: a CI step greps these lines, so a field that quietly changes
 * name or disappears breaks a consumer with no compile error anywhere. Every case below pins a field a script would reach for.
 */
class HeadlessInspectTest {
  private fun model(name: String, repo: String?, size: Long) = ModelFile(name = name, path = "/models/$name", sizeBytes = size, hfRepo = repo)

  /**
   * Definitions built here rather than read from `BenchmarkCatalog.all`.
   *
   * That catalog is process-global and empty until a sync populates it, so on the JVM it is always empty: a test written against it would assert
   * nothing while appearing to cover the listing. The device run is what confirms the catalog is loaded in the activity's process.
   */
  private val prefill = BenchmarkDefinition.PrefillThroughput(BenchmarkId.parse("prefill_throughput_100"), prefillTokens = 100)
  private val decode = BenchmarkDefinition.DecodeThroughput(BenchmarkId.parse("decode_throughput_512_100"), prefillTokens = 512, decodeTokens = 100)
  private val catalog = listOf(prefill, decode)

  @Suppress("LongParameterList") // one optional per field a verb can report; the alternative is a fixture per verb
  private fun facts(
    models: List<ModelFile> = emptyList(),
    manifests: List<JobManifest> = emptyList(),
    registered: Boolean = true,
    running: Boolean = false,
    runtime: HeadlessInspect.Runtime = HeadlessInspect.Runtime(id = "llama_cpp", available = true, build = "b4567"),
    results: List<HeadlessInspect.Result> = emptyList(),
    registration: RegistrationData? = null,
    privateKeyHex: PrivateKeyHex? = null,
    hfToken: String? = null,
    profile: ManagementClient.ClientProfile? = null,
    profileFailure: String? = null,
  ) =
    HeadlessInspect.Facts(
      buildVersion = "dev-gabc1234-debug",
      packageName = "ai.liquid.pipette.debug",
      engineFlavor = "rust",
      registered = registered,
      jobRunning = running,
      models = models,
      manifests = manifests,
      benchmarks = catalog,
      selectableIds = setOf(prefill.benchmarkId.toString()),
      modelsDirPath = "/sdcard/models",
      freeBytes = 633_000_000_000L,
      runtime = runtime,
      results = results,
      registration = registration,
      privateKeyHex = privateKeyHex,
      hfToken = hfToken,
      profile = profile,
      profileFailure = profileFailure,
    )

  /** An identity both halves of which are present, which is the state every auth verb answers from. */
  private fun identity(hfToken: String? = null) =
    facts(registration = HeadlessTestFixtures.registration(), privateKeyHex = HeadlessTestFixtures.privateKeyHex(), hfToken = hfToken)

  private fun clientProfile(tags: List<String> = listOf("lab", "pixel"), capabilities: List<String> = listOf("runtime:llama_cpp")) =
    ManagementClient.ClientProfile(
      clientId = "ev1_a3f8c2d1",
      // A space, which is the invariant this verb breaks and the reason the class KDoc records an exception for it.
      organization = "Liquid AI",
      clientDetails = "Pixel 10 Pro Fold",
      contactEmail = "ops@liquid.ai",
      status = "approved",
      tags = tags,
      reindexPending = false,
      capabilities = capabilities,
    )

  private fun result(
    cellId: String = "cell-a",
    jobId: String = "3f03f6e5",
    serverJobId: String? = null,
    metricValue: Double? = 305.331,
    metricUnit: String? = "tok/s",
    quant: String? = "q4_0",
  ) =
    HeadlessInspect.Result(
      jobId = jobId,
      cellId = cellId,
      benchmarkId = "prefill_throughput_100",
      benchmarkType = "prefill_throughput",
      modelName = "LiquidAI/LFM2.5-230M-GGUF",
      quant = quant,
      serverJobId = serverJobId,
      metricValue = metricValue,
      metricUnit = metricUnit,
      createdAt = "2026-08-12T00:00:00Z",
    )

  private fun report(
    verb: HeadlessVerb,
    type: String? = null,
    benchmarkId: String? = null,
    resultId: String? = null,
    facts: HeadlessInspect.Facts = facts(),
  ) = HeadlessInspect.report(HeadlessCommand.Query(verb, type, benchmarkId, resultId), facts)

  @Test
  fun `version reports the build a submission would carry`() {
    assertEquals(listOf("version build=dev-gabc1234-debug package=ai.liquid.pipette.debug flavor=rust"), report(HeadlessVerb.VERSION))
  }

  /** `running` is the field a poller waits on, since the surface refuses a second job. */
  @Test
  fun `status reports what a script checks before starting a run`() {
    val line = report(HeadlessVerb.STATUS, facts = facts(models = listOf(model("a.gguf", "org/repo", 1L)), running = true))!!.single()

    assertTrue(line, line.startsWith("status "))
    assertTrue(line, line.contains("registered=true"))
    assertTrue(line, line.contains("running=true"))
    assertTrue(line, line.contains("models=1"))
    assertTrue(line, line.contains("jobs=0"))
  }

  @Test
  fun `models lists each file and ends with a count`() {
    val lines =
      report(
        HeadlessVerb.MODELS,
        facts =
          facts(
            models =
              listOf(
                model("LFM2.5-230M-Q4_0.gguf", "LiquidAI/LFM2.5-230M-GGUF", 149_080_928L),
                model("mmproj-LFM2-VL-450M-f16.gguf", "LiquidAI/x", 5L),
              )
          ),
      )!!

    assertEquals("models count=2", lines.last())
    assertTrue(lines[0], lines[0].contains("name=LFM2.5-230M-Q4_0.gguf"))
    assertTrue(lines[0], lines[0].contains("repo=LiquidAI/LFM2.5-230M-GGUF"))
    assertTrue(lines[0], lines[0].contains("bytes=149080928"))
    assertTrue(lines[0], lines[0].contains("quant=Q4_0"))
    assertTrue("a projector is marked as one: ${lines[1]}", lines[1].contains("mmproj=true"))
  }

  /** An imported file has no repo, and `repo=` with nothing after it would read as a repo named the empty string. */
  @Test
  fun `a model with no repo omits the key rather than emitting it empty`() {
    val line = report(HeadlessVerb.MODELS, facts = facts(models = listOf(model("loose.gguf", null, 7L))))!!.first()

    assertTrue(line, !line.contains("repo="))
    assertTrue(line, line.contains("name=loose.gguf"))
  }

  /** An empty list still emits its count, so a consumer can tell "none" from "logcat dropped the rows". */
  @Test
  fun `an empty list still reports a count`() {
    assertEquals(listOf("models count=0"), report(HeadlessVerb.MODELS))
  }

  @Test
  fun `jobs reports per-status cell counts`() {
    val cells = listOf(CellRunStatus.COMPLETED, CellRunStatus.COMPLETED, CellRunStatus.PENDING)
    val lines = report(HeadlessVerb.JOBS, facts = facts(manifests = listOf(HeadlessTestFixtures.manifest("job-1", cells, JobStatus.RUNNING))))!!

    assertEquals("jobs count=1", lines.last())
    val line = lines.first()
    assertTrue(line, line.contains("id=job-1"))
    assertTrue(line, line.contains("status=running"))
    assertTrue(line, line.contains("cells=3"))
    // Lifecycle order (the enum's), not alphabetical: pending before completed reads as progress.
    assertTrue("the breakdown is what says whether it stalled: $line", line.contains("cell_status=pending=1,completed=2"))
  }

  @Test
  fun `benchmarks lists the catalog with a count`() {
    val lines = report(HeadlessVerb.BENCHMARKS)!!

    assertEquals("benchmarks count=2", lines.last())
    assertTrue("every line is a benchmark record", lines.dropLast(1).all { it.startsWith("benchmark id=") })
  }

  @Test
  fun `benchmarks narrows to a type`() {
    val lines = report(HeadlessVerb.BENCHMARKS, type = "prefill_throughput")!!

    assertEquals("benchmarks count=1", lines.last())
    assertTrue(lines[0], lines[0].contains("id=prefill_throughput_100"))
  }

  /** A filter matching nothing is a true answer; an id matching nothing is not. The two verbs differ deliberately. */
  @Test
  fun `an unknown type is an empty list while an unknown id is unanswerable`() {
    assertEquals(listOf("benchmarks count=0"), report(HeadlessVerb.BENCHMARKS, type = "no_such_type"))
    assertNull(report(HeadlessVerb.BENCHMARKS_SHOW, benchmarkId = "no_such_id"))
  }

  @Test
  fun `benchmarks-show reports the one benchmark`() {
    val line = report(HeadlessVerb.BENCHMARKS_SHOW, benchmarkId = "prefill_throughput_100")!!.single()

    assertTrue(line, line.startsWith("benchmark id=prefill_throughput_100 "))
    assertTrue("selectable says whether a run may name it: $line", line.contains("selectable=true"))
  }

  /** Selectability comes from the facts, not from the process-global catalog, so a hidden benchmark reports as hidden. */
  @Test
  fun `a benchmark outside the picker reports as not selectable`() {
    val line = report(HeadlessVerb.BENCHMARKS_SHOW, benchmarkId = "decode_throughput_512_100")!!.single()

    assertTrue(line, line.contains("selectable=false"))
  }

  @Test
  fun `storage-status reports the directory, usage and free space`() {
    val line =
      report(HeadlessVerb.STORAGE_STATUS, facts = facts(models = listOf(model("a.gguf", "o/r", 100L), model("b.gguf", "o/r", 200L))))!!.single()

    assertTrue(line, line.contains("models_dir=/sdcard/models"))
    assertTrue(line, line.contains("models=2"))
    assertTrue(line, line.contains("bytes=300"))
    assertTrue(line, line.contains("free_bytes=633000000000"))
  }

  /** Android has no quota, so the field is absent rather than reported as unlimited. Its absence is the accurate signal. */
  @Test
  fun `storage-status reports no quota because there is none`() {
    assertTrue(report(HeadlessVerb.STORAGE_STATUS)!!.single().let { !it.contains("quota") })
  }

  /** The run verb is routed to the run path by the activity and never reaches the reporter. */
  @Test
  fun `run is not an inspection`() {
    assertNull(report(HeadlessVerb.RUN))
  }

  // ── runtimes ────────────────────────────────────────────────────────────

  @Test
  fun `runtimes reports the one linked engine and its build`() {
    val lines = report(HeadlessVerb.RUNTIMES)!!

    assertEquals("runtime id=llama_cpp available=true build=b4567", lines[0])
    assertEquals("runtimes count=1", lines.last())
  }

  /**
   * A build whose native library did not load advertises no runtime capability and matches no runtime-gated job. The line has to show that, or a
   * fleet check would read a runtime as present and look elsewhere for why the device never claims work.
   */
  @Test
  fun `runtimes reports an unavailable native library rather than hiding it`() {
    val line =
      report(HeadlessVerb.RUNTIMES, facts = facts(runtime = HeadlessInspect.Runtime("llama_cpp", available = false, build = "native-unavailable")))!!
        .first()

    assertTrue(line, line.contains("available=false"))
    assertTrue("the sentinel is carried through rather than rendered as a build: $line", line.contains("build=native-unavailable"))
  }

  // ── results ─────────────────────────────────────────────────────────────

  @Test
  fun `results lists each completed cell and ends with a count`() {
    val lines = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(cellId = "cell-a"), result(cellId = "cell-b"))))!!

    assertEquals("results count=2", lines.last())
    assertTrue(lines[0], lines[0].contains("cell=cell-a"))
    assertTrue(lines[0], lines[0].contains("job=3f03f6e5"))
    assertTrue(lines[0], lines[0].contains("benchmark=prefill_throughput_100"))
    assertTrue(lines[0], lines[0].contains("metric=305.331"))
    assertTrue(lines[0], lines[0].contains("unit=tok/s"))
  }

  /** The whole point of the field for a fleet: whether this measurement reached the warehouse. */
  @Test
  fun `a result reports whether it was submitted, and names the row when it was`() {
    val submitted = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(serverJobId = "srv-77"))))!!.first()
    val local = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(serverJobId = null))))!!.first()

    assertTrue(submitted, submitted.contains("submitted=true"))
    assertTrue(submitted, submitted.contains("server_job=srv-77"))
    assertTrue(local, local.contains("submitted=false"))
    assertTrue("no row to name, so no key: $local", !local.contains("server_job="))
  }

  /**
   * A blank server job id is what an interrupted submission leaves behind, and it is not a submission. Reading it as one would tell a fleet the rows
   * are safely uploaded when nothing was.
   */
  @Test
  fun `a blank server job id does not count as submitted`() {
    val line = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(serverJobId = "   "))))!!.first()

    assertTrue(line, line.contains("submitted=false"))
    assertTrue("a blank id names no row: $line", !line.contains("server_job="))
  }

  /**
   * The defect a device run found and this file did not: `Double.toString()` switches to scientific notation above 10^7, so a real `max_memory_usage`
   * row read `metric=2.96300544E8`. Every fixture here used a throughput in the hundreds, which formats plainly, so the whole suite passed while the
   * largest-valued benchmark on the device was unreadable to `awk` and to shell arithmetic.
   */
  @Test
  fun `a large metric is a plain decimal, never scientific notation`() {
    val line = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(metricValue = 296_300_544.0, metricUnit = "bytes"))))!!.first()

    assertTrue("no exponent: $line", !line.contains("E") && !line.contains("e+"))
    assertTrue(line, line.contains("metric=296300544.000"))
  }

  /** Three places is `pipette-ops`' `{:.3}` and iOS's `%.3f`, so one measurement compares byte for byte across all three clients. */
  @Test
  fun `a metric is rendered at the three places every client prints`() {
    val line = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(metricValue = 1133.8333172412151))))!!.first()

    assertTrue(line, line.contains("metric=1133.833"))
  }

  /**
   * `String.format` without an explicit locale follows the device's, so a phone set to German would emit `296300544,000` and split the value across
   * the line's own separator. Pinned by forcing the default locale, since that is the only way this regresses.
   */
  @Test
  fun `a metric keeps its period on a device whose separator is a comma`() {
    val original = Locale.getDefault()
    try {
      Locale.setDefault(Locale.GERMANY)
      val line = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(metricValue = 1133.8333172412151))))!!.first()

      assertTrue("a comma would break the key=value split: $line", line.contains("metric=1133.833"))
    } finally {
      Locale.setDefault(original)
    }
  }

  /** `metric=` with nothing after it would read as a measurement of zero, so the pair is omitted together. */
  @Test
  fun `a cell whose payload had no scalar still appears, without the metric keys`() {
    val lines = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(metricValue = null, metricUnit = null))))!!

    assertEquals("the row is still counted: $lines", "results count=1", lines.last())
    assertTrue(lines[0], !lines[0].contains("metric="))
    assertTrue(lines[0], !lines[0].contains("unit="))
  }

  /** `quantLabel` answers an unparseable filename with the literal "unknown"; the activity maps that to an absent key, so nothing here emits it. */
  @Test
  fun `a result with no derivable quant omits the key`() {
    val line = report(HeadlessVerb.RESULTS, facts = facts(results = listOf(result(quant = null))))!!.first()

    assertTrue(line, !line.contains("quant="))
  }

  @Test
  fun `results-show returns the one result named`() {
    val lines = report(HeadlessVerb.RESULTS_SHOW, resultId = "cell-b", facts = facts(results = listOf(result("cell-a"), result("cell-b"))))!!

    assertEquals(1, lines.size)
    assertTrue(lines[0], lines[0].contains("cell=cell-b"))
  }

  /** A cell id that is not here is a refusal rather than an empty listing, so a script that asked about one result can branch on the answer. */
  @Test
  fun `results-show refuses an id with no measurement on disk`() {
    assertNull(report(HeadlessVerb.RESULTS_SHOW, resultId = "cell-missing", facts = facts(results = listOf(result("cell-a")))))
  }

  /** No results is a true answer to a listing, unlike an id that does not exist. */
  @Test
  fun `results on a device that has run nothing is a count of zero, not a refusal`() {
    assertEquals(listOf("results count=0"), report(HeadlessVerb.RESULTS))
  }

  // ── the refusal a `-show` verb gets ─────────────────────────────────────

  /**
   * The refusal has to point at the listing verb, or a script author cannot tell a mistyped id from a device that has none of them. Both are things
   * they would do next, and they are different fixes.
   */
  @Test
  fun `a not-found refusal names the id and where the real ones are listed`() {
    val result = HeadlessInspect.notFoundReason(HeadlessCommand.Query(HeadlessVerb.RESULTS_SHOW, resultId = "cell-missing"), facts())
    val benchmark = HeadlessInspect.notFoundReason(HeadlessCommand.Query(HeadlessVerb.BENCHMARKS_SHOW, benchmarkId = "no_such_benchmark"), facts())

    assertTrue(result, result.contains("cell-missing") && result.contains("verb=results"))
    assertTrue(benchmark, benchmark.contains("no_such_benchmark") && benchmark.contains("verb=benchmarks"))
  }

  /**
   * A verb that cannot reach the refusal still gets a sane line rather than a wrong specific one.
   *
   * Worth pinning because the `else` arm is unreachable through the activity today: if a future verb starts returning null from `report`, this is
   * what its caller will log, and it should not claim a benchmark id was missing.
   */
  @Test
  fun `a verb with no id does not borrow another verb's reason`() {
    val message = HeadlessInspect.notFoundReason(HeadlessCommand.Query(HeadlessVerb.MODELS), facts())

    assertTrue(message, message.contains("models"))
    assertTrue("does not invent a benchmark: $message", !message.contains("benchmark"))
  }

  // ── the identity verbs ──────────────────────────────────────────────────

  /** The two fields an operator needs to approve this client, and no third one — in particular, not the signing key. */
  @Test
  fun `auth-whoami reports the client id and a derived public key`() {
    val line = report(HeadlessVerb.AUTH_WHOAMI, facts = identity())!!.single()

    val publicKey = IdentityBundle.publicKeyHexOf(HeadlessTestFixtures.privateKeyHex())
    assertEquals("auth-whoami client_id=ev1_a3f8c2d1 public_key=$publicKey", line)
    assertEquals("a public key is 32 bytes of hex", 64, publicKey!!.length)
    assertTrue("the private key must not appear", !line.contains(HeadlessTestFixtures.privateKeyHex().value))
  }

  /**
   * `auth-me`'s eight fields, in iOS's order, with the two rules this verb breaks visible in the assertion.
   *
   * The whole line rather than field probes: it is the only line on this surface whose *order* is a contract, because a value can hold a space and a
   * consumer therefore has to read it positionally.
   */
  @Test
  fun `auth-me reports the eight profile fields in iOS's order`() {
    val line = report(HeadlessVerb.AUTH_ME, facts = facts(profile = clientProfile()))!!.single()

    assertEquals(
      "auth-me client_id=ev1_a3f8c2d1 organization=Liquid AI client_details=Pixel 10 Pro Fold contact_email=ops@liquid.ai " +
        "status=approved tags=lab,pixel reindex_pending=false capabilities=runtime:llama_cpp",
      line,
    )
  }

  /**
   * `tags=` and `capabilities=` are emitted empty rather than dropped, unlike every other key on this surface.
   *
   * iOS's choice, and the reason is that a parser reading the profile off either client sees the key either way, so "this client has no tags" stays
   * distinguishable from "this build does not report tags".
   */
  @Test
  fun `auth-me emits empty tags and capabilities rather than omitting them`() {
    val line = report(HeadlessVerb.AUTH_ME, facts = facts(profile = clientProfile(tags = emptyList(), capabilities = emptyList())))!!.single()

    assertTrue(line, line.contains(" tags= "))
    assertTrue(line, line.endsWith(" capabilities="))
  }

  /**
   * `auth-export` prints a summary and then the bundle alone, in that order.
   *
   * The order is load-bearing rather than cosmetic: `HeadlessRunActivity.logAuthEvent` mirrors the *first* line into the audit stream, so the summary
   * going first is what keeps the signing key out of a second log. An operator also lifts the token with one `sed`, which needs it alone on its line.
   */
  @Test
  fun `auth-export prints the summary first and the bundle alone second`() {
    val lines = report(HeadlessVerb.AUTH_EXPORT, facts = identity(hfToken = "hf_user"))!!

    val publicKey = IdentityBundle.publicKeyHexOf(HeadlessTestFixtures.privateKeyHex())
    val token = lines.last().removePrefix("auth-export bundle=")
    assertEquals(2, lines.size)
    assertEquals(
      "auth-export client_id=ev1_a3f8c2d1 server_url=https://pipette.liquid.ai public_key=$publicKey " +
        "device_labels=no settings=no hf_tokens=1 bytes=${token.length}",
      lines.first(),
    )
    assertTrue("the summary carries no key", !lines.first().contains(HeadlessTestFixtures.privateKeyHex().value))
    assertTrue("the bundle line is the token and nothing else", lines.last().startsWith("auth-export bundle=") && !token.contains(' '))
  }

  /** The token the export prints is the one an import reads, which is the property the two verbs are only useful in pairs for. */
  @Test
  fun `the exported token decodes back to this device's identity`() {
    val lines = report(HeadlessVerb.AUTH_EXPORT, facts = identity(hfToken = "hf_user"))!!

    val decoded = IdentityBundle.validated(lines.last().removePrefix("auth-export bundle="))
    assertTrue("expected a bundle, got $decoded", decoded is IdentityBundle.Result.Ok)
    val bundle = (decoded as IdentityBundle.Result.Ok).bundle
    assertEquals(HeadlessTestFixtures.registration(), bundle.registration)
    assertEquals(HeadlessTestFixtures.privateKeyHex(), bundle.privateKeyHex)
    assertEquals("hf_user", bundle.userToken)
  }

  /** A device holding no token exports a bundle carrying none, so `hf_tokens=` counts what could actually be restored. */
  @Test
  fun `auth-export counts no token on a device that holds none`() {
    val summary = report(HeadlessVerb.AUTH_EXPORT, facts = identity())!!.first()

    assertTrue(summary, summary.contains(" hf_tokens=0 "))
  }

  /**
   * Both halves of the identity are required, and the refusal names the one that is missing.
   *
   * Registration first, matching iOS's `mgmtSession`: a device that never registered is the ordinary case with a one-step fix, and a registration
   * with no key is a fault, since the device reports itself registered and can sign nothing.
   */
  @Test
  fun `the identity verbs refuse a half identity and name the missing half`() {
    for (verb in listOf(HeadlessVerb.AUTH_WHOAMI, HeadlessVerb.AUTH_EXPORT)) {
      val noRegistration = facts(privateKeyHex = HeadlessTestFixtures.privateKeyHex())
      val noKey = facts(registration = HeadlessTestFixtures.registration())

      assertNull("$verb must refuse without a registration", report(verb, facts = noRegistration))
      assertNull("$verb must refuse without a key", report(verb, facts = noKey))

      val forNoRegistration = HeadlessInspect.notFoundReason(HeadlessCommand.Query(verb), noRegistration)
      val forNoKey = HeadlessInspect.notFoundReason(HeadlessCommand.Query(verb), noKey)
      assertTrue(forNoRegistration, forNoRegistration.contains("no registration"))
      assertTrue(forNoKey, forNoKey.contains("no signing key") && forNoKey.contains("ev1_a3f8c2d1"))
    }
  }

  /**
   * `auth-me` refuses with the fetch's own reason when there was one.
   *
   * The distinction the test is for: by the time a fetch failed, both halves of the identity were present, so falling back to the identity reason
   * would describe a device that is fine and send the operator to fix nothing.
   */
  @Test
  fun `auth-me refuses with the network failure rather than with an identity reason`() {
    val failed =
      facts(
        registration = HeadlessTestFixtures.registration(),
        privateKeyHex = HeadlessTestFixtures.privateKeyHex(),
        profileFailure = "Management server returned HTTP 401",
      )

    assertNull(report(HeadlessVerb.AUTH_ME, facts = failed))
    assertEquals("Management server returned HTTP 401", HeadlessInspect.notFoundReason(HeadlessCommand.Query(HeadlessVerb.AUTH_ME), failed))
  }

  /** With no registration there is nothing to ask the server about, so `auth-me` refuses before the network is reached at all. */
  @Test
  fun `auth-me refuses an unregistered device with the identity reason`() {
    val reason = HeadlessInspect.notFoundReason(HeadlessCommand.Query(HeadlessVerb.AUTH_ME), facts())

    assertTrue(reason, reason.contains("no registration"))
  }
}
