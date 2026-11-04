package ai.liquid.pipette.headless

import ai.liquid.pipette.ModelUri
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the headless argument grammar (PIP-437). The Android counterpart to iOS `HeadlessCommandTests`.
 *
 * Most of these assert a *refusal*. That is the point of the surface: a headless run is started by a script nobody is watching, so an argument the
 * parser quietly reinterprets produces a confidently wrong measurement rather than a visible failure.
 */
class HeadlessCommandTest {

  private fun ok(args: Map<String, String>): HeadlessCommand.Request {
    val parsed = HeadlessCommand.parse(args)
    assertTrue("expected a parse, got $parsed", parsed is HeadlessCommand.Parsed.Ok)
    return (parsed as HeadlessCommand.Parsed.Ok).request
  }

  private fun usageError(args: Map<String, String>): String {
    val parsed = HeadlessCommand.parse(args)
    assertTrue("expected a usage error, got $parsed", parsed is HeadlessCommand.Parsed.UsageError)
    return (parsed as HeadlessCommand.Parsed.UsageError).message
  }

  @Test
  fun readinessKnobsAreUnsetWhenTheCallerNamesNeither() {
    // Null, not the client default: an unset knob has to stay distinguishable from one set to the value the client would have picked anyway, because
    // only the first defers to the device's own Settings toggle for the waiver.
    val request = ok(emptyMap())
    assertEquals(null, request.readinessMaxWaitSecs)
    assertEquals(null, request.readinessSkipThermal)
  }

  @Test
  fun parsesTheReadinessKnobs() {
    val request = ok(mapOf("readiness_max_wait_secs" to "1800", "readiness_skip_thermal" to "1"))
    assertEquals(1800L, request.readinessMaxWaitSecs)
    assertEquals(true, request.readinessSkipThermal)
  }

  @Test
  fun refusesAZeroReadinessDeadline() {
    // Zero would expire the gate before its first poll and fail every cell, reporting a device that never cooled when nothing was ever waited for.
    assertTrue(usageError(mapOf("readiness_max_wait_secs" to "0")).contains("readiness_max_wait_secs"))
  }

  @Test
  fun refusesAReadinessDeadlinePastTheCeiling() {
    // Unbounded seconds would overflow the seconds→millis conversion into a negative deadline, which expires before the first poll and reports a
    // device that never cooled. Refusing at the boundary is how the caller learns they mistyped.
    assertTrue(usageError(mapOf("readiness_max_wait_secs" to "99999999")).contains("readiness_max_wait_secs"))
  }

  @Test
  fun refusesANonNumericReadinessDeadline() {
    assertTrue(usageError(mapOf("readiness_max_wait_secs" to "soon")).contains("readiness_max_wait_secs"))
  }

  @Test
  fun refusesANonBinaryThermalWaiver() {
    assertTrue(usageError(mapOf("readiness_skip_thermal" to "yes")).contains("readiness_skip_thermal"))
  }

  @Test
  fun emptyInvocationRunsEverythingAndSubmits() {
    val request = ok(emptyMap())

    // Submitting is the default, matching iOS: an automated run that records nothing is rarely what
    // was meant, and the caller who wants that says so with submit=0.
    assertTrue("submit must default to true", request.submit)
    assertEquals(null, request.matchQuery)
    assertEquals(emptyList<String>(), request.benchmarkIds)
    assertEquals(HeadlessCommand.DEFAULT_N_GPU_LAYERS, request.nGpuLayers)
    assertEquals(HeadlessCommand.DEFAULT_PREFILL_BATCH, request.prefillBatch)
  }

  @Test
  fun parsesAFullInvocation() {
    val request =
      ok(mapOf("match" to "LFM2.5-350M", "benchmark" to "prefill_512,decode_128", "submit" to "0", "batch" to "256", "number_gpu_layers" to "0"))

    assertEquals("LFM2.5-350M", request.matchQuery)
    assertEquals(listOf("prefill_512", "decode_128"), request.benchmarkIds)
    assertEquals(false, request.submit)
    assertEquals(256, request.prefillBatch)
    assertEquals(0, request.nGpuLayers)
  }

  /**
   * Every argument is spelled the way the CLI and `pipette-plan-types` spell it, so one runbook covers all three clients. Pinned by name because a
   * rename here silently breaks the CI scripts this surface exists for.
   */
  @Test
  fun argumentNamesMatchTheCliAndPlanTypes() {
    assertEquals(0, ok(mapOf("number_gpu_layers" to "0")).nGpuLayers)
    assertEquals(256, ok(mapOf("batch" to "256")).prefillBatch)
    assertEquals("LFM2.5-350M", ok(mapOf("match" to "LFM2.5-350M")).matchQuery)
  }

  /** `benchmark=` is the CLI's spelling, `benchmarks=` the plural alias iOS also keeps. Either resolves to the same ids. */
  @Test
  fun benchmarkAndBenchmarksAreTheSameArgument() {
    assertEquals(listOf("prefill_512"), ok(mapOf("benchmark" to "prefill_512")).benchmarkIds)
    assertEquals(listOf("prefill_512"), ok(mapOf("benchmarks" to "prefill_512")).benchmarkIds)
  }

  /** Both at once would usually disagree, and silently picking one is how a script measures a set nobody asked for. */
  @Test
  fun benchmarkTogetherWithBenchmarksIsRefused() {
    val message = usageError(mapOf("benchmark" to "prefill_512", "benchmarks" to "decode_128"))

    assertTrue("says they are the same argument, got: $message", message.contains("pass one"))
  }

  /** A renamed key is refused by name with a pointer, not swept up in the generic unknown-argument message. */
  @Test
  fun renamedKeysPointAtTheirReplacement() {
    assertTrue(usageError(mapOf("ngl" to "99")).contains("number_gpu_layers"))
  }

  /**
   * `model=` takes the compact coordinate URI, the CLI's own `--model` spelling.
   *
   * It used to be refused outright as a key this surface "cannot resolve". A cell can carry a coordinate now, so it resolves.
   */
  @Test
  fun modelTakesACoordinate() {
    val request = ok(mapOf("model" to "gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf"))
    val text = (request.modelUri as ai.liquid.pipette.Model.GgufText).model

    assertEquals("LiquidAI/LFM2.5-230M-GGUF", text.repo.toString())
    assertEquals("LFM2.5-230M-Q4_0.gguf", text.filename.toString())
    assertEquals(null, request.matchQuery)
  }

  /** A bare name is still answered by pointing at `match=`, which is where substring selection lives. The grammar changed; that advice did not. */
  @Test
  fun aBareNameInModelPointsAtMatch() {
    assertTrue(usageError(mapOf("model" to "LFM2.5-350M")).contains("match="))
  }

  /**
   * The two selectors ask different questions, so both at once is a contradiction rather than a narrowing. iOS refuses the pair for the same reason,
   * having previously carried substring selection on `model=` and found that it left a plan-supplied coordinate silently matching nothing.
   */
  @Test
  fun modelAndMatchAreRefusedTogether() {
    val message = usageError(mapOf("model" to "gguf-text://repo=a/b&path=c.gguf", "match" to "LFM"))

    assertTrue("names both keys, got: $message", message.contains("model") && message.contains("match"))
    assertTrue("says to pass one, got: $message", message.contains("pass one"))
  }

  /**
   * Submission flips to opt-in when a coordinate is named. Nothing sanctioned that model, so an ad-hoc experiment must not quietly publish warehouse
   * rows; a `match=` run keeps the opt-out default because the models it can select were all obtained deliberately. Mirrors iOS.
   */
  @Test
  fun anExactCoordinateDoesNotSubmitUnlessAsked() {
    val coordinate = mapOf("model" to "gguf-text://repo=a/b&path=c.gguf")

    assertTrue("a match run still submits by default", ok(mapOf("match" to "LFM")).submit)
    assertTrue("a coordinate must not submit by default", !ok(coordinate).submit)
    assertTrue("and says so explicitly when asked", ok(coordinate + ("submit" to "1")).submit)
  }

  private fun inspect(args: Map<String, String>): HeadlessCommand.Query {
    val parsed = HeadlessCommand.parse(args)
    assertTrue("expected an inspection, got $parsed", parsed is HeadlessCommand.Parsed.Inspect)
    return (parsed as HeadlessCommand.Parsed.Inspect).query
  }

  /**
   * The whole point of defaulting the verb: every invocation written before verbs existed still means what it meant.
   *
   * This surface is driven by scripts nobody is watching, so a silent change of meaning is the worst available outcome.
   */
  @Test
  fun anInvocationWithNoVerbIsStillARun() {
    val parsed = HeadlessCommand.parse(mapOf("match" to "LFM"))

    assertTrue("no verb must still parse as a run, got $parsed", parsed is HeadlessCommand.Parsed.Ok)
    assertEquals("LFM", (parsed as HeadlessCommand.Parsed.Ok).request.matchQuery)
    assertEquals("and verb=run is the same thing", "LFM", ok(mapOf("verb" to "run", "match" to "LFM")).matchQuery)
  }

  /**
   * The argument a verb refuses to run without, or nothing.
   *
   * Supplying it here rather than excluding the verb keeps [readOnlyVerbsParseToAQuery] exhaustive: a new `-show` verb that forgot to declare its id
   * fails this rather than quietly joining an exclusion list.
   */
  private fun requiredArg(verb: HeadlessVerb): Map<String, String> =
    when (verb) {
      HeadlessVerb.BENCHMARKS_SHOW -> mapOf("benchmark" to "prefill_throughput_100")
      HeadlessVerb.RESULTS_SHOW,
      HeadlessVerb.RESULTS_DELETE -> mapOf("result" to "cell-a")
      HeadlessVerb.JOB_RM,
      HeadlessVerb.JOB_EXPORT -> mapOf("id" to "job-a")
      HeadlessVerb.MODELS_DELETE -> mapOf("model" to "gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf")
      HeadlessVerb.MODELS_RM -> mapOf("name" to "LFM2.5-230M-Q4_0.gguf")
      HeadlessVerb.AUTH_IMPORT -> mapOf("bundle" to "eyJ2ZXJzaW9uIjoxfQ")
      else -> emptyMap()
    }

  @Test
  fun readOnlyVerbsParseToAQuery() {
    for (verb in HeadlessVerb.entries.filter { it.kind == HeadlessVerb.Kind.READ }) {
      assertEquals(verb, inspect(mapOf("verb" to verb.spelling) + requiredArg(verb)).verb)
    }
  }

  /**
   * Every mutating verb parses, and parses to [HeadlessCommand.Parsed.Mutate] rather than to an inspection.
   *
   * Exhaustive over `kind` for the same reason the read-only case above is: a verb added with `Kind.MUTATE` and no case in [requiredArg] falls to its
   * `else`, is handed no arguments, and fails here unless the verb genuinely needs none. What a *declared* requirement buys is the refusal below,
   * `aMutatingVerbWithoutItsTargetIsAUsageError`, which is where the empty invocation is pinned.
   */
  @Test
  fun mutatingVerbsParseToAMutation() {
    for (verb in HeadlessVerb.entries.filter { it.mutates }) {
      val parsed = HeadlessCommand.parse(mapOf("verb" to verb.spelling) + requiredArg(verb))
      assertTrue("$verb parsed as $parsed", parsed is HeadlessCommand.Parsed.Mutate)
      assertEquals(verb, (parsed as HeadlessCommand.Parsed.Mutate).query.verb)
    }
  }

  /** The `-show` verbs' rule, applied to the deletes: naming nothing is a usage error, not a listing and not a mass delete. */
  @Test
  fun aMutatingVerbWithoutItsTargetIsAUsageError() {
    for (verb in HeadlessVerb.entries.filter { it.mutates }) {
      val message = usageError(mapOf("verb" to verb.spelling))
      assertTrue("$verb should refuse an invocation naming nothing, got: $message", message.contains(verb.spelling))
    }
  }

  /**
   * `models-rm` takes either selector, so its rule cannot be the one-required-key map the others use.
   *
   * The message says what it will not do, because the dangerous reading of `models-rm` with no arguments is "remove models" and a caller who meant
   * that should have to say which.
   */
  @Test
  fun modelsRmNeedsAtLeastOneSelector() {
    val message = usageError(mapOf("verb" to "models-rm"))

    assertTrue("offers both selectors, got: $message", message.contains("name=") && message.contains("repo="))
    assertTrue("says what it will not do, got: $message", message.contains("every model"))
  }

  /** Either selector alone is enough, and both together is the narrowing form. */
  @Test
  fun modelsRmAcceptsEitherSelectorOrBoth() {
    for (args in listOf(mapOf("name" to "a.gguf"), mapOf("repo" to "org/repo"), mapOf("name" to "a.gguf", "repo" to "org/repo"))) {
      val parsed = HeadlessCommand.parse(mapOf("verb" to "models-rm") + args)
      assertTrue("$args parsed as $parsed", parsed is HeadlessCommand.Parsed.Mutate)
    }
  }

  /** `models-delete` takes the same coordinate grammar `run` does, so a URI it cannot parse is refused with the crate's reason. */
  @Test
  fun modelsDeleteRefusesACoordinateItCannotParse() {
    val message = usageError(mapOf("verb" to "models-delete", "model" to "LFM2.5-230M"))

    assertTrue("names the key, got: $message", message.contains("model"))
    // Against the grammar's own reason rather than against words the surrounding sentence already contains: "coordinate" is in the constant prefix,
    // so asserting on it would pass even if the reason were dropped, which is the only part of this message the caller cannot guess.
    val reason = (ModelUri.parse("LFM2.5-230M") as ModelUri.Result.Error).failure.reason
    assertTrue("carries the grammar's reason '$reason', got: $message", message.contains(reason))
  }

  @Test
  fun anUnknownVerbListsTheKnownOnes() {
    val message = usageError(mapOf("verb" to "nonsense"))

    assertTrue("names the bad verb, got: $message", message.contains("nonsense"))
    assertTrue("lists the real ones, got: $message", message.contains("models") && message.contains("version"))
  }

  /**
   * Keys are checked against the selected verb, not a union of every verb's.
   *
   * A union would accept this and silently ignore the batch, which is the quiet misreading the rest of the grammar exists to refuse.
   */
  @Test
  fun aVerbRefusesAnArgumentItCannotUse() {
    val message = usageError(mapOf("verb" to "version", "batch" to "256"))

    assertTrue("names the verb, got: $message", message.contains("version"))
    assertTrue("names the offending key, got: $message", message.contains("batch"))
    assertTrue("says the verb takes nothing, got: $message", message.contains("no arguments"))
  }

  /** And a run-only key is not silently accepted by an inspection either way round. */
  @Test
  fun anInspectionRefusesRunKeys() {
    assertTrue(usageError(mapOf("verb" to "models", "match" to "LFM")).contains("models"))
    assertTrue(usageError(mapOf("verb" to "jobs", "submit" to "1")).contains("jobs"))
  }

  @Test
  fun benchmarksTakesAnOptionalTypeFilter() {
    assertEquals("prefill_throughput", inspect(mapOf("verb" to "benchmarks", "type" to "prefill_throughput")).type)
    assertEquals(null, inspect(mapOf("verb" to "benchmarks")).type)
  }

  /** A caller who wrote `benchmarks-show` and forgot the id asked about one benchmark; answering with all of them would not be that question. */
  @Test
  fun benchmarksShowRequiresAnId() {
    assertTrue(usageError(mapOf("verb" to "benchmarks-show")).contains("benchmark="))
    assertEquals("prefill_throughput_100", inspect(mapOf("verb" to "benchmarks-show", "benchmark" to "prefill_throughput_100")).benchmarkId)
  }

  /** Same rule, same reason, for the other `-show` verb. */
  @Test
  fun resultsShowRequiresACellId() {
    assertTrue(usageError(mapOf("verb" to "results-show")).contains("result="))
    assertEquals("cell-a", inspect(mapOf("verb" to "results-show", "result" to "cell-a")).resultId)
  }

  /**
   * `result=` belongs to the two result verbs.
   *
   * The plain listing takes no filter, so accepting one there would silently list everything for a caller who thought they had narrowed it, which is
   * the class of quiet misreading the per-verb key sets exist to refuse.
   */
  @Test
  fun theResultsListingTakesNoFilter() {
    val message = usageError(mapOf("verb" to "results", "result" to "cell-a"))

    assertTrue("names the verb, got: $message", message.contains("results"))
    assertTrue("names the offending key, got: $message", message.contains("result"))
  }

  /** A present-but-blank id reads as an unexpanded shell variable, not as a request for everything. */
  @Test
  fun aBlankShowIdIsRefused() {
    assertTrue(usageError(mapOf("verb" to "results-show", "result" to "  ")).contains("must not be blank"))
  }

  /** The refusal comes from `ModelUri`, so the reason a caller reads is the crate's reason rather than a second opinion invented here. */
  @Test
  fun anUnrunnableCoordinateIsRefusedWithItsReason() {
    assertTrue(usageError(mapOf("model" to "mlx://repo=mlx-community/Qwen3.5-0.8B-4bit")).contains("does not link"))
    assertTrue(usageError(mapOf("model" to "gguf-text://repo=a/b&path=c.gguf&sha256=abc")).contains("digest"))
  }

  /**
   * Neither spelling of the context knob is accepted, matching iOS. Accepting it would stamp a number on the manifest that the measurement then
   * ignores, since `JobRunner.cellContextSize` computes the window from the benchmark.
   */
  @Test
  fun contextSizeIsRefusedRatherThanAcceptedAndDropped() {
    for (key in listOf("ctx", "ctx_size")) {
      val message = usageError(mapOf(key to "2048"))
      assertTrue("$key must explain itself, got: $message", message.contains("computed from the benchmark"))
    }
  }

  @Test
  fun unknownArgumentIsRefusedRatherThanIgnored() {
    // The failure this catches is a typo in a CI script. Ignoring `sumbit` would run a submitting
    // benchmark for a caller who asked for the opposite and report success.
    val message = usageError(mapOf("sumbit" to "0"))

    assertTrue("names the offending key, got: $message", message.contains("sumbit"))
    assertTrue("lists what is accepted, got: $message", message.contains("submit"))
  }

  @Test
  fun submitAcceptsOnlyZeroOrOne() {
    // Deliberately stricter than pipette_readiness's truthy-string grammar, which reads an env var a
    // human exports. Accepting `true` here while rejecting `yes` would invite a caller to guess.
    assertEquals(false, ok(mapOf("submit" to "0")).submit)
    assertEquals(true, ok(mapOf("submit" to "1")).submit)
    for (raw in listOf("true", "yes", "on", "", " ", "2", "-1")) {
      assertTrue("submit='$raw' must be refused", HeadlessCommand.parse(mapOf("submit" to raw)) is HeadlessCommand.Parsed.UsageError)
    }
  }

  @Test
  fun numericArgumentsRejectNonsense() {
    for (raw in listOf("0", "-1", "abc", "", "1.5")) {
      assertTrue("batch='$raw' must be refused", HeadlessCommand.parse(mapOf("batch" to raw)) is HeadlessCommand.Parsed.UsageError)
    }
    // number_gpu_layers=0 is meaningful (CPU only), unlike a zero batch, so it is the one that parses.
    assertEquals(0, ok(mapOf("number_gpu_layers" to "0")).nGpuLayers)
    for (raw in listOf("-1", "abc", "")) {
      assertTrue(
        "number_gpu_layers='$raw' must be refused",
        HeadlessCommand.parse(mapOf("number_gpu_layers" to raw)) is HeadlessCommand.Parsed.UsageError,
      )
    }
  }

  @Test
  fun presentButBlankSelectorsAreRefused() {
    // `-e match ""` reads as "I meant to name a model", not "run every model". Defaulting a blank to
    // the everything case would run the whole catalog for a script whose variable failed to expand.
    assertTrue(HeadlessCommand.parse(mapOf("match" to "  ")) is HeadlessCommand.Parsed.UsageError)
    assertTrue(HeadlessCommand.parse(mapOf("benchmark" to " , ")) is HeadlessCommand.Parsed.UsageError)
    assertTrue(HeadlessCommand.parse(mapOf("benchmarks" to " , ")) is HeadlessCommand.Parsed.UsageError)
  }

  @Test
  fun toleratesWhitespaceAroundValues() {
    val request = ok(mapOf("match" to "  LFM2.5-350M  ", "benchmark" to "prefill_512, decode_128", "batch" to " 256 "))

    assertEquals("LFM2.5-350M", request.matchQuery)
    assertEquals(listOf("prefill_512", "decode_128"), request.benchmarkIds)
    assertEquals(256, request.prefillBatch)
  }

  /** The usage line has to name a value the caller can produce, and `bundle=<id>` describes something that does not exist. */
  @Test
  fun authImportWithoutABundleSaysWhereTheValueComesFrom() {
    val message = usageError(mapOf("verb" to "auth-import"))

    assertTrue("names the key, got: $message", message.contains("bundle="))
    assertTrue("says where the value comes from, got: $message", message.contains("auth-export"))
  }

  /**
   * A present-but-blank `bundle=` is a usage error rather than a bundle that fails to decode.
   *
   * `-e bundle ""` is an unexpanded shell variable, and reporting it as "not base64url" would send the caller looking at the token they copied rather
   * than at the line that dropped it.
   */
  @Test
  fun aBlankBundleIsAUsageErrorRatherThanADecodeFailure() {
    val message = usageError(mapOf("verb" to "auth-import", "bundle" to "  "))

    assertTrue("names the key, got: $message", message.contains("bundle"))
    assertTrue("says it was blank, got: $message", message.contains("blank"))
  }

  /** `force=` is the same strict `0`/`1` grammar every flag on this surface takes; anything else is a typo worth reporting. */
  @Test
  fun forceIsStrictlyZeroOrOne() {
    assertEquals(false, inspectOrMutate(mapOf("verb" to "auth-import", "bundle" to "abc")).force)
    assertEquals(false, inspectOrMutate(mapOf("verb" to "auth-import", "bundle" to "abc", "force" to "0")).force)
    assertEquals(true, inspectOrMutate(mapOf("verb" to "auth-import", "bundle" to "abc", "force" to "1")).force)

    val message = usageError(mapOf("verb" to "auth-import", "bundle" to "abc", "force" to "yes"))
    assertTrue("must not guess at 'yes', got: $message", message.contains("force must be 0 or 1"))
  }

  /** The bundle reaches resolution as the caller wrote it, trimmed, so nothing here can mangle a token it does not understand. */
  @Test
  fun theBundleIsCarriedThroughVerbatim() {
    assertEquals("eyJ2ZXJzaW9uIjoxfQ", inspectOrMutate(mapOf("verb" to "auth-import", "bundle" to "  eyJ2ZXJzaW9uIjoxfQ  ")).bundle)
  }

  /** The identity verbs that only read take no arguments at all, so a `bundle=` aimed at one of them is a usage error. */
  @Test
  fun theReadOnlyIdentityVerbsTakeNoArguments() {
    for (verb in listOf("auth-whoami", "auth-me", "auth-export")) {
      val message = usageError(mapOf("verb" to verb, "bundle" to "abc"))
      assertTrue("$verb should refuse bundle=, got: $message", message.contains(verb) && message.contains("no arguments"))
    }
  }

  /** The query behind a well-formed invocation, whichever side of the read/mutate line the verb falls on. */
  private fun inspectOrMutate(args: Map<String, String>): HeadlessCommand.Query =
    when (val parsed = HeadlessCommand.parse(args)) {
      is HeadlessCommand.Parsed.Mutate -> parsed.query
      is HeadlessCommand.Parsed.Inspect -> parsed.query
      else -> error("expected a query, got $parsed")
    }
}
