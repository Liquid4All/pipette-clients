package ai.liquid.pipette

import org.json.JSONObject

/**
 * Serializes the `model_descriptor` / `runtime_descriptor` / `runtime_flags` / `benchmark_flags` submission specs: the full, lossless typed spec for
 * the model and runtime that produced a run, plus the harness configuration it ran under.
 *
 * `model_descriptor` / `runtime_descriptor` travel on the wire as opaque JSON **strings** (not nested objects). The management server never
 * interprets the schema — it only canonicalizes each string (object keys sorted, whitespace stripped) before storing, and rejects a
 * present-but-invalid descriptor with `400`. Shapes mirror the reshaped `pipette-plan-types::{Model, Runtime}` families (PIP-340) exactly, so a
 * stored descriptor round-trips into the warehouse `model_descriptor` / `runtime_descriptor` columns. This is the Android counterpart to iOS
 * `SubmissionRef.swift`; both clients emit the same canonical (sorted-key, compact) form.
 */
object SubmissionRef {
  /**
   * The llama.cpp build target the Android pipette app runs on — the single member of the plan-types `LlamacppApkPipetteFlavor` enum
   * (`#[serde(rename_all = "kebab-case")]`), so the wire value is `android-arm64-v8` (the enum variant name kebab-cased, NOT the `arm64-v8a` NDK ABI
   * spelling — the type family dropped the trailing `a`).
   */
  const val ANDROID_FLAVOR = "android-arm64-v8"

  /**
   * The canonical upstream llama.cpp repo, scheme-less per the plan-types `RepositoryUrl` sanitizer — matches
   * `pipette_plan_types::default_repository_url` and iOS `SubmissionRef.llamaCppRepositoryUrl`.
   */
  const val LLAMACPP_REPOSITORY_URL = "github.com/ggml-org/llama.cpp"

  /**
   * `model_descriptor`: the model coordinate as the plan-types `Model` descriptor, reconstructed from the run's [modelName] (the HF `org/repo_name`
   * slug), on-disk [modelFilename], and optional [mmprojFilename] (present ⇒ a VL model). Returns null when the coordinate is incomplete — an
   * imported file with no HF slug, or a filename that isn't a valid `*.gguf` — so a source-less legacy cell elides the field rather than sending an
   * invalid descriptor the server would reject with `400` (mirrors iOS).
   */
  /**
   * The typed [Model] for the run's coordinate, or null when it can't be formed (no HF slug, or a non-`*.gguf` model/projector filename).
   *
   * Public rather than an internal step of a descriptor helper because the caller needs the model itself, not just its rendering: `model_descriptor`
   * and `model_flags` are two fields off one coordinate, and building it twice would let them disagree.
   */
  fun typedModelOrNull(modelName: String, modelFilename: String, mmprojFilename: String?): Model? {
    val repo = HfRepo.parseSlugOrNull(modelName)
    val filename = GgufFilename.parseOrNull(modelFilename)
    if (repo == null || filename == null) return null
    return if (mmprojFilename == null) {
      Model.GgufText(HfGgufText(repo, filename))
    } else {
      GgufFilename.parseOrNull(mmprojFilename)?.let { mmproj -> Model.GgufVision(HfGgufVision(repo, filename, mmproj)) }
    }
  }

  /** `model_descriptor` for an already-typed [Model]. Emits the tagged plan-types shape (`type` + `source` sub-tag + the flattened HF coordinate). */
  fun model(model: Model): String = canonical(modelObject(model))

  /**
   * `runtime_descriptor`: the engine identity as the plan-types `Runtime::LlamacppApkPipette` descriptor — the in-process llama.cpp surface the app
   * runs. Carries the flattened `SourceRepository` (`repository_url` + `repository_version`, the latter the same lossless string submitted as
   * `runtime_version`) plus the [ANDROID_FLAVOR] target. Runtime load knobs are NOT part of the descriptor — they ship separately as `runtime_flags`
   * (see [runtimeFlags]).
   */
  fun runtime(version: String): String =
    canonical(
      JSONObject()
        .put("type", "llamacpp_apk_pipette")
        .put("repository_url", LLAMACPP_REPOSITORY_URL)
        .put("repository_version", version)
        .put("flavor", ANDROID_FLAVOR)
    )

  /**
   * `runtime_flags`: the llama.cpp load knobs as the `llama-bench`/`llama-cli` CLI-flag string, the cheap grouping/display companion the server
   * stores verbatim (never interpreted) alongside `runtime_descriptor`. Byte-identical to the iOS client (`PayloadBuilder.swift`: `-ngl <n> -c
   * <ctx>`) so the warehouse `runtime_flags` column stays consistent across platforms. `n_ubatch` is intentionally not recorded here (iOS omits it
   * too).
   */
  fun runtimeFlags(nGpuLayers: Int, contextSize: Int): String = "-ngl $nGpuLayers -c $contextSize"

  /**
   * `benchmark_flags`: what the *harness* ran under, as opposed to `runtime_flags` (how the engine was loaded). Canonical JSON matching
   * `pipette_plan_types::BenchmarkFlags::submission_value`, which for a timing cell is a single `readiness` block.
   *
   * Resolved, not authored. Readiness is decided entirely client-side, so this is the only record of it the server ever sees: without it a waived
   * thermal gate is invisible in a result and a reader has to infer the gate state from the very numbers they are trying to interpret.
   *
   * Returns null for the two cell kinds whose flag variants carry no `readiness` field, matching the partition in `pipette-cli/src/run.rs`
   * (`resolved_flags`): **eval**, whose variant has no field to hold one, and **max_memory_usage**, which has no variant at all. Android's
   * between-cell gate does run before both, unlike the CLI's, so the omission means "nothing to report in this schema", not "no wait happened"; the
   * per-rep thermal telemetry still records the state each rep started at. Note the two are not identical upstream either: an eval cell has a
   * variant, so the CLI emits whichever of its knobs were authored (`{}` when none were), while max-memory has none and is omitted outright. So only
   * max-memory is a true match. The divergence is unobservable for now, since `BenchmarkCatalog.SELECTABLE_TYPES` offers no eval cell to run.
   *
   * Caveat worth knowing before extending this: `BenchmarkFlags` models no `llamacpp_apk_pipette` variant at all yet, so `try_from` would answer
   * `NoSuchCombination` for *every* Android cell, not just the two elided here. What this emits is shape-compatible with the schema without being
   * typed by it. Adding the mobile variants is the same change for iOS and belongs with PIP-429.
   */
  fun benchmarkFlagsOrNull(benchmarkType: BenchmarkType, policy: ReadinessPolicy): String? {
    val gates =
      when (benchmarkType) {
        BenchmarkType.PREFILL_THROUGHPUT,
        BenchmarkType.DECODE_THROUGHPUT,
        BenchmarkType.END_TO_END_LATENCY,
        BenchmarkType.VL_THROUGHPUT -> true
        BenchmarkType.EVAL,
        BenchmarkType.MAX_MEMORY_USAGE -> false
      }
    if (!gates) return null
    val readiness = JSONObject().put("max_wait_secs", policy.maxWaitSecs).put("skip_thermal", policy.skipThermal)
    return canonical(JSONObject().put("readiness", readiness))
  }

  private fun modelObject(model: Model): JSONObject {
    val obj = JSONObject()
    when (model) {
      is Model.GgufText -> {
        obj.put("type", "gguf_text")
        obj.put("source", "huggingface")
        putRepo(obj, model.model.repo)
        obj.put("path", model.model.filename.value)
      }
      is Model.GgufVision -> {
        obj.put("type", "gguf_vision")
        obj.put("source", "huggingface")
        putRepo(obj, model.model.repo)
        obj.put("model", model.model.filename.value)
        obj.put("mmproj", model.model.mmprojFilename.value)
      }
    }
    return obj
  }

  /**
   * The flattened plan-types `HfRepo`: `org` + `repo_name`. Optional `revision` / `auth_token` are `skip_serializing_if`, and the slug-reconstructed
   * repo carries neither, so neither is emitted (matches iOS).
   */
  private fun putRepo(obj: JSONObject, repo: HfRepo) {
    obj.put("org", repo.org.value)
    obj.put("repo_name", repo.repoName.value)
  }

  /**
   * Canonical JSON: object keys sorted, compact (no whitespace). The server canonicalizes anyway, but emitting the canonical form here keeps the
   * stored value byte-identical to what we send, makes the encoding deterministic for tests, and matches iOS's `.sortedKeys` output. `org.json`'s own
   * key ordering is unspecified on the JVM test classpath, so this cannot rely on `JSONObject.toString()`.
   */
  private fun canonical(value: Any?): String =
    when (value) {
      is JSONObject ->
        value.keys().asSequence().sorted().joinToString(separator = ",", prefix = "{", postfix = "}") { key ->
          "${quoteJson(key)}:${canonical(value.get(key))}"
        }
      is String -> quoteJson(value)
      is Boolean,
      is Int,
      is Long,
      is Double -> value.toString()
      else -> quoteJson(value.toString())
    }

  /**
   * RFC 8259 JSON string quoting: escapes `"`, `\`, and control chars — but NOT `/`. Android's framework `org.json` (and thus Robolectric) escapes
   * `/` → `\/`, while the JVM `org.json`, Swift's `JSONEncoder`, and the plan-types serde emit a bare `/`; using our own escaper keeps the on-device
   * `repository_url` (`github.com/ggml-org/llama.cpp`) byte-identical to iOS and to the JVM unit-test output instead of drifting per-runtime.
   */
  private fun quoteJson(value: String): String {
    val sb = StringBuilder(value.length + 2)
    sb.append('"')
    for (c in value) {
      when (c) {
        '"' -> sb.append("\\\"")
        '\\' -> sb.append("\\\\")
        '\n' -> sb.append("\\n")
        '\r' -> sb.append("\\r")
        '\t' -> sb.append("\\t")
        '\b' -> sb.append("\\b")
        '\u000C' -> sb.append("\\f")
        else -> if (c < '\u0020') sb.append("\\u%04x".format(c.code)) else sb.append(c)
      }
    }
    sb.append('"')
    return sb.toString()
  }
}
