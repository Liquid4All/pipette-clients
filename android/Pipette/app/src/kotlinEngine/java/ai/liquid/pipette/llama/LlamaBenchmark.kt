package ai.liquid.pipette.llama

import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.LlamacppLoadMode
import ai.liquid.pipette.ReadinessOutcome
import ai.liquid.pipette.ops.MeasurementObserver
import ai.liquid.pipette.ops.PromptSeed
import ai.liquid.pipette.ops.REPS
import ai.liquid.pipette.ops.Rep
import ai.liquid.pipette.ops.Stats
import ai.liquid.pipette.ops.meanStddev
import ai.liquid.pipette.ops.runMeasurement
import ai.liquid.pipette.thermal.AndroidThermalStatus
import ai.liquid.pipette.thermal.androidThermalStatusFromCode
import android.util.Log
import java.util.Locale
import java.util.concurrent.CancellationException
import kotlin.math.max
import kotlin.math.min
import org.json.JSONArray
import org.json.JSONObject

/** Pure-Kotlin llama.cpp benchmark methodology kernel. Mirrors android/native/benchmarks.rs. */
object LlamaBenchmark {
  private const val TAG = "pipette-llama"
  private const val MEDIA_MARKER = "<__media__>"
  private const val VL_NUM_MEASUREMENT_RUNS = 5
  const val EVAL_PROMPT_BUDGET = 8192

  data class BenchmarkDef(
    val benchmarkId: String,
    val benchmarkType: String,
    val parameterPrefillTokens: Int? = null,
    val parameterDecodeTokens: Int? = null,
    val parameterMaxTokens: Int? = null,
    val parameterMcqChoices: List<String>? = null,
    val samples: JSONArray? = null,
    val parameterImageWidth: Int? = null,
    val parameterImageHeight: Int? = null,
    val parameterTextTokens: Int? = null,
    val parameterNumImages: Int? = null,
    /**
     * Requests the entry keeps in flight at once, on the kinds that carry it. Absent means the reference procedure, one at a time, which is all this
     * engine runs: see [checkBatchSize].
     */
    val parameterBatchSize: Int? = null,
  )

  fun parseBenchmarkDef(jsonStr: String): BenchmarkDef {
    val obj = JSONObject(jsonStr)
    val benchmarkId = obj.getString("benchmark_id")
    val benchmarkType = obj.getString("benchmark_type")
    val prefill =
      if (obj.has("parameter_prefill_tokens") && !obj.isNull("parameter_prefill_tokens")) {
        obj.getInt("parameter_prefill_tokens")
      } else {
        null
      }
    val decode =
      if (obj.has("parameter_decode_tokens") && !obj.isNull("parameter_decode_tokens")) {
        obj.getInt("parameter_decode_tokens")
      } else {
        null
      }
    val maxTokens =
      if (obj.has("parameter_max_tokens") && !obj.isNull("parameter_max_tokens")) {
        obj.getInt("parameter_max_tokens")
      } else {
        null
      }
    val mcqChoices =
      if (obj.has("parameter_mcq_choices") && !obj.isNull("parameter_mcq_choices")) {
        val arr = obj.getJSONArray("parameter_mcq_choices")
        (0 until arr.length()).map { arr.getString(it) }
      } else {
        null
      }
    val samples = if (obj.has("samples") && !obj.isNull("samples")) obj.getJSONArray("samples") else null
    val imageWidth =
      if (obj.has("parameter_image_width") && !obj.isNull("parameter_image_width")) {
        obj.getInt("parameter_image_width")
      } else {
        null
      }
    val imageHeight =
      if (obj.has("parameter_image_height") && !obj.isNull("parameter_image_height")) {
        obj.getInt("parameter_image_height")
      } else {
        null
      }
    val textTokens =
      if (obj.has("parameter_text_tokens") && !obj.isNull("parameter_text_tokens")) {
        obj.getInt("parameter_text_tokens")
      } else {
        null
      }
    val numImages =
      if (obj.has("parameter_num_images") && !obj.isNull("parameter_num_images")) {
        obj.getInt("parameter_num_images")
      } else {
        null
      }
    val batchSize =
      if (obj.has("parameter_batch_size") && !obj.isNull("parameter_batch_size")) {
        obj.getInt("parameter_batch_size")
      } else {
        null
      }

    return BenchmarkDef(
      benchmarkId = benchmarkId,
      benchmarkType = benchmarkType,
      parameterPrefillTokens = prefill,
      parameterDecodeTokens = decode,
      parameterMaxTokens = maxTokens,
      parameterMcqChoices = mcqChoices,
      samples = samples,
      parameterImageWidth = imageWidth,
      parameterImageHeight = imageHeight,
      parameterTextTokens = textTokens,
      parameterNumImages = numImages,
      parameterBatchSize = batchSize,
    )
  }

  fun requiredCtxSize(def: BenchmarkDef): Int? {
    val prefill = def.parameterPrefillTokens ?: 512
    val decode = def.parameterDecodeTokens ?: 128
    val maxTokens = def.parameterMaxTokens ?: 256
    return when (def.benchmarkType) {
      "prefill_throughput" -> prefill
      "max_memory_usage" -> prefill + 1
      "decode_throughput",
      "end_to_end_latency" -> prefill + decode
      "vl_throughput" -> {
        val w = def.parameterImageWidth ?: 0
        val h = def.parameterImageHeight ?: 0
        val imageTokens = (w / 14) * (h / 14)
        val text = def.parameterTextTokens ?: 0
        imageTokens + text + decode
      }
      "eval",
      // No per-sample image resolution is available to calibrate a tighter bound (unlike vl_throughput, whose
      // width/height are the benchmark's own parameters) — reuse eval's text budget, generous for the vision-tower
      // token counts real eval images produce.
      "vlm_eval" -> EVAL_PROMPT_BUDGET + maxTokens
      else -> null
    }
  }

  fun checkCtxSize(def: BenchmarkDef, contextSize: Int) {
    val required = requiredCtxSize(def) ?: return
    if (contextSize < required) {
      error(
        "context_size $contextSize is too small for benchmark '${def.benchmarkId}' " +
          "(needs at least $required; prefill=${def.parameterPrefillTokens ?: 512}, " +
          "decode=${def.parameterDecodeTokens ?: 128}, max_tokens=${def.parameterMaxTokens ?: 256}). " +
          "Increase Context Size in the Job settings."
      )
    }
  }

  /**
   * Refuse an entry that wants more than one request in flight.
   *
   * This engine decodes one sequence through one `llama_context`: there is no second slot to put a request on, and running a batched entry serially
   * would file the reference measurement under an id (`_bs<n>`) that promises the measurement under load. Checked beside [checkCtxSize], before the
   * weights are read, because no amount of waiting makes it runnable. Mirrors `android/native/benchmarks.rs`'s `check_batch_size`.
   */
  fun checkBatchSize(def: BenchmarkDef) {
    val batchSize = def.parameterBatchSize ?: 1
    if (batchSize < 1) error("parameter_batch_size must be at least 1, got $batchSize")
    if (batchSize != 1) {
      error(
        "benchmark '${def.benchmarkId}' keeps $batchSize requests in flight, and this engine " +
          "decodes one sequence at a time; a batched ${def.benchmarkType} entry needs a server " +
          "with that many slots: vLLM or SGLang, or llama-server for end_to_end_latency"
      )
    }
  }

  fun readinessGate(readiness: BenchmarkCooldownCallback?) {
    if (readiness == null) return
    val outcome = readiness.waitUntilReady() ?: return
    if (outcome.startsWith(ReadinessOutcome.CANCELLED_PREFIX)) {
      throw CancellationException("cancelled during thermal cooldown")
    }
    val observed = outcome.removePrefix(ReadinessOutcome.TIMED_OUT_PREFIX)
    error("device failed to cool within the time budget: last seen $observed")
  }

  data class ThermalReading(val androidThermalHeadroom: Float? = null, val androidThermalStatus: AndroidThermalStatus? = null)

  fun sampleThermal(thermal: BenchmarkThermalCallback?): ThermalReading? {
    if (thermal == null) return null
    val headroom = runCatching { thermal.sampleHeadroom() }.getOrNull()?.takeIf { it.isFinite() }
    val status = runCatching { thermal.sampleStatus() }.getOrNull()?.let { androidThermalStatusFromCode(it) }
    return ThermalReading(headroom, status)
  }

  fun mergeThermal(result: JSONObject, before: List<ThermalReading>, after: List<ThermalReading>) {
    if (before.isNotEmpty() && before.all { it.androidThermalHeadroom != null }) {
      val arr = JSONArray()
      before.forEach { arr.put(it.androidThermalHeadroom!!.toDouble()) }
      result.put("device_android_thermal_headroom_before", arr)
    }
    if (after.isNotEmpty() && after.all { it.androidThermalHeadroom != null }) {
      val arr = JSONArray()
      after.forEach { arr.put(it.androidThermalHeadroom!!.toDouble()) }
      result.put("device_android_thermal_headroom_after", arr)
    }
    if (before.isNotEmpty() && before.all { it.androidThermalStatus != null }) {
      val arr = JSONArray()
      before.forEach { arr.put(it.androidThermalStatus!!.wireValue) }
      result.put("device_android_thermal_status_before", arr)
    }
    if (after.isNotEmpty() && after.all { it.androidThermalStatus != null }) {
      val arr = JSONArray()
      after.forEach { arr.put(it.androidThermalStatus!!.wireValue) }
      result.put("device_android_thermal_status_after", arr)
    }
  }

  fun reportMeasurement(progress: BenchmarkProgressCallback?, idx: Int, total: Int) {
    if (progress == null) return
    val n = idx + 1
    if (!progress.onProgress(n, total, "Measurement $n/$total")) {
      throw CancellationException("cancelled at measurement $n/$total")
    }
  }

  fun seedPromptText(model: LlamaModel, targetTokens: Int, contextSize: Int, addSpecial: Boolean): String {
    val target = min(targetTokens, contextSize)
    return PromptSeed.buildPromptText(target) { text -> LlamaCpp.tokenize(model, text, addSpecial).size }
  }

  fun seedPromptTokens(model: LlamaModel, targetTokens: Int, contextSize: Int, addSpecial: Boolean): IntArray {
    val target = min(targetTokens, contextSize)
    val text = seedPromptText(model, targetTokens, contextSize, addSpecial)
    val tokens = LlamaCpp.tokenize(model, text, addSpecial)
    val keepLen = max(target, if (tokens.isNotEmpty()) 1 else 0)
    return if (tokens.size > keepLen) {
      tokens.copyOf(keepLen)
    } else {
      tokens
    }
  }

  data class MeasuredCell(val stats: Stats, val thermal: List<Pair<ThermalReading, ThermalReading>>)

  fun <T> measureCell(
    label: String,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
    prepare: () -> Unit,
    work: () -> T,
    sample: (Rep<T>) -> Double,
  ): MeasuredCell {
    val series = mutableListOf<Pair<ThermalReading, ThermalReading>>()
    var openStart: ThermalReading? = null

    val gate = { readinessGate(readiness) }
    val observer =
      MeasurementObserver(
        onBeforeRepStart = {
          check(openStart == null) { "a repetition was started while the previous one was still open" }
          sampleThermal(thermal)?.let { openStart = it }
        },
        onAfterRepEnd = {
          val start = openStart
          openStart = null
          val end = sampleThermal(thermal)
          if (start != null && end != null) {
            series.add(start to end)
          }
        },
      )

    val measurement =
      runMeasurement(
        label = label,
        gate = gate,
        observer = observer,
        prepare = { prepare() },
        work = { work() },
        sample = { idx, rep ->
          val s = sample(rep)
          reportMeasurement(progress, idx, REPS)
          s
        },
      )

    return MeasuredCell(measurement.stats, series)
  }

  fun withThreadCount(result: JSONObject, model: LlamaModel?) {
    val count =
      if (model != null && !model.isClosed) {
        LlamaCpp.nThreads(model)
      } else {
        LlamaCpp.defaultThreads()
      }
    if (count > 0) {
      result.put("runtime_thread_count", count)
    }
  }

  fun runPrefillThroughputImpl(
    def: BenchmarkDef,
    model: LlamaModel,
    contextSize: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val prefillTokens = def.parameterPrefillTokens ?: 512
    val benchmarkId = def.benchmarkId

    val promptTokens = seedPromptTokens(model, prefillTokens, contextSize, true)

    // Warm-up at the measured shape, untimed and ungated.
    LlamaCpp.resetContext(model)
    LlamaCpp.resetSampler(model)
    LlamaCpp.prefill(model, promptTokens)
    Log.i(TAG, "prefill_throughput: warm-up run (${promptTokens.size}p)")

    val measured =
      measureCell(
        label = "prefill_throughput",
        progress = progress,
        readiness = readiness,
        thermal = thermal,
        prepare = {
          LlamaCpp.resetContext(model)
          LlamaCpp.resetSampler(model)
        },
        work = { LlamaCpp.prefill(model, promptTokens) },
        sample = { rep -> rep.elapsedMs() },
      )
    val meanMs = measured.stats.meanMs
    val stddevMs = measured.stats.stddevMs
    Log.i(TAG, String.format(Locale.US, "prefill_throughput: prefill_time_ms=%.3f stddev=%.3f", meanMs, stddevMs))

    val result =
      JSONObject().apply {
        put("benchmark_id", benchmarkId)
        put("prefill_time_ms", meanMs)
        put("prefill_time_ms_stddev", stddevMs)
      }
    val (before, after) = measured.thermal.unzip()
    mergeThermal(result, before, after)
    return result
  }

  fun runDecodeThroughputImpl(
    def: BenchmarkDef,
    model: LlamaModel,
    contextSize: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val prefillTokens = def.parameterPrefillTokens ?: 512
    val decodeTokens = def.parameterDecodeTokens ?: 128
    val benchmarkId = def.benchmarkId

    val promptTokens = seedPromptTokens(model, prefillTokens, contextSize, true)

    // Warm-up at the measured shape, untimed and ungated.
    LlamaCpp.resetContext(model)
    LlamaCpp.resetSampler(model)
    LlamaCpp.prefill(model, promptTokens)
    val warmRc = LlamaCpp.decodeNGreedyIgnoreEog(model, decodeTokens)
    check(warmRc == 0) { "warmup decode failed with code $warmRc" }
    Log.i(TAG, "decode_throughput: warm-up run (${promptTokens.size}p/${decodeTokens}g)")

    val measured =
      measureCell(
        label = "decode_throughput",
        progress = progress,
        readiness = readiness,
        thermal = thermal,
        prepare = {
          LlamaCpp.resetContext(model)
          LlamaCpp.resetSampler(model)
          // Untimed: re-establishes KV depth before the timed decode.
          LlamaCpp.prefill(model, promptTokens)
        },
        work = {
          val rc = LlamaCpp.decodeNGreedyIgnoreEog(model, decodeTokens)
          check(rc == 0) { "decode_n_greedy_ignore_eog failed with code $rc" }
        },
        sample = { rep -> rep.elapsedMs() },
      )
    val meanMs = measured.stats.meanMs
    val stddevMs = measured.stats.stddevMs
    Log.i(TAG, String.format(Locale.US, "decode_throughput: decode_time_ms=%.3f stddev=%.3f", meanMs, stddevMs))

    val result =
      JSONObject().apply {
        put("benchmark_id", benchmarkId)
        put("decode_time_ms", meanMs)
        put("decode_time_ms_stddev", stddevMs)
      }
    val (before, after) = measured.thermal.unzip()
    mergeThermal(result, before, after)
    return result
  }

  fun runEndToEndLatencyImpl(
    def: BenchmarkDef,
    model: LlamaModel,
    contextSize: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val prefillTokens = def.parameterPrefillTokens ?: 512
    val decodeTokens = def.parameterDecodeTokens ?: 128
    val benchmarkId = def.benchmarkId

    val promptText = seedPromptText(model, prefillTokens, contextSize, true)
    val expectedPromptTokens = min(prefillTokens, contextSize)

    // Warm-up at the measured shape, untimed and ungated.
    val warmupPrompt = seedPromptTokens(model, prefillTokens, contextSize, true)
    LlamaCpp.resetContext(model)
    LlamaCpp.resetSampler(model)
    LlamaCpp.prefill(model, warmupPrompt)
    val warmRc = LlamaCpp.decodeNGreedyIgnoreEog(model, decodeTokens)
    check(warmRc == 0) { "warmup decode failed with code $warmRc" }
    Log.i(TAG, "end_to_end_latency: warm-up run (${prefillTokens}p/${decodeTokens}g)")

    val measured =
      measureCell(
        label = "end_to_end_latency",
        progress = progress,
        readiness = readiness,
        thermal = thermal,
        prepare = {
          LlamaCpp.resetContext(model)
          LlamaCpp.resetSampler(model)
        },
        work = {
          // Tokenize in-window, then prefill + decode.
          val tokens = LlamaCpp.tokenize(model, promptText, true)
          val promptLen = tokens.size
          LlamaCpp.prefill(model, tokens)
          val rc = LlamaCpp.decodeNGreedyIgnoreEog(model, decodeTokens)
          check(rc == 0) { "decode_n_greedy_ignore_eog failed with code $rc" }
          promptLen
        },
        sample = { rep ->
          if (rep.value != expectedPromptTokens) {
            error("e2e prompt tokenized to ${rep.value} tokens in-window, expected $expectedPromptTokens: prompt-seed / tokenizer drift")
          }
          rep.elapsedMs()
        },
      )
    val meanMs = measured.stats.meanMs
    val stddevMs = measured.stats.stddevMs
    Log.i(TAG, String.format(Locale.US, "end_to_end_latency: total_time_ms=%.3f stddev=%.3f", meanMs, stddevMs))

    val result =
      JSONObject().apply {
        put("benchmark_id", benchmarkId)
        put("total_time_ms", meanMs)
        put("total_time_ms_stddev", stddevMs)
      }
    val (before, after) = measured.thermal.unzip()
    mergeThermal(result, before, after)
    return result
  }

  fun runMaxMemoryUsage(
    def: BenchmarkDef,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    onModelLoaded: ((Int) -> Unit)? = null,
  ): JSONObject {
    val prefillTokens = def.parameterPrefillTokens ?: 512
    val benchmarkId = def.benchmarkId

    val poller = spawnMemoryAllocationPoller()
    var liveMemorySample = 0L
    var executionError: Throwable? = null

    try {
      withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
        onModelLoaded?.invoke(LlamaCpp.nThreads(model))
        val tokens = seedPromptTokens(model, prefillTokens, contextSize, true)
        LlamaCpp.prefill(model, tokens)
        val rc = LlamaCpp.decodeNGreedyIgnoreEog(model, 1)
        check(rc == 0) { "decode_n_greedy_ignore_eog failed with code $rc" }
        liveMemorySample = currentRssBytes() ?: 0L
      }
    } catch (t: Throwable) {
      executionError = t
    }

    val maxRamBytes: Long
    try {
      maxRamBytes = poller.stopAndJoinWithSample(liveMemorySample)
    } catch (pollerError: Throwable) {
      if (executionError != null) {
        executionError.addSuppressed(pollerError)
        throw executionError
      }
      throw pollerError
    }
    if (executionError != null) {
      throw executionError
    }

    return JSONObject().apply {
      put("benchmark_id", benchmarkId)
      put("max_ram_bytes", maxRamBytes)
    }
  }

  fun runEvalImpl(def: BenchmarkDef, model: LlamaModel, progress: BenchmarkProgressCallback?, readiness: BenchmarkCooldownCallback?): JSONObject {
    val samples = def.samples ?: error("eval benchmark missing samples")
    val maxTokens = def.parameterMaxTokens ?: 256
    val mcqChoices = def.parameterMcqChoices?.takeIf { it.isNotEmpty() }
    val benchmarkId = def.benchmarkId

    val total = samples.length()

    data class EvalSample(val id: String, val messagesJson: String)
    val sampleData = mutableListOf<EvalSample>()
    for (i in 0 until total) {
      val sampleObj = samples.getJSONObject(i)
      val sampleId = sampleObj.optString("id", "")
      val messages = sampleObj.optJSONArray("messages") ?: JSONArray()
      sampleData.add(EvalSample(sampleId, messages.toString()))
    }

    val completions = JSONArray()

    for (index in 0 until sampleData.size) {
      val sample = sampleData[index]
      readinessGate(readiness)
      LlamaCpp.resetContext(model)

      val effectiveMax = if (mcqChoices != null) 1 else maxTokens

      try {
        val generation = LlamaCpp.chatCompletion(model, sample.messagesJson, effectiveMax, mcqChoices)
        val compObj =
          JSONObject().apply {
            put("id", sample.id)
            put("completion", generation.text)
            put("failed", false)
            put("stop_reason", generation.stopReason.wireValue)
            generation.stopDetail?.let { put("stop_detail", it) }
            put("completion_tokens", generation.completionTokens)
          }
        completions.put(compObj)
      } catch (err: Throwable) {
        if (abortsCell(err)) throw err
        val detail = err.message ?: err.javaClass.simpleName
        Log.w(TAG, "eval sample ${sample.id} failed, recording and continuing: $detail")
        val compObj =
          JSONObject().apply {
            put("id", sample.id)
            put("completion", "")
            put("failed", true)
            put("failed_reason", detail)
            put("stop_reason", BenchmarkEvalCompletionStopReason.FAILURE.wireValue)
            put("stop_detail", detail)
          }
        completions.put(compObj)
      }

      val completed = index + 1
      if (progress != null && !progress.onProgress(completed, total, "Sample $completed/$total")) {
        throw CancellationException("cancelled after $completed/$total samples")
      }
    }

    return JSONObject().apply {
      put("benchmark_id", benchmarkId)
      put("completions", completions)
    }
  }

  /**
   * A vlm_eval sample's `messages` arrive pre-built by the server as OpenAI content arrays that already embed `image_url` data-URI blocks (mirrors
   * the macOS/Linux server's own vlm_eval samples). This engine has no chat server in front of it, so it has to do what llama-server's own multimodal
   * chat handler does: fold each content array into a single templatable string (image blocks become the mtmd media marker, in place, so the rendered
   * prompt reads the same as the original interleaving) and pull the one image's raw bytes out separately for [LlamaCpp.mtmdAllocChunksFromImage].
   * Only a single image per sample is supported, matching the single-bitmap mtmd path vl_throughput already uses.
   */
  private fun extractTemplatableMessagesAndImage(messages: JSONArray): Pair<String, ByteArray> {
    val templatable = JSONArray()
    var imageBytes: ByteArray? = null
    for (i in 0 until messages.length()) {
      val message = messages.getJSONObject(i)
      val role = message.optString("role", "user")
      val (text, blockImage) = templatableContent(message.opt("content"))
      if (blockImage != null) {
        check(imageBytes == null) { "vlm_eval sample has more than one image; only single-image samples run on-device" }
        imageBytes = blockImage
      }
      templatable.put(JSONObject().put("role", role).put("content", text))
    }
    val bytes = imageBytes ?: error("vlm_eval sample missing an image_url content block")
    return templatable.toString() to bytes
  }

  /**
   * One message's `content` folded into templatable text (image blocks become the mtmd media marker) plus that message's image bytes, if it carried
   * one. A plain string `content` passes through unchanged, matching [LlamaCpp.applyChatTemplate]'s existing text-message shape.
   */
  private fun templatableContent(content: Any?): Pair<String, ByteArray?> {
    if (content !is JSONArray) return (content as? String ?: "") to null
    val text = StringBuilder()
    var imageBytes: ByteArray? = null
    for (j in 0 until content.length()) {
      val block = content.getJSONObject(j)
      when (block.optString("type")) {
        "text" -> text.append(block.optString("text", ""))
        "image_url" -> {
          check(imageBytes == null) { "vlm_eval sample has more than one image; only single-image samples run on-device" }
          imageBytes = decodeDataUriImage(block.getJSONObject("image_url").getString("url"))
          text.append(MEDIA_MARKER)
        }
      }
    }
    return text.toString() to imageBytes
  }

  /** Raw bytes out of a `data:<mime>;base64,<...>` image URL, the shape mgmt embeds vlm_eval samples' images in. */
  private fun decodeDataUriImage(url: String): ByteArray {
    val marker = ";base64,"
    val idx = url.indexOf(marker)
    check(url.startsWith("data:") && idx >= 0) { "vlm_eval sample image_url is not a base64 data URI" }
    return android.util.Base64.decode(url.substring(idx + marker.length), android.util.Base64.DEFAULT)
  }

  fun runVlmEvalImpl(
    def: BenchmarkDef,
    model: LlamaModel,
    mmprojPath: String,
    nGpuLayers: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
  ): JSONObject {
    // Only a local definition carries its samples inline. A catalog entry's
    // are a corpus this engine does not download: those run through the
    // pipette CLI on the device (the adb transport).
    val samples = def.samples ?: error("vlm_eval: this engine runs inline samples only, not a corpus")
    val maxTokens = def.parameterMaxTokens ?: 256
    val mcqChoices = def.parameterMcqChoices?.takeIf { it.isNotEmpty() }
    val benchmarkId = def.benchmarkId
    val useGpu = nGpuLayers > 0
    val total = samples.length()

    val completions = JSONArray()
    val mtmd = LlamaCpp.mtmdInit(model, mmprojPath, useGpu)
    try {
      for (index in 0 until total) {
        val sampleObj = samples.getJSONObject(index)
        readinessGate(readiness)
        LlamaCpp.resetContext(model)
        val effectiveMax = if (mcqChoices != null) 1 else maxTokens
        completions.put(runOneVlmEvalSample(model, mtmd, sampleObj, effectiveMax, mcqChoices))

        val completed = index + 1
        if (progress != null && !progress.onProgress(completed, total, "Sample $completed/$total")) {
          throw CancellationException("cancelled after $completed/$total samples")
        }
      }
    } finally {
      mtmd.close()
    }

    return JSONObject().apply {
      put("benchmark_id", benchmarkId)
      put("completions", completions)
    }
  }

  /** One vlm_eval sample's completion JSON: a success row, or (mirroring [runEvalImpl]) a `failed: true` row recording why. */
  private fun runOneVlmEvalSample(
    model: LlamaModel,
    mtmd: MtmdContext,
    sampleObj: JSONObject,
    effectiveMax: Int,
    mcqChoices: List<String>?,
  ): JSONObject {
    val sampleId = sampleObj.optString("id", "")
    return try {
      val messages = sampleObj.optJSONArray("messages") ?: JSONArray()
      val (templatableMessagesJson, imageBytes) = extractTemplatableMessagesAndImage(messages)
      val generation = LlamaCpp.chatCompletionWithImage(model, mtmd, templatableMessagesJson, imageBytes, effectiveMax, mcqChoices)
      JSONObject().apply {
        put("id", sampleId)
        put("completion", generation.text)
        put("failed", false)
        put("stop_reason", generation.stopReason.wireValue)
        generation.stopDetail?.let { put("stop_detail", it) }
        put("completion_tokens", generation.completionTokens)
      }
    } catch (err: Throwable) {
      if (abortsCell(err)) throw err
      val detail = err.message ?: err.javaClass.simpleName
      Log.w(TAG, "vlm_eval sample $sampleId failed, recording and continuing: $detail")
      JSONObject().apply {
        put("id", sampleId)
        put("completion", "")
        put("failed", true)
        put("failed_reason", detail)
        put("stop_reason", BenchmarkEvalCompletionStopReason.FAILURE.wireValue)
        put("stop_detail", detail)
      }
    }
  }

  fun runVlThroughputImpl(
    def: BenchmarkDef,
    model: LlamaModel,
    mmprojPath: String,
    nGpuLayers: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val useGpu = nGpuLayers > 0
    val imageWidth = def.parameterImageWidth ?: error("vl_throughput missing parameter_image_width")
    val imageHeight = def.parameterImageHeight ?: error("vl_throughput missing parameter_image_height")
    val textTokens = def.parameterTextTokens ?: error("vl_throughput missing parameter_text_tokens")
    val decodeTokens = def.parameterDecodeTokens ?: error("vl_throughput missing parameter_decode_tokens")
    val numImages = max(def.parameterNumImages ?: 1, 1)
    if (numImages > 1) {
      error("vl_throughput num_images=$numImages not supported on the native (mtmd) path yet; only single-image cells run on-device")
    }
    val benchmarkId = def.benchmarkId

    val textPrompt = PromptSeed.buildPromptText(textTokens) { text -> LlamaCpp.tokenize(model, text, false).size }
    val promptWithMarker = "$MEDIA_MARKER$textPrompt"

    val mtmd = LlamaCpp.mtmdInit(model, mmprojPath, useGpu)
    try {
      return runVlMeasurements(
        model = model,
        mtmd = mtmd,
        promptWithMarker = promptWithMarker,
        width = imageWidth,
        height = imageHeight,
        decodeTokens = decodeTokens,
        benchmarkId = benchmarkId,
        progress = progress,
        readiness = readiness,
        thermal = thermal,
      )
    } finally {
      mtmd.close()
    }
  }

  private fun runVlMeasurements(
    model: LlamaModel,
    mtmd: MtmdContext,
    promptWithMarker: String,
    width: Int,
    height: Int,
    decodeTokens: Int,
    benchmarkId: String,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val totalRuns = 1 + VL_NUM_MEASUREMENT_RUNS
    val promptMsSamples = mutableListOf<Double>()
    val predictedMsSamples = mutableListOf<Double>()
    var promptTokensObserved: Int? = null
    var imageTokensObserved: Int? = null

    val thermalSeries = mutableListOf<Pair<ThermalReading, ThermalReading>>()
    var openStart: ThermalReading? = null

    for (runIndex in 0 until totalRuns) {
      if (progress != null) {
        val label = if (runIndex == 0) "VL warmup run" else "VL run $runIndex/$VL_NUM_MEASUREMENT_RUNS"
        if (!progress.onProgress(runIndex, totalRuns, label)) {
          throw CancellationException("cancelled at vl run $runIndex/$totalRuns")
        }
      }

      if (runIndex > 0) {
        readinessGate(readiness)
        openStart = sampleThermal(thermal)
      }

      LlamaCpp.resetContext(model)
      LlamaCpp.resetSampler(model)

      val chunks = LlamaCpp.mtmdAllocGrayChunks(mtmd, promptWithMarker, width, height)
      try {
        val t0 = System.nanoTime()
        val evalRc = LlamaCpp.mtmdEvalChunks(mtmd, model, chunks)
        val promptMs = (System.nanoTime() - t0) / 1_000_000.0
        check(evalRc == 0) { "mtmd_eval_chunks failed with code $evalRc" }

        val t1 = System.nanoTime()
        val decodeRc = LlamaCpp.decodeNGreedyIgnoreEog(model, decodeTokens)
        val predictedMs = (System.nanoTime() - t1) / 1_000_000.0
        check(decodeRc == 0) { "decode_n_greedy_ignore_eog failed with code $decodeRc" }

        if (runIndex == 0) {
          promptTokensObserved = chunks.totalTokens.toInt()
          imageTokensObserved = chunks.imageTokens.toInt()
        } else {
          promptMsSamples.add(promptMs)
          predictedMsSamples.add(predictedMs)
          val end = sampleThermal(thermal)
          val start = openStart
          openStart = null
          if (start != null && end != null) {
            thermalSeries.add(start to end)
          }
        }
      } finally {
        chunks.close()
      }
    }

    val promptTokens = promptTokensObserved ?: 0
    val imageTokens = imageTokensObserved ?: 0
    val promptStats = meanStddev(promptMsSamples)
    val predictedStats = meanStddev(predictedMsSamples)

    val result =
      JSONObject().apply {
        put("benchmark_id", benchmarkId)
        put("prompt_tokens", promptTokens)
        put("image_tokens", imageTokens)
        put("prompt_ms", promptStats.meanMs)
        put("prompt_ms_stddev", promptStats.stddevMs)
        put("predicted_ms", predictedStats.meanMs)
        put("predicted_ms_stddev", predictedStats.stddevMs)
      }
    val (thermalBefore, thermalAfter) = thermalSeries.unzip()
    mergeThermal(result, thermalBefore, thermalAfter)
    return result
  }

  private fun abortsCell(t: Throwable): Boolean =
    t is CancellationException ||
      t is OutOfMemoryError ||
      (t.message?.contains("cancelled", ignoreCase = true) == true) ||
      (t.message?.contains("unusable", ignoreCase = true) == true)

  private inline fun <T> withModel(
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    block: (LlamaModel) -> T,
  ): T {
    val model = LlamaCpp.loadModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode)
    return model.use { block(it) }
  }

  fun runFresh(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    val def = parseBenchmarkDef(benchmarkJson)
    checkCtxSize(def, contextSize)
    checkBatchSize(def)

    if (progress?.onProgress(0, 1, "Starting benchmark...") == false) {
      throw CancellationException("cancelled before start")
    }

    var modelThreads: Int? = null
    val result =
      when (def.benchmarkType) {
        "prefill_throughput" ->
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runPrefillThroughputImpl(def, model, contextSize, progress, cooldown, thermal)
          }
        "decode_throughput" ->
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runDecodeThroughputImpl(def, model, contextSize, progress, cooldown, thermal)
          }
        "end_to_end_latency" ->
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runEndToEndLatencyImpl(def, model, contextSize, progress, cooldown, thermal)
          }
        "max_memory_usage" -> runMaxMemoryUsage(def, modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { threads -> modelThreads = threads }
        "eval" ->
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runEvalImpl(def, model, progress, cooldown)
          }
        "vl_throughput" -> {
          val mmproj = mmprojPath ?: error("vl_throughput benchmark requires mmproj_path")
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runVlThroughputImpl(def, model, mmproj, nGpuLayers, progress, cooldown, thermal)
          }
        }
        "vlm_eval" -> {
          val mmproj = mmprojPath ?: error("vlm_eval benchmark requires mmproj_path")
          withModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode) { model ->
            modelThreads = LlamaCpp.nThreads(model)
            runVlmEvalImpl(def, model, mmproj, nGpuLayers, progress, cooldown)
          }
        }
        else -> error("unsupported benchmark type: ${def.benchmarkType}")
      }

    val finalThreads = modelThreads ?: LlamaCpp.defaultThreads()
    if (finalThreads > 0) {
      result.put("runtime_thread_count", finalThreads)
    }
    return result.toString()
  }

  fun runOnModel(
    benchmarkJson: String,
    model: LlamaModel,
    nGpuLayers: Int,
    contextSize: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    val def = parseBenchmarkDef(benchmarkJson)
    checkCtxSize(def, contextSize)
    checkBatchSize(def)

    if (progress?.onProgress(0, 1, "Starting benchmark...") == false) {
      throw CancellationException("cancelled before start")
    }

    LlamaCpp.resetContext(model)
    LlamaCpp.resetSampler(model)

    val result =
      when (def.benchmarkType) {
        "prefill_throughput" -> runPrefillThroughputImpl(def, model, contextSize, progress, cooldown, thermal)
        "decode_throughput" -> runDecodeThroughputImpl(def, model, contextSize, progress, cooldown, thermal)
        "end_to_end_latency" -> runEndToEndLatencyImpl(def, model, contextSize, progress, cooldown, thermal)
        "eval" -> runEvalImpl(def, model, progress, cooldown)
        "vl_throughput" -> {
          val mmproj = mmprojPath ?: error("vl_throughput benchmark requires mmproj_path")
          runVlThroughputImpl(def, model, mmproj, nGpuLayers, progress, cooldown, thermal)
        }
        "vlm_eval" -> {
          val mmproj = mmprojPath ?: error("vlm_eval benchmark requires mmproj_path")
          runVlmEvalImpl(def, model, mmproj, nGpuLayers, progress, cooldown)
        }
        "max_memory_usage" -> error("max_memory_usage requires a fresh model load; call run_benchmark instead")
        else -> error("unsupported benchmark type: ${def.benchmarkType}")
      }

    withThreadCount(result, model)
    return result.toString()
  }
}
