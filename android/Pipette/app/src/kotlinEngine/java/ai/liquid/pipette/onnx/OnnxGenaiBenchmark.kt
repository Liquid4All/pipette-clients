package ai.liquid.pipette.onnx

import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.OnnxGenaiCells
import ai.liquid.pipette.llama.LlamaBenchmark
import ai.liquid.pipette.llama.LlamaBenchmark.BenchmarkDef
import ai.liquid.pipette.llama.currentRssBytes
import ai.liquid.pipette.llama.spawnMemoryAllocationPoller
import ai.liquid.pipette.ops.PromptSeed
import ai.liquid.pipette.ops.positiveFinite
import android.util.Log
import java.util.Locale
import java.util.concurrent.CancellationException
import org.json.JSONObject

/**
 * The ONNX Runtime GenAI benchmark kernel. Mirrors iOS's `OnnxGenaiExecute`, cell for cell, over the harness [LlamaBenchmark] uses (five gated
 * repetitions after one warm-up at the measured shape).
 *
 * The engine has no prefill-only entry point, so every timing cell runs one generation and reads a different interval off it ([GenerationTiming]).
 * The policy for everything else is iOS's: the same metric, or an error, never an approximation.
 */
@Suppress("TooManyFunctions") // the four cells, their shared harness and the two entry points EngineActor calls
object OnnxGenaiBenchmark {
  private const val TAG = "pipette-onnx"
  private const val DEFAULT_PREFILL_TOKENS = 512
  private const val DEFAULT_DECODE_TOKENS = 128

  /** Refuse an entry that wants more than one request in flight: the engine generates one sequence at a time. As [LlamaBenchmark.checkBatchSize]. */
  fun checkBatchSize(def: BenchmarkDef) {
    val batchSize = def.parameterBatchSize ?: 1
    if (batchSize < 1) error("parameter_batch_size must be at least 1, got $batchSize")
    if (batchSize != 1) {
      error(
        "benchmark '${def.benchmarkId}' keeps $batchSize requests in flight, and " +
          "the in-process ONNX Runtime GenAI engine generates one sequence at a time"
      )
    }
  }

  /**
   * The shared corpus text under this model's own tokenizer, not synthetic ids and not chat-templated: a prefill number only compares with
   * llama.cpp's if both ran the same text through their own tokenizers. Cut to [targetTokens] as [LlamaBenchmark.seedPromptTokens] cuts, and refused
   * when the text falls short, so a cell never files a number under a prompt length it did not run.
   */
  fun promptIds(model: OnnxGenaiModel, targetTokens: Int): IntArray {
    val text = PromptSeed.buildPromptText(targetTokens) { OnnxGenai.encode(model, it).size }
    val ids = OnnxGenai.encode(model, text)
    check(ids.size >= targetTokens) { "the prompt tokenized to ${ids.size} tokens, expected $targetTokens: prompt-seed / tokenizer drift" }
    return if (ids.size > targetTokens) ids.copyOf(targetTokens) else ids
  }

  /** The time to the first token of a one-token generation over the cell's prompt ([GenerationTiming.ttftMs]). */
  fun runPrefillThroughputImpl(
    def: BenchmarkDef,
    model: OnnxGenaiModel,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val ids = promptIds(model, def.parameterPrefillTokens ?: DEFAULT_PREFILL_TOKENS)
    return timingCell("prefill_throughput", "prefill_time_ms", def, model, ids, 1, progress, readiness, thermal) { timing -> timing.ttftMs }
  }

  /** The mean interval after the first token, times the cell's decode count: the span the other runtimes report. */
  fun runDecodeThroughputImpl(
    def: BenchmarkDef,
    model: OnnxGenaiModel,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val decodeTokens = def.parameterDecodeTokens ?: DEFAULT_DECODE_TOKENS
    val ids = promptIds(model, def.parameterPrefillTokens ?: DEFAULT_PREFILL_TOKENS)
    return timingCell("decode_throughput", "decode_time_ms", def, model, ids, decodeTokens, progress, readiness, thermal) { timing ->
      // Null only for a single-token generation, which has no interval.
      val tpotMs = timing.tpotMs ?: error("a $decodeTokens-token generation reported no per-token interval")
      tpotMs * decodeTokens
    }
  }

  /** `ttft + tpot * (n - 1)` ([GenerationTiming.endToEndMs]): reconstructed, so tokenizing and building the generator are left out. */
  fun runEndToEndLatencyImpl(
    def: BenchmarkDef,
    model: OnnxGenaiModel,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject {
    val decodeTokens = def.parameterDecodeTokens ?: DEFAULT_DECODE_TOKENS
    val ids = promptIds(model, def.parameterPrefillTokens ?: DEFAULT_PREFILL_TOKENS)
    return timingCell("end_to_end_latency", "total_time_ms", def, model, ids, decodeTokens, progress, readiness, thermal) { timing ->
      timing.endToEndMs
    }
  }

  /** One warm-up generation at the measured shape, untimed and ungated, then the gated repetitions, each sampled by [read]. */
  @Suppress("LongParameterList") // one cell: its label, its metric, its shape and the harness's three callbacks
  private fun timingCell(
    label: String,
    metric: String,
    def: BenchmarkDef,
    model: OnnxGenaiModel,
    ids: IntArray,
    decodeTokens: Int,
    progress: BenchmarkProgressCallback?,
    readiness: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
    read: (GenerationTiming) -> Double,
  ): JSONObject {
    OnnxGenai.generate(model, ids, decodeTokens)
    Log.i(TAG, "$label: warm-up run (${ids.size}p/${decodeTokens}g)")

    val measured =
      LlamaBenchmark.measureCell(
        label = label,
        progress = progress,
        readiness = readiness,
        thermal = thermal,
        prepare = {},
        work = { OnnxGenai.generate(model, ids, decodeTokens) },
        sample = { rep -> positiveFinite(metric, read(rep.value)) },
      )
    val meanMs = measured.stats.meanMs
    val stddevMs = measured.stats.stddevMs
    Log.i(TAG, String.format(Locale.US, "%s: %s=%.3f stddev=%.3f", label, metric, meanMs, stddevMs))

    val result =
      JSONObject().apply {
        put("benchmark_id", def.benchmarkId)
        put(metric, meanMs)
        put("${metric}_stddev", stddevMs)
      }
    val (before, after) = measured.thermal.unzip()
    LlamaBenchmark.mergeThermal(result, before, after)
    return result
  }

  /**
   * One prefill and one decode on a fresh load, under the RSS poller [LlamaBenchmark.runMaxMemoryUsage] uses. ORT maps a package's external-data
   * file, and RSS counts resident file-backed pages, so the mapped weights are in the figure without iOS's separate file-backed term.
   */
  fun runMaxMemoryUsage(def: BenchmarkDef, modelPath: String): JSONObject {
    val prefillTokens = def.parameterPrefillTokens ?: DEFAULT_PREFILL_TOKENS

    val poller = spawnMemoryAllocationPoller()
    var liveMemorySample = 0L
    val execution = runCatching {
      OnnxGenai.loadModel(modelPath).use { model ->
        OnnxGenai.generate(model, promptIds(model, prefillTokens), 1)
        liveMemorySample = currentRssBytes() ?: 0L
      }
    }
    // The poller stops whether or not the cell ran, and the cell's own failure is the one reported.
    val peak = runCatching { poller.stopAndJoinWithSample(liveMemorySample) }
    execution.getOrThrow()

    return JSONObject().apply {
      put("benchmark_id", def.benchmarkId)
      put("max_ram_bytes", peak.getOrThrow())
    }
  }

  /** The kinds this engine refuses, each with its reason; null for the four it runs. The rule is [OnnxGenaiCells]', which the runner reads too. */
  fun unsupportedReason(benchmarkType: String): String? = OnnxGenaiCells.unsupportedReason(benchmarkType)

  private fun parse(benchmarkJson: String): BenchmarkDef {
    val def = LlamaBenchmark.parseBenchmarkDef(benchmarkJson)
    unsupportedReason(def.benchmarkType)?.let { error("benchmark '${def.benchmarkId}' is unsupported: $it") }
    if (def.benchmarkType != "max_memory_usage") checkBatchSize(def)
    return def
  }

  private fun start(progress: BenchmarkProgressCallback?) {
    if (progress?.onProgress(0, 1, "Starting benchmark...") == false) {
      throw CancellationException("cancelled before start")
    }
  }

  /** A cell that loads its own package and frees it when done. */
  fun runFresh(
    benchmarkJson: String,
    modelPath: String,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    val def = parse(benchmarkJson)
    start(progress)
    if (def.benchmarkType == "max_memory_usage") return runMaxMemoryUsage(def, modelPath).toString()
    return OnnxGenai.loadModel(modelPath).use { model -> runTiming(def, model, progress, cooldown, thermal) }.toString()
  }

  /** A cell against a package that is already loaded. */
  fun runOnModel(
    benchmarkJson: String,
    model: OnnxGenaiModel,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String {
    val def = parse(benchmarkJson)
    start(progress)
    if (def.benchmarkType == "max_memory_usage") error("max_memory_usage requires a fresh model load; call run_benchmark instead")
    return runTiming(def, model, progress, cooldown, thermal).toString()
  }

  private fun runTiming(
    def: BenchmarkDef,
    model: OnnxGenaiModel,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): JSONObject =
    when (def.benchmarkType) {
      "prefill_throughput" -> runPrefillThroughputImpl(def, model, progress, cooldown, thermal)
      "decode_throughput" -> runDecodeThroughputImpl(def, model, progress, cooldown, thermal)
      "end_to_end_latency" -> runEndToEndLatencyImpl(def, model, progress, cooldown, thermal)
      else -> error("unsupported benchmark type: ${def.benchmarkType}")
    }
}
