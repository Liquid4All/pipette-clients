package ai.liquid.pipette

import java.io.File
import java.util.Locale
import kotlin.math.abs
import org.json.JSONObject

data class CompletedRunMetric(val name: String, val unit: String, val displayValue: String, val numericValue: Double, val higherIsBetter: Boolean)

object CompletedResultsCsvExporter {
  val headers =
    listOf(
      "job_id",
      "job_title",
      "created_at",
      "cell_id",
      "model_name",
      "model_display_name",
      "model_quant",
      "benchmark_id",
      "benchmark_type",
      "benchmark_name",
      "benchmark_parameters",
      "status",
      "metric_name",
      "metric_value",
      "metric_unit",
      "metric_display_value",
      "submitted_at",
      "server_job_id",
      "runtime_name",
      "runtime_version",
      "runtime_flags",
      "benchmark_flags",
      "runtime_cpu_variant",
      "runtime_thread_count",
      "device_name",
      "device_form_factor",
      "device_os_name",
      "device_os_version",
      "device_os_build",
      "device_os_security_patch",
      "device_chip_model",
      "device_ram_bytes",
      "device_battery_level",
      "device_power_state",
      "device_power_save_mode",
      "device_android_cpuset",
      "device_android_cpu_affinity_list",
      "device_android_cpu_affinity_excludes_top_tier",
      "error_message",
    )

  fun filename(manifest: JobManifest): String {
    val jobPrefix = manifest.jobId.take(8)
    return "pipette-results-${DateFormats.shortDate(manifest.createdAt)}-$jobPrefix.csv"
  }

  fun csv(storage: LocalStorage, manifest: JobManifest): String = csv(manifest, payloadsByCellId(storage, manifest))

  fun csv(manifest: JobManifest, payloadsByCellId: Map<String, JSONObject>): String {
    val metricsByCellId = metricsByCellId(manifest, payloadsByCellId)
    val rows =
      csvCells(manifest).map { cell ->
        val payload = payloadsByCellId[cell.cellId]
        val metric = metricsByCellId[cell.cellId]
        val benchmarkType = benchmarkType(cell)
        listOf(
          manifest.jobId,
          manifest.displayTitle,
          manifest.createdAt,
          cell.cellId,
          cell.modelName,
          modelDisplayName(cell),
          quantLabel(cell),
          cell.benchmarkId,
          benchmarkType,
          BenchmarkCatalog.displayName(benchmarkType),
          parameterSummary(cell.benchmarkId) ?: "",
          cell.runStatus.wire,
          metric?.name ?: "",
          metric?.numericValue?.let { csvNumber(it) } ?: "",
          metric?.unit ?: "",
          metric?.displayValue ?: "",
          payloadString(payload, "submitted_at"),
          cell.serverJobId ?: "",
          payloadString(payload, "runtime_name"),
          payloadString(payload, "runtime_version"),
          payloadString(payload, "runtime_flags"),
          payloadString(payload, "benchmark_flags"),
          payloadString(payload, "runtime_cpu_variant"),
          payloadString(payload, "runtime_thread_count"),
          payloadString(payload, "device_name"),
          payloadString(payload, "device_form_factor"),
          payloadString(payload, "device_os_name"),
          payloadString(payload, "device_os_version"),
          payloadString(payload, "device_os_build"),
          payloadString(payload, "device_os_security_patch"),
          payloadString(payload, "device_chip_model"),
          payloadString(payload, "device_ram_bytes"),
          payloadString(payload, "device_battery_level"),
          payloadString(payload, "device_power_state"),
          payloadString(payload, "device_power_save_mode"),
          payloadString(payload, "device_android_cpuset"),
          payloadString(payload, "device_android_cpu_affinity_list"),
          payloadString(payload, "device_android_cpu_affinity_excludes_top_tier"),
          cell.errorMessage ?: "",
        )
      }
    return (listOf(headers) + rows).joinToString(separator = "\n") { csvLine(it) } + "\n"
  }

  fun metricsByCellId(manifest: JobManifest, payloadsByCellId: Map<String, JSONObject>): Map<String, CompletedRunMetric> =
    manifest.cells.mapNotNull { cell -> metric(cell, payloadsByCellId[cell.cellId])?.let { cell.cellId to it } }.toMap()

  fun metric(cell: JobCell, payload: JSONObject?): CompletedRunMetric? {
    if (payload == null) return null
    val type = benchmarkType(cell)
    val params = benchmarkParams(cell)
    return when (type) {
      "prefill_throughput" -> {
        val ms = payload.doubleOrNull("prefill_time_ms")?.takeIf { it > 0 } ?: return null
        val tokens = params.uint("parameter_prefill_tokens")
        val throughput = tokens / (ms / 1000.0)
        CompletedRunMetric("Prefill throughput", "tok/s", formatNumber(throughput), throughput, true)
      }
      "decode_throughput" -> {
        val ms = payload.doubleOrNull("decode_time_ms")?.takeIf { it > 0 } ?: return null
        val tokens = params.uint("parameter_decode_tokens")
        val throughput = tokens / (ms / 1000.0)
        CompletedRunMetric("Decode throughput", "tok/s", formatNumber(throughput), throughput, true)
      }
      "end_to_end_latency" -> {
        val ms = payload.doubleOrNull("total_time_ms") ?: return null
        CompletedRunMetric("E2E latency", "ms", formatMilliseconds(ms), ms, false)
      }
      "max_memory_usage" -> {
        val bytes = payload.doubleOrNull("max_ram_bytes") ?: return null
        CompletedRunMetric("Max memory", "bytes", ByteFormat.fileSize(bytes.toLong()), bytes, false)
      }
      "vl_throughput" -> {
        val promptMs = payload.doubleOrNull("prompt_ms") ?: return null
        val predictedMs = payload.doubleOrNull("predicted_ms") ?: return null
        if (promptMs + predictedMs <= 0) return null
        val promptTokens = payload.doubleOrNull("prompt_tokens") ?: params.uint("parameter_text_tokens")
        val decodeTokens = params.uint("parameter_decode_tokens")
        val throughput = (promptTokens + decodeTokens) / ((promptMs + predictedMs) / 1000.0)
        CompletedRunMetric("VL throughput", "tok/s", formatNumber(throughput), throughput, true)
      }
      else -> null
    }
  }

  fun displayMetric(metric: CompletedRunMetric): String =
    when (metric.unit) {
      "ms",
      "bytes" -> metric.displayValue
      else -> "${metric.displayValue} ${metric.unit}".trim()
    }

  fun payloadsByCellId(storage: LocalStorage, manifest: JobManifest): Map<String, JSONObject> =
    manifest.cells
      .mapNotNull { cell ->
        val payload =
          runCatching {
              val file = storage.cellPayloadFile(manifest.jobId, cell.cellId)
              if (file.exists()) JSONObject(file.readText()) else null
            }
            .getOrNull()
        payload?.let { cell.cellId to it }
      }
      .toMap()

  fun quantLabel(cell: JobCell): String {
    val filename = File(cell.modelPath).name
    return (LocalStorage.parseQuant(filename) ?: "unknown").lowercase().replace("_k_m", "_km")
  }

  fun resultModelGroupKey(cell: JobCell): String {
    val filename = File(cell.modelPath).name
    if (filename.isNotBlank()) return LocalStorage.normalizedModelStem(filename)
    return modelDisplayName(cell).lowercase()
  }

  fun modelDisplayName(cell: JobCell): String {
    val modelKey = resultModelGroupKeyFromPath(cell.modelPath)
    ModelTemplateCatalog.byFamilyId[modelKey]?.let {
      return it.displayName
    }
    ModelTemplateCatalog.repoToName[cell.modelName]?.let {
      return it
    }
    if (cell.modelName.contains("/")) {
      return cell.modelName.substringAfterLast('/').removeSuffix("-GGUF").replace('-', ' ')
    }
    if (cell.modelName.endsWith(".gguf", ignoreCase = true)) {
      return LocalStorage.modelStem(File(cell.modelPath).name)
    }
    return cell.modelName
  }

  fun parameterSummary(benchmarkId: String): String? {
    val item = BenchmarkCatalog.resolve(benchmarkId) ?: return null
    val params = item.rawJson
    return when (item.benchmarkType) {
      "prefill_throughput",
      "max_memory_usage" -> {
        val prefill = params.uint("parameter_prefill_tokens")
        if (prefill > 0) csvNumber(prefill) else null
      }
      "decode_throughput",
      "end_to_end_latency" -> {
        val prefill = params.uint("parameter_prefill_tokens")
        val decode = params.uint("parameter_decode_tokens")
        if (prefill > 0 && decode > 0) "${csvNumber(prefill)}-${csvNumber(decode)}" else null
      }
      "vl_throughput" -> {
        val width = params.uint("parameter_image_width")
        val height = params.uint("parameter_image_height")
        if (width > 0 && height > 0) "${csvNumber(width)}x${csvNumber(height)}" else null
      }
      else -> null
    }
  }

  fun benchmarkType(cell: JobCell): String = cell.benchmarkType ?: BenchmarkCatalog.resolve(cell.benchmarkId)?.benchmarkType ?: cell.benchmarkId

  private fun benchmarkParams(cell: JobCell): JSONObject = BenchmarkCatalog.resolve(cell.benchmarkId)?.rawJson ?: JSONObject()

  private fun csvCells(manifest: JobManifest): List<JobCell> {
    val columns = orderedUnique(manifest.cells.map { it.benchmarkId })
    val modelOrder = mutableListOf<String>()
    val byModel = linkedMapOf<String, MutableList<JobCell>>()
    manifest.cells.forEach { cell ->
      val modelKey = resultModelGroupKey(cell)
      if (!byModel.containsKey(modelKey)) modelOrder += modelKey
      byModel.getOrPut(modelKey) { mutableListOf() } += cell
    }

    val ordered = mutableListOf<JobCell>()
    modelOrder.forEach { modelKey ->
      val modelCells = byModel[modelKey].orEmpty()
      val quantOrder = mutableListOf<String>()
      val byQuant = linkedMapOf<String, MutableList<JobCell>>()
      modelCells.forEach { cell ->
        val quant = quantLabel(cell)
        if (!byQuant.containsKey(quant)) quantOrder += quant
        byQuant.getOrPut(quant) { mutableListOf() } += cell
      }
      quantOrder.forEach { quant ->
        val quantCells = byQuant[quant].orEmpty()
        columns.forEach { column -> ordered += quantCells.filter { it.benchmarkId == column } }
      }
    }
    return ordered
  }

  private fun orderedUnique(values: List<String>): List<String> {
    val seen = mutableSetOf<String>()
    return values.filter { seen.add(it) }
  }

  private fun resultModelGroupKeyFromPath(path: String): String {
    val filename = File(path).name
    return if (filename.isNotBlank()) LocalStorage.normalizedModelStem(filename) else ""
  }

  private fun csvLine(values: List<String>): String = values.joinToString(",") { csvEscape(it) }

  private fun csvEscape(value: String): String =
    if (value.any { it == '"' || it == ',' || it == '\n' || it == '\r' }) {
      "\"${value.replace("\"", "\"\"")}\""
    } else {
      value
    }

  private fun payloadString(payload: JSONObject?, key: String): String = scalarString(payload?.opt(key)) ?: ""

  private fun scalarString(value: Any?): String? {
    if (value == null || value == JSONObject.NULL) return null
    return when (value) {
      is String -> value
      is Boolean -> if (value) "true" else "false"
      is Number -> csvNumber(value.toDouble())
      else -> null
    }
  }

  private fun formatNumber(value: Double): String = if (value >= 100.0) value.toLong().toString() else String.format(Locale.US, "%.1f", value)

  private fun formatMilliseconds(value: Double): String = if (value >= 100.0) "${value.toLong()} ms" else String.format(Locale.US, "%.1f ms", value)

  private fun csvNumber(value: Double): String {
    if (!value.isFinite()) return ""
    val rounded = kotlin.math.round(value)
    if (abs(rounded - value) < 0.0001) return rounded.toLong().toString()
    return String.format(Locale.US, "%.6g", value)
  }

  private fun JSONObject.doubleOrNull(key: String): Double? {
    if (!has(key) || isNull(key)) return null
    return when (val value = opt(key)) {
      is Number -> value.toDouble()
      is String -> value.toDoubleOrNull()
      else -> null
    }
  }

  private fun JSONObject.uint(key: String): Double {
    if (!has(key) || isNull(key)) return 0.0
    return when (val value = opt(key)) {
      is Number -> value.toDouble().coerceAtLeast(0.0)
      is String -> value.toDoubleOrNull()?.coerceAtLeast(0.0) ?: 0.0
      else -> 0.0
    }
  }
}
