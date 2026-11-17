package ai.liquid.pipette

/**
 * Which benchmark kinds the ONNX Runtime GenAI engine runs. Here, and not with the engine in the `kotlinEngine` source set, because the runner asks
 * before it fetches or loads a package: refused only inside the engine, a cell of a kind it does not run would cost a multi-gigabyte load to learn
 * so, and heat the device before the next timed cell.
 */
object OnnxGenaiCells {
  /** Why the engine refuses [benchmarkType], or null for the four kinds it runs: the four iOS runs. */
  fun unsupportedReason(benchmarkType: String): String? =
    when (benchmarkType) {
      "prefill_throughput",
      "decode_throughput",
      "end_to_end_latency",
      "max_memory_usage" -> null
      // A borrowed renderer would template with one tokenizer and generate with another: a prompt no other runtime ran.
      "eval" -> "ONNX Runtime GenAI's tokenizer applies no chat template, so this engine does not run eval"
      else -> "the ONNX Runtime GenAI engine does not run $benchmarkType"
    }
}
