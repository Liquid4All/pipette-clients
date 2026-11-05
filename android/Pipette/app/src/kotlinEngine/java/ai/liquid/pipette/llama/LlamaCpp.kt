package ai.liquid.pipette.llama

import ai.liquid.pipette.LlamacppLoadMode
import ai.liquid.pipette.NativeLib
import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

enum class BenchmarkEvalCompletionStopReason(val wireValue: String) {
  EOS("eos"),
  TRUNCATED("truncated"),
  FAILURE("failure"),
  UNKNOWN("unknown");

  override fun toString(): String = wireValue
}

data class Generation(val text: String, val stopReason: BenchmarkEvalCompletionStopReason, val stopDetail: String?, val completionTokens: Long)

sealed class RawStop {
  object StopToken : RawStop()

  object CapReached : RawStop()
}

fun classifyStop(stop: RawStop): BenchmarkEvalCompletionStopReason =
  when (stop) {
    is RawStop.StopToken -> BenchmarkEvalCompletionStopReason.EOS
    is RawStop.CapReached -> BenchmarkEvalCompletionStopReason.TRUNCATED
  }

fun memoryPeakDelta(baseline: Long, peak: Long, finalSample: Long): Long {
  val highest = maxOf(peak, finalSample)
  return if (highest > baseline) highest - baseline else 0L
}

private val procSelfStatus = File("/proc/self/status")
private val whitespaceRegex = "\\s+".toRegex()

fun currentRssBytes(): Long? {
  val nativeBytes =
    try {
      LlamaCpp.nativeCurrentRssBytes()
    } catch (_: Throwable) {
      0L
    }
  if (nativeBytes > 0L) {
    return nativeBytes
  }
  return try {
    procSelfStatus.useLines { lines ->
      lines.firstNotNullOfOrNull { line ->
        if (line.startsWith("VmRSS:")) {
          val parts = line.substring(6).trim().split(whitespaceRegex)
          parts.firstOrNull()?.toLongOrNull()?.let { it * 1024L }
        } else {
          null
        }
      }
    }
  } catch (_: Throwable) {
    null
  }
}

typealias MetalAllocationPoller = MemoryAllocationPoller

fun metalAllocatedSizeBytes(): Long = currentRssBytes() ?: 0L

class MemoryAllocationPoller internal constructor(private val baseline: Long) {
  private val stopRequested = AtomicBoolean(false)
  private val peak = AtomicLong(baseline)
  private val thread =
    thread(start = true, isDaemon = true, name = "MemoryAllocationPoller") {
      while (!stopRequested.get()) {
        val current = currentRssBytes() ?: 0L
        peak.updateAndGet { prev -> maxOf(prev, current) }
        try {
          Thread.sleep(20)
        } catch (_: InterruptedException) {
          break
        }
      }
    }

  fun stopAndJoinWithSample(extraSample: Long = 0L): Long {
    stopRequested.set(true)
    thread.interrupt()
    try {
      thread.join()
    } catch (_: InterruptedException) {
      Thread.currentThread().interrupt()
    }
    val finalSample = maxOf(currentRssBytes() ?: 0L, extraSample)
    val maxPeak = peak.updateAndGet { prev -> maxOf(prev, finalSample) }
    return memoryPeakDelta(baseline, maxPeak, finalSample)
  }
}

fun spawnMemoryAllocationPoller(): MemoryAllocationPoller = MemoryAllocationPoller(currentRssBytes() ?: 0L)

fun spawnMetalAllocationPoller(): MemoryAllocationPoller = spawnMemoryAllocationPoller()

class LlamaModel(val handle: Long, val path: String, val nGpuLayers: Int, val contextSize: Int, val nUbatch: Int, val loadMode: LlamacppLoadMode) :
  AutoCloseable {
  private val closed = AtomicBoolean(false)
  val isClosed: Boolean
    get() = closed.get()

  override fun close() {
    if (closed.compareAndSet(false, true)) {
      LlamaCpp.rawModelFree(handle)
    }
  }
}

class MtmdContext(val handle: Long, val mmprojPath: String, val useGpu: Boolean) : AutoCloseable {
  private val closed = AtomicBoolean(false)
  val isClosed: Boolean
    get() = closed.get()

  override fun close() {
    if (closed.compareAndSet(false, true)) {
      LlamaCpp.rawMtmdFree(handle)
    }
  }
}

class MtmdChunks(val handle: Long, val totalTokens: Long, val imageTokens: Long) : AutoCloseable {
  private val closed = AtomicBoolean(false)
  val isClosed: Boolean
    get() = closed.get()

  override fun close() {
    if (closed.compareAndSet(false, true)) {
      LlamaCpp.rawMtmdFreeChunks(handle)
    }
  }
}

/** Direct Kotlin/JNI primitive layer wrapping llama.cpp for the kotlinEngine flavor. */
object LlamaCpp {
  init {
    check(NativeLib.isAvailable) { "libpipette_android.so is unavailable for LlamaCpp" }
  }

  fun loadModel(
    path: String,
    nGpuLayers: Int = 99,
    contextSize: Int = 0,
    nUbatch: Int = 512,
    loadMode: LlamacppLoadMode = LlamacppLoadMode.READ,
  ): LlamaModel {
    val handle = nativeModelLoad(path, nGpuLayers, contextSize, nUbatch, loadMode.cValue)
    check(handle != 0L) { "nativeModelLoad returned null handle" }
    return LlamaModel(handle, path, nGpuLayers, contextSize, nUbatch, loadMode)
  }

  fun modelFree(model: LlamaModel) {
    model.close()
  }

  internal fun rawModelFree(handle: Long) {
    if (handle != 0L) {
      nativeModelFree(handle)
    }
  }

  fun nThreads(model: LlamaModel): Int {
    check(!model.isClosed) { "model is closed" }
    return nativeNThreads(model.handle)
  }

  fun defaultThreads(): Int = nativeDefaultThreads()

  fun tokenize(model: LlamaModel, text: String, addSpecial: Boolean = false): IntArray {
    check(!model.isClosed) { "model is closed" }
    val textBytes = text.toByteArray(StandardCharsets.UTF_8)
    val tokens = nativeTokenize(model.handle, textBytes, addSpecial)
    checkNotNull(tokens) { "tokenization returned null" }
    return tokens
  }

  fun prefill(model: LlamaModel, tokens: IntArray) {
    check(!model.isClosed) { "model is closed" }
    if (tokens.isEmpty()) return
    val rc = nativeDecodeBatch(model.handle, tokens, 0, tokens.size)
    check(rc == 0) { "prefill failed with code $rc" }
  }

  fun decodeBatch(model: LlamaModel, tokens: IntArray, offset: Int = 0, length: Int = tokens.size): Int {
    check(!model.isClosed) { "model is closed" }
    return nativeDecodeBatch(model.handle, tokens, offset, length)
  }

  fun sampleGreedy(model: LlamaModel): Int {
    check(!model.isClosed) { "model is closed" }
    val token = nativeSampleGreedy(model.handle)
    check(token >= 0) { "sampleGreedy returned invalid token $token" }
    return token
  }

  fun tokenToStr(model: LlamaModel, token: Int): ByteArray {
    check(!model.isClosed) { "model is closed" }
    val bytes = nativeTokenToStr(model.handle, token)
    return bytes ?: ByteArray(0)
  }

  fun tokenToPiece(model: LlamaModel, token: Int, accumulator: RuneAccumulator): String {
    val bytes = tokenToStr(model, token)
    return accumulator.accumulate(bytes)
  }

  fun tokenToBuffer(model: LlamaModel, token: Int, buffer: ByteBuffer, offset: Int = 0, capacity: Int = buffer.capacity() - offset): Int {
    check(!model.isClosed) { "model is closed" }
    require(buffer.isDirect) { "buffer must be a direct ByteBuffer" }
    require(offset >= 0 && capacity > 0 && capacity <= buffer.capacity() - offset) { "offset and capacity exceed direct buffer bounds" }
    return nativeTokenToBuffer(model.handle, token, buffer, offset, capacity)
  }

  fun applyChatTemplate(model: LlamaModel, messagesJson: String): String {
    check(!model.isClosed) { "model is closed" }
    val jsonBytes = messagesJson.toByteArray(StandardCharsets.UTF_8)
    val bytes = nativeApplyChatTemplate(model.handle, jsonBytes)
    checkNotNull(bytes) { "applyChatTemplate returned null" }
    return String(bytes, StandardCharsets.UTF_8)
  }

  fun tokenEosId(model: LlamaModel): Int {
    check(!model.isClosed) { "model is closed" }
    return nativeTokenEosId(model.handle)
  }

  fun resetContext(model: LlamaModel) {
    check(!model.isClosed) { "model is closed" }
    nativeKvCacheClear(model.handle)
  }

  fun resetSampler(model: LlamaModel) {
    check(!model.isClosed) { "model is closed" }
    nativeSamplerReset(model.handle)
  }

  fun decodeNGreedyIgnoreEog(model: LlamaModel, count: Int): Int {
    check(!model.isClosed) { "model is closed" }
    return nativeDecodeNGreedyIgnoreEog(model.handle, count)
  }

  fun decodeGreedy(model: LlamaModel, maxTokens: Int, stopTokens: IntArray): Generation {
    check(!model.isClosed) { "model is closed" }
    val accumulator = RuneAccumulator()
    val output = StringBuilder()
    val singleToken = IntArray(1)
    var completionTokens = 0L
    var stop: RawStop = RawStop.CapReached

    // `completionTokens` steps once per full iteration, so it is also the budget.
    while (completionTokens < maxTokens) {
      val token = sampleGreedy(model)
      if (stopTokens.contains(token)) {
        stop = RawStop.StopToken
        break
      }

      completionTokens++
      val tokenBytes = tokenToStr(model, token)
      val piece = accumulator.accumulate(tokenBytes)
      output.append(piece)

      singleToken[0] = token
      val rc = decodeBatch(model, singleToken)
      check(rc == 0) { "decode failed with code $rc" }
    }

    output.append(accumulator.flush())
    return Generation(text = output.toString(), stopReason = classifyStop(stop), stopDetail = null, completionTokens = completionTokens)
  }

  /**
   * The MCQ-arm / streaming-arm decode that follows a populated KV cache, shared by [chatCompletion] (after a text [prefill]) and
   * [chatCompletionWithImage] (after an [mtmdEvalChunks]) — the decode loop only depends on KV state, not on how it got there.
   */
  private fun completeFromKv(model: LlamaModel, maxTokens: Int, mcqChoices: List<String>?): Generation {
    return if (mcqChoices != null) {
      val token = sampleGreedy(model)
      val bytes = tokenToStr(model, token)
      val generated = String(bytes, StandardCharsets.UTF_8).trim()

      val matched = mcqChoices.contains(generated)
      val text = if (matched) generated else (mcqChoices.firstOrNull() ?: "")
      val stopDetail =
        if (matched) {
          "mcq arm (single greedy token, matched a choice)"
        } else {
          "mcq arm (single greedy token, no choice matched; fell back to the first)"
        }

      Generation(text = text, stopReason = BenchmarkEvalCompletionStopReason.UNKNOWN, stopDetail = stopDetail, completionTokens = 1L)
    } else {
      val eos = tokenEosId(model)
      decodeGreedy(model, maxTokens, intArrayOf(eos))
    }
  }

  fun chatCompletion(model: LlamaModel, messagesJson: String, maxTokens: Int, mcqChoices: List<String>? = null): Generation {
    check(!model.isClosed) { "model is closed" }
    val prompt = applyChatTemplate(model, messagesJson)
    val tokens = tokenize(model, prompt, addSpecial = true)
    prefill(model, tokens)
    return completeFromKv(model, maxTokens, mcqChoices)
  }

  /**
   * Chat completion over one image: [templatableMessagesJson] is the sample's messages with every `image_url` content block replaced by the mtmd
   * media marker (a plain string `content`, so [applyChatTemplate] — which requires string content — can render it), and [imageBytes] is the
   * corresponding raw image file bytes. Mirrors [chatCompletion] but tokenizes and evaluates the prompt+image together via mtmd instead of the plain
   * tokenizer.
   */
  fun chatCompletionWithImage(
    model: LlamaModel,
    mtmd: MtmdContext,
    templatableMessagesJson: String,
    imageBytes: ByteArray,
    maxTokens: Int,
    mcqChoices: List<String>? = null,
  ): Generation {
    check(!model.isClosed) { "model is closed" }
    check(!mtmd.isClosed) { "mtmd context is closed" }
    val promptWithMarker = applyChatTemplate(model, templatableMessagesJson)
    val chunks = mtmdAllocChunksFromImage(mtmd, promptWithMarker, imageBytes)
    try {
      val rc = mtmdEvalChunks(mtmd, model, chunks)
      check(rc == 0) { "mtmd_eval_chunks failed with code $rc" }
    } finally {
      chunks.close()
    }
    return completeFromKv(model, maxTokens, mcqChoices)
  }

  fun mtmdInit(model: LlamaModel, mmprojPath: String, useGpu: Boolean = false): MtmdContext {
    check(!model.isClosed) { "model is closed" }
    val handle = nativeMtmdInit(model.handle, mmprojPath, useGpu)
    check(handle != 0L) { "nativeMtmdInit returned null handle for $mmprojPath" }
    return MtmdContext(handle, mmprojPath, useGpu)
  }

  fun mtmdFree(mtmd: MtmdContext) {
    mtmd.close()
  }

  internal fun rawMtmdFree(handle: Long) {
    if (handle != 0L) {
      nativeMtmdFree(handle)
    }
  }

  fun mtmdAllocGrayChunks(mtmd: MtmdContext, textWithMarker: String, width: Int, height: Int): MtmdChunks {
    check(!mtmd.isClosed) { "mtmd context is closed" }
    val textBytes = textWithMarker.toByteArray(StandardCharsets.UTF_8)
    val info = nativeMtmdAllocGrayChunks(mtmd.handle, textBytes, width, height)
    checkNotNull(info) { "mtmdAllocGrayChunks failed" }
    check(info.size >= 3) { "mtmdAllocGrayChunks returned incomplete info" }
    return MtmdChunks(info[0], info[1], info[2])
  }

  fun mtmdAllocChunksFromImage(mtmd: MtmdContext, textWithMarker: String, imageBytes: ByteArray): MtmdChunks {
    check(!mtmd.isClosed) { "mtmd context is closed" }
    val textBytes = textWithMarker.toByteArray(StandardCharsets.UTF_8)
    val info = nativeMtmdAllocChunksFromImage(mtmd.handle, textBytes, imageBytes)
    checkNotNull(info) { "mtmdAllocChunksFromImage failed" }
    check(info.size >= 3) { "mtmdAllocChunksFromImage returned incomplete info" }
    return MtmdChunks(info[0], info[1], info[2])
  }

  fun mtmdEvalChunks(mtmd: MtmdContext, model: LlamaModel, chunks: MtmdChunks): Int {
    check(!mtmd.isClosed) { "mtmd context is closed" }
    check(!model.isClosed) { "model is closed" }
    check(!chunks.isClosed) { "chunks is closed" }
    return nativeMtmdEvalChunks(mtmd.handle, model.handle, chunks.handle)
  }

  fun mtmdFreeChunks(chunks: MtmdChunks) {
    chunks.close()
  }

  internal fun rawMtmdFreeChunks(handle: Long) {
    if (handle != 0L) {
      nativeMtmdFreeChunks(handle)
    }
  }

  // ===========================================================================
  // Native JNI Declarations
  // ===========================================================================

  @JvmStatic external fun nativeCurrentRssBytes(): Long

  @JvmStatic private external fun nativeModelLoad(path: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: Int): Long

  @JvmStatic private external fun nativeModelFree(handle: Long)

  @JvmStatic private external fun nativeNThreads(handle: Long): Int

  @JvmStatic private external fun nativeDefaultThreads(): Int

  @JvmStatic private external fun nativeTokenize(handle: Long, textUtf8: ByteArray, addSpecial: Boolean): IntArray?

  @JvmStatic private external fun nativeDecodeBatch(handle: Long, tokens: IntArray, offset: Int, length: Int): Int

  @JvmStatic private external fun nativeSampleGreedy(handle: Long): Int

  @JvmStatic private external fun nativeTokenToStr(handle: Long, token: Int): ByteArray?

  @JvmStatic private external fun nativeTokenToBuffer(handle: Long, token: Int, directBuffer: ByteBuffer, offset: Int, capacity: Int): Int

  @JvmStatic private external fun nativeApplyChatTemplate(handle: Long, messagesJsonUtf8: ByteArray): ByteArray?

  @JvmStatic private external fun nativeTokenEosId(handle: Long): Int

  @JvmStatic private external fun nativeKvCacheClear(handle: Long)

  @JvmStatic private external fun nativeSamplerReset(handle: Long)

  @JvmStatic private external fun nativeDecodeNGreedyIgnoreEog(handle: Long, n: Int): Int

  @JvmStatic private external fun nativeMtmdInit(modelHandle: Long, mmprojPath: String, useGpu: Boolean): Long

  @JvmStatic private external fun nativeMtmdFree(mtmdHandle: Long)

  @JvmStatic private external fun nativeMtmdAllocGrayChunks(mtmdHandle: Long, textWithMarkerUtf8: ByteArray, width: Int, height: Int): LongArray?

  @JvmStatic private external fun nativeMtmdAllocChunksFromImage(mtmdHandle: Long, textWithMarkerUtf8: ByteArray, imageBytes: ByteArray): LongArray?

  @JvmStatic private external fun nativeMtmdEvalChunks(mtmdHandle: Long, modelHandle: Long, chunksPtr: Long): Int

  @JvmStatic private external fun nativeMtmdFreeChunks(chunksPtr: Long)
}
