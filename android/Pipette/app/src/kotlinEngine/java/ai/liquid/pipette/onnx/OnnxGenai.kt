package ai.liquid.pipette.onnx

import ai.liquid.pipette.NativeLib
import ai.liquid.pipette.OnnxGenaiPackage
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicBoolean

/** A loaded ONNX Runtime GenAI package: the model and the tokenizer built from it. */
class OnnxGenaiModel(val handle: Long, val path: String) : AutoCloseable {
  private val closed = AtomicBoolean(false)
  val isClosed: Boolean
    get() = closed.get()

  override fun close() {
    if (closed.compareAndSet(false, true)) {
      OnnxGenai.rawModelFree(handle)
    }
  }
}

/**
 * What one generation shows. The engine has no prefill/decode split: `AppendTokens` may defer the prefill to the first `GenerateNextToken`, so this
 * records what is observable either way. Mirrors iOS's `OnnxGenaiRuntime.GenerationTiming`, so an Android number and an iOS number measure the same
 * thing.
 */
data class GenerationTiming(
  val promptTokens: Int,
  val generatedTokens: Int,
  /** `AppendTokens` and the first `GenerateNextToken` together, since the prefill may run in either. */
  val ttftMs: Double,
  /** Mean per token after the first; null for a one-token generation. */
  val tpotMs: Double?,
) {
  /** Reconstructed from the intervals rather than wall clock, which would include tokenizing and building the generator. */
  val endToEndMs: Double
    get() = ttftMs + (tpotMs ?: 0.0) * maxOf(0, generatedTokens - 1)

  companion object {
    private const val NANOS_PER_MS = 1_000_000.0

    /** From the native call's three longs: nanoseconds to the first token, nanoseconds over the rest, tokens generated. */
    fun fromNative(promptTokens: Int, values: LongArray): GenerationTiming {
      check(values.size == NATIVE_FIELDS) { "the engine returned ${values.size} timing fields, expected $NATIVE_FIELDS" }
      val generated = values[2].toInt()
      val remaining = generated - 1
      return GenerationTiming(
        promptTokens = promptTokens,
        generatedTokens = generated,
        ttftMs = values[0] / NANOS_PER_MS,
        tpotMs = if (remaining > 0) values[1] / NANOS_PER_MS / remaining else null,
      )
    }

    private const val NATIVE_FIELDS = 3
  }
}

/** Direct Kotlin/JNI primitive layer over ONNX Runtime GenAI for the kotlinEngine flavor (`android/native/onnx_genai_jni.cpp`). CPU only. */
@Suppress("TooManyFunctions") // one per native primitive plus its checked wrapper, as LlamaCpp is
object OnnxGenai {
  /** The file every ONNX Runtime GenAI package has, and a plain ONNX export has not. */
  const val CONFIG_FILE = OnnxGenaiPackage.CONFIG_FILE

  /** Whether the native library was built with the engine (`build-onnx-genai-android.sh`). */
  val isBuilt: Boolean
    get() = NativeLib.isAvailable && nativeIsBuilt()

  /** The onnxruntime release the engine was built from, as in the submitted stack. */
  fun onnxRuntimeVersion(): String = nativeOnnxRuntimeVersion()

  /** The onnxruntime-genai commit the engine was built from: the pin has no release tag. */
  fun genaiCommit(): String = nativeGenaiCommit()

  /** Whether [path] is an ONNX Runtime GenAI package rather than a GGUF file: the directory that holds a [CONFIG_FILE]. */
  fun isPackage(path: String): Boolean = OnnxGenaiPackage.isPackage(path)

  fun loadModel(path: String): OnnxGenaiModel {
    check(isBuilt) { "this build has no ONNX Runtime GenAI engine" }
    val handle = nativeModelLoad(path)
    check(handle != 0L) { "nativeModelLoad returned null handle" }
    return OnnxGenaiModel(handle, path)
  }

  internal fun rawModelFree(handle: Long) {
    if (handle != 0L) {
      nativeModelFree(handle)
    }
  }

  /** [text]'s token ids under the model's own tokenizer, with no chat template. */
  fun encode(model: OnnxGenaiModel, text: String): IntArray {
    check(!model.isClosed) { "model is closed" }
    val tokens = nativeEncode(model.handle, text.toByteArray(StandardCharsets.UTF_8))
    return checkNotNull(tokens) { "encode returned null" }
  }

  /** One generation of exactly [decodeTokens] tokens over [promptIds], timed in native code. */
  fun generate(model: OnnxGenaiModel, promptIds: IntArray, decodeTokens: Int): GenerationTiming {
    check(!model.isClosed) { "model is closed" }
    require(decodeTokens > 0) { "a generation of $decodeTokens tokens measures nothing" }
    val values = checkNotNull(nativeGenerate(model.handle, promptIds, decodeTokens)) { "generate returned null" }
    val timing = GenerationTiming.fromNative(promptIds.size, values)
    check(timing.generatedTokens == decodeTokens) { "the engine generated ${timing.generatedTokens} tokens, expected $decodeTokens" }
    return timing
  }

  // ===========================================================================
  // Native JNI Declarations
  // ===========================================================================

  @JvmStatic private external fun nativeIsBuilt(): Boolean

  @JvmStatic private external fun nativeOnnxRuntimeVersion(): String

  @JvmStatic private external fun nativeGenaiCommit(): String

  @JvmStatic private external fun nativeModelLoad(path: String): Long

  @JvmStatic private external fun nativeModelFree(handle: Long)

  @JvmStatic private external fun nativeEncode(handle: Long, textUtf8: ByteArray): IntArray?

  @JvmStatic private external fun nativeGenerate(handle: Long, promptIds: IntArray, decodeTokens: Int): LongArray?
}
