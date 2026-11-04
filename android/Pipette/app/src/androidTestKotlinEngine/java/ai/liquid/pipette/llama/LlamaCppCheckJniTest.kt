package ai.liquid.pipette.llama

import ai.liquid.pipette.LlamacppLoadMode
import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests running under CheckJNI against the kotlinEngine C++ backend. Validates reference discipline, direct ByteBuffer bounds safety,
 * and stress decode stability.
 */
@RunWith(AndroidJUnit4::class)
class LlamaCppCheckJniTest {
  private val tag = "LlamaCppCheckJniTest"

  private fun findTestModel(context: Context): File? {
    val candidates =
      listOf(
        File(context.filesDir, "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File(context.filesDir, "LFM2.5-350M-Q4_0.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf"),
        File(context.getExternalFilesDir(null), "models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q5_K_M.gguf"),
        File("/sdcard/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
        File("/storage/emulated/0/Android/data/ai.liquid.pipette/files/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_0.gguf"),
      )
    return candidates.firstOrNull { it.exists() && it.canRead() }
  }

  @Test
  fun smokeTestLoadTokenizePrefillDecodeFree() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found under external files directory", modelFile != null)
    checkNotNull(modelFile)

    Log.i(tag, "Loading model: ${modelFile.absolutePath}")
    val model = LlamaCpp.loadModel(path = modelFile.absolutePath, nGpuLayers = 0, contextSize = 512, nUbatch = 128, loadMode = LlamacppLoadMode.READ)

    try {
      assertFalse(model.isClosed)
      val threads = LlamaCpp.nThreads(model)
      assertTrue("nThreads must be positive: $threads", threads > 0)

      val prompt = "The capital of France is"
      val tokens = LlamaCpp.tokenize(model, prompt, addSpecial = true)
      assertTrue("Tokenized tokens must not be empty", tokens.isNotEmpty())
      Log.i(tag, "Prompt '$prompt' tokenized to ${tokens.size} tokens")

      LlamaCpp.prefill(model, tokens)

      val directBuf = ByteBuffer.allocateDirect(128)
      val accumulator = RuneAccumulator()
      val generatedText = StringBuilder()

      for (i in 0 until 8) {
        val nextToken = LlamaCpp.sampleGreedy(model)
        assertTrue("Sampled token must be non-negative", nextToken >= 0)

        // Test direct ByteBuffer detokenization
        directBuf.clear()
        val bytesWritten = LlamaCpp.tokenToBuffer(model, nextToken, directBuf, 0, directBuf.capacity())
        assertTrue("Direct buffer detokenize must write bytes", bytesWritten >= 0)

        // Test ByteArray detokenization with RuneAccumulator
        val piece = LlamaCpp.tokenToPiece(model, nextToken, accumulator)
        generatedText.append(piece)

        val rc = LlamaCpp.decodeBatch(model, intArrayOf(nextToken))
        assertEquals(0, rc)
      }

      generatedText.append(accumulator.flush())
      Log.i(tag, "Generated 8 tokens: '$generatedText'")
      assertTrue("Generated output should not be empty", generatedText.isNotEmpty())
    } finally {
      model.close()
      assertTrue(model.isClosed)
    }
  }

  @Test
  fun directByteBufferSafetyValidation() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val model = LlamaCpp.loadModel(path = modelFile.absolutePath, nGpuLayers = 0, contextSize = 256, nUbatch = 128)

    try {
      val tokens = LlamaCpp.tokenize(model, "Hello", addSpecial = true)
      LlamaCpp.prefill(model, tokens)
      val token = LlamaCpp.sampleGreedy(model)

      // 1. Non-direct buffer must throw IllegalArgumentException
      val nonDirectBuf = ByteBuffer.allocate(64)
      assertThrows(IllegalArgumentException::class.java) { LlamaCpp.tokenToBuffer(model, token, nonDirectBuf, 0, 64) }

      // 2. Out of bounds offset/capacity must throw IllegalArgumentException
      val directBuf = ByteBuffer.allocateDirect(32)
      assertThrows(IllegalArgumentException::class.java) { LlamaCpp.tokenToBuffer(model, token, directBuf, -1, 16) }
      assertThrows(IllegalArgumentException::class.java) {
        LlamaCpp.tokenToBuffer(model, token, directBuf, 16, 20) // 16 + 20 > 32
      }

      // 3. Valid direct buffer succeeds
      val rc = LlamaCpp.tokenToBuffer(model, token, directBuf, 0, 32)
      assertTrue("tokenToBuffer should return byte length >= 0", rc >= 0)
    } finally {
      model.close()
    }
  }

  @Test
  fun stressDecodeNativeAndJniHopUnderCheckJni() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val model = LlamaCpp.loadModel(path = modelFile.absolutePath, nGpuLayers = 0, contextSize = 2048, nUbatch = 512)

    try {
      val prompt = "Stress test prompt for CheckJNI verification."
      val tokens = LlamaCpp.tokenize(model, prompt, addSpecial = true)
      LlamaCpp.prefill(model, tokens)

      Log.i(tag, "Starting native decodeNGreedyIgnoreEog for 256 tokens")
      val rc = LlamaCpp.decodeNGreedyIgnoreEog(model, 256)
      assertEquals(0, rc)

      Log.i(tag, "Starting JNI-hopping decodeBatch loop for 64 tokens")
      for (i in 0 until 64) {
        val nextToken = LlamaCpp.sampleGreedy(model)
        val decodeRc = LlamaCpp.decodeBatch(model, intArrayOf(nextToken))
        assertEquals(0, decodeRc)
      }

      Log.i(tag, "Stress decode completed with zero JNI errors")
    } finally {
      model.close()
    }
  }

  @Test
  fun chatTemplateFormattingWithUtf8Characters() {
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val modelFile = findTestModel(context)
    assumeTrue("Test GGUF model not found", modelFile != null)
    checkNotNull(modelFile)

    val model = LlamaCpp.loadModel(path = modelFile.absolutePath, nGpuLayers = 0, contextSize = 512, nUbatch = 128)
    try {
      val messagesJson = """[{"role":"user","content":"Hello world! 🚀 🌟 Testing emojis and accents: café, naïve."}]"""
      val templated = LlamaCpp.applyChatTemplate(model, messagesJson)
      assertTrue("Templated text must contain input content", templated.contains("Hello world! 🚀 🌟 Testing emojis and accents: café, naïve."))
      Log.i(tag, "Templated output successfully: $templated")
    } finally {
      model.close()
    }
  }
}
