package ai.liquid.pipette.onnx

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests running under CheckJNI against the ONNX Runtime GenAI JNI (`android/native/onnx_genai_jni.cpp`), the counterpart of
 * `LlamaCppCheckJniTest`. CheckJNI aborts the process on a JNI call made with an exception pending, a bad array region or a leaked reference, so what
 * these exercise is every primitive's failure path as much as its success: each engine error has to come back as one Java exception and leave the
 * runtime usable.
 *
 * CheckJNI is on by default on an emulator; on a phone, `adb shell setprop debug.checkjni 1` before the run. The package is found as
 * [OnnxGenaiBenchmarkInstrumentedTest] finds it.
 */
@RunWith(AndroidJUnit4::class)
class OnnxGenaiCheckJniTest {
  private val arguments = InstrumentationRegistry.getArguments()

  private fun requirePackage(): File {
    assumeTrue("this build has no ONNX Runtime GenAI engine", OnnxGenai.isBuilt)
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val candidates =
      listOfNotNull(
        arguments.getString("onnxModelDir")?.let { File(it) },
        File(context.getExternalFilesDir(null), PHI3_PACKAGE),
        File(context.filesDir, PHI3_PACKAGE),
      )
    val dir = candidates.firstOrNull { OnnxGenai.isPackage(it.path) }
    assumeTrue("ONNX Runtime GenAI package not found", dir != null)
    return checkNotNull(dir)
  }

  /** An engine error crosses JNI as one exception, with the engine's own reason, and the next call works. */
  @Test
  fun testALoadThatFailsThrowsAndLeavesTheEngineUsable() {
    val dir = requirePackage()
    val context = InstrumentationRegistry.getInstrumentation().targetContext
    val empty = File(context.cacheDir, "not-a-package").apply { mkdirs() }

    val ex = assertThrows(IllegalStateException::class.java) { OnnxGenai.loadModel(empty.path) }
    assertTrue(ex.message.orEmpty(), ex.message.orEmpty().contains("genai_config.json"))

    OnnxGenai.loadModel(dir.path).use { model -> assertTrue(OnnxGenai.encode(model, "after a failed load").isNotEmpty()) }
  }

  /** Text crosses as bytes and ids come back as a region copy, whatever the text is: empty, multi-byte, or long. */
  @Test
  fun testEncodeCopiesItsRegions() {
    val dir = requirePackage()
    OnnxGenai.loadModel(dir.path).use { model ->
      val plain = OnnxGenai.encode(model, "The capital of France is")
      assertTrue(plain.isNotEmpty())
      assertEquals("the same text encodes the same way", plain.toList(), OnnxGenai.encode(model, "The capital of France is").toList())

      assertTrue(OnnxGenai.encode(model, "naïve café — 東京 🙂").isNotEmpty())
      OnnxGenai.encode(model, "")

      val long = OnnxGenai.encode(model, "word ".repeat(LONG_WORDS))
      assertTrue("a long text is not truncated: ${long.size}", long.size >= LONG_WORDS)
    }
  }

  /** A generation that cannot start is refused without a native call made in a bad state, and a closed model is refused before one is made. */
  @Test
  fun testAGenerationThatCannotRunIsRefused() {
    val dir = requirePackage()
    val model = OnnxGenai.loadModel(dir.path)
    val ids = OnnxGenai.encode(model, "The capital of France is")

    assertThrows(IllegalStateException::class.java) { OnnxGenai.generate(model, IntArray(0), 1) }
    assertThrows(IllegalArgumentException::class.java) { OnnxGenai.generate(model, ids, 0) }
    // Past the package's context length: the engine's error, not a crash.
    assertThrows(IllegalStateException::class.java) { OnnxGenai.generate(model, ids, OVER_CONTEXT) }

    // The model still works after each refusal.
    assertEquals(GENERATED, OnnxGenai.generate(model, ids, GENERATED).generatedTokens)

    model.close()
    assertTrue(model.isClosed)
    model.close()
    assertThrows(IllegalStateException::class.java) { OnnxGenai.encode(model, "closed") }
    assertThrows(IllegalStateException::class.java) { OnnxGenai.generate(model, ids, 1) }
  }

  /** Loading and freeing in a loop leaks no reference and leaves no handle behind. */
  @Test
  fun testRepeatedLoadAndFree() {
    val dir = requirePackage()
    repeat(CYCLES) {
      val model = OnnxGenai.loadModel(dir.path)
      assertFalse(model.isClosed)
      assertEquals(GENERATED, OnnxGenai.generate(model, OnnxGenai.encode(model, "Hello"), GENERATED).generatedTokens)
      model.close()
    }
  }

  private companion object {
    const val PHI3_PACKAGE = "models/microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32"
    const val LONG_WORDS = 2000
    const val GENERATED = 4
    const val CYCLES = 3
    // More tokens than any package's context: Phi-3's is 4096.
    const val OVER_CONTEXT = 1_000_000
  }
}
