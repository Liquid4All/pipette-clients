package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The compact `model=` URI grammar, against `crates/pipette-cli/src/model_uri.rs`.
 *
 * Every refusal has its own case, because the value of this parser is which refusal a caller gets. A `rev=` or `sha256=` accepted and ignored would
 * run bytes the caller did not pin and then submit the row as though it had, so those two are the cases most worth pinning.
 */
class ModelUriTest {
  private fun ok(raw: String): Model {
    val result = ModelUri.parse(raw)
    assertTrue("expected a coordinate from '$raw', got ${(result as? ModelUri.Result.Error)?.failure?.reason}", result is ModelUri.Result.Ok)
    return (result as ModelUri.Result.Ok).model
  }

  private fun failure(raw: String): ModelUri.Failure {
    val result = ModelUri.parse(raw)
    assertTrue("expected '$raw' to be refused", result is ModelUri.Result.Error)
    return (result as ModelUri.Result.Error).failure
  }

  @Test
  fun `a text coordinate parses to its repo and filename`() {
    val model = ok("gguf-text://repo=LiquidAI/LFM2.5-230M-GGUF&path=LFM2.5-230M-Q4_0.gguf")
    val text = (model as Model.GgufText).model

    assertEquals("LiquidAI/LFM2.5-230M-GGUF", text.repo.toString())
    assertEquals("LFM2.5-230M-Q4_0.gguf", text.filename.toString())
  }

  /** Keys are unordered, which the crate states and a plan emitting them in any order relies on. */
  @Test
  fun `key order does not matter`() {
    val model = ok("gguf-text://path=LFM2.5-230M-Q4_0.gguf&repo=LiquidAI/LFM2.5-230M-GGUF")

    assertEquals("LFM2.5-230M-Q4_0.gguf", (model as Model.GgufText).model.filename.toString())
  }

  @Test
  fun `a vision coordinate carries both halves`() {
    val model = ok("gguf-vision://repo=LiquidAI/LFM2-VL-450M-GGUF&model=LFM2-VL-450M-Q4_0.gguf&mmproj=mmproj-LFM2-VL-450M-f16.gguf")
    val vision = (model as Model.GgufVision).model

    assertEquals("LiquidAI/LFM2-VL-450M-GGUF", vision.repo.toString())
    assertEquals("LFM2-VL-450M-Q4_0.gguf", vision.filename.toString())
    assertEquals("mmproj-LFM2-VL-450M-f16.gguf", vision.mmprojFilename.toString())
  }

  /** The likeliest mistake, and the one whose message has to point at `match=` rather than explain a grammar the caller was not using. */
  @Test
  fun `a bare name is not a uri and says where names belong`() {
    val failure = failure("LFM2.5-230M")

    assertTrue(failure is ModelUri.Failure.NotAUri)
    assertTrue(failure.reason, failure.reason.contains("match="))
  }

  @Test
  fun `an engine this build does not link is refused by name`() {
    val failure = failure("mlx://repo=mlx-community/Qwen3.5-0.8B-4bit")

    assertTrue(failure is ModelUri.Failure.UnsupportedScheme)
    assertTrue(failure.reason, failure.reason.contains("does not link"))

    val onnx = failure("onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx")

    assertTrue(onnx is ModelUri.Failure.UnsupportedScheme)
    assertTrue(onnx.reason, onnx.reason.contains("does not link"))
  }

  /** The crate separates a real-but-non-importable source from an unknown one, so the taxonomy is mirrored rather than flattened. */
  @Test
  fun `a non-importable source is not reported as an unknown scheme`() {
    assertTrue(failure("apple-foundation://") is ModelUri.Failure.NotImportable)
    assertTrue(failure("local://repo=a/b") is ModelUri.Failure.NotImportable)
    assertTrue(failure("nonsense://repo=a/b") is ModelUri.Failure.UnknownScheme)
  }

  /**
   * The important pair. A pin this client cannot honour must be refused, not dropped: running unpinned bytes and submitting them as pinned is a worse
   * outcome than refusing the invocation.
   */
  @Test
  fun `a pinned digest or revision is refused rather than ignored`() {
    val digest = failure("gguf-text://repo=a/b&path=c.gguf&sha256=abc123")
    val revision = failure("gguf-text://repo=a/b&path=c.gguf&rev=main")

    assertTrue(digest is ModelUri.Failure.NotRepresentable)
    assertTrue(digest.reason, digest.reason.contains("digest"))
    assertTrue(revision is ModelUri.Failure.NotRepresentable)
    assertTrue(revision.reason, revision.reason.contains("revision"))
  }

  /** `Model` has no absolute-URL arm on this client, exactly as on iOS, so the URL spellings cannot be honoured either. */
  @Test
  fun `a url source is refused`() {
    assertTrue(failure("gguf-text://url=https://example.invalid/m.gguf") is ModelUri.Failure.NotRepresentable)
  }

  @Test
  fun `a missing required key names the key`() {
    assertTrue(failure("gguf-text://repo=a/b") is ModelUri.Failure.MissingKey)
    assertTrue(failure("gguf-text://path=c.gguf") is ModelUri.Failure.MissingKey)
    assertTrue(failure("gguf-vision://repo=a/b&model=c.gguf") is ModelUri.Failure.MissingKey)
    assertTrue(failure("gguf-text://") is ModelUri.Failure.MissingKey)
  }

  @Test
  fun `an unknown key is reported as unknown`() {
    val failure = failure("gguf-text://repo=a/b&path=c.gguf&quant=Q4_0")

    assertTrue(failure is ModelUri.Failure.UnknownKey)
    assertTrue(failure.reason, failure.reason.contains("quant"))
  }

  /** Last-wins would let a script run a model it did not name, so a repeat is an error. */
  @Test
  fun `a duplicate key is refused rather than resolved`() {
    assertTrue(failure("gguf-text://repo=a/b&path=c.gguf&path=d.gguf") is ModelUri.Failure.DuplicateKey)
  }

  @Test
  fun `a pair with no equals is malformed`() {
    assertTrue(failure("gguf-text://repo=a/b&path") is ModelUri.Failure.MalformedPair)
  }

  @Test
  fun `an unparseable value reports the key it came from`() {
    val repo = failure("gguf-text://repo=notaslug&path=c.gguf")
    val file = failure("gguf-text://repo=a/b&path=weights.bin")

    assertTrue(repo is ModelUri.Failure.InvalidValue)
    assertTrue(repo.reason, repo.reason.contains("repo"))
    assertTrue(file is ModelUri.Failure.InvalidValue)
    assertTrue(file.reason, file.reason.contains(".gguf"))
  }

  /** The scheme splits on the FIRST `://`, which is what lets a value hold one. */
  @Test
  fun `the scheme splits on the first separator`() {
    assertTrue(failure("gguf-text://url=file:///models/m.gguf") is ModelUri.Failure.NotRepresentable)
  }

  @Test
  fun `surrounding whitespace is tolerated`() {
    assertEquals("a/b", ((ok("  gguf-text://repo=a/b&path=c.gguf  ")) as Model.GgufText).model.repo.toString())
  }
}
