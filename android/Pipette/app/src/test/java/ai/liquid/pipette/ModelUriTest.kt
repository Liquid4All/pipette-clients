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

    val openvino = failure("openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw")

    assertTrue(openvino is ModelUri.Failure.UnsupportedScheme)
    assertTrue(openvino.reason, openvino.reason.contains("does not link"))
  }

  /** The directory arm, as the crate's `parse_dir_source` reads it: a repo, and the subdirectory of it that holds the package. */
  @Test
  fun `an onnx-genai coordinate parses to its repo and prefix`() {
    val model = ok("onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32")
    val onnx = (model as Model.OnnxGenai).model

    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx", onnx.repo.toString())
    assertEquals("cpu_and_mobile/cpu-int4-rtn-block-32", onnx.prefix.toString())
    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32", onnx.relativeDir)
    assertEquals("cpu-int4-rtn-block-32", onnx.name)
  }

  /** A package at the root of its repo names no prefix, and is then named for the repo. */
  @Test
  fun `an onnx-genai coordinate needs no prefix`() {
    val onnx = (ok("onnx-genai://repo=org/some-genai-package") as Model.OnnxGenai).model

    assertEquals(null, onnx.prefix)
    assertEquals("org/some-genai-package", onnx.relativeDir)
    assertEquals("some-genai-package", onnx.name)
  }

  /**
   * A prefix is joined onto a path under the models directory that a delete later removes, so one that climbs out is refused here, before it is a
   * path at all. A GGUF key is refused rather than ignored, and a digest is refused as it is on every scheme.
   */
  @Test
  fun `an onnx-genai coordinate refuses a traversal, a GGUF key and a digest`() {
    for (prefix in listOf("../other", "a/../b", "/absolute", "a//b", "a/", ".")) {
      val bad = failure("onnx-genai://repo=org/name&prefix=$prefix")
      assertTrue("$prefix: ${bad.reason}", bad is ModelUri.Failure.InvalidValue)
    }

    val ggufKey = failure("onnx-genai://repo=org/name&path=model.gguf")
    assertTrue(ggufKey.reason, ggufKey is ModelUri.Failure.UnknownKey)

    val digest = failure("onnx-genai://repo=org/name&sha256=abc123")
    assertTrue(digest.reason, digest is ModelUri.Failure.NotRepresentable)

    val badRevision = failure("onnx-genai://repo=org/name&rev=-leading-dash")
    assertTrue(badRevision.reason, badRevision is ModelUri.Failure.InvalidValue)

    val noRepo = failure("onnx-genai://prefix=a/b")
    assertTrue(noRepo.reason, noRepo is ModelUri.Failure.MissingKey)
  }

  /**
   * The one pin this client honours: a directory model's revision, which the catalog's own row carries so a resumed multi-gigabyte pull cannot meet a
   * tree that moved on `main`. On a GGUF coordinate `rev` is still refused, as the pin there would be accepted and ignored.
   */
  @Test
  fun `an onnx-genai coordinate keeps its revision`() {
    val commit = "5f5f794c1c23c9d5ee142af85df02a6cc52d6945"
    val onnx = (ok("onnx-genai://repo=org/name&prefix=a/b&rev=$commit") as Model.OnnxGenai).model

    assertEquals(commit, onnx.revision.toString())
    assertEquals("the pin is not part of where the package sits", "org/name/a/b", onnx.relativeDir)

    val gguf = failure("gguf-text://repo=org/name&path=model.gguf&rev=$commit")
    assertTrue(gguf.reason, gguf is ModelUri.Failure.NotRepresentable)
  }

  /** `prefix` is a directory model's key alone: on a GGUF coordinate it stays the refusal it was. */
  @Test
  fun `a prefix on a GGUF coordinate is still refused`() {
    val failure = failure("gguf-text://repo=org/name&path=model.gguf&prefix=a/b")

    assertTrue(failure.reason, failure is ModelUri.Failure.NotRepresentable)
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
