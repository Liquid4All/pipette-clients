package ai.liquid.pipette

import java.io.ByteArrayOutputStream
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * What makes a directory an ONNX Runtime GenAI package this client will load, against iOS's `OnnxGenaiModels.validatePackage`.
 *
 * The graphs here are a few bytes of hand-written protobuf, the smallest that carry what the check reads: a `ModelProto` whose graph has an
 * initializer naming an external-data file.
 */
class OnnxGenaiPackageTest {
  @get:Rule val temp = TemporaryFolder()

  private fun write(dir: File, name: String, text: String) = File(dir, name).apply { parentFile?.mkdirs() }.writeText(text)

  /** A whole package: config, graph, the sidecar the graph names, and the tokenizer's two files. */
  private fun wholePackage(name: String = "package"): File {
    val dir = temp.newFolder(name)
    write(dir, OnnxGenaiPackage.CONFIG_FILE, """{"model": {"decoder": {"filename": "model.onnx"}}}""")
    File(dir, "model.onnx").writeBytes(OnnxGraphBytes.graphNaming("model.onnx.data"))
    write(dir, "model.onnx.data", "weights")
    write(dir, "tokenizer_config.json", "{}")
    write(dir, "tokenizer.json", "{}")
    return dir
  }

  private fun refusal(dir: File): String = assertThrows(OnnxGenaiPackage.Invalid::class.java) { OnnxGenaiPackage.validate(dir) }.message!!

  @Test
  fun `a whole package passes`() {
    val dir = wholePackage()

    OnnxGenaiPackage.validate(dir)
    assertTrue(OnnxGenaiPackage.isPackage(dir.path))
    assertTrue(OnnxGenaiPackage.isValid(dir))
  }

  /** The case most LiquidAI ONNX repos are: real graphs, no `genai_config.json`. Named, so the reader learns which kind of ONNX this is. */
  @Test
  fun `a plain ONNX export is refused by name`() {
    val dir = temp.newFolder("plain")
    File(dir, "model.onnx").writeBytes(OnnxGraphBytes.graphNaming("model.onnx.data"))

    assertFalse(OnnxGenaiPackage.isPackage(dir.path))
    assertTrue(refusal(dir).contains("transformers.js"))
  }

  /** The sidecar is the multi-gigabyte half a truncated transfer loses, and without this check the graph loads and fails at the first inference. */
  @Test
  fun `a graph missing its external data is refused`() {
    val dir = wholePackage()
    File(dir, "model.onnx.data").delete()

    val reason = refusal(dir)
    assertTrue(reason, reason.contains("model.onnx.data"))
    assertTrue(reason, reason.contains("incomplete"))
  }

  @Test
  fun `a config that names no graph, or a graph that is absent, is refused`() {
    val noGraph = wholePackage("no-graph")
    write(noGraph, OnnxGenaiPackage.CONFIG_FILE, """{"model": {}}""")
    assertTrue(refusal(noGraph).contains("model.decoder.filename"))

    val absent = wholePackage("absent")
    File(absent, "model.onnx").delete()
    assertTrue(refusal(absent).contains("missing the decoder graph"))

    val notJson = wholePackage("not-json")
    write(notJson, OnnxGenaiPackage.CONFIG_FILE, "not json")
    assertTrue(refusal(notJson).contains("does not parse"))
  }

  /** Metadata paths are untrusted: one that climbs out of the package is refused rather than followed, in the config and in the graph. */
  @Test
  fun `a path that leaves the package is refused`() {
    val config = wholePackage("config")
    write(config, OnnxGenaiPackage.CONFIG_FILE, """{"model": {"decoder": {"filename": "../outside.onnx"}}}""")
    assertTrue(refusal(config).contains("not a normalized relative path"))

    val graph = wholePackage("graph")
    File(graph, "model.onnx").writeBytes(OnnxGraphBytes.graphNaming("../outside.data"))
    assertTrue(refusal(graph).contains("not a normalized relative path"))
  }

  @Test
  fun `a graph that is not ONNX is refused`() {
    val dir = wholePackage()
    File(dir, "model.onnx").writeText("this is not a protobuf")

    assertTrue(refusal(dir).contains("does not parse as an ONNX graph"))
  }

  /** A tensor declared external that names no file cannot be checked, so the graph is refused rather than passed. */
  @Test
  fun `an external tensor with no location is refused`() {
    val dir = wholePackage()
    File(dir, "model.onnx").writeBytes(OnnxGraphBytes.graphWithUnnamedExternalTensor())

    assertTrue(refusal(dir).contains("does not parse as an ONNX graph"))
  }

  /** A tensor inside a node's attribute, and one inside a subgraph of it, are tensors too. */
  @Test
  fun `external data named inside a node attribute and a subgraph is found`() {
    val dir = wholePackage()
    File(dir, "model.onnx").writeBytes(OnnxGraphBytes.graphNamingInSubgraph("nested.data"))

    assertTrue(refusal(dir).contains("nested.data"))
    write(dir, "nested.data", "weights")
    OnnxGenaiPackage.validate(dir)
  }

  @Test
  fun `a package without its tokenizer is refused`() {
    val noConfig = wholePackage("no-config")
    File(noConfig, "tokenizer_config.json").delete()
    assertTrue(refusal(noConfig).contains("tokenizer_config.json"))

    val noVocabulary = wholePackage("no-vocabulary")
    File(noVocabulary, "tokenizer.json").delete()
    assertEquals("`${noVocabulary.path}` is missing tokenizer vocabulary `tokenizer.json`", refusal(noVocabulary))
  }

  /** `model.tokenizer_dir` moves both files, and `tiktoken_file` names the vocabulary. */
  @Test
  fun `a tokenizer directory and a tiktoken vocabulary are followed`() {
    val dir = wholePackage()
    write(dir, OnnxGenaiPackage.CONFIG_FILE, """{"model": {"decoder": {"filename": "model.onnx"}, "tokenizer_dir": "tok"}}""")
    File(dir, "tokenizer_config.json").delete()
    File(dir, "tokenizer.json").delete()
    write(dir, "tok/tokenizer_config.json", """{"tiktoken_file": "vocab.tiktoken"}""")

    assertTrue(refusal(dir).contains("vocab.tiktoken"))
    write(dir, "tok/vocab.tiktoken", "vocabulary")
    OnnxGenaiPackage.validate(dir)
  }
}

/** Hand-written ONNX protobuf, field numbers as in onnx.proto. */
internal object OnnxGraphBytes {
  private const val VARINT = 0
  private const val BYTES = 2

  private fun varint(value: Long): ByteArray {
    val out = ByteArrayOutputStream()
    var rest = value
    while (rest >= 0x80) {
      out.write(((rest and 0x7F) or 0x80).toInt())
      rest = rest ushr 7
    }
    out.write(rest.toInt())
    return out.toByteArray()
  }

  private fun field(number: Int, payload: ByteArray): ByteArray = varint(((number shl 3) or BYTES).toLong()) + varint(payload.size.toLong()) + payload

  private fun flag(number: Int, value: Long): ByteArray = varint(((number shl 3) or VARINT).toLong()) + varint(value)

  private fun text(number: Int, value: String): ByteArray = field(number, value.toByteArray())

  /** `TensorProto` stored externally at [location]: `external_data` (13) with a `location` entry, and `data_location` (14) = EXTERNAL. */
  private fun externalTensor(location: String?): ByteArray =
    (location?.let { field(13, text(1, "location") + text(2, it)) } ?: ByteArray(0)) + flag(14, 1)

  /** `ModelProto.graph` (7) wrapping [graph]. */
  private fun model(graph: ByteArray): ByteArray = flag(1, 8) + field(7, graph)

  /** A graph with one initializer (5) stored at [location]. */
  fun graphNaming(location: String): ByteArray = model(field(5, externalTensor(location)))

  fun graphWithUnnamedExternalTensor(): ByteArray = model(field(5, externalTensor(null)))

  /** A node (1) whose attribute (5) holds a subgraph (6), whose own node's attribute holds a tensor (5) stored at [location]. */
  fun graphNamingInSubgraph(location: String): ByteArray {
    val inner = field(1, field(5, field(5, externalTensor(location))))
    return model(field(1, field(5, field(6, inner))))
  }
}
