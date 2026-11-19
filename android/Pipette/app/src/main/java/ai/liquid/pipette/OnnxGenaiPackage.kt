package ai.liquid.pipette

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import org.json.JSONObject

/**
 * An ONNX Runtime GenAI package on disk: a directory whose `genai_config.json` names a graph that is there, external data included.
 *
 * The Kotlin counterpart of iOS's `OnnxGenaiModels.validatePackage` and the host crate's `pipette-onnx-genai/src/models.rs`. Checked here rather than
 * left to the engine for the reasons iOS gives: a graph missing its external-data sidecar loads and then fails at the first inference as if the
 * engine were at fault, the sidecar is the multi-gigabyte half a truncated transfer loses, and most of LiquidAI's `-ONNX` repos are transformers.js
 * exports with real `.onnx` graphs and no `genai_config.json`, which named here says which kind of ONNX the directory holds.
 *
 * No provider check: this client's runtime has the CPU provider alone.
 */
// One function per message of the protobuf walk and per part of a package, each refusing by name: the refusals are what the checks are for.
@Suppress("TooManyFunctions", "ThrowsCount")
object OnnxGenaiPackage {
  /** The file every ONNX Runtime GenAI package has, and a plain ONNX export has not. Everything else a package needs is named inside it. */
  const val CONFIG_FILE = "genai_config.json"

  /** Why a directory is not a package this engine can load. */
  class Invalid(message: String, cause: Throwable? = null) : IllegalStateException(message, cause)

  /** Whether [path] is a package's directory rather than a GGUF file: the directory that holds a [CONFIG_FILE]. Says nothing of the rest. */
  fun isPackage(path: String): Boolean = File(path, CONFIG_FILE).isFile

  /** Whether [dir] passes [validate]. */
  fun isValid(dir: File): Boolean = runCatching { validate(dir) }.isSuccess

  /** Check [dir] as iOS checks it: the config, the decoder graph it names, the graph's external data and the tokenizer's assets. */
  fun validate(dir: File) {
    val config = File(dir, CONFIG_FILE)
    if (!config.isFile) {
      throw Invalid(
        "`${dir.path}` has no $CONFIG_FILE, so it is not an ONNX Runtime GenAI package; a transformers.js export has `.onnx` graphs and no such file"
      )
    }
    val root =
      runCatching { JSONObject(config.readText()) }.getOrNull() ?: throw Invalid("`${dir.path}` has a $CONFIG_FILE that does not parse as JSON")
    val model = root.optJSONObject("model")
    val filename = model?.optJSONObject("decoder")?.optStringOrNull("filename")
    if (model == null || filename == null) {
      throw Invalid("`${dir.path}`'s $CONFIG_FILE names no `model.decoder.filename`, so there is no graph to load")
    }
    val graph = packageFile(filename, dir, dir, "decoder graph")
    if (!graph.isFile) throw Invalid("`${dir.path}` is missing the decoder graph `$filename` its $CONFIG_FILE names")
    validateExternalData(graph, dir)
    validateTokenizer(model, dir)
  }

  private fun JSONObject.optStringOrNull(key: String): String? = (opt(key) as? String)?.takeIf { it.isNotEmpty() }

  /**
   * Resolves an untrusted metadata path without letting it leave the package. A normalized spelling is required, so `graphs/../model.onnx` is refused
   * rather than reinterpreted. The second check follows symlinks, since a package-supplied symlink can escape without any `..`.
   */
  private fun packageFile(path: String, base: File, root: File, kind: String): File {
    val normalized =
      path.isNotEmpty() && !path.startsWith("/") && !path.contains("\\") && path.split("/").none { it.isEmpty() || it == "." || it == ".." }
    if (!normalized) throw Invalid("`${root.path}`'s $kind path `$path` is not a normalized relative path inside the package")
    val candidate = File(base, path)
    if (!contains(candidate.absoluteFile.normalize(), root.absoluteFile.normalize())) {
      throw Invalid("`${root.path}`'s $kind path `$path` escapes the package")
    }
    if (!contains(candidate.canonicalFile, root.canonicalFile)) {
      throw Invalid("`${root.path}`'s $kind path `$path` escapes the package through a symbolic link")
    }
    return candidate
  }

  private fun contains(candidate: File, root: File): Boolean = candidate.path.startsWith(root.path.trimEnd('/') + "/")

  /**
   * The assets ORT Extensions requires in `model.tokenizer_dir`, checked so a completed download cannot become an installed model that fails when its
   * tokenizer is built.
   */
  private fun validateTokenizer(model: JSONObject, root: File) {
    val tokenizerDir =
      if (model.has("tokenizer_dir")) {
        val raw = model.opt("tokenizer_dir") as? String ?: throw Invalid("`${root.path}`'s $CONFIG_FILE has a non-string `model.tokenizer_dir`")
        if (raw.isEmpty()) root else packageFile(raw, root, root, "tokenizer directory")
      } else {
        root
      }
    val config = packageFile("tokenizer_config.json", tokenizerDir, root, "tokenizer config")
    val parsed =
      runCatching { JSONObject(config.readText()) }.getOrNull()
        ?: throw Invalid("`${root.path}` is missing a valid tokenizer_config.json in its tokenizer directory")
    val vocabulary =
      if (parsed.has("tiktoken_file")) {
        val raw = parsed.optStringOrNull("tiktoken_file") ?: throw Invalid("`${root.path}`'s tokenizer_config.json has an invalid `tiktoken_file`")
        packageFile(raw, tokenizerDir, root, "tokenizer vocabulary")
      } else {
        packageFile("tokenizer.json", tokenizerDir, root, "tokenizer vocabulary")
      }
    if (!vocabulary.isFile) throw Invalid("`${root.path}` is missing tokenizer vocabulary `${vocabulary.name}`")
  }

  /** Every external-data file the graph references is there. The graph is mapped, so its tensor bytes are skipped by range and never copied. */
  private fun validateExternalData(graph: File, root: File) {
    val names =
      try {
        RandomAccessFile(graph, "r").use { file -> externalDataLocations(file.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())) }
      } catch (error: InvalidProtobuf) {
        throw Invalid("`${graph.path}` does not parse as an ONNX graph", error)
      } catch (error: java.io.IOException) {
        throw Invalid("`${graph.path}` could not be read: ${error.message}", error)
      } catch (error: IllegalArgumentException) {
        // A graph over 2 GiB cannot be mapped, and cannot be a protobuf either: its weights belong in external data.
        throw Invalid("`${graph.path}` is too large to be an ONNX graph with external data", error)
      }
    val graphDir = graph.parentFile ?: root
    for (name in names) {
      val sidecar = packageFile(name, graphDir, root, "external-data")
      if (!sidecar.isFile) {
        throw Invalid("`${root.path}` is missing `$name`, the external-data file its decoder graph references; the download is incomplete")
      }
    }
  }

  // ===========================================================================
  // The ONNX graph's external-data locations
  // ===========================================================================

  private class InvalidProtobuf : Exception()

  private const val WIRE_VARINT = 0
  private const val WIRE_FIXED64 = 1
  private const val WIRE_BYTES = 2
  private const val WIRE_FIXED32 = 5
  private const val MAX_GRAPH_DEPTH = 64
  private const val VARINT_SHIFT_LIMIT = 63
  private const val VARINT_PAYLOAD_BITS = 7
  private const val BYTE_MASK = 0xFF
  private const val VARINT_PAYLOAD_MASK = 0x7F
  private const val VARINT_CONTINUATION = 0x80
  private const val WIRE_TYPE_BITS = 3
  private const val WIRE_TYPE_MASK = 0x07L
  private const val FIXED64_BYTES = 8
  private const val FIXED32_BYTES = 4

  // ONNX field numbers, as in onnx.proto.
  private const val MODEL_GRAPH = 7
  private const val GRAPH_NODE = 1
  private const val GRAPH_INITIALIZER = 5
  private const val GRAPH_SPARSE_INITIALIZER = 15
  private const val NODE_ATTRIBUTE = 5
  private const val ATTRIBUTE_TENSOR = 5
  private const val ATTRIBUTE_TENSORS = 10
  private const val ATTRIBUTE_GRAPH = 6
  private const val ATTRIBUTE_GRAPHS = 11
  private const val ATTRIBUTE_SPARSE_TENSOR = 22
  private const val ATTRIBUTE_SPARSE_TENSORS = 23
  private const val SPARSE_VALUES = 1
  private const val SPARSE_INDICES = 2
  private const val TENSOR_EXTERNAL_DATA = 13
  private const val TENSOR_DATA_LOCATION = 14
  private const val DATA_LOCATION_EXTERNAL = 1L
  private const val ENTRY_KEY = 1
  private const val ENTRY_VALUE = 2

  /** A field's value: a varint, or the range of a length-delimited one. Fixed-width fields are skipped and never reach a visitor. */
  private class Field(val number: Int, val varint: Long, val start: Int, val end: Int) {
    val isBytes: Boolean
      get() = start >= 0
  }

  /**
   * ONNX field path: ModelProto.graph(7) → GraphProto.initializer(5) → TensorProto.external_data(13) → StringStringEntryProto key/value(1/2). Node
   * attributes and sparse initializers can contain tensors too, so those paths are traversed as well. Unknown fields are skipped according to their
   * wire type.
   */
  private fun externalDataLocations(bytes: ByteBuffer): List<String> {
    val found = sortedSetOf<String>()
    var sawGraph = false
    fields(bytes, 0, bytes.limit()) { field ->
      if (field.number == MODEL_GRAPH && field.isBytes) {
        sawGraph = true
        scanGraph(bytes, field, 0, found)
      }
    }
    if (!sawGraph) throw InvalidProtobuf()
    return found.toList()
  }

  private fun scanGraph(bytes: ByteBuffer, graph: Field, depth: Int, found: MutableSet<String>) {
    if (depth >= MAX_GRAPH_DEPTH) throw InvalidProtobuf()
    fields(bytes, graph.start, graph.end) { field ->
      if (!field.isBytes) return@fields
      when (field.number) {
        GRAPH_NODE -> scanNode(bytes, field, depth, found)
        GRAPH_INITIALIZER -> scanTensor(bytes, field, found)
        GRAPH_SPARSE_INITIALIZER -> scanSparseTensor(bytes, field, found)
      }
    }
  }

  private fun scanNode(bytes: ByteBuffer, node: Field, depth: Int, found: MutableSet<String>) {
    fields(bytes, node.start, node.end) { field -> if (field.number == NODE_ATTRIBUTE && field.isBytes) scanAttribute(bytes, field, depth, found) }
  }

  private fun scanAttribute(bytes: ByteBuffer, attribute: Field, depth: Int, found: MutableSet<String>) {
    fields(bytes, attribute.start, attribute.end) { field ->
      if (!field.isBytes) return@fields
      when (field.number) {
        ATTRIBUTE_TENSOR,
        ATTRIBUTE_TENSORS -> scanTensor(bytes, field, found)
        ATTRIBUTE_GRAPH,
        ATTRIBUTE_GRAPHS -> scanGraph(bytes, field, depth + 1, found)
        ATTRIBUTE_SPARSE_TENSOR,
        ATTRIBUTE_SPARSE_TENSORS -> scanSparseTensor(bytes, field, found)
      }
    }
  }

  private fun scanSparseTensor(bytes: ByteBuffer, sparse: Field, found: MutableSet<String>) {
    fields(bytes, sparse.start, sparse.end) { field ->
      if ((field.number == SPARSE_VALUES || field.number == SPARSE_INDICES) && field.isBytes) scanTensor(bytes, field, found)
    }
  }

  private fun scanTensor(bytes: ByteBuffer, tensor: Field, found: MutableSet<String>) {
    var external = false
    var location: String? = null
    fields(bytes, tensor.start, tensor.end) { field ->
      if (field.number == TENSOR_DATA_LOCATION && !field.isBytes) {
        external = field.varint == DATA_LOCATION_EXTERNAL
      } else if (field.number == TENSOR_EXTERNAL_DATA && field.isBytes) {
        val (key, value) = stringPair(bytes, field)
        if (key == "location") location = value
      }
    }
    val name = location
    if (name != null) {
      found.add(name)
    } else if (external) {
      // Declared external but naming no file: it cannot be checked, so the graph is refused rather than passed.
      throw InvalidProtobuf()
    }
  }

  private fun stringPair(bytes: ByteBuffer, entry: Field): Pair<String?, String?> {
    var key: String? = null
    var value: String? = null
    fields(bytes, entry.start, entry.end) { field ->
      if (!field.isBytes) return@fields
      val raw = ByteArray(field.end - field.start)
      bytes.duplicate().apply { position(field.start) }.get(raw)
      val string = String(raw, StandardCharsets.UTF_8)
      if (field.number == ENTRY_KEY) key = string
      if (field.number == ENTRY_VALUE) value = string
    }
    return key to value
  }

  private fun fields(bytes: ByteBuffer, start: Int, end: Int, visit: (Field) -> Unit) {
    val cursor = intArrayOf(start)
    while (cursor[0] < end) {
      val key = varint(bytes, cursor, end)
      val rawNumber = key ushr WIRE_TYPE_BITS
      if (rawNumber <= 0 || rawNumber > Int.MAX_VALUE) throw InvalidProtobuf()
      val number = rawNumber.toInt()
      when ((key and WIRE_TYPE_MASK).toInt()) {
        WIRE_VARINT -> visit(Field(number, varint(bytes, cursor, end), -1, -1))
        WIRE_FIXED64 -> skip(cursor, end, FIXED64_BYTES)
        WIRE_BYTES -> {
          val length = varint(bytes, cursor, end)
          if (length < 0 || length > end - cursor[0]) throw InvalidProtobuf()
          val fieldEnd = cursor[0] + length.toInt()
          visit(Field(number, 0, cursor[0], fieldEnd))
          cursor[0] = fieldEnd
        }
        WIRE_FIXED32 -> skip(cursor, end, FIXED32_BYTES)
        else -> throw InvalidProtobuf()
      }
    }
  }

  private fun skip(cursor: IntArray, end: Int, width: Int) {
    if (cursor[0] > end - width) throw InvalidProtobuf()
    cursor[0] += width
  }

  private fun varint(bytes: ByteBuffer, cursor: IntArray, end: Int): Long {
    var value = 0L
    var shift = 0
    while (shift <= VARINT_SHIFT_LIMIT) {
      if (cursor[0] >= end) throw InvalidProtobuf()
      val byte = bytes.get(cursor[0]).toInt() and BYTE_MASK
      cursor[0] += 1
      if (shift == VARINT_SHIFT_LIMIT && byte > 1) throw InvalidProtobuf()
      value = value or ((byte and VARINT_PAYLOAD_MASK).toLong() shl shift)
      if (byte and VARINT_CONTINUATION == 0) return value
      shift += VARINT_PAYLOAD_BITS
    }
    throw InvalidProtobuf()
  }
}
