package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The ONNX Runtime GenAI package as a [Model]: what it is on the wire, in a stored manifest and to a run, and that its result stays on the device.
 */
class OnnxGenaiModelTest {
  private val phi3 =
    Model.OnnxGenai(HfOnnxGenai(HfRepo.parseSlug("microsoft/Phi-3-mini-4k-instruct-onnx"), RepoSubpath.parse("cpu_and_mobile/cpu-int4-rtn-block-32")))
  private val atRoot = Model.OnnxGenai(HfOnnxGenai(HfRepo.parseSlug("org/some-genai-package")))

  /** plan-types' own example for this kind (`model/mod.rs`), in the canonical sorted-key form the other descriptors are pinned in. */
  @Test
  fun `the model descriptor is plan-types' onnx_genai shape`() {
    assertEquals(
      """{"org":"microsoft","prefix":"cpu_and_mobile/cpu-int4-rtn-block-32",""" +
        """"repo_name":"Phi-3-mini-4k-instruct-onnx","source":"huggingface","type":"onnx_genai"}""",
      SubmissionRef.model(phi3),
    )
    // `prefix` is skipped when absent, as `skip_serializing_if` skips it there.
    assertEquals("""{"org":"org","repo_name":"some-genai-package","source":"huggingface","type":"onnx_genai"}""", SubmissionRef.model(atRoot))
  }

  @Test
  fun `a stored manifest reads back the coordinate it wrote`() {
    assertEquals(phi3, Model.fromJsonOrNull(phi3.toJson()))
    assertEquals(atRoot, Model.fromJsonOrNull(atRoot.toJson()))

    val gated = Model.OnnxGenai(phi3.model.copy(repo = phi3.model.repo.copy(requiresAuth = true)))
    assertEquals(gated, Model.fromJsonOrNull(gated.toJson()))
  }

  /** A pin rides in the descriptor as plan-types' `HfRepo.revision`, and survives a stored manifest: lost there, a rerun would fetch `main`. */
  @Test
  fun `a pinned coordinate keeps its revision on the wire and in a manifest`() {
    val pinned = Model.OnnxGenai(phi3.model.copy(revision = HfRevision.parse("5f5f794c1c23c9d5ee142af85df02a6cc52d6945")))

    assertEquals(
      """{"org":"microsoft","prefix":"cpu_and_mobile/cpu-int4-rtn-block-32","repo_name":"Phi-3-mini-4k-instruct-onnx",""" +
        """"revision":"5f5f794c1c23c9d5ee142af85df02a6cc52d6945","source":"huggingface","type":"onnx_genai"}""",
      SubmissionRef.model(pinned),
    )
    assertEquals(pinned, Model.fromJsonOrNull(pinned.toJson()))
    assertNull(Model.fromJsonOrNull(pinned.toJson().put("revision", "-not-a-revision")))
  }

  /**
   * A pinned coordinate takes the package on disk only when its record is that commit. For another commit, or a package with no record, it keeps the
   * offer form, so the runner asks the fetcher, which refuses a mismatch.
   */
  @Test
  fun `a pinned coordinate takes the package on disk only at its own commit`() {
    val commit = HfRevision.parse("5f5f794c1c23c9d5ee142af85df02a6cc52d6945")
    val pinned = Model.OnnxGenai(phi3.model.copy(revision = commit))
    val path = "/models/${phi3.model.relativeDir}"
    val recorded = ModelCatalog.packageFile(pinned, path = path, sizeBytes = 9L, onDisk = true)
    val unrecorded = ModelCatalog.packageFile(phi3, path = path, sizeBytes = 9L, onDisk = true)
    val other =
      ModelCatalog.packageFile(Model.OnnxGenai(phi3.model.copy(revision = HfRevision.parse("1111111"))), path = path, sizeBytes = 9L, onDisk = true)

    val taken = ModelCatalog.resolveCoordinate(pinned, downloaded = listOf(recorded)).weights
    assertEquals(path, taken.path)
    assertTrue(taken.onDisk)
    assertEquals(pinned, taken.modelRef)

    for (mismatch in listOf(unrecorded, other)) {
      val offered = ModelCatalog.resolveCoordinate(pinned, downloaded = listOf(mismatch)).weights
      assertEquals(pinned.model.relativeDir, offered.path)
      assertFalse(offered.onDisk)
      assertEquals(pinned, offered.modelRef)
    }

    // An unpinned coordinate takes what is there, and stays unpinned.
    val unpinned = ModelCatalog.resolveCoordinate(phi3, downloaded = listOf(recorded)).weights
    assertTrue(unpinned.onDisk)
    assertEquals(phi3, unpinned.modelRef)
  }

  /** A prefix that no longer parses is a malformed entry, not a package at the repo's root: read as one, a run would fetch the whole repo. */
  @Test
  fun `a manifest with a malformed prefix is not a coordinate`() {
    val json = phi3.toJson().put("prefix", "../escape")

    assertNull(Model.fromJsonOrNull(json))
  }

  @Test
  fun `a repo subpath is the plan-types rule`() {
    for (good in listOf("a", "a/b", "cpu_and_mobile/cpu-int4-rtn-block-32", "v1.0/x_y-z")) {
      assertEquals(good, RepoSubpath.parse(good).toString())
    }
    for (bad in listOf("", ".", "..", "a/..", "../a", "/a", "a/", "a//b", "a b", "a\\b")) {
      assertNull(bad, RepoSubpath.parseOrNull(bad))
    }
    assertEquals("cpu-int4-rtn-block-32", RepoSubpath.parse("cpu_and_mobile/cpu-int4-rtn-block-32").leaf)
  }

  /** What a headless `model=onnx-genai://…` plans against before anything is on disk: the directory the package will have. */
  @Test
  fun `a coordinate with nothing on disk resolves to the offer form`() {
    val resolved = ModelCatalog.resolveCoordinate(phi3, downloaded = emptyList())

    assertNull(resolved.projector)
    assertEquals("cpu-int4-rtn-block-32", resolved.weights.name)
    assertEquals("microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32", resolved.weights.path)
    assertEquals(phi3, resolved.weights.modelRef)
    assertFalse(resolved.weights.onDisk)
  }

  /** The downloaded row wins, as it does for a GGUF file: only it knows the real path and size. Matched on the coordinate, not the name. */
  @Test
  fun `a coordinate prefers the package on disk`() {
    val onDisk = ModelCatalog.packageFile(phi3, path = "/models/microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32", 9L, true)
    val sameNameOtherRepo =
      ModelCatalog.packageFile(
        Model.OnnxGenai(HfOnnxGenai(HfRepo.parseSlug("other/repo"), RepoSubpath.parse("cpu-int4-rtn-block-32"))),
        path = "/models/other/repo/cpu-int4-rtn-block-32",
        sizeBytes = 1L,
        onDisk = true,
      )

    val resolved = ModelCatalog.resolveCoordinate(phi3, downloaded = listOf(sameNameOtherRepo, onDisk))

    assertEquals(onDisk, resolved.weights)
  }

  /** A package has no projector half, and is fetched by its own path rather than through the GGUF download layer. */
  @Test
  fun `provisioning a package goes to the package fetcher`() {
    val asked = mutableListOf<HfOnnxGenai>()
    val provisioner =
      ModelProvisioner(
        resolvePath = { null },
        downloads =
          object : ModelProvisioner.Downloads {
            override fun resume(key: String) = error("a package is not a GGUF transfer")

            override fun enqueue(url: String, repo: String?) = error("a package is not a GGUF transfer")
          },
        packages = { ref, onStatus ->
          asked += ref
          onStatus("Resolving ${ref.name}")
          Result.success("/models/${ref.relativeDir}")
        },
        records = { emptyList() },
        isOnline = { true },
      )
    val statuses = mutableListOf<String>()

    val path = provisioner.ensureWeights(phi3) { statuses += it }.getOrThrow()

    assertEquals("/models/microsoft/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32", path)
    assertEquals(listOf(phi3.model), asked)
    assertEquals(listOf("Resolving cpu-int4-rtn-block-32"), statuses)
    assertNull(provisioner.ensureMmproj(phi3))
  }

  /** Where no fetcher is wired the package is refused by name, not left to fail as a missing file. */
  @Test
  fun `a provisioner with no package fetcher refuses a package`() {
    val provisioner =
      ModelProvisioner(
        resolvePath = { null },
        downloads =
          object : ModelProvisioner.Downloads {
            override fun resume(key: String) = Unit

            override fun enqueue(url: String, repo: String?) = Unit
          },
        records = { emptyList() },
        isOnline = { true },
      )

    val reason = provisioner.ensureWeights(phi3).exceptionOrNull()?.message.orEmpty()

    assertTrue(reason, reason.contains("cpu-int4-rtn-block-32"))
  }

  /**
   * A result from the ONNX Runtime GenAI engine is submittable only where the collector knows the runtime, which the build says. Elsewhere it is
   * counted neither as sent nor as waiting to be sent, or a run would report results it can never upload.
   */
  @Test
  fun `a result from the ONNX engine is not submittable`() {
    fun cell(ref: Model?) =
      JobCell(
        benchmarkId = "prefill_throughput_512",
        benchmarkType = "prefill_throughput",
        modelPath = "/models/x",
        modelName = "x",
        runStatus = CellRunStatus.COMPLETED,
        modelRef = ref,
      )
    val gguf = Model.GgufText(HfGgufText(HfRepo.parseSlug("LiquidAI/LFM2.5-350M-GGUF"), GgufFilename.parse("LFM2.5-350M-Q4_0.gguf")))

    assertFalse(cell(phi3).isSubmittable(onnxGenaiSubmit = false))
    assertTrue("where the collector knows the runtime, the result goes up like any other", cell(phi3).isSubmittable(onnxGenaiSubmit = true))
    assertTrue(cell(gguf).isSubmittable(onnxGenaiSubmit = false))
    assertTrue("a cell with no coordinate is an imported GGUF file", cell(null).isSubmittable(onnxGenaiSubmit = false))
  }
}
