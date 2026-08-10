package ai.liquid.pipette

import ai.liquid.pipette.fakes.BenchmarkCatalogFixture
import android.os.PowerManager
import java.io.File
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class ExampleUnitTest {
  // Several tests resolve benchmarks by id (including VL, which isn't structurally parseable), so install the post-sync catalog around each.
  @Before
  fun installCatalog() {
    BenchmarkCatalogFixture.install()
  }

  @After
  fun resetCatalog() {
    BenchmarkCatalogFixture.reset()
  }

  @Test
  fun parseQuantReadsTrailingQuantToken() {
    assertEquals("Q4_K_M", LocalStorage.parseQuant("LFM2.5-350M-Q4_K_M.gguf"))
    assertEquals("BF16", LocalStorage.parseQuant("model-BF16.gguf"))
    assertNull(LocalStorage.parseQuant("model.gguf"))
  }

  @Test
  fun completedFailedAndCancelledCellsCanBeRerun() {
    assertFalse(CellRunStatus.PENDING.isRerunnable)
    assertFalse(CellRunStatus.RUNNING.isRerunnable)
    assertTrue(CellRunStatus.COMPLETED.isRerunnable)
    assertTrue(CellRunStatus.FAILED.isRerunnable)
    assertTrue(CellRunStatus.CANCELLED.isRerunnable)
  }

  @Test
  fun registrationDataPreservesOptionalClerkLinkFields() {
    val registration =
      RegistrationData(
          clientId = "client-1",
          status = "active",
          serverUrl = "https://example.test",
          organization = "Liquid",
          contactEmail = "user@example.test",
          registeredAt = "2026-06-08T12:00:00Z",
        )
        .withClerkLink(userId = "user_123", sessionId = "sess_123", primaryEmail = "user@example.test")

    val roundTripped = RegistrationData.fromJson(registration.toJson())

    assertEquals("user_123", roundTripped.clerkUserId)
    assertEquals("sess_123", roundTripped.clerkSessionId)
    assertEquals("user@example.test", roundTripped.clerkPrimaryEmail)
    assertFalse(roundTripped.clerkLinkedAt.isNullOrBlank())
  }

  @Test
  fun jobRunnerOnlyExecutesCellsResetToPending() {
    val benchmark = BenchmarkCatalog.byId("decode_throughput_512_100")!!
    val manifest =
      JobManifest(
        nGpuLayers = 99,
        contextSize = 4096,
        status = JobStatus.RUNNING,
        cells =
          mutableListOf(
            testCell("completed", benchmark, CellRunStatus.COMPLETED, "/tmp/a.gguf"),
            testCell("cancelled", benchmark, CellRunStatus.CANCELLED, "/tmp/b.gguf"),
            testCell("failed", benchmark, CellRunStatus.FAILED, "/tmp/c.gguf"),
            testCell("pending", benchmark, CellRunStatus.PENDING, "/tmp/d.gguf"),
          ),
      )

    val order = JobRunner.pendingExecutionOrder(manifest, mapOf(benchmark.benchmarkId.value to benchmark))

    assertEquals(listOf(3), order)
  }

  @Test
  fun jobQuantFiltersMatchIosQuantPills() {
    assertTrue(JobQuantFilter.ALL.matches("Q4_K_M"))
    assertFalse(JobQuantFilter.ALL.matches(null))
    assertTrue(JobQuantFilter.Q4_0.matches("q4_0"))
    assertTrue(JobQuantFilter.Q4_K_M.matches("Q4_K_M"))
    assertTrue(JobQuantFilter.Q5_K_M.matches("Q5_K_M"))
    assertFalse(JobQuantFilter.Q4_0.matches("Q4_K_M"))
    assertFalse(JobQuantFilter.Q5_K_M.matches("BF16"))
    assertFalse(JobQuantFilter.Q4_K_M.matches(null))
  }

  @Test
  fun modelCatalogGroupsQuantVariantsByFamilyId() {
    val q4 = testModel(name = "shared-Q4_0.gguf", hfRepo = "org/shared-q4-GGUF", displayName = "Shared", familyId = "shared-family")
    val q5 = testModel(name = "shared-Q5_K_M.gguf", hfRepo = "org/shared-q5-GGUF", displayName = "Shared", familyId = "shared-family")

    val groups = ModelCatalog.groups(listOf(q4, q5))

    assertEquals(1, groups.size)
    assertEquals("shared-family", groups.single().key)
    assertEquals("Shared", groups.single().name)
    assertEquals(listOf(q4, q5), groups.single().files)
    assertEquals("Q4_0, Q5_K_M", groups.single().quantSummary)
  }

  @Test
  fun modelCatalogFallsBackToNormalizedStemForSideloadGrouping() {
    val q4 = testModel("local-model-Q4_0.gguf")
    val q5 = testModel("local-model-Q5_K_M.gguf")

    val group = ModelCatalog.groups(listOf(q4, q5)).single()

    assertEquals("local-model", group.key)
    assertEquals(listOf(q4, q5), group.files)
  }

  @Test
  fun modelCatalogResolvesSelectedGroupsThroughQuantFilter() {
    val q4 = testModel(name = "shared-Q4_0.gguf", familyId = "shared-family")
    val q5 = testModel(name = "shared-Q5_K_M.gguf", familyId = "shared-family")
    val other = testModel(name = "other-Q4_0.gguf", familyId = "other-family")
    val groups = ModelCatalog.groups(listOf(q4, q5, other))

    val resolved =
      ModelCatalog.resolveSelectedFiles(groups = groups, selectedKeys = setOf("shared-family"), quantMatches = JobQuantFilter.Q5_K_M::matches)

    assertEquals(listOf(q5), resolved)
  }

  @Test
  fun modelCatalogFindsSelectedGroupsMissingActiveQuant() {
    val shared = testModel(name = "shared-Q4_0.gguf", displayName = "Shared", familyId = "shared-family")
    val other = testModel(name = "other-Q5_K_M.gguf", displayName = "Other", familyId = "other-family")
    val groups = ModelCatalog.groups(listOf(shared, other))

    val missing =
      ModelCatalog.selectedGroupsMissingQuant(
        groups = groups,
        selectedKeys = setOf("shared-family", "other-family"),
        quantMatches = JobQuantFilter.Q5_K_M::matches,
      )

    assertEquals(listOf("Shared"), missing.map { it.name })
  }

  @Test
  fun thermalStatusProviderLabelsAndroidThermalStatuses() {
    // ThermalCooldown was removed in favor of the native readiness gate
    // (see Readiness.kt); the status-label helpers it owned now live on
    // AndroidThermalStatusProvider, which the UI uses for display only.
    assertEquals("nominal", AndroidThermalStatusProvider.statusLabel(PowerManager.THERMAL_STATUS_NONE))
    assertEquals("moderate", AndroidThermalStatusProvider.statusLabel(PowerManager.THERMAL_STATUS_MODERATE))
    assertEquals("critical", AndroidThermalStatusProvider.statusLabel(PowerManager.THERMAL_STATUS_CRITICAL))
    assertEquals("unknown", AndroidThermalStatusProvider.statusLabel(-1))
    assertEquals("Normal", AndroidThermalStatusProvider.displayStatusLabel(PowerManager.THERMAL_STATUS_NONE))
    assertEquals("Severe", AndroidThermalStatusProvider.displayStatusLabel(PowerManager.THERMAL_STATUS_SEVERE))
  }

  @Test
  fun benchmarkSearchMatchesTypeDisplayNameAndParameters() {
    val decode = BenchmarkCatalog.byId("decode_throughput_512_100")!!

    assertTrue(BenchmarkCatalog.matchesSearch(decode, "decode"))
    assertTrue(BenchmarkCatalog.matchesSearch(decode, "Decode Throughput"))
    assertTrue(BenchmarkCatalog.matchesSearch(decode, "512tok"))
    assertTrue(BenchmarkCatalog.matchesSearch(decode, "throughput_512"))
    assertFalse(BenchmarkCatalog.matchesSearch(decode, "vision-language"))
  }

  @Test
  fun vlCompatibilityMatchesRepoOrNormalizedStem() {
    val base = ModelFile(name = "model-Q4_K_M.gguf", path = "/tmp/model-Q4_K_M.gguf", sizeBytes = 1, hfRepo = "org/model-GGUF")
    val sameRepoMmproj = ModelFile(name = "mmproj-other.gguf", path = "/tmp/mmproj-other.gguf", sizeBytes = 1, hfRepo = "org/model-GGUF")
    val sameStemMmproj = ModelFile(name = "mmproj-model-Q4_K_M.gguf", path = "/tmp/mmproj-model-Q4_K_M.gguf", sizeBytes = 1, hfRepo = null)
    val unrelatedMmproj = ModelFile(name = "mmproj-unrelated.gguf", path = "/tmp/mmproj-unrelated.gguf", sizeBytes = 1, hfRepo = "org/other-GGUF")

    assertTrue(JobRunner.isVlCompatible(base, listOf(sameRepoMmproj)))
    assertTrue(JobRunner.isVlCompatible(base.copy(hfRepo = null), listOf(sameStemMmproj)))
    assertFalse(JobRunner.isVlCompatible(base, listOf(unrelatedMmproj)))
  }

  @Test
  fun jobPlannerBuildsNonVlCartesianProduct() {
    val first = testModel("first-Q4_K_M.gguf")
    val second = testModel("second-Q4_K_M.gguf")
    val benchmarks = listOf(BenchmarkCatalog.byId("decode_throughput_512_100")!!, BenchmarkCatalog.byId("prefill_throughput_512")!!)

    val cells =
      JobRunner.planCells(models = listOf(first, second), mmprojFiles = emptyList(), benchmarks = benchmarks, selectedMmprojPaths = emptySet())

    assertEquals(4, cells.size)
    assertTrue(cells.all { it.mmprojPath == null })
    assertEquals(setOf(first.path, second.path), cells.map { it.modelPath }.toSet())
    assertEquals(benchmarks.map { it.benchmarkId.value }.toSet(), cells.map { it.benchmarkId }.toSet())
  }

  @Test
  fun jobPlannerExpandsVlCellsBySelectedMmprojsAndSkipsIncompatibleModels() {
    val compatible = testModel(name = "model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")
    val incompatible = testModel(name = "plain-Q4_K_M.gguf", hfRepo = "org/plain-GGUF")
    val firstMmproj = testModel(name = "mmproj-model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")
    val secondMmproj = testModel(name = "mmproj-model-Q5_K_M.gguf", hfRepo = "org/model-GGUF")

    val cells =
      JobRunner.planCells(
        models = listOf(compatible, incompatible),
        mmprojFiles = listOf(firstMmproj, secondMmproj),
        benchmarks = listOf(BenchmarkCatalog.byId("vl_throughput_256x256_32_128")!!),
        selectedMmprojPaths = setOf(secondMmproj.path),
      )

    assertEquals(1, cells.size)
    assertEquals(compatible.path, cells.single().modelPath)
    assertEquals(secondMmproj.path, cells.single().mmprojPath)
  }

  @Test
  fun jobPlannerCreatesNoVlCellsWhenNoMmprojIsSelected() {
    val compatible = testModel(name = "model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")
    val mmproj = testModel(name = "mmproj-model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")

    val cells =
      JobRunner.planCells(
        models = listOf(compatible),
        mmprojFiles = listOf(mmproj),
        benchmarks = listOf(BenchmarkCatalog.byId("vl_throughput_256x256_32_128")!!),
        selectedMmprojPaths = emptySet(),
      )

    assertTrue(cells.isEmpty())
  }

  @Test
  fun jobPlannerCountsMixedVlAndNonVlCells() {
    val compatible = testModel(name = "model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")
    val incompatible = testModel(name = "plain-Q4_K_M.gguf", hfRepo = "org/plain-GGUF")
    val mmproj = testModel(name = "mmproj-model-Q4_K_M.gguf", hfRepo = "org/model-GGUF")
    val benchmarks = listOf(BenchmarkCatalog.byId("decode_throughput_512_100")!!, BenchmarkCatalog.byId("vl_throughput_256x256_32_128")!!)

    val cells =
      JobRunner.planCells(
        models = listOf(compatible, incompatible),
        mmprojFiles = listOf(mmproj),
        benchmarks = benchmarks,
        selectedMmprojPaths = setOf(mmproj.path),
      )

    assertEquals(3, cells.size)
    assertEquals(2, cells.count { it.benchmarkType == "decode_throughput" })
    assertEquals(1, cells.count { it.benchmarkType == "vl_throughput" })
  }

  @Test
  fun downloadParserAcceptsTemplateIdentifier() {
    val parsed = DownloadCoordinator.parseDownloadInput("LiquidAI/LFM2.5-350M-GGUF:Q4_K_M")
    assertEquals("LiquidAI/LFM2.5-350M-GGUF", parsed.repo)
    assertEquals("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf", parsed.url)
  }

  @Test
  fun downloadParserNormalizesHuggingFaceBlobLinks() {
    val parsed =
      DownloadCoordinator.parseDownloadInput("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/blob/main/LFM2.5-350M-Q4_K_M.gguf?download=true")

    assertEquals("LiquidAI/LFM2.5-350M-GGUF", parsed.repo)
    assertEquals("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q4_K_M.gguf?download=true", parsed.url)
  }

  @Test
  fun downloadParserExtractsRepoFromResolveLinks() {
    val parsed = DownloadCoordinator.parseDownloadInput("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf")

    assertEquals("unsloth/Qwen3.5-2B-GGUF", parsed.repo)
    assertEquals("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf", parsed.url)
  }

  @Test
  fun modelTemplateCatalogDeclaresMinistralAcrossRepos() {
    val ministral = ModelTemplateCatalog.byFamilyId["ministral-3-3b-instruct-2512"]!!
    assertTrue(ministral.variants.map { it.repo }.toSet().size > 1)
    assertEquals("unsloth/Ministral-3-3B-Instruct-2512-GGUF", ministral.variants.first { it.quant == "Q4_0" }.repo)
    assertEquals(setOf("mistralai/Ministral-3-3B-Instruct-2512-GGUF"), ministral.variants.filter { it.quant != "Q4_0" }.map { it.repo }.toSet())
  }

  @Test
  fun everyTemplateFamilyIdMatchesSideloadGroupingKey() {
    ModelTemplateCatalog.defaults.forEach { preset -> assertEquals(LocalStorage.normalizedModelStem(preset.filename), preset.familyId) }
  }

  @Test
  fun presetEstimatedBytesParseMbAndGbLabels() {
    val small = PresetModel(id = "test-small", name = "Small", detail = "Q4_K_M · 267 MB", identifier = "LiquidAI/LFM2.5-350M-GGUF:Q4_K_M")
    val large = PresetModel(id = "test-large", name = "Large", detail = "Q4_K_M · 1.28 GB", identifier = "unsloth/Qwen3.5-2B-GGUF:Q4_K_M")

    assertEquals(267L * 1024L * 1024L, small.estimatedBytes)
    assertEquals((1.28 * 1024.0 * 1024.0 * 1024.0).toLong(), large.estimatedBytes)
  }

  @Test
  fun modelRelativePathBucketsRepoFiles() {
    assertEquals(
      "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf",
      LocalStorage.modelRelativePath("LiquidAI/LFM2.5-350M-GGUF", "LFM2.5-350M-Q4_K_M.gguf"),
    )
    assertEquals("local.gguf", LocalStorage.modelRelativePath(null, "local.gguf"))
  }

  @Test
  fun resolveModelPathRecoversNamespacedTailUnderCurrentModelsRoot() {
    val modelsDir = File("/tmp/pipette-model-resolve-test-${System.nanoTime()}/models")
    val file = File(modelsDir, "LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf")
    file.parentFile!!.mkdirs()
    file.writeText("test")

    val resolved = LocalStorage.resolveModelPath("/old/container/Pipette/models/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q4_K_M.gguf", modelsDir)

    assertEquals(file.absolutePath, resolved)
    modelsDir.parentFile!!.deleteRecursively()
  }

  @Test
  fun resolveModelPathFallsBackByFilenameButRefusesAmbiguousMatches() {
    val modelsDir = File("/tmp/pipette-model-ambiguous-test-${System.nanoTime()}/models")
    val first = File(modelsDir, "repo-a/shared.gguf")
    val second = File(modelsDir, "repo-b/shared.gguf")
    val unique = File(modelsDir, "repo-a/unique.gguf")
    first.parentFile!!.mkdirs()
    second.parentFile!!.mkdirs()
    first.writeText("a")
    second.writeText("b")
    unique.writeText("unique")

    assertNull(LocalStorage.resolveModelPath("/old/path/models/repo-c/shared.gguf", modelsDir))
    assertEquals(unique.absolutePath, LocalStorage.resolveModelPath("/old/path/models/repo-c/unique.gguf", modelsDir))
    modelsDir.parentFile!!.deleteRecursively()
  }

  @Test
  fun deletingModelPathCanPruneEmptyRepoBucketShape() {
    val modelsDir = File("/tmp/pipette-model-delete-test/models")
    val bucket = File(modelsDir, "LiquidAI/LFM2.5-350M-GGUF")
    val file = File(bucket, "LFM2.5-350M-Q4_K_M.gguf")
    file.parentFile!!.mkdirs()
    file.writeText("test")

    assertTrue(file.delete())
    var current: File? = file.parentFile
    while (current != null && current.exists() && current.canonicalPath != modelsDir.canonicalPath) {
      val children = current.listFiles()
      if (children == null || children.isNotEmpty()) break
      assertTrue(current.delete())
      current = current.parentFile
    }

    assertFalse(bucket.exists())
    assertTrue(modelsDir.exists())
    modelsDir.deleteRecursively()
  }

  @Test
  fun completedResultsCsvIncludesMetricsAndEscapesTitles() {
    val cell =
      JobCell(
        cellId = "cell-1",
        benchmarkId = "decode_throughput_512_100",
        benchmarkType = "decode_throughput",
        modelPath = "/tmp/LFM2.5-350M-Q4_K_M.gguf",
        modelName = "LiquidAI/LFM2.5-350M-GGUF",
        runStatus = CellRunStatus.COMPLETED,
        serverJobId = "server-1",
      )
    val manifest =
      JobManifest(
        jobId = "abcdef123456",
        createdAt = "2026-06-08T12:00:00Z",
        nGpuLayers = 99,
        contextSize = 4096,
        cells = mutableListOf(cell),
        status = JobStatus.COMPLETED,
        title = "Run, One",
      )
    val payload =
      JSONObject()
        .put("decode_time_ms", 250.0)
        .put("submitted_at", "2026-06-08T12:01:00Z")
        .put("runtime_name", "ggml-org/llama.cpp:android")
        .put("runtime_version", "test")
        .put("runtime_flags", "-ngl 99 -c 612")
        .put("device_name", "Pixel Test")
        .put("device_form_factor", "phone")
        .put("device_os_name", "Android")
        .put("device_os_version", "16")
        .put("device_chip_model", "test-chip")
        .put("device_ram_bytes", 8_000_000_000L)

    val csv = CompletedResultsCsvExporter.csv(manifest, mapOf(cell.cellId to payload))

    assertTrue(csv.startsWith("job_id,job_title,created_at,cell_id,model_name"))
    assertTrue(csv.contains("\"Run, One\""))
    assertTrue(csv.contains("LFM 2.5 350M"))
    assertTrue(csv.contains("Decode throughput"))
    assertTrue(csv.contains(",400,tok/s,400,"))
    assertTrue(csv.contains("server-1"))
  }

  @Test
  fun completedResultsCsvIncludesBatteryState() {
    val cell =
      JobCell(
        cellId = "cell-1",
        benchmarkId = "decode_throughput_512_100",
        benchmarkType = "decode_throughput",
        modelPath = "/tmp/LFM2.5-350M-Q4_K_M.gguf",
        modelName = "LiquidAI/LFM2.5-350M-GGUF",
        runStatus = CellRunStatus.COMPLETED,
      )
    val manifest =
      JobManifest(
        jobId = "abcdef123456",
        createdAt = "2026-06-08T12:00:00Z",
        nGpuLayers = 99,
        contextSize = 4096,
        cells = mutableListOf(cell),
        status = JobStatus.COMPLETED,
        title = "Battery run",
      )
    val payload =
      JSONObject()
        .put("decode_time_ms", 250.0)
        .put("device_battery_level", 42)
        .put("device_power_state", "not_charging")
        .put("device_power_save_mode", true)
        // Comma-free so the naive split below stays column-aligned; the
        // real comma-bearing descriptor is CSV-quoted by csvEscape (covered
        // by csvEscape's own handling), not asserted via this split.
        .put("runtime_cpu_variant", "neon")

    val csv = CompletedResultsCsvExporter.csv(manifest, mapOf(cell.cellId to payload))
    val header = csv.lineSequence().first().split(",")
    val row = csv.lineSequence().drop(1).first().split(",")

    assertEquals("42", row[header.indexOf("device_battery_level")])
    assertEquals("not_charging", row[header.indexOf("device_power_state")])
    assertEquals("true", row[header.indexOf("device_power_save_mode")])
    assertEquals("neon", row[header.indexOf("runtime_cpu_variant")])
  }

  @Test
  fun completedResultsCsvIncludesOsBuildAndSecurityPatch() {
    val cell =
      JobCell(
        cellId = "cell-1",
        benchmarkId = "decode_throughput_512_100",
        benchmarkType = "decode_throughput",
        modelPath = "/tmp/LFM2.5-350M-Q4_K_M.gguf",
        modelName = "LiquidAI/LFM2.5-350M-GGUF",
        runStatus = CellRunStatus.COMPLETED,
      )
    val manifest =
      JobManifest(
        jobId = "abcdef123456",
        createdAt = "2026-06-08T12:00:00Z",
        nGpuLayers = 99,
        contextSize = 4096,
        cells = mutableListOf(cell),
        status = JobStatus.COMPLETED,
        title = "OS build run",
      )
    val payload = JSONObject().put("decode_time_ms", 250.0).put("device_os_build", "12621605").put("device_os_security_patch", "2025-06-01")

    val csv = CompletedResultsCsvExporter.csv(manifest, mapOf(cell.cellId to payload))
    val header = csv.lineSequence().first().split(",")
    val row = csv.lineSequence().drop(1).first().split(",")

    assertEquals("12621605", row[header.indexOf("device_os_build")])
    assertEquals("2025-06-01", row[header.indexOf("device_os_security_patch")])
  }

  @Test
  fun metricDisplayDoesNotDuplicateEmbeddedUnits() {
    assertEquals("1708 ms", CompletedResultsCsvExporter.displayMetric(CompletedRunMetric("E2E latency", "ms", "1708 ms", 1708.0, false)))
    assertEquals("3.87 GB", CompletedResultsCsvExporter.displayMetric(CompletedRunMetric("Max memory", "bytes", "3.87 GB", 3.87, false)))
    assertEquals("400 tok/s", CompletedResultsCsvExporter.displayMetric(CompletedRunMetric("Decode throughput", "tok/s", "400", 400.0, true)))
  }

  @Test
  fun perCellContextMatchesBenchmarkShape() {
    val decode = BenchmarkCatalog.byId("decode_throughput_512_100")!!
    assertEquals(612, BenchmarkContextSize.perCell(decode.benchmarkType, decode.rawJson))

    val memory = BenchmarkCatalog.byId("max_memory_usage_512")!!
    assertEquals(513, BenchmarkContextSize.perCell(memory.benchmarkType, memory.rawJson))

    val vl = BenchmarkCatalog.byId("vl_throughput_256x256_32_128")!!
    assertEquals(8192, BenchmarkContextSize.perCell(vl.benchmarkType, vl.rawJson))
  }

  private fun testModel(name: String, hfRepo: String? = null, displayName: String? = null, familyId: String? = null): ModelFile =
    ModelFile(name = name, path = "/tmp/$name", sizeBytes = 1, hfRepo = hfRepo, displayName = displayName, familyId = familyId)

  private fun testCell(id: String, benchmark: BenchmarkDefinition, status: CellRunStatus, modelPath: String = "/tmp/model.gguf"): JobCell =
    JobCell(
      cellId = id,
      benchmarkId = benchmark.benchmarkId.value,
      benchmarkType = benchmark.benchmarkType,
      modelPath = modelPath,
      modelName = modelPath.substringAfterLast('/'),
      runStatus = status,
    )
}
