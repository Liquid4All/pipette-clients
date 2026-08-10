package ai.liquid.pipette

import android.app.AlertDialog
import android.graphics.Typeface
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.CheckBox
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONObject

/** Run setup (model/benchmark/MMProjector selection + planning), the job list, and per-job detail. */
class JobsScreen(ctx: ScreenContext) : Screen(ctx) {
  override fun renderBody(body: LinearLayout) {
    selectedJobId?.let { jobId ->
      val selected = storage.loadJobManifest(jobId)
      if (selected != null) {
        renderJobDetail(body, selected)
        return
      }
      selectedJobId = null
      expandedCellIds.clear()
      selectedRerunCellIds.clear()
    }

    if (vm.newJobStep != null) {
      renderWizard(body, vm.newJobStep!!.coerceIn(0, NewJobWizard.LAST_STEP))
      return
    }
    renderJobList(body)
  }

  // --- Job list / wizard entry -------------------------------------------------

  private fun renderJobList(body: LinearLayout) {
    val hasModels = storage.availableModels().any { !it.isMmproj }
    body.addView(displayTitle("Your jobs"))
    if (!vm.container.benchmarkEngine.isAvailable) {
      body.addView(
        card {
          addView(mutedLabel("Native benchmark engine missing: jobs can be planned, but cells will fail until libpipette_android.so is packaged."))
        }
      )
    }
    body.addView(
      card {
        addView(sectionTitle("Benchmark jobs"))
        if (!hasModels) {
          addView(mutedLabel("Download or add a model before creating a job."))
          addView(
            outlineButton("Go to Models") {
              selectedTab = Tab.MODELS
              render()
            }
          )
        } else {
          addView(mutedLabel("Pick models and benchmarks, then run them as a job."))
          addView(
            primaryButton("Create a benchmark job") {
              vm.newJobStep = 0
              statusText = ""
              render()
            }
          )
        }
      }
    )

    val manifests = storage.loadAllJobManifests()
    val filteredManifests = manifests.filter { jobMatchesSearch(it, jobSearchText) }
    body.addView(
      card {
        addView(sectionTitle("Jobs"))
        addView(searchBlock("Search jobs", jobSearchText) { jobSearchText = it })
        if (manifests.isEmpty()) {
          addView(mutedLabel("No jobs yet."))
        } else if (filteredManifests.isEmpty()) {
          addView(mutedLabel("No jobs match \"$jobSearchText\"."))
        } else {
          filteredManifests.forEach { manifest -> addView(jobCard(manifest)) }
        }
      }
    )
  }

  // --- New-job wizard (3 steps: models → benchmarks → review) ------------------

  private fun renderWizard(body: LinearLayout, step: Int) {
    val allModels = storage.availableModels()
    val baseModels = allModels.filterNot { it.isMmproj }
    val modelGroups = ModelCatalog.groups(baseModels)
    selectedModelKeys.retainAll(modelGroups.map { it.key }.toSet())
    val filteredModelGroups = modelGroups.filter { modelGroupMatchesSearch(it, jobModelSearchText) }
    val mmprojs = allModels.filter { it.isMmproj }
    val selectedRunnableBaseModels =
      ModelCatalog.resolveSelectedFiles(groups = modelGroups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
    val modelsMissingQuant =
      ModelCatalog.selectedGroupsMissingQuant(groups = modelGroups, selectedKeys = selectedModelKeys, quantMatches = ::jobQuantMatches)
    val selectedBaseModelCount = modelGroups.count { selectedModelKeys.contains(it.key) }
    val hasVlModelSelected = selectedRunnableBaseModels.any { JobRunner.isVlCompatible(it, mmprojs) }
    if (!hasVlModelSelected) pruneVlBenchmarks()
    ensureMmprojSelectionInitialized(mmprojs)
    val selectedBenchmarksForPlanning = BenchmarkCatalog.selectable.filter { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
    val hasVlBenchmarkSelected = selectedBenchmarksForPlanning.any { it.type == BenchmarkType.VL_THROUGHPUT }
    val plannedCells =
      JobRunner.planCells(
        models = selectedRunnableBaseModels,
        mmprojFiles = mmprojs,
        benchmarks = selectedBenchmarksForPlanning,
        selectedMmprojPaths = selectedMmprojPaths,
      )

    // Header: cancel + step progress.
    body.addView(
      textButton("✕ Cancel") {
        vm.newJobStep = null
        statusText = ""
        render()
      }
    )
    body.addView(linearProgress((step + 1) / NewJobWizard.STEP_TITLES.size.toDouble()))
    body.addView(sectionTitle("Step ${step + 1} of ${NewJobWizard.STEP_TITLES.size} · ${NewJobWizard.STEP_TITLES[step]}"))

    when (step) {
      0 -> {
        body.addView(
          card {
            addView(sectionTitle("Models"))
            addView(searchBlock("Search models for this job", jobModelSearchText) { jobModelSearchText = it })
            if (modelGroups.isEmpty()) {
              addView(mutedLabel("No base GGUF models available. Add one in Models."))
              addView(
                outlineButton("Go to Models") {
                  vm.newJobStep = null
                  selectedTab = Tab.MODELS
                  render()
                }
              )
            } else if (filteredModelGroups.isEmpty()) {
              addView(mutedLabel("No models match \"$jobModelSearchText\"."))
            } else {
              addView(mutedLabel("$selectedBaseModelCount selected"))
              filteredModelGroups.forEach { group -> addView(modelGroupRow(group)) }
            }
          }
        )
        body.addView(
          card {
            addView(sectionTitle("Quants"))
            addView(mutedLabel("Filter selected model files by quantization before planning the job."))
            addView(
              chipGroup {
                JobQuantFilter.entries.forEach { filter ->
                  addView(
                    filterChip(filter.label, selectedJobQuantFilters.contains(filter)) {
                      updateJobQuantFilter(filter, it)
                      render()
                    }
                  )
                }
              }
            )
          }
        )
        body.addView(wizardNav(step, "Next", NewJobWizard.canAdvance(step, selectedBaseModelCount, selectedBenchmarksForPlanning.size)))
      }
      1 -> {
        val filteredBenchmarks = BenchmarkCatalog.selectable.filter { BenchmarkCatalog.matchesSearch(it, benchmarkSearchText) }
        body.addView(
          card {
            addView(sectionTitle("Benchmarks"))
            addView(searchBlock("Search benchmarks", benchmarkSearchText) { benchmarkSearchText = it })
            if (filteredBenchmarks.isEmpty()) {
              addView(mutedLabel("No benchmarks match \"$benchmarkSearchText\"."))
            }
            filteredBenchmarks
              .groupBy { it.benchmarkType }
              .toSortedMap(compareBy<String> { BenchmarkCatalog.typeRank(it) })
              .forEach { (type, items) ->
                val isVlGroupDisabled = type == BenchmarkType.VL_THROUGHPUT.wire && !hasVlModelSelected
                val enabledItems = if (isVlGroupDisabled) emptyList() else items
                val allGroupSelected = enabledItems.isNotEmpty() && enabledItems.all { selectedBenchmarkIds.contains(it.benchmarkId.toString()) }
                addView(label(BenchmarkCatalog.displayName(type)))
                if (isVlGroupDisabled) {
                  addView(mutedLabel("Requires a selected model with a matching MMProjector."))
                } else {
                  addView(
                    textButton(
                      if (allGroupSelected) {
                        "Clear ${BenchmarkCatalog.displayName(type)}"
                      } else {
                        "Select all ${BenchmarkCatalog.displayName(type)}"
                      }
                    ) {
                      toggleBenchmarkGroup(enabledItems, allGroupSelected)
                    }
                  )
                }
                items
                  .sortedBy { it.label }
                  .forEach { item ->
                    addView(
                      CheckBox(activity).apply {
                        text = item.label
                        val itemDisabled = item.type == BenchmarkType.VL_THROUGHPUT && !hasVlModelSelected
                        isEnabled = !itemDisabled
                        val itemId = item.benchmarkId.toString()
                        isChecked = selectedBenchmarkIds.contains(itemId) && !itemDisabled
                        setOnCheckedChangeListener { _, checked ->
                          if (itemDisabled) return@setOnCheckedChangeListener
                          if (checked) selectedBenchmarkIds += itemId else selectedBenchmarkIds -= itemId
                          render()
                        }
                      }
                    )
                  }
              }
          }
        )
        if (hasVlBenchmarkSelected) {
          body.addView(
            card {
              addView(sectionTitle("MMProjectors"))
              addView(mutedLabel("Required for VL benchmarks. Each selected MMProjector is paired with each selected compatible model."))
              if (mmprojs.isEmpty()) {
                addView(mutedLabel("No MMProjector files. Download one from the Models tab."))
              } else {
                val selectedMmprojCount = mmprojs.count { selectedMmprojPaths.contains(it.path) }
                val allSelected = selectedMmprojCount == mmprojs.size
                addView(
                  textButton(if (allSelected) "Deselect All" else "Select All") {
                    if (allSelected) {
                      selectedMmprojPaths.clear()
                    } else {
                      selectedMmprojPaths.clear()
                      selectedMmprojPaths += mmprojs.map { it.path }
                    }
                    render()
                  }
                )
                mmprojs.forEach { model ->
                  addView(
                    CheckBox(activity).apply {
                      text = "${model.name} (${model.sizeFormatted})"
                      isChecked = selectedMmprojPaths.contains(model.path)
                      setOnCheckedChangeListener { _, checked ->
                        if (checked) selectedMmprojPaths += model.path else selectedMmprojPaths -= model.path
                        render()
                      }
                    }
                  )
                }
              }
            }
          )
        }
        body.addView(wizardNav(step, "Next", NewJobWizard.canAdvance(step, selectedBaseModelCount, selectedBenchmarksForPlanning.size)))
      }
      else -> {
        val ngl = input("GPU layers", nGpuLayers.toString(), InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_SIGNED)
        val ctxField = input("Context size floor", contextSize.toString(), InputType.TYPE_CLASS_NUMBER)
        val ubatch = input("Prefill batch (n_ubatch)", prefillBatch.toString(), InputType.TYPE_CLASS_NUMBER)
        body.addView(
          card {
            addView(sectionTitle("Run setup"))
            // The fields are pre-filled, so the EditText hint is never visible —
            // give each one a standing label describing what it sets.
            addView(label("GPU layers (n_gpu_layers): model layers offloaded to the GPU"))
            addView(ngl)
            addView(label("Context size floor (tokens): minimum context per cell"))
            addView(ctxField)
            addView(label("Prefill batch (n_ubatch): prompt tokens processed per batch"))
            addView(ubatch)
            addView(
              CheckBox(activity).apply {
                text = "Auto-submit completed results"
                isChecked = contributeResults
                isEnabled = storage.isRegistered()
                setOnCheckedChangeListener { _, checked -> contributeResults = storage.isRegistered() && checked }
              }
            )
          }
        )
        body.addView(
          card {
            addView(sectionTitle("Review"))
            addView(
              label(
                plannedJobSummary(
                  selectedBaseModelCount = selectedBaseModelCount,
                  selectedRunnableModelCount = selectedRunnableBaseModels.size,
                  selectedBenchmarkCount = selectedBenchmarksForPlanning.size,
                  selectedMmprojCount = mmprojs.count { selectedMmprojPaths.contains(it.path) },
                  hasVlBenchmark = hasVlBenchmarkSelected,
                  plannedCellCount = plannedCells.size,
                )
              )
            )
            if (modelsMissingQuant.isNotEmpty()) {
              addView(statusBadge("Skipped", ui.colorThermalSerious()))
              addView(mutedLabel(missingQuantWarning(modelsMissingQuant)))
            }
          }
        )
        // Footer: Back + Run.
        body.addView(
          row {
            addView(
              outlineButton("Back") {
                  vm.newJobStep = step - 1
                  render()
                }
                .apply { layoutParams = LinearLayout.LayoutParams(0, wrap, 1f).also { it.setMargins(0, dp(6), dp(4), dp(6)) } }
            )
            val running = runnerState.runningJobId != null
            addView(
              primaryButton(if (running) "A job is running" else "Run job") {
                  nGpuLayers = ngl.text.toString().toIntOrNull() ?: nGpuLayers
                  contextSize = ctxField.text.toString().toIntOrNull() ?: contextSize
                  prefillBatch = ubatch.text.toString().toIntOrNull() ?: prefillBatch
                  runCatching {
                      val jobId =
                        runner.startNewJob(
                          models = selectedRunnableBaseModels,
                          mmprojFiles = mmprojs,
                          benchmarks = selectedBenchmarksForPlanning,
                          selectedMmprojPaths = selectedMmprojPaths,
                          nGpuLayers = nGpuLayers,
                          contextSize = contextSize,
                          prefillBatch = prefillBatch,
                          contributeResults = contributeResults,
                        )
                      statusText = "Started job $jobId"
                      vm.newJobStep = null
                      selectedJobId = jobId
                      render()
                    }
                    .onFailure { showError(it) }
                }
                .apply {
                  isEnabled = NewJobWizard.canRun(plannedCells.size, running)
                  layoutParams = LinearLayout.LayoutParams(0, wrap, 1f).also { it.setMargins(dp(4), dp(6), 0, dp(6)) }
                }
            )
          }
        )
      }
    }
  }

  /** A Back/Next footer row for wizard steps that advance (steps before Review). */
  private fun wizardNav(step: Int, primaryLabel: String, primaryEnabled: Boolean): View = row {
    if (step > 0) {
      addView(
        outlineButton("Back") {
            vm.newJobStep = step - 1
            render()
          }
          .apply { layoutParams = LinearLayout.LayoutParams(0, wrap, 1f).also { it.setMargins(0, dp(6), dp(4), dp(6)) } }
      )
    }
    addView(
      primaryButton(primaryLabel) {
          vm.newJobStep = step + 1
          render()
        }
        .apply {
          isEnabled = primaryEnabled
          layoutParams = LinearLayout.LayoutParams(0, wrap, 1f).also { it.setMargins(if (step > 0) dp(4) else 0, dp(6), 0, dp(6)) }
        }
    )
  }

  private fun renderJobDetail(body: LinearLayout, manifest: JobManifest) {
    body.addView(
      textButton("‹ Back to jobs") {
        selectedJobId = null
        expandedCellIds.clear()
        selectedRerunCellIds.clear()
        statusText = ""
        render()
      }
    )
    selectedRerunCellIds.retainAll(manifest.cells.map { it.cellId }.toSet())

    body.addView(
      card {
        addView(sectionTitle(manifest.displayTitle))
        addView(statusBadge(manifest.status.wire, jobStatusAccent(manifest.status)))
        addView(
          mutedLabel(
            "Created: ${DateFormats.shortDate(manifest.createdAt)}\n" +
              "${manifest.completedCells}/${manifest.totalCells} completed, " +
              "${manifest.failedCells} failed, ${manifest.cancelledCells} cancelled, ${manifest.submittedCells} submitted"
          )
        )
        if (runnerState.runningJobId == manifest.jobId) {
          addView(linearProgress(runnerState.currentCellFraction))
          addView(mutedLabel("${runnerState.currentCellLabel}\n${runnerState.currentProgressText}"))
          addView(outlineButton("Open in Pocket Mode") { setPocketMode(manifest.jobId) })
          addView(outlineButton("Cancel running job") { runner.cancel() })
        }
        addView(
          CheckBox(activity).apply {
            text = "Auto-submit completed results"
            isChecked = manifest.contributeResults == true
            isEnabled = storage.isRegistered()
            setOnCheckedChangeListener { _, checked -> setJobAutoSubmit(manifest.jobId, checked) }
          }
        )
      }
    )

    body.addView(
      card {
        addView(sectionTitle("Actions"))
        addView(outlineButton("Rename") { renameJob(manifest) })
        if (manifest.status == JobStatus.PAUSED && !runner.isRunning()) {
          addView(primaryButton("Resume paused cells") { runCatching { runner.resume(manifest.jobId) }.onFailure { showError(it) } })
        }
        if (manifest.failedCells > 0 && !runner.isRunning()) {
          addView(
            outlineButton("Retry failed cells") {
              runCatching {
                  runner.retryFailed(manifest.jobId)
                  statusText = "Retrying failed cells"
                  render()
                }
                .onFailure { showError(it) }
            }
          )
        }
        if (manifest.completedCells > 0) {
          addView(outlineButton("Export CSV") { exportResultsCsv(manifest) })
        }
        val selectedRerunnableCount = selectedRerunCellIds.count { cellId -> manifest.cells.any { it.cellId == cellId && it.isRerunnable } }
        if (selectedRerunnableCount > 0 && !runner.isRunning()) {
          addView(
            primaryButton("Rerun $selectedRerunnableCount selected ${plural("cell", selectedRerunnableCount)}") {
              runCatching {
                  runner.rerunCells(manifest.jobId, selectedRerunCellIds.toSet())
                  selectedRerunCellIds.clear()
                  statusText = "Rerunning selected cells"
                  render()
                }
                .onFailure { showError(it) }
            }
          )
          addView(
            textButton("Clear selected cells") {
              selectedRerunCellIds.clear()
              render()
            }
          )
        }
        val unsubmitted = storage.unsubmittedResultCount(manifest)
        if (unsubmitted > 0 && storage.isRegistered()) {
          addView(primaryButton("Submit $unsubmitted Result${if (unsubmitted == 1) "" else "s"}") { submitJobResults(manifest, unsubmitted) })
        }
        addView(
          textButton("Delete job") {
            confirm("Delete job ${manifest.displayTitle}?") {
              storage.deleteJob(manifest.jobId)
              selectedJobId = null
              expandedCellIds.clear()
              selectedRerunCellIds.clear()
              render()
            }
          }
        )
      }
    )

    val payloads = CompletedResultsCsvExporter.payloadsByCellId(storage, manifest)
    val metrics = CompletedResultsCsvExporter.metricsByCellId(manifest, payloads)
    if (metrics.isNotEmpty()) {
      body.addView(resultsCard(manifest, payloads, metrics))
    }
    body.addView(
      card {
        addView(sectionTitle("Cells"))
        manifest.cells.forEachIndexed { index, cell -> addView(cellCard(manifest, cell, index + 1, payloads[cell.cellId], metrics[cell.cellId])) }
      }
    )
  }

  // --- Results heatmap grid (model×quant rows × benchmark columns) -------------

  /**
   * A model×benchmark results grid. Rows are model+quant; columns are benchmarks. Each cell is shaded by its value's rank *within its column*, made
   * direction-aware via [CompletedRunMetric.higherIsBetter] (throughput is higher- better; latency/memory are lower-better) so "better → brighter"
   * reads correctly regardless of the metric. The model column is frozen; benchmark columns scroll horizontally. Tapping a cell opens its detail
   * (preserving inspect + rerun).
   */
  private fun resultsCard(manifest: JobManifest, payloads: Map<String, JSONObject>, metrics: Map<String, CompletedRunMetric>): View {
    val cells = manifest.cells
    val columns = cells.map { it.benchmarkId }.distinct()
    val rowOrder = mutableListOf<String>()
    val rowCells = linkedMapOf<String, MutableMap<String, JobCell>>()
    val rowLabels = linkedMapOf<String, String>()
    cells.forEach { cell ->
      val key = CompletedResultsCsvExporter.resultModelGroupKey(cell) + "|" + CompletedResultsCsvExporter.quantLabel(cell)
      if (!rowCells.containsKey(key)) {
        rowOrder += key
        rowCells[key] = mutableMapOf()
        rowLabels[key] = "${CompletedResultsCsvExporter.modelDisplayName(cell)}\n${CompletedResultsCsvExporter.quantLabel(cell)}"
      }
      rowCells.getValue(key)[cell.benchmarkId] = cell
    }
    // Per-column min/max over completed values (for normalization).
    val colRange =
      columns.associateWith { col ->
        val values = rowOrder.mapNotNull { rk -> rowCells[rk]?.get(col)?.cellId?.let { metrics[it]?.numericValue } }
        if (values.isEmpty()) null else (values.min() to values.max())
      }

    val labelW = dp(132)
    val colW = dp(108)
    val rowH = dp(60)

    fun gridCell(text: String, width: Int, bg: Int?, bold: Boolean, textColor: Int, onClick: (() -> Unit)?): TextView =
      TextView(activity).apply {
        this.text = text
        textSize = 12f
        gravity = Gravity.CENTER
        setTextColor(textColor)
        if (bold) setTypeface(typeface, Typeface.BOLD)
        setPadding(dp(6), dp(4), dp(6), dp(4))
        layoutParams = LinearLayout.LayoutParams(width, rowH).apply { setMargins(dp(1), dp(1), dp(1), dp(1)) }
        if (bg != null) setBackgroundColor(bg)
        if (onClick != null) setOnClickListener { onClick() }
      }

    // Frozen left column: corner + one label per row.
    val leftColumn =
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        addView(gridCell("Model", labelW, null, true, ui.colorMuted(), null))
        rowOrder.forEach { rk -> addView(gridCell(rowLabels.getValue(rk), labelW, null, false, ui.colorOnSurface(), null)) }
      }

    // Scrollable benchmark columns: header row + a value row per model.
    val grid = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL }
    grid.addView(
      LinearLayout(activity).apply {
        orientation = LinearLayout.HORIZONTAL
        columns.forEach { col -> addView(gridCell(columnLabel(cells.first { it.benchmarkId == col }), colW, null, true, ui.colorMuted(), null)) }
      }
    )
    rowOrder.forEach { rk ->
      grid.addView(
        LinearLayout(activity).apply {
          orientation = LinearLayout.HORIZONTAL
          columns.forEach { col ->
            val cell = rowCells[rk]?.get(col)
            val metric = cell?.cellId?.let { metrics[it] }
            when {
              cell == null -> addView(gridCell("—", colW, null, false, ui.colorMuted(), null))
              metric != null -> {
                val (lo, hi) = colRange[col] ?: (metric.numericValue to metric.numericValue)
                val intensity = ResultsGrid.heatmapIntensity(metric.numericValue, lo, hi, metric.higherIsBetter)
                // Dark text on bright (high-intensity) cells, light text on dim ones.
                val textColor = if (intensity > 0.55) ui.colorOnPrimary() else ui.colorOnSurface()
                addView(
                  gridCell(CompletedResultsCsvExporter.displayMetric(metric), colW, ui.heatmapColor(intensity), false, textColor) {
                    showCellDetailDialog(manifest, cell, payloads[cell.cellId], metric)
                  }
                )
              }
              else ->
                addView(
                  gridCell(cell.runStatus.wire, colW, null, false, ui.colorMuted()) {
                    showCellDetailDialog(manifest, cell, payloads[cell.cellId], null)
                  }
                )
            }
          }
        }
      )
    }

    val table =
      LinearLayout(activity).apply {
        orientation = LinearLayout.HORIZONTAL
        addView(leftColumn)
        addView(HorizontalScrollView(activity).apply { addView(grid) })
      }

    return card {
      addView(sectionTitle("Results"))
      addView(mutedLabel("Tap a cell to inspect. Brighter = better within each column."))
      addView(table)
    }
  }

  private fun columnLabel(cell: JobCell): String {
    val type = CompletedResultsCsvExporter.benchmarkType(cell)
    val short =
      when (type) {
        "prefill_throughput" -> "Prefill"
        "decode_throughput" -> "Decode"
        "end_to_end_latency" -> "E2E"
        "max_memory_usage" -> "Memory"
        "vl_throughput" -> "VL"
        else -> BenchmarkCatalog.displayName(type)
      }
    val params = CompletedResultsCsvExporter.parameterSummary(cell.benchmarkId)
    return if (params != null) "$short\n$params" else short
  }

  private fun showCellDetailDialog(manifest: JobManifest, cell: JobCell, payload: JSONObject?, metric: CompletedRunMetric?) {
    val type = CompletedResultsCsvExporter.benchmarkType(cell)
    val params = CompletedResultsCsvExporter.parameterSummary(cell.benchmarkId)
    val message =
      buildString {
          appendLine("${CompletedResultsCsvExporter.modelDisplayName(cell)} · ${CompletedResultsCsvExporter.quantLabel(cell)}")
          appendLine(BenchmarkCatalog.displayName(type) + (params?.let { " · $it" } ?: ""))
          appendLine("Status: ${cell.runStatus.wire}")
          if (!cell.errorMessage.isNullOrBlank()) appendLine("Error: ${cell.errorMessage}")
          if (payload != null) {
            // payloadDetailRows already prepends the metric row, so don't add it twice.
            payloadDetailRows(payload, metric).forEach { (k, v) -> appendLine("$k: $v") }
          } else if (metric != null) {
            appendLine("${metric.name}: ${CompletedResultsCsvExporter.displayMetric(metric)}")
          }
        }
        .trim()
    val builder = AlertDialog.Builder(activity).setTitle("Cell result").setMessage(message).setPositiveButton("Close", null)
    if (cell.isRerunnable && !runner.isRunning()) {
      builder.setNeutralButton("Rerun") { _, _ ->
        runCatching {
            runner.rerunCells(manifest.jobId, setOf(cell.cellId))
            statusText = "Rerunning cell"
            render()
          }
          .onFailure { showError(it) }
      }
    }
    builder.show()
  }

  /** Accent color for a job status badge. */
  private fun jobStatusAccent(status: JobStatus): Int =
    when (status) {
      JobStatus.COMPLETED,
      JobStatus.RUNNING -> ui.colorThermalNominal()
      JobStatus.PAUSED -> ui.colorThermalSerious()
      JobStatus.CANCELLED -> ui.colorThermalCritical()
      JobStatus.PLANNED -> ui.colorMuted()
    }

  private fun cellCard(manifest: JobManifest, cell: JobCell, position: Int, payload: JSONObject?, metric: CompletedRunMetric?): View = tile {
    val benchmarkType = CompletedResultsCsvExporter.benchmarkType(cell)
    val parameter = CompletedResultsCsvExporter.parameterSummary(cell.benchmarkId)
    val benchmarkLabel = buildString {
      append(BenchmarkCatalog.displayName(benchmarkType))
      if (parameter != null) append(" - $parameter")
    }
    val submission = storage.loadSubmission(manifest.jobId, cell.cellId)
    addView(label("Cell $position - ${CompletedResultsCsvExporter.modelDisplayName(cell)}"))
    addView(statusBadge(cell.runStatus.wire, cellStatusAccent(cell.runStatus)))
    addView(mutedLabel("${CompletedResultsCsvExporter.quantLabel(cell)}\n$benchmarkLabel"))
    if (metric != null) {
      addView(label("${metric.name}: ${CompletedResultsCsvExporter.displayMetric(metric)}"))
    }
    if (!cell.errorMessage.isNullOrBlank()) {
      addView(mutedLabel("Error: ${cell.errorMessage}"))
    }
    when {
      !cell.serverJobId.isNullOrBlank() -> addView(mutedLabel("Submitted: ${cell.serverJobId}"))
      submission?.status == "submitted" -> addView(mutedLabel("Submitted: ${submission.serverJobId ?: "recorded"}"))
      submission?.status == "failed" -> addView(mutedLabel("Submission failed: ${submission.errors.joinToString("; ")}"))
    }

    val expanded = expandedCellIds.contains(cell.cellId)
    if (cell.isRerunnable) {
      addView(
        CheckBox(context).apply {
          text = "Select for rerun"
          isChecked = selectedRerunCellIds.contains(cell.cellId)
          setOnCheckedChangeListener { _, checked ->
            if (checked) selectedRerunCellIds += cell.cellId else selectedRerunCellIds -= cell.cellId
            render()
          }
        }
      )
    }
    val canSubmit = storage.isUnsubmitted(manifest.jobId, cell) && storage.isRegistered()
    addView(
      row {
        addView(
          textButton(if (expanded) "Hide details" else "Details") {
            if (expanded) expandedCellIds -= cell.cellId else expandedCellIds += cell.cellId
            render()
          }
        )
        if (cell.isRerunnable) {
          addView(
            textButton("Rerun cell") {
              runCatching {
                  runner.rerunCells(manifest.jobId, setOf(cell.cellId))
                  selectedRerunCellIds -= cell.cellId
                  statusText = "Rerunning cell"
                  render()
                }
                .onFailure { showError(it) }
            }
          )
        }
        if (canSubmit) {
          addView(textButton("Submit cell") { submitCellResult(manifest.jobId, cell.cellId) })
        }
      }
    )

    if (expanded) {
      addView(label("Cell details"))
      addView(mutedLabel("Model path: ${cell.modelPath}"))
      cell.mmprojPath?.let { addView(mutedLabel("MMProjector: $it")) }
      if (payload == null) {
        addView(mutedLabel("No saved result payload was found for this cell."))
      } else {
        payloadDetailRows(payload, metric).forEach { (title, value) -> addView(mutedLabel("$title: $value")) }
      }
    }
  }

  /** Accent color for a cell run-status badge. */
  private fun cellStatusAccent(status: CellRunStatus): Int =
    when (status) {
      CellRunStatus.COMPLETED -> ui.colorThermalNominal()
      CellRunStatus.RUNNING -> ui.colorThermalNominal()
      CellRunStatus.FAILED -> ui.colorThermalCritical()
      CellRunStatus.CANCELLED -> ui.colorThermalSerious()
      CellRunStatus.PENDING -> ui.colorMuted()
    }

  private fun payloadDetailRows(payload: JSONObject, metric: CompletedRunMetric?): List<Pair<String, String>> {
    val hiddenKeys =
      setOf(
        "benchmark_flags",
        "benchmark_id",
        "cell_id",
        "completions",
        "device_name",
        "job_id",
        "model_descriptor",
        "model_name",
        "model_quant",
        "runtime_descriptor",
        "runtime_flags",
        "submitted_at",
      )
    val rows = mutableListOf<Pair<String, String>>()
    if (metric != null) rows += metric.name to CompletedResultsCsvExporter.displayMetric(metric)
    payload
      .keys()
      .asSequence()
      .filterNot { hiddenKeys.contains(it) }
      .sorted()
      .forEach { key -> payloadScalarString(payload.opt(key))?.let { value -> rows += humanizedKey(key) to value } }
    return rows
  }

  private fun renameJob(manifest: JobManifest) {
    val field = input("Name", manifest.title ?: "")
    AlertDialog.Builder(activity)
      .setTitle("Rename job")
      .setMessage("Leave empty to reset to the default name.")
      .setView(field)
      .setPositiveButton("Save") { _, _ -> saveJobTitle(manifest.jobId, field.text.toString()) }
      .setNeutralButton("Reset") { _, _ -> saveJobTitle(manifest.jobId, "") }
      .setNegativeButton("Cancel", null)
      .show()
  }

  private fun saveJobTitle(jobId: String, title: String) {
    val manifest = storage.loadJobManifest(jobId) ?: return
    manifest.title = title.trim().takeIf { it.isNotEmpty() }
    storage.saveJobManifest(manifest)
    render()
  }

  private fun setJobAutoSubmit(jobId: String, enabled: Boolean) {
    val manifest = storage.loadJobManifest(jobId) ?: return
    manifest.contributeResults = storage.isRegistered() && enabled
    storage.saveJobManifest(manifest)
    render()
  }

  private fun submitCellResult(jobId: String, cellId: String) {
    val registration = storage.loadRegistration()
    if (registration == null) {
      showError(IllegalStateException("Register the device before submitting results"))
      return
    }
    runInBackground("Submitting cell...") {
      val record = submissionService.submitCell(jobId, cellId, registration) ?: throw IllegalStateException("No payload is available for this cell")
      val manifest = storage.loadJobManifest(jobId)
      if (manifest != null && record.status == "submitted") {
        manifest.cells.firstOrNull { it.cellId == cellId }?.serverJobId = record.serverJobId
        storage.saveJobManifest(manifest)
      }
      if (record.status == "submitted") {
        "Submitted cell"
      } else {
        "Cell submission failed: ${record.errors.joinToString("; ")}"
      }
    }
  }

  private fun payloadScalarString(value: Any?): String? {
    if (value == null || value == JSONObject.NULL) return null
    return when (value) {
      is String -> value
      is Boolean -> if (value) "true" else "false"
      is Number -> {
        val double = value.toDouble()
        val rounded = kotlin.math.round(double)
        if (kotlin.math.abs(rounded - double) < 0.0001) {
          rounded.toLong().toString()
        } else {
          String.format("%.2f", double)
        }
      }
      else -> null
    }
  }

  private fun humanizedKey(key: String): String =
    key.replace("_ms", " ms").replace("_bytes", " bytes").split("_").joinToString(" ") { word -> word.replaceFirstChar { it.titlecase() } }

  private fun modelGroupRow(group: ModelGroup): View =
    CheckBox(activity).apply {
      text = "${group.name}\n${group.quantSummary} - ${ByteFormat.fileSize(group.files.sumOf { it.sizeBytes })}"
      isChecked = selectedModelKeys.contains(group.key)
      setOnCheckedChangeListener { _, checked ->
        if (checked) selectedModelKeys += group.key else selectedModelKeys -= group.key
        render()
      }
    }

  private fun modelGroupMatchesSearch(group: ModelGroup, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    return listOf(group.name, group.key, group.quantSummary).any { it.lowercase().contains(q) } || group.files.any { modelMatchesSearch(it, q) }
  }

  private fun jobMatchesSearch(manifest: JobManifest, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    val benchmarkLabels =
      manifest.cells.mapNotNull { cell ->
        BenchmarkCatalog.resolve(cell.benchmarkId)?.let { item ->
          "${item.benchmarkId} ${BenchmarkCatalog.displayName(item.benchmarkType)} ${item.label}"
        }
      }
    return listOf(
        manifest.displayTitle,
        manifest.jobId,
        manifest.createdAt,
        manifest.cells.joinToString(" ") { it.modelName },
        manifest.cells.joinToString(" ") { it.benchmarkId },
        benchmarkLabels.joinToString(" "),
      )
      .any { it.lowercase().contains(q) }
  }

  private fun updateJobQuantFilter(filter: JobQuantFilter, checked: Boolean) {
    if (checked) {
      if (filter == JobQuantFilter.ALL) {
        selectedJobQuantFilters.clear()
        selectedJobQuantFilters += JobQuantFilter.ALL
      } else {
        selectedJobQuantFilters -= JobQuantFilter.ALL
        selectedJobQuantFilters += filter
      }
    } else {
      selectedJobQuantFilters -= filter
    }
    if (selectedJobQuantFilters.isEmpty()) selectedJobQuantFilters += JobQuantFilter.ALL
  }

  private fun jobQuantMatches(quant: String?): Boolean =
    selectedJobQuantFilters.contains(JobQuantFilter.ALL) || selectedJobQuantFilters.any { it.matches(quant) }

  private fun quantSummaryForWarning(): String =
    if (selectedJobQuantFilters.contains(JobQuantFilter.ALL)) {
      "selected quant"
    } else {
      JobQuantFilter.entries.filter { it != JobQuantFilter.ALL && selectedJobQuantFilters.contains(it) }.joinToString(", ") { it.label }
    }

  private fun missingQuantWarning(groups: List<ModelGroup>): String {
    val names = groups.joinToString(", ") { it.name }
    val verb = if (groups.size == 1) "has" else "have"
    return "$names $verb no ${quantSummaryForWarning()} build downloaded and will be skipped."
  }

  private fun plannedJobSummary(
    selectedBaseModelCount: Int,
    selectedRunnableModelCount: Int,
    selectedBenchmarkCount: Int,
    selectedMmprojCount: Int,
    hasVlBenchmark: Boolean,
    plannedCellCount: Int,
  ): String {
    val base = buildString {
      append("Planned cells: $plannedCellCount\n")
      append("$selectedRunnableModelCount runnable model ${plural("file", selectedRunnableModelCount)}")
      append(" - $selectedBenchmarkCount selected ${plural("benchmark", selectedBenchmarkCount)}")
      if (hasVlBenchmark) {
        append(" - $selectedMmprojCount selected ${plural("MMProjector", selectedMmprojCount)}")
      }
    }
    val warning =
      when {
        selectedBaseModelCount == 0 -> "Select at least one model."
        selectedRunnableModelCount == 0 -> "Selected models are filtered out by the active quant filter."
        selectedBenchmarkCount == 0 -> "Select at least one benchmark."
        plannedCellCount == 0 && hasVlBenchmark && selectedMmprojCount == 0 -> "Select at least one MMProjector for VL benchmarks."
        plannedCellCount == 0 && hasVlBenchmark -> "No selected model/MMProjector pairing can run the selected VL benchmarks."
        plannedCellCount == 0 -> "No runnable cells for the selected model/benchmark set."
        else -> null
      }
    return warning?.let { "$base\n$it" } ?: base
  }

  private fun pruneVlBenchmarks() {
    selectedBenchmarkIds.removeAll(
      BenchmarkCatalog.selectable.filter { it.type == BenchmarkType.VL_THROUGHPUT }.map { it.benchmarkId.toString() }.toSet()
    )
  }

  private fun toggleBenchmarkGroup(items: List<BenchmarkDefinition>, allSelected: Boolean) {
    if (allSelected) {
      selectedBenchmarkIds.removeAll(items.map { it.benchmarkId.toString() }.toSet())
    } else {
      selectedBenchmarkIds.addAll(items.map { it.benchmarkId.toString() })
    }
    render()
  }

  private fun jobCard(manifest: JobManifest): View = tile {
    addView(label(manifest.displayTitle))
    addView(statusBadge(manifest.status.wire, jobStatusAccent(manifest.status)))
    if (runnerState.runningJobId == manifest.jobId) {
      addView(linearProgress(runnerState.currentCellFraction))
    }
    addView(
      mutedLabel(
        "${manifest.completedCells}/${manifest.totalCells} completed, " +
          "${manifest.failedCells} failed, ${manifest.cancelledCells} cancelled, ${manifest.submittedCells} submitted"
      )
    )
    val failed = manifest.cells.firstOrNull { it.runStatus == CellRunStatus.FAILED && it.errorMessage != null }
    if (failed != null) addView(mutedLabel("First failure: ${failed.benchmarkId} - ${failed.errorMessage}"))
    addView(
      primaryButton("Details") {
        selectedJobId = manifest.jobId
        expandedCellIds.clear()
        statusText = ""
        render()
      }
    )
    if (runnerState.runningJobId == manifest.jobId) {
      addView(outlineButton("Open in Pocket Mode") { setPocketMode(manifest.jobId) })
    }
    if (manifest.status == JobStatus.PAUSED && !runner.isRunning()) {
      addView(outlineButton("Resume") { runCatching { runner.resume(manifest.jobId) }.onFailure { showError(it) } })
    }
    if (manifest.completedCells > 0) {
      addView(textButton("Export CSV") { exportResultsCsv(manifest) })
    }
    val unsubmitted = storage.unsubmittedResultCount(manifest)
    if (unsubmitted > 0 && storage.isRegistered()) {
      addView(textButton("Submit $unsubmitted Result${if (unsubmitted == 1) "" else "s"}") { submitJobResults(manifest, unsubmitted) })
    }
    addView(
      textButton("Delete") {
        confirm("Delete job ${manifest.displayTitle}?") {
          storage.deleteJob(manifest.jobId)
          render()
        }
      }
    )
  }

  private fun exportResultsCsv(manifest: JobManifest) {
    runCatching { ctx.exportCsv(CompletedResultsCsvExporter.filename(manifest), CompletedResultsCsvExporter.csv(storage, manifest)) }
      .onFailure { showError(it) }
  }

  private fun submitJobResults(manifest: JobManifest, unsubmitted: Int) {
    val registration = storage.loadRegistration()
    if (registration == null) {
      showError(IllegalStateException("Register the device before submitting results"))
      return
    }
    confirm("Submit $unsubmitted completed result${if (unsubmitted == 1) "" else "s"}?", "Submit") {
      runInBackground("Submitting results...") {
        val latest = storage.loadJobManifest(manifest.jobId) ?: manifest
        val outcome = submissionService.submit(latest, registration)
        if (outcome.errors.isEmpty()) {
          "Submitted ${outcome.submitted} result${if (outcome.submitted == 1) "" else "s"}"
        } else {
          "Submitted ${outcome.submitted}; ${outcome.errors.size} error${if (outcome.errors.size == 1) "" else "s"}"
        }
      }
    }
  }
}
