package ai.liquid.pipette

import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Model management, matched to the iOS client. The main screen is "Your models": a title + "Add models" pill, a search field, and either a rich empty
 * state or downloaded models grouped by family (expandable to per-quant rows). "Add models" is a sub-screen: a family list with vendor logos and
 * sizes, a quantization checkbox card, and a "Download N models" footer. The HF token lives in Settings; there's no manual URL field (local GGUF
 * sideload stays, via the system picker).
 */
class ModelsScreen(ctx: ScreenContext) : Screen(ctx) {
  override fun renderBody(body: LinearLayout) {
    if (vm.modelsShowAddScreen) renderAddModels(body) else renderMainModels(body)
  }

  // --- Main screen: "Your models" ---

  private fun renderMainModels(body: LinearLayout) {
    val models = storage.availableModels()
    ensureMmprojSelectionInitialized(models.filter { it.isMmproj })
    val activeDownloads = downloadCoordinator.activeDownloads()
    val filtered = models.filter { modelMatchesSearch(it, downloadedModelSearchText) }
    val groups = ModelCatalog.groups(filtered.filterNot { it.isMmproj })
    val mmprojs = filtered.filter { it.isMmproj }

    body.addView(
      row {
        layoutParams = LinearLayout.LayoutParams(match, wrap)
        addView(ui.displayTitle("Your models").apply { layoutParams = LinearLayout.LayoutParams(0, wrap, 1f) })
        addView(ui.pillButton("Add models", R.drawable.ic_search) { openAddModels() })
      }
    )
    body.addView(
      ui.iconSearchField("Search your downloaded models", downloadedModelSearchText) {
        downloadedModelSearchText = it
        render()
      }
    )

    when {
      models.isEmpty() && activeDownloads.isEmpty() -> body.addView(emptyStateCard())
      groups.isEmpty() && mmprojs.isEmpty() && activeDownloads.isEmpty() ->
        body.addView(card { addView(mutedLabel("No downloaded models match \"$downloadedModelSearchText\".")) })
      else -> body.addView(downloadedCard(activeDownloads, groups, mmprojs))
    }

    body.addView(
      ui
        .pillButton("Add local GGUF") { ctx.openModel() }
        .apply { layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 6.dp, 0, 6.dp) } }
    )
  }

  private fun emptyStateCard(): View = card {
    gravity = Gravity.CENTER_HORIZONTAL
    addView(
      ImageView(activity).apply {
        setImageResource(R.drawable.models_empty_state)
        scaleType = ImageView.ScaleType.FIT_CENTER
        layoutParams =
          LinearLayout.LayoutParams(ILLUSTRATION_W_DP.dp, ILLUSTRATION_H_DP.dp).apply {
            gravity = Gravity.CENTER_HORIZONTAL
            setMargins(0, 24.dp, 0, 16.dp)
          }
      }
    )
    addView(
      ui.displayTitle("No models downloaded").apply {
        gravity = Gravity.CENTER
        layoutParams = LinearLayout.LayoutParams(match, wrap)
      }
    )
    addView(
      mutedLabel("No models downloaded. Download models to select for a benchmarking job.").apply {
        gravity = Gravity.CENTER
        layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 4.dp, 0, 0) }
      }
    )
    addView(
      ui
        .pillButton("Search models", R.drawable.ic_search, filled = true) { openAddModels() }
        .apply {
          (layoutParams as LinearLayout.LayoutParams).apply {
            gravity = Gravity.CENTER_HORIZONTAL
            topMargin = 16.dp
          }
        }
    )
  }

  private fun downloadedCard(activeDownloads: List<ActiveDownload>, groups: List<ModelGroup>, mmprojs: List<ModelFile>): View = card {
    val blocks = buildList {
      activeDownloads.forEach { add(activeDownloadRow(it)) }
      groups.forEach { add(groupBlock(it)) }
      mmprojs.forEach { add(modelManagementRow(it)) }
    }
    blocks.forEachIndexed { index, view ->
      if (index > 0) addView(ui.divider())
      addView(view)
    }
  }

  /** An expandable family block: header (logo + name + quant summary + size + chevron); expanding lists each downloaded quant with a delete. */
  private fun groupBlock(group: ModelGroup): View {
    val expanded = expandedModelGroupKeys.contains(group.key)
    val totalBytes = group.files.sumOf { it.sizeBytes }
    return LinearLayout(activity).apply {
      orientation = LinearLayout.VERTICAL
      layoutParams = LinearLayout.LayoutParams(match, wrap)
      val header = row {
        layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 12.dp, 0, 12.dp) }
        addView(ui.brandLogo(group.name, group.files.firstOrNull()?.hfRepo))
        addView(
          LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, wrap, 1f)
            addView(label(group.name))
            addView(mutedLabel("${group.quantSummary} · ${ByteFormat.fileSize(totalBytes)}"))
          }
        )
        addView(
          TextView(activity).apply {
            text = if (expanded) "▾" else "▸"
            textSize = CHEVRON_TEXT_SP
            setTextColor(ui.colorMuted())
          }
        )
      }
      header.setOnClickListener {
        if (expanded) expandedModelGroupKeys -= group.key else expandedModelGroupKeys += group.key
        render()
      }
      addView(header)
      if (expanded) group.files.forEach { addView(quantRow(it)) }
    }
  }

  private fun quantRow(file: ModelFile): View = row {
    layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(8.dp, 6.dp, 0, 6.dp) }
    addView(
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        layoutParams = LinearLayout.LayoutParams(0, wrap, 1f)
        addView(label(file.quant ?: file.name))
        addView(mutedLabel("${file.hfRepo ?: "sideloaded"} · ${file.sizeFormatted}"))
      }
    )
    addView(textButton("Delete") { deleteModel(file) })
  }

  private fun modelManagementRow(model: ModelFile): View = row {
    layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 10.dp, 0, 10.dp) }
    addView(
      LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        layoutParams = LinearLayout.LayoutParams(0, wrap, 1f)
        addView(label(model.displayName ?: model.name))
        addView(mutedLabel("${model.hfRepo ?: "sideloaded"} · ${model.sizeFormatted}" + (model.quant?.let { " · $it" } ?: "")))
      }
    )
    addView(textButton("Delete") { deleteModel(model) })
  }

  private fun deleteModel(model: ModelFile) {
    confirm("Delete ${model.name}?\n\nThis permanently removes ${model.sizeFormatted} from this device.") {
      storage.deleteModel(model)
      selectedModelKeys -= ModelCatalog.groupKey(model)
      selectedMmprojPaths -= model.path
      statusText = "Deleted ${model.name}"
      render()
    }
  }

  private fun activeDownloadRow(download: ActiveDownload): View {
    val progress =
      if (download.totalBytes > 0) {
        "${ByteFormat.fileSize(download.bytesRead)} / ${ByteFormat.fileSize(download.totalBytes)}"
      } else {
        ByteFormat.fileSize(download.bytesRead)
      }
    val fraction = if (download.totalBytes > 0) download.bytesRead.toDouble() / download.totalBytes.toDouble() else 0.0
    return LinearLayout(activity).apply {
      orientation = LinearLayout.VERTICAL
      layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 10.dp, 0, 10.dp) }
      addView(label(download.filename))
      addView(statusBadge(download.displayLabel, ui.colorMuted()))
      addView(linearProgress(fraction))
      addView(mutedLabel("${download.repo ?: "direct URL"} · $progress\n${download.message}"))
      addView(
        row {
          if (download.canPause) {
            addView(
              textButton("Pause") {
                downloadCoordinator.pause(download.key)
                render()
              }
            )
          }
          if (download.canResume) {
            addView(
              textButton("Resume") {
                downloadCoordinator.resume(download.key)
                render()
              }
            )
          }
          addView(
            textButton("Cancel") {
              downloadCoordinator.cancel(download.key)
              statusText = "Cancelling ${download.filename}"
              render()
            }
          )
        }
      )
    }
  }

  // --- "Add models" sub-screen ---

  private fun renderAddModels(body: LinearLayout) {
    val families = ModelTemplateCatalog.families
    val filtered = families.filter { familyMatchesSearch(it, templateSearchText) }
    val allFamilyIds = families.map { it.id }.toSet()
    val allSelected = allFamilyIds.isNotEmpty() && selectedAddFamilyIds.containsAll(allFamilyIds)
    val resolved = resolvedAddPresets()

    body.addView(
      row {
        layoutParams = LinearLayout.LayoutParams(match, wrap)
        addView(textButton("‹") { closeAddModels() })
        addView(
          sectionTitle("Add models").apply {
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0, wrap, 1f)
          }
        )
        addView(View(activity).apply { layoutParams = LinearLayout.LayoutParams(44.dp, 1) })
      }
    )
    body.addView(ui.divider())

    body.addView(
      row {
        layoutParams = LinearLayout.LayoutParams(match, wrap)
        addView(sectionTitle("Download models").apply { layoutParams = LinearLayout.LayoutParams(0, wrap, 1f) })
        addView(
          ui.pillButton(if (allSelected) "Selected all" else "Select all") {
            selectedAddFamilyIds.clear()
            if (!allSelected) selectedAddFamilyIds.addAll(allFamilyIds)
            render()
          }
        )
      }
    )
    body.addView(mutedLabel("Select the models to download for benchmarking."))
    body.addView(
      ui.iconSearchField("Search models", templateSearchText) {
        templateSearchText = it
        render()
      }
    )

    body.addView(
      card {
        if (filtered.isEmpty()) {
          addView(mutedLabel("No models match \"$templateSearchText\"."))
        } else {
          filtered.forEachIndexed { index, family ->
            if (index > 0) addView(ui.divider())
            addView(familyRow(family))
          }
        }
      }
    )

    body.addView(sectionTitle("Quantizations"))
    body.addView(mutedLabel("Specify level of quantization to download"))
    body.addView(
      card {
        ModelQuantFilter.pills.forEachIndexed { index, pill ->
          if (index > 0) addView(ui.divider())
          addView(quantSelectorRow(pill))
        }
      }
    )

    body.addView(
      ui.pillButton(
        "Download ${resolved.size} model${if (resolved.size == 1) "" else "s"}",
        R.drawable.ic_download,
        filled = true,
        fullWidth = true,
        enabled = resolved.isNotEmpty(),
      ) {
        attemptAddDownload(resolved)
      }
    )
  }

  private fun familyRow(family: ModelFamily): View {
    val selected = selectedAddFamilyIds.contains(family.id)
    val repo = family.variants.firstOrNull()?.repo
    val size = family.variants.firstOrNull()?.sizeLabel.orEmpty()
    val view = row {
      layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 12.dp, 0, 12.dp) }
      addView(ui.brandLogo(family.displayName, repo))
      addView(
        LinearLayout(activity).apply {
          orientation = LinearLayout.VERTICAL
          layoutParams = LinearLayout.LayoutParams(0, wrap, 1f)
          addView(label(family.displayName))
          if (size.isNotBlank()) addView(mutedLabel(size))
        }
      )
      addView(ui.wizardCheckbox(selected))
    }
    view.setOnClickListener {
      if (selected) selectedAddFamilyIds -= family.id else selectedAddFamilyIds += family.id
      render()
    }
    return view
  }

  private fun quantSelectorRow(pill: ModelQuantFilter): View {
    val selected = selectedAddQuants.contains(pill)
    val view = row {
      layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, 12.dp, 0, 12.dp) }
      addView(label(pill.label).apply { layoutParams = LinearLayout.LayoutParams(0, wrap, 1f) })
      addView(ui.wizardCheckbox(selected))
    }
    view.setOnClickListener {
      val next = ModelQuantFilter.toggled(selectedAddQuants, pill)
      selectedAddQuants.clear()
      selectedAddQuants.addAll(next)
      render()
    }
    return view
  }

  /** Presets to download: selected families × selected quants, minus anything already downloaded or in flight. */
  private fun resolvedAddPresets(): List<PresetModel> {
    val unavailable = unavailableDownloadKeys()
    return ModelTemplateCatalog.defaults.filter { preset ->
      selectedAddFamilyIds.contains(preset.familyId) &&
        ModelQuantFilter.matchesSelection(selectedAddQuants, preset.quant) &&
        !unavailable.contains(preset.downloadKey)
    }
  }

  private fun unavailableDownloadKeys(): Set<String> {
    val downloaded = storage.availableModels().map { LocalStorage.modelRelativePath(it.hfRepo, it.name) }.toSet()
    val active = downloadCoordinator.activeDownloads().map { it.key }.toSet()
    return downloaded + active
  }

  private fun attemptAddDownload(selected: List<PresetModel>) {
    if (selected.isEmpty()) {
      showError(IllegalStateException("Select at least one model"))
      return
    }
    val totalBytes = selected.sumOf { it.estimatedBytes }
    if (totalBytes > LARGE_DOWNLOAD_WARNING_BYTES) {
      confirm(
        message =
          "Download size over ${ByteFormat.fileSize(LARGE_DOWNLOAD_WARNING_BYTES)}.\n\n" +
            "Selected models total about ${ByteFormat.fileSize(totalBytes)}.",
        positiveText = "Proceed",
      ) {
        startDownloads(selected)
      }
      return
    }
    startDownloads(selected)
  }

  private fun startDownloads(selected: List<PresetModel>) {
    // Capture only the retained ViewModel in the download callbacks — never `this` Screen (which holds the Activity). The callbacks live in the
    // process-global DownloadRegistry for the whole download, so capturing the Activity would leak it across a rotation.
    val vm = this.vm
    val total = selected.size
    runInBackground("Queueing $total model${if (total == 1) "" else "s"}...") {
      var queued = 0
      val skipped = mutableListOf<String>()
      selected.forEachIndexed { index, preset ->
        val label = "${preset.name} ${preset.quant ?: ""}".trim()
        try {
          downloadCoordinator.enqueueDownload(
            urlString = preset.identifier,
            repo = preset.repoIdentifier,
            familyId = preset.familyId,
            displayName = preset.name,
            onProgress = { progress -> vm.postDownloadStatus("Downloading ${index + 1}/$total: $label - ${progress.message}") },
            onComplete = { model -> vm.postDownloadStatus("Downloaded ${model.name}") },
            onFailure = { error -> vm.postDownloadStatus("Download failed: ${error.message ?: label}") },
          )
          queued++
        } catch (error: IllegalArgumentException) {
          // Already downloaded / already in flight / unparseable preset — skip it and keep queueing the rest of the batch.
          skipped += (error.message ?: label)
        } catch (error: IllegalStateException) {
          skipped += (error.message ?: label)
        }
      }
      // Selection sets aren't thread-safe; mutate on the main thread. Leaving the
      // add screen drops back to the (now-updated) downloaded list.
      onMain { closeAddModels() }
      if (skipped.isEmpty()) "Queued $queued model${if (queued == 1) "" else "s"}" else "Queued $queued, skipped ${skipped.size} already present"
    }
  }

  private fun openAddModels() {
    vm.modelsShowAddScreen = true
    templateSearchText = ""
    render()
  }

  private fun closeAddModels() {
    selectedAddFamilyIds.clear()
    vm.modelsShowAddScreen = false
    render()
  }

  private fun familyMatchesSearch(family: ModelFamily, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    val fields = listOf(family.displayName, family.id) + family.variants.flatMap { listOf(it.quant, it.repo) }
    return fields.any { it.lowercase().contains(q) }
  }

  private companion object {
    const val LARGE_DOWNLOAD_WARNING_BYTES = 200L * 1024L * 1024L
    const val CHEVRON_TEXT_SP = 16f
    const val ILLUSTRATION_W_DP = 250
    const val ILLUSTRATION_H_DP = 103
  }
}
