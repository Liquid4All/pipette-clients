package ai.liquid.pipette

import android.text.InputType
import android.view.View
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity

/**
 * Wiring handed to each per-screen view-controller: the host Activity (used as a view [android.content.Context] and for launchers), the shared
 * [MainViewModel], the [UiKit] factory, and the system-document launchers that can only be driven from the Activity.
 */
class ScreenContext(
  val activity: AppCompatActivity,
  val vm: MainViewModel,
  val ui: UiKit,
  val openModel: () -> Unit,
  val exportCsv: (filename: String, csv: String) -> Unit,
)

/**
 * Base for the per-tab view-controllers. The delegating accessors below let the render bodies read like the old monolithic `MainActivity` methods
 * (same `storage`, `runner`, `selectedModelKeys`, `button(...)`, `render()` names), which is what keeps the split a near-verbatim move rather than a
 * rewrite. The mutable state now lives in [MainViewModel] so it survives Activity recreation.
 */
abstract class Screen(protected val ctx: ScreenContext) {
  protected val vm
    get() = ctx.vm

  protected val activity
    get() = ctx.activity

  // Collaborators (app-scoped).
  protected val storage
    get() = vm.container.storage

  protected val settingsStore
    get() = vm.container.settingsStore

  protected val secrets
    get() = vm.container.secrets

  protected val registrationService
    get() = vm.container.registrationService

  protected val downloadCoordinator
    get() = vm.container.downloadCoordinator

  protected val submissionService
    get() = vm.container.submissionService

  protected val thermalStatusProvider
    get() = vm.container.thermalStatusProvider

  protected val runner
    get() = vm.container.jobController

  protected val runnerState
    get() = vm.runnerState.value

  // Shared UI / navigation state (backed by the ViewModel).
  protected var selectedTab: Tab
    get() = vm.selectedTab
    set(value) {
      vm.selectedTab = value
    }

  protected var statusText: String
    get() = vm.statusText
    set(value) {
      vm.statusText = value
    }

  protected val selectedModelKeys
    get() = vm.selectedModelKeys

  protected val selectedBenchmarkIds
    get() = vm.selectedBenchmarkIds

  protected val selectedMmprojPaths
    get() = vm.selectedMmprojPaths

  protected val selectedJobQuantFilters
    get() = vm.selectedJobQuantFilters

  protected val expandedCellIds
    get() = vm.expandedCellIds

  protected val selectedRerunCellIds
    get() = vm.selectedRerunCellIds

  protected val selectedAddFamilyIds
    get() = vm.selectedAddFamilyIds

  protected val selectedAddQuants
    get() = vm.selectedAddQuants

  protected val expandedModelGroupKeys
    get() = vm.expandedModelGroupKeys

  protected var mmprojSelectionInitialized: Boolean
    get() = vm.mmprojSelectionInitialized
    set(value) {
      vm.mmprojSelectionInitialized = value
    }

  protected var downloadedModelSearchText: String
    get() = vm.downloadedModelSearchText
    set(value) {
      vm.downloadedModelSearchText = value
    }

  protected var templateSearchText: String
    get() = vm.templateSearchText
    set(value) {
      vm.templateSearchText = value
    }

  protected var jobSearchText: String
    get() = vm.jobSearchText
    set(value) {
      vm.jobSearchText = value
    }

  protected var benchmarkSearchText: String
    get() = vm.benchmarkSearchText
    set(value) {
      vm.benchmarkSearchText = value
    }

  protected var jobModelSearchText: String
    get() = vm.jobModelSearchText
    set(value) {
      vm.jobModelSearchText = value
    }

  protected var nGpuLayers: Int
    get() = vm.nGpuLayers
    set(value) {
      vm.nGpuLayers = value
    }

  protected var contextSize: Int
    get() = vm.contextSize
    set(value) {
      vm.contextSize = value
    }

  protected var prefillBatch: Int
    get() = vm.prefillBatch
    set(value) {
      vm.prefillBatch = value
    }

  protected var contributeResults: Boolean
    get() = vm.contributeResults
    set(value) {
      vm.contributeResults = value
    }

  protected var pendingCsvExportText: String?
    get() = vm.pendingCsvExportText
    set(value) {
      vm.pendingCsvExportText = value
    }

  protected var selectedJobId: String?
    get() = vm.selectedJobId
    set(value) {
      vm.selectedJobId = value
    }

  // UI factory passthrough.
  protected val ui
    get() = ctx.ui

  protected val match = UiKit.MATCH
  protected val wrap = UiKit.WRAP

  protected fun dp(value: Int) = ctx.ui.dp(value)

  protected fun label(text: String) = ctx.ui.label(text)

  protected fun mutedLabel(text: String) = ctx.ui.mutedLabel(text)

  protected fun sectionTitle(text: String) = ctx.ui.sectionTitle(text)

  protected fun displayTitle(text: String) = ctx.ui.displayTitle(text)

  protected fun input(hint: String, value: String, type: Int = InputType.TYPE_CLASS_TEXT) = ctx.ui.input(hint, value, type)

  protected fun button(text: String, onClick: () -> Unit) = ctx.ui.button(text, onClick)

  protected fun primaryButton(text: String, onClick: () -> Unit) = ctx.ui.primaryButton(text, onClick)

  protected fun outlineButton(text: String, onClick: () -> Unit) = ctx.ui.outlineButton(text, onClick)

  protected fun textButton(text: String, onClick: () -> Unit) = ctx.ui.textButton(text, onClick)

  protected fun filterChip(text: String, selected: Boolean, onToggle: (Boolean) -> Unit) = ctx.ui.filterChip(text, selected, onToggle)

  protected fun infoChip(text: String) = ctx.ui.infoChip(text)

  protected fun chipGroup(build: com.google.android.material.chip.ChipGroup.() -> Unit) = ctx.ui.chipGroup(build)

  protected fun statusBadge(text: String, accent: Int) = ctx.ui.statusBadge(text, accent)

  protected fun searchField(hint: String, value: String) = ctx.ui.searchField(hint, value)

  protected fun linearProgress(fraction: Double) = ctx.ui.linearProgress(fraction)

  protected fun card(build: LinearLayout.() -> Unit) = ctx.ui.card(build)

  protected fun tile(build: LinearLayout.() -> Unit) = ctx.ui.tile(build)

  protected fun row(build: LinearLayout.() -> Unit) = ctx.ui.row(build = build)

  protected fun confirm(message: String, positiveText: String = "Delete", onConfirm: () -> Unit) = ctx.ui.confirm(message, positiveText, onConfirm)

  protected fun showError(error: Throwable) = ctx.ui.showError(error)

  /**
   * A search field plus an Apply/Clear button row. Search stays explicit (not search-as-you-type) on purpose: the imperative full-rebuild render
   * would otherwise wipe the field's focus on every keystroke. [onChange] receives the new query (empty string for Clear); this helper re-renders.
   */
  protected fun searchBlock(hint: String, current: String, onChange: (String) -> Unit): View {
    val (field, edit) = searchField(hint, current)
    return LinearLayout(activity).apply {
      orientation = LinearLayout.VERTICAL
      addView(field)
      addView(
        row {
          addView(
            textButton("Apply") {
              onChange(edit.text.toString())
              render()
            }
          )
          if (current.isNotBlank())
            addView(
              textButton("Clear") {
                onChange("")
                render()
              }
            )
        }
      )
    }
  }

  /** Request a re-render of the current screen. */
  protected fun render() = vm.invalidate()

  /** Post to the main thread (download/poll callbacks fire off-main). */
  protected fun onMain(block: () -> Unit) = vm.onMain(block)

  protected fun runInBackground(startMessage: String, block: () -> String) =
    vm.runInBackground(startMessage, onError = { showError(it) }, block = block)

  protected fun setPocketMode(jobId: String?) {
    vm.pocketModeJobId = jobId
    render()
  }

  // --- Shared pure helpers used across more than one screen ---

  protected fun thermalStateLabel(): String = thermalLabel(thermalStatusProvider)

  protected fun thermalStateDescription(): String = thermalDescription(thermalStatusProvider)

  protected fun modelMatchesSearch(model: ModelFile, query: String): Boolean {
    val q = query.trim().lowercase()
    if (q.isBlank()) return true
    return listOfNotNull(model.name, model.displayName, model.hfRepo, model.familyId, model.quant).any { it.lowercase().contains(q) }
  }

  protected fun ensureMmprojSelectionInitialized(mmprojs: List<ModelFile>) {
    val availablePaths = mmprojs.map { it.path }.toSet()
    selectedMmprojPaths.retainAll(availablePaths)
    if (availablePaths.isEmpty()) {
      mmprojSelectionInitialized = false
      return
    }
    if (!mmprojSelectionInitialized) {
      selectedMmprojPaths += availablePaths
      mmprojSelectionInitialized = true
    }
  }

  abstract fun renderBody(body: LinearLayout)
}
