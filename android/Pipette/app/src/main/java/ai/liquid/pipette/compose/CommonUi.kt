package ai.liquid.pipette.compose

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.BenchmarkDefinition
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobQuantFilter
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.PresetModel
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// UI value types and the accent resolver shared by more than one screen. Per-screen state lives with its own ViewModel; only genuinely cross-screen
// rendering shapes (model rows, the results grid, pocket mode) live here. The AccentKind value type and its thermalAccentKind classifier live in the
// Compose-free UiFormatting so the :benchmark-process BenchmarkActivity can reuse them without loading Compose.

/** Maps a severity [AccentKind] to the themed Compose color. */
@Composable
fun accentColor(kind: AccentKind): Color {
  val colors = PipetteTheme.colors
  return when (kind) {
    AccentKind.NOMINAL -> colors.thermalNominal
    AccentKind.SERIOUS -> colors.orange
    AccentKind.CRITICAL -> colors.red
    AccentKind.MUTED -> colors.gray
  }
}

data class ModelRowUi(val model: ModelFile, val title: String, val subtitle: String)

/** A downloaded model family in the Models list: brand (from name) + size + collapsible quant chips. */
data class DownloadedGroupUi(
  val key: String,
  val name: String,
  val sizeLabel: String,
  val quantCount: Int,
  val quants: List<String>,
  val files: List<ModelFile>,
)

/** A selectable model family in the Add-models cover. */
data class AddModelGroupUi(val id: String, val name: String, val sizeLabel: String, val checked: Boolean)

data class TemplateGroupUi(val name: String, val rows: List<PresetRowUi>)

data class PresetRowUi(val preset: PresetModel, val label: String, val enabled: Boolean, val checked: Boolean)

data class JobCardUi(
  val manifest: JobManifest,
  val statusAccent: AccentKind,
  val runningHere: Boolean,
  val runProgress: Double,
  val countsLine: String,
  val rowPrimaryMeta: String,
  val rowSecondaryMeta: String,
  val firstFailure: String?,
  val canResume: Boolean,
  val completedCells: Int,
  val unsubmittedCount: Int,
  val isRegistered: Boolean,
)

data class ModelGroupRowUi(val key: String, val label: String, val name: String, val sizeLabel: String, val checked: Boolean)

data class QuantChipUi(val filter: JobQuantFilter, val label: String, val selected: Boolean)

data class BenchmarkGroupUi(
  val type: String,
  val displayName: String,
  val description: String,
  val disabled: Boolean,
  val allSelected: Boolean,
  val someSelected: Boolean,
  val toggleLabel: String,
  val items: List<BenchmarkItemUi>,
)

data class BenchmarkItemUi(val id: String, val label: String, val enabled: Boolean, val checked: Boolean, val definition: BenchmarkDefinition)

data class MmprojRowUi(val path: String, val label: String, val checked: Boolean)

data class ResultsGridUi(val columnLabels: List<String>, val rows: List<ResultsRowUi>)

data class ResultsRowUi(val label: String, val modelName: String, val quant: String, val cells: List<ResultsCellUi>)

/** Accent for a non-metric result cell: failed (red) / cancelled (orange), else none (iOS status colors). */
enum class ResultCellAccent {
  NONE,
  FAILED,
  CANCELLED,
}

data class ResultsCellUi(
  val text: String,
  val intensity: Double?,
  val brightText: Boolean,
  val cell: JobCell?,
  val hasDetail: Boolean,
  val accent: ResultCellAccent = ResultCellAccent.NONE,
)

data class CellUi(
  val cell: JobCell,
  val position: Int,
  val title: String,
  val modelName: String,
  val quant: String,
  val benchmarkLabel: String,
  val statusLabel: String,
  val statusAccent: AccentKind,
  val subtitle: String,
  val metricLine: String?,
  val errorLine: String?,
  val submissionLine: String?,
  val rerunSelectable: Boolean,
  val rerunSelected: Boolean,
  val expanded: Boolean,
  val canSubmit: Boolean,
  val submitting: Boolean,
  val detailRows: List<Pair<String, String>>,
  val detailModelPath: String,
  val detailMmprojPath: String?,
  val hasPayload: Boolean,
)

/** Colors for [JobLiveActivity] so the same block reads on both the themed running page and the dark Pocket card. */
data class JobActivityColors(val primaryText: Color, val secondaryText: Color, val accent: Color)

data class PocketUi(
  val jobId: String,
  val title: String,
  val subtitle: String,
  val progress: Double,
  val cellsDone: String,
  val timeLeft: String,
  val thermalLabel: String,
  val thermalAccent: AccentKind,
  val estTimeLine: String,
  val currentCellLabel: String = "",
  val progressText: String = "",
  // Non-null while the thermal-readiness gate is cooling: anchors the live cooldown timer + wash.
  val coolingSinceMillis: Long? = null,
)

/** Case-insensitive match of a model against a free-text query (shared by the Models + Jobs searches). */
fun ModelFile.matchesSearch(query: String): Boolean {
  val q = query.trim().lowercase()
  if (q.isBlank()) return true
  return listOfNotNull(name, displayName, hfRepo, familyId, quant).any { it.lowercase().contains(q) }
}
