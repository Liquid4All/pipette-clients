package ai.liquid.pipette.compose

import ai.liquid.pipette.compose.shell.ShellViewModel
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** One-off side effects a screen asks the host Activity to perform (SAF launchers, transient errors). */
sealed interface Effect {
  data object PickModel : Effect

  /**
   * The CSV text itself is held in JobsViewModel (see consumePendingCsvExport) to keep it out of the saved-state Bundle; this carries only the name.
   */
  data class ExportCsv(val filename: String) : Effect

  data class ShowError(val message: String) : Effect
}

/**
 * Base for the per-screen ViewModels. Each screen owns a single `state: StateFlow<…UiState>` and a single `onIntent(…)` entry point; this base just
 * provides the shared plumbing: the [effects] channel and a background-work runner whose errors surface as [Effect.ShowError] toasts. Screen VMs
 * depend on the [shell] hub for navigation/registration, never on each other.
 */
abstract class ScreenViewModel(app: Application, protected val shell: ShellViewModel) : AndroidViewModel(app) {
  // Buffered Channel (not a replay=0 SharedFlow): effects emitted while the collector is
  // detached — e.g. an activity-recreation gap — are queued and delivered when collection
  // resumes, instead of being silently dropped.
  private val _effects = Channel<Effect>(Channel.BUFFERED)
  val effects: Flow<Effect> = _effects.receiveAsFlow()

  protected fun emit(effect: Effect) {
    _effects.trySend(effect)
  }

  protected fun showError(message: String) = emit(Effect.ShowError(message))

  /**
   * Run [block] off the main thread; surface any failure as an error effect (Toast). Success is silent — like iOS, transient success feedback lives
   * per-screen (in-button spinners, refreshed lists), not in a shared status bar.
   */
  protected fun runInBackground(block: () -> Unit) {
    viewModelScope.launch { runCatching { withContext(Dispatchers.IO) { block() } }.onFailure { showError(it.message ?: it.javaClass.simpleName) } }
  }

  /** Surface an error from an off-main callback. [showError]'s channel send is thread-safe, so no dispatch hop is needed. */
  protected fun postError(error: Throwable) {
    showError(error.message ?: error.javaClass.simpleName)
  }
}
