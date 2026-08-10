package ai.liquid.pipette.compose

import ai.liquid.pipette.compose.jobs.JobsViewModel
import ai.liquid.pipette.compose.models.ModelsViewModel
import ai.liquid.pipette.compose.settings.SettingsViewModel
import ai.liquid.pipette.compose.setup.SetupViewModel
import ai.liquid.pipette.compose.shell.ShellViewModel
import android.app.Application
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

/** Builds the per-screen ViewModels, injecting the app + the shared [ShellViewModel] hub. */
class PipetteViewModelFactory(private val app: Application, private val shell: ShellViewModel) : ViewModelProvider.Factory {
  @Suppress("UNCHECKED_CAST")
  override fun <T : ViewModel> create(modelClass: Class<T>): T =
    when {
      modelClass.isAssignableFrom(SetupViewModel::class.java) -> SetupViewModel(app, shell)
      modelClass.isAssignableFrom(ModelsViewModel::class.java) -> ModelsViewModel(app, shell)
      modelClass.isAssignableFrom(JobsViewModel::class.java) -> JobsViewModel(app, shell)
      modelClass.isAssignableFrom(SettingsViewModel::class.java) -> SettingsViewModel(app, shell)
      else -> throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
    }
      as T
}
