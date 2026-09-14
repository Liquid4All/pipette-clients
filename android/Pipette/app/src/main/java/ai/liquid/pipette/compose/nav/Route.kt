package ai.liquid.pipette.compose.nav

import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

/**
 * Navigation 3 destinations for the signed-in + registered app (i.e. inside the auth / setup gates, which remain plain state-driven overlays in
 * [ai.liquid.pipette.compose.PipetteAppRoot]).
 *
 * Keys are `@Serializable` so `rememberNavBackStack` can persist and restore the back stack across process death.
 */
@Serializable
sealed interface Route : NavKey {
  /** Top-level tab destinations surfaced in the floating pill bar. */
  @Serializable sealed interface TopLevel : Route

  @Serializable data object Jobs : TopLevel

  @Serializable data object Models : TopLevel

  @Serializable data object Settings : TopLevel

  /** Pushed detail: Settings → open-source licenses. */
  @Serializable data object Acknowledgements : Route

  /** Pushed full-screen cover: Models → Add models. Visibility mirrors the Models VM's addModelsOpen. */
  @Serializable data object AddModels : Route

  /** Pushed full-screen cover: Jobs → Create a job (the new-job wizard). Mirrors JobsUiState.Wizard. */
  @Serializable data object CreateJob : Route

  /** Pushed detail: Jobs → per-job detail (results + progress). Mirrors JobsUiState.Detail. */
  @Serializable data object JobDetail : Route

  /** Pushed full-screen cover: Jobs → cell detail (opened from the results grid). Mirrors JobsUiState.CellDetail. */
  @Serializable data object CellDetail : Route
}
