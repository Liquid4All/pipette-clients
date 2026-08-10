package ai.liquid.pipette

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first

private val Context.pipetteSettingsDataStore by preferencesDataStore(name = "pipette_settings")

data class SetupSettings(val serverUrl: String = AppSettingsStore.DEFAULT_SERVER_URL, val organization: String = "", val contactEmail: String = "")

class AppSettingsStore(context: Context) {
  private val appContext = context.applicationContext

  // ── Suspend API (preferred) ──────────────────────────────────────────
  // DataStore's `data.first()` and `edit { }` are already main-safe
  // suspend functions (they hop to Dispatchers.IO internally), so these
  // need no explicit dispatcher. ViewModels (Phase B) call these directly.

  suspend fun readSetupSettings(): SetupSettings {
    val prefs = appContext.pipetteSettingsDataStore.data.first()
    return SetupSettings(
      serverUrl = prefs[SERVER_URL] ?: DEFAULT_SERVER_URL,
      organization = prefs[ORGANIZATION] ?: "",
      contactEmail = prefs[CONTACT_EMAIL] ?: "",
    )
  }

  suspend fun writeSetupSettings(settings: SetupSettings) {
    appContext.pipetteSettingsDataStore.edit { prefs ->
      prefs[SERVER_URL] = settings.serverUrl
      prefs[ORGANIZATION] = settings.organization
      prefs[CONTACT_EMAIL] = settings.contactEmail
    }
  }

  suspend fun readDefaultContributeResults(): Boolean {
    val prefs = appContext.pipetteSettingsDataStore.data.first()
    return prefs[DEFAULT_CONTRIBUTE_RESULTS] ?: false
  }

  suspend fun writeDefaultContributeResults(enabled: Boolean) {
    appContext.pipetteSettingsDataStore.edit { prefs -> prefs[DEFAULT_CONTRIBUTE_RESULTS] = enabled }
  }

  /**
   * Debug-only Clerk-gate bypass. Defaults to false (auth enforced) so debug behaves like release; a developer can opt out via the Settings toggle to
   * run benchmarks un-gated. The gate is always enforced in release regardless of this value (see MainViewModel). Only read/written in debug builds.
   */
  suspend fun readClerkGateBypass(): Boolean {
    val prefs = appContext.pipetteSettingsDataStore.data.first()
    return prefs[CLERK_GATE_BYPASS] ?: false
  }

  suspend fun writeClerkGateBypass(enabled: Boolean) {
    appContext.pipetteSettingsDataStore.edit { prefs -> prefs[CLERK_GATE_BYPASS] = enabled }
  }

  /**
   * Debug-only thermal-gate waiver (PIP-434). Defaults to false (the gate is enforced), so a build behaves the same whether or not the toggle exists.
   * A waived run is recorded as such in `benchmark_flags.readiness.skip_thermal`, so its results stay distinguishable from gated ones rather than
   * silently polluting the warehouse.
   */
  suspend fun readSkipThermalGate(): Boolean {
    val prefs = appContext.pipetteSettingsDataStore.data.first()
    return prefs[SKIP_THERMAL_GATE] ?: false
  }

  suspend fun writeSkipThermalGate(enabled: Boolean) {
    appContext.pipetteSettingsDataStore.edit { prefs -> prefs[SKIP_THERMAL_GATE] = enabled }
  }

  // The transitional `runBlocking` bridges were removed in Phase B: the UI is
  // coroutine-aware now and reads/writes these settings through MainViewModel
  // (cached in memory, persisted on viewModelScope), so nothing touches
  // DataStore on the main thread.

  companion object {
    // Production collector, matching iOS (CollectorEndpoint.productionURL). Previously defaulted to
    // the awsdev backend, which shipped in release builds too — registrations went to the wrong host.
    const val DEFAULT_SERVER_URL = "https://collector.pipette.liquid.ai"

    private val SERVER_URL = stringPreferencesKey("server_url")
    private val ORGANIZATION = stringPreferencesKey("organization")
    private val CONTACT_EMAIL = stringPreferencesKey("contact_email")
    private val DEFAULT_CONTRIBUTE_RESULTS = booleanPreferencesKey("default_contribute_results")
    private val CLERK_GATE_BYPASS = booleanPreferencesKey("clerk_gate_bypass")
    private val SKIP_THERMAL_GATE = booleanPreferencesKey("skip_thermal_gate")
  }
}
