package ai.liquid.pipette

import ai.liquid.pipette.compose.PipetteAppRoot
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen

/**
 * Compose entry point for the Android client; the launcher activity (see AndroidManifest). The whole UI is built from the per-screen MVI ViewModels
 * and design-system composables under [ai.liquid.pipette.compose]; [PipetteAppRoot] hosts the shell + the four screens.
 *
 * State and background work live in the app-scoped AppContainer (reached through the ViewModels), so a config-change recreation here leaves a running
 * benchmark and its :benchmark process untouched — same contract as the legacy view-based [MainActivity].
 */
class ComposeMainActivity : ComponentActivity() {
  // Lets the download foreground notification (with pause/cancel) show on Android 13+. Result
  // ignored — downloads still run if the user declines; the notification just won't appear.
  private val notificationPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

  override fun onCreate(savedInstanceState: Bundle?) {
    // Must run before super.onCreate so the system swaps the launch theme for the AndroidX
    // splash screen (Theme.Pipette.Starting -> Theme.Pipette via postSplashScreenTheme).
    installSplashScreen()
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    maybeRequestNotificationPermission()
    setContent { PipetteAppRoot() }
  }

  private fun maybeRequestNotificationPermission() {
    if (
      Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
    ) {
      notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
  }
}
