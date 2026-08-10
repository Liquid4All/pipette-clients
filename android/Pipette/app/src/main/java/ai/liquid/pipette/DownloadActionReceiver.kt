package ai.liquid.pipette

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Routes the download notification's Pause / Resume / Cancel buttons to the app-scoped [DownloadCoordinator]. Runs in the main process. */
class DownloadActionReceiver : BroadcastReceiver() {
  override fun onReceive(context: Context, intent: Intent) {
    val key = intent.getStringExtra(EXTRA_KEY)
    val action = intent.action
    val coordinator = (context.applicationContext as? PipetteApp)?.containerOrNull?.downloadCoordinator
    if (key == null || action == null || coordinator == null) return
    // onReceive runs on the main thread, but pause/resume/cancel touch SharedPreferences and the filesystem; hop off-main via goAsync() to avoid
    // jank.
    val pending = goAsync()
    Thread {
        try {
          when (action) {
            ACTION_PAUSE -> coordinator.pause(key)
            ACTION_RESUME -> coordinator.resume(key)
            ACTION_CANCEL -> coordinator.cancel(key)
          }
        } finally {
          pending.finish()
        }
      }
      .start()
  }

  companion object {
    const val ACTION_PAUSE = "ai.liquid.pipette.action.PAUSE_DOWNLOAD"
    const val ACTION_RESUME = "ai.liquid.pipette.action.RESUME_DOWNLOAD"
    const val ACTION_CANCEL = "ai.liquid.pipette.action.CANCEL_DOWNLOAD"
    const val EXTRA_KEY = "download_key"
  }
}
