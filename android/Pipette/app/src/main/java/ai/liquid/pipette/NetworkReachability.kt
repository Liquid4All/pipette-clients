package ai.liquid.pipette

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities

/**
 * Is there a network good enough to reach the management server right now?
 *
 * The Android counterpart of the iOS client's `NetworkReachability.shared.isConnected`, and it exists for one reason: gating the per-cell result
 * upload during a benchmark run. An offline device should skip the attempt rather than spend the uploader's retries, and the run's wall-clock, on a
 * request that cannot succeed. The results are already on disk; the end-of-run sweep or the next run sends them.
 *
 * Queried on demand rather than observed: a callback would have to be registered, unregistered and thread-confined for the life of a run, and the
 * only question ever asked is "right now?", immediately before a single upload.
 */
object NetworkReachability {
  /**
   * True when there is an active network claiming internet access and not currently blocked for this app.
   *
   * Deliberately **not** requiring [NetworkCapabilities.NET_CAPABILITY_VALIDATED]. Validation means Android's captive-portal probe reached *Google's*
   * generate_204 endpoint, which says nothing about whether this app's management server is reachable, and it is permanently false on a firewalled
   * corporate network or a LAN-only self-hosted server, precisely the deployments most likely to be running an on-prem server. Requiring it would
   * turn a wall-clock optimization into "auto-submit silently never happens here". The iOS twin uses `NWPath.status == .satisfied`, which is this,
   * not validation.
   *
   * Fails **open** when the service itself is unavailable, so a platform quirk can never be the reason results stop being submitted. The upload has
   * its own error handling; this is an optimization, not a correctness gate.
   */
  fun isOnline(context: Context): Boolean {
    val manager = context.getSystemService(ConnectivityManager::class.java) ?: return true
    val capabilities = manager.getNetworkCapabilities(manager.activeNetwork)
    return capabilities != null &&
      capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
      capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_SUSPENDED)
  }

  /** Bind [isOnline] to an application context, for injecting into [JobRunner] as a plain lambda so tests stay hermetic. */
  fun checker(context: Context): () -> Boolean {
    val appContext = context.applicationContext
    return { isOnline(appContext) }
  }
}
