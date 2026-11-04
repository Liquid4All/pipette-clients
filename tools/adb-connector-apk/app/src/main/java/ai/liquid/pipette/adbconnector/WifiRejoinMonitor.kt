package ai.liquid.pipette.adbconnector

import android.content.Context
import android.net.ConnectivityManager
import android.net.LinkProperties
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.SystemClock
import android.util.Log
import java.net.Inet4Address
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Re-arms wireless debugging when wifi re-associates, so adbd announces its port again.
 *
 * adbd registers `_adb-tls-connect._tcp` when its TLS listener starts, so re-announcing the port is a 0 -> 1 flip of `adb_wifi_enabled`
 * ([AdbWifi.forceRearm]). A rejoin is a safe moment for it, because every wireless transport died with the link.
 *
 * Runs in [WifiDebuggingDialogService]'s process because a manifest receiver no longer gets the implicit wifi broadcasts (Android 8+) and a runtime
 * callback needs a process that stays alive. The request omits NET_CAPABILITY_INTERNET: a lab access point with no internet route never satisfies it.
 *
 * ## The first callback is not a rejoin
 *
 * Registration reports the network already connected, and a host transport can be live then (the service rebinds on every reinstall). So the first
 * link takes [AdbWifi.enable], which cannot move a live listener's port; only a different link earns the flip. With wifi down at registration there
 * is no initial callback, so [start] records whether a routable wifi network existed, and if none did the first link is a genuine rejoin.
 *
 * ## Why the edge is a routable address, not availability
 *
 * `onAvailable` can arrive before DHCP completes, and a re-arm then registers no useful A record.
 *
 * ## Anti-storm
 *
 * One association emits several `onLinkPropertiesChanged` calls, and a flip per callback would move the port several times. The debounce is on
 * identity (network handle plus IPv4 address); [REARM_FLOOR_MS] backs it up against an identity that keeps changing. A link the floor rejects is
 * re-checked when the floor elapses, because no later callback would carry it. The identity is one slot, so two concurrent wifi networks would flip
 * at the floor rate.
 *
 * The callback thread and a deferred re-check both reach the debounce state, so a lock guards the decision, never the re-arm itself.
 */
internal class WifiRejoinMonitor(private val context: Context, private val scope: CoroutineScope) {

  /** Set only once registration succeeds, so [stop] unregisters only what was registered. */
  private var connectivity: ConnectivityManager? = null

  private val debounce = Any()

  /** Guarded by [debounce]. */
  private var retryJob: Job? = null

  @Volatile private var sawNetworkAtRegistration = false

  /** Network handle and IPv4 address the last re-arm ran for, or null before the first one. */
  @Volatile private var lastRearmFor: Pair<Long, String>? = null

  /** Elapsed-realtime stamp of the last rejoin's re-arm, or 0 before the first one. */
  @Volatile private var lastRearmAt = 0L

  private val callback =
    object : ConnectivityManager.NetworkCallback() {
      override fun onLinkPropertiesChanged(network: Network, linkProperties: LinkProperties) {
        // Caught: a throwable here would take down the dialog accepter's process too.
        try {
          val address = linkProperties.routableIpv4() ?: return
          onRoutableAddress(network.networkHandle, address)
        } catch (e: Exception) {
          Log.e(AdbWifi.TAG, "wifi rejoin callback failed", e)
        }
      }
    }

  /** Registers the wifi callback. Safe to call when the platform refuses: the watchdog remains the recovery floor. */
  fun start() {
    val manager = context.getSystemService(ConnectivityManager::class.java)
    if (manager == null) {
      Log.e(AdbWifi.TAG, "no ConnectivityManager; a wifi rejoin will not re-arm wireless debugging")
      return
    }
    val request = NetworkRequest.Builder().addTransportType(NetworkCapabilities.TRANSPORT_WIFI).build()
    // Read before registering, so no callback sees the default. A network appearing in between
    // costs one unnecessary flip, better than a missed rejoin stranding a stale 1.
    sawNetworkAtRegistration = manager.hasRoutableWifi()
    // Throws on a missing ACCESS_NETWORK_STATE grant or too many outstanding requests; neither is
    // worth crashing the accepter over.
    runCatching { manager.registerNetworkCallback(request, callback) }
      .onSuccess {
        connectivity = manager
        Log.i(AdbWifi.TAG, "wifi rejoin monitor registered; a rejoin will re-arm wireless debugging")
      }
      .onFailure { Log.e(AdbWifi.TAG, "could not register the wifi rejoin monitor", it) }
  }

  fun stop() {
    val manager = connectivity ?: return
    connectivity = null
    synchronized(debounce) {
      retryJob?.cancel()
      retryJob = null
    }
    runCatching { manager.unregisterNetworkCallback(callback) }
      .onSuccess { Log.i(AdbWifi.TAG, "wifi rejoin monitor unregistered") }
      .onFailure { Log.w(AdbWifi.TAG, "could not unregister the wifi rejoin monitor", it) }
  }

  /**
   * Acts on a routable wifi link, unless the debounce rejects it.
   *
   * Records the identity and stamp before the work starts: the flip suspends for seconds while the callback thread re-enters, so recording them
   * afterwards would let every queued callback through.
   */
  private fun onRoutableAddress(handle: Long, address: String) {
    synchronized(debounce) {
      val identity = handle to address
      if (identity == lastRearmFor) {
        Log.i(AdbWifi.TAG, "wifi link net=$handle ip=$address already re-armed; skipping")
        return
      }
      val now = SystemClock.elapsedRealtime()
      val sinceLast = now - lastRearmAt
      // elapsedRealtime counts from boot, so an unset 0 stamp would put the first REARM_FLOOR_MS of
      // uptime under the floor.
      if (lastRearmAt != 0L && sinceLast < REARM_FLOOR_MS) {
        // Re-checked rather than skipped: no later callback would carry this link.
        val wait = REARM_FLOOR_MS - sinceLast
        Log.i(AdbWifi.TAG, "wifi link net=$handle ip=$address is ${sinceLast}ms after the last re-arm; below the floor, re-checking in ${wait}ms")
        scheduleRecheck(handle, address, wait)
        return
      }
      val rejoined = lastRearmFor != null || !sawNetworkAtRegistration
      lastRearmFor = identity
      if (rejoined) {
        // Only a flip leaves the device without a listener for seconds, so only it starts the floor.
        lastRearmAt = now
        Log.i(AdbWifi.TAG, "wifi rejoined on net=$handle ip=$address; re-arming in ${AdbWifi.HOLDER_GRACE_MS}ms to re-announce the port")
      } else {
        Log.i(AdbWifi.TAG, "wifi link net=$handle ip=$address present at registration; enabling without a flip")
      }

      scope.launch {
        try {
          val enabled = if (rejoined) AdbWifi.forceRearm(context) else AdbWifi.enable(context)
          Log.i(AdbWifi.TAG, "wireless debugging after the wifi callback: enabled=$enabled")
        } catch (e: Exception) {
          // Includes PermissionDenied before `pm grant`. Never rethrown: this scope belongs to the
          // dialog accepter, which must not die over a failed write.
          Log.e(AdbWifi.TAG, "rearm after a wifi callback failed", e)
        }
      }
    }
  }

  /** Replaces any pending re-check, so a burst of rejected callbacks re-checks only the newest link. The caller holds [debounce]. */
  private fun scheduleRecheck(handle: Long, address: String, delayMs: Long) {
    retryJob?.cancel()
    retryJob =
      scope.launch {
        delay(delayMs)
        // Cleared first, so a re-check that lands below the floor again does not cancel itself.
        synchronized(debounce) { retryJob = null }
        onRoutableAddress(handle, address)
      }
  }

  /**
   * Whether any wifi network already holds a routable IPv4 address.
   *
   * `allNetworks` is deprecated for a network callback, but only an immediate read answers what [start]'s registration will report on arrival.
   */
  @Suppress("DEPRECATION")
  private fun ConnectivityManager.hasRoutableWifi(): Boolean =
    allNetworks.any { network ->
      getNetworkCapabilities(network)?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true && getLinkProperties(network)?.routableIpv4() != null
    }

  /**
   * IPv4 because the host connects over it and adbd's A record carries it. Link-local is what the interface holds before DHCP answers, the state this
   * waits out.
   */
  private fun LinkProperties.routableIpv4(): String? =
    linkAddresses.map { it.address }.filterIsInstance<Inet4Address>().firstOrNull { !it.isLoopbackAddress && !it.isLinkLocalAddress }?.hostAddress

  private companion object {
    /**
     * Shortest gap between two re-arms. Longer than a re-arm's own window ([AdbWifi.HOLDER_GRACE_MS] plus the ~11s OEM freeze), so a callback during
     * a re-arm is deferred to a re-check rather than refused by [AdbWifi.forceRearm].
     */
    const val REARM_FLOOR_MS = 30_000L
  }
}
