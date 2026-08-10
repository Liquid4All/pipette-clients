package ai.liquid.pipette

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

/**
 * Re-reports the device profile and capability set to the management server at startup (`PATCH /clients/me`).
 *
 * The planner matches jobs against this profile, and its inputs drift between launches — an OS update, a monthly security patch, a new APK built from
 * a different llama.cpp commit — so the client re-reports on every start rather than only at registration. An unchanged resubmit is cheap: the server
 * voids the client's queue standing only when the matching input actually changed (`client-integration.md` §2).
 *
 * Unlike the CLI and iOS clients, this one does **not** wait out the `reindex_pending` gate. Those poll because they hold a lease and claim work, so
 * a voided standing means discarding in-flight jobs before continuing. This client has no planner loop — it never claims — so there is nothing in
 * flight to discard and nothing to wait for. The flag is logged and otherwise ignored.
 */
class ProfileRefreshService(private val context: Context, private val storage: LocalStorage, private val managementClient: ManagementClient) {
  /**
   * Re-report the profile, returning the server's updated record — or null when this device has no registration yet, in which case there is nothing
   * to refresh and no identity to sign with. Network and server failures propagate: the refresh is advisory, so callers log and carry on.
   */
  fun refresh(): ManagementClient.ClientProfile? {
    val registration = storage.loadRegistration() ?: return null
    val capabilities = Capabilities.current()
    val profile = managementClient.updateMe(registration.serverUrl, registration.clientId, profileUpdateBody(context, capabilities))
    Log.i(TAG, "device profile refreshed: status=${profile.status} reindexPending=${profile.reindexPending} capabilities=$capabilities")
    return profile
  }

  internal companion object {
    private const val TAG = "ProfileRefresh"

    /**
     * The `PATCH /clients/me` body, carrying only fields the server's schema accepts (`httpapi.md` §2.4.1).
     *
     * Deliberately absent:
     * - `device_os_build` / `device_os_security_patch` — collected by [DeviceInfo] but with no profile field to hold them; they ride on benchmark
     *   submissions instead (see [LocalStorage.writePayload]).
     * - the run-environment fields (battery level, power state, power-save mode) — they change minute to minute and belong on a result, not on a
     *   profile that gates queue standing.
     * - `device_gpu_*` / `device_npu_*` — omitted entirely rather than sent as null, since a `*_vram_bytes` without its matching `*_model` is a
     *   `400`.
     * - `client_details` — set at registration and never stale (it is the device model name), so re-sending it would only add wire noise.
     *
     * `device_os_name` is always paired with `device_os_version` because a version without a name is likewise a `400`.
     */
    internal fun profileUpdateBody(context: Context, capabilities: List<String>): JSONObject =
      JSONObject()
        .put("device_name", DeviceInfo.modelName())
        .put("device_form_factor", DeviceInfo.formFactor(context))
        .put("device_os_name", "Android")
        .put("device_os_version", DeviceInfo.osVersion())
        .put("device_chip_model", DeviceInfo.chipModel())
        .put("device_ram_bytes", DeviceInfo.ramBytes(context))
        // Set-granular: the value replaces the server's stored set wholesale, so this is always the full current set. An empty array is
        // meaningful — it clears the set for a build with no native library, which genuinely supports no runtime.
        .put("capabilities", JSONArray(capabilities))
  }
}
