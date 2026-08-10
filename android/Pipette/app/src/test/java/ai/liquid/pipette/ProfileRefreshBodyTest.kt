package ai.liquid.pipette

import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Pins the `PATCH /clients/me` body against the server's schema (`pipette-mgmt` `docs/httpapi.md` §2.4.1). Robolectric supplies the real Android
 * [android.content.Context] the [DeviceInfo] form-factor and RAM probes need.
 *
 * Every assertion here is about a `400` the server would return, or a field that has no profile column to land in.
 */
// A stock Application, not the manifest PipetteApp: the body builder needs only a Context, while PipetteApp.onCreate wires up WorkManager/Clerk/etc.
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class ProfileRefreshBodyTest {
  private fun body(capabilities: List<String> = listOf("runtime:llama_cpp")) =
    ProfileRefreshService.profileUpdateBody(ApplicationProvider.getApplicationContext(), capabilities)

  @Test
  fun `carries the device profile fields the server accepts`() {
    val json = body()
    assertEquals("Android", json.getString("device_os_name"))
    assertTrue(json.getString("device_name").isNotBlank())
    assertTrue(json.getString("device_os_version").isNotBlank())
    assertTrue(json.getString("device_chip_model").isNotBlank())
    // Presence and type only: Robolectric's shadowed ActivityManager leaves MemoryInfo.totalMem at 0, so the value is 0 here and real on a device.
    assertTrue(json.has("device_ram_bytes"))
    assertTrue(json.getLong("device_ram_bytes") >= 0)
  }

  @Test
  fun `pairs os_version with os_name`() {
    // `device_os_version` present without `device_os_name` is a 400.
    val json = body()
    assertTrue(json.has("device_os_name"))
    assertTrue(json.has("device_os_version"))
  }

  @Test
  fun `reports a form factor from the server's enum`() {
    // Anything outside this set is a 400.
    assertTrue(body().getString("device_form_factor") in setOf("phone", "tablet", "laptop", "desktop", "server", "embedded"))
  }

  @Test
  fun `omits the gpu and npu fields entirely rather than sending nulls`() {
    // A `*_vram_bytes` without its matching `*_model` is a 400, and a null leaves the stored value unchanged anyway — so absent, not null.
    val json = body()
    listOf("device_gpu_model", "device_gpu_vram_bytes", "device_npu_model", "device_npu_vram_bytes").forEach {
      assertFalse("$it must not be sent", json.has(it))
    }
  }

  @Test
  fun `omits fields the profile schema has no column for`() {
    // Collected by DeviceInfo and sent on benchmark submissions, but not part of the client profile. The run-environment values additionally change
    // minute to minute, and a capabilities/profile change voids the client's queue standing.
    val json = body()
    listOf("device_os_build", "device_os_security_patch", "device_battery_level", "device_power_state", "device_power_save_mode").forEach {
      assertFalse("$it is not a profile field", json.has(it))
    }
  }

  @Test
  fun `sends capabilities as the full replacement set`() {
    // Set-granular on the server: the value replaces the stored set wholesale.
    val json = body(listOf("runtime:llama_cpp", "runtime:llama_cpp:b8683"))
    val flags = json.getJSONArray("capabilities")
    assertEquals(2, flags.length())
    assertEquals("runtime:llama_cpp", flags.getString(0))
    assertEquals("runtime:llama_cpp:b8683", flags.getString(1))
  }

  @Test
  fun `sends an empty capabilities array rather than omitting it`() {
    // Absent would leave a stale set on the server; an empty array truthfully clears it for a build with no native library.
    val json = body(emptyList())
    assertTrue(json.has("capabilities"))
    assertEquals(0, json.getJSONArray("capabilities").length())
  }
}
