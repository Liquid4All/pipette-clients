package ai.liquid.pipette

import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowBuild

/**
 * Pins the `POST /clients/register` body against the server's schema (`pipette-mgmt` `docs/httpapi.md`). Robolectric supplies [org.json.JSONObject].
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class RegisterBodyTest {
  private fun body(preauthKey: String? = null) =
    registerBody(
      "Example",
      "user@example.com",
      "Pixel 9",
      "a3f8",
      preauthKey,
      ProfileRefreshService.profileUpdateBody(ApplicationProvider.getApplicationContext(), listOf("runtime:llama_cpp")),
    )

  /** The server derives `os:` and the other device flags from these fields, and the catalog offers models by them from the first fetch. */
  @Test
  fun `carries the device profile and the capabilities`() {
    val json = body()
    assertEquals(DeviceInfo.OS_NAME, json.getString("device_os_name"))
    assertTrue(json.has("device_os_version"))
    assertTrue(json.has("device_ram_bytes"))
    assertEquals("runtime:llama_cpp", json.getJSONArray("capabilities").getString(0))
  }

  @Test
  fun `carries the arch and os family`() {
    ShadowBuild.setSupportedAbis(arrayOf("arm64-v8a"))
    val json = body()
    assertEquals("arm64-v8a", json.getString("device_cpu_arch"))
    assertEquals("Android", json.getString("device_os_family"))
  }

  @Test
  fun `omits the cpu arch when the platform lists no ABI`() {
    ShadowBuild.setSupportedAbis(emptyArray())
    assertFalse("device_cpu_arch must be absent, not null", body().has("device_cpu_arch"))
  }

  @Test
  fun `omits a blank preauth key`() {
    assertFalse(body(preauthKey = " ").has("client_preauth_key"))
    assertEquals("client_preauth_k.s", body(preauthKey = "client_preauth_k.s").getString("client_preauth_key"))
  }
}
