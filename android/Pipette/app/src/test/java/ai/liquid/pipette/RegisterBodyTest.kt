package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
  private fun body(preauthKey: String? = null) = registerBody("Example", "user@example.com", "Pixel 9", "a3f8", preauthKey)

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
