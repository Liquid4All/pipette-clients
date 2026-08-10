package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the capability flags this client advertises. The failure mode these guard against is silent: a malformed or wrongly-spelled flag is still
 * valid canonical input the server accepts, so the client keeps matching zero jobs with no error anywhere.
 */
class CapabilitiesTest {
  @Test
  fun `advertises both the general and versioned llama_cpp flags`() {
    assertEquals(listOf("runtime:llama_cpp", "runtime:llama_cpp:b8683"), Capabilities.flags(nativeAvailable = true, llamaCppCommit = "b8683"))
  }

  @Test
  fun `spells the runtime with an underscore to match the planner and the other clients`() {
    // `runtime:llamacpp` would pass server-side validation and match nothing. The CLI generates `runtime:llama_cpp`
    // (pipette_cli::client::worker::profile::runtime_capability_flags) and the plan rules require that spelling.
    assertTrue(Capabilities.flags(nativeAvailable = true, llamaCppCommit = "b8683").contains("runtime:llama_cpp"))
  }

  @Test
  fun `canonicalizes the build id to lowercase without whitespace`() {
    // A non-canonical flag is a 400 that fails the whole profile update, not just the one flag.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:b8683-dirty"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "B8683 -Dirty"),
    )
  }

  @Test
  fun `omits the versioned flag when the commit is a native sentinel`() {
    // "native-unavailable" / "native-pending" are placeholders NativeLib returns when the library or the bound engine isn't there yet.
    assertEquals(listOf("runtime:llama_cpp"), Capabilities.flags(nativeAvailable = true, llamaCppCommit = "native-pending"))
    assertEquals(listOf("runtime:llama_cpp"), Capabilities.flags(nativeAvailable = true, llamaCppCommit = "native-unavailable"))
  }

  @Test
  fun `omits the versioned flag when the commit is blank`() {
    assertEquals(listOf("runtime:llama_cpp"), Capabilities.flags(nativeAvailable = true, llamaCppCommit = ""))
  }

  @Test
  fun `advertises nothing when the native library is missing`() {
    // No engine means no runtime; claiming one would win jobs this build cannot run.
    assertEquals(emptyList<String>(), Capabilities.flags(nativeAvailable = false, llamaCppCommit = "b8683"))
  }
}
