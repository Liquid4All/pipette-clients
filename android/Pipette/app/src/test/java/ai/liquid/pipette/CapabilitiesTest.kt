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
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:876a43211"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `spells the runtime with an underscore to match the planner and the other clients`() {
    // `runtime:llamacpp` would pass server-side validation and match nothing. The CLI generates `runtime:llama_cpp`
    // (pipette_cli::client::worker::profile::runtime_capability_flags) and the plan rules require that spelling.
    assertTrue(
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = null).contains("runtime:llama_cpp")
    )
  }

  @Test
  fun `canonicalizes the build id to lowercase without whitespace`() {
    // A non-canonical flag is a 400 that fails the whole profile update, not just the one flag.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:876a43211-dirty"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876A43211 -Dirty", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `omits the versioned flag when the commit is a native sentinel`() {
    // "native-unavailable" / "native-pending" are placeholders NativeLib returns when the library or the bound engine isn't there yet.
    assertEquals(
      listOf("runtime:llama_cpp"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "native-pending", llamaCppTag = null, buildVersion = null),
    )
    assertEquals(
      listOf("runtime:llama_cpp"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "native-unavailable", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `omits the versioned flag when the commit is blank`() {
    assertEquals(
      listOf("runtime:llama_cpp"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `advertises the upstream tag beside the commit`() {
    // Advertising only the commit is what kept an Android phone out of every tag-pinned plan. See the `Capabilities` class doc.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:876a43211", "runtime:llama_cpp:b10216"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = "b10216", buildVersion = null),
    )
  }

  @Test
  fun `omits the tag level when the pin carries no tag`() {
    // An untagged pin is normal: a bump lands on a commit before upstream tags it.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:876a43211"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `omits the versioned flag when the build id is the unknown placeholder`() {
    // `engine::llama_cpp_commit` returns this literal when the vendored checkout carried no git metadata. It is not a build id: advertising it
    // publishes a flag matching no real job and churns the server's stored set, which costs a reindex and the client's queue standing.
    assertEquals(
      listOf("runtime:llama_cpp"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "unknown", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `still advertises the tag when the commit is unusable`() {
    // The two levels are independent. An unbound engine costs the commit level, not the tag, which is stamped at build time either way.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:b10216"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "native-pending", llamaCppTag = "b10216", buildVersion = null),
    )
  }

  @Test
  fun `reports one flag when the commit and the tag are the same string`() {
    // Nothing stops the two agreeing. A duplicate would be a second identical entry in the reported set, which the server stores as one and this
    // client would then diff against forever.
    assertEquals(
      listOf("runtime:llama_cpp", "runtime:llama_cpp:b10216"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "b10216", llamaCppTag = "b10216", buildVersion = null),
    )
  }

  @Test
  fun `advertises nothing when the native library is missing`() {
    // No engine means no runtime; claiming one would win jobs this build cannot run.
    assertEquals(
      emptyList<String>(),
      Capabilities.flags(nativeAvailable = false, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = null),
    )
  }

  @Test
  fun `advertises the build version of a released build`() {
    // The harness build reaches the planner from the profile, so a plan can pin the client that produces the measurement.
    assertEquals(
      listOf("client_version:2026.08.11-640d96e8", "runtime:llama_cpp", "runtime:llama_cpp:876a43211"),
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = "2026.08.11-640d96e8"),
    )
  }

  @Test
  fun `omits the client_version flag when the build names no release`() {
    // A local build and a debug build both name no release, and the flat flag has no general level to fall back to.
    assertTrue(
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = null).none {
        it.startsWith("client_version:")
      }
    )
  }

  @Test
  fun `canonicalizes the build version`() {
    // The value arrives from an environment variable, so nothing guarantees its form.
    assertTrue(
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = "2026.08.11 640D96E8")
        .contains("client_version:2026.08.11640d96e8")
    )
  }

  @Test
  fun `omits the client_version flag when the build version is blank`() {
    // Canonicalization strips whitespace, so a whitespace-only version leaves nothing to advertise.
    assertTrue(
      Capabilities.flags(nativeAvailable = true, llamaCppCommit = "876a43211", llamaCppTag = null, buildVersion = "   ").none {
        it.startsWith("client_version:")
      }
    )
  }
}
