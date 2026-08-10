package ai.liquid.pipette

/**
 * The capability flags this client reports to the management server. The planner matches them against a job's `requires` set, comparing each flag as
 * a whole, opaque string — so every level we support is advertised: the general `runtime:llama_cpp` plus the versioned `runtime:llama_cpp:<build>`. A
 * client that reported only the versioned flag would match jobs pinned to that exact build and nothing else.
 *
 * llama.cpp is the only runtime this app has, and it is compiled into the APK as `libpipette_android.so` rather than downloaded, so this set is a
 * build-time property of the binary — there is no on-disk runtime inventory to enumerate (unlike the CLI's `installed_runtime_capabilities`).
 *
 * Flags must be canonical — lowercase, no whitespace — or the server rejects the whole request with `400`, and must avoid the server-owned reserved
 * namespaces (`os:`, `os_version:`, `device:`, `chip:`, `form_factor:`, `ram_bytes:`, `gpu:`, `gpu_vram_bytes:`, `npu:`, `npu_vram_bytes:`), which
 * the server derives from the `device_*` profile itself. Reporting `runtime:` flags only keeps us clear of both rules.
 *
 * See `pipette-mgmt` `docs/client-integration.md` §2 and `docs/httpapi.md` §2.4.1; the CLI counterpart is
 * `pipette_cli::client::worker::profile::runtime_capability_flags` and the iOS one is `PlannerWorker.capabilityFlags`.
 */
object Capabilities {
  /** Prefix of the placeholder values [NativeLib.llamaCppCommit] returns in place of a real build id. */
  private const val NATIVE_SENTINEL_PREFIX = "native-"

  /**
   * Flags for a build whose native library availability is [nativeAvailable] and whose llama.cpp build id is [llamaCppCommit]. Split from [current]
   * so the branching is testable without a packaged `.so`.
   */
  fun flags(nativeAvailable: Boolean, llamaCppCommit: String): List<String> {
    // No native library means no runtime at all, so advertise nothing rather than claim a capability we cannot honor. The client then matches no
    // runtime-gated job, which is the truthful outcome.
    if (!nativeAvailable) return emptyList()
    val flags = mutableListOf("runtime:llama_cpp")
    val build = llamaCppCommit.lowercase().filterNot { it.isWhitespace() }
    // "native-unavailable" / "native-pending" are placeholders, not build ids. Advertising one would publish a versioned flag that matches no
    // real job, and would churn the server's stored set every time the engine happened to be unbound at launch.
    if (build.isNotEmpty() && !build.startsWith(NATIVE_SENTINEL_PREFIX)) {
      flags += "runtime:llama_cpp:$build"
    }
    return flags
  }

  /**
   * Flags for this build, read from the packaged native library. Touching [NativeLib] here loads `libpipette_android.so` into the calling process
   * (the UI process at startup, not just the isolated `:benchmark` one) — the cost of reporting the versioned flag without waiting for a service
   * binding.
   */
  fun current(): List<String> = flags(NativeLib.isAvailable, NativeLib.llamaCppCommit())
}
