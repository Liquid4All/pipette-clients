package ai.liquid.pipette

/**
 * The capability flags this client reports to the management server. The planner matches them against a job's `requires` set, comparing each flag as
 * a whole, opaque string — so every level we support is advertised: the general `runtime:llama_cpp`, plus a versioned `runtime:llama_cpp:<build>` for
 * BOTH the vendored commit and the upstream `bNNNN` tag on it. A client that reported only the versioned flag would match jobs pinned to that exact
 * build and nothing else, and one that reported only the commit would miss every plan pinned the way the rest of the fleet pins llama.cpp: by tag, as
 * in a desktop runtime's `repository_version = "b10216"`. That miss is invisible from the plan side, because an unmatched job simply never dispatches
 * here.
 *
 * llama.cpp is the only runtime this app has, and it is compiled into the APK as `libpipette_android.so` rather than downloaded, so this set is a
 * build-time property of the binary — there is no on-disk runtime inventory to enumerate (unlike the CLI's `installed_runtime_capabilities`).
 *
 * Flags must be canonical — lowercase, no whitespace — or the server rejects the whole request with `400`, and must avoid the server-owned reserved
 * namespaces (`os:`, `os_version:`, `device:`, `chip:`, `form_factor:`, `ram_bytes:`, `gpu:`, `gpu_vram_bytes:`, `npu:`, `npu_vram_bytes:`), which
 * the server derives from the `device_*` profile itself. This client reports two namespaces, `runtime:` and `client_version:`, and the server
 * reserves neither.
 *
 * See `pipette-mgmt` `docs/client-integration.md` §2 and `docs/httpapi.md` §2.4.1; the CLI counterpart is
 * `pipette_cli::client::worker::profile::runtime_capability_flags` and the iOS one is `PlannerWorker.capabilityFlags`.
 */
object Capabilities {
  /** Prefix of the placeholder values [NativeLib.llamaCppCommit] returns in place of a real build id. */
  private const val NATIVE_SENTINEL_PREFIX = "native-"

  /** The placeholder `engine::llama_cpp_commit` returns when the vendored checkout carried no git metadata. The iOS client drops the same literal. */
  private const val UNKNOWN_BUILD = "unknown"

  /**
   * [value] canonicalized for the wire — lowercase, no whitespace — or null when nothing is left to advertise.
   *
   * One helper for both the llama.cpp build id and the build version, so the two cannot drift. The iOS client keeps the same rule in
   * `Capabilities.canonicalBuild`.
   */
  private fun canonical(value: String?): String? = value?.lowercase()?.filterNot { it.isWhitespace() }?.ifEmpty { null }

  /**
   * Flags for a build whose native library availability is [nativeAvailable], whose vendored llama.cpp commit is [llamaCppCommit] and whose upstream
   * tag on that commit is [llamaCppTag] (null when the pin carries none). The build publishes as the release [buildVersion], or names no release when
   * [buildVersion] is null. Split from [current] so the branching is testable without a packaged `.so`.
   *
   * The result is sorted. The server voids the client's queue standing whenever the reported set differs from the stored one. A stable order
   * therefore keeps a relaunch from costing a reindex and a lease relinquish.
   */
  fun flags(nativeAvailable: Boolean, llamaCppCommit: String, llamaCppTag: String?, buildVersion: String?): List<String> {
    // No native library means no runtime at all, so advertise nothing rather than claim a capability we cannot honor. The client then matches no
    // runtime-gated job, which is the truthful outcome.
    if (!nativeAvailable) return emptyList()
    val flags = mutableListOf("runtime:llama_cpp")
    // Both levels, per the class doc: the commit is this build's identity, the tag is what the rest of the fleet pins by. By the containment rule
    // above, the extra level can only widen what this client matches.
    for (build in listOfNotNull(buildId(llamaCppCommit), buildId(llamaCppTag))) {
      flags += "runtime:llama_cpp:$build"
    }
    // The harness build, so a plan can pin the client that produces the measurement and an operator can read the fleet's build distribution off the
    // profile alone — before any job finishes. The flag carries the string this build submits as `client_version` whenever that string is already
    // canonical, which holds for every release version CI stamps. The value reaches the build through an environment variable, so canonicalize it
    // here: a non-canonical flag is a `400` on the whole profile update.
    canonical(buildVersion)?.let { flags += "client_version:$it" }
    return flags.distinct().sorted()
  }

  /**
   * [value] as a build id worth advertising, or null when it is not one.
   *
   * Three things are dropped: blank, the `native-*` placeholders [NativeLib.llamaCppCommit] returns when the engine is unbound, and [UNKNOWN_BUILD]
   * from a checkout with no git metadata. All three would publish a versioned flag that matches no real job and would churn the server's stored set,
   * and with it this client's queue standing, whenever the value changed.
   */
  private fun buildId(value: String?): String? = canonical(value)?.takeUnless { it.startsWith(NATIVE_SENTINEL_PREFIX) || it == UNKNOWN_BUILD }

  /**
   * Flags for this build, read from the packaged native library. Touching [NativeLib] here loads `libpipette_android.so` into the calling process
   * (the UI process at startup, not just the isolated `:benchmark` one) — the cost of reporting the versioned flag without waiting for a service
   * binding.
   */
  fun current(): List<String> =
    flags(
      NativeLib.isAvailable,
      NativeLib.llamaCppCommit(),
      NativeLib.llamaCppTag(),
      BuildConfig.BUILD_VERSION.takeIf { BuildConfig.HAS_RELEASE_VERSION },
    )
}
