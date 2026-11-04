import Foundation

// What this client advertises to the planner (capability flags from the engines compiled
// into this build, and the device profile) and the reindex wait that follows a profile
// change: `pipette-cli`'s `client/worker/profile.rs`. The server unions the `device_*`
// fields, as reserved-namespace flags, with `capabilities` into the effective set jobs
// match against (the mgmt repo's `docs/plan-ingestion.md` §5).
//
// Only the planner worker reports: whatever claims as this client owns its profile, and a
// report from elsewhere would replace the claimer's (`pipette-ios-proxy` claims as a farm
// phone). Resubmitting an unchanged profile is cheap: the server voids queue standing only
// when a matching input changed (the mgmt repo's `docs/client-integration.md` §2).

/// The crate's `REINDEX_POLL_INTERVAL`.
nonisolated let reindexPollInterval: Duration = .seconds(5)
/// The crate's `REINDEX_WAIT_BUDGET`: a local budget only, as the gate is server-side.
/// Measured from the first poll, so a slow `GET` spends it too.
nonisolated let reindexWaitBudget: Duration = .seconds(5 * 60)

/// The crate's `ReindexWait`. Not a sleep closure: the budget is compared against the clock
/// inside the loop, so a no-op sleep would still wait out the full five minutes.
nonisolated struct ReindexWait: Sendable, Equatable {
    var pollInterval: Duration = reindexPollInterval
    var budget: Duration = reindexWaitBudget

    static let `default` = ReindexWait()
}

/// A llama.cpp build's commit when the vendored checkout carried no git metadata:
/// `ios/build-llama.sh` substitutes this literal rather than failing the build.
private nonisolated let unknownBuild = "unknown"

/// The `client_version:` flag, so a plan can pin the build that produces the measurement;
/// nil when `buildVersion` names no release (a local or TestFlight build).
///
/// It carries the flavor suffix, as a submission's `client_version` does: an internal build
/// reports `<release>-internal` and matches no bare `<release>` requirement, rightly, since
/// its readiness gate differs. The version is canonicalized before the suffix joins it, so
/// a blank one earns no flag.
nonisolated func clientVersionCapabilityFlag(_ buildVersion: String?) -> String? {
    buildVersion.flatMap(canonicalBuild).map { "client_version:\(Bundle.releaseVersion($0))" }
}

/// The runtime flags this build reports as `capabilities`.
///
/// Matching compares whole strings by set containment, so every level is advertised: the
/// general `runtime:mlx` and each versioned `runtime:mlx:<pin>`. A versioned flag does not
/// imply the general one, and an extra flag can only widen what this client matches.
///
/// A non-canonical flag (uppercase, whitespace) fails the whole update with a `400`, as
/// does one in a server-reserved namespace (the mgmt repo's `docs/httpapi.md` §2.2.1).
///
/// No `job_schema:<n>` flag, matching the Rust client: no plan emits one yet (the mgmt
/// repo's `docs/plan-ingestion.md` §7). Add it on both sides when one does.
///
/// Sorted so the set is byte-stable: the server voids queue standing whenever it changes,
/// so a reordered set would cost a reindex and a lease relinquish on every launch.
///
/// llama.cpp is main's build alone: the in-app worker runs every llama.cpp claim on it.
///
/// The build ids are parameters only so tests can reach the canonicalization and
/// placeholder branches. llama.cpp arrives as one `LlamaPin.Build`, not a commit/tag pair,
/// so a test cannot construct a combination `LlamaPins` never builds.
nonisolated func installedRuntimeCapabilities(
    llamaCpp: LlamaPin.Build = LlamaPins.main.build,
    mlxSwiftVersion: String = MLXBuildInfo.mlxSwift.submissionVersion,
    mlxSwiftLMVersion: String = MLXBuildInfo.mlxSwiftLM.submissionVersion,
    swiftTransformersVersion: String = MLXBuildInfo.swiftTransformers.submissionVersion,
    onnxGenaiVersion: String = OnnxGenaiBuildInfo.onnxruntimeGenai.submissionVersion,
    onnxRuntimeVersion: String = OnnxGenaiBuildInfo.onnxruntime.submissionVersion
) -> [String] {

    var flags: Set<String> = [
        "runtime:llama_cpp",
        "runtime:mlx",
        // Ships with the OS and exposes no build id, so no versioned level (as the
        // crate's `runtime_capability_flags`).
        "runtime:apple_foundation",
        "runtime:onnx_genai",
    ]
    // Commit and tag both: desktop runtimes are pinned by tag (`b10216`), so one plan can
    // pin iOS the way it pins the rest of the fleet.
    [llamaCpp.commit, llamaCpp.tag]
        .compactMap { $0 }
        .compactMap(canonicalBuild)
        .forEach { flags.insert("runtime:llama_cpp:\($0)") }
    // All three MLX packages affect output (swift-transformers changes tokenization), so
    // each gets its own flag; requiring several is equivalent to one composite string.
    //
    // `tools/pipette-ios-proxy/src/pool/capabilities.rs` replicates these four levels, and
    // its tests mirror `CapabilitiesTests`: a level added here and not there makes a
    // proxied phone stop matching a plan that pins it.
    if let build = canonicalBuild(mlxSwiftVersion) {
        // Bare form: what a plan generated from a runtime spec carries.
        flags.insert("runtime:mlx:\(build)")
        flags.insert("runtime:mlx:mlx-swift=\(build)")
    }
    if let build = canonicalBuild(mlxSwiftLMVersion) {
        flags.insert("runtime:mlx:mlx-swift-lm=\(build)")
    }
    if let build = canonicalBuild(swiftTransformersVersion) {
        flags.insert("runtime:mlx:swift-transformers=\(build)")
    }
    // Both ONNX source trees affect output (the runtime executes the graph, genai drives
    // it), so each gets a flag as MLX's packages do. The bare form is what the crate's
    // `runtime_capability_flags` derives from the onnxruntime-genai ref.
    if let build = canonicalBuild(onnxGenaiVersion) {
        flags.insert("runtime:onnx_genai:\(build)")
        flags.insert("runtime:onnx_genai:onnxruntime-genai=\(build)")
    }
    if let build = canonicalBuild(onnxRuntimeVersion) {
        flags.insert("runtime:onnx_genai:onnxruntime=\(build)")
    }
    return flags.sorted()
}

/// The runtimes the in-app worker runs, as their descriptors name them: the active llama.cpp pin,
/// which runs every llama.cpp claim, then MLX and ONNX GenAI.
nonisolated func workerRuntimes() -> [Runtime] {
    [Runtime.carried(LlamaPins.current)]
        + [RuntimeType.mlxIosPipette, .onnxGenaiIosPipette].compactMap { Runtime.thisBuild(for: $0) }
}

/// One build flag (`runtimeBuildFlag`) per runtime in `runtimes`, sorted. The worker sends
/// them with `installedRuntimeCapabilities` in one profile update.
///
/// A descriptor that does not encode throws. The worker then stops, because a profile
/// without the flag would withhold every catalog offer of that build.
nonisolated func installedRuntimeBuildFlags(runtimes: [Runtime] = workerRuntimes()) throws -> [String] {
    try runtimes.compactMap(runtimeBuildFlag).sorted()
}

/// `build` canonicalized for the wire, or nil for blank and the `unknown` placeholder: a
/// flag from a non-build-id matches no job and would churn the client's queue standing.
/// Checked through `CapabilityFlag` so no segment can earn the whole update a `400`.
private nonisolated func canonicalBuild(_ build: String) -> String? {
    let canonical = build.lowercased().filter { !$0.isWhitespace }
    guard canonical != unknownBuild, let flag = try? CapabilityFlag(canonical) else { return nil }
    return flag.value
}

/// The `PATCH /clients/me` body, over a device detected under the operator's labels. No
/// `client_details`, as the CLI sends none: the description is the operator's, set at
/// registration, and a refresh would overwrite it with the probed model name.
nonisolated func buildProfileUpdate(deviceStore: ClientDeviceStore,
                                    capabilities: [String]) throws -> UpdateClientRequest {
    let labels = try deviceStore.get()
    let device = detectDeviceInfo(deviceName: labels.deviceName,
                                  deviceFormFactor: labels.deviceFormFactor)
    return profileUpdate(device: device, capabilities: capabilities)
}

/// The body over a caller-supplied snapshot.
///
/// Deliberately absent: `device_gpu_*` / `device_npu_*`, as iOS has no separately
/// addressable VRAM on a unified-memory device; and `device_os_build` and the battery and
/// power state, which ride on submissions, describing one measurement rather than gating
/// queue standing.
nonisolated func profileUpdate(device: DeviceInfo, capabilities: [String]) -> UpdateClientRequest {
    let hardware = hardwareCapabilities(device: device)
    AppLog.profile.info("hardware capability flags: [\(hardware.joined(separator: ", "))]")
    var flags = capabilities + hardware
    if let clientVersion = clientVersionCapabilityFlag(Bundle.main.buildVersion) {
        AppLog.profile.info("client build capability flag: \(clientVersion)")
        flags.append(clientVersion)
    } else {
        AppLog.profile.info("build version names no release; no client_version capability flag")
    }
    return UpdateClientRequest(
        clientDetails: nil,
        device: DeviceProfileFields(
            deviceName: device.deviceName,
            deviceFormFactor: device.deviceFormFactor,
            // Always sent as a pair: `device_os_version` without `device_os_name` is a `400`.
            deviceOsName: device.deviceOsName,
            deviceOsVersion: device.deviceOsVersion,
            deviceCpuArch: device.deviceCpuArch,
            deviceOsFamily: device.deviceOsFamily,
            deviceChipModel: device.deviceChipModel,
            deviceRamBytes: device.deviceRamBytes,
            // A present value replaces the server's stored set wholesale.
            capabilities: Array(Set(flags)).sorted()
        )
    )
}

/// Report the profile, then hold until `reindex_pending` clears or the budget runs out:
/// the crate's `refresh_profile_at_startup`. Nil when there is no registration. A claim
/// before the gate lifts is answered `204` and idles the worker for a whole idle wait.
nonisolated func refreshProfileAtStartup(
    registration: ClientRegistrationStore,
    deviceStore: ClientDeviceStore,
    http: HttpClient = .shared,
    capabilities: [String],
    wait: ReindexWait = .default
) async throws -> ClientProfile? {
    guard let auth = registration.signingIdentity(),
          let server = registration.get()?.serverUrl
    else { return nil }
    let request = try buildProfileUpdate(deviceStore: deviceStore, capabilities: capabilities)
    let buildFlagCount = request.device.capabilities?
        .filter { $0.hasPrefix(runtimeBuildFlagPrefix) }.count ?? 0
    AppLog.profile.info(
        "refreshing device profile (\(request.device.capabilities?.count ?? 0) "
            + "capability flag(s), \(buildFlagCount) runtime build flag(s), "
            + "cpu_arch=\(request.device.deviceCpuArch ?? "unset"), "
            + "os_family=\(request.device.deviceOsFamily ?? "unset"))"
    )
    let profile = try await PipetteMgmtClient.ClientApi.updateMe(
        server: server, http: http, auth: auth, request: request)
    AppLog.profile.info(
        "profile ready (status=\(profile.status), reindex_pending=\(profile.reindexPending), "
            + "tags=\(profile.tags), caps=\(profile.capabilities))"
    )
    guard profile.reindexPending else { return profile }
    AppLog.profile.info(
        "reindex_pending=true after profile update; waiting for queue-maintenance "
            + "(poll every \(wait.pollInterval.components.seconds)s, "
            + "budget \(wait.budget.components.seconds)s)"
    )
    return try await waitForReindex(server: server, http: http, auth: auth, wait: wait)
}

/// Poll `GET /clients/me` until `reindex_pending` is false or the budget is spent: the
/// crate's `wait_for_reindex`. On expiry returns the last, still pending, profile; a claim
/// simply answers `204` until the gate lifts.
nonisolated func waitForReindex(
    server: ServerUrl,
    http: HttpClient = .shared,
    auth: ClientIdentity,
    wait: ReindexWait = .default
) async throws -> ClientProfile {
    let clock = ContinuousClock()
    let deadline = clock.now + wait.budget
    while true {
        try await Task.sleep(for: wait.pollInterval)
        let profile = try await PipetteMgmtClient.ClientApi.me(
            server: server, http: http, auth: auth)
        if !profile.reindexPending {
            AppLog.profile.info("reindex gate lifted")
            return profile
        }
        if clock.now >= deadline {
            AppLog.profile.warning(
                "reindex still pending after \(wait.budget.components.seconds)s; continuing "
                    + "(claim will idle until gate lifts)"
            )
            return profile
        }
    }
}
