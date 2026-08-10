import Foundation

/// Reports this device's profile and capability set to the management server.
///
/// Both halves drive job matching: the server slugifies each populated `device_*`
/// field into a reserved-namespace flag (`device_os_name: "iOS"` → `os:ios`) and
/// unions those with the `capabilities` reported here directly, forming the
/// client's effective capability set (mgmt `planner.md` §Client Matching Rules).
/// A client with no profile and no capabilities has an empty effective set and
/// matches nothing.
///
/// The inputs drift between launches — an OS update, a monthly security patch, an
/// app built against a different llama.cpp commit — so the profile is reported at
/// registration *and* re-reported at every launch. Resubmitting an unchanged
/// profile is cheap: the server voids the client's queue standing only when a
/// matching input actually changed (mgmt `client-integration.md` §2).
///
/// MainActor-isolated by the project default, which `DeviceProbe` requires; the
/// network call inside `refresh` hops off on its own `await`.
enum ProfileReporter {
    /// The profile + capability payload, single-sourced so the registration body
    /// and the per-launch patch can't drift apart.
    ///
    /// Always carries `client_details`. Registration also has a `client_details`
    /// argument of its own, and `RegisterRequest.encode(to:)` drops this copy so
    /// the two can't collide — callers don't have to think about it.
    ///
    /// Deliberately absent:
    /// - `device_gpu_*` / `device_npu_*` — a `*_vram_bytes` without its matching
    ///   `*_model` is a `400`, and iOS exposes no separately addressable VRAM to
    ///   report on a unified-memory device.
    /// - `device_os_build` and the run-environment values (battery level, power
    ///   state, low-power mode) — detected by `DeviceProbe`, but not profile
    ///   fields. They ride on benchmark submissions instead, where they describe
    ///   the conditions of one measurement rather than gating queue standing.
    ///
    /// Field-by-field from a `DeviceInfo`, as the CLI's `build_profile_update` is.
    static func profile() -> DeviceProfileUpdate {
        profile(device: DeviceProbe.detectDeviceInfo())
    }

    /// Testable seam: the mapping, over a caller-supplied snapshot.
    static func profile(device: DeviceInfo) -> DeviceProfileUpdate {
        DeviceProfileUpdate(
            clientDetails: device.deviceName,
            deviceName: device.deviceName,
            deviceFormFactor: device.deviceFormFactor,
            // Always sent as a pair: `device_os_version` without
            // `device_os_name` is a `400`.
            deviceOsName: device.deviceOsName,
            deviceOsVersion: device.deviceOsVersion,
            deviceChipModel: device.deviceChipModel,
            deviceRamBytes: device.deviceRamBytes,
            // Set-granular: a present value replaces the server's stored set
            // wholesale, so this is always the full current set.
            capabilities: Capabilities.flags()
        )
    }

    /// `PATCH /clients/me` — re-report the profile and return the server's
    /// updated record.
    ///
    /// Returns nil when this device has no registration yet: there is nothing to
    /// refresh and no identity to sign with. Network and server failures
    /// propagate — reporting is advisory, so callers log and carry on.
    ///
    /// Note this does **not** wait out the `reindex_pending` gate. Waiting is the
    /// concern of a caller that claims work and must discard in-flight jobs when
    /// its standing is voided; see `PlannerWorker.refreshProfile`.
    @discardableResult
    static func refresh(storage: Storage) async throws -> ClientProfile? {
        guard let session = try? storage.mgmtSession() else { return nil }

        let update = profile()
        let result = try await ManagementClient.updateMe(
            serverUrl: session.serverUrl,
            auth: session.auth,
            update: update
        )
        AppLog.profile.info(
            "profile reported: status=\(result.status) "
                + "reindex=\(result.reindexPending) caps=\(update.capabilities ?? [])"
        )
        return result
    }
}
