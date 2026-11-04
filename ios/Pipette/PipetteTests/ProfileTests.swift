import Foundation
import Testing

@testable import Pipette

/// Pins the device-profile payload against the server's schema (pipette-mgmt's
/// `docs/httpapi.md` §2.2.1, §2.4.1): a `400` it would return, or a field with no profile
/// column to land in.
@MainActor
struct ProfileTests {
    private func encoded(_ update: DeviceProfileFields) throws -> [String: Any] {
        let data = try JSONEncoder().encode(update)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func device(_ labels: DeviceLabels? = nil) throws -> ClientDeviceStore {
        let store = ClientDeviceStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("profile-\(UUID().uuidString)", isDirectory: true))
        if let labels { try store.put(labels) }
        return store
    }

    @Test func carriesTheProfileFieldsTheServerAccepts() throws {
        let json = try encoded(buildProfileUpdate(deviceStore: try device(), capabilities: installedRuntimeCapabilities()).device)
        #expect(json["device_os_name"] as? String == "iOS" || json["device_os_name"] as? String == "iPadOS")
        #expect((json["device_name"] as? String)?.isEmpty == false)
        #expect((json["device_os_version"] as? String)?.isEmpty == false)
        #expect((json["device_chip_model"] as? String)?.isEmpty == false)
        #expect(((json["device_ram_bytes"] as? NSNumber)?.int64Value ?? 0) > 0)
        #expect(json["device_cpu_arch"] as? String == detectCpuArch())
        #expect(json["device_os_family"] as? String == "iOS")
    }

    // The crate's `every_form_factor_is_wired_as_the_type_serializes_it`. The words
    // themselves are pinned in `DeviceInfoTests.formFactorWireSpellingsMatchTheServer`.
    @Test func everyFormFactorIsWiredAsTheTypeSerializesIt() throws {
        for formFactor in DeviceFormFactor.allCases {
            let json = try encoded(profileUpdate(
                device: deviceInfoFixture(formFactor: formFactor), capabilities: []).device)
            #expect(json["device_form_factor"] as? String == formFactor.rawValue,
                    "\(formFactor) is wired as something other than its own raw value")
        }
    }

    // A `*_vram_bytes` without its matching `*_model` is a 400, and a null would leave
    // the stored value unchanged anyway, so absent, never null.
    @Test func omitsGpuAndNpuFieldsEntirely() throws {
        let json = try encoded(buildProfileUpdate(deviceStore: try device(), capabilities: installedRuntimeCapabilities()).device)
        for key in ["device_gpu_model", "device_gpu_vram_bytes", "device_npu_model", "device_npu_vram_bytes"] {
            #expect(json[key] == nil, "\(key) must not be sent")
        }
    }

    // Sent on benchmark submissions, not the profile. The run-environment values change
    // minute to minute, and a profile change voids the client's queue standing.
    @Test func aProfileUpdateOmitsWhatTheSchemaHasNoColumnFor() throws {
        let json = try encoded(buildProfileUpdate(deviceStore: try device(), capabilities: installedRuntimeCapabilities()).device)
        for key in [
            "device_os_build", "device_os_security_patch",
            "device_battery_level", "device_power_state", "device_power_save_mode",
        ] {
            #expect(json[key] == nil, "\(key) is not a profile field")
        }
    }

    // A present value replaces the server's stored set wholesale, so the payload carries
    // the full set, as `build_profile_update` merges it.
    @Test func aProfileUpdateSendsTheFullMergedCapabilitySet() throws {
        let device = deviceInfoFixture()
        let sent = try #require(
            try encoded(profileUpdate(device: device, capabilities: installedRuntimeCapabilities()).device)["capabilities"] as? [String])
        let clientVersion = clientVersionCapabilityFlag(Bundle.main.buildVersion).map { [$0] } ?? []
        #expect(Set(sent) == Set(installedRuntimeCapabilities() + hardwareCapabilities(device: device)
                                 + clientVersion))
        #expect(sent.contains("cpu_vendor:apple"))
        #expect(sent == sent.sorted())
        #expect(Set(sent).count == sent.count)
    }

    // The crate's `build_profile_update` leaves it `None`, so a worker's refresh cannot
    // overwrite the description the operator registered with.
    @Test func aProfileUpdateSendsNoClientDetails() throws {
        #expect(try encoded(buildProfileUpdate(deviceStore: try device(), capabilities: installedRuntimeCapabilities()).device)["client_details"] == nil)

        let patch = try #require(
            JSONSerialization.jsonObject(
                with: try JSONEncoder().encode(profileUpdate(device: deviceInfoFixture(), capabilities: installedRuntimeCapabilities()))
            ) as? [String: Any]
        )
        #expect(patch["client_details"] == nil)
        #expect(patch["device_name"] as? String == "iPhone 17 Pro")
    }

    // The server rejects the whole update over one non-canonical flag, costing this
    // device every other flag it reports.
    @Test func everyCapabilityItSendsIsCanonical() throws {
        for flag in profileUpdate(device: deviceInfoFixture(),
                                  capabilities: installedRuntimeCapabilities()).device.capabilities ?? [] {
            #expect((try? CapabilityFlag(flag)) != nil, "\(flag) is not canonical")
        }
    }

    // The server can gate an incompatible job-body change on it, but no plan emits one.
    // The crate's `no_job_schema_flag_is_reported`.
    @Test func noJobSchemaFlagIsReported() {
        #expect(!(profileUpdate(device: deviceInfoFixture(),
                                capabilities: installedRuntimeCapabilities()).device.capabilities ?? [])
            .contains { $0.hasPrefix("job_schema:") })
    }

    // As the crate's `build_profile_update` passes them to `detect_device_info`.
    @Test func theOperatorsLabelsNameTheDeviceOnTheDetectedProfile() throws {
        let json = try encoded(buildProfileUpdate(
            deviceStore: try device(DeviceLabels(deviceName: "  lab-07  ", deviceFormFactor: .tablet)),
            capabilities: []).device)
        #expect(json["device_name"] as? String == "lab-07")
        #expect(json["device_form_factor"] as? String == "tablet")
    }

    // The host-based tests can only assert shape; a fixed snapshot shows a field wired to
    // the wrong source. The crate's `profile_update`.
    @Test func profileUpdateMapsEachFieldFromTheDeviceSnapshot() throws {
        let json = try encoded(profileUpdate(device: deviceInfoFixture(formFactor: .tablet), capabilities: installedRuntimeCapabilities()).device)

        #expect(json["device_name"] as? String == "iPhone 17 Pro")
        #expect(json["device_form_factor"] as? String == "tablet")
        #expect(json["device_os_name"] as? String == "iOS")
        #expect(json["device_os_version"] as? String == "26.4")
        #expect(json["device_chip_model"] as? String == "Apple A19 Pro")
        #expect((json["device_ram_bytes"] as? NSNumber)?.uint64Value == 8_589_934_592)
    }
}

/// As the crate's `profile_body` answers both `PATCH` and `GET /clients/me` in its tests.
private func profileBody(pending: Bool) -> String {
    """
    {"client_id": "ev1_test", "organization": "o", "client_details": "d",
     "contact_email": "a@b.com", "status": "approved", "reindex_pending": \(pending)}
    """
}

/// The crate's `wait_for_reindex`, over a stubbed server.
@MainActor
struct ProfileReindexTests {
    @Test func theWaitEndsWhenTheGateLiftsOnALaterPoll() async throws {
        let (pending, lifted) = (profileBody(pending: true), profileBody(pending: false))
        let (http, server, seen) = try RoutedURLProtocol.serve { _, index in
            .reply(200, index < 2 ? pending : lifted)
        }

        let profile = try await waitForReindex(
            server: server, http: http, auth: try testIdentity(),
            wait: ReindexWait(pollInterval: .zero, budget: .seconds(60)))

        #expect(!profile.reindexPending)
        #expect(seen.methodsAndPaths == Array(repeating: "GET /clients/me", count: 3))
    }

    /// The wait does not spend its whole budget on a host that is not coming back.
    @Test func theWaitSurfacesAServerError() async throws {
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(503, "down") }

        await #expect(throws: (any Error).self) {
            try await waitForReindex(
                server: server, http: http, auth: try testIdentity(),
                wait: ReindexWait(pollInterval: .zero, budget: .seconds(60)))
        }
        #expect(seen.requests.count == 1, "a failing poll must not be retried in the loop")
    }
}

/// The crate's `refresh_profile_at_startup`: the one path that reports the whole
/// capability set, and the only hand-over to the reindex wait. A client that skipped the
/// wait would spend the first minutes of every launch on claims that could not succeed.
@MainActor
struct ProfileStartupTests {
    private static let wait = ReindexWait(pollInterval: .zero, budget: .seconds(60))

    /// As the crate's `seed_registration`: both halves of the identity, so the refresh
    /// reaches the stub rather than the real collector.
    private func seedRegistration(_ ws: PipetteWorkspace, server: ServerUrl) throws {
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(
            IdentityRegistration(clientId: try ClientId("ev1_test"), status: "approved",
                                 serverUrl: server, organization: "Example",
                                 contactEmail: "user@example.com",
                                 registeredAt: "2026-05-28T16:41:00Z"))
    }

    @Test func refreshProfileAtStartupPatchesOnceWhenTheGateIsClear() async throws {
        let clear = profileBody(pending: false)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, clear) }
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedRegistration(ws, server: server)

        let profile = try await refreshProfileAtStartup(
            registration: ws.registration, deviceStore: ws.device, http: http,
            capabilities: installedRuntimeCapabilities(), wait: Self.wait)

        #expect(try #require(profile).reindexPending == false)
        #expect(seen.methodsAndPaths == ["PATCH /clients/me"])
    }

    /// The server replaces its stored set with whatever arrives, so a set that left
    /// something out silently narrows what this device matches.
    @Test func refreshProfileAtStartupSendsTheCapabilitySet() async throws {
        let clear = profileBody(pending: false)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, clear) }
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedRegistration(ws, server: server)

        let buildFlags = try installedRuntimeBuildFlags()
        try await refreshProfileAtStartup(
            registration: ws.registration, deviceStore: ws.device, http: http,
            capabilities: installedRuntimeCapabilities() + buildFlags, wait: Self.wait)

        let body = try #require(seen.bodies.first)
        let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let sent = try #require(json["capabilities"] as? [String])
        #expect(sent.contains("runtime:llama_cpp"))
        #expect(sent.contains("runtime:mlx"))
        #expect(sent.contains("runtime:apple_foundation"))
        // The build flags go in the same update: a second one would trip the reindex gate
        // again.
        #expect(!buildFlags.isEmpty)
        #expect(buildFlags.allSatisfy(sent.contains))
        #expect(seen.methodsAndPaths == ["PATCH /clients/me"])
        #expect(sent == sent.sorted())
        #expect(Set(sent).count == sent.count)
        // A worker's refresh must not overwrite the description the operator registered.
        #expect(json["client_details"] == nil)
    }

    /// `reindex_pending` means the queue standing is being rebuilt, and a claim before it
    /// lifts is answered `204` and idles the worker for a whole idle wait.
    @Test func refreshProfileAtStartupWaitsOutTheReindexGate() async throws {
        let (pending, lifted) = (profileBody(pending: true), profileBody(pending: false))
        let (http, server, seen) = try RoutedURLProtocol.serve { _, index in
            // 0 is the PATCH; the polls that follow hold the gate once, then lift it.
            .reply(200, index < 2 ? pending : lifted)
        }
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedRegistration(ws, server: server)

        let profile = try await refreshProfileAtStartup(
            registration: ws.registration, deviceStore: ws.device, http: http,
            capabilities: installedRuntimeCapabilities(), wait: Self.wait)

        #expect(try #require(profile).reindexPending == false)
        #expect(seen.methodsAndPaths
            == ["PATCH /clients/me", "GET /clients/me", "GET /clients/me"])
    }

    /// The caller claims anyway and the server answers `204` until the gate lifts, which
    /// costs less than holding the launch open.
    @Test func aSpentBudgetEndsTheStartupWait() async throws {
        let pending = profileBody(pending: true)
        let (http, server, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, pending) }
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seedRegistration(ws, server: server)

        let profile = try await refreshProfileAtStartup(
            registration: ws.registration, deviceStore: ws.device, http: http,
            capabilities: installedRuntimeCapabilities(), wait: ReindexWait(pollInterval: .zero, budget: .zero))

        #expect(try #require(profile).reindexPending)
        #expect(seen.methodsAndPaths == ["PATCH /clients/me", "GET /clients/me"])
    }

    /// An unenrolled device must not throw at every launch. The crate errors here instead,
    /// because its worker cannot run unregistered at all; this client's UI can.
    @Test func refreshProfileAtStartupReportsNothingForAnUnregisteredDevice() async throws {
        let clear = profileBody(pending: false)
        let (http, _, seen) = try RoutedURLProtocol.serve { _, _ in .reply(200, clear) }
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        let profile = try await refreshProfileAtStartup(
            registration: ws.registration, deviceStore: ws.device, http: http,
            capabilities: installedRuntimeCapabilities(), wait: Self.wait)

        #expect(profile == nil)
        #expect(seen.requests.isEmpty)
    }
}
