import Foundation
import Testing

@testable import Pipette

struct ClientAuthTests {
    static func registerInput() throws -> RegisterInput {
        RegisterInput(
            organization: "Example", contactEmail: try ContactEmail("user@example.com"),
            clientDetails: "ci box")
    }

    @Test(arguments: [
        // both none
        (nil, nil, false, nil, nil),
        // input wins
        ("new", .laptop, true, "new", .laptop),
        // inherit prior
        (nil, nil, true, "prior-name", .desktop),
        // empty name inherits
        ("  ", .laptop, true, "prior-name", .laptop),
    ] as [(String?, DeviceFormFactor?, Bool, String?, DeviceFormFactor?)])
    func mergeLabelsForRegisterCases(
        inputName: String?, inputFormFactor: DeviceFormFactor?, usePrior: Bool,
        wantName: String?, wantFormFactor: DeviceFormFactor?
    ) {
        let prior = usePrior
            ? DeviceLabels(deviceName: "prior-name", deviceFormFactor: .desktop)
            : DeviceLabels.empty
        let got = mergeLabelsForRegister(
            deviceName: inputName, deviceFormFactor: inputFormFactor, prior: prior)
        #expect(got.deviceName == wantName)
        #expect(got.deviceFormFactor == wantFormFactor)
    }

    // On the registration path a 401 is the server's pre-auth-key rejection
    // (invalid/expired/used); the user sees a key-specific cause, not the raw
    // "HTTP 401: {body}".
    @Test func humanizes401AsPreauthKeyRejection() {
        let message = humanizedRegistrationError(
            PipetteMgmtClient.Error.httpStatus(
                method: "POST", url: "https://mgmt.test/clients/register", status: 401,
                body: "consumed"),
            preauthContext: true
        )
        #expect(message.contains("pre-auth key"))
        #expect(!message.contains("401"))
    }

    // A 403 means the collector requires a key but none was supplied.
    @Test func humanizes403AsPreauthKeyRequired() {
        let message = humanizedRegistrationError(
            PipetteMgmtClient.Error.httpStatus(
                method: "POST", url: "https://mgmt.test/clients/register", status: 403,
                body: ""),
            preauthContext: true
        )
        #expect(message.contains("requires a pre-auth key"))
        #expect(!message.contains("403"))
    }

    // Without the pre-auth context (e.g. the public benchmark-catalog sync), a
    // 401/403 must not be mislabeled as a key problem.
    @Test func doesNotRemap401OutsidePreauthContext() {
        let message = humanizedRegistrationError(
            PipetteMgmtClient.Error.httpStatus(
                method: "POST", url: "https://mgmt.test/clients/register", status: 401,
                body: "nope")
        )
        #expect(!message.contains("pre-auth key"))
    }

    @Test func doesNotRemapUnrelatedHTTPStatus() {
        let message = humanizedRegistrationError(
            PipetteMgmtClient.Error.httpStatus(
                method: "POST", url: "https://mgmt.test/clients/register", status: 500,
                body: "boom"),
            preauthContext: true
        )
        #expect(!message.contains("pre-auth key"))
    }

    /// Registration carries the two fields that the server derives `arch:` and `os_family:`
    /// from, and none of the profile that the planner worker reports.
    @Test func registrationProfileCarriesTheArchAndOsFamilyOnly() {
        #expect(
            registrationProfile()
                == DeviceProfileFields(deviceCpuArch: detectCpuArch(), deviceOsFamily: "iOS"))
    }

    /// The crate's `ensure_empty` runs first in `register`, before any keypair or request, so
    /// a second registration cannot silently rotate the identity and orphan every result
    /// submitted under the first.
    @Test func registerRefusesOverARecordTheDeviceHolds() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.put(try registrationData())
        let held = ws

        await #expect(throws: IdentityError.self) {
            _ = try await register(
                registrationStore: held.registration, device: held.device,
                server: try ServerUrl("https://collector.example.com"),
                input: try Self.registerInput())
        }
        #expect(held.registration.get()?.clientId.value == "client-1")
    }

    /// A key that no record claims names no client, so nothing local can sign as one.
    /// Refusing over it would strand the device on `SetupView`, with `auth reset` (no UI) as
    /// the only way out.
    @Test func anOrphanedKeyIsDiscardedSoRegistrationCanProceed() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(String(repeating: "ab", count: 32)))

        #expect(discardOrphanedSigningKey(ws: ws))

        #expect(ws.registration.getPrivateKey() == nil)
        #expect(throws: Never.self) { try ws.registration.ensureEmpty() }
    }

    /// Gated on nothing that awaits submission, the way `auth reset` gates on it: the new
    /// client id would otherwise claim those results, and a silent re-attribution of a
    /// measurement is the harm `ensureEmpty` exists to prevent.
    @Test func anOrphanedKeyIsKeptWhileResultsAwaitSubmission() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(String(repeating: "ab", count: 32)))
        let pending = try #require(ws.results.resultDir(.remotePending, CellId("r-1")))
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: true)

        #expect(!discardOrphanedSigningKey(ws: ws))

        #expect(ws.registration.getPrivateKey() != nil)
        #expect(throws: IdentityError.self) { try ws.registration.ensureEmpty() }
    }
}
