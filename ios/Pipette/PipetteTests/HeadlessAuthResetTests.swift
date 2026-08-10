import Foundation
import Testing

@testable import Pipette

/// What `auth reset` actually does to stored state, as opposed to how it parses.
///
/// Worth its own suite: the command's whole purpose is discarding a credential, and an
/// earlier version reported that it had while leaving the signing key in place.
///
/// The Keychain half is injected. A Simulator test host cannot delete a Keychain item, so
/// calling the real one here would assert the harness's limits rather than the command's
/// behaviour — and did, until this suite failed on CI while passing on device.
@MainActor struct HeadlessAuthResetTests {
    @Test func resetWithoutForceLeavesTheRegistrationAlone() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        var keyDeletes = 0
        #expect(AuthCommands.reset(
            force: false, storage: storage, deleteKey: { keyDeletes += 1; return true }) == false)
        // A refusal must not have touched the credential either.
        #expect(keyDeletes == 0)
        #expect(storage.identity.getRegistration()?.clientId == registrationData().clientId)
    }

    @Test func resetWithForceClearsTheRegistration() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        var keyDeletes = 0
        #expect(AuthCommands.reset(
            force: true, storage: storage, deleteKey: { keyDeletes += 1; return true }))
        #expect(storage.identity.getRegistration() == nil)
        #expect(!storage.identity.isRegistered)
        #expect(keyDeletes == 1)
    }

    /// A device with nothing stored is already in the state the command produces, so it
    /// succeeds rather than reporting a failure the operator cannot act on.
    @Test func resetOnAnUnregisteredDeviceSucceeds() {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        #expect(AuthCommands.reset(
            force: true, storage: storage, deleteKey: { true }))
        #expect(storage.identity.getRegistration() == nil)
    }

    /// The command reports the credential's fate, so a Keychain delete that fails has to
    /// fail the command — leaving the key behind is the outcome it exists to prevent.
    @Test func aFailedKeyDeleteFailsTheReset() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        try storage.identity.putRegistration(registrationData())

        #expect(AuthCommands.reset(
            force: true, storage: storage, deleteKey: { false }) == false)
        // The record still goes: a half-cleared identity is reported, not silently kept.
        #expect(storage.identity.getRegistration() == nil)
    }
}
