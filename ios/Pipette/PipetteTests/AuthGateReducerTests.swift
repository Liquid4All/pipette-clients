import Testing

@testable import Pipette

/// The gate's routing truth table. `reduceAuthGate` is a plain function so that the ordering
/// that admits a registered device with no Clerk session can be asserted.
struct AuthGateReducerTests {
    /// One row of the table: the six inputs and the screen they reduce to. `name` is what a
    /// failing row reports, so each row says which rule it stands for.
    struct Case: Sendable, CustomTestStringConvertible {
        let name: String
        /// Defaulted, because every row but the headless ones is an app run.
        var isHeadless = false
        let clerkLoaded: Bool
        let clerkUserId: String?
        /// The two halves separately, because three of the four pairs route differently.
        let hasRegistration: Bool
        let hasPrivateKey: Bool
        let linkedClerkUserId: String?
        let expected: AuthGateScreen

        var testDescription: String { name }
    }

    nonisolated static let cases: [Case] = [
        // A CLI run shows a readout and nothing else, whatever the device holds. These two
        // rows stand for the arm that short-circuits every other one.
        Case(
            name: "headless, with an identity",
            isHeadless: true,
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: "user_source",
            expected: .headless),
        Case(
            name: "headless, signed in as a different account",
            isHeadless: true,
            clerkLoaded: true, clerkUserId: "user_b",
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: "user_a",
            expected: .headless),
        // An app run on a device a CLI run registered: it holds both halves of its identity
        // and has no session. This row also proves the identity check runs before
        // `clerkLoaded`, so the device does not wait out `ClerkLoadingView`'s deadline.
        Case(
            name: "identity, no session, Clerk not loaded",
            clerkLoaded: false, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: nil,
            expected: .app),
        // The `auth import` case: the bundle carries the source device's Clerk link, so the
        // record names an account this install was never signed in to.
        Case(
            name: "identity, no session, a link from the imported bundle",
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: "user_source",
            expected: .app),
        Case(
            name: "identity, signed in as the linked account",
            clerkLoaded: true, clerkUserId: "user_a",
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: "user_a",
            expected: .app),
        Case(
            name: "identity, signed in with no link recorded",
            clerkLoaded: true, clerkUserId: "user_a",
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: nil,
            expected: .app),
        Case(
            name: "identity, signed in as a different account",
            clerkLoaded: true, clerkUserId: "user_b",
            hasRegistration: true, hasPrivateKey: true, linkedClerkUserId: "user_a",
            expected: .mismatch),
        // Registering needs a session, because `SetupView` records the Clerk link on the
        // record it creates.
        Case(
            name: "no identity, signed in",
            clerkLoaded: true, clerkUserId: "user_a",
            hasRegistration: false, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .setup),
        // The App Review path: a fresh install holds nothing and is asked to sign in.
        Case(
            name: "no identity, signed out",
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: false, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .signIn),
        Case(
            name: "no identity, Clerk not loaded",
            clerkLoaded: false, clerkUserId: nil,
            hasRegistration: false, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .loading),
        // A record with no signing key cannot claim or submit work, and setup cannot mint a
        // replacement over it either: `ensureEmpty()` refuses over the record. These four
        // rows stand for the arm that names that state instead of offering registration.
        Case(
            name: "a record with no key, signed in",
            clerkLoaded: true, clerkUserId: "user_a",
            hasRegistration: true, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .keyMissing),
        Case(
            name: "a record with no key, signed out",
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .keyMissing),
        // Ahead of `clerkLoaded`, as the other identity arms are: the read is local, and the
        // screen is the same either way.
        Case(
            name: "a record with no key, Clerk not loaded",
            clerkLoaded: false, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .keyMissing),
        // A stale link does not make it a mismatch: the device has nothing to sign with, so
        // the account it was pinned to is not the problem to report.
        Case(
            name: "a record with no key and a stale link, signed in as another account",
            clerkLoaded: true, clerkUserId: "user_b",
            hasRegistration: true, hasPrivateKey: false, linkedClerkUserId: "user_a",
            expected: .keyMissing),
        // A CLI run still reaches the readout, because `auth import` is the repair for it.
        Case(
            name: "headless, a record with no key",
            isHeadless: true,
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: true, hasPrivateKey: false, linkedClerkUserId: nil,
            expected: .headless),
        // A key no record claims is dead weight `discardOrphanedSigningKey` drops, so the
        // device registers fresh rather than stopping.
        Case(
            name: "a key with no record, signed in",
            clerkLoaded: true, clerkUserId: "user_a",
            hasRegistration: false, hasPrivateKey: true, linkedClerkUserId: nil,
            expected: .setup),
        Case(
            name: "a key with no record, signed out",
            clerkLoaded: true, clerkUserId: nil,
            hasRegistration: false, hasPrivateKey: true, linkedClerkUserId: nil,
            expected: .signIn),
    ]

    @Test(arguments: cases) nonisolated func theGateRoutes(testCase: Case) {
        let screen = reduceAuthGate(
            isHeadless: testCase.isHeadless,
            clerkLoaded: testCase.clerkLoaded,
            clerkUserId: testCase.clerkUserId,
            hasRegistration: testCase.hasRegistration,
            hasPrivateKey: testCase.hasPrivateKey,
            linkedClerkUserId: testCase.linkedClerkUserId
        )
        #expect(screen == testCase.expected)
    }
}
