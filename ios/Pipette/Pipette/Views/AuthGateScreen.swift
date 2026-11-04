import Foundation

/// What the auth gate shows. Pulled out of the view, like Android's `reduceAuthGate`
/// (`ClerkAuth.kt`), because a `ViewBuilder` branch is unreachable from a test.
nonisolated enum AuthGateScreen: Equatable {
    case headless
    case loading
    case signIn
    case setup
    case keyMissing
    case mismatch
    case app
}

/// Reduce the launch mode, the Clerk session, and the local device identity to one screen.
///
/// `isHeadless` wins, like Android's `bypass`: nobody is there to answer a screen, and
/// `auth import` is how an operator repairs a device the other arms would stop.
///
/// A registration plus its signing key is what `mgmtSession()` signs with, so a device
/// holding both is admitted without a Clerk session (e.g. one restored by `auth import`).
/// Clerk is only required to create a registration. A record without its key cannot sign,
/// and setup would be refused by `ensureEmpty()`, so it gets `.keyMissing`.
///
/// The identity arms precede `clerkLoaded` because they are a local read: a registered
/// device starts work without waiting out `ClerkLoadingView`'s deadline.
///
/// A mismatch needs both ids. With no session, an imported bundle's link belongs to the
/// source device, so the device is admitted.
nonisolated func reduceAuthGate(
    isHeadless: Bool,
    clerkLoaded: Bool,
    clerkUserId: String?,
    hasRegistration: Bool,
    hasPrivateKey: Bool,
    linkedClerkUserId: String?
) -> AuthGateScreen {
    if isHeadless { return .headless }
    let hasDeviceIdentity = hasRegistration && hasPrivateKey
    if hasDeviceIdentity,
       let linkedClerkUserId,
       let clerkUserId,
       linkedClerkUserId != clerkUserId {
        return .mismatch
    }
    if hasDeviceIdentity { return .app }
    if hasRegistration { return .keyMissing }
    if !clerkLoaded { return .loading }
    return clerkUserId == nil ? .signIn : .setup
}
