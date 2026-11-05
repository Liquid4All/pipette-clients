import Foundation

/// Whether this device may submit results at all: it has to hold a registration, since
/// the collector only takes a result signed by a registered client.
enum ResultSubmissionFeatureGate {
    nonisolated static func canSubmitResults(registration: IdentityRegistration?) -> Bool {
        registration != nil
    }
}
