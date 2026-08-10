import Foundation

enum ClerkConfiguration {
    static var isComplete: Bool {
        publishableKey != nil && frontendApiDomain != nil
    }

    static var publishableKey: String? {
        Bundle.main.normalizedInfoString("ClerkPublishableKey")
    }

    static var frontendApiDomain: String? {
        Bundle.main.normalizedInfoString("ClerkFrontendApiDomain")
    }
}
