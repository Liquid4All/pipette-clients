import SwiftUI

/// SwiftUI injection point for the `Storage` dependency. `PipetteApp` seeds it
/// with the production instance at the composition root; views read it with
/// `@Environment(\.storage)` and thread it into the services they call, so no
/// view reaches a global. The default value is the production store — the same
/// composition-root instance the app injects, and the only place `.production`
/// is named.
private struct StorageEnvironmentKey: EnvironmentKey {
    static let defaultValue: Storage = FileStorage.production
}

extension EnvironmentValues {
    var storage: Storage {
        get { self[StorageEnvironmentKey.self] }
        set { self[StorageEnvironmentKey.self] = newValue }
    }
}
