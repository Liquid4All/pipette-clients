import SwiftUI

/// Views read the workspace with `@Environment(\.ws)` so none reaches a global.
private struct PipetteWorkspaceEnvironmentKey: EnvironmentKey {
    static let defaultValue = PipetteWorkspace.production
}

extension EnvironmentValues {
    var ws: PipetteWorkspace {
        get { self[PipetteWorkspaceEnvironmentKey.self] }
        set { self[PipetteWorkspaceEnvironmentKey.self] = newValue }
    }
}
