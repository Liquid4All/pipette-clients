import Foundation

/// The downloaded-models list every screen renders, so a download or deletion updates all
/// of them at once. The root view loads it on appear and on each finished download.
@MainActor
@Observable
final class ModelStore {
    private(set) var models: [DiscoveredModel] = []
    private let ws: PipetteWorkspace

    init(ws: PipetteWorkspace) {
        self.ws = ws
    }

    func reload() {
        models = ws.availableModels()
    }

    /// Reloads rather than removing in memory: the disk scan is the canonical view.
    func delete(_ modelsToDelete: [DiscoveredModel]) {
        modelsToDelete.forEach(ws.deleteModel)
        reload()
    }
}
