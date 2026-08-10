import Foundation

/// Metadata written to disk when a download starts so the UI can reconstruct
/// its state after the app is terminated and relaunched by the system.
struct DownloadRecord: Codable {
    let filename: String
    let urlString: String
    let repo: String?
    /// Catalog family id. On relaunch `restoreState()` rehydrates it into the
    /// in-flight `Download`, so the Models-tab download row stays grouped under its
    /// family; not part of the on-disk manifest.
    var familyId: String?
    /// The typed model definition this file was ignited from, so completion can
    /// write an exact manifest even after an app relaunch redelivers the event.
    /// Nil for sideloads / legacy records (provenance then falls back to strings).
    var source: Model?
}
