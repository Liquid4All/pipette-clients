import Foundation

/// Versioned decoding for persisted job manifests.
///
/// An additive change (a new optional field) needs no version bump. Bump `currentVersion`
/// and add a `migrationStep` case for an incompatible one (a rename, split, or type change)
/// so manifests already on devices still load. Version 1 (2026-06) is the baseline, and a
/// manifest with no version reads as version 1.
///
/// Version 2 (2026-09) splits a cell's catalog reference into the id/side pair.
nonisolated enum JobManifestSchema {
    static let currentVersion = 2

    /// Decode a persisted manifest, upgrading older schema versions first.
    /// Throws when the data isn't a manifest at any known version.
    static func decode(_ data: Data) throws -> JobManifest {
        guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.coderReadCorrupt)
        }
        let storedVersion = json["schemaVersion"] as? Int ?? 1
        guard storedVersion < currentVersion else {
            // Current — or newer than this app knows (a downgrade). Decode
            // as-is: Codable ignores unknown keys, so newer manifests load
            // best-effort rather than disappearing.
            return try JSONDecoder().decode(JobManifest.self, from: data)
        }
        for version in storedVersion..<currentVersion {
            json = migrationStep(from: version, json)
        }
        json["schemaVersion"] = currentVersion
        let upgraded = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(JobManifest.self, from: upgraded)
    }

    /// One rung of the ladder: transform a version-N manifest dictionary into
    /// version N+1. Add a case here when bumping `currentVersion`.
    private static func migrationStep(from version: Int, _ json: [String: Any]) -> [String: Any] {
        switch version {
        case 1:
            return splittingCellBenchmarkReferences(json)
        default:
            return json
        }
    }

    /// Version 1 to 2: an older build wrote the whole `source/id` reference into
    /// `benchmarkId`. `BenchmarkId` refuses the `/`, and every id-keyed lookup expects the
    /// bare form.
    ///
    /// The side is recovered too: an absent `benchmarkSource` reads as `remote`, so an
    /// unsplit `local/` cell would submit a result the server never sanctioned. A side
    /// already recorded wins. Any other `benchmarkId` is left as it was.
    private static func splittingCellBenchmarkReferences(
        _ json: [String: Any]
    ) -> [String: Any] {
        guard let cells = json["cells"] as? [[String: Any]] else { return json }
        var json = json
        json["cells"] = cells.map { cell -> [String: Any] in
            guard let stored = cell["benchmarkId"] as? String, stored.contains("/"),
                  let reference = try? SourcedBenchmarkId(parsing: stored)
            else { return cell }
            var cell = cell
            cell["benchmarkId"] = reference.id.value
            // `as? String`, not a nil-key check: a `null` here deserializes to `NSNull`,
            // which is not absent to a dictionary but is to `decodeIfPresent`.
            if cell["benchmarkSource"] as? String == nil {
                cell["benchmarkSource"] = reference.source.rawValue
            }
            return cell
        }
        return json
    }
}
