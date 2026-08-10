import Foundation

/// Versioned decoding for persisted job manifests.
///
/// Purely additive changes (a new optional field) never need a version bump —
/// Codable already handles them in both directions. Bump `currentVersion` and
/// add a `migrationStep` case when the shape changes incompatibly (a rename,
/// split, or type change), so manifests already on user devices are upgraded
/// on load instead of failing to decode. Each step is a plain JSON-dictionary
/// transform, unit-testable against string fixtures of the old shape.
///
/// Version 1 (2026-06) is the compatibility baseline: manifests written before
/// versioning existed owe no migration support, but every shape change from
/// version 1 onward must keep older versions loading.
nonisolated enum JobManifestSchema {
    static let currentVersion = 1

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
        default:
            return json
        }
    }
}
