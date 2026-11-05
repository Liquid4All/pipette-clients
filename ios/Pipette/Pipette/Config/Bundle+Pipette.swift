import Foundation

extension Bundle {
    /// An Info.plist string, with empty values and unresolved `$(...)` placeholders as nil.
    ///
    /// `nonisolated` because the target defaults to `MainActor` isolation and this is a
    /// plain read of an immutable dictionary; the members below are read off the main actor.
    nonisolated func normalizedInfoString(_ key: String) -> String? {
        guard let raw = object(forInfoDictionaryKey: key) as? String else {
            return nil
        }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !value.contains("$(") else {
            return nil
        }
        return value
    }

    /// Stamped by `ios/stamp-git-commit.sh`: `"a1b2c3d"`, or `"a1b2c3d-dirty"`. Nil when the
    /// build had no git metadata.
    nonisolated var gitCommit: String? {
        normalizedInfoString("PipetteGitCommit")
    }

    /// The GitHub release this build publishes as (e.g. `"2026.08.11-640d96e8"`), stamped by
    /// `ios/stamp-git-commit.sh` from CI's `PIPETTE_BUILD_VERSION`, so a submitted row names
    /// a downloadable artifact. Nil on a local or TestFlight build.
    ///
    /// Not `CFBundleVersion`: App Store Connect validates that key as a build number.
    nonisolated var buildVersion: String? {
        normalizedInfoString("PipetteBuildVersion")
    }

    /// One composition, so ``appVersionDisplayString`` and the `client_version:` capability
    /// flag cannot drift apart.
    nonisolated static func releaseVersion(_ version: String) -> String {
        "\(version)\(BuildFlavor.versionSuffix)"
    }

    /// A released build reports the release verbatim (`"2026.08.11-640d96e8-internal"`), so
    /// the value matches its release page; anything else the local `"0.1.1 (1, a1b2c3d)"`.
    ///
    /// The `-internal` suffix stays: without it a row gated by a real die temperature is
    /// indistinguishable from one gated by `thermalState`. `client_version` is free-form.
    nonisolated var appVersionDisplayString: String {
        if let buildVersion {
            return Self.releaseVersion(buildVersion)
        }
        let version = object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown"
        let build = object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "Unknown"
        return Self.appVersionDisplayString(version: version, build: build, commit: gitCommit)
    }

    /// Over caller-supplied parts: a test host's bundle carries the runner's version and no
    /// commit.
    nonisolated static func appVersionDisplayString(
        version: String, build: String, commit: String?
    ) -> String {
        // One spelling everywhere (Settings, the Sentry tag, `client_version`), as in the
        // crate, so a number traces back from whatever an operator pasted.
        let build = commit.map { "\(build), \($0)" } ?? build
        return "\(version)\(BuildFlavor.versionSuffix) (\(build))"
    }

    /// The PostHog `app_version`, e.g. "0.1.1.1": the build number is kept so a regression
    /// can be bisected to a build.
    var appVersionAnalyticsString: String {
        let version = object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown"
        guard let build = object(forInfoDictionaryKey: "CFBundleVersion") as? String, !build.isEmpty else {
            return version
        }
        return "\(version).\(build)"
    }
}
