import Foundation

extension Bundle {
    /// Reads an Info.plist string, treating empty values or unresolved `$(…)`
    /// build-setting placeholders as absent. Shared by `ClerkConfiguration` and
    /// `SentryConfiguration` so the resolution rule lives in one place.
    func normalizedInfoString(_ key: String) -> String? {
        guard let raw = object(forInfoDictionaryKey: key) as? String else {
            return nil
        }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !value.contains("$(") else {
            return nil
        }
        return value
    }

    /// The commit this build was made from, stamped into the built Info.plist by
    /// `ios/stamp-git-commit.sh` — `"a1b2c3d"`, or `"a1b2c3d-dirty"` when the tree carried
    /// uncommitted changes. Nil when the build had no git metadata to read, which
    /// `normalizedInfoString` also reports for the unresolved placeholder.
    var gitCommit: String? {
        normalizedInfoString("PipetteGitCommit")
    }

    /// The version this build publishes as — `ci/version.sh`'s output, stamped into the built
    /// Info.plist by `ios/stamp-git-commit.sh` when CI passes `PIPETTE_BUILD_VERSION`
    /// (e.g. `"2026.08.1-3-ga1b2c3d4ab"`). Also the GitHub release's tag and name, which is the
    /// point: a submitted row names a downloadable artifact.
    ///
    /// Nil on a local build and on the TestFlight path, neither of which has a release to name.
    ///
    /// Not `CFBundleVersion`: that key is the build *number*, hardcoded to `1` in the project,
    /// stamped by `agvtool` only on the TestFlight path, and validated by App Store Connect —
    /// so it can neither carry this value nor be replaced by it.
    var buildVersion: String? {
        normalizedInfoString("PipetteBuildVersion")
    }

    /// A released build reports the release, verbatim: `"2026.08.1-3-ga1b2c3d4ab-internal"`.
    /// Anything else falls back to the composed local form, `"0.1.0 (1, a1b2c3d)"`.
    ///
    /// The release string is not wrapped in the marketing version and build number. Those add
    /// nothing a release has to say — `MARKETING_VERSION` has never moved off `0.1.0` and
    /// `CFBundleVersion` is `1` — and wrapping them around it would stop the value matching the
    /// release page it came from. It already ends in the commit, so nothing identifying is lost.
    ///
    /// The `-internal` suffix survives because it is not version metadata: a result gated by a
    /// real die temperature is not comparable to one gated by `thermalState`, and without the
    /// marker two such rows are indistinguishable. `client_version` is free-form upstream
    /// (`Option<String>`), so the suffix breaks no parser.
    var appVersionDisplayString: String {
        if let buildVersion {
            return "\(buildVersion)\(BuildFlavor.versionSuffix)"
        }
        let version = object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown"
        let build = object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "Unknown"
        return Self.appVersionDisplayString(version: version, build: build, commit: gitCommit)
    }

    /// The composition, over caller-supplied parts. A test host's bundle carries the test
    /// runner's version and no commit, so the shape is otherwise unassertable.
    static func appVersionDisplayString(
        version: String, build: String, commit: String?
    ) -> String {
        // The commit joins the build number inside the existing parenthetical rather than
        // taking a field of its own: this string is already the one that reaches Settings,
        // the Sentry tag, and `client_version` on every submission, and the crate makes the
        // same call — one spelling everywhere, so a number can be traced back from whatever
        // an operator pasted. Omitted entirely when unknown, so it never reads as a commit
        // named "Unknown".
        let build = commit.map { "\(build), \($0)" } ?? build
        return "\(version)\(BuildFlavor.versionSuffix) (\(build))"
    }

    /// Dotted version + build, e.g. "0.1.0.1". The PostHog `app_version` super property.
    ///
    /// Android fills the same key from `BuildConfig.VERSION_NAME`, which its build script composes as
    /// `"<base>.<CI run number>"` (e.g. "1.0.42"). Matching that shape matters twice over: both
    /// platforms report into one project, so a differing grammar means every version filter has to be
    /// written twice, and the build component has to survive, or every build of a given marketing
    /// version collapses into one value and a regression can't be bisected to the build that caused it.
    ///
    /// ``appVersionDisplayString`` stays the human-facing "0.1.0 (1)" form for Settings and the Sentry tag.
    var appVersionAnalyticsString: String {
        let version = object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Unknown"
        guard let build = object(forInfoDictionaryKey: "CFBundleVersion") as? String, !build.isEmpty else {
            return version
        }
        return "\(version).\(build)"
    }
}
