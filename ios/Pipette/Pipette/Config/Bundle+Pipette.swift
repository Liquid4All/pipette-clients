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

    /// "<short version>[-internal] (<build>[, <commit>])" — e.g. "0.1.0 (1, a1b2c3d)", or
    /// "0.1.0-internal (1)" for an internal build with no git metadata. Shared by the
    /// Settings debug row, the Sentry `app_version` tag, and the `client_version` on every
    /// submitted result, so the displayed version and the reported version can never drift.
    ///
    /// The commit is what makes the string identifying: `CFBundleVersion` is `1` on every
    /// local build, so without it two results from different source trees are
    /// indistinguishable in the warehouse.
    ///
    /// The `-internal` marker is here rather than in a separate field because this string is
    /// already the one thing that reaches all of them: a result gated by a real die
    /// temperature is not comparable to one gated by `thermalState`, and without the
    /// marker two such rows are indistinguishable. `client_version` is free-form
    /// upstream (`Option<String>`), so the suffix breaks no parser.
    /// The commit this build was made from, stamped into the built Info.plist by
    /// `ios/stamp-git-commit.sh` — `"a1b2c3d"`, or `"a1b2c3d-dirty"` when the tree carried
    /// uncommitted changes. Nil when the build had no git metadata to read, which
    /// `normalizedInfoString` also reports for the unresolved placeholder.
    var gitCommit: String? {
        normalizedInfoString("PipetteGitCommit")
    }

    var appVersionDisplayString: String {
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
