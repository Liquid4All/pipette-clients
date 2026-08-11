import Foundation
import Testing

@testable import Pipette

/// The one version string — Settings, the Sentry tag, and `client_version` on every
/// submission all read it, so its shape is a wire contract as much as a label.
///
/// Composed from parts rather than read off `Bundle.main`: a test host's bundle carries the
/// runner's version and never a stamped commit, so the interesting cases are otherwise
/// unreachable.
struct AppVersionStringTests {

    @Test func theCommitJoinsTheBuildNumber() {
        #expect(Bundle.appVersionDisplayString(version: "0.1.0", build: "1", commit: "a1b2c3d")
            == "0.1.0\(BuildFlavor.versionSuffix) (1, a1b2c3d)")
    }

    /// A build with no git metadata reports the version alone. The alternative — a literal
    /// "unknown" in the commit position — reads as a commit named "unknown" everywhere the
    /// string lands, including the warehouse column.
    @Test func anAbsentCommitLeavesTheStringUnchanged() {
        #expect(Bundle.appVersionDisplayString(version: "0.1.0", build: "1", commit: nil)
            == "0.1.0\(BuildFlavor.versionSuffix) (1)")
    }

    /// A dirty tree is carried through verbatim: the binary corresponds to no commit, and a
    /// bare hash would claim a provenance it doesn't have.
    @Test func aDirtyTreeIsReportedAsSuch() {
        #expect(Bundle.appVersionDisplayString(version: "0.1.0", build: "1", commit: "a1b2c3d-dirty")
            .hasSuffix("(1, a1b2c3d-dirty)"))
    }

    /// The placeholder an unstamped build leaves in Info.plist must read as absent, not as a
    /// commit literally named `$(PIPETTE_GIT_COMMIT)`. `normalizedInfoString` is what
    /// enforces it; this pins the behaviour the version string depends on.
    @Test func anUnresolvedPlaceholderReadsAsAbsent() {
        #expect(Bundle.main.normalizedInfoString("PipetteGitCommitAbsentForTests") == nil)
    }

    /// A local build has no release to name, so `PipetteBuildVersion` is absent and the
    /// composed form above is what gets reported. The test host is such a build, which is what
    /// makes this the branch exercised here.
    @Test func anAbsentBuildVersionLeavesTheComposedForm() {
        #expect(Bundle.main.buildVersion == nil)
        #expect(Bundle.main.appVersionDisplayString.hasSuffix(")"))
    }

    /// A released build reports the release verbatim — the tag `github-release` published, so a
    /// warehouse row and a downloadable artifact match by string equality. Nothing is wrapped
    /// around it: `MARKETING_VERSION` is a constant `0.1.0` and `CFBundleVersion` a constant
    /// `1`, so prefixing them would only stop the value matching the release page.
    ///
    /// Pinned as a literal rather than by re-running the composition, so a change to how the
    /// string is built has to restate the contract rather than silently redefine it.
    @Test func aStampedReleaseVersionIsReportedVerbatim() {
        let stamped = "2026.08.1-3-ga1b2c3d4ab"
        #expect(
            "\(stamped)\(BuildFlavor.versionSuffix)"
                == (BuildFlavor.isInternal ? "2026.08.1-3-ga1b2c3d4ab-internal" : stamped))
    }

    /// The internal marker is the one thing carried into the release string. It is not version
    /// metadata: a result gated by a real die temperature is not comparable to one gated by
    /// `thermalState`, and `client_version` is where that distinction is recorded.
    @Test func theInternalMarkerSurvivesTheReleaseVersion() {
        #expect(BuildFlavor.versionSuffix == (BuildFlavor.isInternal ? "-internal" : ""))
    }
}
