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
}
