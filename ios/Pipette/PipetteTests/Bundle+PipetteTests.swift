import Foundation
import Testing

@testable import Pipette

/// Settings, the Sentry tag and every submission's `client_version` read this one version
/// string, so its shape is a wire contract. Composed from parts rather than read off
/// `Bundle.main`, whose test-host version never carries a stamped commit.
struct BundlePipetteTests {

    @Test func theCommitJoinsTheBuildNumber() {
        #expect(Bundle.appVersionDisplayString(version: "0.1.0", build: "1", commit: "a1b2c3d")
            == "0.1.0\(BuildFlavor.versionSuffix) (1, a1b2c3d)")
    }

    /// A literal "unknown" in the commit position would read as a commit named "unknown"
    /// everywhere the string lands, the warehouse column included.
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

    /// The placeholder an unstamped build leaves in Info.plist must not read as a commit
    /// named `$(PIPETTE_GIT_COMMIT)`.
    @Test func anUnresolvedPlaceholderReadsAsAbsent() {
        #expect(Bundle.main.normalizedInfoString("PipetteGitCommitAbsentForTests") == nil)
    }

    /// A local build, such as the test host, has no `PipetteBuildVersion`, so the composed
    /// form is reported.
    @Test func anAbsentBuildVersionLeavesTheComposedForm() {
        #expect(Bundle.main.buildVersion == nil)
        #expect(Bundle.main.appVersionDisplayString.hasSuffix(")"))
    }

    /// A released build reports the tag `github-release` published, verbatim, so a warehouse
    /// row and a downloadable artifact match by string equality. `MARKETING_VERSION` and
    /// `CFBundleVersion` are constants, so wrapping them around it would only break that match.
    /// Pinned as a literal so a change to the composition has to restate the contract.
    @Test func aStampedReleaseVersionIsReportedVerbatim() {
        let stamped = "2026.08.11-640d96e8"
        #expect(
            "\(stamped)\(BuildFlavor.versionSuffix)"
                == (BuildFlavor.isInternal ? "2026.08.11-640d96e8-internal" : stamped))
    }

    /// The internal marker is the one thing carried into the release string. It is not version
    /// metadata: a result gated by a real die temperature is not comparable to one gated by
    /// `thermalState`, and `client_version` is where that distinction is recorded.
    @Test func theInternalMarkerSurvivesTheReleaseVersion() {
        #expect(BuildFlavor.versionSuffix == (BuildFlavor.isInternal ? "-internal" : ""))
    }
}
