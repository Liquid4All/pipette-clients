import Testing
import Foundation
@testable import Pipette

/// One case for the auto-submit gate table. `hasRegistration` (not a
/// `IdentityRegistration?`) keeps the case `Sendable` for `@Test(arguments:)`; the
/// test builds the registration itself.
private struct GateCase: Sendable, CustomTestStringConvertible {
    let name: String
    let autoSubmit: Bool
    let contribute: Bool?
    let online: Bool
    let hasRegistration: Bool
    let expected: Bool
    var testDescription: String { name }
}

/// PIP-358: the auto-submit gate shared by the per-cell upload and the job-end
/// sweep. Exercised directly as a pure predicate — the offline case is the one
/// the reachability change added, and the rest guard against a regression that
/// would submit when the run/job/registration didn't opt in.
@Suite @MainActor struct JobExecutorAutoSubmitGateTests {
    private func manifest(contributeResults: Bool?) -> JobManifest {
        var manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [],
            status: .completed
        )
        manifest.contributeResults = contributeResults
        return manifest
    }

    @Test(arguments: [
        GateCase(name: "all conditions met", autoSubmit: true, contribute: true, online: true, hasRegistration: true, expected: true),
        GateCase(name: "offline skips", autoSubmit: true, contribute: true, online: false, hasRegistration: true, expected: false),
        GateCase(name: "not opted in", autoSubmit: true, contribute: false, online: true, hasRegistration: true, expected: false),
        GateCase(name: "opt-in unset", autoSubmit: true, contribute: nil, online: true, hasRegistration: true, expected: false),
        GateCase(name: "autoSubmit off (headless)", autoSubmit: false, contribute: true, online: true, hasRegistration: true, expected: false),
        GateCase(name: "no registration", autoSubmit: true, contribute: true, online: true, hasRegistration: false, expected: false),
    ])
    fileprivate func gateRequiresEveryCondition(_ testCase: GateCase) {
        let got = JobExecutor.shouldAutoSubmit(
            manifest(contributeResults: testCase.contribute),
            autoSubmit: testCase.autoSubmit,
            online: testCase.online,
            registration: testCase.hasRegistration ? registrationData() : nil
        )
        #expect(got == testCase.expected)
    }
}
