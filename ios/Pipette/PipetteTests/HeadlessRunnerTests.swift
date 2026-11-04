import Foundation
import Testing

@testable import Pipette

/// The run's terminal line is the only status `pipette-ctl` can see: `devicectl` does not
/// pass the app's exit code through, and `scan_sentinel` reads the first token after
/// `BENCH_DONE` as the code, so a bare sentinel reads as 0 and the cell as measured.
@Suite struct HeadlessRunnerTests {

    @Test(arguments: [(HeadlessExit.ok, "BENCH_DONE 0"),
                      (HeadlessExit.failure, "BENCH_DONE 1"),
                      (HeadlessExit.usage, "BENCH_DONE 2")])
    func theLineCarriesTheExitCode(_ code: Int32, _ expected: String) {
        #expect(HeadlessRunner.benchDoneLine(code) == expected)
    }

    /// Same literal as `pipette-ios-proxy`'s `the_launch_marker` case: pins the prefix on both sides.
    @Test func theLaunchMarkerNamesTheProcess() {
        #expect(HeadlessRunner.diagnosticLine(HeadlessRunner.helloLine(pid: 4242))
            == "[HEADLESS] hello pid=4242\n")
    }

    @Test func aBoolHandlerReportsOkOrFailure() {
        #expect(HeadlessRunner.benchDoneLine(ok: true) == "BENCH_DONE 0")
        #expect(HeadlessRunner.benchDoneLine(ok: false) == "BENCH_DONE 1")
    }

    /// Detail rides after the code, where upstream's scanner ignores it: it reads the
    /// first token only, so a job status stays legible without breaking the parse.
    @Test func detailFollowsTheCode() {
        #expect(HeadlessRunner.benchDoneLine(HeadlessExit.failure, "cancelled")
            == "BENCH_DONE 1 cancelled")
    }

    /// A claimed run is one cell, so its error is the run's, on one line for
    /// `pipette-ios-proxy` to file. An unclaimed sweep reports its job's status.
    @Test func aClaimedRunFailsWithItsCell() {
        func line(_ status: JobStatus, _ failure: String?, claimed: Bool) -> String {
            let done = HeadlessRunner.sweepDone(status: status, cellFailure: failure,
                                                claimed: claimed)
            return HeadlessRunner.benchDoneLine(done.code, done.detail)
        }
        #expect(line(.completed, "Engine backend failed: prefill decode rc=-3.", claimed: true)
            == "BENCH_DONE 1 failed: Engine backend failed: prefill decode rc=-3.")
        #expect(line(.completed, "Engine error: load failed:\nunknown model architecture",
                     claimed: true)
            == "BENCH_DONE 1 failed: Engine error: load failed: unknown model architecture")
        #expect(line(.completed, nil, claimed: true) == "BENCH_DONE 0 completed")
        #expect(line(.completed, "Engine error: x.", claimed: false) == "BENCH_DONE 0 completed")
        #expect(line(.paused, nil, claimed: true) == "BENCH_DONE 1 paused")
    }

    /// Only a UI launch and the resident `worker` drain results at start. Every other launch
    /// runs alone, as `pipette-ios-proxy`'s verbs between a claim and its submit do.
    @Test(arguments: [
        (["/app/Pipette"], false),
        (["/app/Pipette", "headlessrun", "worker"], false),
        (["/app/Pipette", "headlessrun", "settings", "run"], false),
        (["/app/Pipette", "headlessrun", "results", "list"], true),
        (["/app/Pipette", "headlessrun", "runtimes", "list"], true),
        (["/app/Pipette", "headlessrun", "auth", "me", "local=1"], true),
        (["/app/Pipette", "headlessrun", "status"], true),
        // Refused, so it exits before anything else.
        (["/app/Pipette", "headlessrun", "no-such-verb"], true),
    ])
    func onlyAResidentLaunchActsAsTheClient(_ args: [String], _ oneShot: Bool) {
        #expect(HeadlessRunner.isOneShotLaunch(args) == oneShot)
    }

    /// The launch the proxy measures a memory cell in. Parsed first, so the refusal arm
    /// cannot be what answers.
    @Test func theProxysMemoryCellLaunchIsOneShot() throws {
        let runtime = try String(
            decoding: JSONEncoder().encode(#require(Runtime.thisBuild(for: .mlxIosPipette))),
            as: UTF8.self)
        let model = #"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#
        let args = ["/app/Pipette", "headlessrun", "runtime=\(runtime)", "model=\(model)",
                    "benchmarks=max_memory_usage_512", "submit=0"]
        _ = try HeadlessCommand.parse(args).get()
        #expect(HeadlessRunner.isOneShotLaunch(args))
    }
}
