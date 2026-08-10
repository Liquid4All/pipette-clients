import Testing

@testable import Pipette

/// The run's terminal line, which is the only status `pipette-plan` can see: `devicectl`
/// does not pass the app's exit code through, and its `scan_sentinel` reads the first
/// token after `BENCH_DONE` as an integer — so a bare sentinel reads there as 0.
///
/// A refused invocation used to print exactly that, and the runner recorded the cell as
/// done: never retried, and indistinguishable from one that measured.
@Suite struct HeadlessSentinelTests {

    /// The three exit codes a run can end on, each carried by the line rather than only by
    /// the process.
    @Test(arguments: [(HeadlessExit.ok, "BENCH_DONE 0"),
                      (HeadlessExit.failure, "BENCH_DONE 1"),
                      (HeadlessExit.usage, "BENCH_DONE 2")])
    func theLineCarriesTheExitCode(_ code: Int32, _ expected: String) {
        #expect(HeadlessRunner.benchDoneLine(code) == expected)
    }

    /// A `Bool`-reporting handler maps onto the same two codes.
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
}
