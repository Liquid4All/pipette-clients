import Foundation
import Testing

@testable import Pipette

/// What a public build says about Max Memory accuracy, and when; an internal build says
/// nothing.
@Suite struct MemoryFreshnessTests {
    private func notice(
        _ kind: MemoryFreshness.RunKind, publicBuild: Bool = true, memory: Int, total: Int,
        ranEngineWork: Bool
    ) -> MemoryFreshness.Notice? {
        MemoryFreshness.runNotice(
            kind, publicBuild: publicBuild, memoryCells: memory, totalCells: total,
            ranEngineWork: ranEngineWork)
    }

    @Test func onlyMemoryCellsOnAPublicBuildKeepARecord() {
        #expect(MemoryFreshness.record(for: .maxMemoryUsage, ranEngineWork: true, publicBuild: true) == true)
        #expect(MemoryFreshness.record(for: .maxMemoryUsage, ranEngineWork: false, publicBuild: true) == false)
        #expect(MemoryFreshness.record(for: .vlMaxMemory, ranEngineWork: true, publicBuild: true) == true)
        #expect(MemoryFreshness.record(for: .asrMaxMemory, ranEngineWork: true, publicBuild: true) == true)
        #expect(MemoryFreshness.record(for: .asrThroughput, ranEngineWork: true, publicBuild: true) == nil)
        #expect(MemoryFreshness.record(for: .prefillThroughput, ranEngineWork: true, publicBuild: true) == nil)
        #expect(MemoryFreshness.record(for: nil, ranEngineWork: true, publicBuild: true) == nil)
        #expect(MemoryFreshness.record(for: .maxMemoryUsage, ranEngineWork: true, publicBuild: false) == nil)
    }

    /// Switching back to the app keeps its process, so every instruction names the way out.
    @Test func thePickerNoteSaysHowToStartFresh() {
        #expect(MemoryFreshness.pickerNote.contains("App Switcher"))
    }

    @Test(arguments: [MemoryFreshness.RunKind.newJob, .rerun])
    func nothingToSayOnAnInternalBuildOrWithoutMemoryCells(kind: MemoryFreshness.RunKind) {
        #expect(notice(kind, publicBuild: false, memory: 3, total: 9, ranEngineWork: true) == nil)
        #expect(notice(kind, publicBuild: false, memory: 1, total: 1, ranEngineWork: true) == nil)
        #expect(notice(kind, memory: 0, total: 9, ranEngineWork: true) == nil)
    }

    /// Any other cell in the job may run before a memory cell, so even a lone memory test
    /// among other benchmarks gets the job-of-its-own advice, fresh app or not.
    @Test(arguments: zip([1, 2, 3], [2, 2, 9]))
    func aNewJobWithAnyOtherCellAsksForAJobOfItsOwn(memory: Int, total: Int) throws {
        for ranEngineWork in [false, true] {
            let said = try #require(notice(.newJob, memory: memory, total: total,
                                           ranEngineWork: ranEngineWork))
            #expect(said.isWarning)
            #expect(said.message.contains("a job of its own, with no other benchmarks"))
            #expect(said.message.contains("App Switcher"))
        }
    }

    @Test(arguments: zip([1, 2], [3, 3]))
    func aRerunWithAnyOtherCellAsksForEachMemoryCellOnItsOwn(memory: Int, total: Int) throws {
        let said = try #require(notice(.rerun, memory: memory, total: total, ranEngineWork: false))
        #expect(said.isWarning)
        #expect(said.message.contains("each Max Memory cell on its own (select only that cell)"))
        #expect(said.message.contains("App Switcher"))
    }

    @Test(arguments: [MemoryFreshness.RunKind.newJob, .rerun])
    func aLoneMemoryCellAfterOtherWorkCanReadHigh(kind: MemoryFreshness.RunKind) throws {
        let said = try #require(notice(kind, memory: 1, total: 1, ranEngineWork: true))
        #expect(said.isWarning)
        #expect(said.message.contains("can read high"))
        #expect(said.message.contains("App Switcher"))
        #expect(!said.message.contains("its own"))
    }

    /// The good case: a new job informs without warning, and a rerun needs no question.
    @Test func aLoneFreshMemoryCellInformsANewJobAndLetsARerunThrough() throws {
        let said = try #require(notice(.newJob, memory: 1, total: 1, ranEngineWork: false))
        #expect(!said.isWarning)
        #expect(said.message.contains("if you start it now"))
        #expect(notice(.rerun, memory: 1, total: 1, ranEngineWork: false) == nil)
    }

    @Test func aResultNoteFollowsTheCellsRecord() throws {
        let after = try #require(MemoryFreshness.resultNote(
            type: .maxMemoryUsage, completed: true, measuredAfterOtherWork: true,
            publicBuild: true))
        #expect(after.contains("can read high"))
        #expect(after.contains("App Switcher"))

        #expect(MemoryFreshness.resultNote(
            type: .maxMemoryUsage, completed: true, measuredAfterOtherWork: false,
            publicBuild: true) == nil)
    }

    /// No record: a public build can only say when the number holds.
    @Test func anUnrecordedResultGetsTheConditionOnAPublicBuildOnly() throws {
        let note = try #require(MemoryFreshness.resultNote(
            type: .maxMemoryUsage, completed: true, measuredAfterOtherWork: nil,
            publicBuild: true))
        #expect(note.contains("only if it was the first benchmark"))

        #expect(MemoryFreshness.resultNote(
            type: .maxMemoryUsage, completed: true, measuredAfterOtherWork: nil,
            publicBuild: false) == nil)
    }

    @Test func noResultNoteForOtherKindsOrUnfinishedCells() {
        #expect(MemoryFreshness.resultNote(
            type: .decodeThroughput, completed: true, measuredAfterOtherWork: true,
            publicBuild: true) == nil)
        #expect(MemoryFreshness.resultNote(
            type: .maxMemoryUsage, completed: false, measuredAfterOtherWork: true,
            publicBuild: true) == nil)
    }

    @Test func onlyACompletedMemoryResultMeasuredAfterOtherWorkIsFlagged() {
        func cell(_ type: BenchmarkType, _ status: CellRunStatus, after: Bool?) -> JobCell {
            var cell = JobCell(
                cellId: CellId(UUID().uuidString), benchmarkId: "\(type.rawValue)_512",
                benchmarkType: type, runStatus: status, source: .appleFoundationText)
            cell.measuredAfterOtherWork = after
            return cell
        }
        let flagged = cell(.maxMemoryUsage, .completed, after: true)
        let clean = [
            cell(.maxMemoryUsage, .completed, after: false),
            cell(.maxMemoryUsage, .completed, after: nil),
            cell(.maxMemoryUsage, .failed, after: true),
            cell(.prefillThroughput, .completed, after: true),
        ]

        #expect(MemoryFreshness.flagsResult(flagged))
        #expect(clean.allSatisfy { !MemoryFreshness.flagsResult($0) })
        #expect(MemoryFreshness.tableNote(cells: clean + [flagged]) != nil)
        #expect(MemoryFreshness.tableNote(cells: clean) == nil)
    }
}
