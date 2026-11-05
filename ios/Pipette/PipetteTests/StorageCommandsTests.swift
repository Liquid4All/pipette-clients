import Foundation
import Testing

@testable import Pipette

/// `storage status` and `storage gc` output, as pipette-cli's `commands/storage.rs` prints it.
///
/// Every key is the crate's or the wire's `snake_case` spelling, so a token greps the same
/// in both outputs. Swift converts its `camelCase` names at the print, and nothing but a test
/// stops a property name leaking into a line.
struct StorageCommandsTests {

    private let lastUsed = Date(timeIntervalSince1970: 1_767_225_600)

    private let garbage = StorageEntry(
        path: URL(fileURLWithPath: "/models/junk"), label: "junk", sizeBytes: 2048,
        kind: .garbage(reason: "no readable manifest"))

    private func model(size: Int64) throws -> StorageEntry {
        let key = try #require(ModelStorageKey.of(ggufTextFixture("org/r-GGUF", "r-Q4_0.gguf")))
        return StorageEntry(path: URL(fileURLWithPath: "/models/\(key.value)"), label: key.value,
                            sizeBytes: size,
                            kind: .model(key: key, lastUsedAt: lastUsed, fetchedAt: lastUsed))
    }

    /// The keys in a `<group> k=v k=v` line, in order.
    private func keys(of line: String) -> [String] {
        line.split(separator: " ").compactMap { token in
            guard let eq = token.firstIndex(of: "=") else { return nil }
            return String(token[..<eq])
        }
    }

    private func expectAllSnakeCase(_ lines: [String]) {
        for line in lines {
            for key in keys(of: line) {
                #expect(key == key.lowercased() && !key.contains("-"), "\(key) in \(line)")
            }
        }
    }

    // MARK: - the resolved quota

    /// `quota_source` has to come off the same resolution as the number beside it. A
    /// settings file holding a quota that was refused falls back to the default, and
    /// crediting the file for the default's value misreports where the cap came from.
    @Test(arguments: [Int64(0), -1])
    func aRefusedSettingsQuotaReportsTheDefaultAsItsSource(_ configured: Int64) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.settings.put(ClientSettings(storageQuotaBytes: configured))

        let quota = ws.storageQuota

        #expect(quota.source == .default)
        #expect(quota.bytes == ws.defaultStorageQuotaBytes)
    }

    @Test func aConfiguredQuotaIsReportedAsComingFromSettings() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(4096)

        #expect(ws.storageQuota == StorageQuota(bytes: 4096, source: .settings))
    }

    @Test func noSettingsFileIsTheBuiltInDefault() {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        #expect(ws.storageQuota.source == .default)
        #expect(QuotaSource.default.label == "built-in default")
        #expect(QuotaSource.settings.label == "identity/settings.json")
    }

    // MARK: - status

    /// The crate's `quota:` / `used:` / `free:` header, then one row per entry in the
    /// order `gc` would reclaim them: garbage first, with its reason as the note.
    @Test func statusReportsTheQuotaTheUsageAndEveryEntry() throws {
        let model = try model(size: 4096)
        let lines = StorageCommands.statusLines(
            quota: 10_000, quotaSource: "settings",
            survey: StorageSurvey(entries: [model, garbage]))

        #expect(lines == [
            "storage quota_bytes=10000 quota_source=settings used_bytes=6144 "
                + "reserved_bytes=0 used_percent=61 free_bytes=3856",
            "storage count=2",
            "storage entry kind=garbage entry=junk size_bytes=2048 last_used=- "
                + "note=no readable manifest",
            "storage entry kind=model entry=\(model.label) size_bytes=4096 "
                + "last_used=\(JobDateFormat.iso8601.string(from: lastUsed))",
        ])
        expectAllSnakeCase(lines)
    }

    /// `free_bytes` saturates as the crate's does, and the share reads past 100: an
    /// over-quota store is reported as over rather than as a smaller overage.
    @Test func anOverQuotaStoreHasNoFreeBytes() {
        let over = StorageEntry(path: URL(fileURLWithPath: "/models/big"), label: "big",
                                sizeBytes: 250, kind: .garbage(reason: "staging orphan"))
        let lines = StorageCommands.statusLines(quota: 100, quotaSource: "default",
                                                survey: StorageSurvey(entries: [over]))
        #expect(lines.first == "storage quota_bytes=100 quota_source=default used_bytes=250 "
            + "reserved_bytes=0 used_percent=250 free_bytes=0")
    }

    /// Bytes promised to a running transfer count against the quota exactly as stored
    /// bytes do, because that is what the next admission will do. Reporting them as free
    /// would have the status line contradict the refusal a download is about to get.
    @Test func reservedBytesCountAgainstTheQuota() {
        let lines = StorageCommands.statusLines(quota: 1000, quotaSource: "settings",
                                                survey: StorageSurvey(entries: []),
                                                reservedBytes: 400)
        #expect(lines.first == "storage quota_bytes=1000 quota_source=settings used_bytes=0 "
            + "reserved_bytes=400 used_percent=40 free_bytes=600")
        expectAllSnakeCase(lines)
    }

    /// A zero quota, which nothing lets through, reads full instead of dividing by zero
    /// (the crate's `percent_of_reads_as_a_share_of_the_quota`).
    @Test(arguments: [
        (Int64(0), Int64(100), "used_percent=0"),
        (50, 100, "used_percent=50"),
        (150, 100, "used_percent=150"),
        (10, 0, "used_percent=100"),
    ])
    func usedPercentReadsAsAShareOfTheQuota(used: Int64, quota: Int64, expected: String) {
        let entries = used > 0
            ? [StorageEntry(path: URL(fileURLWithPath: "/models/x"), label: "x",
                            sizeBytes: used, kind: .garbage(reason: "staging orphan"))]
            : []
        let header = StorageCommands.statusLines(quota: quota, quotaSource: "default",
                                                 survey: StorageSurvey(entries: entries)).first
        #expect(header?.split(separator: " ").contains { $0 == expected } == true, "\(header ?? "")")
    }

    @Test func anEmptyStoreListsNoEntries() {
        #expect(StorageCommands.statusLines(quota: 100, quotaSource: "default",
                                            survey: StorageSurvey(entries: []))
            == ["storage quota_bytes=100 quota_source=default used_bytes=0 "
                    + "reserved_bytes=0 used_percent=0 free_bytes=100",
                "storage count=0"])
    }

    // MARK: - gc

    @Test func aDryRunLineReadsAsAPlanAndARealOneAsAFact() throws {
        #expect(StorageCommands.evictionLine(garbage, dryRun: true)
            == "storage gc would_reclaim kind=garbage entry=junk size_bytes=2048 "
                + "reason=no readable manifest")
        #expect(StorageCommands.evictionLine(garbage, dryRun: false)
            == "storage gc reclaimed kind=garbage entry=junk size_bytes=2048 "
                + "reason=no readable manifest")

        let model = try model(size: 4096)
        let stamp = JobDateFormat.iso8601.string(from: lastUsed)
        #expect(StorageCommands.evictionLine(model, dryRun: true)
            == "storage gc would_evict kind=model entry=\(model.label) size_bytes=4096 "
                + "last_used=\(stamp)")
        #expect(StorageCommands.evictionLine(model, dryRun: false)
            == "storage gc evicted kind=model entry=\(model.label) size_bytes=4096 "
                + "last_used=\(stamp)")
    }

    /// `freed X; using Y of Q`, and `would free X; leaving Y of Q` on a dry run: the
    /// figures a dry run projects are named apart from the ones a sweep reports.
    @Test func theSummaryReportsWhatTheSweepLeft() {
        let projected = StorageCommands.gcSummaryLine(freed: 40, used: 100, quota: 80,
                                                      dryRun: true)
        let actual = StorageCommands.gcSummaryLine(freed: 40, used: 100, quota: 80,
                                                   dryRun: false)
        #expect(projected == "storage gc would_free_bytes=40 would_leave_bytes=60 quota_bytes=80")
        #expect(actual == "storage gc freed_bytes=40 used_bytes=60 quota_bytes=80")
        expectAllSnakeCase([projected, actual])
    }

    /// The crate's `warns_from_what_the_sweep_left`. Absent when the store fits: an operator
    /// greps for the key's presence, so a `still_over_bytes=0` would read as a failure.
    @Test(arguments: [(Int64(12_000), Int64(10_000), Int64?(2_000)),
                      (9_000, 10_000, nil),
                      (10_000, 10_000, nil)])
    func warnsFromWhatTheSweepLeft(remaining: Int64, quota: Int64, over: Int64?) {
        let line = StorageCommands.gcSummaryLine(freed: 0, used: remaining, quota: quota,
                                                 dryRun: false)
        if let over {
            #expect(line.hasSuffix(" still_over_bytes=\(over)"))
        } else {
            #expect(!line.contains("still_over_bytes"))
        }
    }
}
