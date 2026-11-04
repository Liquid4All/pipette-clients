import Foundation

/// The `storage` group, the crate's `commands/storage.rs` fields as `key=value`.
///
/// `reserved_bytes` is this client's: what admission promises to running transfers.
/// Without it a refusal is unreadable, since the promised bytes are not yet on disk.
enum StorageCommands {
    /// `storage status`: the quota, what is against it, and every entry in the order `gc`
    /// would reclaim them.
    @MainActor
    static func status(ws: PipetteWorkspace) async {
        // `quota_source` comes from the same resolution as the number, so a refused quota
        // (zero, or absent) that fell back to the default reads as the default.
        let quota = ws.storageQuota
        let survey = ws.survey(pinning: SweepPins())
        let reserved = await DownloadCoordinator.shared.admission.outstandingBytes
        for line in statusLines(quota: quota.bytes, quotaSource: quota.source.rawValue,
                                survey: survey, reservedBytes: reserved) {
            HeadlessRunner.log(line)
        }
    }

    /// Split out so the values are assertable without a store or stdout.
    ///
    /// `free_bytes` clamps at zero; an over-quota store shows through `used_percent`, which
    /// passes 100. It counts `reserved_bytes` against the quota, as the next admission
    /// will, so the line does not contradict the refusal a download is about to get.
    static func statusLines(quota: Int64, quotaSource: String,
                            survey: StorageSurvey, reservedBytes: Int64 = 0) -> [String] {
        let committed = survey.usedBytes + reservedBytes
        return ["storage quota_bytes=\(quota) quota_source=\(quotaSource) "
            + "used_bytes=\(survey.usedBytes) reserved_bytes=\(reservedBytes) "
            + "used_percent=\(percentOf(committed, quota)) "
            + "free_bytes=\(max(0, quota - committed))",
         "storage count=\(survey.entries.count)"]
            + survey.entries.map { entryLine($0) }
    }

    /// `note=` is the garbage reason, last because it is prose; a live model prints no key.
    static func entryLine(_ entry: StorageEntry) -> String {
        var line = "storage entry kind=\(kindLabel(entry.kind)) entry=\(entry.label) "
            + "size_bytes=\(entry.sizeBytes) last_used=\(lastUsedLabel(entry.kind))"
        if case let .garbage(reason) = entry.kind { line += " note=\(reason)" }
        return line
    }

    /// `storage gc`: reclaim garbage first, then least-recently-used artifacts.
    ///
    /// Unlike the crate, this pins what a resident app can have in flight (cells of a
    /// running or paused job, models a transfer is writing): reclaiming a model from under
    /// a live job would turn it into a re-download.
    ///
    /// `freed` counts quota-accounted bytes only. Hub-cache orphans are reclaimed and
    /// listed too, but sit outside the store root, so counting them would inflate a figure
    /// measured against the cap.
    static func gc(dryRun: Bool, ws: PipetteWorkspace, pins injected: SweepPins? = nil) async -> Bool {
        let pins = if let injected { injected } else {
            await MainActor.run { DownloadCoordinator.shared.sweepPins(justInstalled: nil) }
        }
        let quota = ws.storageQuotaBytes
        // Planned once and carried out as-is: re-planning inside the sweep would execute a
        // different plan than the one reported.
        let survey = ws.survey(pinning: pins)
        let plan = Pipette.plan(survey, quotaBytes: quota, pins: pins)

        let freed: Int64
        if dryRun {
            guard !plan.evictions.isEmpty || plan.stillOverByBytes != nil else {
                return nothingToReclaim(survey.usedBytes, quota)
            }
            for entry in plan.evictions { HeadlessRunner.log(evictionLine(entry, dryRun: true)) }
            freed = plan.freedBytes
        } else {
            // Run even on an empty plan: hub-cache orphans sit outside the store root the
            // plan covers, and only `reclaim` removes them.
            let report = ws.reclaim(plan, pinning: pins)
            guard !report.removed.isEmpty || !report.failed.isEmpty
                    || plan.stillOverByBytes != nil else {
                return nothingToReclaim(survey.usedBytes, quota)
            }
            for entry in report.removed { HeadlessRunner.log(evictionLine(entry, dryRun: false)) }
            for (entry, reason) in report.failed {
                HeadlessRunner.log("storage gc could_not_reclaim entry=\(entry.label) "
                    + "path=\(entry.path.path) reason=\(reason)")
            }
            freed = report.freedBytes
        }

        HeadlessRunner.log(
            gcSummaryLine(freed: freed, used: survey.usedBytes, quota: quota, dryRun: dryRun))
        return true
    }

    /// One reported eviction, verb first so no delete is silent.
    static func evictionLine(_ entry: StorageEntry, dryRun: Bool) -> String {
        switch entry.kind {
        case let .garbage(reason):
            "storage gc \(dryRun ? "would_reclaim" : "reclaimed") kind=garbage "
                + "entry=\(entry.label) size_bytes=\(entry.sizeBytes) reason=\(reason)"
        case .model:
            "storage gc \(dryRun ? "would_evict" : "evicted") kind=\(kindLabel(entry.kind)) "
                + "entry=\(entry.label) size_bytes=\(entry.sizeBytes) "
                + "last_used=\(lastUsedLabel(entry.kind))"
        }
    }

    /// The dry-run figures are named apart because they are a projection. Measured against
    /// what the sweep freed, not what the plan predicted: a failed unlink is tolerated, so
    /// the store can still be over, which `still_over_bytes` reports.
    static func gcSummaryLine(freed: Int64, used: Int64, quota: Int64, dryRun: Bool) -> String {
        let left = max(0, used - freed)
        let over = max(0, left - quota)
        return "storage gc \(dryRun ? "would_free_bytes" : "freed_bytes")=\(freed) "
            + "\(dryRun ? "would_leave_bytes" : "used_bytes")=\(left) quota_bytes=\(quota)"
            + (over > 0 ? " still_over_bytes=\(over)" : "")
    }

    private static func nothingToReclaim(_ used: Int64, _ quota: Int64) -> Bool {
        HeadlessRunner.log("storage gc nothing to reclaim used_bytes=\(used) quota_bytes=\(quota)")
        return true
    }

    /// No runtime kinds: a phone has no runtime store.
    private static func kindLabel(_ kind: EntryKind) -> String {
        switch kind {
        case .garbage: "garbage"
        case .model: "model"
        }
    }

    /// RFC 3339, or `-` for garbage.
    private static func lastUsedLabel(_ kind: EntryKind) -> String {
        kind.lastUsedAt.map(JobDateFormat.iso8601.string(from:)) ?? "-"
    }

    /// Reads past 100 when the store is over, and 100 for a zero quota, as the crate's does.
    private static func percentOf(_ used: Int64, _ quota: Int64) -> Int64 {
        quota > 0 ? used.multipliedReportingOverflow(by: 100).partialValue / quota : 100
    }
}
