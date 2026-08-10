import Foundation

/// The `storage` group — `status` and `gc`, as the crate has them.
enum StorageCommands {
    /// `storage status`: the quota the sweeper enforces, and what is against it.
    static func status(storage: Storage) async {
        let used = storage.storageUsageBytes()
        let limit = storage.storageQuotaBytes
        let models = await MainActor.run { storage.availableModels().count }
        // `remaining` clamps at zero: an over-limit store is reported by `over`, and a
        // negative "remaining" would read as a smaller overage than it is.
        HeadlessRunner.log("storage usedBytes=\(used) limitBytes=\(limit) "
            + "remainingBytes=\(max(0, limit - used)) over=\(used > limit ? "yes" : "no") "
            + "models=\(models)")
    }

    /// `storage gc`: reclaim disk — garbage first, then least-recently-used artifacts.
    ///
    /// Where upstream plans against no pins at all ("nothing is in flight during `gc`"),
    /// this pins what a resident app can have in flight: every cell of a running or paused
    /// job, and every model a transfer is still writing. A CLI process cannot be mid-run
    /// while it runs `gc`; this one can, and reclaiming a model out from under a live job
    /// would turn it into a re-download.
    ///
    /// `pins` is injectable so a test can state them; production reads the coordinator.
    ///
    /// `freed` counts quota-accounted bytes only. Hub-cache orphans are reclaimed too and
    /// appear in the per-entry lines, but they sit outside the store root, so adding them
    /// would inflate a figure measured against the cap.
    static func gc(dryRun: Bool, storage: Storage, pins injected: SweepPins? = nil) async -> Bool {
        let pins = if let injected { injected } else {
            await MainActor.run { DownloadCoordinator.shared.sweepPins(justInstalled: nil) }
        }
        let limit = storage.storageQuotaBytes
        // Planned once and, on the reclaiming path, carried out as-is — upstream surveys
        // once and hands that plan to `apply_sweep`. Re-planning inside the sweep would
        // execute a different plan than the one reported, and re-walk the store to do it.
        let plan = storage.sweepPlan(pinning: pins)

        let freed: Int64
        if dryRun {
            guard !plan.evictions.isEmpty || plan.stillOverByBytes != nil else {
                return nothingToReclaim(plan.usedBytes, limit)
            }
            for entry in plan.evictions { logEntry("would-reclaim", entry) }
            freed = plan.freedBytes
        } else {
            // Run even on an empty plan: the plan covers the store root, while hub-cache
            // orphans sit outside it and are reclaimed by `reclaim` alone. Upstream can
            // return early here because its survey already covers everything it reclaims;
            // ours does not, and a store under quota with a stranded pull is the ordinary
            // case for those bytes.
            let report = storage.reclaim(plan, pinning: pins)
            guard !report.removed.isEmpty || !report.failed.isEmpty
                    || plan.stillOverByBytes != nil else {
                return nothingToReclaim(plan.usedBytes, limit)
            }
            for entry in report.removed { logEntry("reclaimed", entry) }
            for (entry, reason) in report.failed {
                HeadlessRunner.log(
                    "storage gc could-not-reclaim path=\(entry.path.path) reason=\(reason)")
            }
            freed = report.freedBytes
        }

        // Against what the sweep left rather than what the plan predicted: a failed unlink
        // is tolerated, so a plan that expected to fit can still leave the store over —
        // and `stillOverByBytes` would say nothing about it. On a dry run this is the
        // projection, which is why it is named apart from `status`'s factual `usedBytes`.
        let left = max(0, plan.usedBytes - freed)
        let over = max(0, left - limit)
        HeadlessRunner.log("storage gc \(dryRun ? "wouldFreeBytes" : "freedBytes")=\(freed) "
            + "usedBytes=\(plan.usedBytes) limitBytes=\(limit) "
            + "\(dryRun ? "wouldLeaveBytes" : "remainingBytes")=\(left)"
            + (over > 0 ? " stillOverBytes=\(over)" : ""))
        return true
    }

    private static func nothingToReclaim(_ used: Int64, _ limit: Int64) -> Bool {
        HeadlessRunner.log("storage gc nothing to reclaim usedBytes=\(used) limitBytes=\(limit)")
        return true
    }

    private static func logEntry(_ verb: String, _ entry: StorageEntry) {
        HeadlessRunner.log("storage gc \(verb) path=\(entry.path.path) bytes=\(entry.sizeBytes)")
    }
}
