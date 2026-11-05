import Foundation

/// Storage helpers that need no data root, so not `PipetteWorkspace` members.
nonisolated enum LocalStorage {
    static let defaultContributeResultsKey = "defaultContributeResults"
    static let lastReconciledCollectorKey = "lastReconciledCollector"
    static let lastUndecodableJobSweepBuildKey = "lastUndecodableJobSweepBuild"
    static let analyticsOptOutKey = "analyticsOptOut"

    /// Lets a refresh skip the whole-store scan when the collector hasn't changed since the
    /// last reconcile (`SettingsView.resendResultsFromOtherCollectors`).
    static var lastReconciledCollector: String? {
        get { UserDefaults.standard.string(forKey: lastReconciledCollectorKey) }
        set { UserDefaults.standard.set(newValue, forKey: lastReconciledCollectorKey) }
    }

    /// The app build that last ran `discardUndecodableJobs()`, so a launch on the same build
    /// skips that whole-store decode.
    static var lastUndecodableJobSweepBuild: String? {
        get { UserDefaults.standard.string(forKey: lastUndecodableJobSweepBuildKey) }
        set { UserDefaults.standard.set(newValue, forKey: lastUndecodableJobSweepBuildKey) }
    }

    @discardableResult
    static func markExcludedFromBackup(_ url: URL) -> Bool {
        var mutableURL = url
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        do {
            try mutableURL.setResourceValues(resourceValues)
            return true
        } catch {
            return false
        }
    }

    /// The authority on the opt-out, seeded into `PostHogConfig.optOut` before `setup`. Not
    /// the SDK's own persisted copy: on posthog-android 3.19.0 its restore was measured not to
    /// work (Android's `AnalyticsOptOutStore`), and iOS takes the same defensive shape
    /// unmeasured. Absent means collecting.
    static var analyticsOptOut: Bool {
        get { UserDefaults.standard.bool(forKey: analyticsOptOutKey) }
        set { UserDefaults.standard.set(newValue, forKey: analyticsOptOutKey) }
    }

    static var defaultContributeResults: Bool {
        get {
            // On by default (the payload is performance-only). Keep in sync with
            // SettingsView's @AppStorage default.
            if UserDefaults.standard.object(forKey: defaultContributeResultsKey) == nil {
                return true
            }
            return UserDefaults.standard.bool(forKey: defaultContributeResultsKey)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: defaultContributeResultsKey)
        }
    }

    // MARK: - Per-download keys

    /// The per-transfer download key, repo-qualified so two repos publishing the same
    /// filename get distinct rows. Not a storage path: that is the `ModelStorageKey` entry.
    static func modelRelativePath(repo: String?, filename: String) -> String {
        if let repo, !repo.isEmpty { return "\(repo)/\(filename)" }
        return filename
    }

    // MARK: - Model / benchmark metadata helpers

    /// Q*, F*, BF*, IQ* tokens identify gguf quant/precision suffixes.
    static func isQuantToken(_ s: String) -> Bool {
        let u = s.uppercased()
        return u.hasPrefix("Q") || u.hasPrefix("F") || u.hasPrefix("BF") || u.hasPrefix("IQ")
    }

    static func parseQuant(from filename: String) -> String? {
        let stem = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        for part in stem.split(separator: "-").reversed() where isQuantToken(String(part)) {
            return String(part)
        }
        return nil
    }

    /// Takes a repo leaf, not an `org/repo` slug. Case-insensitive because publishers
    /// disagree (`prism-ml` lowercases where most uppercase).
    static func repoStem(_ repoName: String) -> String {
        repoName.lowercased().hasSuffix("-gguf")
            ? String(repoName.dropLast("-GGUF".count)) : repoName
    }

    /// `LFM2.5-350M-Q4_0.gguf` to `LFM2.5-350M`.
    static func modelStem(from filename: String) -> String {
        let stem = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        let parts = stem.split(separator: "-")
        if let last = parts.last, isQuantToken(String(last)) {
            return parts.dropLast().joined(separator: "-")
        }
        return stem
    }

    /// `350M`, `1.5B`; rejects a version like `2.5`, which has no unit.
    static func isParamToken(_ s: String) -> Bool {
        guard let unit = s.last, "MmBb".contains(unit) else { return false }
        let num = s.dropLast()
        return num.contains(where: \.isNumber)
            && num.allSatisfy { $0.isNumber || $0 == "." }
    }

    /// The parameter count, not the byte size. The HF repo is preferred as more canonical.
    static func parseParamSize(name: String, hfRepo: String?) -> String? {
        for source in [hfRepo, name].compactMap({ $0 }) {
            let stem = source.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
            for part in stem.split(whereSeparator: { $0 == "-" || $0 == "/" }) where isParamToken(String(part)) {
                return String(part).uppercased()
            }
        }
        return nil
    }

    /// Pairs base models with their mmproj files across quant variants.
    static func normalizedModelStem(_ filename: String) -> String {
        var base = filename.replacingOccurrences(of: ".gguf", with: "", options: .caseInsensitive)
        if base.lowercased().hasPrefix("mmproj-") {
            base = String(base.dropFirst("mmproj-".count))
        }
        let parts = base.split(separator: "-")
        if let last = parts.last, isQuantToken(String(last)) {
            return parts.dropLast().joined(separator: "-").lowercased()
        }
        return base.lowercased()
    }

    /// Recovers a stale `storedPath` under `modelsRoot`.
    static func resolveModelPath(_ storedPath: String, modelsRoot: URL) -> String? {
        let fm = FileManager.default
        let comps = storedPath.split(separator: "/").map(String.init)
        guard let filename = comps.last, !filename.isEmpty else { return nil }
        if let modelsIdx = comps.lastIndex(of: "models"), modelsIdx + 1 < comps.count {
            let tail = comps[(modelsIdx + 1)...].joined(separator: "/")
            let candidate = modelsRoot.appendingPathComponent(tail).path
            if fm.fileExists(atPath: candidate) { return candidate }
        }
        // By name, refusing to guess when two repos hold it: the other repo's bytes would
        // mislabel the result's provenance.
        let matches = modelURLs(named: filename, under: modelsRoot)
        return matches.count == 1 ? matches[0].path : nil
    }

    private static func modelURLs(named filename: String, under root: URL) -> [URL] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return [] }
        var matches: [URL] = []
        for case let url as URL in enumerator
        where url.pathExtension.lowercased() == "gguf" && url.lastPathComponent == filename {
            matches.append(url)
        }
        return matches
    }
}
