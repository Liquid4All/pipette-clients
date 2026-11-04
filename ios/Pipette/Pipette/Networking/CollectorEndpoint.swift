import Foundation

enum CollectorEndpoint {
    nonisolated static let productionURL = "https://collector.pipette.liquid.ai"

    /// The registration wins: its identity signs for that server alone.
    nonisolated static func effectiveServerUrl(registered: ServerUrl?) -> ServerUrl? {
        registered ?? (try? ServerUrl(productionURL))
    }

    /// Compares normalized forms, so a trailing slash or omitted scheme isn't a
    /// mismatch. A nil or blank `stored` (a submission with no recorded
    /// collector) counts as a different collector, so it re-syncs.
    static func isSameCollector(_ stored: String?, as current: String) -> Bool {
        guard let stored, !stored.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return false }
        let normalizedStored = normalizedCustomURL(stored) ?? stored.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedCurrent = normalizedCustomURL(current) ?? current.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalizedStored == normalizedCurrent
    }

    static func normalizedCustomURL(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let candidate = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        guard var components = URLComponents(string: candidate),
              let scheme = components.scheme?.lowercased(),
              scheme == "https",
              let host = components.host,
              !host.isEmpty,
              components.user == nil,
              components.password == nil,
              components.query == nil,
              components.fragment == nil
        else {
            return nil
        }

        components.scheme = scheme
        // DNS is case-insensitive: two spellings of one host are one collector.
        components.host = host.lowercased()
        components.path = components.path.trimmingTrailingSlashes()

        return components.url?.absoluteString
    }
}

enum CollectorEndpointOption: String, CaseIterable, Identifiable {
    case production
    case custom

    var id: String { rawValue }

    var title: String {
        switch self {
        case .production:
            return "Liquid AI"
        case .custom:
            return "Custom"
        }
    }

    func serverURL(customURL: String) -> String? {
        switch self {
        case .production:
            return CollectorEndpoint.productionURL
        case .custom:
            return CollectorEndpoint.normalizedCustomURL(customURL)
        }
    }
}

private extension String {
    func trimmingTrailingSlashes() -> String {
        var result = self
        while result.last == "/" {
            result.removeLast()
        }
        return result
    }
}
