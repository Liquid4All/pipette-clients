import Foundation

/// Parse server `time_window` values like `PT10M` / `PT1H2M3S` into seconds.
enum Iso8601Duration {
    static func seconds(from raw: String) -> TimeInterval? {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard s.uppercased().hasPrefix("PT"), s.count > 2 else { return nil }
        var total: TimeInterval = 0
        var num = 0.0
        var sawDigit = false
        for ch in s.dropFirst(2) {
            if ch.isNumber {
                sawDigit = true
                num = num * 10 + TimeInterval(ch.wholeNumberValue ?? 0)
            } else if sawDigit {
                switch ch {
                case "H", "h": total += num * 3600
                case "M", "m": total += num * 60
                case "S", "s": total += num
                default: return nil
                }
                num = 0
                sawDigit = false
            } else {
                return nil
            }
        }
        if sawDigit { return nil }
        return total
    }

    /// Half of `time_window`, floored at 1 s (protocol default heartbeat cadence).
    static func heartbeatInterval(timeWindow: String, overrideSeconds: TimeInterval? = nil) -> TimeInterval {
        if let overrideSeconds { return max(1, overrideSeconds) }
        let window = seconds(from: timeWindow) ?? 600
        return max(1, window / 2)
    }
}
