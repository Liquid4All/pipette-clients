import Foundation

/// Shared date formatters: one definition each so a date renders the same on every
/// screen (and in the same time zone), and no per-call allocation.
///
/// `nonisolated(unsafe)` for the non-`Sendable` formatters: configured once and only
/// read afterwards, which is thread-safe in practice.
enum JobDateFormat {
    nonisolated(unsafe) static let relative: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    nonisolated(unsafe) static let iso8601 = ISO8601DateFormatter()

    /// `DateFormatter` is `Sendable`, so no `unsafe` here.
    nonisolated static let shortDate: DateFormatter = {
        let f = DateFormatter()
        f.calendar = Calendar(identifier: .gregorian)
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = .current
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
}
