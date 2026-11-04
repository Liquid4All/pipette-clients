import Foundation

// Port of `crates/pipette-readiness/src/callbacks.rs`.

/// Which gate produced a `ReadinessStatus`: this client's own, or the simulator's, which
/// has no probes (`other.rs`).
nonisolated enum ReadinessPlatform: Sendable, Equatable, CustomStringConvertible {
    case ios
    case other

    func asStr() -> String {
        switch self {
        case .ios: "ios"
        case .other: "other"
        }
    }

    var description: String { asStr() }
}

nonisolated enum ReadinessPhase: Sendable, Equatable {
    /// `pollIn` is the topped-up remainder of the poll interval, not the whole interval.
    case waiting(pollIn: Duration)
    case proceeding
    case timedOut

    /// Recorded with the wait, so it must match the crate's `ReadinessPhase::as_str`.
    func asStr() -> String {
        switch self {
        case .waiting: "waiting"
        case .proceeding: "proceeding"
        case .timedOut: "timed_out"
        }
    }
}

/// One poll of a readiness wait. `description` derives the log line's closing phrase from
/// `phase` rather than storing it.
nonisolated struct ReadinessStatus: Sendable, Equatable, CustomStringConvertible {
    var phase: ReadinessPhase
    var platform: ReadinessPlatform
    /// The platform's summary of this poll's readings. A string, not a type: the vocabulary
    /// is per platform.
    var observed: String
    /// Measured from before the first probe.
    var elapsed: Duration
    var maxWait: Duration

    /// The only phase worth a live line.
    func isWaiting() -> Bool {
        if case .waiting = phase { true } else { false }
    }

    /// `readiness: <platform> <observed> → proceeding`, `→ waiting 1.8s`,
    /// `→ timed out after 300s`: the crate's line, durations as Rust prints them. Keep the
    /// phrases verbatim: `pipette-ios-proxy` keys its engine clock on them (`READINESS` in
    /// `tools/pipette-ios-proxy/src/app/mod.rs`).
    var description: String {
        let action = switch phase {
        case let .waiting(pollIn): "waiting \(rustDebug(pollIn))"
        case .proceeding: "proceeding"
        case .timedOut: "timed out after \(rustDebug(maxWait))"
        }
        return "readiness: \(platform) \(observed) → \(action)"
    }
}

extension ReadinessStatus: WaitStatus {
    /// Used when the caller passes no status handler.
    nonisolated func reportDefault() {
        AppLog.jobRun.info("\(self)")
    }
}

typealias ReadinessCallbacks = WaitCallbacks<ReadinessStatus>

/// Rust's `{:?}` for a `Duration`: the largest unit that keeps the integer part nonzero,
/// and the fraction to the nanosecond without trailing zeros.
nonisolated private func rustDebug(_ duration: Duration) -> String {
    let (seconds, attoseconds) = duration.components
    let nanos = UInt64(max(0, attoseconds) / 1_000_000_000)
    func decimal(_ whole: UInt64, _ fraction: UInt64, width: Int, _ unit: String) -> String {
        guard fraction > 0 else { return "\(whole)\(unit)" }
        var digits = String(fraction)
        digits = String(repeating: "0", count: width - digits.count) + digits
        while digits.hasSuffix("0") { digits.removeLast() }
        return "\(whole).\(digits)\(unit)"
    }
    if seconds > 0 { return decimal(UInt64(seconds), nanos, width: 9, "s") }
    if nanos >= 1_000_000 { return decimal(nanos / 1_000_000, nanos % 1_000_000, width: 6, "ms") }
    if nanos >= 1_000 { return decimal(nanos / 1_000, nanos % 1_000, width: 3, "µs") }
    return "\(nanos)ns"
}
