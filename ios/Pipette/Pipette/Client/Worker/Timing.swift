import Foundation

// Claim-loop timing, with no I/O: `pipette-cli/src/client/worker/timing.rs`. The ISO 8601
// parse lives here because only a claim's `time_window` has that shape.

/// Idle wait after a `204` claim (the mgmt repo's `docs/client-integration.md` §5).
nonisolated let defaultIdleWait: TimeInterval = 5 * 60
/// Jitter added to the idle wait, `0...this`.
nonisolated let defaultIdleJitter: TimeInterval = 60
/// The window assumed when a claim's `time_window` does not parse.
private nonisolated let fallbackTimeWindow: TimeInterval = 10 * 60

/// Parse the `PTnHnMnS` duration the server emits for `time_window` (no days, weeks or
/// months). As strict as the crate's: the prefix is `PT` or `pt`, never a mix, and only
/// ASCII digits count (`Character.isNumber` would take `٣` or `½`).
nonisolated func parseIso8601Duration(_ raw: String) -> TimeInterval? {
    let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    let rest: Substring
    if s.hasPrefix("PT") || s.hasPrefix("pt") {
        rest = s.dropFirst(2)
    } else {
        return nil
    }
    guard !rest.isEmpty else { return nil }
    var total: UInt64 = 0
    var num: UInt64 = 0
    var sawDigit = false
    for ch in rest.unicodeScalars {
        switch ch {
        case "0"..."9":
            sawDigit = true
            let digit = UInt64(ch.value - UnicodeScalar("0").value)
            num = saturatingAdd(saturatingMultiply(num, 10), digit)
        case "H", "h", "M", "m", "S", "s":
            guard sawDigit else { return nil }
            let unit: UInt64 = switch ch {
            case "H", "h": 3600
            case "M", "m": 60
            default: 1
            }
            total = saturatingAdd(total, saturatingMultiply(num, unit))
            num = 0
            sawDigit = false
        default:
            return nil
        }
    }
    if sawDigit { return nil }
    return TimeInterval(total)
}

/// The crate's `resolve_heartbeat_interval`: half of `time_window`, or 1 s only when that
/// half is zero. A `PT1S` window heartbeats every 0.5 s; the floor only stops a spin.
/// An override longer than half the window risks the lease lapsing between ticks, but is
/// honoured so an operator can slow heartbeats deliberately (the worker warns).
nonisolated func resolveHeartbeatInterval(timeWindow: String,
                                     overrideSecs: TimeInterval? = nil) -> TimeInterval {
    if let overrideSecs { return max(1, overrideSecs) }
    let half = (parseIso8601Duration(timeWindow) ?? fallbackTimeWindow) / 2
    return half == 0 ? 1 : half
}

/// The crate's `idle_wait_with_jitter`.
nonisolated func idleWaitWithJitter(base: TimeInterval, jitter: TimeInterval) -> TimeInterval {
    base + jittered(jitter)
}

/// Run `attempt` once per entry of `attempts` until it answers non-nil, sleeping
/// ``exponentialBackoff(attempt:first:cap:)`` between tries: the crate's
/// `retry_with_backoff`. `nil` when the attempts run out.
nonisolated func retryWithBackoff<T>(
    attempts: some Sequence<Int>,
    first: TimeInterval,
    cap: TimeInterval,
    attempt: (Int) async -> T?,
    sleep: (TimeInterval) async -> Void
) async -> T? {
    for n in attempts {
        if let answer = await attempt(n) { return answer }
        await sleep(exponentialBackoff(attempt: n, first: first, cap: cap))
    }
    return nil
}

private nonisolated func saturatingAdd(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let (sum, overflow) = a.addingReportingOverflow(b)
    return overflow ? .max : sum
}

private nonisolated func saturatingMultiply(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let (product, overflow) = a.multipliedReportingOverflow(by: b)
    return overflow ? .max : product
}
