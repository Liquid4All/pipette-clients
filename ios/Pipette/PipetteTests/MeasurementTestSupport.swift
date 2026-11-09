@testable import Pipette

/// For tests that record no series and run under no limit: the crate's
/// `MeasurementObserver::QUIET`. Kept out of the app so no forwarding layer can default to it.
nonisolated func noObserver() -> MeasurementObserver {
    MeasurementObserver(onBeforeRepStart: {}, onAfterRepEnd: {})
}

/// The steps a measurement entry ran, in order, failing the one named: the crate entry
/// tests' `step` closure. A reference so the observer's `@Sendable` hooks can append.
nonisolated final class StepLog: @unchecked Sendable {
    private(set) var events: [String] = []
    let failing: String?

    init(failing: String? = nil) { self.failing = failing }

    func step(_ event: String) throws {
        events.append(event)
        if event == failing { throw StepFailed(event: event) }
    }

    /// An observer whose hooks record `start` and `end`.
    var observer: MeasurementObserver {
        MeasurementObserver(onBeforeRepStart: { try self.step("start") },
                            onAfterRepEnd: { try self.step("end") })
    }
}

/// What `StepLog.step` throws for the step it was told to fail.
nonisolated struct StepFailed: Error, Equatable {
    let event: String
}

/// The failure an attempt raises in `theContextOutlivesAFailedAttempt`.
nonisolated struct ServerWentAway: Error, Equatable {}

/// `Result(catching:)` for an async body.
func outcome<T>(_ body: () async throws -> T) async -> Result<T, any Error> {
    do {
        return .success(try await body())
    } catch {
        return .failure(error)
    }
}

/// What the attempts of one entry run recorded: a reference, as the crate lends a
/// `&mut Vec`.
nonisolated final class Recorded: @unchecked Sendable {
    var attempts: [String] = []
}

/// Literal counts in fixtures. A literal out of range is a bug in the test, so it traps.
extension PositiveCount: @retroactive ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        do {
            try self.init(value)
        } catch {
            preconditionFailure("\(error)")
        }
    }
}

extension NonNegativeCount: @retroactive ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        do {
            try self.init(value)
        } catch {
            preconditionFailure("\(error)")
        }
    }
}

/// Literal ids in fixtures, likewise: an id the grammar refuses is a bug in the test.
extension BenchmarkId: @retroactive ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        do {
            try self.init(value)
        } catch {
            preconditionFailure("\(error)")
        }
    }
}

extension BatchSize: @retroactive ExpressibleByIntegerLiteral {
    public init(integerLiteral value: UInt32) {
        do {
            try self.init(value)
        } catch {
            preconditionFailure("\(error)")
        }
    }
}

/// The next scripted value for an attempt, or a failure for one that ran past the script.
/// Not `#require(values.popFirst())`: the macro cannot take a mutating call.
func next<T>(_ values: inout ArraySlice<T>) throws -> T {
    guard let value = values.popFirst() else {
        throw StepFailed(event: "attempted past the script")
    }
    return value
}
