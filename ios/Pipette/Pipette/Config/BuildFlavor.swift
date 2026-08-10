import Foundation

/// What this binary was compiled with, for a human — or a collector — confirming which
/// build a device is running.
///
/// One capability today, and it is the one that changes what a benchmark measures:
/// `PIPETTE_PRIVATE_THERMAL` decides whether the readiness gate waits on a real SoC die
/// temperature or falls back to `ProcessInfo.thermalState` plus the calibrated IMU
/// estimate. Two devices running the same marketing version can therefore produce numbers
/// that are not comparable, and nothing on the wire said so — which is why the marker
/// rides on the version string rather than sitting in a debug menu.
nonisolated enum BuildFlavor {
    /// Whether the private SoC die-temp read is compiled in. Resolved at build time from
    /// `SWIFT_ACTIVE_COMPILATION_CONDITIONS` — `ios/build.sh` sets it when
    /// `PIPETTE_PRIVATE_THERMAL=1`, and nothing else does.
    static let hasPrivateThermal: Bool = {
        #if PIPETTE_PRIVATE_THERMAL
        return true
        #else
        return false
        #endif
    }()

    /// Whether this build carries anything App Review would reject, and so must never
    /// reach TestFlight or the App Store.
    ///
    /// Derived from the capabilities above rather than being a flag of its own: the
    /// question "may this ship?" is answered by what got compiled in, so the two cannot
    /// drift. One capability implies it today; a second would be OR'd in here, leaving
    /// every caller — and the version suffix — unchanged.
    static var isInternal: Bool { hasPrivateThermal }

    /// Appended to the marketing version: `-internal` when the build must not ship,
    /// nothing otherwise.
    ///
    /// Named for the restriction, not the capability that causes it — of the two facts,
    /// "must not ship" is the one a reader must not miss, and it stays accurate if the
    /// set of internal-only capabilities grows. `headlessrun version` reports which
    /// capabilities outright. Absent rather than `-public` so an ordinary build's version
    /// reads unchanged.
    static var versionSuffix: String { isInternal ? "-internal" : "" }

    /// One line for `headlessrun version` and the Settings debug row.
    static var thermalDescription: String {
        hasPrivateThermal
            ? "private SoC die temp (internal build)"
            : "thermalState + calibrated IMU (no private read)"
    }
}
