import Foundation

/// What this binary was compiled with. Each private capability changes what a benchmark
/// measures (a real SoC die temperature for the readiness gate; a reclaim-ring flush and
/// the kernel footprint peak for `max_memory_usage`), so two devices on the same version
/// can produce incomparable numbers. That is why the marker rides on the version string.
nonisolated enum BuildFlavor {
    /// Only `ios/build.sh` sets the condition, when `PIPETTE_PRIVATE_THERMAL=1`.
    static let hasPrivateThermal: Bool = {
        #if PIPETTE_PRIVATE_THERMAL
        return true
        #else
        return false
        #endif
    }()

    /// `ios/build.sh` sets this alongside `PIPETTE_PRIVATE_THERMAL`, so one switch still
    /// means "internal".
    static let hasPrivateMemory: Bool = {
        #if PIPETTE_PRIVATE_MEMORY
        return true
        #else
        return false
        #endif
    }()

    /// Whether this build carries anything App Review would reject, and so must never
    /// reach TestFlight or the App Store. Derived from what got compiled in, so the two
    /// cannot drift.
    static var isInternal: Bool { hasPrivateThermal || hasPrivateMemory }

    /// Named for the restriction, not the capability: "must not ship" is what a reader must
    /// not miss. `headlessrun version` reports the capabilities themselves.
    static var versionSuffix: String { isInternal ? "-internal" : "" }

    /// For `headlessrun version` and the Settings debug row.
    static var thermalDescription: String {
        hasPrivateThermal
            ? "private SoC die temp (internal build)"
            : "thermalState + calibrated IMU (no private read)"
    }

    /// For `headlessrun version` and the Settings debug row.
    static var memoryDescription: String {
        hasPrivateMemory
            ? "private reclaim-ring flush + kernel footprint peak + 20 ms sampler (internal build)"
            : "20 ms footprint sampler, no reclaim-ring flush (no private calls)"
    }
}
