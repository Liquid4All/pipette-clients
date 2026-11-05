import Darwin
import Foundation

// This client's platform gate, the peer of `crates/pipette-readiness/src/macos.rs`: the OS
// thermal state and a die temperature (waived together by `ThermalGate.skip`), plus a
// busy-core CPU check that always applies. `maxDieC` and `maxImuC` are measured on iPhone
// sensors no other platform has, so do not sync them to a Rust value.

/// The CPU budget the device must stay *under* during `cpuSampleWindow`, in busy cores
/// (`busy ratio × online cores`). A reading equal to it fails.
///
/// Cores, not a fraction of capacity: one pinned core is the smallest competition that
/// matters. The value is `macos.rs`'s, measured on Macs (idle 0.23–0.37 busy cores against
/// 2.36 for one pinned core). No iPhone idle floor has been measured, so a device whose
/// background daemons hold a core will wait here and time out.
nonisolated let maxBusyCores: Double = 1.0

/// SoC die-temperature ceiling (°C) for a rep to proceed, when the private sensor is available
/// (`PIPETTE_PRIVATE_THERMAL` build). The measured idle floor is ~35°C (room ambient +
/// ~5–7°C), so 36°C is the deepest reliable setpoint: 34–35°C times out, 40°C leaves less
/// headroom. The floor tracks ambient, so raise this in a warmer room.
nonisolated let maxDieC: Double = 36.0

/// The setpoint for the public IMU-estimate fallback (°C), used only when the die sensor is
/// unavailable and the IMU has been calibrated. Higher than `maxDieC` on purpose: the
/// estimate is ~±1.5–2 °C, so gating at 38 keeps the true temperature reliably under 40 on
/// both test units, where a near-floor 36 lets the IMU noise false-trigger.
nonisolated let maxImuC: Double = 38.0

nonisolated enum TemperatureSignal: Sendable, Equatable {
    /// The private SoC die sensor (`dieTempMaxC()`).
    case die(Double)
    /// The public IMU estimate (`IMUThermometer`).
    case imu(Double)

    var celsius: Double {
        switch self {
        case let .die(celsius), let .imu(celsius): celsius
        }
    }

    var maxC: Double {
        if case .imu = self { maxImuC } else { maxDieC }
    }

    fileprivate var label: String {
        if case .imu = self { "imu" } else { "die" }
    }
}

/// Whether a temperature clears its setpoint. A nil reading waives the criterion rather than
/// failing it, as on macOS: a device with no signal falls back to the thermal state alone.
nonisolated func dieOk(_ signal: TemperatureSignal?) -> Bool {
    signal.map { $0.celsius < $0.maxC } ?? true
}

/// `macos.rs`'s inline `thermal_ok`, named here so it is tested. Any state above `.nominal`
/// waits whatever the temperature.
nonisolated func thermalOk(state: ProcessInfo.ThermalState, signal: TemperatureSignal?,
                           thermal: ThermalGate) -> Bool {
    thermal.skipped() || (state == .nominal && dieOk(signal))
}

/// `macos.rs`'s `Bounds::cpu_ok`. An unreadable load fails: any default would be wrong in the
/// unsafe direction.
nonisolated func cpuOk(busyCores: Double?) -> Bool {
    busyCores.map { $0 < maxBusyCores } ?? false
}

#if !targetEnvironment(simulator)
nonisolated func waitUntilReady(maxWait: Duration, thermal: ThermalGate,
                                callbacks: ReadinessCallbacks) -> ReadinessOutcome {
    // `readBusyCores` spends ~1 s of every poll interval sampling, which the loop's top-up
    // accounts for.
    waitLoop(platform: .ios, maxWait: maxWait, callbacks: callbacks) {
        // Read even when the gate is skipped: the line then says what the device's thermal
        // state was on a run that chose not to wait for it.
        let signal = readTemperatureSignal()
        let state = ProcessInfo.processInfo.thermalState
        let busyCores = readBusyCores()
        return Probe(ready: thermalOk(state: state, signal: signal, thermal: thermal)
                         && cpuOk(busyCores: busyCores),
                     summary: formatSummary(state: state, signal: signal, busyCores: busyCores,
                                            thermal: thermal))
    }
}
#endif

/// The log and timeout line, e.g. `thermal=state:nominal die:42C(hot) cpu=busy:0.4cores(ok)`.
/// A skipped gate is marked `[SKIPPED]` and an absent reading `unreadable`: a waived criterion
/// is not a passed one, and the line has to say which.
nonisolated func formatSummary(state: ProcessInfo.ThermalState, signal: TemperatureSignal?,
                               busyCores: Double?, thermal: ThermalGate) -> String {
    let skipped = thermal.skipped() ? "[SKIPPED]" : ""
    let tempField = signal.map {
        String(format: " %@:%.0fC(%@)", $0.label, $0.celsius, dieOk($0) ? "ok" : "hot")
    } ?? " die:unreadable"
    let cpuField = busyCores.map {
        String(format: "cpu=busy:%.1fcores(%@)", $0, cpuOk(busyCores: $0) ? "ok" : "busy")
    } ?? "cpu=busy:unreadable"
    return "thermal=state:\(formatPressureWord(state))\(skipped)\(tempField) \(cpuField)"
}

/// Named for `macos.rs`'s `format_pressure_word`, but in `ProcessInfo.ThermalState`'s names.
nonisolated func formatPressureWord(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal:  return "nominal"
    case .fair:     return "fair"
    case .serious:  return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// The private die sensor if this build has it, else the IMU estimate if calibrated.
nonisolated func readTemperatureSignal() -> TemperatureSignal? {
    if let celsius = dieTempMaxC() { return .die(celsius) }
    if let celsius = IMUThermometer.estimate() { return .imu(celsius) }
    return nil
}

/// Width of the CPU sampling window. `HOST_CPU_LOAD_INFO` refreshes on a coarse (~1 s)
/// cadence, so a shorter window often spans no update, and the samples that do land on one
/// are biased high. Do not shorten below ~800 ms.
nonisolated let cpuSampleWindow: Duration = .seconds(1)

/// One `HOST_CPU_LOAD_INFO` snapshot as `[user, system, idle, nice]` ticks. Public,
/// unprivileged Mach API on iOS as on macOS.
nonisolated func cpuTicks() -> [UInt32]? {
    var info = host_cpu_load_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
    let host = mach_host_self()
    // `mach_host_self` hands out a send right per call; release it, or every poll leaks one.
    defer { mach_port_deallocate(mach_task_self_, host) }
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics(host, HOST_CPU_LOAD_INFO, $0, &count)
        }
    }
    guard status == KERN_SUCCESS else { return nil }
    let ticks = info.cpu_ticks
    return [ticks.0, ticks.1, ticks.2, ticks.3]
}

nonisolated func readBusyCores() -> Double? {
    guard let ratio = readCpuLoad(), let cores = onlineCpus() else { return nil }
    return ratio * cores
}

/// Online logical CPUs, nil rather than a default when unreadable: assuming 1 would keep
/// every reading under `maxBusyCores`.
nonisolated func onlineCpus() -> Double? {
    let cores = ProcessInfo.processInfo.activeProcessorCount
    return cores > 0 ? Double(cores) : nil
}

/// Resampled once if the first window spanned no counter refresh.
nonisolated func readCpuLoad() -> Double? {
    machCpuBusy() ?? machCpuBusy()
}

nonisolated func machCpuBusy() -> Double? {
    guard let before = cpuTicks() else { return nil }
    Thread.sleep(forTimeInterval: cpuSampleWindow / .seconds(1))
    guard let after = cpuTicks() else { return nil }
    return busyFromTicks(before: before, after: after)
}

/// Nil if the counters did not advance. They are 32-bit and monotonic, so the wrapping
/// subtraction reads a wrapped one as the small forward delta it is.
nonisolated func busyFromTicks(before: [UInt32], after: [UInt32]) -> Double? {
    let idleIndex = Int(CPU_STATE_IDLE)
    guard before.count == after.count, before.count > idleIndex else { return nil }
    let deltas = zip(after, before).map { UInt64($0 &- $1) }
    let total = deltas.reduce(0, +)
    guard total > 0 else { return nil }
    return min(max(1 - Double(deltas[idleIndex]) / Double(total), 0), 1)
}
