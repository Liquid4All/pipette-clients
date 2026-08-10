import Foundation

// Typed, Codable mirror of the management server's submission schema. The
// canonical source of truth is the Rust `pipette-plan-types::benchmark`
// (`BenchmarkSubmissionPayload` / `BenchmarkResultData` / `BenchmarkEvalCompletion`);
// the desktop and Android clients serialize the same shape.
//
// The result is modeled as a sum type (`BenchmarkResult`) mirroring Rust's
// `BenchmarkResultData` enum: the wire format is an untagged union (the variant
// is identified purely by which fields are present), so a sum type is the only
// representation that cannot express a non-member — no empty result, no partial
// variant, no two variants at once. Decoding resolves to exactly one variant or
// throws, which *is* the coherence guard.
//
// Wire conventions mirrored here:
// - Optional fields elide when nil (matches serde `skip_serializing_if`), via the
//   payload's compiler-synthesized `encode(to:)` (`encodeIfPresent` for optionals).
// - The result is flattened into the top-level object (serde `flatten`). Swift
//   `Codable` has no `flatten`, so the payload carries the result fields directly,
//   populated from a `BenchmarkResult` by an exhaustive `switch` in its init.
// - `max_memory_usage` field names mirror Rust's methodology terms
//   (`max_host_bytes`/`max_gpu_bytes`); the wire keys stay `max_ram_bytes`/
//   `max_vram_bytes` via CodingKeys, exactly as Rust does with `#[serde(rename)]`.

/// Canonical per-sample stop reason.
///
/// The enum is **owned by pipette-mgmt** (`docs/scoring-service.md`); this is the
/// iOS mirror, named as the Rust client names it, so every producer emits the
/// same snake_case wire tokens. `doomLoop` has no on-device producer today
/// (the mobile engines have no doom-loop detector) but is modeled for a complete
/// contract; `unknown` is the fallback whenever a runtime can't classify the stop.
///
/// Provenance caveat: mgmt marks any client-produced value `recorded`, whether it
/// came from an authoritative engine signal (llama.cpp EOG, MLX EOS break) or a
/// client-side heuristic (AFM: generated-token count vs the cap, approximate). The
/// enum does not distinguish authoritative from heuristic.
enum BenchmarkEvalCompletionStopReason: String, Codable, Equatable {
    case eos
    case truncated
    case doomLoop = "doom_loop"
    case failure
    case unknown
}

/// One eval sample's outcome (`BenchmarkResultData::Eval`'s `completions` element).
/// Modeled as a sum — a sample either completed or failed — so an invalid state
/// (e.g. failed *with* a real completion) can't be constructed. The `Codable` below
/// produces the server's flat shape verbatim: a success is
/// `{id, completion, stop_reason, [completion_tokens]}`; a failure is
/// `{id, completion: "", failed: true, failed_reason, stop_reason: "failure"}`.
/// `stop_reason` is **required** on the client — every sample is classified
/// (`unknown` when indeterminate) — so it is always on the wire.
nonisolated enum BenchmarkEvalCompletion: Equatable {
    case completed(id: String, text: String, stopReason: BenchmarkEvalCompletionStopReason, stopDetail: String?, completionTokens: Int?)
    case failed(id: String, reason: String)

    var id: String {
        switch self {
        case .completed(let id, _, _, _, _), .failed(let id, _): return id
        }
    }
}

nonisolated extension BenchmarkEvalCompletion: Codable {
    private enum CodingKeys: String, CodingKey {
        case id, completion, failed
        case failedReason = "failed_reason"
        case stopReason = "stop_reason"
        case stopDetail = "stop_detail"
        case completionTokens = "completion_tokens"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let id = try c.decode(String.self, forKey: .id)
        if try c.decodeIfPresent(Bool.self, forKey: .failed) == true {
            self = .failed(id: id, reason: try c.decodeIfPresent(String.self, forKey: .failedReason) ?? "")
        } else {
            self = .completed(
                id: id,
                text: try c.decode(String.self, forKey: .completion),
                // Required on the client, but tolerate a pre-feature payload
                // that predates the field by defaulting to `unknown`.
                stopReason: try c.decodeIfPresent(BenchmarkEvalCompletionStopReason.self, forKey: .stopReason) ?? .unknown,
                stopDetail: try c.decodeIfPresent(String.self, forKey: .stopDetail),
                completionTokens: try c.decodeIfPresent(Int.self, forKey: .completionTokens))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        switch self {
        case .completed(_, let text, let stopReason, let stopDetail, let completionTokens):
            try c.encode(text, forKey: .completion)
            // `stop_reason` is required on the client, so always emit it;
            // `stop_detail` / `completion_tokens` elide when unset.
            try c.encode(stopReason, forKey: .stopReason)
            try c.encodeIfPresent(stopDetail, forKey: .stopDetail)
            try c.encodeIfPresent(completionTokens, forKey: .completionTokens)
        case .failed(_, let reason):
            try c.encode("", forKey: .completion)   // server schema keeps the field present
            try c.encode(true, forKey: .failed)
            try c.encode(reason, forKey: .failedReason)
            // The client owns `stop_reason`: a sample that never produced a
            // completion is a `failure`, with the crash detail in `stop_detail`.
            // `failed` / `failed_reason` are dual-written for legacy consumers
            // until they are retired.
            try c.encode(BenchmarkEvalCompletionStopReason.failure, forKey: .stopReason)
            try c.encode(reason, forKey: .stopDetail)
        }
    }
}

/// A benchmark's measured result — exactly one variant, mirroring Rust's
/// `BenchmarkResultData`. Decoded from the runtime's raw result JSON by which
/// fields are present, serde-untagged style; a result with no recognized metric
/// throws.
enum BenchmarkResult: Decodable, Equatable {
    case prefillThroughput(timeMs: Double, stddev: Double?)
    case decodeThroughput(timeMs: Double, stddev: Double?)
    case endToEndLatency(timeMs: Double, stddev: Double?)
    case maxMemoryUsage(hostBytes: UInt64, gpuBytes: UInt64?, npuBytes: UInt64?)
    case eval(completions: [BenchmarkEvalCompletion])
    case vlThroughput(promptTokens: UInt32, promptMs: Double, promptMsStddev: Double?,
                      predictedMs: Double, predictedMsStddev: Double?)

    enum CodingKeys: String, CodingKey {
        case prefillTimeMs = "prefill_time_ms"
        case prefillTimeMsStddev = "prefill_time_ms_stddev"
        case decodeTimeMs = "decode_time_ms"
        case decodeTimeMsStddev = "decode_time_ms_stddev"
        case totalTimeMs = "total_time_ms"
        case totalTimeMsStddev = "total_time_ms_stddev"
        case maxHostBytes = "max_ram_bytes"
        case maxGpuBytes = "max_vram_bytes"
        case maxNpuBytes = "max_npu_bytes"
        case completions
        case promptTokens = "prompt_tokens"
        case promptMs = "prompt_ms"
        case promptMsStddev = "prompt_ms_stddev"
        case predictedMs = "predicted_ms"
        case predictedMsStddev = "predicted_ms_stddev"
    }

    // Resolve by the variant's *primary* field, in the same declaration order
    // as Rust's untagged `BenchmarkResultData`. Stddev fields are never
    // discriminators (meaningless without their mean). Variants needing more
    // than one field (max-memory, vl) decode their remaining required fields
    // with `decode` (not `decodeIfPresent`), so an incomplete variant throws
    // rather than silently producing a payload the server can't deserialize.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try c.decodeIfPresent(Double.self, forKey: .prefillTimeMs) {
            self = .prefillThroughput(
                timeMs: value, stddev: try c.decodeIfPresent(Double.self, forKey: .prefillTimeMsStddev))
        } else if let value = try c.decodeIfPresent(Double.self, forKey: .decodeTimeMs) {
            self = .decodeThroughput(
                timeMs: value, stddev: try c.decodeIfPresent(Double.self, forKey: .decodeTimeMsStddev))
        } else if let value = try c.decodeIfPresent(Double.self, forKey: .totalTimeMs) {
            self = .endToEndLatency(
                timeMs: value, stddev: try c.decodeIfPresent(Double.self, forKey: .totalTimeMsStddev))
        } else if let hostBytes = try c.decodeIfPresent(UInt64.self, forKey: .maxHostBytes) {
            self = .maxMemoryUsage(
                hostBytes: hostBytes,
                gpuBytes: try c.decodeIfPresent(UInt64.self, forKey: .maxGpuBytes),
                npuBytes: try c.decodeIfPresent(UInt64.self, forKey: .maxNpuBytes))
        } else if let completions = try c.decodeIfPresent([BenchmarkEvalCompletion].self, forKey: .completions) {
            self = .eval(completions: completions)
        } else if let promptTokens = try c.decodeIfPresent(UInt32.self, forKey: .promptTokens) {
            self = .vlThroughput(
                promptTokens: promptTokens,
                promptMs: try c.decode(Double.self, forKey: .promptMs),
                promptMsStddev: try c.decodeIfPresent(Double.self, forKey: .promptMsStddev),
                predictedMs: try c.decode(Double.self, forKey: .predictedMs),
                predictedMsStddev: try c.decodeIfPresent(Double.self, forKey: .predictedMsStddev))
        } else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "benchmark result has no recognized metric fields"))
        }
    }
}

/// A runtime's result as returned over the wire: the `benchmark_id` sibling plus
/// the flattened `BenchmarkResult` variant (`{"benchmark_id": …, <metric fields>}`).
/// Mirrors how Rust keeps `benchmark_id` as a field of the payload, outside the
/// flattened `BenchmarkResultData`.
struct BenchmarkRun: Decodable, Equatable {
    let benchmarkId: String
    let result: BenchmarkResult

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try container.decode(String.self, forKey: .benchmarkId)
        result = try BenchmarkResult(from: decoder)  // re-reads the same flat object
    }
}

/// The full payload POSTed to the management server: device/runtime/model
/// identity plus the flattened result fields. There is deliberately no `job_id`:
/// it isn't part of the schema (the server assigns one on submission).
///
/// `encode(to:)` is compiler-synthesized — every stored property is emitted with
/// its `CodingKeys` wire name, and nil optionals elide — so no field can be
/// dropped by a hand-written encoder. The result fields are populated from a
/// `BenchmarkResult` by the exhaustive `switch` in the initializer (a new
/// variant forces that switch to be updated).
struct BenchmarkSubmissionPayload: Encodable {
    let benchmarkId: String
    // Device (auto-detected on-device; GPU/NPU fields are nil on iOS).
    let deviceName: String
    let deviceFormFactor: DeviceFormFactor
    let deviceOsName: String
    let deviceOsVersion: String
    // Precise OS build (e.g. "22F76"), finer than deviceOsVersion. Optional —
    // elides when unreadable.
    let deviceOsBuild: String?
    // Always nil here — no iOS API exposes a security-patch level. Carried because
    // the crate's flattened `DeviceInfo` has it and Android fills it.
    let deviceOsSecurityPatch: String?
    let deviceChipModel: String
    let deviceRamBytes: UInt64
    let deviceGpuModel: String?
    let deviceGpuVramBytes: UInt64?
    let deviceNpuModel: String?
    let deviceNpuVramBytes: UInt64?
    // Run-environment power state.
    let deviceBatteryLevel: Int32?
    let devicePowerState: DevicePowerState?
    let devicePowerSaveMode: Bool
    // Per-iteration Apple thermal telemetry: one `ProcessInfo.thermalState`
    // token (`nominal`/`fair`/`serious`/`critical`) per measured rep, sampled at
    // the rep's gate pass (`before`) and after its timed work (`after`). `nil`
    // when nothing was captured (e.g. non-throughput runs) — elided on encode.
    // Reuses the shared Apple family; the public build ships no temperature
    // column (die temp is a private API — see `deviceAppleSocTempC*`).
    let deviceAppleThermalStateBefore: [String]?
    let deviceAppleThermalStateAfter: [String]?
    // Raw SoC die temperature (°C), one fractional reading per rep, in rep order
    // (array position is the iteration index). Collected only on a
    // `PIPETTE_PRIVATE_THERMAL` build. Sampled at the same points as the state
    // series but NOT element-aligned with it: this series is always full-length
    // per rep, whereas the state series drops a rep whose token is unknown, so a
    // shared index may not name the same iteration. `nil` (elided) unless every
    // rep reported a valid reading.
    let deviceAppleSocTempCBefore: [Float]?
    let deviceAppleSocTempCAfter: [Float]?
    // Model + runtime identity: the descriptors state it, and nothing else does. A
    // reader that wants a display name or a quant decodes one
    // (`SubmissionRef.model(fromDescriptor:)`) rather than reading a second field.
    let modelFlags: String?
    /// Full, lossless model spec (plan-types `Model` as a JSON string). Required — every
    /// cell has a known model source. The server stores it opaquely.
    let modelDescriptor: String
    let runtimeFlags: String?
    /// The readiness gate the run was held by, as the crate's `benchmark_flags` — the only
    /// place a cell that waived the temperature criterion is told apart from a gated one.
    let benchmarkFlags: String?
    // Full, lossless runtime spec (plan-types `Runtime` as a JSON string), stored
    // opaquely by the server under `runtime_descriptor`.
    let runtimeDescriptor: String
    let runtimeCpuVariant: String?
    // Version of this app build — the harness, not the runtime it drove. The
    // server stores it opaquely as a grouping key, so a shift in the numbers
    // can be attributed to an app change rather than to the device. Same string
    // the Settings screen shows and Sentry tags as `app_version`, so a warehouse
    // row, a bug report, and a crash all name the build identically.
    let clientVersion: String
    let submittedAt: String
    // Result fields, flattened from the `BenchmarkResult` variant (see init).
    let prefillTimeMs: Double?
    let prefillTimeMsStddev: Double?
    let decodeTimeMs: Double?
    let decodeTimeMsStddev: Double?
    let totalTimeMs: Double?
    let totalTimeMsStddev: Double?
    let maxHostBytes: UInt64?
    let maxGpuBytes: UInt64?
    let maxNpuBytes: UInt64?
    let completions: [BenchmarkEvalCompletion]?
    let promptTokens: UInt32?
    let promptMs: Double?
    let promptMsStddev: Double?
    let predictedMs: Double?
    let predictedMsStddev: Double?

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceOsBuild = "device_os_build"
        case deviceOsSecurityPatch = "device_os_security_patch"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
        case deviceBatteryLevel = "device_battery_level"
        case devicePowerState = "device_power_state"
        case devicePowerSaveMode = "device_power_save_mode"
        case deviceAppleThermalStateBefore = "device_apple_thermal_state_before"
        case deviceAppleThermalStateAfter = "device_apple_thermal_state_after"
        case deviceAppleSocTempCBefore = "device_apple_soc_temp_c_before"
        case deviceAppleSocTempCAfter = "device_apple_soc_temp_c_after"
        case modelFlags = "model_flags"
        case modelDescriptor = "model_descriptor"
        case runtimeFlags = "runtime_flags"
        case benchmarkFlags = "benchmark_flags"
        case runtimeDescriptor = "runtime_descriptor"
        case runtimeCpuVariant = "runtime_cpu_variant"
        case clientVersion = "client_version"
        case submittedAt = "submitted_at"
        case prefillTimeMs = "prefill_time_ms"
        case prefillTimeMsStddev = "prefill_time_ms_stddev"
        case decodeTimeMs = "decode_time_ms"
        case decodeTimeMsStddev = "decode_time_ms_stddev"
        case totalTimeMs = "total_time_ms"
        case totalTimeMsStddev = "total_time_ms_stddev"
        case maxHostBytes = "max_ram_bytes"
        case maxGpuBytes = "max_vram_bytes"
        case maxNpuBytes = "max_npu_bytes"
        case completions
        case promptTokens = "prompt_tokens"
        case promptMs = "prompt_ms"
        case promptMsStddev = "prompt_ms_stddev"
        case predictedMs = "predicted_ms"
        case predictedMsStddev = "predicted_ms_stddev"
    }

    /// `device` and `power` arrive as whole values rather than as twelve loose
    /// arguments, mirroring the crate's builder, which takes the flattened
    /// `DeviceInfo` and `PowerState` the same way. Fifteen same-typed strings and
    /// optionals in a row is where a transposition hides.
    init(
        benchmarkId: String,
        device: DeviceInfo,
        power: PowerState,
        deviceAppleThermalStateBefore: [String]? = nil,
        deviceAppleThermalStateAfter: [String]? = nil,
        deviceAppleSocTempCBefore: [Float]? = nil,
        deviceAppleSocTempCAfter: [Float]? = nil,
        modelFlags: String?,
        modelDescriptor: String,
        runtimeFlags: String?,
        benchmarkFlags: String?,
        runtimeDescriptor: String,
        runtimeCpuVariant: String?,
        clientVersion: String = Bundle.main.appVersionDisplayString,
        submittedAt: String,
        result: BenchmarkResult
    ) {
        self.benchmarkId = benchmarkId
        self.deviceName = device.deviceName
        self.deviceFormFactor = device.deviceFormFactor
        self.deviceOsName = device.deviceOsName
        self.deviceOsVersion = device.deviceOsVersion
        self.deviceOsBuild = device.deviceOsBuild
        self.deviceOsSecurityPatch = device.deviceOsSecurityPatch
        self.deviceChipModel = device.deviceChipModel
        self.deviceRamBytes = device.deviceRamBytes
        self.deviceGpuModel = device.deviceGpuModel
        self.deviceGpuVramBytes = device.deviceGpuVramBytes
        self.deviceNpuModel = device.deviceNpuModel
        self.deviceNpuVramBytes = device.deviceNpuVramBytes
        self.deviceBatteryLevel = power.batteryLevel
        self.devicePowerState = power.powerState
        self.devicePowerSaveMode = power.powerSaveMode
        self.deviceAppleThermalStateBefore = deviceAppleThermalStateBefore
        self.deviceAppleThermalStateAfter = deviceAppleThermalStateAfter
        self.deviceAppleSocTempCBefore = deviceAppleSocTempCBefore
        self.deviceAppleSocTempCAfter = deviceAppleSocTempCAfter
        self.modelFlags = modelFlags
        self.modelDescriptor = modelDescriptor
        self.runtimeFlags = runtimeFlags
        self.benchmarkFlags = benchmarkFlags
        self.runtimeDescriptor = runtimeDescriptor
        self.runtimeCpuVariant = runtimeCpuVariant
        self.clientVersion = clientVersion
        self.submittedAt = submittedAt

        // Flatten the active variant into the result fields; the rest stay nil
        // (and elide on encode). Exhaustive — a new variant won't compile until
        // it's mapped here.
        var prefillTimeMs: Double?
        var prefillTimeMsStddev: Double?
        var decodeTimeMs: Double?
        var decodeTimeMsStddev: Double?
        var totalTimeMs: Double?
        var totalTimeMsStddev: Double?
        var maxHostBytes: UInt64?
        var maxGpuBytes: UInt64?
        var maxNpuBytes: UInt64?
        var completions: [BenchmarkEvalCompletion]?
        var promptTokens: UInt32?
        var promptMs: Double?
        var promptMsStddev: Double?
        var predictedMs: Double?
        var predictedMsStddev: Double?
        switch result {
        case let .prefillThroughput(timeMs, stddev):
            prefillTimeMs = timeMs; prefillTimeMsStddev = stddev
        case let .decodeThroughput(timeMs, stddev):
            decodeTimeMs = timeMs; decodeTimeMsStddev = stddev
        case let .endToEndLatency(timeMs, stddev):
            totalTimeMs = timeMs; totalTimeMsStddev = stddev
        case let .maxMemoryUsage(hostBytes, gpuBytes, npuBytes):
            maxHostBytes = hostBytes; maxGpuBytes = gpuBytes; maxNpuBytes = npuBytes
        case let .eval(samples):
            completions = samples
        case let .vlThroughput(tokens, promptMsValue, promptStddev, predictedMsValue, predictedStddev):
            promptTokens = tokens
            promptMs = promptMsValue
            promptMsStddev = promptStddev
            predictedMs = predictedMsValue
            predictedMsStddev = predictedStddev
        }
        self.prefillTimeMs = prefillTimeMs
        self.prefillTimeMsStddev = prefillTimeMsStddev
        self.decodeTimeMs = decodeTimeMs
        self.decodeTimeMsStddev = decodeTimeMsStddev
        self.totalTimeMs = totalTimeMs
        self.totalTimeMsStddev = totalTimeMsStddev
        self.maxHostBytes = maxHostBytes
        self.maxGpuBytes = maxGpuBytes
        self.maxNpuBytes = maxNpuBytes
        self.completions = completions
        self.promptTokens = promptTokens
        self.promptMs = promptMs
        self.promptMsStddev = promptMsStddev
        self.predictedMs = predictedMs
        self.predictedMsStddev = predictedMsStddev
    }
}

/// What a finished run leaves beside its payload for diagnosis — the crate's
/// `BenchmarkResultExtras`, stored as `extras.json` and never submitted.
///
/// `executable` and `command` name a shelled-out invocation, which an in-process engine
/// does not make; they are kept so the artifact reads the same on either side.
nonisolated struct BenchmarkResultExtras: Codable, Equatable {
    var executable: String?
    var command: [String]
    var stdout: String
    var stderr: String
}
