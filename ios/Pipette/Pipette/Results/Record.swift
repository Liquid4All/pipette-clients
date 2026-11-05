import Foundation

// The record step: `pipette-cli`'s `results/record.rs`. The submit the crate can make in
// the same call is `ResultUploader`'s drain here, so `recordResult` only saves.

/// The crate's `(BenchmarkSubmissionPayload, BenchmarkResultExtras)`.
nonisolated struct RunRecord: Sendable {
    let payload: BenchmarkSubmissionPayload
    let extras: BenchmarkResultExtras
}

/// Guards a payload built from a record that predates the descriptor format.
nonisolated func ensureDescriptor(field: String, value: String) throws {
    guard !value.isEmpty else {
        throw ResultSubmissionError.missingDescriptor(field: field)
    }
    do {
        _ = try JSONSerialization.jsonObject(with: Data(value.utf8), options: .fragmentsAllowed)
    } catch {
        throw ResultSubmissionError.invalidDescriptor(field: field,
                                                      reason: error.localizedDescription)
    }
}

/// An engine that returned another cell's flags would record a run that didn't happen.
private nonisolated func ensureFlagsRoundTripped(req: RunRequest, outcome: RunResponse) throws {
    guard let returned = outcome.runtimeFlags else { return }
    let benchmarkType = req.benchmark.benchmarkType
    guard returned.matches(benchmark: benchmarkType, runtime: req.runtime.declared,
                           model: req.model.declared)
    else {
        throw RunRequestError.runtimeFlagsForAnotherCell(
            flags: returned,
            cell: (benchmarkType, RuntimeType.of(req.runtime.declared),
                   ModelType.of(req.model.declared)))
    }
}

/// Over the cap is refused here, as the crate's `client_extras` does: mgmt would answer
/// `400`, and the result would stay pending for good.
nonisolated func checkedClientExtras(_ extras: String?) throws -> String? {
    guard let extras else { return nil }
    guard extras.utf8.count <= maxClientExtrasBytes else {
        throw ResultSubmissionError.clientExtrasTooLarge(bytes: extras.utf8.count)
    }
    return extras
}

/// The device and power state are detected here, under the operator's `device.json` labels.
nonisolated func finishedRunPayload(
    deviceStore: ClientDeviceStore,
    req: RunRequest,
    outcome: RunResponse,
    telemetry: RepTelemetry
) async throws -> RunRecord {
    let labels = try deviceStore.get()
    let device = detectDeviceInfo(deviceName: labels.deviceName,
                                  deviceFormFactor: labels.deviceFormFactor)
    let power = await detectPowerState()
    // No CPU variant: only the Android client detects one.
    return try buildSubmissionPayloadFromRun(
        outcome: outcome, req: req, runtimeCpuVariant: nil, device: device, power: power,
        telemetry: telemetry)
}

/// The crate's `record_and_maybe_submit_run`, less the submit. Keyed by the cell, where the
/// crate mints an id per run: a cell has one result here, and a re-run replaces it.
nonisolated func recordResult(
    results: ResultsStore,
    payload: BenchmarkSubmissionPayload,
    extras: BenchmarkResultExtras,
    location: BenchmarkResultLocation,
    resultId: CellId
) throws {
    // Written before the previous attempt's progress is cleared, never after: a kill
    // between the two must leave a result on disk, and this device gets killed mid-run
    // routinely.
    try results.saveResult(
        location, resultId,
        payload: try Coding.encoder.encode(payload),
        extras: try Coding.encoder.encode(extras))
    // A re-run of a synced cell lands in `remote/pending`, and `location(of:)` would keep
    // finding the synced copy first. Dropped after the write so a kill never leaves no
    // result; a kill in between marks the cell failed, and a retry replaces both.
    for other in BenchmarkResultLocation.allCases where other != location {
        if let dir = results.resultDir(other, resultId) {
            try? FileManager.default.removeItem(at: dir)
        }
    }
    results.clearProgress(resultId)
}

/// Sets `device_time_ms` to the time since the claim; a run no claim started is unchanged.
nonisolated func timedFromClaim(
    _ payload: BenchmarkSubmissionPayload,
    claimedAt: ContinuousClock.Instant?,
    now: ContinuousClock.Instant
) -> BenchmarkSubmissionPayload {
    guard let claimedAt else { return payload }
    var timed = payload
    timed.deviceTimeMs = deviceTimeMs(now - claimedAt)
    return timed
}

/// Over its cap a field loses its first elements, as the server would cut it, so the
/// measurement still goes out. Keys sorted because `JSONEncoder` guarantees no key order.
nonisolated func telemetryField(_ name: String, _ elements: [some Encodable],
                                maxBytes: Int) throws -> String? {
    guard !elements.isEmpty else { return nil }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let sent = String(decoding: try encoder.encode(elements), as: UTF8.self)
    let kept = try keepLastElements(sent, maxBytes: maxBytes)
    if kept.utf8.count < sent.utf8.count {
        AppLog.jobRun.warning("device telemetry over its cap: kept its last elements "
            + "field=\(name) sent_bytes=\(sent.utf8.count) kept_bytes=\(kept.utf8.count)")
    }
    return kept
}

/// No `job_id`: the server assigns one on submission.
nonisolated func buildSubmissionPayloadFromRun(
    outcome: RunResponse,
    req: RunRequest,
    runtimeCpuVariant: String?,
    device: DeviceInfo,
    power: PowerReading,
    telemetry: RepTelemetry
) throws -> RunRecord {
    // The plan coordinate, never the bound form, whose host paths mean nothing off device.
    let modelDescriptor = try SubmissionRef.model(req.model.declared)
    let runtimeDescriptor = try SubmissionRef.runtime(req.runtime.declared)
    try ensureDescriptor(field: "model_descriptor", value: modelDescriptor)
    try ensureDescriptor(field: "runtime_descriptor", value: runtimeDescriptor)
    try ensureFlagsRoundTripped(req: req, outcome: outcome)
    let benchmarkType = req.benchmark.benchmarkType
    let payload = BenchmarkSubmissionPayload(
        benchmarkId: req.benchmark.benchmarkId.value,
        // Checked by the server against its catalog, so a run against a stale or edited
        // definition is refused rather than filed under the id.
        benchmarkParams: BenchmarkParams(req.benchmark),
        device: device,
        power: power,
        deviceHardwareReadings: try telemetryField("device_hardware_readings",
                                                   telemetry.hardwareReadings,
                                                   maxBytes: maxDeviceReadingsBytes),
        deviceOsThrottlingReadings: try telemetryField("device_os_throttling_readings",
                                                       telemetry.osThrottlingReadings,
                                                       maxBytes: maxDeviceReadingsBytes),
        readinessGateWaits: try telemetryField("readiness_gate_waits", telemetry.gateWaits,
                                               maxBytes: maxReadinessGateWaitsBytes),
        // The cell's generation settings on an eval row, nothing elsewhere. The claim path
        // still refuses `enable_thinking` and `reasoning_effort`, so no cell reaching here
        // sets one and this stays absent for now.
        modelFlags: req.modelFlags?.submissionString(benchmarkType),
        modelDescriptor: modelDescriptor,
        // Absent only for Apple Foundation, whose axes name no flags variant.
        runtimeFlags: try outcome.runtimeFlags.map { try $0.submissionValue() },
        benchmarkFlags: try outcome.benchmarkFlags.map { try $0.submissionValue() },
        runtimeDescriptor: runtimeDescriptor,
        runtimeCpuVariant: runtimeCpuVariant,
        clientExtras: try checkedClientExtras(outcome.clientExtras),
        submittedAt: JobDateFormat.iso8601.string(from: Date()),
        // The corpus that was actually traversed, matched or not: the server records the
        // verdict rather than rejecting a mismatch.
        observationDatasetDigest: outcome.datasetDigest,
        // The engine's: the digest is over the turns it walked.
        observationOutputDigest: outcome.outputDigest,
        result: outcome.resultData
    )
    // Never submitted. No `command` or `executable`: an in-process engine runs no argv.
    let extras = BenchmarkResultExtras(command: [],
                                       stdout: outcome.runtimeOutput.runtimeStdout ?? "",
                                       stderr: outcome.runtimeOutput.runtimeStderr ?? "",
                                       turns: outcome.turns.isEmpty ? nil : outcome.turns)
    return RunRecord(payload: payload, extras: extras)
}
