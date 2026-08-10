import Foundation

/// Enriches a raw benchmark result JSON with device/runtime/model fields
/// required by the server's SubmissionPayload and writes it to local storage.
enum PayloadBuilder {
    /// Takes the request the run was handed, as the crate's `finished_run_payload(req,
    /// outcome)` does, rather than the pieces unpacked: the benchmark, the model and the
    /// runtime all come off it, so a caller cannot pair a response with another cell's
    /// identity.
    ///
    /// `cellId`/`source`/`storage` are not upstream's concern — the crate returns the
    /// payload and lets its caller store it. The job is not part of a result's address:
    /// the result is keyed by its cell and filed by location, as the crate files it.
    static func writeLocal(
        request: RunRequest,
        response: RunResponse,
        cellId: CellId,
        source: BenchmarkSource,
        storage: Storage
    ) throws {
        let result = response.resultData
        let flags = response.runtimeFlags
        let thermal = response.thermal
        let benchmarkId = request.benchmark.benchmarkId
        // The plan coordinate, as `req.model.declared` is upstream — never the bound form,
        // whose host paths are meaningless off this device.
        let modelSource = request.model.declared
        let runtime = request.runtime.bound
        // One snapshot per submission, as the crate's `finished_run_payload` takes one:
        // the host's identity and the run's power state are read once and mapped in,
        // rather than each field re-probing at its own line.
        let device = DeviceProbe.detectDeviceInfo()
        let power = DeviceProbe.detectPowerState()

        // Flatten the grouped snapshots back into the per-iteration wire lists, as the
        // crate's submission builder does.
        let appleThermalStateBefore = thermal.before.compactMap(\.appleThermalState).map(\.rawValue)
        let appleThermalStateAfter = thermal.after.compactMap(\.appleThermalState).map(\.rawValue)

        // Emit a SoC series only when every rep reported a value, so array position stays
        // the iteration. A nil rep (a read that failed mid-run) or an empty series (public
        // build with the collection compiled out, or a non-throughput run) elides the
        // field rather than shipping a misaligned one.
        func socWireSeries(_ readings: [ThermalReading]) -> [Float]? {
            let raw = readings.map(\.appleSocTempC)
            guard !raw.isEmpty, !raw.contains(where: { $0 == nil }) else { return nil }
            return raw.compactMap { $0 }
        }

        // The flags the response carries have to be this run's — the crate's
        // `ensure_flags_round_tripped`, which refuses to record rather than describe a run
        // that did not happen.
        if let flags {
            let cell: FlagAxes = (try BenchmarkType(benchmarkId: benchmarkId),
                                  RuntimeType.of(runtime), ModelType.of(modelSource))
            guard flags.axes == cell else {
                throw RuntimeFlagsAxisMismatch(carried: flags.axes, cell: cell)
            }
        }

        // The knobs the run reported, whatever engine ran it — one derivation, as the
        // crate's record builder has. Absent only when the cell's axes name no flags
        // variant, which is Apple Foundation.
        let runtimeFlags = try flags.map { try $0.submissionValue() }
        let benchmarkFlags = try response.benchmarkFlags.map { try $0.submissionValue() }

        // The lossless typed specs (see `SubmissionRef`): model_descriptor is the
        // model coordinate, runtime_descriptor the engine identity + build.
        let modelDescriptor = try SubmissionRef.model(modelSource)
        // Identity is this binary's, never what the plan declared: the descriptor has
        // to record what actually ran.
        let runtimeDescriptor = try SubmissionRef.runtime(runtime)

        // Compose the submission payload from the typed `result` (the runtime
        // already produced a `BenchmarkResult` directly, rejecting a
        // metricless/partial result there).
        // The payload is a typed `Codable` value, so wire field names are
        // compiler-checked; the shape mirrors the Rust source of truth
        // `pipette_plan_types::result::BenchmarkSubmissionPayload` and `SubmissionPayloadTests`
        // guards against drift. (No `job_id` — the server assigns one on
        // submission; absent device fields elide rather than serialize as null.)
        let payload = BenchmarkSubmissionPayload(
            benchmarkId: benchmarkId,
            device: device,
            power: power,
            // Per-rep Apple thermal series; empty (non-throughput runs) elides.
            deviceAppleThermalStateBefore: appleThermalStateBefore.isEmpty ? nil : appleThermalStateBefore,
            deviceAppleThermalStateAfter: appleThermalStateAfter.isEmpty ? nil : appleThermalStateAfter,
            // Per-rep raw SoC die temp (°C); a nil rep or empty series elides.
            deviceAppleSocTempCBefore: socWireSeries(thermal.before),
            deviceAppleSocTempCAfter: socWireSeries(thermal.after),
            modelFlags: nil,
            modelDescriptor: modelDescriptor,
            runtimeFlags: runtimeFlags,
            benchmarkFlags: benchmarkFlags,
            runtimeDescriptor: runtimeDescriptor,
            runtimeCpuVariant: nil,
            submittedAt: JobDateFormat.iso8601.string(from: Date()),
            result: result
        )

        // `extras.json` beside the payload, as the crate's results store keeps them: a
        // local diagnostic artifact, never submitted. `command` is empty and there is no
        // `executable` because an in-process engine runs no argv; `stdout` likewise, and
        // `stderr` carries what the engine wrote while loading.
        let extras = BenchmarkResultExtras(command: [], stdout: "", stderr: response.stderr)
        // Filed by the half its benchmark came from, as the crate's
        // `BenchmarkResultLocation::from(BenchmarkSource)` files it. Written before the
        // previous attempt's progress is cleared, never after: a kill between the two
        // must leave a result on disk, and this device gets killed mid-run routinely.
        try storage.results.saveResult(
            BenchmarkResultLocation(recordedFrom: source), cellId,
            payload: try Coding.encoder.encode(payload),
            extras: try Coding.encoder.encode(extras))
        storage.results.clearProgress(cellId)
    }
}
