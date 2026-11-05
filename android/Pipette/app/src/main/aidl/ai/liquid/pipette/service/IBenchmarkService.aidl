package ai.liquid.pipette.service;

import ai.liquid.pipette.service.BenchmarkResult;
import ai.liquid.pipette.service.IBenchmarkRunCallback;
import ai.liquid.pipette.service.IJobCancelCallback;

// Mirrors the in-process BenchmarkEngine surface. load/run/unload are blocking
// two-way calls: the caller's (off-main) thread parks until the service's
// engine worker finishes, and binder propagates service death as a
// RemoteException so the proxy can fail the cell and move on. requestCancel is
// oneway so it reaches the service while a run holds a binder thread.
interface IBenchmarkService {
    String llamaCppCommit();
    // Active CPU-backend feature descriptor of the runtime-selected variant
    // (null before a model has loaded / when native is missing). Java String
    // returns cross binder as nullable. Surfaced as runtime_cpu_variant.
    String cpuBackendDescriptor();
    // Diagnostic: the :benchmark process's cpuset / CPU-affinity snapshot, as a
    // JSON string (CpuAffinitySnapshot.toJson()). Read in this process so it
    // reflects the (possibly OEM-demoted) scheduling group inference runs under.
    // Always non-null from the service (individual fields degrade to null when a
    // /proc entry is unreadable); the proxy returns null when unbound or the read
    // fails (RemoteException / unparseable payload).
    String benchmarkProcessCpuAffinity();

    // Why the last run stopped, when it stopped because this process was
    // descheduled mid-measurement (screen off, the pocket screen losing window
    // focus, another app taking top-app): a CpusetDemotion.toJson(), else null.
    //
    // The watch behind this is scoped to a CELL, not a run: it starts at
    // loadModel and covers the model load, where a screen-off most often lands
    // and where no run exists yet. A demotion seen during the load is reported
    // against the run that follows it.
    //
    // Read AFTER the run call returns rather than delivered through its result,
    // because the demotion cancels the run and the run can then come back any of
    // three ways: throwing, returning a cancelled/partial result, or completing
    // successfully in the window before the cancel takes effect. All three are
    // invalid measurements and all three have to be distinguishable from a user
    // cancel, which files the cell CANCELLED rather than FAILED.
    //
    // DESTRUCTIVE READ: the first call returns the verdict and clears it, so a
    // second call returns null. That is what stops one demotion explaining more
    // than one cell. The watch stays armed between cells (a demotion confirmed in
    // a run's last moments still has to be readable once the run returns), so a
    // demotion seen in the gap between cells would otherwise still be sitting here
    // when the next cell asks, and that cell may have failed for an unrelated
    // reason such as never reaching this process at all.
    //
    // Also cleared when the next CELL starts (a loadModel, or a runBenchmarkFresh).
    String lastRunInterruption();

    boolean isAvailable();

    // Load modelPath fresh (freeing any resident model first). ok=true on
    // success; ok=false carries the native error message. loadMode is llama's
    // own llama_load_mode discriminant (LlamacppLoadMode.cValue), passed as the
    // number rather than the word so neither side parses anything.
    BenchmarkResult loadModel(String modelPath, int nGpuLayers, int contextSize, int nUbatch, int loadMode);

    // Run against the already-loaded model, streaming progress to callback.
    BenchmarkResult runBenchmark(String benchmarkJson, int nGpuLayers, String mmprojPath, IBenchmarkRunCallback callback);

    // Fresh load + run + unload (the max_memory_usage clean-baseline path).
    BenchmarkResult runBenchmarkFresh(String benchmarkJson, String modelPath, int nGpuLayers, int contextSize, int nUbatch, int loadMode, String mmprojPath, IBenchmarkRunCallback callback);

    BenchmarkResult unloadModel();

    // End of a JOB, not of a cell: stop the cpuset watch AND drop the baseline it
    // carries between cells.
    //
    // Distinct from unloadModel because that is not a job boundary: JobRunner also
    // unloads from its per-cell error path, and dropping the baseline there would
    // let the next cell adopt an already-demoted cpuset as its own normal and
    // measure on in silence. An unload keeps the baseline; only this clears it.
    //
    // Needed at all because this process outlives a job by a short grace period, so
    // a job started inside that window reuses it and would otherwise be judged
    // against the previous job's top-app state.
    //
    // Deliberately NOT oneway, unlike the fire-and-forget calls below. The whole
    // point is that the baseline is gone before the next job's first load arrives,
    // and a oneway dispatch could land after that load has already seeded its
    // tracker from the state this call was meant to erase.
    void endJob();

    // Cooperative cancel: flips a per-run flag the engine's progress/cooldown
    // shims honor. The hard-kill watchdog for an uninterruptible decode lives in
    // the proxy (it tears the service down, which kills this process).
    oneway void requestCancel();

    // Job pocket snapshot pushed main -> :benchmark so BenchmarkActivity's pocket
    // UI (top-app, for the cpuset boost) can render it, mirroring the Compose
    // PocketModeScreen. JSON is BenchmarkProgressBus.Progress.toJson() (title,
    // subtitle, cell label/status, cells done, overall permil, eta, cooling,
    // running=false signals the run ended → the activity finishes). oneway:
    // frequent, fire-and-forget.
    oneway void updateJobProgress(String progressJson);

    // Register the reverse cancel channel (see IJobCancelCallback). Set once the
    // proxy binds; BenchmarkActivity's Cancel invokes it to cancel the whole job.
    oneway void setJobCancelCallback(IJobCancelCallback cb);
}
