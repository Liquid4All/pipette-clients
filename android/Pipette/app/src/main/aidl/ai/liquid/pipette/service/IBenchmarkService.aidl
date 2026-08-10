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
    boolean isAvailable();

    // Load modelPath fresh (freeing any resident model first). ok=true on
    // success; ok=false carries the native error message.
    BenchmarkResult loadModel(String modelPath, int nGpuLayers, int contextSize, int nUbatch);

    // Run against the already-loaded model, streaming progress to callback.
    BenchmarkResult runBenchmark(String benchmarkJson, int nGpuLayers, String mmprojPath, IBenchmarkRunCallback callback);

    // Fresh load + run + unload (the max_memory_usage clean-baseline path).
    BenchmarkResult runBenchmarkFresh(String benchmarkJson, String modelPath, int nGpuLayers, int contextSize, int nUbatch, String mmprojPath, IBenchmarkRunCallback callback);

    BenchmarkResult unloadModel();

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
