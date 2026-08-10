package ai.liquid.pipette

/**
 * Pure process-classification logic, split out from [PipetteApp] so it can be unit-tested on the JVM (the crash it guards against only reproduces
 * across real processes, which CI's unit tests can't spin up — but the classification the guard depends on can be pinned here).
 *
 * `Application.onCreate` runs in every process the app starts. This app has two: the main (UI) process, named exactly [packageName], and the isolated
 * benchmark service, named `"$packageName:benchmark"` (see `AndroidManifest.xml`). Only the main process is [isMainProcess]; a `:suffix` process is
 * not.
 */
internal object ProcessGate {
  /**
   * Whether [processName] (from `Application.getProcessName()`) is the app's main process for [packageName]. Only an exact match qualifies — the
   * benchmark process reports `"$packageName:benchmark"`, so a prefix/substring check would wrongly classify it as main and reintroduce the
   * WorkManager-in-`:benchmark` crash (see [PipetteApp]). A null name (should not happen on minSdk 31) is treated as not-main.
   */
  fun isMainProcess(processName: String?, packageName: String): Boolean = processName == packageName

  /** The suffix the isolated benchmark service runs under — `android:process=":benchmark"` in `AndroidManifest.xml`. Single source of truth. */
  const val BENCHMARK_PROCESS_SUFFIX = ":benchmark"

  /** The benchmark service's full process name for [packageName] (e.g. as reported by `ApplicationExitInfo.getProcessName()`). */
  fun benchmarkProcessName(packageName: String): String = "$packageName$BENCHMARK_PROCESS_SUFFIX"
}
