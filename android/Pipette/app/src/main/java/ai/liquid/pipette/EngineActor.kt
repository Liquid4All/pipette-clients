package ai.liquid.pipette

import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

/**
 * Owns the native [LlamaEngine] and serializes every native call onto a single worker thread, behind an explicit state machine. Modeled on the
 * Samsung sample app's `EngineActor` — the engine reference lives inside the [State], and the actor (not the caller) decides whether to reuse the
 * loaded model or reload it.
 *
 * [JobRunner] runs a sequential cell loop on its own executor, so the engine operations it needs are exposed as blocking `*Sync` helpers that hop
 * onto the actor thread and wait. All model ownership and the reuse/reload/destroy decisions that [JobRunner] used to make inline now live here.
 */
@Suppress("TooManyFunctions") // delegating actor mirroring the growing BenchmarkEngine surface
class EngineActor : BenchmarkEngine {
  sealed class State {
    /** No model loaded. */
    object Empty : State()

    /** A model is loaded and ready to benchmark. */
    data class Ready(val engine: LlamaEngine, val modelPath: String, val contextSize: Int) : State()

    /** A benchmark is currently running on [engine]. */
    data class Busy(val engine: LlamaEngine, val modelPath: String, val contextSize: Int) : State()

    val loadedEngine: LlamaEngine?
      get() =
        when (this) {
          is Ready -> engine
          is Busy -> engine
          else -> null
        }
  }

  private val workerThread = HandlerThread("EngineActor").apply { start() }
  private val handler = Handler(workerThread.looper)
  private val _state = AtomicReference<State>(State.Empty)

  val state: State
    get() = _state.get()

  override val isAvailable: Boolean
    get() = NativeLib.isAvailable

  override fun llamaCppCommit(): String = NativeLib.llamaCppCommit()

  override fun cpuBackendDescriptor(): String? = NativeLib.cpuBackendDescriptor()

  // Read here so the snapshot reflects whatever process this actor runs in (the
  // :benchmark process under the service, or the caller's process in-process).
  override fun benchmarkProcessCpuAffinity(): CpuAffinitySnapshot = CpuAffinityProbe.snapshot()

  // In-process path (no separate :benchmark process): talk to the bus directly.
  override fun publishJobProgress(progress: BenchmarkProgressBus.Progress) {
    BenchmarkProgressBus.publish(progress)
  }

  override var onJobCancelRequested: (() -> Unit)? = null
    set(value) {
      field = value
      BenchmarkProgressBus.setCancelAction(value)
    }

  // Cancellation is cooperative: callers pass a progress/cooldown callback
  // that returns false to abort, which the native kernel honors mid-run.

  /**
   * Load [modelPath] fresh at [contextSize], freeing any resident model first. Always reloads — every benchmark runs on a cold model sized to itself,
   * matching the mlx/llama.cpp clients (which start a fresh server per benchmark) so a benchmark never inherits another's warmed or differently-sized
   * context. Blocks until the load completes; throws the native load error on failure.
   */
  override fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int) {
    onWorker {
      destroyEngineLocked()
      val engine = LlamaEngine.create(modelPath, nGpuLayers, contextSize, nUbatch)
      _state.set(State.Ready(engine, modelPath, contextSize))
    }
  }

  /** Run a benchmark against the already-loaded model. Requires a prior [loadSync]; transitions Ready→Busy→Ready around the run. */
  override fun runBenchmarkSync(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String = onWorker {
    val ready = _state.get() as? State.Ready ?: error("Engine not ready")
    _state.set(State.Busy(ready.engine, ready.modelPath, ready.contextSize))
    try {
      ready.engine.runBenchmark(benchmarkJson, nGpuLayers, mmprojPath, progress, cooldown, thermal)
    } finally {
      // Restore Ready unless a reload/destroy swapped the engine meanwhile.
      if ((_state.get() as? State.Busy)?.engine === ready.engine) _state.set(ready)
    }
  }

  /**
   * Run a benchmark that loads its own model and unloads when done — the `max_memory_usage` case. Any resident engine is destroyed first so the
   * fresh-load measurement starts from a clean memory baseline.
   */
  override fun runFreshSync(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String = onWorker {
    destroyEngineLocked()
    LlamaEngine.runFresh(benchmarkJson, modelPath, nGpuLayers, contextSize, nUbatch, mmprojPath, progress, cooldown, thermal)
  }

  /** Free the resident model (if any) on the worker thread, then block. */
  override fun unloadSync() {
    onWorker { destroyEngineLocked() }
  }

  /** Free the model and tear down the worker thread. */
  override fun destroy() {
    handler.post {
      destroyEngineLocked()
      workerThread.quitSafely()
    }
  }

  /** Destroy the resident engine and reset to [State.Empty]. Worker-thread only. */
  private fun destroyEngineLocked() {
    _state.getAndSet(State.Empty).loadedEngine?.let { runCatching { it.destroy() } }
  }

  /**
   * Run [block] on the actor's worker thread and block the caller until it returns, re-throwing any exception on the caller's thread. Runs inline if
   * already on the worker thread (defensive; callers are off-thread).
   */
  private fun <T> onWorker(block: () -> T): T {
    if (Thread.currentThread() === workerThread) return block()
    val latch = CountDownLatch(1)
    val result = AtomicReference<T>()
    val error = AtomicReference<Throwable>()
    val posted =
      handler.post {
        try {
          result.set(block())
        } catch (t: Throwable) {
          error.set(t)
        } finally {
          latch.countDown()
        }
      }
    // post() returns false once the looper is quitting (destroy() raced us).
    // Without this guard the runnable never runs, latch never counts down,
    // and the caller's thread blocks on await() forever. Surface it as an
    // error instead so JobRunner marks the cell and moves on.
    if (!posted) error("EngineActor has been destroyed")
    latch.await()
    error.get()?.let { throw it }
    @Suppress("UNCHECKED_CAST")
    return result.get() as T
  }
}
