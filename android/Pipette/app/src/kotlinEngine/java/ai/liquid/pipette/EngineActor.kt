package ai.liquid.pipette

import ai.liquid.pipette.llama.LlamaBenchmark
import ai.liquid.pipette.llama.LlamaCpp
import ai.liquid.pipette.llama.LlamaModel
import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

/** Kotlin engine actor implementation for kotlinEngine flavor. Mirrors the BenchmarkEngine interface for pure-Kotlin llama.cpp execution. */
@Suppress("TooManyFunctions")
class EngineActor : BenchmarkEngine {
  sealed class State {
    object Empty : State()

    data class Ready(val model: LlamaModel, val modelPath: String, val contextSize: Int) : State()

    data class Busy(val model: LlamaModel, val modelPath: String, val contextSize: Int) : State()

    val loadedModel: LlamaModel?
      get() =
        when (this) {
          is Ready -> model
          is Busy -> model
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

  override fun benchmarkProcessCpuAffinity(): CpuAffinitySnapshot = CpuAffinityProbe.snapshot()

  override fun lastRunInterruption(): CpusetDemotion? = null

  override fun endJob() = Unit

  override fun publishJobProgress(progress: BenchmarkProgressBus.Progress) {
    BenchmarkProgressBus.publish(progress)
  }

  override var onJobCancelRequested: (() -> Unit)? = null
    set(value) {
      field = value
      BenchmarkProgressBus.setCancelAction(value)
    }

  override fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: LlamacppLoadMode) {
    onWorker {
      destroyModelLocked()
      val model = LlamaCpp.loadModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode)
      _state.set(State.Ready(model, modelPath, contextSize))
    }
  }

  override fun runBenchmarkSync(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String = onWorker {
    val ready = _state.get() as? State.Ready ?: error("Engine not ready")
    _state.set(State.Busy(ready.model, ready.modelPath, ready.contextSize))
    try {
      LlamaBenchmark.runOnModel(
        benchmarkJson = benchmarkJson,
        model = ready.model,
        nGpuLayers = nGpuLayers,
        contextSize = ready.contextSize,
        mmprojPath = mmprojPath,
        progress = progress,
        cooldown = cooldown,
        thermal = thermal,
      )
    } finally {
      if ((_state.get() as? State.Busy)?.model === ready.model) _state.set(ready)
    }
  }

  override fun runFreshSync(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    cooldown: BenchmarkCooldownCallback?,
    thermal: BenchmarkThermalCallback?,
  ): String = onWorker {
    destroyModelLocked()
    LlamaBenchmark.runFresh(
      benchmarkJson = benchmarkJson,
      modelPath = modelPath,
      nGpuLayers = nGpuLayers,
      contextSize = contextSize,
      nUbatch = nUbatch,
      loadMode = loadMode,
      mmprojPath = mmprojPath,
      progress = progress,
      cooldown = cooldown,
      thermal = thermal,
    )
  }

  override fun unloadSync() {
    onWorker { destroyModelLocked() }
  }

  override fun destroy() {
    handler.post {
      destroyModelLocked()
      workerThread.quitSafely()
    }
  }

  private fun destroyModelLocked() {
    _state.getAndSet(State.Empty).loadedModel?.let { runCatching { it.close() } }
  }

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
    if (!posted) error("EngineActor has been destroyed")
    latch.await()
    error.get()?.let { throw it }
    @Suppress("UNCHECKED_CAST")
    return result.get() as T
  }
}
