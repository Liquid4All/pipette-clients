package ai.liquid.pipette

import ai.liquid.pipette.llama.LlamaBenchmark
import ai.liquid.pipette.llama.LlamaCpp
import ai.liquid.pipette.llama.LlamaModel
import ai.liquid.pipette.onnx.OnnxGenai
import ai.liquid.pipette.onnx.OnnxGenaiBenchmark
import ai.liquid.pipette.onnx.OnnxGenaiModel
import android.os.Handler
import android.os.HandlerThread
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

/**
 * Kotlin engine actor implementation for kotlinEngine flavor. Mirrors the BenchmarkEngine interface for pure-Kotlin execution on llama.cpp, or on
 * ONNX Runtime GenAI when the model path is a package ([OnnxGenai.isPackage]): the two engines take the same calls, so nothing above this class tells
 * them apart.
 */
@Suppress("TooManyFunctions")
class EngineActor : BenchmarkEngine {
  /** The resident model of whichever engine loaded it. */
  sealed class Resident : AutoCloseable {
    data class Llama(val model: LlamaModel) : Resident() {
      override fun close() = model.close()
    }

    data class Onnx(val model: OnnxGenaiModel) : Resident() {
      override fun close() = model.close()
    }
  }

  sealed class State {
    object Empty : State()

    data class Ready(val model: Resident, val modelPath: String, val contextSize: Int) : State()

    data class Busy(val model: Resident, val modelPath: String, val contextSize: Int) : State()

    val loadedModel: Resident?
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
      val model =
        if (OnnxGenai.isPackage(modelPath)) {
          Resident.Onnx(OnnxGenai.loadModel(modelPath))
        } else {
          Resident.Llama(LlamaCpp.loadModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode))
        }
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
      when (val resident = ready.model) {
        is Resident.Llama ->
          LlamaBenchmark.runOnModel(
            benchmarkJson = benchmarkJson,
            model = resident.model,
            nGpuLayers = nGpuLayers,
            contextSize = ready.contextSize,
            mmprojPath = mmprojPath,
            progress = progress,
            cooldown = cooldown,
            thermal = thermal,
          )
        is Resident.Onnx -> OnnxGenaiBenchmark.runOnModel(benchmarkJson, resident.model, progress, cooldown, thermal)
      }
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
    if (OnnxGenai.isPackage(modelPath)) {
      OnnxGenaiBenchmark.runFresh(benchmarkJson, modelPath, progress, cooldown, thermal)
    } else {
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
