package ai.liquid.pipette

/**
 * What [JobRunner] needs of the provisioning layer, so its tests can drive the four arms without WorkManager, a network, or a Context.
 *
 * A seam rather than the concrete class for the same reason `isOnline` is injected there: the runner's suite is hermetic, and the interesting cases
 * here are the ones a real downloader is least able to produce on demand.
 */
interface ModelProvisioning {
  fun ensureWeights(ref: Model, onStatus: (String) -> Unit = {}): Result<String>

  fun ensureMmproj(ref: Model, onStatus: (String) -> Unit = {}): Result<String>?
}

/**
 * Puts a cell's weights on disk before the cell runs, from the coordinate the cell names ([JobCell.modelRef]).
 *
 * The Android counterpart of iOS `Artifacts/Ensure.swift` (`ensureModel`), which `RunCell.prepare` calls before every cell. Until this existed a cell
 * whose file was absent failed outright, which is why a run could only ever name a model somebody had already downloaded through the Models tab, and
 * why the headless surface refused `model=` at all.
 *
 * Blocking on purpose. [JobRunner] drives its cells on a single-thread executor in the MAIN process (`AndroidManifest.xml` puts
 * `JobOrchestratorService` there, and only `BenchmarkActivity`/`PipetteBenchmarkService` in `:benchmark`), so this runs on the same thread that
 * already blocks for the length of an inference. There is no process boundary between the runner and the downloader.
 *
 * ## Why it polls rather than taking a callback
 *
 * [DownloadCoordinator.enqueueDownload] offers `onComplete`/`onFailure`, but [DownloadRegistry] holds exactly ONE callback slot per key and
 * `setCallbacks` overwrites it. Registering here would silently take the slot the Models screen is using for its progress row, so a user watching a
 * download would see it stop moving. Polling the state the worker already persists costs a `SharedPreferences` read per second and steals nothing.
 *
 * It also survives a transfer this provisioner did not start: the four arms below include a download already in flight (the user began it from the
 * Models tab, or a previous cell in this same run needed the same weights) and one left `paused` by a process death, which is the case iOS had to fix
 * separately in PIP-490 once its headless runs started hitting it.
 */
class ModelProvisioner(
  /** `LocalStorage.resolveModelPath`. A function rather than the store, because that one call is all of it this needs. */
  private val resolvePath: (String) -> String?,
  private val downloads: Downloads,
  /** The persisted transfer rows, re-read each poll: the worker updates them from another thread and a snapshot would never see the change. */
  private val records: () -> List<DownloadStore.Record>,
  private val isOnline: () -> Boolean,
  private val deadlineMillis: Long = DEFAULT_DEADLINE_MILLIS,
  private val pollMillis: Long = DEFAULT_POLL_MILLIS,
  private val sleep: (Long) -> Unit = { Thread.sleep(it) },
  private val now: () -> Long = { System.currentTimeMillis() },
) : ModelProvisioning {
  /**
   * The two things this asks of the download layer, so its own suite needs no WorkManager and no Context.
   *
   * [DownloadCoordinator] satisfies it through an adapter at the construction site rather than by implementing it directly: the coordinator is the
   * app's download API and should not grow a second, narrower identity just because one caller wants less of it.
   */
  interface Downloads {
    fun resume(key: String)

    fun enqueue(url: String, repo: String?)
  }

  /**
   * Ensure the weights [ref] names are on disk, returning their resolved absolute path.
   *
   * Weights and projector are separate entry points rather than one call returning a pair, because [JobRunner] resolves `modelPath` and `mmprojPath`
   * at two sites with two different failure messages, and a pair would have to be taken apart to reach them. It also means a vision pair left
   * half-fetched completes only the half that is missing, which is worth more than it sounds: the two share a repo but not a size, and re-fetching
   * the multi-gigabyte half to recover the small one would be the whole cost of the cell.
   */
  override fun ensureWeights(ref: Model, onStatus: (String) -> Unit): Result<String> =
    when (ref) {
      is Model.GgufText -> ensureOne(ref.model.repo, ref.model.filename, onStatus)
      is Model.GgufVision -> ensureOne(ref.model.repo, ref.model.filename, onStatus)
    }

  /** Ensure the projector a vision [ref] names, or null when [ref] names none. */
  override fun ensureMmproj(ref: Model, onStatus: (String) -> Unit): Result<String>? =
    when (ref) {
      is Model.GgufText -> null
      is Model.GgufVision -> ensureOne(ref.model.repo, ref.model.mmprojFilename, onStatus)
    }

  /**
   * Ensure one file, returning its resolved absolute path.
   *
   * The four arms, in the order they are checked:
   * 1. already on disk, which is every cell after the first that names the same model,
   * 2. left `paused`, which is resumed rather than reported as terminal,
   * 3. in flight, which is waited on rather than re-enqueued (and could not be re-enqueued: `enqueueDownload` rejects a duplicate),
   * 4. absent, which is enqueued.
   */
  @Suppress("ReturnCount") // the already-present arm and the two ways starting a transfer fails, each returning its own reason
  private fun ensureOne(repo: HfRepo, filename: GgufFilename, onStatus: (String) -> Unit): Result<String> {
    val relative = LocalStorage.modelRelativePath(repo.toString(), filename.toString())
    resolvePath(relative)?.let {
      return Result.success(it)
    }

    val key = relative
    val existing = records().firstOrNull { it.key == key }
    when {
      existing?.state == DownloadWorker.STATE_PAUSED -> {
        onStatus("Resuming ${filename}")
        downloads.resume(key)
      }
      existing?.state == DownloadWorker.STATE_FAILED -> {
        // A row this run has not tried yet, left failed by an earlier attempt. Resume rather than refuse: the reason it failed (no network, a
        // full disk) is usually gone by the time a new run starts, and refusing would strand the device on a stale record until someone opened
        // the app. A genuine repeat failure lands in the poll below and reports there.
        onStatus("Retrying ${filename}")
        downloads.resume(key)
      }
      existing != null -> onStatus("Waiting for ${filename}")
      else -> {
        onStatus("Downloading $filename")
        // Through the coordinator's own parser rather than composing a URL here: it owns the `org/repo/file.gguf` form AND the per-segment URL
        // encoding, and a second spelling of the second one would quietly break on the first filename that needs escaping.
        val parsed = runCatching { DownloadCoordinator.parseDownloadInput("$repo/$filename") }
        val input =
          parsed.getOrElse {
            return Result.failure(IllegalStateException("Could not resolve a download URL for $filename: ${it.message}"))
          }
        runCatching { downloads.enqueue(input.url, input.repo) }
          .onFailure {
            return Result.failure(IllegalStateException("Could not start the download for $filename: ${it.message}"))
          }
      }
    }
    return await(key, filename, onStatus)
  }

  /**
   * Wait for the transfer under [key] to reach a terminal state.
   *
   * Terminal is read off what the worker persists, not off a callback: a cleared record with the file present is success (`DownloadWorker` clears the
   * record and then fires `onComplete`), and `STATE_FAILED` is failure. Everything else (`queued`, `running`, `waiting_network`) is still working.
   *
   * `waiting_network` is only patience while the device is online, because WorkManager's backoff genuinely does re-run the transfer. Offline it is a
   * wait with no end, so it reports immediately instead of holding the run to the deadline. iOS draws the same distinction, and had to add
   * `hasReading` to its reachability seam to draw it correctly at launch.
   */
  @Suppress("ReturnCount") // one return per terminal state; a single exit would need a sentinel to say which one was reached
  private fun await(key: String, filename: GgufFilename, onStatus: (String) -> Unit): Result<String> {
    val started = now()
    while (true) {
      val record = records().firstOrNull { it.key == key }
      val resolved = resolvePath(key)
      when {
        // Present wins over any record state: the bytes are what the cell needs, and a row that has not been cleared yet is bookkeeping.
        resolved != null -> return Result.success(resolved)
        record == null -> return Result.failure(IllegalStateException("The download for $filename ended without producing a file"))
        record.state == DownloadWorker.STATE_FAILED ->
          return Result.failure(IllegalStateException("The download for $filename failed. Retry it from the Models tab."))
        record.state == DownloadWorker.STATE_WAITING_NETWORK && !isOnline() ->
          return Result.failure(IllegalStateException("$filename is waiting for a network this device does not have"))
        now() - started > deadlineMillis ->
          return Result.failure(
            IllegalStateException("The download for $filename did not finish within ${deadlineMillis / MILLIS_PER_MINUTE} minutes")
          )
        else -> {
          onStatus("Downloading $filename")
          sleep(pollMillis)
        }
      }
    }
  }

  companion object {
    /**
     * How long one file may take before the cell gives up.
     *
     * Generous because the thing being waited on is a multi-gigabyte transfer over whatever connection a bench device happens to have, and a cell
     * failed for impatience costs the whole run. It bounds only a transfer that is neither progressing nor failing, since a real failure and a
     * confirmed-offline wait both report immediately above.
     */
    const val DEFAULT_DEADLINE_MILLIS = 60L * 60L * 1000L

    /** Cheap: one `SharedPreferences` read and one `File.exists`, against a transfer measured in minutes. */
    const val DEFAULT_POLL_MILLIS = 1_000L

    /** Only to render the deadline in the message a failure carries. */
    private const val MILLIS_PER_MINUTE = 60_000L
  }
}
