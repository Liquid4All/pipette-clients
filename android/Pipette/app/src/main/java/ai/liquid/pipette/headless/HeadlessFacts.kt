package ai.liquid.pipette.headless

import ai.liquid.pipette.AppContainer
import ai.liquid.pipette.BenchmarkCatalog
import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.CompletedResultsCsvExporter
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.LocalStorage
import ai.liquid.pipette.ManagementClient
import ai.liquid.pipette.NativeLib
import ai.liquid.pipette.PrivateKeyHex
import ai.liquid.pipette.RegistrationData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking

/**
 * Gathers what a verb needs off the device, so [HeadlessInspect] and [HeadlessMutate] can stay free of the container and therefore unit-testable.
 *
 * The split is the point: everything that touches disk, a process-global catalog, or the native library lives here, and everything that decides how
 * an answer is *worded* or *which thing to touch* lives there. A format regression is caught by a JVM test; this half is confirmed by the device run.
 */
object HeadlessFacts {

  /** The id `verb=runtimes` reports for the linked engine. See [runtime]. */
  const val RUNTIME_ID = "llama_cpp"

  /** What `CompletedResultsCsvExporter.quantLabel` returns for a filename it cannot parse a quantization out of. */
  private const val UNKNOWN_QUANT = "unknown"

  /**
   * Reads the whole world [query] could ask about.
   *
   * Most of it is gathered unconditionally: it is one directory scan plus one manifest load, both of which the app already does on every Jobs screen
   * render. Three things are conditional, each because of what it costs. [HeadlessInspect.Facts.results] costs a file read per completed cell across
   * every job. The identity costs an *exposure* rather than time, so the private key is read only for the verbs that need it. The profile costs a
   * network round trip, and it is the only part of this function whose cost a server on the other end decides: `ManagementClient.fetchMe`'s own
   * connect and read timeouts are the bound, since a blocking socket read cannot be cancelled from outside.
   */
  fun gather(
    query: HeadlessCommand.Query,
    container: AppContainer,
    packageName: String,
    buildVersion: String,
    engineFlavor: String = ai.liquid.pipette.BuildConfig.ENGINE_FLAVOR,
  ): HeadlessInspect.Facts {
    val storage = container.storage
    val models = storage.availableModels()
    val manifests = storage.loadAllJobManifests()
    val registration = if (query.verb.readsIdentity) storage.loadRegistration() else null
    val privateKeyHex = if (query.verb.readsIdentity) container.secrets.exportPrivateKeyHex()?.let(::PrivateKeyHex) else null
    // Skipped when either half is absent, so the refusal names the missing half rather than reporting whatever a request signed with nothing failed
    // with. Both halves are needed to sign: iOS composes the same pair in `mgmtSession` before it calls the endpoint.
    val fetched =
      if (query.verb.needsNetwork && registration != null && privateKeyHex != null) fetchProfile(container.managementClient, registration) else null
    return HeadlessInspect.Facts(
      buildVersion = buildVersion,
      packageName = packageName,
      engineFlavor = engineFlavor,
      registered = storage.isRegistered(),
      jobRunning = container.jobController.isRunning(),
      models = models,
      manifests = manifests,
      benchmarks = BenchmarkCatalog.all,
      selectableIds = BenchmarkCatalog.selectable.map { it.benchmarkId.toString() }.toSet(),
      modelsDirPath = storage.modelsDir.absolutePath,
      freeBytes = runCatching { storage.modelsDir.usableSpace }.getOrDefault(0L),
      runtime = runtime(),
      results = if (query.verb.readsResults) results(storage, manifests) else emptyList(),
      registration = registration,
      privateKeyHex = privateKeyHex,
      privateKeyUnreadable = privateKeyHex == null && query.verb.readsIdentity && container.secrets.hasUnreadablePrivateKey(),
      hfToken = if (query.verb.readsIdentity) container.secrets.loadHfToken() else null,
      profile = fetched?.getOrNull(),
      profileFailure = fetched?.exceptionOrNull()?.let(HeadlessAct::reason),
    )
  }

  /**
   * `GET /clients/me`, from a synchronous `onCreate`.
   *
   * `runBlocking(Dispatchers.IO)` rather than a suspending surface: `ManagementClient` is blocking `HttpURLConnection`, and Android throws
   * `NetworkOnMainThreadException` for a socket opened on the main thread, so the work has to move to an IO thread. The main thread then blocks on
   * the result, which is what keeps `onCreate` synchronous, and every other verb on this surface assumes that.
   *
   * The main thread it blocks is the app process's own: `HeadlessRunActivity` declares no `android:process`, so it shares one with
   * `ComposeMainActivity` and with `JobOrchestratorService`, which runs there for the length of a job. What that costs is a stall, not a lost run. An
   * unattended device delivers no input to freeze and starts no service to time out, and the benchmark itself runs in `:benchmark`. A `verb=auth-me`
   * against a stalled server holds this thread, and the job's progress mirror catches up afterwards.
   *
   * The wait is bounded by `ManagementClient.fetchMe`'s own connect and read timeouts, because nothing outside the call can cancel a blocking socket
   * read. `fetchMe` asks for a read budget sized to a short record rather than the submission default, so that bound is the caller's whole cost.
   *
   * The failure is carried rather than thrown, so how a refusal is *worded* stays [HeadlessInspect]'s decision like every other refusal here.
   */
  private fun fetchProfile(client: ManagementClient, registration: RegistrationData): Result<ManagementClient.ClientProfile> = runCatching {
    runBlocking(Dispatchers.IO) { client.fetchMe(registration.serverUrl, registration.clientId) }
  }

  /**
   * The one runtime this build links, read from the packaged native library.
   *
   * [HeadlessInspect.Runtime.id] is the capability flag's own spelling (`runtime:llama_cpp`, see `Capabilities`) rather than the plan-types runtime
   * type `llamacpp_apk_pipette`. The flag is what a fleet operator sees against this device on the server, so it is the name they would grep a device
   * log for.
   */
  private fun runtime(): HeadlessInspect.Runtime =
    HeadlessInspect.Runtime(id = RUNTIME_ID, available = NativeLib.isAvailable, build = NativeLib.llamaCppCommit())

  /**
   * Whether a cell is a result: it completed, and its payload file is still on disk.
   *
   * The status alone is not enough, and the difference is the whole of `results-delete`. Clearing a cell's artifacts leaves its manifest entry
   * COMPLETED, because the cell did complete, so filtering on status kept listing a result whose measurement had just been deleted. Found on device,
   * where `results count=` did not move after a delete that reported success.
   *
   * [cellIdsWithPayloads] must be built from the payload file's **existence**, not from whether it could be parsed. The distinction is not academic:
   * a payload that is on disk but unreadable is the one case where the two answers differ, and taking parse-success there would hide the file from
   * `results` and then refuse to delete it, leaving bytes that no verb on this surface can reach. It is the same test `LocalStorage.isUnsubmitted`
   * makes for the same reason, and what iOS spells `payloadPath(of:) != nil`. Note the two counts still differ, and legitimately: the Submit button
   * also requires `serverJobId` to be blank, so a result that has already gone up is listed here and not there.
   *
   * Split out from the walk so the rule can be pinned by a test. The walk itself needs a `LocalStorage` and a real payload directory; this is the
   * half that decides, and it was the half that was wrong.
   */
  internal fun isReportable(cell: JobCell, cellIdsWithPayloads: Set<String>): Boolean =
    cell.runStatus == CellRunStatus.COMPLETED && cell.cellId in cellIdsWithPayloads

  /**
   * Every completed cell's measurement, flattened across [manifests].
   *
   * A cell counts as a result when it completed *and* its payload file is still on disk, which is [isReportable]'s rule and `submittablePayloadDir`'s
   * test. `payloadsByCellId` is not that test even though it looks like one: it parses, and returns nothing for a file it could not read.
   *
   * A payload that is present but yields no metric still produces a row, with [HeadlessInspect.Result.metricValue] null. Two different cells land
   * there, and both are worth seeing: a benchmark type with nothing to reduce to a single number, which ran and submitted normally, and a payload
   * that is on disk but unreadable, which is a fault the caller can act on and bytes they can now reclaim.
   */
  internal fun results(storage: LocalStorage, manifests: List<JobManifest>): List<HeadlessInspect.Result> =
    manifests.flatMap { manifest ->
      val payloads = CompletedResultsCsvExporter.payloadsByCellId(storage, manifest)
      // A stat per completed cell, on top of the read `payloadsByCellId` already pays for every cell in the manifest. Worth naming rather than
      // eliminating: the two answer different questions, and collapsing them is what the device regression was. The exporter's map says "could this
      // payload be parsed", and membership here has to mean "is the file there", so a payload that is present but unreadable stays listed and stays
      // deletable. Restricted to the completed cells because [isReportable] rejects the rest on status alone, so a stat for them could not change
      // the outcome.
      val stored =
        manifest.cells
          .filter { it.runStatus == CellRunStatus.COMPLETED }
          .map { it.cellId }
          .filterTo(mutableSetOf()) { storage.submittablePayloadDir(manifest.jobId, it) != null }
      manifest.cells
        .filter { isReportable(it, stored) }
        .map { cell ->
          val metric = CompletedResultsCsvExporter.metric(cell, payloads[cell.cellId])
          HeadlessInspect.Result(
            jobId = manifest.jobId,
            cellId = cell.cellId,
            benchmarkId = cell.benchmarkId,
            benchmarkType = CompletedResultsCsvExporter.benchmarkType(cell),
            modelName = cell.modelName,
            // `quantLabel` answers an unparseable filename with the string "unknown" rather than null, which the CSV wants as a cell value. Here the
            // key is omitted instead, matching how `model` treats a file with no quant: `quant=unknown` reads as a quantization by that name.
            quant = CompletedResultsCsvExporter.quantLabel(cell).takeIf { it != UNKNOWN_QUANT },
            serverJobId = cell.serverJobId,
            metricValue = metric?.numericValue,
            metricUnit = metric?.unit,
            createdAt = manifest.createdAt,
          )
        }
    }
}
