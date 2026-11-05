package ai.liquid.pipette.headless

import ai.liquid.pipette.IdentityBundle
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobRunner
import ai.liquid.pipette.ModelCatalog
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.PrivateKeyHex

/**
 * What a mutating verb targets, decided against [HeadlessInspect.Facts] and nothing else.
 *
 * The same split the read-only verbs use: [HeadlessFacts] gathers, [HeadlessInspect] words the answer, and neither decides anything the other could
 * have. Here the deciding is separated from the doing for a sharper reason than symmetry. A verb that deletes has exactly one interesting question,
 * "which thing, and should it happen at all", and answering it against a plain data structure is what lets that be settled off a device.
 * [HeadlessAct] carries the decision out, and is tested separately against a real filesystem.
 *
 * A refusal names what was looked for whenever the caller supplied a selector, so a script batching several invocations can tell which one was turned
 * away. It always names *why*: "no such job" and "a job is running" want different responses, and only one of them is worth waiting out.
 */
object HeadlessMutate {

  /** The resolved target of a mutating verb, or the reason there is not one. */
  sealed interface Outcome {
    /** Delete the whole job directory: manifest, payloads, results. */
    data class DeleteJob(val jobId: String) : Outcome

    /** Write [manifest]'s cells to a CSV, every one of them, each carrying its own status column. */
    data class ExportJob(val manifest: JobManifest) : Outcome

    /**
     * Clear one cell's stored artifacts.
     *
     * Carries the job id as well as the cell id because that is what the store's `clearCellArtifacts` takes, and a cell id alone cannot find its job
     * without the walk this resolution already did.
     */
    data class DeleteResult(val jobId: String, val cellId: String) : Outcome

    /**
     * Delete these files.
     *
     * A list rather than one file because both model verbs can legitimately name more than one: a coordinate resolves to its weights and, for a
     * vision pair, its projector; and `repo=` on its own names every quant of that repo. Deleting the weights of a VL pair while leaving the
     * projector behind would leave an unusable orphan taking up the space the caller was trying to reclaim.
     */
    data class DeleteModels(val files: List<ModelFile>) : Outcome

    /**
     * Restore [bundle] onto this device.
     *
     * Carries the decoded bundle rather than the token, so the whole decision — is it a bundle, is its key a key, does it replace another identity —
     * is settled here and nothing downstream can reach a half-read record.
     *
     * [observedKey] is the promoted signing key this decision read, and it does two jobs. It answers [replacesAnotherKey], the fact that made
     * `force=` necessary, which `HeadlessAct` cannot recompute because it writes the bundle's key first. It is also what `Secrets` compares the
     * promoted slot against before it writes, so an import that a concurrent registration overtook writes nothing.
     *
     * [forced] is `force=1` as the caller gave it, carried because it answers a second question this decision cannot see. A registration killed
     * between staging a key and promoting it strands that key, and nothing on this surface clears the slot, so an unforced import would refuse for as
     * long as the install lasts. `force=1` is how an operator who knows the device is idle says so.
     */
    data class ImportIdentity(val bundle: IdentityBundle, val observedKey: PrivateKeyHex?, val forced: Boolean = false) : Outcome {
      /** Whether this import overwrites a key that signs for another client. */
      val replacesAnotherKey: Boolean
        get() = observedKey != null && !observedKey.value.equals(bundle.privateKeyHex.value, ignoreCase = true)
    }

    /** Nothing was done, and [reason] says why. */
    data class Refused(val reason: String) : Outcome
  }

  /**
   * Resolve [query] against [facts]. Pure: it reads the gathered world and returns a decision, touching no disk.
   *
   * The `else` is not a hole to fill in later. `HeadlessCommand` only produces a `Parsed.Mutate` for a verb whose `Kind` is MUTATE, so a verb
   * reaching that arm means the parse and this function disagree about what mutates, which is worth saying out loud rather than crashing. What the
   * arm does cost is compile-time exhaustiveness: a MUTATE verb added to [HeadlessVerb] and forgotten here would build and then refuse at runtime.
   * `mutatingVerbsAllResolve` in the tests closes that gap by walking `HeadlessVerb.entries` and failing on any MUTATE verb that lands here.
   */
  fun resolve(query: HeadlessCommand.Query, facts: HeadlessInspect.Facts): Outcome =
    when (query.verb) {
      // Not wrapped in [guardRunning]: it makes the same refusal itself, and can name the job it declined to delete.
      HeadlessVerb.JOB_RM -> deleteJob(query.jobId, facts)
      HeadlessVerb.JOB_EXPORT -> exportJob(query.jobId, facts)
      HeadlessVerb.RESULTS_DELETE -> guardRunning(facts) { deleteResult(query.resultId, facts) }
      HeadlessVerb.MODELS_DELETE -> guardRunning(facts) { deleteByCoordinate(query, facts) }
      HeadlessVerb.MODELS_RM -> guardRunning(facts) { deleteByName(query, facts) }
      HeadlessVerb.AUTH_IMPORT -> guardRunning(facts) { importIdentity(query, facts) }
      else -> Outcome.Refused("${query.verb.spelling} is not a mutating verb")
    }

  /**
   * Nothing is deleted while a job runs.
   *
   * Covers the three verbs wrapped here, `models-delete`, `models-rm` and `results-delete`, and each fails silently in its own way. `JobRunner` loads
   * a model file per cell, so deleting weights it has not reached yet fails every later cell of a run that was going to succeed, and
   * `clearCellArtifacts` races the payload write of the cell in flight. Neither throws anywhere the caller can see: the delete reports success and
   * the damage surfaces later, in a run nobody was watching. [deleteJob] refuses for its own third reason and says so itself.
   *
   * `auth-import` is wrapped too, and its quiet failure is the worst of the set: `ResultSubmissionService` signs each submission as it is made, so a
   * key swapped mid-run makes every cell that finishes afterwards report under the restored `client_id`. The run reports success and the warehouse
   * ends up with one device's measurements split across two clients. Nothing throws, and nothing on the device records that it happened.
   *
   * `job-export` is deliberately outside the guard: it reads the manifest and writes a CSV of its own, so a run in flight makes its output incomplete
   * but not wrong, and refusing it would deny a caller the results they already have.
   *
   * The target is resolved first and a refusal from that resolution wins, which keeps the ordering [deleteJob] argues for.
   */
  private inline fun guardRunning(facts: HeadlessInspect.Facts, resolveTarget: () -> Outcome): Outcome =
    resolveTarget().let { outcome ->
      if (facts.jobRunning && outcome !is Outcome.Refused) Outcome.Refused("a job is running; stop it and try again") else outcome
    }

  /**
   * A job that is currently running is refused rather than deleted.
   *
   * `JobRunner` holds the manifest open and writes payloads into the directory as cells finish, so removing it mid-run leaves the runner writing into
   * a path that no longer exists and the caller with a partial delete they did not ask for. The Jobs screen has the same hole and does not check;
   * this surface is the unattended one, so it does not inherit that. Refused whichever job is running, since `jobRunning` reports only that one is,
   * and a script can wait. Worded here rather than by [guardRunning] because at this point the job is in hand and can be named.
   */
  private fun deleteJob(jobId: String?, facts: HeadlessInspect.Facts): Outcome {
    val manifest = facts.manifests.firstOrNull { it.jobId == jobId }
    // Ordered: a job that is not here is reported as absent even while another one runs, because "no such job" is the actionable answer and "a job is
    // running" would send the caller to wait for something that was never going to help. [guardRunning] keeps the same order for the verbs it wraps.
    return when {
      manifest == null -> Outcome.Refused("no job with id '$jobId'")
      facts.jobRunning -> Outcome.Refused("a job is running; stop it before deleting job ${manifest.jobId}")
      else -> Outcome.DeleteJob(manifest.jobId)
    }
  }

  /** Export needs the manifest itself, not just its id, since the CSV is built from its cells. */
  private fun exportJob(jobId: String?, facts: HeadlessInspect.Facts): Outcome =
    facts.manifests.firstOrNull { it.jobId == jobId }?.let { Outcome.ExportJob(it) } ?: Outcome.Refused("no job with id '$jobId'")

  /**
   * A cell id is a UUID and unique across jobs, so the walk that gathered the results is what maps it back to its job.
   *
   * The population is every cell that completed with a payload still on disk, which is the right one to delete from: a cell with nothing stored has
   * no result to delete, and saying "no result" is more use than silently succeeding. Note it is payload-on-disk and not metric-derivable, so a
   * payload this client cannot read is still reachable here, which is the only way its bytes can be reclaimed.
   */
  private fun deleteResult(cellId: String?, facts: HeadlessInspect.Facts): Outcome =
    facts.results.firstOrNull { it.cellId == cellId }?.let { Outcome.DeleteResult(it.jobId, it.cellId) }
      ?: Outcome.Refused("no stored result with cell id '$cellId'")

  /**
   * The coordinate is resolved the same way a run resolves it, through [ModelCatalog.resolveCoordinate], so `models-delete` and `run` agree about
   * which bytes a URI names. A coordinate that resolves to something not on disk is refused rather than reported as a no-op delete: the caller asked
   * to reclaim space and none was reclaimed.
   */
  private fun deleteByCoordinate(query: HeadlessCommand.Query, facts: HeadlessInspect.Facts): Outcome {
    // Defensive rather than reachable: `HeadlessCommand` requires `model=` for this verb, so a caller sees the parser's usage error first. Worded as
    // a statement of fact instead of repeating that usage sentence, which would be a second copy to keep in step with it.
    val uri = query.modelUri ?: return Outcome.Refused("no coordinate was supplied")
    // The projector is null for a text coordinate, and the weights entry is synthesized even when nothing is on disk, so both filters below earn
    // their place: `listOfNotNull` for the half a coordinate does not name, `onDisk` for the half it names but the device does not have. Keeping the
    // two apart is what makes "resolved but nothing here" a distinct answer from "resolved to bytes": the caller asked to reclaim space, and if none
    // would be reclaimed they should hear so rather than read a successful delete of nothing.
    val resolved = ModelCatalog.resolveCoordinate(uri, facts.models)
    val weights = resolved.weights.takeIf { it.onDisk }
    val projector = resolved.projector?.takeIf { it.onDisk && !stillPairedWithAnother(it, weights, facts.models) }
    val files = listOfNotNull(weights, projector)
    // Quotes `model=` as the caller wrote it rather than the parsed value: see [HeadlessCommand.Query.modelSpelling].
    return if (files.isEmpty()) Outcome.Refused("no bytes on device for model='${query.modelSpelling}'") else Outcome.DeleteModels(files)
  }

  /**
   * `name=` matches the file name exactly and `repo=` the Hugging Face repo, and passing both narrows rather than widens.
   *
   * Exact rather than substring, unlike `run`'s `match=`. The two read alike and mean opposite things: a substring that selects one model to
   * benchmark costs a wrong measurement, while a substring that selects models to delete costs the models. `run` is where a caller wants a loose
   * handle; a delete is where they want to have named the thing.
   *
   * **A deliberate divergence from iOS, which lets `name=` win when both are given** (`ModelCommands.remove`, whose filter returns on `name` before
   * it looks at `repo`). There, `name=X repo=Y` deletes every X on the device; here it deletes the X under Y and nothing else. Narrowing is the safe
   * reading of two selectors on a delete, and a caller who did mean every copy can say `name=X` alone. Recorded in `docs/client-parity.md`.
   *
   * `name=` on its own does span repos, which is iOS's behaviour and worth knowing for a file name several repos reuse: `mmproj-model-f16.gguf` names
   * a projector in every VL repo that has one. The `files=` list printed after the delete says which ones went.
   */
  private fun deleteByName(query: HeadlessCommand.Query, facts: HeadlessInspect.Facts): Outcome {
    // With neither selector every filter below passes and this resolves to every model on the device. `HeadlessCommand` refuses that invocation and
    // is the only thing that builds a Query, so the guard is unreachable today; it is here because of what it guards. The same defence sits on
    // `deleteByCoordinate`, which risks less.
    val matched =
      facts.models.filter { model -> (query.name == null || model.name == query.name) && (query.repo == null || model.hfRepo == query.repo) }
    return when {
      query.name == null && query.repo == null -> Outcome.Refused("no selector was supplied; models-rm will not delete every model")
      matched.isEmpty() -> Outcome.Refused("no model on device matches ${describeSelector(query)}")
      else -> Outcome.DeleteModels(matched)
    }
  }

  /**
   * Decide the whole import before anything is written.
   *
   * Four questions in order, and the order is the point: a bundle that is not readable refuses before the device is consulted at all, so a mistyped
   * token leaves the identity exactly as it was.
   * 1. [IdentityBundle.validated] — base64url, version, the full record, and proof the hex is a key.
   * 2. Whether the device holds a key it cannot decrypt, which [HeadlessInspect.Facts.privateKeyUnreadable] reports.
   * 3. Whether the device holds a *different* key. Compared against the bundle's own rather than merely tested for presence, and compared
   *    case-insensitively because the value is hex. That comparison is what lets a rerun after a half-finished import complete the restore: the key
   *    it finds is the one that same bundle wrote, which is no conflict.
   * 4. Whether `force=1` was given, which is only asked once a genuine conflict is established.
   *
   * The refusal in step 4 guards data this verb never touches — the results already submitted under the key it would overwrite — which is why it is
   * the one refusal on this surface that a caller has to override rather than resolve. iOS refuses in the same place and for the same reason, and the
   * CLI refuses `auth register` over an existing identity on the same grounds.
   *
   * Step 2 is the one refusal `force=1` does not lift, because forcing cannot express a decision the operator is able to make. A slot the Keystore
   * cannot open right now reads exactly like an empty one, so the caller cannot tell a fresh device from the identity they are about to destroy, and
   * step 3 cannot either. The outage passes and the key comes back, so the refusal is worth retrying rather than overriding. A device whose Keystore
   * is broken for good still restores, by clearing the identity from Settings first.
   */
  @Suppress("ReturnCount") // one refusal per question, in the order the KDoc lists them; folding them in would hide that ordering
  private fun importIdentity(query: HeadlessCommand.Query, facts: HeadlessInspect.Facts): Outcome {
    // Defensive rather than reachable, as on [deleteByCoordinate]: `HeadlessCommand` requires `bundle=` for this verb.
    val token = query.bundle ?: return Outcome.Refused("no bundle was supplied")
    val bundle =
      when (val decoded = IdentityBundle.validated(token)) {
        is IdentityBundle.Result.Error -> return Outcome.Refused(decoded.failure.reason)
        is IdentityBundle.Result.Ok -> decoded.bundle
      }
    // Appended to both refusals below, because `HeadlessRunActivity` mirrors an `auth-import` refusal into the `pipette-auth` stream, and a refusal
    // recorded there without a client id says something was declined and nothing about what. Two ids, because a refusal fires exactly when they
    // differ: `client_id` is the identity that was protected, and it keeps meaning "this device" as it does on every other line of this surface, so
    // the one somebody offered takes a key of its own. The device's id is dropped when there is no registration, on the same grounds as `model`'s
    // absent `repo`: an empty `client_id=` sends a reader looking for a value that was never there.
    val identities = buildString {
      facts.registration?.clientId?.let { append(" client_id=$it") }
      append(" bundle_client_id=${bundle.registration.clientId}")
    }
    if (facts.privateKeyUnreadable) {
      return Outcome.Refused(
        "this device holds a signing key the Android Keystore cannot open, so this import cannot tell whether it would replace another client's " +
          "identity. Try again; force=1 does not override this. Clear the identity from Settings if the key is gone for good" +
          identities
      )
    }
    val outcome = Outcome.ImportIdentity(bundle, facts.privateKeyHex, forced = query.force)
    return if (outcome.replacesAnotherKey && !query.force) {
      Outcome.Refused(
        "refusing over the signing key this device already holds: it signs for a different client, and an overwrite would orphan that client's " +
          "results. Clear the identity from Settings, or pass force=1" +
          identities
      )
    } else {
      outcome
    }
  }

  /**
   * Whether a model other than the one being deleted still needs [projector].
   *
   * The projector travels with its weights on iOS, where `ModelArtifactStore` keeps each spec's files as its own entry, so removing a vision spec
   * cannot take a file another spec is using. Android stores by coordinate instead, `models/<repo>/<filename>`, which makes the projector **one file
   * that every quant of that repo pairs with**: `LFM2.5-VL-GGUF/mmproj-model-f16.gguf` is the same bytes for the Q4_0 and the Q8_0.
   *
   * So deleting a vision coordinate unconditionally would take the projector out from under the models left behind, and leave them unable to run the
   * thing they were kept for.
   *
   * Which models those are is [JobRunner.isVlCompatible]'s question, not one to answer again here. Its rule is same repo **or** the same normalized
   * stem, and the second half is load-bearing: a projector can serve a VL model in another repo entirely, which a repo-only test would miss and then
   * delete out from under it.
   *
   * The other direction is already handled: when nothing else needs it, the projector goes with the weights, because leaving it would be an orphan
   * occupying the space the caller asked to reclaim.
   */
  private fun stillPairedWithAnother(projector: ModelFile, deleting: ModelFile?, models: List<ModelFile>): Boolean =
    models.any { it.onDisk && !it.isMmproj && it.path != deleting?.path && JobRunner.isVlCompatible(it, listOf(projector)) }

  /** The selector as the caller wrote it, so a refusal quotes back what was looked for rather than a rephrasing of it. */
  private fun describeSelector(query: HeadlessCommand.Query): String =
    listOfNotNull(query.name?.let { "name='$it'" }, query.repo?.let { "repo='$it'" }).joinToString(" ")
}
