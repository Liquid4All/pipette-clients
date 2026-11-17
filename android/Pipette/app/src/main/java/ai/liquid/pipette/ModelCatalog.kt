package ai.liquid.pipette

data class ModelGroup(val key: String, val name: String, val files: List<ModelFile>) {
  val quantCountLabel: String
    get() = "${files.size} quant${if (files.size == 1) "" else "s"}"

  val quantSummary: String
    get() {
      val quants = files.mapNotNull { it.quant }.distinct()
      return if (quants.isEmpty()) quantCountLabel else quants.joinToString(", ")
    }
}

object ModelCatalog {
  /**
   * Everything a run may be started against: what is on disk, plus the rest of the preset catalog.
   *
   * A benchmark may name a model that has not been fetched, because [JobRunner] fetches it before the cell runs ([ModelProvisioning]). So restricting
   * the picker to downloads was a rule the run path never needed, and it was the reason a fresh device could not start a run at all without a trip
   * through the Models tab first. The headless surface hit the same wall harder, having no tab to send anyone to.
   *
   * A downloaded row wins a collision, because only it knows the real on-disk size and path. The catalog's figure is the declared download size, an
   * estimate parsed out of a label.
   *
   * Collision is judged on the model's coordinate, not on the whole [ModelFile]: the two forms disagree about size and about whether the path exists
   * yet, so comparing values would list the same model twice. iOS keys `offerable` the same way and for the same reason.
   */
  fun offerable(
    downloaded: List<ModelFile>,
    presets: List<PresetModel> = ModelTemplateCatalog.defaults,
    packages: List<OnnxPreset> = ModelTemplateCatalog.onnxDefaults,
    // Whether this build has the engine that runs an ONNX package. Without it none is offered, from the catalog or from disk: each would be a
    // run that fails at its first cell.
    onnxEngine: Boolean = BuildConfig.ONNX_GENAI_ENGINE,
  ): List<ModelFile> {
    val byCoordinate = linkedMapOf<String, ModelFile>()
    presets.forEach { preset -> preset.asOffer()?.let { byCoordinate[it.path] = it } }
    if (onnxEngine) packages.forEach { preset -> preset.asOffer()?.let { byCoordinate[it.path] = it } }
    // Second, so it overwrites the offer for a model already present. Keyed on the same relative path the offer used, which is what
    // `LocalStorage.modelRelativePath` produces for both.
    // A package on disk is offered only where an engine loads it: it is still listed and can be deleted, but not started.
    downloaded.filter { onnxEngine || it.modelRef !is Model.OnnxGenai }.forEach { model -> byCoordinate[coordinateKey(model)] = model }
    return byCoordinate.values.toList()
  }

  /**
   * The key both forms of one model agree on: its `repo/filename` relative path.
   *
   * A downloaded file's [ModelFile.path] may be absolute, so it is recomposed from the repo and name rather than used directly. An import with no
   * repo keys on its filename alone, which is also what its download path would have been.
   */
  private fun coordinateKey(model: ModelFile): String =
    // A package's key is its directory, not its name: the name is the directory's last segment alone.
    (model.modelRef as? Model.OnnxGenai)?.model?.relativeDir ?: LocalStorage.modelRelativePath(model.hfRepo, model.name)

  /**
   * One file of a coordinate as something a run can be planned against, preferring the downloaded row when there is one.
   *
   * Preferring it matters for the same reason it wins in [offerable]: only the on-disk row knows the real size and the absolute path, and a run
   * planned against the offer form would send the runner through a fetch to discover what was already there.
   *
   * The ref is always the **text** variant, even for one half of a vision pair. `JobRunner.visionRef` composes the vision coordinate itself from the
   * weights' text ref plus whichever projector the run paired with them, so handing it a `GgufVision` ref here would leave it with nothing to read.
   *
   * [sizeBytes] is 0 on the offer form because nothing has measured it. A coordinate carries no declared size the way a preset does, so an invented
   * figure would be a guess presented as a fact in the run summary.
   */
  fun resolve(repo: HfRepo, filename: GgufFilename, downloaded: List<ModelFile>): ModelFile {
    val relative = LocalStorage.modelRelativePath(repo.toString(), filename.toString())
    downloaded
      .firstOrNull { coordinateKey(it) == relative }
      ?.let {
        return it
      }
    return ModelFile(
      name = filename.toString(),
      path = relative,
      sizeBytes = 0L,
      hfRepo = repo.toString(),
      displayName = ModelTemplateCatalog.repoToName[repo.toString()],
      familyId = LocalStorage.normalizedModelStem(filename.toString()),
      modelRef = Model.GgufText(HfGgufText(repo, filename)),
      onDisk = false,
    )
  }

  /** The one or two files a `model=` coordinate names, each already resolved against what is on disk. */
  data class ResolvedCoordinate(val weights: ModelFile, val projector: ModelFile?)

  /**
   * A parsed coordinate as the files a run is planned against.
   *
   * Both halves of a vision pair go through [resolve], so either can already be on disk while the other is fetched. That is the case worth getting
   * right: the projector is small and the weights are not, and recovering one should not re-transfer the other.
   */
  fun resolveCoordinate(ref: Model, downloaded: List<ModelFile>): ResolvedCoordinate =
    when (ref) {
      is Model.GgufText -> ResolvedCoordinate(resolve(ref.model.repo, ref.model.filename, downloaded), null)
      is Model.GgufVision ->
        ResolvedCoordinate(resolve(ref.model.repo, ref.model.filename, downloaded), resolve(ref.model.repo, ref.model.mmprojFilename, downloaded))
      is Model.OnnxGenai -> ResolvedCoordinate(resolvePackage(ref, downloaded), null)
    }

  /**
   * An ONNX Runtime GenAI package as something a run can be planned against, preferring the downloaded row when there is one, as [resolve] does.
   *
   * Matched on the coordinate itself: a package's name is its directory's last segment, which two packages of one repo can share with nothing but the
   * directories above it to tell them apart. The offer form's path is the directory the package will have under the models directory, which the
   * runner resolves through [ModelProvisioning.ensureWeights].
   */
  fun resolvePackage(ref: Model.OnnxGenai, downloaded: List<ModelFile>): ModelFile {
    val onDisk = downloaded.firstOrNull { (it.modelRef as? Model.OnnxGenai)?.model?.relativeDir == ref.model.relativeDir }
    val pin = ref.model.revision
    // The listed row carries the commit the fetcher recorded for it ([ModelStore]). A pinned coordinate takes the row only when that is the
    // commit it names; for another, for a tag or a branch, or for a package with no record, it keeps the offer form, and the fetcher, which
    // can resolve a name and refuses a mismatch, answers when the cell runs.
    val satisfied = pin == null || (onDisk?.modelRef as? Model.OnnxGenai)?.model?.revision == pin
    return onDisk?.takeIf { satisfied }?.copy(modelRef = ref) ?: packageFile(ref, path = ref.model.relativeDir, sizeBytes = 0L, onDisk = false)
  }

  /** A package as a [ModelFile]: one model whose path is its directory. Shared by the offer above and [ModelStore]'s scan, so both name it alike. */
  fun packageFile(ref: Model.OnnxGenai, path: String, sizeBytes: Long, onDisk: Boolean): ModelFile =
    ModelFile(
      name = ref.model.name,
      path = path,
      sizeBytes = sizeBytes,
      hfRepo = ref.model.repo.toString(),
      // The repo's name with the directory's, since a repo holding several packages gives each the same first half.
      displayName =
        ModelTemplateCatalog.repoToName[ref.model.repo.toString()]
          ?: listOfNotNull(ref.model.repo.repoName.value, ref.model.prefix?.leaf).joinToString(" "),
      familyId = ref.model.relativeDir.lowercase(),
      modelRef = ref,
      onDisk = onDisk,
    )

  fun groups(from: List<ModelFile>): List<ModelGroup> {
    val keys = mutableListOf<String>()
    val seen = mutableSetOf<String>()
    from.forEach { model ->
      val key = groupKey(model)
      if (seen.add(key)) keys += key
    }
    val byKey = from.groupBy { groupKey(it) }
    return keys.mapNotNull { key ->
      val files = byKey[key] ?: return@mapNotNull null
      ModelGroup(
        key = key,
        name =
          ModelTemplateCatalog.byFamilyId[key]?.displayName
            ?: files.firstOrNull()?.displayName
            ?: LocalStorage.modelStem(files.firstOrNull()?.name ?: key),
        files = files,
      )
    }
  }

  fun groupKey(model: ModelFile): String = model.familyId ?: LocalStorage.normalizedModelStem(model.name)

  fun resolveSelectedFiles(groups: List<ModelGroup>, selectedKeys: Set<String>, quantMatches: (String?) -> Boolean): List<ModelFile> =
    groups.filter { selectedKeys.contains(it.key) }.flatMap { group -> group.files.filter { quantMatches(it.quant) } }

  fun selectedGroupsMissingQuant(groups: List<ModelGroup>, selectedKeys: Set<String>, quantMatches: (String?) -> Boolean): List<ModelGroup> =
    groups.filter { selectedKeys.contains(it.key) }.filterNot { group -> group.files.any { quantMatches(it.quant) } }
}
