package ai.liquid.pipette

import java.io.File

/**
 * Filesystem-backed model catalog. Like the desktop/ops clients (whose `ModelStore` trait is just a directory), Android keeps no model database: the
 * catalog is a scan of [modelsDir] with metadata derived from the repo-bucketed path, the filename, and the bundled model templates. (Replaces the
 * former Room-backed cache.)
 */
class ModelStore(private val modelsDir: File) {

  // Cache the directory scan so the imperative full-rebuild render (which runs every ~500ms while a download progresses) doesn't walk the model
  // directory and stat every .gguf on the main thread each tick. Invalidated whenever the catalog changes (register/delete/clear).
  @Volatile private var cache: List<ModelFile>? = null

  /** The current catalog, scanning the model directory only when the cache is cold. */
  fun availableModels(): List<ModelFile> = cache ?: scan().also { cache = it }

  private fun scan(): List<ModelFile> {
    modelsDir.mkdirs()
    return modelsDir
      .walkTopDown()
      .filter { it.isFile && it.extension.equals("gguf", ignoreCase = true) }
      .sortedBy { it.name.lowercase() }
      .map { modelFileFor(it) }
      .toList()
  }

  /** Drop the cached scan so the next [availableModels] re-reads the directory. */
  fun invalidate() {
    cache = null
  }

  /**
   * A downloaded/imported file needs no registration — the next scan surfaces it. Kept for call-site compatibility; returns the derived [ModelFile],
   * honoring any provenance the caller already knows, and invalidates the cache so the new file shows up.
   */
  fun registerModel(file: File, repo: String?, displayName: String? = null, familyId: String? = null): ModelFile {
    invalidate()
    return modelFileFor(file, repoOverride = repo, displayNameOverride = displayName, familyIdOverride = familyId)
  }

  /** Nothing is persisted, so there is nothing to clear but the in-memory cache. */
  fun clear() = invalidate()

  private fun modelFileFor(
    file: File,
    repoOverride: String? = null,
    displayNameOverride: String? = null,
    familyIdOverride: String? = null,
  ): ModelFile {
    val repo = (repoOverride ?: repoFromBucketPath(file))?.takeIf { it.isNotBlank() }
    val familyId = familyIdOverride ?: LocalStorage.normalizedModelStem(file.name)
    val displayName = displayNameOverride ?: repo?.let { ModelTemplateCatalog.repoToName[it] }
    return ModelFile(
      name = file.name,
      path = file.absolutePath,
      sizeBytes = file.length(),
      hfRepo = repo,
      displayName = displayName,
      familyId = familyId,
      modelRef = refFor(repo, file.name),
    )
  }

  /**
   * The coordinate a discovered file can be re-fetched from, or null when its layout does not name one.
   *
   * Derived from the bucket path rather than looked up in [ModelTemplateCatalog], so a model downloaded from a repo the preset catalog has never
   * heard of still gets one. The Models tab accepts any `org/repo/file.gguf`, and those are exactly the models no preset could describe.
   *
   * Always the text variant. A vision pairing is composed later by `JobRunner.visionRef`, which reads this ref for the weights half and takes the
   * projector from the pairing the picker made; a projector's own ref is unused.
   *
   * Without this, the models with no coordinate would be precisely the ones already on disk, so a cell whose file was deleted between planning and
   * execution could not recover it, which is the case [JobCell.modelRef] exists to cover.
   */
  private fun refFor(repo: String?, filename: String): Model? {
    val hfRepo = repo?.let { HfRepo.parseSlugOrNull(it) }
    val gguf = GgufFilename.parseOrNull(filename)
    return if (hfRepo == null || gguf == null) null else Model.GgufText(HfGgufText(hfRepo, gguf))
  }

  private fun relativePath(file: File): String = file.relativeTo(modelsDir).invariantSeparatorsPath

  private fun repoFromBucketPath(file: File): String? {
    val parts = relativePath(file).split("/")
    return if (parts.size > 1) parts.dropLast(1).joinToString("/") else null
  }
}
