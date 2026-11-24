package ai.liquid.pipette

import java.io.File

/** Files that these cells can provision. Multiple cells can refer to the same file. */
internal fun pendingJobDownloads(cells: List<JobCell>, models: List<ModelFile>): List<ModelFile> {
  val paths = cells.filter { it.modelRef != null }.flatMap { listOfNotNull(it.modelPath, it.mmprojPath) }.toSet()
  return models.filter { !it.onDisk && it.path in paths }.distinctBy { it.storageKey }
}

/** Check the whole batch before the runner begins provisioning any cell. */
internal fun checkJobDownloadSpace(modelsRoot: File, downloads: List<ModelFile>, lookupSize: (ModelFile) -> Long?) {
  if (downloads.isEmpty()) return
  val filesAndSizes =
    downloads.mapNotNull { model ->
      val file = File(modelsRoot, model.storageKey)
      if (file.isFile) return@mapNotNull null
      val size =
        model.sizeBytes.takeIf { it > 0 }
          ?: lookupSize(model)?.takeIf { it > 0 }
          ?: error("Could not check the download size for ${model.name}. Try again when the model host is available.")
      file to size
    }
  val remainingBytes =
    filesAndSizes.fold(0L) { total, (file, size) ->
      if (file.isFile) return@fold total
      val partialBytes = File(file.parentFile, "${file.name}.part").length()
      val remaining = (size - partialBytes).coerceAtLeast(0L)
      total + remaining.coerceAtMost(Long.MAX_VALUE - total)
    }
  val free = modelsRoot.usableSpace
  check(remainingBytes <= free) {
    "Not enough storage for this job's downloads. ${ByteFormat.fileSize(remainingBytes)} needed, ${ByteFormat.fileSize(free)} available."
  }
}
