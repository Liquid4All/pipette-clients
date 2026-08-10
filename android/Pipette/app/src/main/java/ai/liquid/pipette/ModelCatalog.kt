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
