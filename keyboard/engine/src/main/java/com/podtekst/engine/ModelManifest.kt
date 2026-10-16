package com.podtekst.engine

import org.json.JSONObject

/** One downloadable model as listed in the private repo's manifest.json. */
data class ManifestModel(val tier: ModelTier, val file: String, val version: String, val bytes: Long, val sha256: String) {
    fun toJson(): String = JSONObject()
        .put("tier", tier.id).put("file", file).put("version", version).put("bytes", bytes).put("sha256", sha256)
        .toString()

    companion object {
        fun fromJson(json: String): ManifestModel = ModelManifest.parse("""{"models": [$json]}""").single()
    }
}

object ModelManifest {
    /** Parses manifest.json. Unknown tiers are skipped; unsafe file names are rejected. */
    fun parse(json: String): List<ManifestModel> {
        val arr = JSONObject(json).getJSONArray("models")
        return (0 until arr.length()).mapNotNull { i ->
            val o = arr.getJSONObject(i)
            val tier = ModelTier.fromId(o.optString("tier")) ?: return@mapNotNull null
            val file = o.getString("file")
            require(file.endsWith(".litertlm") && '/' !in file && '\\' !in file && !file.startsWith(".")) {
                "bad model file name: $file"
            }
            ManifestModel(tier, file, o.getString("version"), o.getLong("bytes"), o.getString("sha256").lowercase())
        }
    }

    fun pick(models: List<ManifestModel>, tier: ModelTier): ManifestModel? = models.firstOrNull { it.tier == tier }
}
