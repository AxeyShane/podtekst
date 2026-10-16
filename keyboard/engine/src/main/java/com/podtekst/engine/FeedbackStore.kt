package com.podtekst.engine

import org.json.JSONObject
import java.io.File

/** One rated translation. rating: 1 = 👍, -1 = 👎. Field names match the dataset files (snake_case). */
data class FeedbackEntry(
    val id: Long,
    val time: Long,
    val sourceLang: String,
    val sourceText: String,
    val translation: String,
    val category: String,
    val note: String,
    val rating: Int,
    val comment: String,
    val model: String,
    val totalMs: Long,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("id", id).put("time", time).put("source_lang", sourceLang).put("source_text", sourceText)
        .put("translation", translation).put("category", category).put("nuance_note", note)
        .put("rating", rating).put("comment", comment).put("model", model).put("total_ms", totalMs)

    companion object {
        fun fromJson(o: JSONObject) = FeedbackEntry(
            id = o.getLong("id"), time = o.getLong("time"), sourceLang = o.getString("source_lang"),
            sourceText = o.getString("source_text"), translation = o.getString("translation"),
            category = o.getString("category"), note = o.optString("nuance_note"), rating = o.getInt("rating"),
            comment = o.optString("comment"), model = o.optString("model"), totalMs = o.optLong("total_ms"),
        )
    }
}

/** Rated translations in a private JSON-lines file. Small (tens of entries), so rewrites are fine. */
class FeedbackStore(private val file: File) {

    @Synchronized
    fun add(e: FeedbackEntry) {
        file.parentFile?.mkdirs()
        file.appendText(e.toJson().toString() + "\n", Charsets.UTF_8)
    }

    @Synchronized
    fun list(): List<FeedbackEntry> {
        if (!file.exists()) return emptyList()
        return file.readLines(Charsets.UTF_8)
            .filter { it.isNotBlank() }
            .mapNotNull { runCatching { FeedbackEntry.fromJson(JSONObject(it)) }.getOrNull() }
    }

    @Synchronized
    fun update(id: Long, transform: (FeedbackEntry) -> FeedbackEntry) =
        rewrite(list().map { if (it.id == id) transform(it) else it })

    @Synchronized
    fun delete(id: Long) = rewrite(list().filterNot { it.id == id })

    /** Writes all entries to [out] and returns how many. */
    @Synchronized
    fun exportTo(out: File): Int {
        val entries = list()
        out.parentFile?.mkdirs()
        out.writeText(entries.joinToString("") { it.toJson().toString() + "\n" }, Charsets.UTF_8)
        return entries.size
    }

    @Synchronized
    fun clear() {
        file.delete()
    }

    private fun rewrite(entries: List<FeedbackEntry>) {
        file.parentFile?.mkdirs()
        val tmp = File(file.path + ".tmp")
        tmp.writeText(entries.joinToString("") { it.toJson().toString() + "\n" }, Charsets.UTF_8)
        if (!tmp.renameTo(file)) {
            file.delete()
            check(tmp.renameTo(file)) { "could not save feedback" }
        }
    }
}
