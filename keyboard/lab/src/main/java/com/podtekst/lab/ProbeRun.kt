package com.podtekst.lab

import com.podtekst.engine.Translation
import com.podtekst.engine.TranslationResult
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** One input row of a probe file: dataset rows, or sft_test.jsonl items that wrap them in "row". */
data class ProbeRow(val json: JSONObject) {
    val sourceText: String get() = json.optString("source_text")
    val speakerGender: String? get() = text("speaker_gender") ?: text("_speaker_gender")
    val addresseeGender: String? get() = text("addressee_gender") ?: text("_addressee_gender")

    private fun text(key: String) = json.opt(key) as? String
}

object ProbeFiles {

    fun list(dir: File): List<File> =
        (dir.listFiles() ?: emptyArray()).filter { it.isFile && it.name.endsWith(".jsonl") }.sortedBy { it.name }

    fun read(file: File): List<ProbeRow> =
        file.readLines(Charsets.UTF_8).filter { it.isNotBlank() }.map { line ->
            val obj = JSONObject(line)
            ProbeRow(obj.optJSONObject("row") ?: obj)
        }.filter { it.sourceText.isNotBlank() }

    /** results/<model>__<probe>__<yyyyMMdd-HHmm>.jsonl — score_device.py reads the probe part of the name. */
    fun resultsFile(resultsDir: File, model: String, probe: File): File {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date())
        resultsDir.mkdirs()
        return File(resultsDir, "${model.removeSuffix(".litertlm")}__${probe.nameWithoutExtension}__$stamp.jsonl")
    }

    fun resultLine(row: ProbeRow, result: TranslationResult, model: String, backend: String): String =
        JSONObject()
            .put("row", row.json)
            .put("reply", result.raw)
            .put("parsed", result.parsed?.toJson() ?: JSONObject.NULL)
            .put("ms", JSONObject().put("first_token", result.timings.firstTokenMs).put("total", result.timings.totalMs))
            .put("chunks", result.timings.chunks)
            .put("model", model)
            .put("backend", backend)
            .toString()

    private fun Translation.toJson() = JSONObject()
        .put("translation", translation)
        .put("has_subtext", hasSubtext)
        .put("category", category)
        .put("nuance_note", nuanceNote)
}

/** Running totals shown on the Runs screen while and after a run. */
class RunStats {
    var done = 0
        private set
    var jsonValid = 0
        private set
    var totalMsSum = 0L
        private set
    var maxMs = 0L
        private set
    val categories = sortedMapOf<String, Int>()

    fun add(result: TranslationResult) {
        done++
        if (result.parsed != null) jsonValid++
        totalMsSum += result.timings.totalMs
        maxMs = maxOf(maxMs, result.timings.totalMs)
        val cat = result.parsed?.category ?: "invalid"
        categories[cat] = (categories[cat] ?: 0) + 1
    }

    fun summary(): String {
        if (done == 0) return "No rows run."
        val valid = 100.0 * jsonValid / done
        val mean = totalMsSum / done
        return "Rows $done · valid JSON %.1f%% · mean %d ms · max %d ms\n".format(valid, mean, maxMs) +
            categories.entries.joinToString("  ") { "${it.key} ${it.value}" }
    }
}
