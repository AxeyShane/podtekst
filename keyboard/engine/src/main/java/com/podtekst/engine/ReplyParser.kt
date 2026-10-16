package com.podtekst.engine

import org.json.JSONException
import org.json.JSONObject

/** One parsed model answer. */
data class Translation(
    val translation: String,
    val hasSubtext: Boolean,
    val category: String,
    val nuanceNote: String,
)

/** Kotlin port of parse_reply() in model/podtekst_sft/prompt.py — keep the two in step. */
object ReplyParser {

    private val THINK = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)
    private val JSON_OBJECT = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)

    /** Model reply -> normalised answer, or null when there is no usable JSON. */
    fun parse(text: String?): Translation? {
        val cleaned = THINK.replace(text ?: "", "").trim()
            .removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        val match = JSON_OBJECT.find(cleaned) ?: return null
        val obj = try {
            JSONObject(match.value)
        } catch (e: JSONException) {
            return null
        }
        val translation = obj.opt("translation") as? String ?: return null
        val rawCategory = obj.opt("category")
        val category = if (rawCategory is String && rawCategory in PodtekstPrompt.CATEGORIES) rawCategory else "none"
        val hasSubtext = truthy(obj.opt("has_subtext"), stringsByValue = true) && category != "none"
        val note = obj.opt("nuance_note")
        return Translation(
            translation = translation,
            hasSubtext = hasSubtext,
            category = if (hasSubtext) category else "none",
            nuanceNote = if (truthy(note, stringsByValue = false)) note?.toString().orEmpty() else "",
        )
    }

    /** Python truthiness; for has_subtext a string counts only when it reads "true". */
    private fun truthy(value: Any?, stringsByValue: Boolean): Boolean = when (value) {
        null, JSONObject.NULL -> false
        is Boolean -> value
        is Number -> value.toDouble() != 0.0
        is String -> if (stringsByValue) value.trim().lowercase() == "true" else value.isNotEmpty()
        else -> true
    }
}
