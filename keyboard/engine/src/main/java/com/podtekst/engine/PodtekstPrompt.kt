package com.podtekst.engine

/**
 * The prompt the model was fine-tuned on. Kotlin copy of model/podtekst_sft/prompt.py:
 * the same wording must be used at runtime, and PodtekstPromptTest fails if the two drift apart.
 */
object PodtekstPrompt {

    val CATEGORIES = listOf("none", "formality_shift", "idiom", "sarcasm", "emotional_subtext")

    const val SYSTEM_PROMPT =
        "You translate chat messages between Russian and English for a phone keyboard. " +
            "Translate the message into the other language, then say whether a plain translation " +
            "loses nuance. Reply with JSON only: " +
            "{\"translation\": str, \"has_subtext\": bool, " +
            "\"category\": \"none|formality_shift|idiom|sarcasm|emotional_subtext\", \"nuance_note\": str}. " +
            "Use category \"none\" and an empty nuance_note when nothing is lost."

    /**
     * Chat template for LiteRT-LM. The base model's own template (RuadaptQwen3-4B-Instruct, ChatML
     * without a think block) builds strings with `+ message.content`, but LiteRT-LM always passes
     * content as a list of parts, so rendering fails. This template produces the same text from
     * the parts: `<|im_start|>role\n…<|im_end|>\n` per message, then `<|im_start|>assistant\n`.
     */
    const val CHAT_TEMPLATE =
        "{%- for message in messages -%}" +
            "{{- '<|im_start|>' + message.role + '\\n' -}}" +
            "{%- if message.content is string -%}{{- message.content -}}" +
            "{%- else -%}{%- for part in message.content -%}" +
            "{%- if part.type == 'text' -%}{{- part.text -}}{%- endif -%}" +
            "{%- endfor -%}{%- endif -%}" +
            "{{- '<|im_end|>\\n' -}}" +
            "{%- endfor -%}" +
            "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"

    private val GENDERS = setOf("female", "male")

    /** The user turn: the message, plus the optional gender context the keyboard can know. */
    fun userContent(text: String, speakerGender: String? = null, addresseeGender: String? = null): String {
        val hints = buildList {
            if (speakerGender in GENDERS) add("speaker: $speakerGender")
            if (addresseeGender in GENDERS) add("addressee: $addresseeGender")
        }
        return if (hints.isEmpty()) text else "$text\n(${hints.joinToString(", ")})"
    }
}
