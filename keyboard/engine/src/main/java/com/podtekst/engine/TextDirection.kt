package com.podtekst.engine

object TextDirection {
    /** Longest message the Translate button accepts (the model was trained on chat-length messages). */
    const val MAX_CHARS = 500

    /** "ru" when Cyrillic letters are at least as many as Latin ones (and there is at least one), else "en". */
    fun sourceLang(text: String): String {
        var cyrillic = 0
        var latin = 0
        for (c in text) {
            when {
                c in '\u0400'..'\u04FF' -> cyrillic++
                c in 'a'..'z' || c in 'A'..'Z' -> latin++
            }
        }
        return if (cyrillic > 0 && cyrillic >= latin) "ru" else "en"
    }
}
