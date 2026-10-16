package com.podtekst.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class PodtekstPromptTest {

    /** The model was trained on the prompt in model/podtekst_sft/prompt.py; any drift breaks it. */
    @Test
    fun systemPromptMatchesTrainingPrompt() {
        val py = File("../../model/podtekst_sft/prompt.py")
        assertTrue("prompt.py not found at ${py.absolutePath}", py.isFile)
        val block = py.readText().substringAfter("SYSTEM_PROMPT = (").substringBefore("\n)")
        val literal = Regex("\"([^\"]*)\"|'([^']*)'")
        val trained = literal.findAll(block).joinToString("") { it.groupValues[1] + it.groupValues[2] }
        assertTrue("could not read SYSTEM_PROMPT from prompt.py", trained.length > 100)
        assertEquals(trained, PodtekstPrompt.SYSTEM_PROMPT)
    }

    @Test
    fun genderHintsMatchUserContent() {
        assertEquals("Я устала.\n(speaker: female)", PodtekstPrompt.userContent("Я устала.", "female"))
        assertEquals("Hi\n(addressee: male)", PodtekstPrompt.userContent("Hi", null, "male"))
        assertEquals(
            "Hi\n(speaker: male, addressee: female)",
            PodtekstPrompt.userContent("Hi", "male", "female"),
        )
        assertEquals("Hi", PodtekstPrompt.userContent("Hi"))
        assertEquals("Hi", PodtekstPrompt.userContent("Hi", "unknown", ""))
    }
}
