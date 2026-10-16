package com.podtekst.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Mirrors parse_reply() in model/podtekst_sft/prompt.py. */
class ReplyParserTest {

    @Test
    fun plainJson() {
        val t = ReplyParser.parse(
            """{"translation": "Thanks a lot.", "has_subtext": true, "category": "sarcasm", "nuance_note": "Ironic."}"""
        )
        assertEquals(Translation("Thanks a lot.", true, "sarcasm", "Ironic."), t)
    }

    @Test
    fun stripsThinkBlockAndCodeFence() {
        val raw = "<think>\nhmm\n</think>\n```json\n" +
            """{"translation": "Hi", "has_subtext": false, "category": "none", "nuance_note": ""}""" +
            "\n```"
        assertEquals(Translation("Hi", false, "none", ""), ReplyParser.parse(raw))
    }

    @Test
    fun leadingChatterIsIgnored() {
        val raw = """Sure! {"translation": "Hi", "has_subtext": false, "category": "none", "nuance_note": ""}"""
        assertEquals("Hi", ReplyParser.parse(raw)?.translation)
    }

    @Test
    fun hasSubtextAsString() {
        val t = ReplyParser.parse("""{"translation": "x", "has_subtext": "True", "category": "idiom", "nuance_note": "n"}""")
        assertEquals(Translation("x", true, "idiom", "n"), t)
    }

    @Test
    fun unknownCategoryBecomesNone() {
        val t = ReplyParser.parse("""{"translation": "x", "has_subtext": true, "category": "irony", "nuance_note": "n"}""")
        assertEquals(Translation("x", false, "none", "n"), t)
    }

    @Test
    fun categoryNoneForcesNoSubtext() {
        val t = ReplyParser.parse("""{"translation": "x", "has_subtext": true, "category": "none", "nuance_note": ""}""")
        assertEquals(false, t?.hasSubtext)
    }

    @Test
    fun noSubtextHidesCategory() {
        val t = ReplyParser.parse("""{"translation": "x", "has_subtext": false, "category": "sarcasm", "nuance_note": ""}""")
        assertEquals("none", t?.category)
    }

    @Test
    fun nullNoteBecomesEmpty() {
        val t = ReplyParser.parse("""{"translation": "x", "has_subtext": false, "category": "none", "nuance_note": null}""")
        assertEquals("", t?.nuanceNote)
    }

    @Test
    fun unusableRepliesGiveNull() {
        assertNull(ReplyParser.parse(null))
        assertNull(ReplyParser.parse(""))
        assertNull(ReplyParser.parse("I can't help with that."))
        assertNull(ReplyParser.parse("""{"translation": "x", "has_subtext": """))
        assertNull(ReplyParser.parse("""{"has_subtext": false, "category": "none"}"""))
        assertNull(ReplyParser.parse("""{"translation": 5, "has_subtext": false}"""))
    }
}
