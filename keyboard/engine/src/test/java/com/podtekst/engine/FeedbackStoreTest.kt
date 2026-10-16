package com.podtekst.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.json.JSONObject

class FeedbackStoreTest {
    @get:Rule val tmp = TemporaryFolder()

    private fun entry(id: Long, rating: Int = 1) = FeedbackEntry(
        id = id, time = 1_760_000_000_000 + id, sourceLang = "ru", sourceText = "Ну ты, конечно, молодец",
        translation = "Well, you sure did great", category = "sarcasm", note = "Ironic praise.",
        rating = rating, comment = "", model = "podtekst-rua-q3-int8.litertlm", totalMs = 4600,
    )

    @Test fun addListUpdateDelete() {
        val store = FeedbackStore(tmp.root.resolve("sub/feedback.jsonl"))
        assertTrue(store.list().isEmpty())
        store.add(entry(1))
        store.add(entry(2, rating = -1))
        assertEquals(listOf(1L, 2L), store.list().map { it.id })
        store.update(2) { it.copy(rating = 1, comment = "точно сарказм") }
        assertEquals("точно сарказм", store.list()[1].comment)
        assertEquals(1, store.list()[1].rating)
        store.delete(1)
        assertEquals(listOf(2L), store.list().map { it.id })
    }

    @Test fun exportWritesJsonLinesAndClearEmpties() {
        val store = FeedbackStore(tmp.root.resolve("feedback.jsonl"))
        store.add(entry(1))
        store.add(entry(2))
        val out = tmp.root.resolve("export.jsonl")
        assertEquals(2, store.exportTo(out))
        val lines = out.readLines(Charsets.UTF_8).filter { it.isNotBlank() }
        assertEquals(2, lines.size)
        assertEquals("Ну ты, конечно, молодец", JSONObject(lines[0]).getString("source_text"))
        store.clear()
        assertTrue(store.list().isEmpty())
    }

    @Test fun brokenLinesAreSkipped() {
        val file = tmp.root.resolve("feedback.jsonl")
        val store = FeedbackStore(file)
        store.add(entry(1))
        file.appendText("{not json\n")
        assertEquals(1, store.list().size)
    }
}
