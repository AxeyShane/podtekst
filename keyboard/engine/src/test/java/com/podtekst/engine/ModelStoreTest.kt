package com.podtekst.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream

class ModelStoreTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun listsOnlyModelFilesSortedByName() {
        val dir = tmp.newFolder("models")
        File(dir, "b.litertlm").writeBytes(ByteArray(3))
        File(dir, "A.LITERTLM").writeBytes(ByteArray(5))
        File(dir, "c.litertlm.part").writeBytes(ByteArray(1))
        File(dir, "notes.txt").writeText("x")
        File(dir, "folder.litertlm").mkdir()

        val models = ModelStore(dir).list()
        assertEquals(listOf("A.LITERTLM", "b.litertlm"), models.map { it.name })
        assertEquals(5L, models[0].sizeBytes)
    }

    @Test
    fun missingFolderListsNothing() {
        assertTrue(ModelStore(File(tmp.root, "absent")).list().isEmpty())
    }

    @Test
    fun importCopiesAndReportsProgress() {
        val dir = File(tmp.root, "models")
        val data = ByteArray(3_000_000) { (it % 251).toByte() }
        val seen = mutableListOf<Long>()

        val model = ModelStore(dir).import("m.litertlm", data.size.toLong(), { ByteArrayInputStream(data) }) { seen += it }

        assertTrue(model.file.readBytes().contentEquals(data))
        assertEquals(data.size.toLong(), seen.last())
        assertFalse(File(dir, "m.litertlm.part").exists())
    }

    @Test
    fun failedImportLeavesNothingBehind() {
        val dir = File(tmp.root, "models")
        val broken = object : InputStream() {
            private var n = 0
            override fun read(): Int = if (n++ < 10) 1 else throw IOException("cable pulled")
        }
        try {
            ModelStore(dir).import("m.litertlm", -1, { broken })
            fail("expected IOException")
        } catch (e: IOException) {
            // expected
        }
        assertEquals(emptyList<String>(), dir.list()?.toList().orEmpty())
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsOtherFileTypes() {
        ModelStore(tmp.newFolder("models")).import("model.gguf", 1, { ByteArrayInputStream(ByteArray(1)) })
    }
}
