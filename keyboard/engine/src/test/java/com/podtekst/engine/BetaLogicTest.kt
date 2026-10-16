package com.podtekst.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class BetaLogicTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test fun tierByRam() {
        assertEquals(ModelTier.INT8, ModelTier.forTotalRam(11_200_000_000)) // a "12 GB" phone
        assertEquals(ModelTier.INT4, ModelTier.forTotalRam(7_400_000_000))  // an "8 GB" phone
        assertNull(ModelTier.forTotalRam(5_600_000_000))                    // a "6 GB" phone
        assertEquals(ModelTier.INT8, ModelTier.forTotalRam(ModelTier.INT8_MIN_RAM))
        assertEquals(ModelTier.INT4, ModelTier.fromId("INT4"))
        assertNull(ModelTier.fromId("fp16"))
    }

    private val manifestJson = """
        {"models": [
          {"tier": "int4", "file": "podtekst-rua-q3-int4.litertlm", "version": "rua-q3", "bytes": 3, "sha256": "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"},
          {"tier": "int8", "file": "podtekst-rua-q3-int8.litertlm", "version": "rua-q3", "bytes": 4, "sha256": "00"},
          {"tier": "fp16", "file": "x.litertlm", "version": "v", "bytes": 1, "sha256": "00"}
        ]}
    """.trimIndent()

    @Test fun manifestParsesKnownTiersAndLowercasesHash() {
        val models = ModelManifest.parse(manifestJson)
        assertEquals(2, models.size)
        val int4 = ModelManifest.pick(models, ModelTier.INT4)!!
        assertEquals("podtekst-rua-q3-int4.litertlm", int4.file)
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", int4.sha256)
        assertEquals(int4, ManifestModel.fromJson(int4.toJson()))
    }

    @Test fun manifestRejectsPathsAndOtherExtensions() {
        val bad = """{"models": [{"tier": "int4", "file": "../evil.litertlm", "version": "v", "bytes": 1, "sha256": "00"}]}"""
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(bad) }
        val apk = """{"models": [{"tier": "int4", "file": "x.apk", "version": "v", "bytes": 1, "sha256": "00"}]}"""
        assertThrows(IllegalArgumentException::class.java) { ModelManifest.parse(apk) }
    }

    @Test fun sha256OfFileAndMatch() {
        val f = tmp.newFile("podtekst-rua-q3-int4.litertlm").apply { writeText("abc") }
        assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", Sha256.of(f))
        val model = ModelManifest.pick(ModelManifest.parse(manifestJson), ModelTier.INT4)!!
        assertTrue(Sha256.matches(f, model))
        f.writeText("abd")
        assertFalse(Sha256.matches(f, model))
        f.writeText("abcd")
        assertFalse(Sha256.matches(f, model)) // wrong size
    }

    @Test fun memoryAndStoragePolicy() {
        val model = 4_044_303_040L
        assertTrue(MemoryPolicy.canLoad(model + MemoryPolicy.LOAD_MARGIN, model))
        assertFalse(MemoryPolicy.canLoad(model, model))
        assertTrue(MemoryPolicy.canDownload(model + MemoryPolicy.STORAGE_MARGIN, model))
        assertFalse(MemoryPolicy.canDownload(model + MemoryPolicy.STORAGE_MARGIN - 1, model))
    }

    @Test fun direction() {
        assertEquals("ru", TextDirection.sourceLang("Ну ты, конечно, молодец"))
        assertEquals("en", TextDirection.sourceLang("Well, you sure did great"))
        assertEquals("ru", TextDirection.sourceLang("Я в офисе, ok"))
        assertEquals("en", TextDirection.sourceLang("Ок, see you tomorrow"))
        assertEquals("en", TextDirection.sourceLang("123 :)"))
    }
}
