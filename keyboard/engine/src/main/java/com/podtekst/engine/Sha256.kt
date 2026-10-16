package com.podtekst.engine

import java.io.File
import java.security.MessageDigest

object Sha256 {
    fun of(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** Size first (cheap), then the hash. */
    fun matches(file: File, model: ManifestModel): Boolean =
        file.isFile && file.length() == model.bytes && of(file) == model.sha256
}
