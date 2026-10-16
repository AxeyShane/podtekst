package com.podtekst.engine

import java.io.File
import java.io.IOException
import java.io.InputStream

/** A .litertlm file in the models folder. */
data class ModelFile(val file: File) {
    val name: String get() = file.name
    val sizeBytes: Long get() = file.length()
}

/**
 * The folder the app loads models from. Files arrive by `adb push` straight into [dir], or
 * through [import] (the Android file picker), which copies via a `.part` file so a failed
 * copy never shows up as a model.
 */
class ModelStore(val dir: File) {

    fun list(): List<ModelFile> =
        (dir.listFiles() ?: emptyArray())
            .filter { it.isFile && it.name.endsWith(EXT, ignoreCase = true) }
            .sortedBy { it.name.lowercase() }
            .map { ModelFile(it) }

    /** Copies a model into [dir]. [sizeBytes] is -1 when unknown; [onProgress] gets bytes copied. */
    fun import(
        name: String,
        sizeBytes: Long,
        open: () -> InputStream,
        onProgress: (Long) -> Unit = {},
    ): ModelFile {
        val fileName = File(name).name
        require(fileName.endsWith(EXT, ignoreCase = true)) { "Not a $EXT file: $name" }
        dir.mkdirs()
        if (sizeBytes > 0 && dir.usableSpace < sizeBytes + SPARE_BYTES) {
            throw IOException("Not enough free space: need ${sizeBytes / MB} MB, have ${dir.usableSpace / MB} MB")
        }
        val target = File(dir, fileName)
        val part = File(dir, "$fileName.part")
        try {
            open().use { input ->
                part.outputStream().use { output ->
                    val buffer = ByteArray(1 shl 20)
                    var total = 0L
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        total += n
                        onProgress(total)
                    }
                }
            }
            if (target.exists() && !target.delete()) throw IOException("Could not replace $fileName")
            if (!part.renameTo(target)) throw IOException("Could not rename ${part.name}")
        } catch (e: Exception) {
            part.delete()
            throw e
        }
        return ModelFile(target)
    }

    companion object {
        const val EXT = ".litertlm"
        private const val MB = 1024L * 1024
        private const val SPARE_BYTES = 200 * MB
    }
}
