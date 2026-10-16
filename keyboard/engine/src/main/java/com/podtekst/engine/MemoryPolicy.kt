package com.podtekst.engine

object MemoryPolicy {
    /** Free RAM needed beyond the model's size before loading it. */
    const val LOAD_MARGIN = 256L * 1024 * 1024
    /** Free storage needed beyond the model's size before downloading it. */
    const val STORAGE_MARGIN = 1_000_000_000L

    fun canLoad(availBytes: Long, modelBytes: Long): Boolean = availBytes >= modelBytes + LOAD_MARGIN
    fun canDownload(freeBytes: Long, modelBytes: Long): Boolean = freeBytes >= modelBytes + STORAGE_MARGIN
}
