package com.podtekst.engine

/** Which quantization a phone gets, chosen from its total RAM (decimal bytes, as ActivityManager reports). */
enum class ModelTier(val approxGb: Double) {
    INT4(2.3),
    INT8(4.0);

    val id: String get() = name.lowercase()

    companion object {
        /** A "12 GB" phone reports about 11.2e9 bytes. */
        const val INT8_MIN_RAM = 10_500_000_000L
        /** An "8 GB" phone reports about 7.4e9 bytes. */
        const val INT4_MIN_RAM = 7_000_000_000L

        fun forTotalRam(totalBytes: Long): ModelTier? = when {
            totalBytes >= INT8_MIN_RAM -> INT8
            totalBytes >= INT4_MIN_RAM -> INT4
            else -> null
        }

        fun fromId(id: String): ModelTier? = entries.firstOrNull { it.id.equals(id, ignoreCase = true) }
    }
}
