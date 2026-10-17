package com.podtekst.engine

/** Which quantization a phone gets, chosen from its total RAM (decimal bytes, as ActivityManager reports). */
enum class ModelTier(val approxGb: Double) {
    /** The 1.7B model at 4-bit, for 6 GB phones that can't hold the 4B model next to the apps. */
    SMALL(1.0),
    INT4(2.3),
    INT8(4.0);

    val id: String get() = name.lowercase()

    companion object {
        /** A "12 GB" phone reports about 11.2e9 bytes. */
        const val INT8_MIN_RAM = 10_500_000_000L
        /** An "8 GB" phone reports about 7.4e9 bytes. */
        const val INT4_MIN_RAM = 7_000_000_000L
        /** A "6 GB" phone reports about 5.6e9 bytes. */
        const val SMALL_MIN_RAM = 5_000_000_000L

        fun forTotalRam(totalBytes: Long): ModelTier? = when {
            totalBytes >= INT8_MIN_RAM -> INT8
            totalBytes >= INT4_MIN_RAM -> INT4
            totalBytes >= SMALL_MIN_RAM -> SMALL
            else -> null
        }

        fun fromId(id: String): ModelTier? = entries.firstOrNull { it.id.equals(id, ignoreCase = true) }
    }
}
