package com.podtekst.engine

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import java.io.File

enum class ComputeBackend { CPU, GPU }

/** Wall-clock timings of one answer. [chunks] is the number of streamed pieces (about one per token). */
data class Timings(val firstTokenMs: Long, val totalMs: Long, val chunks: Int) {
    /** Approximate decode speed after the first token, in tokens per second. */
    val decodePerSecond: Double?
        get() = (totalMs - firstTokenMs).takeIf { it > 0 && chunks > 1 }?.let { (chunks - 1) * 1000.0 / it }
}

data class TranslationResult(val parsed: Translation?, val raw: String, val timings: Timings)

/**
 * One loaded .litertlm model, used with the exact settings of the Kaggle evaluation: greedy
 * decoding, thinking off, at most [MAX_OUTPUT_TOKENS] tokens, and a fresh conversation per message
 * (the training framing: system prompt + one user message). Not thread-safe; callers serialise.
 */
class TranslatorEngine(private val cacheDir: String) : AutoCloseable {

    private var engine: Engine? = null

    var loaded: File? = null
        private set
    var backend: ComputeBackend? = null
        private set

    /** Loads [model], replacing any loaded one. Blocking (seconds): call off the main thread. Returns ms taken. */
    fun load(model: File, backend: ComputeBackend): Long {
        close()
        val start = System.nanoTime()
        val config = EngineConfig(
            modelPath = model.absolutePath,
            backend = when (backend) {
                ComputeBackend.GPU -> Backend.GPU()
                ComputeBackend.CPU -> Backend.CPU()
            },
            cacheDir = cacheDir,
        )
        val e = Engine(config)
        try {
            e.initialize()
        } catch (t: Throwable) {
            runCatching { e.close() }
            throw t
        }
        engine = e
        loaded = model
        this.backend = backend
        return elapsedMs(start)
    }

    /** Translates one message. [onPartial] receives the text generated so far while streaming. */
    suspend fun translate(
        text: String,
        speakerGender: String? = null,
        addresseeGender: String? = null,
        onPartial: (String) -> Unit = {},
    ): TranslationResult {
        val e = checkNotNull(engine) { "No model loaded" }
        val config = ConversationConfig(
            systemInstruction = Contents.of(PodtekstPrompt.SYSTEM_PROMPT),
            samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
            maxOutputToken = MAX_OUTPUT_TOKENS,
            thinkingConfig = ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0),
            chatTemplate = PodtekstPrompt.CHAT_TEMPLATE,
        )
        val start = System.nanoTime()
        var firstMs = -1L
        var chunks = 0
        val out = StringBuilder()
        e.createConversation(config).use { conversation ->
            conversation.sendMessageAsync(PodtekstPrompt.userContent(text, speakerGender, addresseeGender))
                .collect { message ->
                    if (firstMs < 0) firstMs = elapsedMs(start)
                    chunks++
                    out.append(message.toString())
                    onPartial(out.toString())
                }
        }
        val raw = out.toString()
        val total = elapsedMs(start)
        return TranslationResult(ReplyParser.parse(raw), raw, Timings(if (firstMs < 0) total else firstMs, total, chunks))
    }

    override fun close() {
        engine?.let { runCatching { it.close() } }
        engine = null
        loaded = null
        backend = null
    }

    private fun elapsedMs(startNanos: Long) = (System.nanoTime() - startNanos) / 1_000_000

    companion object {
        const val MAX_OUTPUT_TOKENS = 256
    }
}
