package com.podtekst.lab

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.Debug
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.podtekst.engine.ComputeBackend
import com.podtekst.engine.ModelFile
import com.podtekst.engine.ModelStore
import com.podtekst.engine.TranslationResult
import com.podtekst.engine.TranslatorEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

class LabViewModel(app: Application) : AndroidViewModel(app) {

    /**
     * App-private storage. Files that adb puts into /sdcard/Android/data/<app> belong to the shell
     * user and the app cannot read them, so models and probes go in via /data/local/tmp + run-as.
     */
    private val filesDir: File = app.filesDir
    val store = ModelStore(File(filesDir, "models"))
    val probesDir = File(filesDir, "probes")
    val resultsDir = File(filesDir, "results")

    private val engine = TranslatorEngine(app.cacheDir.path)
    /** Only one load, translation or run touches the engine at a time. */
    private val engineLock = Mutex()
    private val prefs = app.getSharedPreferences("lab", Context.MODE_PRIVATE)

    // Models screen
    var models by mutableStateOf<List<ModelFile>>(emptyList())
        private set
    var backend by mutableStateOf(savedBackend())
    var loadedName by mutableStateOf<String?>(null)
        private set
    var loadedBackend by mutableStateOf<ComputeBackend?>(null)
        private set
    /** File name of the last load attempt, for "Retry on CPU". */
    var lastTried by mutableStateOf<String?>(null)
        private set
    var status by mutableStateOf("No model loaded")
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var offerCpuRetry by mutableStateOf(false)
        private set
    var busy by mutableStateOf(false)
        private set
    var importProgress by mutableStateOf<String?>(null)
        private set

    // Try screen
    var partial by mutableStateOf("")
        private set
    var result by mutableStateOf<TranslationResult?>(null)
        private set

    // Runs screen
    var probeFiles by mutableStateOf<List<File>>(emptyList())
        private set
    var running by mutableStateOf<String?>(null)
        private set
    var runProgress by mutableStateOf(0 to 0)
        private set
    var runSummary by mutableStateOf<String?>(null)
        private set
    private var runJob: Job? = null

    init {
        store.dir.mkdirs()
        probesDir.mkdirs()
        resultsDir.mkdirs()
        refresh()
        prefs.getString(KEY_MODEL, null)
            ?.let { File(store.dir, it) }
            ?.takeIf { it.isFile }
            ?.let { load(it, backend) }
    }

    fun refresh() {
        models = store.list()
        probeFiles = ProbeFiles.list(probesDir)
    }

    fun load(file: File, backend: ComputeBackend = this.backend) {
        viewModelScope.launch {
            busy = true
            error = null
            offerCpuRetry = false
            lastTried = file.name
            status = "Loading ${file.name} on $backend…"
            try {
                val ms = withContext(Dispatchers.IO) { engineLock.withLock { engine.load(file, backend) } }
                loadedName = file.name
                loadedBackend = backend
                this@LabViewModel.backend = backend
                prefs.edit().putString(KEY_MODEL, file.name).putString(KEY_BACKEND, backend.name).apply()
                status = "Loaded ${file.name} on $backend in %.1f s · app memory %d MB".format(ms / 1000.0, memoryMb())
            } catch (t: Throwable) {
                loadedName = null
                loadedBackend = null
                error = t.toString()
                offerCpuRetry = backend == ComputeBackend.GPU
                status = "Load failed"
            } finally {
                busy = false
            }
        }
    }

    fun unload() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { engineLock.withLock { engine.close() } }
            loadedName = null
            loadedBackend = null
            status = "No model loaded"
        }
    }

    fun import(uri: Uri) {
        val resolver = getApplication<Application>().contentResolver
        var name = "imported.litertlm"
        var size = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                name = c.getString(0) ?: name
                if (!c.isNull(1)) size = c.getLong(1)
            }
        }
        viewModelScope.launch {
            error = null
            importProgress = "Copying $name…"
            try {
                withContext(Dispatchers.IO) {
                    store.import(name, size, { checkNotNull(resolver.openInputStream(uri)) { "Cannot open $name" } }) { copied ->
                        val pct = if (size > 0) " (${100 * copied / size}%)" else ""
                        importProgress = "Copying $name: ${copied / MB} MB$pct"
                    }
                }
                importProgress = "Imported $name"
            } catch (t: Throwable) {
                importProgress = null
                error = "Import failed: $t"
            }
            refresh()
        }
    }

    fun translate(text: String, speakerGender: String?, addresseeGender: String?) {
        if (text.isBlank() || loadedName == null) return
        viewModelScope.launch {
            busy = true
            error = null
            partial = ""
            result = null
            try {
                result = withContext(Dispatchers.Default) {
                    engineLock.withLock {
                        engine.translate(text.trim(), speakerGender, addresseeGender) { partial = it }
                    }
                }
            } catch (t: Throwable) {
                error = t.toString()
            } finally {
                busy = false
            }
        }
    }

    fun startRun(probe: File) {
        val model = loadedName ?: return
        if (runJob?.isActive == true) return
        runJob = viewModelScope.launch {
            running = probe.name
            runSummary = null
            error = null
            val stats = RunStats()
            try {
                val rows = withContext(Dispatchers.IO) { ProbeFiles.read(probe) }
                runProgress = 0 to rows.size
                val out = ProbeFiles.resultsFile(resultsDir, model, probe)
                val backendName = backend.name
                withContext(Dispatchers.Default) {
                    out.bufferedWriter(Charsets.UTF_8).use { writer ->
                        for (row in rows) {
                            ensureActive()
                            val r = engineLock.withLock {
                                engine.translate(row.sourceText, row.speakerGender, row.addresseeGender)
                            }
                            writer.write(ProbeFiles.resultLine(row, r, model, backendName))
                            writer.write("\n")
                            writer.flush()
                            stats.add(r)
                            runProgress = stats.done to rows.size
                        }
                    }
                }
                runSummary = stats.summary() + "\nSaved ${out.name}"
            } catch (t: Throwable) {
                runSummary = if (t is kotlinx.coroutines.CancellationException) {
                    "Cancelled after ${stats.done} rows.\n" + stats.summary()
                } else {
                    error = t.toString()
                    "Stopped after ${stats.done} rows."
                }
            } finally {
                running = null
            }
        }
    }

    fun cancelRun() {
        runJob?.cancel()
    }

    override fun onCleared() {
        engine.close()
    }

    private fun savedBackend(): ComputeBackend =
        runCatching { ComputeBackend.valueOf(prefs.getString(KEY_BACKEND, null) ?: "GPU") }.getOrDefault(ComputeBackend.GPU)

    private fun memoryMb(): Long {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024L
    }

    companion object {
        private const val KEY_MODEL = "model"
        private const val KEY_BACKEND = "backend"
        private const val MB = 1024L * 1024
    }
}
