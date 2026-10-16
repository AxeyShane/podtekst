package com.podtekst.lab

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunsScreen(vm: LabViewModel) {
    val running = vm.running != null
    val view = LocalView.current
    DisposableEffect(running) {
        view.keepScreenOn = running
        onDispose { view.keepScreenOn = false }
    }

    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(vm.loadedName?.let { "Model: $it (${vm.loadedBackend})" } ?: "Load a model on the Models tab first.",
            style = MaterialTheme.typography.bodySmall)

        if (running) {
            val (done, total) = vm.runProgress
            Text("Running ${vm.running}: $done / $total")
            LinearProgressIndicator(
                progress = { if (total > 0) done.toFloat() / total else 0f },
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedButton(onClick = vm::cancelRun) { Text("Cancel") }
        }
        vm.runSummary?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
        vm.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }

        if (vm.probeFiles.isEmpty()) {
            Text(
                "No probe files yet. From the PC:\nadb push crude_probe.jsonl /data/local/tmp/\nadb shell run-as com.podtekst.lab cp /data/local/tmp/crude_probe.jsonl ${vm.probesDir.absolutePath}/",
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
            )
        }
        Text("Results: ${vm.resultsDir.absolutePath}", style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace)
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(vm.probeFiles, key = { it.name }) { f ->
                Card(
                    onClick = { vm.startRun(f) },
                    enabled = vm.loadedName != null && !running && !vm.busy,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text(f.name, style = MaterialTheme.typography.titleSmall)
                        Text("%.1f KB · tap to run".format(f.length() / 1024.0), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
