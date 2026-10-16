package com.podtekst.lab

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.podtekst.engine.ComputeBackend

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(vm: LabViewModel) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::import)
    }
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ComputeBackend.entries.forEach { b ->
                FilterChip(selected = vm.backend == b, onClick = { vm.backend = b }, label = { Text(b.name) })
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Import") }
            OutlinedButton(onClick = vm::refresh) { Text("Refresh") }
            OutlinedButton(onClick = vm::unload, enabled = vm.loadedName != null && !vm.busy) { Text("Unload") }
        }
        Text(vm.status, style = MaterialTheme.typography.bodyMedium)
        vm.importProgress?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        vm.error?.let { err ->
            Text(err, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            if (vm.offerCpuRetry) {
                val failed = vm.models.firstOrNull { it.name == vm.lastTried }
                if (failed != null) {
                    Button(onClick = { vm.load(failed.file, ComputeBackend.CPU) }) { Text("Retry on CPU") }
                }
            }
        }
        if (vm.models.isEmpty()) {
            Text(
                "No models yet. From the PC:\nadb push model.litertlm /data/local/tmp/\nadb shell run-as com.podtekst.lab cp /data/local/tmp/model.litertlm ${vm.store.dir.absolutePath}/\nor use Import.",
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
            )
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(vm.models, key = { it.name }) { m ->
                val isLoaded = m.name == vm.loadedName
                Card(
                    onClick = { if (!vm.busy) vm.load(m.file) },
                    modifier = Modifier.fillMaxWidth(),
                    colors = if (isLoaded) {
                        CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                    } else {
                        CardDefaults.cardColors()
                    },
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text(m.name, style = MaterialTheme.typography.titleSmall)
                        Text(
                            "%.2f GB".format(m.sizeBytes / 1e9) + if (isLoaded) " · loaded on ${vm.loadedBackend}" else "",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
        }
    }
}
