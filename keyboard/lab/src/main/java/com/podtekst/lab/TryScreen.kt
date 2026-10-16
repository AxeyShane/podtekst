package com.podtekst.lab

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

private val GENDER_CYCLE = listOf<String?>(null, "female", "male")

@Composable
fun TryScreen(vm: LabViewModel) {
    var text by rememberSaveable { mutableStateOf("") }
    var speaker by rememberSaveable { mutableStateOf<String?>(null) }
    var addressee by rememberSaveable { mutableStateOf<String?>(null) }

    Column(
        Modifier.padding(16.dp).verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(vm.loadedName?.let { "Model: $it (${vm.loadedBackend})" } ?: "Load a model on the Models tab first.",
            style = MaterialTheme.typography.bodySmall)
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Message (Russian or English)") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { speaker = next(speaker) }) { Text("Speaker: ${speaker ?: "–"}") }
            OutlinedButton(onClick = { addressee = next(addressee) }) { Text("Addressee: ${addressee ?: "–"}") }
        }
        Button(
            onClick = { vm.translate(text, speaker, addressee) },
            enabled = vm.loadedName != null && !vm.busy && vm.running == null && text.isNotBlank(),
        ) { Text(if (vm.busy) "Working…" else "Translate") }

        vm.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }

        val r = vm.result
        if (r == null && vm.busy && vm.partial.isNotEmpty()) {
            Mono(vm.partial)
        }
        if (r != null) {
            val p = r.parsed
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (p == null) {
                        Text("No valid JSON in the reply", color = MaterialTheme.colorScheme.error)
                    } else {
                        SelectionContainer { Text(p.translation, style = MaterialTheme.typography.titleMedium) }
                        Text(if (p.hasSubtext) "Nuance: ${p.category}" else "No nuance lost (none)",
                            style = MaterialTheme.typography.labelLarge)
                        if (p.nuanceNote.isNotEmpty()) Text(p.nuanceNote, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
            val t = r.timings
            Text(
                "first token ${t.firstTokenMs} ms · total ${t.totalMs} ms · ${t.chunks} chunks" +
                    (t.decodePerSecond?.let { " · ≈%.1f tok/s".format(it) } ?: ""),
                style = MaterialTheme.typography.bodySmall,
            )
            Text("Raw reply", style = MaterialTheme.typography.labelMedium)
            Mono(r.raw)
        }
    }
}

@Composable
private fun Mono(s: String) {
    SelectionContainer {
        Text(s, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
    }
}

private fun next(current: String?): String? = GENDER_CYCLE[(GENDER_CYCLE.indexOf(current) + 1) % GENDER_CYCLE.size]
