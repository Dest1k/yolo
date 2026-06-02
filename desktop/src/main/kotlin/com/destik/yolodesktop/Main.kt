package com.destik.yolodesktop

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import java.awt.FileDialog
import java.awt.Frame
import java.nio.file.Paths

private val Accent    = Color(0xFF00E676)
private val BgDark    = Color(0xFF1A1A1A)
private val BgSide    = Color(0xFF2A2A2A)
private val BgCard    = Color(0xFF333333)
private val TextSub   = Color(0xFFAAAAAA)

fun main(args: Array<String>) {
    if (args.contains("--headless")) { HeadlessRunner.run(args); return }
    application {
        val state       = remember { AppState() }
        val windowState = rememberWindowState(width = 1300.dp, height = 820.dp)
        Window(onCloseRequest = {
            state.stop()
            if (state.mjpegActive) state.mjpegServer.stop()
            state.closeDetectors()
            exitApplication()
        }, title = "YOLO Detector", state = windowState) {
            MaterialTheme(colorScheme = darkColorScheme()) {
                AppScreen(state)
            }
        }
    }
}

@Composable
fun AppScreen(state: AppState) {
    Row(Modifier.fillMaxSize().background(BgDark)) {

        // ── Video panel ──────────────────────────────────────────────────────
        Box(
            Modifier.weight(1f).fillMaxHeight().background(Color.Black),
            contentAlignment = Alignment.Center
        ) {
            val frame = state.currentFrame
            if (frame != null) {
                Image(
                    bitmap = frame.toComposeImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("No video", color = Color.Gray, fontSize = 18.sp)
                    if (state.statusMessage != "Idle")
                        Text(state.statusMessage, color = TextSub, fontSize = 13.sp)
                }
            }
        }

        // ── Settings sidebar ─────────────────────────────────────────────────
        Column(
            Modifier
                .width(310.dp)
                .fillMaxHeight()
                .background(BgSide)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("YOLO Detector", color = Accent, fontSize = 20.sp,
                style = MaterialTheme.typography.titleLarge)

            // ── Model type ──────────────────────────────────────────────────
            SectionCard {
                Label("Model type")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ModelType.entries.forEach { t ->
                        FilterChip(
                            selected = state.modelType == t,
                            onClick  = { state.modelType = t },
                            label    = { Text(t.name) },
                            colors   = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = Accent,
                                selectedLabelColor     = Color.Black
                            )
                        )
                    }
                }

                Spacer(Modifier.height(4.dp))
                Label("Model file")
                Row(verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = if (state.modelPath.isEmpty()) "Not selected"
                               else Paths.get(state.modelPath).fileName.toString(),
                        color = Color.White, fontSize = 12.sp, modifier = Modifier.weight(1f),
                        maxLines = 1
                    )
                    val ext = if (state.modelType == ModelType.ONNX) "onnx" else "pt"
                    OutlinedButton(onClick = {
                        val path = pickFile("Select $ext model", ext)
                        if (path != null) state.modelPath = path
                    }) { Text("Browse") }
                }
            }

            // ── Video source ────────────────────────────────────────────────
            SectionCard {
                Label("Video source")
                OutlinedTextField(
                    value = state.sourcePath,
                    onValueChange = { state.sourcePath = it },
                    label = { Text("Webcam index or http://...") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = tfColors()
                )
            }

            // ── GPU / inference ─────────────────────────────────────────────
            SectionCard {
                Label("GPU / Execution provider")
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    GpuPreference.entries.forEach { g ->
                        FilterChip(
                            selected = state.gpuPref == g,
                            onClick  = { state.gpuPref = g },
                            label    = { Text(g.label(), fontSize = 11.sp) },
                            colors   = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = if (g == GpuPreference.CPU) Color(0xFF555555) else Accent,
                                selectedLabelColor     = Color.Black
                            )
                        )
                    }
                }
                if (state.activeProvider.isNotEmpty())
                    Text("Active: ${state.activeProvider}", color = Accent, fontSize = 11.sp)
            }

            // ── Model params ────────────────────────────────────────────────
            SectionCard {
                Label("Input size: ${state.inputSize}")
                Slider(value = state.inputSize.toFloat(),
                    onValueChange = { state.inputSize = it.toInt() },
                    valueRange = 320f..1280f, steps = 11,
                    colors = sliderColors())

                Label("Confidence: ${"%.2f".format(state.confThreshold)}")
                Slider(value = state.confThreshold,
                    onValueChange = { state.confThreshold = it },
                    valueRange = 0.05f..0.95f,
                    colors = sliderColors())

                Label("Classes: ${state.numClasses}")
                Slider(value = state.numClasses.toFloat(),
                    onValueChange = { state.numClasses = it.toInt() },
                    valueRange = 1f..200f, steps = 198,
                    colors = sliderColors())
            }

            HorizontalDivider(color = Color(0xFF444444))

            // ── Controls ────────────────────────────────────────────────────
            Button(
                onClick = { if (state.running) state.stop() else state.start() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (state.running) Color(0xFFCC3333) else Accent
                )
            ) { Text(if (state.running) "Stop" else "Start", color = Color.Black) }

            OutlinedButton(
                onClick = { state.toggleMjpeg() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = if (state.mjpegActive) Accent else TextSub
                )
            ) {
                Text(if (state.mjpegActive)
                    "MJPEG ON  :8080  (${state.mjpegClients} clients)"
                else "MJPEG OFF")
            }

            OutlinedButton(
                onClick = {
                    val path = saveFileDialog("Save screenshot", "png") ?: return@OutlinedButton
                    state.saveScreenshot(path)
                },
                modifier = Modifier.fillMaxWidth()
            ) { Text("Screenshot") }

            HorizontalDivider(color = Color(0xFF444444))

            // ── Status ──────────────────────────────────────────────────────
            Text(state.statusMessage, color = TextSub, fontSize = 12.sp)

            // ── Per-class counts ─────────────────────────────────────────────
            val counts = state.detections
                .groupBy { it.cls }
                .mapValues { it.value.size }
                .entries.sortedByDescending { it.value }
            if (counts.isNotEmpty()) {
                SectionCard {
                    Label("Detections")
                    counts.take(12).forEach { (cls, n) ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(state.labelFor(cls), color = Color.White, fontSize = 12.sp)
                            Text("$n", color = Accent, fontSize = 12.sp)
                        }
                    }
                }
            }
        }
    }
}

// ── Small helpers ────────────────────────────────────────────────────────────

@Composable
private fun SectionCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors   = CardDefaults.cardColors(containerColor = BgCard),
        shape    = RoundedCornerShape(8.dp)
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp), content = content)
    }
}

@Composable
private fun Label(text: String) = Text(text, color = TextSub, fontSize = 12.sp)

@Composable
private fun sliderColors() = SliderDefaults.colors(thumbColor = Accent, activeTrackColor = Accent)

@Composable
private fun tfColors() = OutlinedTextFieldDefaults.colors(
    focusedTextColor   = Color.White, unfocusedTextColor   = Color.White,
    focusedBorderColor = Accent,      unfocusedBorderColor = Color.Gray
)

private fun GpuPreference.label() = when (this) {
    GpuPreference.CPU      -> "CPU"
    GpuPreference.AUTO     -> "AUTO"
    GpuPreference.CUDA     -> "CUDA"
    GpuPreference.DIRECTML -> "DirectML"
}

private fun pickFile(title: String, extension: String): String? {
    val fd = FileDialog(null as Frame?, title, FileDialog.LOAD)
    fd.file = "*.$extension"
    fd.isVisible = true
    val f = fd.file ?: return null
    return "${fd.directory}$f"
}

private fun saveFileDialog(title: String, extension: String): String? {
    val fd = FileDialog(null as Frame?, title, FileDialog.SAVE)
    fd.file = "screenshot.$extension"
    fd.isVisible = true
    val f = fd.file ?: return null
    return "${fd.directory}$f"
}
