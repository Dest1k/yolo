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
import java.awt.image.BufferedImage
import java.nio.file.Paths

fun main() = application {
    val windowState = rememberWindowState(width = 1280.dp, height = 800.dp)
    Window(onCloseRequest = ::exitApplication, title = "YOLO Detector", state = windowState) {
        MaterialTheme(colorScheme = darkColorScheme()) {
            AppScreen()
        }
    }
}

@Composable
fun AppScreen() {
    val state = remember { AppState() }
    DisposableEffect(Unit) {
        onDispose {
            state.stop()
            if (state.mjpegActive) state.mjpegServer.stop()
            state.detector.close()
        }
    }

    Row(Modifier.fillMaxSize().background(Color(0xFF1A1A1A))) {
        // Video panel
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
                Text("No video", color = Color.Gray, fontSize = 18.sp)
            }
        }

        // Settings sidebar
        Column(
            Modifier
                .width(300.dp)
                .fillMaxHeight()
                .background(Color(0xFF2A2A2A))
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text("YOLO Detector", color = Color(0xFF00E676), fontSize = 20.sp,
                style = MaterialTheme.typography.titleLarge)

            // Model file
            SectionLabel("ONNX Model")
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = if (state.modelPath.isEmpty()) "Not selected" else Paths.get(state.modelPath).fileName.toString(),
                    color = Color.White, fontSize = 12.sp, modifier = Modifier.weight(1f)
                )
                OutlinedButton(onClick = {
                    val path = pickFile("Select ONNX model", "onnx")
                    if (path != null) state.modelPath = path
                }) { Text("Browse") }
            }

            // Source
            SectionLabel("Video Source")
            OutlinedTextField(
                value = state.sourcePath,
                onValueChange = { state.sourcePath = it },
                label = { Text("Webcam index or http://...") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                    focusedBorderColor = Color(0xFF00E676), unfocusedBorderColor = Color.Gray
                )
            )

            // Settings
            SectionLabel("Input size: ${state.inputSize}")
            Slider(value = state.inputSize.toFloat(), onValueChange = { state.inputSize = it.toInt() },
                valueRange = 320f..1280f, steps = 3,
                colors = SliderDefaults.colors(thumbColor = Color(0xFF00E676), activeTrackColor = Color(0xFF00E676)))

            SectionLabel("Conf threshold: ${"%.2f".format(state.confThreshold)}")
            Slider(value = state.confThreshold, onValueChange = { state.confThreshold = it },
                valueRange = 0.05f..0.95f,
                colors = SliderDefaults.colors(thumbColor = Color(0xFF00E676), activeTrackColor = Color(0xFF00E676)))

            SectionLabel("Classes: ${state.numClasses}")
            Slider(value = state.numClasses.toFloat(), onValueChange = { state.numClasses = it.toInt() },
                valueRange = 1f..200f, steps = 198,
                colors = SliderDefaults.colors(thumbColor = Color(0xFF00E676), activeTrackColor = Color(0xFF00E676)))

            Divider(color = Color(0xFF444444))

            // Start/Stop
            Button(
                onClick = { if (state.running) state.stop() else state.start() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (state.running) Color(0xFFCC3333) else Color(0xFF00E676)
                )
            ) {
                Text(if (state.running) "Stop" else "Start", color = Color.Black)
            }

            // MJPEG server
            OutlinedButton(
                onClick = { state.toggleMjpeg() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = if (state.mjpegActive) Color(0xFF00E676) else Color.Gray
                )
            ) {
                Text(if (state.mjpegActive) "MJPEG ON (:8080, ${state.mjpegClients} clients)" else "MJPEG OFF")
            }

            // Screenshot
            OutlinedButton(
                onClick = {
                    val path = saveFileDialog("Save screenshot", "png") ?: return@OutlinedButton
                    state.saveScreenshot(path)
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Screenshot")
            }

            Divider(color = Color(0xFF444444))

            // Status
            Text(state.statusMessage, color = Color(0xFFAAAAAA), fontSize = 12.sp)

            // Detection counts
            if (state.detections.isNotEmpty()) {
                val counts = state.detections.groupBy { it.cls }
                    .mapValues { it.value.size }
                    .entries.sortedByDescending { it.value }
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    for ((cls, count) in counts.take(10)) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(state.labelFor(cls), color = Color.White, fontSize = 12.sp)
                            Text("$count", color = Color(0xFF00E676), fontSize = 12.sp)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(text, color = Color(0xFFAAAAAA), fontSize = 12.sp)
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
