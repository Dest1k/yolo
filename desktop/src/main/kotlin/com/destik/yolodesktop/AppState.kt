package com.destik.yolodesktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.awt.image.BufferedImage

enum class ModelType { ONNX, PT }
enum class GpuPreference { CPU, AUTO, CUDA, DIRECTML }

class AppState {
    private val uiScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    var modelPath       by mutableStateOf("")
    var modelType       by mutableStateOf(ModelType.ONNX)
    var sourcePath      by mutableStateOf("0")
    var inputSize       by mutableStateOf(640)
    var numClasses      by mutableStateOf(80)
    var confThreshold   by mutableStateOf(0.25f)
    var gpuPref         by mutableStateOf(GpuPreference.AUTO)

    var running         by mutableStateOf(false)
    var statusMessage   by mutableStateOf("Idle")
    var activeProvider  by mutableStateOf("")

    var currentFrame    by mutableStateOf<BufferedImage?>(null)
    var detections      by mutableStateOf<List<Detection>>(emptyList())

    var mjpegActive     by mutableStateOf(false)
    var mjpegClients    by mutableStateOf(0)

    val mjpegServer = MjpegServer(8080)
    private val onnxDetector = OnnxDetector()
    private val ptDetector   = PtDetector()
    private val tracker      = DetectionTracker()
    var videoInput: VideoInput? = null

    val cocoLabels get() = Render.cocoLabels

    fun labelFor(cls: Int): String = Render.labelFor(cls)

    fun start() {
        if (running) return
        if (modelPath.isEmpty()) { statusMessage = "Select a model first"; return }
        try {
            when (modelType) {
                ModelType.ONNX -> {
                    onnxDetector.load(modelPath, inputSize, numClasses, confThreshold, gpuPref.toOnnxMode())
                    activeProvider = onnxDetector.activeProvider
                }
                ModelType.PT -> {
                    ptDetector.load(modelPath, inputSize, numClasses, confThreshold, gpuPref.toPtMode())
                    activeProvider = if (gpuPref == GpuPreference.CPU) "CPU" else "PyTorch auto"
                }
            }
        } catch (e: Exception) {
            statusMessage = "Model load failed: ${e.message}"; return
        }
        running = true
        statusMessage = "Running [$activeProvider]"
        videoInput = VideoInput(
            source  = sourcePath,
            onFrame = { img ->
                val raw = runCatching {
                    when (modelType) {
                        ModelType.ONNX -> onnxDetector.detect(img)
                        ModelType.PT   -> ptDetector.detect(img)
                    }
                }.getOrDefault(emptyList())
                val dets = tracker.update(raw, System.currentTimeMillis())
                val composed = Render.draw(img, dets)
                if (mjpegActive) mjpegServer.pushFrame(composed)
                // Compose state must be mutated on the UI thread
                uiScope.launch {
                    currentFrame = composed
                    detections   = dets
                    if (mjpegActive) mjpegClients = mjpegServer.clientCount()
                }
            },
            onError = { msg -> uiScope.launch { statusMessage = msg } }
        ).also { it.start() }
    }

    fun stop() {
        videoInput?.stop(); videoInput = null
        tracker.reset()
        uiScope.launch { running = false; statusMessage = "Stopped" }
    }

    fun toggleMjpeg() {
        if (mjpegActive) {
            mjpegServer.stop(); mjpegActive = false; mjpegClients = 0
        } else {
            mjpegServer.start(); mjpegActive = true
        }
    }

    fun saveScreenshot(path: String) {
        val img  = currentFrame ?: return
        val file = java.io.File(path)
        javax.imageio.ImageIO.write(img, "PNG", file)
        statusMessage = "Saved: ${file.name}"
    }

    fun closeDetectors() {
        onnxDetector.close()
        ptDetector.close()
    }

    private fun GpuPreference.toOnnxMode() = when (this) {
        GpuPreference.CPU       -> OnnxDetector.GpuMode.CPU
        GpuPreference.CUDA      -> OnnxDetector.GpuMode.CUDA
        GpuPreference.DIRECTML  -> OnnxDetector.GpuMode.DIRECTML
        GpuPreference.AUTO      -> OnnxDetector.GpuMode.AUTO
    }

    private fun GpuPreference.toPtMode() = when (this) {
        GpuPreference.CPU  -> PtDetector.GpuMode.CPU
        else               -> PtDetector.GpuMode.AUTO
    }

}
