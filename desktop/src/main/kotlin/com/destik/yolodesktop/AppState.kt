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
    var videoInput: VideoInput? = null

    val cocoLabels = arrayOf(
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    )

    fun labelFor(cls: Int): String = cocoLabels.getOrNull(cls) ?: "cls$cls"

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
                val dets = runCatching {
                    when (modelType) {
                        ModelType.ONNX -> onnxDetector.detect(img)
                        ModelType.PT   -> ptDetector.detect(img)
                    }
                }.getOrDefault(emptyList())
                val composed = drawDetections(img, dets)
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

    private fun drawDetections(src: BufferedImage, dets: List<Detection>): BufferedImage {
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
        val g   = out.createGraphics()
        g.drawImage(src, 0, 0, null)
        g.stroke = java.awt.BasicStroke(2f)
        val palette = arrayOf(
            java.awt.Color(255, 80, 80),  java.awt.Color(80, 200, 80),
            java.awt.Color(80, 120, 255), java.awt.Color(255, 200, 0),
            java.awt.Color(200, 0, 200),  java.awt.Color(0, 200, 200)
        )
        for (d in dets) {
            val color = palette[d.cls % palette.size]
            g.color = color
            g.drawRect(d.x1.toInt(), d.y1.toInt(), (d.x2 - d.x1).toInt(), (d.y2 - d.y1).toInt())
            val label = "${labelFor(d.cls)} ${"%.2f".format(d.conf)}"
            val fm    = g.fontMetrics
            val tw    = fm.stringWidth(label)
            val th    = fm.height
            g.fillRect(d.x1.toInt(), d.y1.toInt() - th, tw + 4, th)
            g.color = java.awt.Color.BLACK
            g.drawString(label, d.x1.toInt() + 2, d.y1.toInt() - 2)
        }
        g.dispose()
        return out
    }
}
