package com.destik.yolodesktop

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.awt.image.BufferedImage

class AppState {
    var modelPath      by mutableStateOf("")
    var sourcePath     by mutableStateOf("0")      // webcam index or http URL
    var inputSize      by mutableStateOf(640)
    var numClasses     by mutableStateOf(80)
    var confThreshold  by mutableStateOf(0.25f)
    var running        by mutableStateOf(false)
    var statusMessage  by mutableStateOf("Idle")

    var currentFrame   by mutableStateOf<BufferedImage?>(null)
    var detections     by mutableStateOf<List<Detection>>(emptyList())

    var mjpegActive    by mutableStateOf(false)
    var mjpegClients   by mutableStateOf(0)

    val mjpegServer = MjpegServer(8080)
    val detector    = OnnxDetector()
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
            detector.load(modelPath, inputSize, numClasses, confThreshold)
        } catch (e: Exception) {
            statusMessage = "Model load failed: ${e.message}"; return
        }
        running = true
        statusMessage = "Running"
        videoInput = VideoInput(
            source = sourcePath,
            onFrame = { img ->
                val dets = runCatching { detector.detect(img) }.getOrDefault(emptyList())
                val composed = drawDetections(img, dets)
                currentFrame = composed
                detections = dets
                if (mjpegActive) {
                    mjpegServer.pushFrame(composed)
                    mjpegClients = mjpegServer.clientCount()
                }
            },
            onError = { msg -> statusMessage = msg }
        ).also { it.start() }
    }

    fun stop() {
        videoInput?.stop(); videoInput = null
        running = false
        statusMessage = "Stopped"
    }

    fun toggleMjpeg() {
        if (mjpegActive) {
            mjpegServer.stop()
            mjpegActive = false
            mjpegClients = 0
        } else {
            mjpegServer.start()
            mjpegActive = true
        }
    }

    fun saveScreenshot(path: String) {
        val img = currentFrame ?: return
        val file = java.io.File(path)
        javax.imageio.ImageIO.write(img, "PNG", file)
        statusMessage = "Saved: ${file.name}"
    }

    private fun drawDetections(src: BufferedImage, dets: List<Detection>): BufferedImage {
        val out = BufferedImage(src.width, src.height, BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        g.drawImage(src, 0, 0, null)
        g.stroke = java.awt.BasicStroke(2f)
        val colors = arrayOf(
            java.awt.Color(255, 80, 80),   java.awt.Color(80, 200, 80),
            java.awt.Color(80, 120, 255),  java.awt.Color(255, 200, 0),
            java.awt.Color(200, 0, 200),   java.awt.Color(0, 200, 200)
        )
        for (d in dets) {
            val color = colors[d.cls % colors.size]
            g.color = color
            g.drawRect(d.x1.toInt(), d.y1.toInt(), (d.x2 - d.x1).toInt(), (d.y2 - d.y1).toInt())
            val label = "${labelFor(d.cls)} ${"%.2f".format(d.conf)}"
            val fm = g.fontMetrics
            val tw = fm.stringWidth(label)
            val th = fm.height
            g.fillRect(d.x1.toInt(), d.y1.toInt() - th, tw + 4, th)
            g.color = java.awt.Color.BLACK
            g.drawString(label, d.x1.toInt() + 2, d.y1.toInt() - 2)
        }
        g.dispose()
        return out
    }
}
