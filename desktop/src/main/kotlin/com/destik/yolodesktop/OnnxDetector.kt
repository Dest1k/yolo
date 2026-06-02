package com.destik.yolodesktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float, val cls: Int)

class OnnxDetector {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputSize = 640
    var confThreshold = 0.25f
    var numClasses = 80

    fun load(modelPath: String, inputSize: Int = 640, numClasses: Int = 80, confThreshold: Float = 0.25f) {
        close()
        this.inputSize = inputSize
        this.numClasses = numClasses
        this.confThreshold = confThreshold
        env = OrtEnvironment.getEnvironment()
        session = env!!.createSession(modelPath, OrtSession.SessionOptions())
    }

    fun detect(image: BufferedImage): List<Detection> {
        val sess = session ?: return emptyList()
        val env = env ?: return emptyList()

        val (letterboxed, scale, padX, padY) = letterbox(image, inputSize)
        val tensor = imageToTensor(letterboxed, inputSize, env)
        val inputName = sess.inputNames.iterator().next()

        val results = sess.run(mapOf(inputName to tensor))
        val output = results[0].value as Array<*>
        val rows = output[0] as Array<*>

        val dets = mutableListOf<Detection>()
        for (row in rows) {
            val r = row as FloatArray
            if (r.size < 6) continue
            val conf = r[4]
            if (conf < confThreshold) continue
            val cls = r[5].toInt().coerceIn(0, numClasses - 1)

            val cx = (r[0] - padX) / scale
            val cy = (r[1] - padY) / scale
            val w  = r[2] / scale
            val h  = r[3] / scale
            dets.add(Detection(
                x1 = (cx - w / 2).coerceIn(0f, image.width.toFloat()),
                y1 = (cy - h / 2).coerceIn(0f, image.height.toFloat()),
                x2 = (cx + w / 2).coerceIn(0f, image.width.toFloat()),
                y2 = (cy + h / 2).coerceIn(0f, image.height.toFloat()),
                conf = conf, cls = cls
            ))
        }

        results.close()
        tensor.close()
        return dets
    }

    private data class LetterboxResult(val img: BufferedImage, val scale: Float, val padX: Float, val padY: Float)

    private fun letterbox(src: BufferedImage, size: Int): LetterboxResult {
        val scale = min(size.toFloat() / src.width, size.toFloat() / src.height)
        val newW = (src.width * scale).toInt()
        val newH = (src.height * scale).toInt()
        val padX = (size - newW) / 2f
        val padY = (size - newH) / 2f

        val dst = BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
        val g = dst.createGraphics()
        g.color = java.awt.Color(114, 114, 114)
        g.fillRect(0, 0, size, size)
        g.drawImage(src, padX.toInt(), padY.toInt(), newW, newH, null)
        g.dispose()
        return LetterboxResult(dst, scale, padX, padY)
    }

    private fun imageToTensor(img: BufferedImage, size: Int, env: OrtEnvironment): OnnxTensor {
        val buf = FloatBuffer.allocate(1 * 3 * size * size)
        val pixels = IntArray(size * size)
        img.getRGB(0, 0, size, size, pixels, 0, size)
        val n = size * size
        for (i in 0 until n) {
            val px = pixels[i]
            buf.put(i,           ((px shr 16 and 0xFF) / 255f))
            buf.put(i + n,       ((px shr 8  and 0xFF) / 255f))
            buf.put(i + n * 2,   ((px        and 0xFF) / 255f))
        }
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, size.toLong(), size.toLong()))
    }

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
    }
}
