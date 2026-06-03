package com.destik.yolodesktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float, val cls: Int)

/**
 * ONNX Runtime detector with optional GPU acceleration.
 *
 * Handles both common YOLO export layouts:
 *  - NMS-free / end-to-end [1,N,6] = x1,y1,x2,y2,score,cls (YOLOv10)
 *  - anchor-free [1,4+nc,N] / [1,N,4+nc] = cx,cy,w,h,cls… (YOLOv8/v9/v11, NMS here)
 *
 * Per-frame buffers (padded image, pixel + float arrays) are reused so a slow
 * single-board CPU isn't fighting the garbage collector on every frame.
 *
 * GPU modes: AUTO (CUDA→DirectML→CPU), CUDA, DIRECTML, CPU.
 */
class OnnxDetector {

    enum class GpuMode { CPU, CUDA, DIRECTML, AUTO }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputSize = 640
    var confThreshold = 0.25f
    var numClasses = 80
    var nmsThreshold = 0.45f

    /** Human-readable description of the execution provider actually used. */
    var activeProvider = "CPU"
        private set

    // Reusable per-frame buffers.
    private var padded: BufferedImage? = null
    private var pixels: IntArray? = null
    private var floats: FloatArray? = null

    fun load(
        modelPath: String,
        inputSize: Int = 640,
        numClasses: Int = 80,
        confThreshold: Float = 0.25f,
        gpuMode: GpuMode = GpuMode.AUTO
    ) {
        close()
        this.inputSize     = inputSize
        this.numClasses    = numClasses
        this.confThreshold = confThreshold

        env = OrtEnvironment.getEnvironment()
        val opts = buildSessionOptions(gpuMode)
        session = env!!.createSession(modelPath, opts)
    }

    fun detect(image: BufferedImage): List<Detection> {
        val sess = session ?: return emptyList()
        val env  = env     ?: return emptyList()

        val sz = inputSize
        val w  = image.width
        val h  = image.height
        val scale = min(sz.toFloat() / w, sz.toFloat() / h)
        val nw = (w * scale).roundToInt()
        val nh = (h * scale).roundToInt()
        val padX = (sz - nw) / 2
        val padY = (sz - nh) / 2

        // Reusable letterbox canvas (gray 114 padding, matches training).
        val pad = padded?.takeIf { it.width == sz && it.height == sz }
            ?: BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB).also { padded = it }
        pad.createGraphics().apply {
            color = Color(114, 114, 114); fillRect(0, 0, sz, sz)
            drawImage(image, padX, padY, nw, nh, null); dispose()
        }

        val n  = sz * sz
        val px = pixels?.takeIf { it.size == n } ?: IntArray(n).also { pixels = it }
        pad.getRGB(0, 0, sz, sz, px, 0, sz)

        val arr = floats?.takeIf { it.size == 3 * n } ?: FloatArray(3 * n).also { floats = it }
        for (i in 0 until n) {
            val p = px[i]
            arr[i]         = ((p shr 16) and 0xFF) / 255f
            arr[i + n]     = ((p shr 8)  and 0xFF) / 255f
            arr[i + 2 * n] = ( p         and 0xFF) / 255f
        }

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr),
            longArrayOf(1L, 3L, sz.toLong(), sz.toLong()))
        return try {
            val results = sess.run(mapOf(sess.inputNames.iterator().next() to tensor))
            results.use {
                val out   = it[0] as OnnxTensor
                val shape = out.info.shape          // [1, A, B]
                if (shape.size < 3) return emptyList()
                val a = shape[1].toInt()
                val b = shape[2].toInt()
                val buf = out.floatBuffer
                if ((b == 6 && a in 7..4000) || (a == 6 && b in 7..4000))
                    parseNmsFree(buf, a, b, sz, padX, padY, scale, w, h)
                else
                    parseAnchorFree(buf, a, b, sz, padX, padY, scale, w, h)
            }
        } catch (e: Exception) {
            emptyList()
        } finally {
            tensor.close()
        }
    }

    private fun parseNmsFree(
        buf: FloatBuffer, a: Int, b: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, ow: Int, oh: Int
    ): List<Detection> {
        val tr = (a < b)
        val nd = if (tr) b else a
        fun g(attr: Int, i: Int) = if (tr) buf[attr * nd + i] else buf[i * 6 + attr]

        // pixel vs normalised coords
        var pixel = false
        for (i in 0 until min(nd, 100)) { val x2 = g(2, i); if (!x2.isNaN() && x2 > 1.5f) { pixel = true; break } }
        val sc = if (pixel) sz.toFloat() else 1f   // normalised → scale up to model px

        val dets = ArrayList<Detection>()
        for (i in 0 until nd) {
            val score = g(4, i)
            if (score.isNaN() || score < confThreshold) continue
            val x1 = g(0, i) * sc; val y1 = g(1, i) * sc
            val x2 = g(2, i) * sc; val y2 = g(3, i) * sc
            if (x2 <= x1 || y2 <= y1) continue
            val cls = g(5, i).toInt().coerceIn(0, numClasses - 1)
            dets += Detection(
                x1 = ((x1 - padX) / scale).coerceIn(0f, ow.toFloat()),
                y1 = ((y1 - padY) / scale).coerceIn(0f, oh.toFloat()),
                x2 = ((x2 - padX) / scale).coerceIn(0f, ow.toFloat()),
                y2 = ((y2 - padY) / scale).coerceIn(0f, oh.toFloat()),
                conf = score, cls = cls
            )
        }
        return dets
    }

    private fun parseAnchorFree(
        buf: FloatBuffer, a: Int, b: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, ow: Int, oh: Int
    ): List<Detection> {
        val tr    = (a < b)
        val nd    = if (tr) b else a
        val attrs = if (tr) a else b
        val nc    = (attrs - 4).coerceAtLeast(1)
        fun g(attr: Int, i: Int) = if (tr) buf[attr * nd + i] else buf[i * attrs + attr]

        val raw = ArrayList<Detection>()
        for (i in 0 until nd) {
            var best = confThreshold; var cls = -1
            for (c in 0 until nc) { val s = g(4 + c, i); if (s > best) { best = s; cls = c } }
            if (cls < 0) continue
            val cx = g(0, i); val cy = g(1, i); val bw = g(2, i); val bh = g(3, i)
            if (bw <= 0f || bh <= 0f || cx.isNaN()) continue
            val x1 = (cx - bw / 2 - padX) / scale
            val y1 = (cy - bh / 2 - padY) / scale
            val x2 = (cx + bw / 2 - padX) / scale
            val y2 = (cy + bh / 2 - padY) / scale
            raw += Detection(
                x1 = x1.coerceIn(0f, ow.toFloat()), y1 = y1.coerceIn(0f, oh.toFloat()),
                x2 = x2.coerceIn(0f, ow.toFloat()), y2 = y2.coerceIn(0f, oh.toFloat()),
                conf = best, cls = cls
            )
        }
        return nms(raw)
    }

    /** Greedy per-class NMS. */
    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.conf }
        val keep = BooleanArray(sorted.size) { true }
        val out = ArrayList<Detection>()
        for (i in sorted.indices) {
            if (!keep[i]) continue
            out += sorted[i]
            for (j in i + 1 until sorted.size)
                if (keep[j] && sorted[i].cls == sorted[j].cls && iou(sorted[i], sorted[j]) > nmsThreshold)
                    keep[j] = false
        }
        return out
    }

    private fun iou(p: Detection, q: Detection): Float {
        val ix1 = maxOf(p.x1, q.x1); val iy1 = maxOf(p.y1, q.y1)
        val ix2 = minOf(p.x2, q.x2); val iy2 = minOf(p.y2, q.y2)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = (p.x2 - p.x1) * (p.y2 - p.y1) + (q.x2 - q.x1) * (q.y2 - q.y1) - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        session?.close(); session = null
        env?.close(); env = null
        padded = null; pixels = null; floats = null
    }

    private fun buildSessionOptions(mode: GpuMode): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
        when (mode) {
            GpuMode.CPU      -> activeProvider = "CPU"
            GpuMode.CUDA     -> { opts.addCUDA(0);     activeProvider = "CUDA GPU:0" }
            GpuMode.DIRECTML -> { opts.addDirectML(0); activeProvider = "DirectML GPU:0" }
            GpuMode.AUTO     -> activeProvider = tryAddGpuAuto(opts)
        }
        return opts
    }

    private fun tryAddGpuAuto(opts: OrtSession.SessionOptions): String {
        try { opts.addCUDA(0); return "CUDA GPU:0" } catch (_: Throwable) {}
        try { opts.addDirectML(0); return "DirectML GPU:0" } catch (_: Throwable) {}
        return "CPU (no GPU available)"
    }
}
