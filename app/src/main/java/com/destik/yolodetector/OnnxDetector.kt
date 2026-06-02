package com.destik.yolodetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.util.EnumSet
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.FloatBuffer
import kotlin.math.min
import kotlin.math.roundToInt

class OnnxDetector(private val config: ModelConfig) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var lastDiag = "onnx|init"

    // Reusable buffers to avoid per-frame GC pressure
    private var paddedBmp: Bitmap? = null
    private var pixelBuf: IntArray? = null
    private var floatBuf: FloatArray? = null
    private val drawPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    fun init(): Boolean {
        return try {
            var activeEP = "CPU"
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(config.numThreads)
                // Graph-level optimisation (fuse ops etc.) — always beneficial
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

                if (config.useGPU) {
                    // NNAPI: Qualcomm/ARM NPU/DSP/GPU (Android 8.1+, API 27+)
                    try {
                        addNnapi(EnumSet.of(OrtSession.SessionOptions.NNAPIFlags.USE_FP16))
                        activeEP = "NNAPI"
                    } catch (_: Throwable) {
                        try { addNnapi(); activeEP = "NNAPI" } catch (_: Throwable) {}
                    }
                }
                // XNNPACK: SIMD-optimised CPU inference; always registered as fallback
                // Handles ops NNAPI can't, and is sole provider when GPU is off
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to config.numThreads.toString()))
                    if (activeEP == "CPU") activeEP = "XNNPACK"
                } catch (_: Throwable) {}
            }
            session = env.createSession(config.onnxPath, opts)
            lastDiag = "onnx|ready|ep=$activeEP"
            true
        } catch (e: Exception) {
            lastDiag = "onnx|init_error|${e.message}"
            false
        }
    }

    fun detect(bitmap: Bitmap): Array<Detection> {
        val sess = session ?: return emptyArray<Detection>().also { lastDiag = "onnx|no_session" }
        return try {
            val sz   = config.inputSize
            val bmpW = bitmap.width
            val bmpH = bitmap.height

            // Letterbox: scale to fit sz×sz, pad with gray 114
            val scale = min(sz.toFloat() / bmpW, sz.toFloat() / bmpH)
            val nw    = (bmpW * scale).roundToInt()
            val nh    = (bmpH * scale).roundToInt()
            val padX  = (sz - nw) / 2
            val padY  = (sz - nh) / 2

            // Reuse padded bitmap (same sz×sz) — it gets fully overwritten each frame
            val padded = paddedBmp?.takeIf { it.width == sz && it.height == sz }
                ?: Bitmap.createBitmap(sz, sz, Bitmap.Config.ARGB_8888).also { paddedBmp = it }
            Canvas(padded).apply {
                drawColor(Color.rgb(114, 114, 114))
                // Draw directly into padded using scaled rect — avoids creating scaled Bitmap
                drawBitmap(bitmap,
                    android.graphics.Rect(0, 0, bmpW, bmpH),
                    android.graphics.RectF(padX.toFloat(), padY.toFloat(),
                                          (padX + nw).toFloat(), (padY + nh).toFloat()),
                    drawPaint)
            }
            val n      = sz * sz
            val pixels = (pixelBuf?.takeIf { it.size == n } ?: IntArray(n).also { pixelBuf = it })
            padded.getPixels(pixels, 0, sz, 0, 0, sz, sz)

            val arr = (floatBuf?.takeIf { it.size == 3 * n } ?: FloatArray(3 * n).also { floatBuf = it })
            val inv = 1f / 255f
            for (i in 0 until n) {
                val p = pixels[i]
                arr[i]         = (p shr 16 and 0xFF) * inv
                arr[i + n]     = (p shr  8 and 0xFF) * inv
                arr[i + n + n] = (p        and 0xFF) * inv
            }

            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr),
                longArrayOf(1L, 3L, sz.toLong(), sz.toLong()))

            val inputName = sess.inputNames.first()
            sess.run(mapOf(inputName to inputTensor)).use { results ->
                inputTensor.close()
                (results.first().value as OnnxTensor).use { out ->
                    postprocess(out, sz, padX, padY, scale, bmpW, bmpH)
                }
            }
        } catch (e: Exception) {
            lastDiag = "onnx|error|${e.message}"
            emptyArray()
        }
    }

    private fun postprocess(
        out: OnnxTensor, sz: Int, padX: Int, padY: Int,
        scale: Float, bmpW: Int, bmpH: Int
    ): Array<Detection> {
        val shape = out.info.shape   // [1, A, B]
        val buf   = out.floatBuffer
        val ct    = config.confThreshold
        if (shape.size < 3) return emptyArray()
        val A = shape[1].toInt()
        val B = shape[2].toInt()

        return if ((B == 6 && A > 6) || (A == 6 && B > 6)) {
            parseNmsFree(buf, A, B, sz, padX, padY, scale, bmpW, bmpH, ct)
        } else {
            parseAnchorFree(buf, A, B, sz, padX, padY, scale, bmpW, bmpH, ct)
        }
    }

    private fun parseNmsFree(
        buf: FloatBuffer, A: Int, B: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, bmpW: Int, bmpH: Int, ct: Float
    ): Array<Detection> {
        val tr = (A == 6)
        val nd = if (tr) B else A

        // Auto-detect pixel vs normalised coordinates
        var pixelCoords = false
        for (i in 0 until min(nd, 100)) {
            val x2 = if (tr) buf[2 * nd + i] else buf[i * 6 + 2]
            if (!x2.isNaN() && x2 > 1.5f) { pixelCoords = true; break }
        }
        val sc = if (pixelCoords) 1f / sz else 1f

        buf.rewind()
        var maxC = 0f
        val dets = mutableListOf<Detection>()
        for (i in 0 until nd) {
            val x1    = get6(tr, 0, nd, i, buf) * sc
            val y1    = get6(tr, 1, nd, i, buf) * sc
            val x2    = get6(tr, 2, nd, i, buf) * sc
            val y2    = get6(tr, 3, nd, i, buf) * sc
            val score = get6(tr, 4, nd, i, buf)
            val cid   = get6(tr, 5, nd, i, buf)
            if (score.isNaN() || score.isInfinite()) continue
            if (score > maxC) maxC = score
            if (score < ct || x2 <= x1 || y2 <= y1) continue
            dets += Detection(
                x  = (x1 * sz - padX) / (scale * bmpW),
                y  = (y1 * sz - padY) / (scale * bmpH),
                w  = (x2 - x1) * sz / (scale * bmpW),
                h  = (y2 - y1) * sz / (scale * bmpH),
                label      = cid.toInt().coerceIn(0, 65535),
                confidence = score
            )
        }
        lastDiag = "onnx|nms-free|${nd}|maxC:${"%.2f".format(maxC)}|dets:${dets.size}"
        return dets.toTypedArray()
    }

    private fun parseAnchorFree(
        buf: FloatBuffer, A: Int, B: Int,
        sz: Int, padX: Int, padY: Int, scale: Float, bmpW: Int, bmpH: Int, ct: Float
    ): Array<Detection> {
        val nc  = config.numClasses
        val tr  = (A == 4 + nc)
        val nd  = if (tr) B else A

        buf.rewind()
        val raw = mutableListOf<Detection>()
        for (i in 0 until nd) {
            val cx = getN(tr, 0, nd, nc, i, buf)
            val cy = getN(tr, 1, nd, nc, i, buf)
            val bw = getN(tr, 2, nd, nc, i, buf)
            val bh = getN(tr, 3, nd, nc, i, buf)
            if (bw <= 0f || bh <= 0f || cx.isNaN()) continue
            var best = ct; var cls = -1
            for (c in 0 until nc) {
                val s = getN(tr, 4 + c, nd, nc, i, buf)
                if (s > best) { best = s; cls = c }
            }
            if (cls < 0) continue
            raw += Detection(
                x  = ((cx - bw * .5f) / sz * sz - padX) / (scale * bmpW),
                y  = ((cy - bh * .5f) / sz * sz - padY) / (scale * bmpH),
                w  = bw / sz * sz / (scale * bmpW),
                h  = bh / sz * sz / (scale * bmpH),
                label      = cls,
                confidence = best
            )
        }
        // Greedy NMS
        val sorted = raw.sortedByDescending { it.confidence }
        val keep   = BooleanArray(sorted.size) { true }
        val result = mutableListOf<Detection>()
        for (i in sorted.indices) {
            if (!keep[i]) continue
            result += sorted[i]
            for (j in i + 1 until sorted.size) {
                if (keep[j] && sorted[i].label == sorted[j].label && iou(sorted[i], sorted[j]) > config.nmsThreshold)
                    keep[j] = false
            }
        }
        lastDiag = "onnx|v8|${nd}|maxC:${"%.2f".format(raw.maxOfOrNull { it.confidence } ?: 0f)}|dets:${result.size}"
        return result.toTypedArray()
    }

    fun getDiagnostics(): String = lastDiag

    fun release() {
        session?.close(); session = null
        paddedBmp?.recycle(); paddedBmp = null
        pixelBuf = null; floatBuf = null
    }

    // Element access helpers
    private fun get6(tr: Boolean, attr: Int, nd: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * 6 + attr]

    private fun getN(tr: Boolean, attr: Int, nd: Int, nc: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * (4 + nc) + attr]

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x, b.x);         val iy1 = maxOf(a.y, b.y)
        val ix2 = minOf(a.x + a.w, b.x + b.w); val iy2 = minOf(a.y + a.h, b.y + b.h)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        val union = a.w * a.h + b.w * b.h - inter
        return if (union <= 0f) 0f else inter / union
    }
}
