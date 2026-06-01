package com.destik.yolodetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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

    fun init(): Boolean {
        return try {
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(config.numThreads)
                try { addNnapi() } catch (_: Throwable) {}
            }
            session = env.createSession(config.onnxPath, opts)
            lastDiag = "onnx|ready"
            true
        } catch (e: Exception) {
            lastDiag = "onnx|init_error|${e.message}"
            false
        }
    }

    fun detect(bitmap: Bitmap): Array<Detection> {
        return try {
            val sess = session ?: run {
                lastDiag = "onnx|no_session"
                return emptyArray()
            }

            val sz = config.inputSize
            val bmpW = bitmap.width
            val bmpH = bitmap.height

            // Letterbox preprocess
            val scale = min(sz.toFloat() / bmpW, sz.toFloat() / bmpH)
            val nw = (bmpW * scale).roundToInt()
            val nh = (bmpH * scale).roundToInt()
            val padX = (sz - nw) / 2
            val padY = (sz - nh) / 2

            val padded = Bitmap.createBitmap(sz, sz, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(padded)
            canvas.drawColor(Color.rgb(114, 114, 114))
            val scaled = Bitmap.createScaledBitmap(bitmap, nw, nh, true)
            canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), Paint())
            scaled.recycle()

            val pixels = IntArray(sz * sz)
            padded.getPixels(pixels, 0, sz, 0, 0, sz, sz)
            padded.recycle()

            val arr = FloatArray(3 * sz * sz)
            val rOffset = 0
            val gOffset = sz * sz
            val bOffset = 2 * sz * sz
            for (idx in pixels.indices) {
                val px = pixels[idx]
                arr[rOffset + idx] = Color.red(px) / 255f
                arr[gOffset + idx] = Color.green(px) / 255f
                arr[bOffset + idx] = Color.blue(px) / 255f
            }

            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(arr),
                longArrayOf(1L, 3L, sz.toLong(), sz.toLong())
            )

            // Run inference
            val results = try {
                val inputName = sess.inputNames.first()
                sess.run(mapOf(inputName to inputTensor))
            } finally {
                inputTensor.close()
            }

            // Postprocess
            val out: OnnxTensor
            val shape: LongArray
            val buf: FloatBuffer
            try {
                out = results.first().value as OnnxTensor
                shape = out.info.shape   // expect [1, A, B]
                buf = out.floatBuffer
            } finally {
                results.close()
            }

            val ct = config.confThreshold
            val A = shape[1].toInt()
            val B = shape[2].toInt()

            // NMS-free branch: shape like [1, nd, 6] or [1, 6, nd]
            if ((B == 6 && A > 6) || (A == 6 && B > 6)) {
                val tr = (A == 6)
                val nd = if (tr) B else A
                val detections = mutableListOf<Detection>()

                // Auto-detect pixel coords by sampling x2 values
                val sampleCount = min(nd, 100)
                var pixelCoords = false
                for (i in 0 until sampleCount) {
                    val x2 = if (tr) buf[2 * nd + i] else buf[i * 6 + 2]
                    if (x2 > 1.5f) { pixelCoords = true; break }
                }
                val sc = if (pixelCoords) 1f / sz else 1f

                buf.rewind()
                for (i in 0 until nd) {
                    val x1    = get(tr, 0, nd, i, buf) * sc
                    val y1    = get(tr, 1, nd, i, buf) * sc
                    val x2    = get(tr, 2, nd, i, buf) * sc
                    val y2    = get(tr, 3, nd, i, buf) * sc
                    val score = get(tr, 4, nd, i, buf)
                    val cid   = get(tr, 5, nd, i, buf)

                    if (score.isNaN() || score.isInfinite() || score < ct) continue
                    if (x2 <= x1 || y2 <= y1) continue

                    // Reverse letterbox to normalised image coords
                    val rx1 = (x1 * sz - padX) / (scale * bmpW)
                    val ry1 = (y1 * sz - padY) / (scale * bmpH)
                    val rx2 = (x2 * sz - padX) / (scale * bmpW)
                    val ry2 = (y2 * sz - padY) / (scale * bmpH)

                    detections.add(
                        Detection(
                            rx1, ry1,
                            rx2 - rx1, ry2 - ry1,
                            cid.toInt().coerceIn(0, 65535),
                            score
                        )
                    )
                }

                lastDiag = "onnx|nms-free|${nd}dets:${detections.size}"
                detections.toTypedArray()

            } else {
                // Anchor-free YOLOv8 branch: shape [1, 4+nc, nd] or [1, nd, 4+nc]
                val nc = config.numClasses
                val tr8 = (A == 4 + nc)
                val nd = if (tr8) B else A
                val raw = mutableListOf<Detection>()

                buf.rewind()
                for (i in 0 until nd) {
                    val cx = safeGet(tr8, 0, nd, nc, i, buf)
                    val cy = safeGet(tr8, 1, nd, nc, i, buf)
                    val bw = safeGet(tr8, 2, nd, nc, i, buf)
                    val bh = safeGet(tr8, 3, nd, nc, i, buf)

                    if (bw <= 0f || bh <= 0f || cx.isNaN()) continue

                    var bestScore = -Float.MAX_VALUE
                    var bestClass = 0
                    for (c in 0 until nc) {
                        val s = safeGet(tr8, 4 + c, nd, nc, i, buf)
                        if (s > bestScore) { bestScore = s; bestClass = c }
                    }
                    if (bestScore <= ct) continue

                    // Normalise to 0..1 in padded image space, then reverse letterbox
                    val nx1 = (cx - bw * 0.5f) / sz
                    val ny1 = (cy - bh * 0.5f) / sz
                    val nw_ = bw / sz
                    val nh_ = bh / sz

                    val rx1 = (nx1 * sz - padX) / (scale * bmpW)
                    val ry1 = (ny1 * sz - padY) / (scale * bmpH)
                    val rw  = nw_ * sz / (scale * bmpW)
                    val rh  = nh_ * sz / (scale * bmpH)

                    raw.add(Detection(rx1, ry1, rw, rh, bestClass, bestScore))
                }

                // NMS
                val sorted = raw.sortedByDescending { it.confidence }.toMutableList()
                val result = mutableListOf<Detection>()
                val suppressed = BooleanArray(sorted.size)
                for (i in sorted.indices) {
                    if (suppressed[i]) continue
                    result.add(sorted[i])
                    for (j in i + 1 until sorted.size) {
                        if (suppressed[j]) continue
                        if (sorted[i].label == sorted[j].label &&
                            iou(sorted[i], sorted[j]) > config.nmsThreshold
                        ) {
                            suppressed[j] = true
                        }
                    }
                }

                lastDiag = "onnx|v8|${nd}|maxC:${"%.2f".format(raw.maxOfOrNull { it.confidence } ?: 0f)}|dets:${result.size}"
                result.toTypedArray()
            }

        } catch (e: Exception) {
            lastDiag = "onnx|detect_error|${e.message}"
            emptyArray()
        }
    }

    fun getDiagnostics(): String = lastDiag

    fun release() {
        session?.close()
        session = null
    }

    /**
     * Get element from a [6, nd] (tr=true) or [nd, 6] (tr=false) buffer.
     */
    private fun get(tr: Boolean, attr: Int, nd: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * 6 + attr]

    /**
     * Get element from a [4+nc, nd] (tr=true) or [nd, 4+nc] (tr=false) buffer.
     */
    private fun safeGet(tr: Boolean, attr: Int, nd: Int, nc: Int, i: Int, buf: FloatBuffer): Float =
        if (tr) buf[attr * nd + i] else buf[i * (4 + nc) + attr]

    private fun iou(a: Detection, b: Detection): Float {
        val ax2 = a.x + a.w; val ay2 = a.y + a.h
        val bx2 = b.x + b.w; val by2 = b.y + b.h

        val ix1 = maxOf(a.x, b.x); val iy1 = maxOf(a.y, b.y)
        val ix2 = minOf(ax2, bx2); val iy2 = minOf(ay2, by2)

        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        if (inter == 0f) return 0f

        val union = a.w * a.h + b.w * b.h - inter
        return if (union <= 0f) 0f else inter / union
    }
}
